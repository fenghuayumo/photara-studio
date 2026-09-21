#include "patchmatch_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace photara::mvs::cuda_patchmatch {
namespace {

constexpr int k_half_window = 4;
constexpr int k_step = 2;
constexpr int k_texels = 25;
constexpr float k_robust = 2.F;
constexpr unsigned k_rows_per_launch = 64;

struct DeviceImage {
    int width{};
    int height{};
    float fx{};
    float fy{};
    float cx{};
    float cy{};
    const float* gray{};
    const std::uint8_t* mask{};
    const float* depth{};
};

struct DeviceSource {
    DeviceImage image;
    float rotation[9]{};
    float translation[3]{};
};

struct DeviceRequest {
    DeviceImage reference;
    float* depth{};
    float3* normal{};
    float* confidence{};
    const DeviceSource* sources{};
    unsigned source_count{};
    float depth_min{};
    float depth_max{};
    unsigned random_iters{};
    unsigned min_patch_views{};
    float geometric_weight{};
    unsigned random_seed{};
    bool use_geometric{};
    bool initialize_invalid{};
};

bool check(
    const cudaError_t status, const char* operation, std::string& error) {
    if (status == cudaSuccess) return true;
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status);
    error = message.str();
    return false;
}

template <class T>
class Buffer {
public:
    Buffer() = default;
    ~Buffer() {
        if (data_ != nullptr) cudaFree(data_);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    bool resize(const std::size_t size, std::string& error) {
        if (size == size_) return true;
        if (data_ != nullptr) {
            if (!check(cudaFree(data_), "cudaFree", error)) return false;
            data_ = nullptr;
            size_ = 0;
        }
        if (size == 0) return true;
        if (!check(
                cudaMalloc(reinterpret_cast<void**>(&data_),
                           size * sizeof(T)),
                "cudaMalloc", error))
            return false;
        size_ = size;
        return true;
    }

    bool upload(
        const T* source, const std::size_t size, std::string& error) {
        if (!resize(size, error)) return false;
        if (size == 0) return true;
        return check(
            cudaMemcpy(
                data_, source, size * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpy HostToDevice", error);
    }

    bool download(
        T* destination, const std::size_t size, std::string& error) const {
        if (size > size_) {
            error = "CUDA PatchMatch download exceeds allocation";
            return false;
        }
        if (size == 0) return true;
        return check(
            cudaMemcpy(
                destination, data_, size * sizeof(T),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy DeviceToHost", error);
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

private:
    T* data_{};
    std::size_t size_{};
};

__device__ __forceinline__ float3 make_vec(
    const float x, const float y, const float z) {
    return make_float3(x, y, z);
}

__device__ __forceinline__ float3 add(
    const float3 a, const float3 b) {
    return make_vec(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 sub(
    const float3 a, const float3 b) {
    return make_vec(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 mul(
    const float3 a, const float scale) {
    return make_vec(a.x * scale, a.y * scale, a.z * scale);
}

__device__ __forceinline__ float dot3(
    const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 normalized(const float3 value) {
    const float squared = dot3(value, value);
    if (!(squared > 1e-20F) || !isfinite(squared))
        return make_vec(0.F, 0.F, -1.F);
    return mul(value, rsqrtf(squared));
}

__device__ __forceinline__ float3 matrix_vector(
    const float matrix[9], const float3 value) {
    return make_vec(
        matrix[0] * value.x + matrix[1] * value.y +
            matrix[2] * value.z,
        matrix[3] * value.x + matrix[4] * value.y +
            matrix[5] * value.z,
        matrix[6] * value.x + matrix[7] * value.y +
            matrix[8] * value.z);
}

__device__ __forceinline__ float3 matrix_transpose_vector(
    const float matrix[9], const float3 value) {
    return make_vec(
        matrix[0] * value.x + matrix[3] * value.y +
            matrix[6] * value.z,
        matrix[1] * value.x + matrix[4] * value.y +
            matrix[7] * value.z,
        matrix[2] * value.x + matrix[5] * value.y +
            matrix[8] * value.z);
}

__device__ __forceinline__ std::uint32_t hash32(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return value;
}

__device__ __forceinline__ float random01(std::uint32_t& state) {
    state = hash32(state + 0x9e3779b9U);
    return static_cast<float>(state & 0x00ffffffU) /
           static_cast<float>(0x01000000U);
}

__device__ __forceinline__ bool foreground(
    const DeviceImage& image, const int x, const int y) {
    return image.mask == nullptr ||
           image.mask[static_cast<std::size_t>(y) * image.width + x] != 0;
}

__device__ bool sample_gray(
    const DeviceImage& image, const float x, const float y, float& value) {
    if (!(x >= 0.F && y >= 0.F &&
          x < static_cast<float>(image.width - 1) &&
          y < static_cast<float>(image.height - 1)))
        return false;
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const std::size_t row0 =
        static_cast<std::size_t>(y0) * image.width;
    const std::size_t row1 = row0 + image.width;
    const float a = image.gray[row0 + x0];
    const float b = image.gray[row0 + x0 + 1];
    const float c = image.gray[row1 + x0];
    const float d = image.gray[row1 + x0 + 1];
    value = (a + tx * (b - a)) +
            ty * ((c + tx * (d - c)) - (a + tx * (b - a)));
    return isfinite(value);
}

struct Patch {
    float texels[k_texels]{};
    float norm_squared{};
    float3 center_ray{};
};

__device__ bool fill_patch(
    const DeviceImage& image, const int x, const int y, Patch& patch) {
    if (x < k_half_window || y < k_half_window ||
        x + k_half_window >= image.width ||
        y + k_half_window >= image.height ||
        !foreground(image, x, y))
        return false;
    float sum = 0.F;
    int count = 0;
    int foreground_count = 0;
    for (int dy = -k_half_window; dy <= k_half_window; dy += k_step) {
        for (int dx = -k_half_window; dx <= k_half_window; dx += k_step) {
            const float value =
                image.gray[static_cast<std::size_t>(y + dy) * image.width +
                           x + dx];
            patch.texels[count++] = value;
            sum += value;
            foreground_count += foreground(image, x + dx, y + dy) ? 1 : 0;
        }
    }
    if (foreground_count < k_texels * 4 / 5) return false;
    const float mean = sum / static_cast<float>(k_texels);
    patch.norm_squared = 0.F;
    for (int i = 0; i < k_texels; ++i) {
        patch.texels[i] -= mean;
        patch.norm_squared += patch.texels[i] * patch.texels[i];
    }
    if (patch.norm_squared < 2.5e-4F) return false;
    patch.center_ray = make_vec(
        (static_cast<float>(x) - image.cx) / image.fx,
        (static_cast<float>(y) - image.cy) / image.fy, 1.F);
    return true;
}

__device__ float source_photo_score(
    const DeviceImage& reference, const Patch& patch,
    const int x, const int y, const DeviceSource& source,
    const float depth, const float3 normal) {
    const float plane_distance = dot3(normal, patch.center_ray) * depth;
    if (!(depth > 0.F) || dot3(normal, patch.center_ray) >= 0.F ||
        fabsf(plane_distance) < 1e-8F)
        return k_robust;
    float sum = 0.F;
    float sum_squared = 0.F;
    float numerator = 0.F;
    int count = 0;
    for (int dy = -k_half_window; dy <= k_half_window; dy += k_step) {
        for (int dx = -k_half_window; dx <= k_half_window; dx += k_step) {
            const float3 ray = make_vec(
                (static_cast<float>(x + dx) - reference.cx) / reference.fx,
                (static_cast<float>(y + dy) - reference.cy) / reference.fy,
                1.F);
            const float denominator = dot3(normal, ray);
            if (fabsf(denominator) < 1e-8F) return k_robust;
            const float3 reference_point =
                mul(ray, plane_distance / denominator);
            const float3 source_point = add(
                matrix_vector(source.rotation, reference_point),
                make_vec(
                    source.translation[0], source.translation[1],
                    source.translation[2]));
            if (!(source_point.z > 1e-6F)) return k_robust;
            const float source_x =
                source.image.fx * source_point.x / source_point.z +
                source.image.cx;
            const float source_y =
                source.image.fy * source_point.y / source_point.z +
                source.image.cy;
            const int mask_x = static_cast<int>(floorf(source_x + 0.5F));
            const int mask_y = static_cast<int>(floorf(source_y + 0.5F));
            if (mask_x < 0 || mask_y < 0 ||
                mask_x >= source.image.width ||
                mask_y >= source.image.height ||
                !foreground(source.image, mask_x, mask_y))
                return k_robust;
            float sample = 0.F;
            if (!sample_gray(source.image, source_x, source_y, sample))
                return k_robust;
            sum += sample;
            sum_squared += sample * sample;
            numerator += patch.texels[count++] * sample;
        }
    }
    const float mean = sum / static_cast<float>(k_texels);
    const float source_norm =
        sum_squared - static_cast<float>(k_texels) * mean * mean;
    const float denominator = patch.norm_squared * source_norm;
    if (!(denominator > 1e-16F)) return k_robust;
    const float ncc = fminf(
        1.F, fmaxf(-1.F, numerator * rsqrtf(denominator)));
    return fminf(2.F, 1.F - ncc);
}

__device__ float geometric_score(
    const DeviceImage& reference, const Patch& patch,
    const int x, const int y, const DeviceSource& source,
    const float depth) {
    if (source.image.depth == nullptr) return 4.F;
    const float3 reference_point = mul(patch.center_ray, depth);
    const float3 translation = make_vec(
        source.translation[0], source.translation[1],
        source.translation[2]);
    const float3 source_point =
        add(matrix_vector(source.rotation, reference_point), translation);
    if (!(source_point.z > 1e-6F)) return 4.F;
    const float u =
        source.image.fx * source_point.x / source_point.z + source.image.cx;
    const float v =
        source.image.fy * source_point.y / source_point.z + source.image.cy;
    const int center_x = static_cast<int>(floorf(u + 0.5F));
    const int center_y = static_cast<int>(floorf(v + 0.5F));
    float best = 1e30F;
    int best_x = -1;
    int best_y = -1;
    float best_depth = 0.F;
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            const int px = center_x + ox;
            const int py = center_y + oy;
            if (px < 0 || py < 0 || px >= source.image.width ||
                py >= source.image.height)
                continue;
            const float candidate =
                source.image.depth[
                    static_cast<std::size_t>(py) * source.image.width + px];
            if (!(candidate > 0.F)) continue;
            const float relative =
                fabsf(source_point.z - candidate) /
                fmaxf(source_point.z, candidate);
            if (relative > 0.03F) continue;
            const float pixel =
                hypotf(static_cast<float>(px) - u,
                       static_cast<float>(py) - v);
            const float score = relative + pixel * 1e-3F;
            if (score < best) {
                best = score;
                best_x = px;
                best_y = py;
                best_depth = candidate;
            }
        }
    }
    if (best_x < 0) return 4.F;
    const float3 source_back = make_vec(
        (static_cast<float>(best_x) - source.image.cx) /
                source.image.fx *
            best_depth,
        (static_cast<float>(best_y) - source.image.cy) /
                source.image.fy *
            best_depth,
        best_depth);
    const float3 reference_back = matrix_transpose_vector(
        source.rotation, sub(source_back, translation));
    if (!(reference_back.z > 1e-6F)) return 4.F;
    const float back_u =
        reference.fx * reference_back.x / reference_back.z + reference.cx;
    const float back_v =
        reference.fy * reference_back.y / reference_back.z + reference.cy;
    const float distance =
        hypotf(back_u - static_cast<float>(x),
               back_v - static_cast<float>(y));
    return fminf(4.F, sqrtf(distance * (distance + 2.F)));
}

__device__ float score_candidate(
    const DeviceRequest& request, const Patch& patch,
    const int x, const int y, const float depth, const float3 normal) {
    if (!(depth >= request.depth_min && depth <= request.depth_max))
        return k_robust;
    float scores[k_max_sources];
    unsigned count = 0;
    for (unsigned source_index = 0;
         source_index < request.source_count; ++source_index) {
        const DeviceSource& source = request.sources[source_index];
        const float photo = source_photo_score(
            request.reference, patch, x, y, source, depth, normal);
        if (!(photo < k_robust)) continue;
        float score = photo;
        if (request.use_geometric && request.geometric_weight > 0.F) {
            const float geometry = geometric_score(
                request.reference, patch, x, y, source, depth);
            if (!(geometry < 4.F)) continue;
            score += request.geometric_weight * geometry;
        }
        unsigned position = count;
        while (position > 0 && scores[position - 1] > score) {
            scores[position] = scores[position - 1];
            --position;
        }
        scores[position] = score;
        ++count;
    }
    unsigned required =
        request.min_patch_views > 0U ? request.min_patch_views : 1U;
    if (required > request.source_count)
        required = request.source_count;
    if (count < required) return k_robust;
    float sum = 0.F;
    for (unsigned i = 0; i < required; ++i) sum += scores[i];
    return sum / static_cast<float>(required);
}

__device__ float3 random_normal(
    std::uint32_t& state, const float3 center_ray) {
    float3 normal = make_vec(
        random01(state) * 2.F - 1.F,
        random01(state) * 2.F - 1.F,
        random01(state) * 2.F - 1.F);
    normal = normalized(normal);
    if (dot3(normal, center_ray) > 0.F) normal = mul(normal, -1.F);
    if (dot3(normal, center_ray) >= -0.1F)
        normal = normalized(mul(center_ray, -1.F));
    return normal;
}

__device__ float3 perturb_normal(
    std::uint32_t& state, const float3 normal,
    const float3 center_ray, const float range) {
    const float3 jitter = random_normal(state, center_ray);
    float3 candidate = normalized(add(normal, mul(jitter, range)));
    if (dot3(candidate, center_ray) > 0.F)
        candidate = mul(candidate, -1.F);
    return candidate;
}

__global__ void initialize_kernel(
    DeviceRequest request, const unsigned row_begin,
    const unsigned row_end) {
    const int x =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(
        row_begin + blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= request.reference.width ||
        y >= static_cast<int>(row_end))
        return;
    const std::size_t index =
        static_cast<std::size_t>(y) * request.reference.width + x;
    Patch patch;
    if (!fill_patch(request.reference, x, y, patch)) {
        request.depth[index] = 0.F;
        request.normal[index] = make_vec(0.F, 0.F, 0.F);
        request.confidence[index] = k_robust;
        return;
    }
    std::uint32_t state = hash32(
        request.random_seed ^
        static_cast<std::uint32_t>(index * 0x9e3779b9ULL));
    if (!(request.depth[index] > 0.F)) {
        if (!request.initialize_invalid) {
            request.confidence[index] = k_robust;
            return;
        }
        const float inverse_min = 1.F / request.depth_max;
        const float inverse_max = 1.F / request.depth_min;
        const float t = random01(state);
        request.depth[index] =
            1.F / (inverse_min + t * (inverse_max - inverse_min));
        request.normal[index] = random_normal(state, patch.center_ray);
    }
    request.normal[index] = normalized(request.normal[index]);
    request.confidence[index] = score_candidate(
        request, patch, x, y, request.depth[index],
        request.normal[index]);
}

__global__ void propagate_kernel(
    DeviceRequest request, const unsigned row_begin,
    const unsigned row_end, const unsigned iteration,
    const unsigned parity) {
    const int x =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(
        row_begin + blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= request.reference.width ||
        y >= static_cast<int>(row_end) ||
        ((static_cast<unsigned>(x + y) & 1U) != parity))
        return;
    const std::size_t index =
        static_cast<std::size_t>(y) * request.reference.width + x;
    if (!(request.depth[index] > 0.F)) return;
    Patch patch;
    if (!fill_patch(request.reference, x, y, patch)) return;

    float best_depth = request.depth[index];
    float3 best_normal = request.normal[index];
    float best_score = request.confidence[index];
    const int direction = (iteration & 1U) == 0U ? -1 : 1;
    for (int neighbor = 0; neighbor < 2; ++neighbor) {
        const int neighbor_x = x + (neighbor == 0 ? direction : 0);
        const int neighbor_y = y + (neighbor == 0 ? 0 : direction);
        if (neighbor_x < 0 || neighbor_y < 0 ||
            neighbor_x >= request.reference.width ||
            neighbor_y >= request.reference.height)
            continue;
        const std::size_t neighbor_index =
            static_cast<std::size_t>(neighbor_y) *
                request.reference.width +
            neighbor_x;
        const float neighbor_depth = request.depth[neighbor_index];
        if (!(neighbor_depth > 0.F)) continue;
        const float3 neighbor_normal = request.normal[neighbor_index];
        const float3 neighbor_ray = make_vec(
            (static_cast<float>(neighbor_x) - request.reference.cx) /
                request.reference.fx,
            (static_cast<float>(neighbor_y) - request.reference.cy) /
                request.reference.fy,
            1.F);
        const float plane_distance =
            dot3(neighbor_normal, neighbor_ray) * neighbor_depth;
        const float denominator =
            dot3(neighbor_normal, patch.center_ray);
        if (fabsf(denominator) < 1e-8F) continue;
        const float candidate_depth = plane_distance / denominator;
        const float candidate_score = score_candidate(
            request, patch, x, y, candidate_depth, neighbor_normal);
        if (candidate_score < best_score) {
            best_depth = candidate_depth;
            best_normal = neighbor_normal;
            best_score = candidate_score;
        }
    }

    std::uint32_t state = hash32(
        request.random_seed ^ (iteration + 1U) * 0x85ebca6bU ^
        (parity + 1U) * 0xc2b2ae35U ^
        static_cast<std::uint32_t>(index));
    float depth_range =
        best_depth * (request.use_geometric ? 0.05F : 0.5F);
    float normal_range = request.use_geometric ? 0.2F : 1.F;
    for (unsigned trial = 0; trial < request.random_iters; ++trial) {
        const float candidate_depth = fminf(
            request.depth_max,
            fmaxf(
                request.depth_min,
                best_depth +
                    (random01(state) * 2.F - 1.F) * depth_range));
        const float3 candidate_normal = perturb_normal(
            state, best_normal, patch.center_ray, normal_range);
        const float candidate_score = score_candidate(
            request, patch, x, y, candidate_depth, candidate_normal);
        if (candidate_score < best_score) {
            best_depth = candidate_depth;
            best_normal = candidate_normal;
            best_score = candidate_score;
        }
        depth_range *= 0.5F;
        normal_range *= 0.5F;
    }
    request.depth[index] = best_depth;
    request.normal[index] = best_normal;
    request.confidence[index] = best_score;
}

DeviceImage make_device_image(
    const HostImage& host, const float* gray,
    const std::uint8_t* mask, const float* depth) {
    return {
        static_cast<int>(host.width), static_cast<int>(host.height),
        host.fx, host.fy, host.cx, host.cy, gray, mask, depth};
}

}  // namespace

bool available(
    const int requested_device, std::string& device_name,
    std::string& error) {
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", error) ||
        count <= 0) {
        if (error.empty()) error = "no CUDA device is available";
        return false;
    }
    int device = requested_device;
    if (device < 0 &&
        !check(cudaGetDevice(&device), "cudaGetDevice", error))
        return false;
    if (device < 0 || device >= count) {
        error = "CUDA PatchMatch device ordinal is out of range";
        return false;
    }
    cudaDeviceProp properties{};
    if (!check(
            cudaGetDeviceProperties(&properties, device),
            "cudaGetDeviceProperties", error))
        return false;
    device_name = properties.name;
    return true;
}

bool run(const Request& request, std::string& error) {
    if (request.reference.gray == nullptr || request.depth == nullptr ||
        request.normal_xyz == nullptr || request.confidence == nullptr ||
        request.source_count == 0 ||
        request.source_count > k_max_sources) {
        error = "invalid CUDA PatchMatch request";
        return false;
    }
    int device = request.device;
    if (device >= 0 &&
        !check(cudaSetDevice(device), "cudaSetDevice", error))
        return false;

    const std::size_t pixel_count =
        static_cast<std::size_t>(request.reference.width) *
        request.reference.height;
    Buffer<float> reference_gray;
    Buffer<std::uint8_t> reference_mask;
    Buffer<float> depth;
    Buffer<float3> normal;
    Buffer<float> confidence;
    if (!reference_gray.upload(
            request.reference.gray, pixel_count, error) ||
        (request.reference.mask != nullptr &&
         !reference_mask.upload(
             request.reference.mask, pixel_count, error)) ||
        !depth.upload(request.depth, pixel_count, error) ||
        !confidence.upload(request.confidence, pixel_count, error))
        return false;

    std::vector<float3> packed_normals(pixel_count);
    for (std::size_t i = 0; i < pixel_count; ++i)
        packed_normals[i] = make_float3(
            request.normal_xyz[i * 3 + 0],
            request.normal_xyz[i * 3 + 1],
            request.normal_xyz[i * 3 + 2]);
    if (!normal.upload(packed_normals.data(), pixel_count, error))
        return false;

    std::vector<float> source_gray_host;
    std::vector<std::uint8_t> source_mask_host;
    std::vector<float> source_depth_host;
    std::vector<std::size_t> gray_offsets(request.source_count);
    std::vector<std::size_t> mask_offsets(request.source_count);
    std::vector<std::size_t> depth_offsets(request.source_count);
    std::vector<bool> has_mask(request.source_count, false);
    std::vector<bool> has_depth(request.source_count, false);
    for (unsigned i = 0; i < request.source_count; ++i) {
        const HostImage& image = request.sources[i].image;
        const std::size_t count =
            static_cast<std::size_t>(image.width) * image.height;
        gray_offsets[i] = source_gray_host.size();
        source_gray_host.insert(
            source_gray_host.end(), image.gray, image.gray + count);
        if (image.mask != nullptr) {
            has_mask[i] = true;
            mask_offsets[i] = source_mask_host.size();
            source_mask_host.insert(
                source_mask_host.end(), image.mask, image.mask + count);
        }
        if (image.depth != nullptr) {
            has_depth[i] = true;
            depth_offsets[i] = source_depth_host.size();
            source_depth_host.insert(
                source_depth_host.end(), image.depth, image.depth + count);
        }
    }

    Buffer<float> source_gray;
    Buffer<std::uint8_t> source_mask;
    Buffer<float> source_depth;
    if (!source_gray.upload(
            source_gray_host.data(), source_gray_host.size(), error) ||
        !source_mask.upload(
            source_mask_host.data(), source_mask_host.size(), error) ||
        !source_depth.upload(
            source_depth_host.data(), source_depth_host.size(), error))
        return false;

    std::vector<DeviceSource> device_sources(request.source_count);
    for (unsigned i = 0; i < request.source_count; ++i) {
        const HostSource& host = request.sources[i];
        DeviceSource& source = device_sources[i];
        source.image = make_device_image(
            host.image, source_gray.data() + gray_offsets[i],
            has_mask[i] ? source_mask.data() + mask_offsets[i] : nullptr,
            has_depth[i] ? source_depth.data() + depth_offsets[i] : nullptr);
        std::copy(
            std::begin(host.rotation), std::end(host.rotation),
            std::begin(source.rotation));
        std::copy(
            std::begin(host.translation), std::end(host.translation),
            std::begin(source.translation));
    }
    Buffer<DeviceSource> sources;
    if (!sources.upload(
            device_sources.data(), device_sources.size(), error))
        return false;

    DeviceRequest device_request;
    device_request.reference = make_device_image(
        request.reference, reference_gray.data(),
        request.reference.mask != nullptr ? reference_mask.data() : nullptr,
        nullptr);
    device_request.depth = depth.data();
    device_request.normal = normal.data();
    device_request.confidence = confidence.data();
    device_request.sources = sources.data();
    device_request.source_count = request.source_count;
    device_request.depth_min = request.depth_min;
    device_request.depth_max = request.depth_max;
    device_request.random_iters = request.random_iters;
    device_request.min_patch_views = request.min_patch_views;
    device_request.geometric_weight = request.geometric_weight;
    device_request.random_seed = request.random_seed;
    device_request.use_geometric = request.use_geometric;
    device_request.initialize_invalid = request.initialize_invalid;

    const dim3 block(16, 8);
    const unsigned rows = request.reference.height;
    for (unsigned row = 0; row < rows; row += k_rows_per_launch) {
        const unsigned row_end = std::min(rows, row + k_rows_per_launch);
        const dim3 grid(
            (request.reference.width + block.x - 1) / block.x,
            (row_end - row + block.y - 1) / block.y);
        initialize_kernel<<<grid, block>>>(
            device_request, row, row_end);
    }
    if (!check(cudaGetLastError(), "initialize kernel", error))
        return false;

    const unsigned iterations =
        request.use_geometric ? 1U : request.estimation_iters;
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        for (unsigned parity = 0; parity < 2; ++parity) {
            for (unsigned row = 0; row < rows;
                 row += k_rows_per_launch) {
                const unsigned row_end =
                    std::min(rows, row + k_rows_per_launch);
                const dim3 grid(
                    (request.reference.width + block.x - 1) / block.x,
                    (row_end - row + block.y - 1) / block.y);
                propagate_kernel<<<grid, block>>>(
                    device_request, row, row_end, iteration, parity);
            }
            if (!check(
                    cudaGetLastError(), "propagation kernel", error))
                return false;
        }
    }
    if (!check(
            cudaDeviceSynchronize(), "CUDA PatchMatch synchronize", error))
        return false;

    if (!depth.download(request.depth, pixel_count, error) ||
        !normal.download(packed_normals.data(), pixel_count, error) ||
        !confidence.download(request.confidence, pixel_count, error))
        return false;
    for (std::size_t i = 0; i < pixel_count; ++i) {
        request.normal_xyz[i * 3 + 0] = packed_normals[i].x;
        request.normal_xyz[i * 3 + 1] = packed_normals[i].y;
        request.normal_xyz[i * 3 + 2] = packed_normals[i].z;
    }
    return true;
}

}  // namespace photara::mvs::cuda_patchmatch
