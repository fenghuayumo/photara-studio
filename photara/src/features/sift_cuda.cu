#include "sift_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace photara::features {
namespace {

constexpr int kFilterWidthFactor = 4;
constexpr int kKernelMaxWidth = 33;
constexpr int kKernelMinWidth = 5;
constexpr int kMaxBlurRadius = (kKernelMaxWidth - 1) / 2;
constexpr double kTwoPi = 6.283185307179586476925286766559;

#define PHOTARA_SIFT_CHECK(expr)                                                 \
    do {                                                                         \
        const cudaError_t photara_sift_status = (expr);                          \
        if (photara_sift_status != cudaSuccess) {                                \
            throw std::runtime_error(                                            \
                std::string("CUDA SIFT: ") +                                     \
                cudaGetErrorString(photara_sift_status));                        \
        }                                                                        \
    } while (0)

struct Buffer {
    void* data = nullptr;
    std::size_t bytes = 0;

    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept : data(other.data), bytes(other.bytes) {
        other.data = nullptr;
        other.bytes = 0;
    }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this == &other) return *this;
        if (data) cudaFree(data);
        data = other.data;
        bytes = other.bytes;
        other.data = nullptr;
        other.bytes = 0;
        return *this;
    }
    ~Buffer() {
        if (data) cudaFree(data);
    }

    void reset() {
        if (data) cudaFree(data);
        data = nullptr;
        bytes = 0;
    }

    void ensure(std::size_t size) {
        if (size <= bytes) return;
        reset();
        if (size == 0) return;
        PHOTARA_SIFT_CHECK(cudaMalloc(&data, size));
        bytes = size;
    }

    template <class T>
    T* as(std::size_t count) {
        ensure(count * sizeof(T));
        return static_cast<T*>(data);
    }

    template <class T>
    T* get() const {
        return static_cast<T*>(data);
    }
};

struct DetKey {
    int x;
    int y;
    float dx;
    float dy;
    float ds;
};

struct OctaveShape {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t pixels;
};

struct LevelSlot {
    int octave = 0;
    int level = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    int count = 0;
    Buffer keys;
};

std::uint32_t aligned_width(std::uint32_t width) {
    return (width + 3U) / 4U * 4U;
}

std::vector<float> gaussian_kernel(float sigma) {
    const auto radius = static_cast<int>(
        std::ceil(static_cast<float>(kFilterWidthFactor) * sigma - 0.5F));
    int width = 2 * radius + 1;
    if (width > kKernelMaxWidth) width = kKernelMaxWidth;
    if (width < kKernelMinWidth) width = kKernelMinWidth;
    std::vector<float> kernel(static_cast<std::size_t>(width));
    const int half = width / 2;
    float sum = 0.0F;
    for (int i = 0; i <= half; ++i) {
        const float value =
            std::exp(-0.5F * static_cast<float>(i * i) / (sigma * sigma));
        kernel[static_cast<std::size_t>(half - i)] = value;
        kernel[static_cast<std::size_t>(half + i)] = value;
        sum += i == 0 ? value : 2.0F * value;
    }
    const float inverse = 1.0F / sum;
    for (auto& value : kernel) value *= inverse;
    return kernel;
}

__device__ int sift_clampi(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

__device__ float sift_fetch(const float* data, int index, int count) {
    return index >= 0 && index < count ? data[index] : 0.0F;
}

// One dynamic shared array for both blur kernels. A second extern __shared__
// declaration in this translation unit is a redefinition under nvcc.
extern __shared__ float sift_blur_tile[];

__global__ void convert_kernel(
    const std::uint8_t* input, float* output, int pixels, int source_stride,
    int destination_stride) {
    const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int row = blockIdx.y;
    if (col >= destination_stride) return;
    for (int i = 0; i < 4; ++i) {
        const int index = row * destination_stride + col + i;
        if (index >= pixels) break;
        const int source = row * source_stride + col + i;
        output[index] = static_cast<float>(input[source]) * (1.0F / 255.0F);
    }
}

__global__ void upsample_kernel(
    const float* source, float* destination, int src_width, int src_height,
    int dst_width, int dst_height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    if (x >= src_width || y >= dst_height || x * 2 + 1 >= dst_width) return;
    const int count = src_width * src_height;
    const int row = y >> 1;
    const int index = row * src_width + x;
    const int dst_index = y * dst_width + x * 2;
    // Linear fetches: the last column blends with the next row, and reads past
    // the buffer are zero. That is the documented tex1Dfetch edge behavior.
    if ((y & 1) != 0) {
        const float v11 = sift_fetch(source, index, count);
        const float v12 = sift_fetch(source, index + 1, count);
        const float v21 = sift_fetch(source, index + src_width, count);
        const float v22 = sift_fetch(source, index + src_width + 1, count);
        const float first = v21 * 0.5F + 0.5F * v11;
        const float second = v22 * 0.5F + 0.5F * v12;
        destination[dst_index] = first;
        destination[dst_index + 1] = first * 0.5F + second * 0.5F;
    } else {
        const float v1 = sift_fetch(source, index, count);
        const float v2 = sift_fetch(source, index + 1, count);
        destination[dst_index] = v1;
        destination[dst_index + 1] = v1 * 0.5F + v2 * 0.5F;
    }
}

__global__ void downsample_kernel(
    const float* source, float* destination, int src_width, int src_height,
    int dst_width, int dst_height, int scale) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_width || y >= dst_height) return;
    int src_col = x * scale;
    if (src_col >= src_width) src_col = src_width - 1;
    const int src_row = y * scale;
    float value = 0.0F;
    if (src_row >= 0 && src_row < src_height && src_width > 0)
        value = source[src_row * src_width + src_col];
    destination[y * dst_width + x] = value;
}

__global__ void blur_horizontal_kernel(
    const float* source, float* destination, int width, int height, int radius,
    const float* kernel) {
    const int tile_width = blockDim.x + 2 * radius;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    for (int i = threadIdx.x; i < tile_width; i += blockDim.x) {
        int sample = static_cast<int>(blockIdx.x * blockDim.x) + i - radius;
        if (sample < 0) sample = 0;
        else if (sample >= width) sample = width - 1;
        sift_blur_tile[i] = source[static_cast<std::size_t>(y) * width + sample];
    }
    __syncthreads();
    if (x >= width) return;
    float value = 0.0F;
    const int base = threadIdx.x + radius;
    for (int k = -radius; k <= radius; ++k)
        value += sift_blur_tile[base + k] * kernel[k + radius];
    destination[static_cast<std::size_t>(y) * width + x] = value;
}

__global__ void blur_vertical_kernel(
    const float* source, float* destination, int width, int height, int radius,
    const float* kernel) {
    const int tile_width = blockDim.x;
    const int tile_height = blockDim.y + 2 * radius;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    for (int i = threadIdx.y; i < tile_height; i += blockDim.y) {
        int sample = static_cast<int>(blockIdx.y * blockDim.y) + i - radius;
        if (sample < 0) sample = 0;
        else if (sample >= height) sample = height - 1;
        if (x < width)
            sift_blur_tile[i * tile_width + threadIdx.x] =
                source[static_cast<std::size_t>(sample) * width + x];
    }
    __syncthreads();
    if (x >= width || y >= height) return;
    float value = 0.0F;
    for (int k = -radius; k <= radius; ++k)
        value += sift_blur_tile[(threadIdx.y + radius + k) * tile_width + threadIdx.x] *
                 kernel[k + radius];
    destination[static_cast<std::size_t>(y) * width + x] = value;
}

__global__ void dog_kernel(
    const float* current, const float* previous, float* dog, float2* gradient,
    int width, int height, int write_gradient) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int count = width * height;
    const int index = y * width + x;
    const float value = current[index];
    dog[index] = value - previous[index];
    if (write_gradient == 0) return;
    const float dx = sift_fetch(current, index + 1, count) -
                     sift_fetch(current, index - 1, count);
    const float dy = sift_fetch(current, index + width, count) -
                     sift_fetch(current, index - width, count);
    const float magnitude = 0.5F * sqrtf(dx * dx + dy * dy);
    const float rotation = magnitude == 0.0F ? 0.0F : atan2f(dy, dx);
    gradient[index] = make_float2(magnitude, rotation);
}

__global__ void gradient_kernel(
    const float* image, float* gradient_magnitude, float* gradient_orientation,
    int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int count = width * height;
    const int index = y * width + x;
    const float dx = sift_fetch(image, index + 1, count) -
                     sift_fetch(image, index - 1, count);
    const float dy = sift_fetch(image, index + width, count) -
                     sift_fetch(image, index - width, count);
    const float magnitude = 0.5F * sqrtf(dx * dx + dy * dy);
    const float rotation = magnitude == 0.0F ? 0.0F : atan2f(dy, dx);
    gradient_magnitude[index] = magnitude;
    gradient_orientation[index] = rotation;
}

// DoG extremum with the same partial-pivot sub-pixel solve as sift_key.hlsl.
// dx/dy/ds are zero when the solver bails out but the pixel is still a key.
__device__ bool detect_extremum(
    const float* previous, const float* current, const float* next, int width,
    int height, int x, int y, float dog_threshold0, float dog_threshold,
    float edge_threshold, float& dx, float& dy, float& ds) {
    dx = 0.0F;
    dy = 0.0F;
    ds = 0.0F;
    if (y <= 0 || x <= 0 || y >= height - 1 || x >= width - 1) return false;
    const int index = y * width + x;
    const int row_above = index - width;
    const int row_center = index;
    const int row_below = index + width;

    const float center = current[row_center];
    if (fabsf(center) <= dog_threshold0) return false;

    float data[3][3];
    float datap[3][3];
    float datan[3][3];
    data[1][1] = center;
    data[1][0] = current[row_center - 1];
    data[1][2] = current[row_center + 1];
    float nmax = fmaxf(data[1][0], data[1][2]);
    float nmin = fminf(data[1][0], data[1][2]);
    if (center <= nmax && center >= nmin) return false;

    for (int pass = 0; pass < 2; ++pass) {
        const int r = pass == 0 ? 0 : 2;
        const int base = r == 0 ? row_above : row_below;
        for (int c = 0; c < 3; ++c) data[r][c] = current[base + c - 1];
        if (center > nmax) {
            nmax = fmaxf(nmax, fmaxf(data[r][0], fmaxf(data[r][1], data[r][2])));
            if (center < nmax) return false;
        } else {
            nmin = fminf(nmin, fminf(data[r][0], fminf(data[r][1], data[r][2])));
            if (center > nmin) return false;
        }
    }

    const float doubled = center * 2.0F;
    const float fxx = data[1][0] + data[1][2] - doubled;
    const float fyy = data[0][1] + data[2][1] - doubled;
    const float fxy = 0.25F * (data[2][2] + data[0][0] - data[2][0] - data[0][2]);
    const float det = fxx * fyy - fxy * fxy;
    const float trace = fxx + fyy;
    if (det <= 0.0F || trace * trace > edge_threshold * det) return false;

    for (int r = 0; r < 3; ++r) {
        const int base = r == 0 ? row_above : (r == 1 ? row_center : row_below);
        for (int c = 0; c < 3; ++c) datap[r][c] = previous[base + c - 1];
        if (center > nmax) {
            nmax = fmaxf(nmax, fmaxf(datap[r][0], fmaxf(datap[r][1], datap[r][2])));
            if (center < nmax) return false;
        } else {
            nmin = fminf(nmin, fminf(datap[r][0], fminf(datap[r][1], datap[r][2])));
            if (center > nmin) return false;
        }
    }
    for (int r = 0; r < 3; ++r) {
        const int base = r == 0 ? row_above : (r == 1 ? row_center : row_below);
        for (int c = 0; c < 3; ++c) datan[r][c] = next[base + c - 1];
        if (center > nmax) {
            nmax = fmaxf(nmax, fmaxf(datan[r][0], fmaxf(datan[r][1], datan[r][2])));
            if (center < nmax) return false;
        } else {
            nmin = fminf(nmin, fminf(datan[r][0], fminf(datan[r][1], datan[r][2])));
            if (center > nmin) return false;
        }
    }

    const float fx = 0.5F * (data[1][2] - data[1][0]);
    const float fy = 0.5F * (data[2][1] - data[0][1]);
    const float fs = 0.5F * (datan[1][1] - datap[1][1]);
    const float fss = datan[1][1] + datap[1][1] - doubled;
    const float fxs =
        0.25F * (datan[1][2] + datap[1][0] - datan[1][0] - datap[1][2]);
    const float fys =
        0.25F * (datan[2][1] + datap[0][1] - datan[0][1] - datap[2][1]);

    float4 row0 = fxx > 0.0F ? make_float4(fxx, fxy, fxs, -fx)
                             : make_float4(-fxx, -fxy, -fxs, fx);
    float4 row1 = fxy > 0.0F ? make_float4(fxy, fyy, fys, -fy)
                             : make_float4(-fxy, -fyy, -fys, fy);
    float4 row2 = fxs > 0.0F ? make_float4(fxs, fys, fss, -fs)
                             : make_float4(-fxs, -fys, -fss, fs);
    const float pivot = fmaxf(row0.x, fmaxf(row1.x, row2.x));
    bool offset_ok = true;
    if (pivot >= 1e-10F) {
        if (pivot == row1.x) {
            const float4 temp = row1;
            row1 = row0;
            row0 = temp;
        } else if (pivot == row2.x) {
            const float4 temp = row2;
            row2 = row0;
            row0 = temp;
        }
        row0.y /= row0.x;
        row0.z /= row0.x;
        row0.w /= row0.x;
        row1.y -= row1.x * row0.y;
        row1.z -= row1.x * row0.z;
        row1.w -= row1.x * row0.w;
        row2.y -= row2.x * row0.y;
        row2.z -= row2.x * row0.z;
        row2.w -= row2.x * row0.w;
        if (fabsf(row2.y) > fabsf(row1.y)) {
            const float4 temp = row2;
            row2 = row1;
            row1 = temp;
        }
        if (fabsf(row1.y) >= 1e-10F) {
            row1.z /= row1.y;
            row1.w /= row1.y;
            row2.z -= row2.y * row1.z;
            row2.w -= row2.y * row1.w;
            if (fabsf(row2.z) >= 1e-10F) {
                ds = row2.w / row2.z;
                dy = row1.w - ds * row1.z;
                dx = row0.w - ds * row0.z - dy * row0.y;
                offset_ok = fabsf(center + 0.5F * (dx * fx + dy * fy + ds * fs)) >
                                dog_threshold &&
                            fabsf(ds) < 1.0F && fabsf(dx) < 1.0F && fabsf(dy) < 1.0F;
            }
        }
    }
    return offset_ok;
}

__global__ void detect_kernel(
    const float* previous, const float* current, const float* next, int width,
    int height, float dog_threshold0, float dog_threshold, float edge_threshold,
    DetKey* keys, int capacity, int* counter) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    float dx, dy, ds;
    if (!detect_extremum(
            previous, current, next, width, height, x, y, dog_threshold0,
            dog_threshold, edge_threshold, dx, dy, ds))
        return;
    const int slot = atomicAdd(counter, 1);
    if (slot < capacity) {
        DetKey key;
        key.x = x;
        key.y = y;
        key.dx = dx;
        key.dy = dy;
        key.ds = ds;
        keys[slot] = key;
    }
}

__device__ float2 sample_gradient(
    const float* magnitude, const float* orientation, int width, int height,
    float x, float y) {
    const float fx = x - 0.5F;
    const float fy = y - 0.5F;
    const int ix = static_cast<int>(floorf(fx));
    const int iy = static_cast<int>(floorf(fy));
    const float tx = fx - static_cast<float>(ix);
    const float ty = fy - static_cast<float>(iy);
    const int x0 = sift_clampi(ix, 0, width - 1);
    const int x1 = sift_clampi(ix + 1, 0, width - 1);
    const int y0 = sift_clampi(iy, 0, height - 1);
    const int y1 = sift_clampi(iy + 1, 0, height - 1);
    const int a_index = y0 * width + x0;
    const int b_index = y0 * width + x1;
    const int c_index = y1 * width + x0;
    const int d_index = y1 * width + x1;
    const float2 a = make_float2(magnitude[a_index], orientation[a_index]);
    const float2 b = make_float2(magnitude[b_index], orientation[b_index]);
    const float2 c = make_float2(magnitude[c_index], orientation[c_index]);
    const float2 d = make_float2(magnitude[d_index], orientation[d_index]);
    const float2 ab = make_float2(a.x + (b.x - a.x) * tx, a.y + (b.y - a.y) * tx);
    const float2 cd = make_float2(c.x + (d.x - c.x) * tx, c.y + (d.y - c.y) * tx);
    return make_float2(ab.x + (cd.x - ab.x) * ty, ab.y + (cd.y - ab.y) * ty);
}

__global__ void orientation_kernel(
    const DetKey* keys, int count, const float* gradient_magnitude,
    const float* gradient_orientation, int width, int height,
    unsigned num_orientation, float sigma, float sigma_step,
    float gaussian_factor, float sample_factor, float* packed_out) {
    constexpr float kTenDegrees = 5.7295779513082320876798154814105F;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const DetKey key_in = keys[index];
    float x = static_cast<float>(key_in.x) + 0.5F + key_in.dx;
    float y = static_cast<float>(key_in.y) + 0.5F + key_in.dy;
    float scale = sigma * powf(sigma_step, key_in.ds);
    float* out = packed_out + static_cast<std::size_t>(index) * 4;
    if (num_orientation == 0) {
        out[0] = x;
        out[1] = y;
        out[2] = scale;
        out[3] = 0.0F;
        return;
    }

    float vote[37];
    for (int i = 0; i < 37; ++i) vote[i] = 0.0F;
    const float gsigma = scale * gaussian_factor;
    const float win = fabsf(scale) * sample_factor;
    const float dist_threshold = win * win + 0.5F;
    const float factor = -0.5F / (gsigma * gsigma);
    const float xmin = fmaxf(1.5F, floorf(x - win) + 0.5F);
    const float ymin = fmaxf(1.5F, floorf(y - win) + 0.5F);
    const float xmax = fminf(static_cast<float>(width) - 1.5F, floorf(x + win) + 0.5F);
    const float ymax = fminf(static_cast<float>(height) - 1.5F, floorf(y + win) + 0.5F);
    for (float sy = ymin; sy <= ymax; sy += 1.0F) {
        for (float sx = xmin; sx <= xmax; sx += 1.0F) {
            const float ddx = sx - x;
            const float ddy = sy - y;
            const float squared = ddx * ddx + ddy * ddy;
            if (squared >= dist_threshold) continue;
            const float2 sample = sample_gradient(
                gradient_magnitude, gradient_orientation, width, height, sx, sy);
            const float weight = sample.x * expf(squared * factor);
            int bin = static_cast<int>(floorf(sample.y * kTenDegrees));
            if (bin < 0) bin += 36;
            if (bin >= 0 && bin <= 36) vote[bin] += weight;
        }
    }
    constexpr float kOneThird = 1.0F / 3.0F;
    for (int pass = 0; pass < 6; ++pass) {
        vote[36] = vote[0];
        float previous = vote[35];
        for (int j = 0; j < 36; ++j) {
            const float smoothed = kOneThird * (previous + vote[j] + vote[j + 1]);
            previous = vote[j];
            vote[j] = smoothed;
        }
    }
    vote[36] = vote[0];

    unsigned packed = 0;
    if (num_orientation == 1) {
        int index_max = 0;
        float max_vote = vote[0];
        for (int i = 1; i < 36; ++i) {
            index_max = vote[i] > max_vote ? i : index_max;
            max_vote = fmaxf(max_vote, vote[i]);
        }
        const float pre = vote[index_max == 0 ? 35 : index_max - 1];
        const float next = vote[index_max + 1];
        const float off =
            0.5F * ((next - pre) / (max_vote + max_vote - next - pre));
        float fraction = (static_cast<float>(index_max) + 0.5F + off) / 36.0F;
        if (fraction < 0.0F) fraction += 1.0F;
        packed = (65535U << 16) |
                 (static_cast<unsigned>(floorf(fraction * 65535.0F)) & 0xFFFFU);
    } else {
        float max_vote = vote[0];
        for (int i = 1; i < 36; ++i) max_vote = fmaxf(max_vote, vote[i]);
        const float vote_threshold = max_vote * 0.8F;
        float previous = vote[35];
        float max_rot0 = 0.0F, max_rot1 = 0.0F;
        float max_vot0 = 0.0F, max_vot1 = 0.0F;
        int orientations = 0;
        for (int i = 0; i < 36; ++i) {
            const float next = vote[i + 1];
            if (vote[i] > vote_threshold && vote[i] > previous && vote[i] > next) {
                const float di =
                    0.5F * ((next - previous) / (vote[i] + vote[i] - next - previous));
                const float rotation = static_cast<float>(i) + di + 0.5F;
                const float weight = vote[i];
                if (weight > max_vot1) {
                    if (weight > max_vot0) {
                        max_vot1 = max_vot0;
                        max_rot1 = max_rot0;
                        max_vot0 = weight;
                        max_rot0 = rotation;
                    } else {
                        max_vot1 = weight;
                        max_rot1 = rotation;
                    }
                    ++orientations;
                }
            }
            previous = vote[i];
        }
        float first = max_rot0 / 36.0F;
        if (first < 0.0F) first += 1.0F;
        unsigned low = orientations == 0
                           ? 65535U
                           : static_cast<unsigned>(floorf(first * 65535.0F));
        unsigned high = 65535U;
        if (orientations > 1) {
            float second = max_rot1 / 36.0F;
            if (second < 0.0F) second += 1.0F;
            high = static_cast<unsigned>(floorf(second * 65535.0F));
        }
        packed = (high << 16) | low;
    }
    out[0] = x;
    out[1] = y;
    out[2] = scale;
    out[3] = __uint_as_float(packed);
}

__global__ void descriptor_kernel(
    const float* keys, int count, const float* gradient_magnitude,
    const float* gradient_orientation, int width, int height, float window_factor,
    float* descriptors) {
    constexpr float kPi = 3.14159265358979323846F;
    constexpr float kTwo = 6.28318530717958647692F;
    constexpr float kBinScale = 4.0F / 3.14159265358979323846F;
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned feature = index >> 4;
    if (feature >= static_cast<unsigned>(count)) return;
    const unsigned bin = index & 15U;
    const float ix = static_cast<float>(bin & 3U);
    const float iy = static_cast<float>(bin >> 2);
    const float key_x = keys[feature * 4 + 0];
    const float key_y = keys[feature * 4 + 1];
    const float key_scale = keys[feature * 4 + 2];
    const float key_angle = keys[feature * 4 + 3];

    const float spacing = fabsf(key_scale * window_factor);
    float sine, cosine;
    sincosf(key_angle, &sine, &cosine);
    const float angle = key_angle > kPi ? key_angle - kTwo : key_angle;
    const float cos_spacing = cosine * spacing;
    const float sin_spacing = sine * spacing;
    const float cos_over = cosine / spacing;
    const float sin_over = sine / spacing;
    const float offset_x = ix - 1.5F;
    const float offset_y = iy - 1.5F;
    const float point_x = cos_spacing * offset_x - sin_spacing * offset_y + key_x;
    const float point_y = cos_spacing * offset_y + sin_spacing * offset_x + key_y;
    const float box = fabsf(cos_spacing) + fabsf(sin_spacing);
    const float xmin = fmaxf(1.5F, floorf(point_x - box) + 0.5F);
    const float ymin = fmaxf(1.5F, floorf(point_y - box) + 0.5F);
    const float xmax =
        fminf(static_cast<float>(width) - 1.5F, floorf(point_x + box) + 0.5F);
    const float ymax =
        fminf(static_cast<float>(height) - 1.5F, floorf(point_y + box) + 0.5F);

    float bins[9];
    for (int i = 0; i < 9; ++i) bins[i] = 0.0F;
    for (float sy = ymin; sy <= ymax; sy += 1.0F) {
        for (float sx = xmin; sx <= xmax; sx += 1.0F) {
            const float ddx = sx - point_x;
            const float ddy = sy - point_y;
            const float nx = cos_over * ddx + sin_over * ddy;
            const float ny = cos_over * ddy - sin_over * ddx;
            const float ax = fabsf(nx);
            const float ay = fabsf(ny);
            if (ax < 1.0F && ay < 1.0F) {
                const float2 sample = sample_gradient(
                    gradient_magnitude, gradient_orientation, width, height, sx, sy);
                const float dnx = nx + offset_x;
                const float dny = ny + offset_y;
                const float weight = expf(-0.125F * (dnx * dnx + dny * dny)) *
                                     (1.0F - ax) * (1.0F - ay) * sample.x;
                float theta = (angle - sample.y) * kBinScale;
                if (theta < 0.0F) theta += 8.0F;
                const float lower = floorf(theta);
                const unsigned start = static_cast<unsigned>(lower);
                const float weight1 = lower + 1.0F - theta;
                const float weight2 = theta - lower;
                for (unsigned k = 0; k < 8; ++k) {
                    if (k == start) {
                        bins[k] += weight1 * weight;
                        bins[k + 1] += weight2 * weight;
                    }
                }
            }
        }
    }
    bins[0] += bins[8];
    float* out = descriptors + static_cast<std::size_t>(index) * 8;
    out[0] = bins[0];
    out[1] = bins[1];
    out[2] = bins[2];
    out[3] = bins[3];
    out[4] = bins[4];
    out[5] = bins[5];
    out[6] = bins[6];
    out[7] = bins[7];
}

__global__ void normalize_kernel(float* descriptors, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float values[128];
    float norm1 = 0.0F;
    float* row = descriptors + static_cast<std::size_t>(index) * 128;
    // Same 32 float4 dots as the Vulkan normalize shader, so the reduction
    // order stays aligned with that pass.
    for (int i = 0; i < 32; ++i) {
        const float x = row[i * 4 + 0];
        const float y = row[i * 4 + 1];
        const float z = row[i * 4 + 2];
        const float w = row[i * 4 + 3];
        values[i * 4 + 0] = x;
        values[i * 4 + 1] = y;
        values[i * 4 + 2] = z;
        values[i * 4 + 3] = w;
        norm1 += x * x + y * y + z * z + w * w;
    }
    norm1 = rsqrtf(norm1);
    float norm2 = 0.0F;
    for (int i = 0; i < 128; ++i) {
        values[i] = fminf(0.2F, values[i] * norm1);
        norm2 += values[i] * values[i];
    }
    norm2 = rsqrtf(norm2);
    for (int i = 0; i < 128; ++i) row[i] = values[i] * norm2;
}

struct Pyramid {
    int dog_level_num = 0;
    int level_min = 0;
    int level_max = 0;
    int level_num = 0;
    int level_ds = 0;
    float sigma0 = 0;
    float sigman = 0;
    float sigmak = 0;
    float sigma_step = 0;
    float dsigma0 = 0;
    std::vector<float> level_sigma;
    std::vector<float> sigma_inc;

    explicit Pyramid(const SiftGpuOptions& options) {
        dog_level_num = static_cast<int>(options.octave_layers);
        level_min = -1;
        level_max = dog_level_num + 1;
        level_num = level_max - level_min + 1;
        level_ds = std::min(level_min + dog_level_num, level_max);
        sigma0 = 1.6F * std::pow(2.0F, 1.0F / static_cast<float>(dog_level_num));
        sigman = 0.5F;
        sigmak = std::pow(2.0F, 1.0F / static_cast<float>(dog_level_num));
        sigma_step = sigmak;
        dsigma0 = sigma0 * std::sqrt(1.0F - 1.0F / (sigmak * sigmak));
        level_sigma.resize(static_cast<std::size_t>(level_num));
        for (int level = level_min; level <= level_max; ++level)
            level_sigma[static_cast<std::size_t>(level - level_min)] =
                sigma0 * std::pow(
                             2.0F, static_cast<float>(level) /
                                       static_cast<float>(dog_level_num));
        sigma_inc.resize(static_cast<std::size_t>(level_num - 1));
        for (int level = level_min + 1; level <= level_max; ++level)
            sigma_inc[static_cast<std::size_t>(level - level_min - 1)] =
                dsigma0 * std::pow(sigmak, static_cast<float>(level));
    }

    [[nodiscard]] float initial_smooth_sigma(int octave_min) const {
        const float sa = sigma0 * std::pow(
                                     2.0F, static_cast<float>(level_min) /
                                               static_cast<float>(dog_level_num));
        const float sb = sigman / std::pow(2.0F, static_cast<float>(octave_min));
        return sa > sb + 0.001F ? std::sqrt(sa * sa - sb * sb) : 0.0F;
    }

    [[nodiscard]] float skip_sigma() const {
        const float sa = sigma0 * std::pow(sigmak, static_cast<float>(level_min));
        const float sb =
            sigma0 * std::pow(sigmak, static_cast<float>(level_ds - dog_level_num));
        return sa > sb + 0.001F ? std::sqrt(sa * sa - sb * sb) : 0.0F;
    }
};

struct Plan {
    int octave_min = 0;
    int octave_num = 0;
    std::uint32_t first_width = 0;
    std::uint32_t first_height = 0;
};

Plan make_plan(
    const SiftGpuOptions& options, std::uint32_t width, std::uint32_t height) {
    Plan result;
    result.octave_min = options.first_octave;
    std::uint32_t working_width = width;
    std::uint32_t working_height = height;
    if (result.octave_min < 0) {
        working_width <<= static_cast<std::uint32_t>(-result.octave_min);
        working_height <<= static_cast<std::uint32_t>(-result.octave_min);
    } else {
        working_width >>= static_cast<std::uint32_t>(result.octave_min);
        working_height >>= static_cast<std::uint32_t>(result.octave_min);
    }
    const std::uint32_t maximum_dimension =
        options.maximum_image_dimension *
        (1U << static_cast<std::uint32_t>(std::max(0, -options.first_octave)));
    while (working_width > maximum_dimension || working_height > maximum_dimension) {
        ++result.octave_min;
        working_width = std::max(1U, working_width >> 1);
        working_height = std::max(1U, working_height >> 1);
    }
    const double smallest =
        static_cast<double>(std::min(working_width, working_height));
    result.octave_num = std::max(1, static_cast<int>(std::floor(std::log2(smallest))) - 3);
    result.first_width = working_width;
    result.first_height = working_height;
    return result;
}

}  // namespace

struct SiftCudaEngine::Impl {
    SiftGpuOptions options;
    Pyramid pyramid;
    bool device_ready = false;
    int device_index = 0;
    cudaStream_t stream = nullptr;
    Buffer image_bytes;
    Buffer input_float;
    Buffer scratch;
    Buffer carry;
    Buffer kernel;
    Buffer counter;
    Buffer packed;
    Buffer descriptor_keys;
    Buffer descriptors;
    std::vector<Buffer> gauss;
    std::vector<Buffer> dogs;
    std::vector<LevelSlot> slots;

    explicit Impl(const SiftGpuOptions& value)
        : options(value), pyramid(value), device_index(value.device_index) {
        if (options.maximum_features == 0 || options.maximum_image_dimension == 0 ||
            options.maximum_orientations == 0 || options.octave_layers == 0 ||
            options.maximum_orientations > 4 || options.octave_layers > 16)
            throw std::invalid_argument("Invalid SiftGPU limits");
        if (options.first_octave < -1 || options.first_octave > 4)
            throw std::invalid_argument("SiftGPU first_octave must be in [-1, 4]");
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0 ||
            device_index < 0 || device_index >= devices)
            return;
        PHOTARA_SIFT_CHECK(cudaSetDevice(device_index));
        PHOTARA_SIFT_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        // Gaussian construction needs only the previous/current levels. DoG
        // detection needs a three-level window around an extremum.
        gauss.resize(2);
        dogs.resize(3);
        device_ready = true;
    }

    ~Impl() {
        if (!device_ready && stream == nullptr) return;
        cudaSetDevice(device_index);
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

    void blur(float* destination, const float* source, int width, int height, float sigma) {
        if (!(sigma > 0.0F)) {
            if (destination != source) {
                PHOTARA_SIFT_CHECK(cudaMemcpyAsync(
                    destination, source,
                    static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                        sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));
            }
            return;
        }
        const std::vector<float> weights = gaussian_kernel(sigma);
        const int radius = static_cast<int>(weights.size() / 2);
        if (radius > kMaxBlurRadius)
            throw std::runtime_error("CUDA SIFT blur radius exceeds the kernel limit");
        PHOTARA_SIFT_CHECK(cudaMemcpyAsync(
            kernel.as<float>(weights.size()), weights.data(),
            weights.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        const dim3 horizontal_block(128);
        const dim3 horizontal_grid((width + 127) / 128, height);
        const std::size_t horizontal_shared =
            static_cast<std::size_t>(128 + 2 * radius) * sizeof(float);
        blur_horizontal_kernel<<<horizontal_grid, horizontal_block, horizontal_shared,
                                 stream>>>(
            source, scratch.get<float>(), width, height, radius, kernel.get<float>());
        const dim3 vertical_block(32, 8);
        const dim3 vertical_grid((width + 31) / 32, (height + 7) / 8);
        const std::size_t vertical_shared =
            static_cast<std::size_t>(32 * (8 + 2 * radius)) * sizeof(float);
        blur_vertical_kernel<<<vertical_grid, vertical_block, vertical_shared, stream>>>(
            scratch.get<float>(), destination, width, height, radius, kernel.get<float>());
    }

    int detect_level(
        LevelSlot& slot, const float* previous, const float* current, const float* next,
        float dog_threshold0, float dog_threshold, float edge_threshold) {
        const int width = static_cast<int>(slot.width);
        const int height = static_cast<int>(slot.height);
        auto launch = [&](int capacity) {
            PHOTARA_SIFT_CHECK(cudaMemsetAsync(counter.get<int>(), 0, sizeof(int), stream));
            const dim3 block(16, 16);
            const dim3 grid((width + 15) / 16, (height + 15) / 16);
            detect_kernel<<<grid, block, 0, stream>>>(
                previous, current, next, width, height, dog_threshold0, dog_threshold,
                edge_threshold, slot.keys.get<DetKey>(), capacity, counter.get<int>());
        };
        int capacity = static_cast<int>(slot.keys.bytes / sizeof(DetKey));
        if (capacity < 4096) {
            slot.keys.as<DetKey>(4096);
            capacity = 4096;
        }
        int count = 0;
        launch(capacity);
        PHOTARA_SIFT_CHECK(cudaMemcpyAsync(
            &count, counter.get<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream));
        PHOTARA_SIFT_CHECK(cudaStreamSynchronize(stream));
        if (count > capacity) {
            slot.keys.as<DetKey>(static_cast<std::size_t>(count));
            capacity = count;
            launch(capacity);
            PHOTARA_SIFT_CHECK(cudaMemcpyAsync(
                &count, counter.get<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream));
            PHOTARA_SIFT_CHECK(cudaStreamSynchronize(stream));
            if (count > capacity)
                throw std::runtime_error("CUDA SIFT keypoint count exceeded the compacted buffer");
        }
        if (count <= 0) return 0;
        std::vector<DetKey> host(static_cast<std::size_t>(count));
        PHOTARA_SIFT_CHECK(cudaMemcpyAsync(
            host.data(), slot.keys.get<DetKey>(), host.size() * sizeof(DetKey),
            cudaMemcpyDeviceToHost, stream));
        PHOTARA_SIFT_CHECK(cudaStreamSynchronize(stream));
        std::sort(host.begin(), host.end(), [](const DetKey& a, const DetKey& b) {
            if (a.y != b.y) return a.y < b.y;
            return a.x < b.x;
        });
        PHOTARA_SIFT_CHECK(cudaMemcpyAsync(
            slot.keys.get<DetKey>(), host.data(), host.size() * sizeof(DetKey),
            cudaMemcpyHostToDevice, stream));
        // The sorted staging vector has to stay alive until the copy finishes.
        PHOTARA_SIFT_CHECK(cudaStreamSynchronize(stream));
        return count;
    }

    FeatureSet extract(
        std::span<const std::uint8_t> pixels, std::uint32_t width, std::uint32_t height,
        std::size_t row_stride) {
        if (!device_ready)
            throw std::runtime_error("CUDA SIFT extractor is not available");
        if (row_stride == 0) row_stride = width;
        if (width < 8 || height < 8 || row_stride < width ||
            pixels.size() < row_stride * static_cast<std::size_t>(height))
            throw std::invalid_argument("Invalid grayscale image view");
        PHOTARA_SIFT_CHECK(cudaSetDevice(device_index));

        std::vector<std::uint8_t> contiguous;
        const std::uint8_t* data = pixels.data();
        if (row_stride != width) {
            contiguous.resize(static_cast<std::size_t>(width) * height);
            for (std::uint32_t row = 0; row < height; ++row) {
                std::copy_n(
                    pixels.data() + static_cast<std::size_t>(row) * row_stride, width,
                    contiguous.data() + static_cast<std::size_t>(row) * width);
            }
            data = contiguous.data();
        }
        const std::uint32_t effective_width = std::max(4U, width & ~3U);
        const Plan plan = make_plan(options, effective_width, height);
        const int octave_num = plan.octave_num;
        const int dog_levels = pyramid.dog_level_num;
        const std::size_t level_total =
            static_cast<std::size_t>(octave_num) * static_cast<std::size_t>(dog_levels);

        std::vector<OctaveShape> shapes(static_cast<std::size_t>(octave_num));
        {
            std::uint32_t w = plan.first_width;
            std::uint32_t h = plan.first_height;
            for (int octave = 0; octave < octave_num; ++octave) {
                const std::uint32_t aligned = aligned_width(w);
                shapes[static_cast<std::size_t>(octave)] =
                    OctaveShape{aligned, h, aligned * h};
                w = std::max(1U, w >> 1);
                h = std::max(1U, h >> 1);
            }
        }
        const std::size_t max_pixels = shapes.front().pixels;
        for (auto& buffer : gauss)
            buffer.as<float>(max_pixels);
        for (auto& buffer : dogs)
            buffer.as<float>(max_pixels);
        scratch.as<float>(max_pixels);
        carry.as<float>(max_pixels);
        kernel.as<float>(kKernelMaxWidth);
        counter.as<int>(1);
        if (slots.size() < level_total) slots.resize(level_total);

        const std::size_t image_bytes_n = static_cast<std::size_t>(width) * height;
        PHOTARA_SIFT_CHECK(cudaMemcpyAsync(
            image_bytes.as<std::uint8_t>(image_bytes_n), data, image_bytes_n,
            cudaMemcpyHostToDevice, stream));
        input_float.as<float>(static_cast<std::size_t>(effective_width) * height);
        {
            const dim3 block(128);
            const dim3 grid(
                (effective_width / 4 + 127) / 128, height);
            convert_kernel<<<grid, block, 0, stream>>>(
                image_bytes.get<std::uint8_t>(), input_float.get<float>(),
                static_cast<int>(effective_width) * static_cast<int>(height),
                static_cast<int>(width), static_cast<int>(effective_width));
        }

        const float dog_threshold = options.peak_threshold;
        const float dog_threshold0 = 0.8F * dog_threshold;
        const float edge_threshold = (options.edge_threshold + 1.0F) *
                                     (options.edge_threshold + 1.0F) /
                                     options.edge_threshold;
        const int source_index = pyramid.level_ds - pyramid.level_min;

        for (int octave = 0; octave < octave_num; ++octave) {
            const auto& shape = shapes[static_cast<std::size_t>(octave)];
            const int w = static_cast<int>(shape.width);
            const int h = static_cast<int>(shape.height);
            float* base = gauss[0].get<float>();
            if (octave == 0) {
                if (plan.octave_min < 0) {
                    const dim3 block(128);
                    const dim3 grid(
                        (static_cast<int>(effective_width) + 127) / 128, h);
                    upsample_kernel<<<grid, block, 0, stream>>>(
                        input_float.get<float>(), base, static_cast<int>(effective_width),
                        static_cast<int>(height), w, h);
                } else {
                    const dim3 block(16, 16);
                    const dim3 grid((w + 15) / 16, (h + 15) / 16);
                    downsample_kernel<<<grid, block, 0, stream>>>(
                        input_float.get<float>(), base, static_cast<int>(effective_width),
                        static_cast<int>(height), w, h,
                        1 << plan.octave_min);
                }
                blur(base, base, w, h, pyramid.initial_smooth_sigma(plan.octave_min));
            } else {
                const auto& previous = shapes[static_cast<std::size_t>(octave - 1)];
                const dim3 block(16, 16);
                const dim3 grid((w + 15) / 16, (h + 15) / 16);
                downsample_kernel<<<grid, block, 0, stream>>>(
                    carry.get<float>(), base, static_cast<int>(previous.width),
                    static_cast<int>(previous.height), w, h, 2);
                blur(base, base, w, h, pyramid.skip_sigma());
            }

            for (int level = 1; level < pyramid.level_num; ++level) {
                float* destination =
                    gauss[static_cast<std::size_t>(level & 1)].get<float>();
                const float* source =
                    gauss[static_cast<std::size_t>((level - 1) & 1)].get<float>();
                blur(destination, source, w, h,
                     pyramid.sigma_inc[static_cast<std::size_t>(level - 1)]);
                const dim3 block(16, 16);
                const dim3 grid((w + 15) / 16, (h + 15) / 16);
                const int dog = (level - 1) % 3;
                dog_kernel<<<grid, block, 0, stream>>>(
                    destination, source,
                    dogs[static_cast<std::size_t>(dog)].get<float>(), nullptr,
                    w, h, 0);
                if (level == source_index) {
                    PHOTARA_SIFT_CHECK(cudaMemcpyAsync(
                        carry.get<float>(), destination,
                        static_cast<std::size_t>(shape.pixels) * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream));
                }
                if (level >= 3 && level - 3 < dog_levels) {
                    const int feature_level = level - 3;
                    LevelSlot& slot = slots[
                        static_cast<std::size_t>(octave) * dog_levels +
                        static_cast<std::size_t>(feature_level)];
                    slot.octave = octave;
                    slot.level = feature_level;
                    slot.width = shape.width;
                    slot.height = shape.height;
                    slot.count = detect_level(
                        slot,
                        dogs[static_cast<std::size_t>(feature_level % 3)].get<float>(),
                        dogs[static_cast<std::size_t>((feature_level + 1) % 3)].get<float>(),
                        dogs[static_cast<std::size_t>((feature_level + 2) % 3)].get<float>(),
                        dog_threshold0, dog_threshold, edge_threshold);
                }
            }
        }
        PHOTARA_SIFT_CHECK(cudaStreamSynchronize(stream));
        PHOTARA_SIFT_CHECK(cudaGetLastError());

        std::vector<LevelSlot*> kept;
        {
            long long accumulated = 0;
            const long long threshold = static_cast<long long>(options.maximum_features);
            for (std::size_t index = level_total; index-- > 0;) {
                if (threshold > 0 && accumulated > threshold) continue;
                accumulated += slots[index].count;
                if (slots[index].count > 0) kept.push_back(&slots[index]);
            }
            std::reverse(kept.begin(), kept.end());
        }

        FeatureSet result;
        result.image_width = width;
        result.image_height = height;
        result.descriptor_dimension = 128;
        result.metric = DescriptorMetric::l2;
        result.extractor_name = "siftgpu";
        if (kept.empty()) return result;

        struct Expanded {
            LevelSlot* slot;
            std::vector<float> values;
            std::vector<float> descriptors;
        };
        std::vector<Expanded> expanded(kept.size());
        for (std::size_t i = 0; i < kept.size(); ++i)
            expanded[i].slot = kept[i];

        // Gradients are the dominant SIFT allocation (float2 for every pixel
        // in every detected level). Rebuild the Gaussian pyramid and consume
        // one reusable gradient buffer per level instead of retaining all
        // octave gradients until descriptor generation.
        std::size_t total_features = 0;
        const double orientation_factor = kTwoPi / 65535.0;
        for (int octave = 0; octave < octave_num; ++octave) {
            const auto& shape = shapes[static_cast<std::size_t>(octave)];
            const int w = static_cast<int>(shape.width);
            const int h = static_cast<int>(shape.height);
            float* base = gauss[0].get<float>();
            if (octave == 0) {
                if (plan.octave_min < 0) {
                    const dim3 block(128);
                    const dim3 grid(
                        (static_cast<int>(effective_width) + 127) / 128, h);
                    upsample_kernel<<<grid, block, 0, stream>>>(
                        input_float.get<float>(), base,
                        static_cast<int>(effective_width), static_cast<int>(height),
                        w, h);
                } else {
                    const dim3 block(16, 16);
                    const dim3 grid((w + 15) / 16, (h + 15) / 16);
                    downsample_kernel<<<grid, block, 0, stream>>>(
                        input_float.get<float>(), base,
                        static_cast<int>(effective_width), static_cast<int>(height),
                        w, h, 1 << plan.octave_min);
                }
                blur(base, base, w, h,
                     pyramid.initial_smooth_sigma(plan.octave_min));
            } else {
                const auto& previous = shapes[static_cast<std::size_t>(octave - 1)];
                const dim3 block(16, 16);
                const dim3 grid((w + 15) / 16, (h + 15) / 16);
                downsample_kernel<<<grid, block, 0, stream>>>(
                    carry.get<float>(), base, static_cast<int>(previous.width),
                    static_cast<int>(previous.height), w, h, 2);
                blur(base, base, w, h, pyramid.skip_sigma());
            }
            auto process_level = [&](const int feature_level,
                                     const float* gaussian) {
                LevelSlot* slot = &slots[
                    static_cast<std::size_t>(octave) * dog_levels +
                    static_cast<std::size_t>(feature_level)];
                const auto selected = std::find(kept.begin(), kept.end(), slot);
                if (selected == kept.end()) return;
                Expanded& output = expanded[
                    static_cast<std::size_t>(selected - kept.begin())];
                const dim3 block(16, 16);
                const dim3 grid((w + 15) / 16, (h + 15) / 16);
                // Detection is complete before this second pyramid pass, so two
                // DoG ring buffers can hold the gradient planes without adding a
                // separate two-float allocation for every pixel.
                float* gradient_magnitude = dogs[0].get<float>();
                float* gradient_orientation = dogs[1].get<float>();
                gradient_kernel<<<grid, block, 0, stream>>>(
                    gaussian, gradient_magnitude, gradient_orientation, w, h);

                const int count = slot->count;
                float* packed_device = packed.as<float>(
                    static_cast<std::size_t>(count) * 4);
                orientation_kernel<<<(count + 63) / 64, 64, 0, stream>>>(
                    slot->keys.get<DetKey>(), count, gradient_magnitude,
                    gradient_orientation, w, h, options.maximum_orientations,
                    pyramid.level_sigma[
                        static_cast<std::size_t>(feature_level + 1)],
                    pyramid.sigma_step, 1.5F, 3.0F, packed_device);
                std::vector<float> packed_host(
                    static_cast<std::size_t>(count) * 4);
                PHOTARA_SIFT_CHECK(cudaMemcpyAsync(
                    packed_host.data(), packed_device,
                    packed_host.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream));
                PHOTARA_SIFT_CHECK(cudaStreamSynchronize(stream));

                auto& values = output.values;
                values.reserve(static_cast<std::size_t>(count) * 8);
                for (int feature = 0; feature < count; ++feature) {
                    const float x = packed_host[feature * 4 + 0];
                    const float y = packed_host[feature * 4 + 1];
                    const float scale = packed_host[feature * 4 + 2];
                    std::uint32_t bits = 0;
                    std::memcpy(
                        &bits, &packed_host[feature * 4 + 3], sizeof(bits));
                    const std::uint32_t first = bits & 0xFFFFU;
                    const std::uint32_t second = bits >> 16;
                    if (first == 65535U) continue;
                    values.insert(values.end(), {
                        x, y, scale,
                        static_cast<float>(orientation_factor * first)});
                    if (second != 65535U && second != first) {
                        values.insert(values.end(), {
                            x, y, scale,
                            static_cast<float>(orientation_factor * second)});
                    }
                }
                const int oriented_count =
                    static_cast<int>(values.size() / 4);
                total_features += static_cast<std::size_t>(oriented_count);
                if (oriented_count == 0) return;
                float* key_device = descriptor_keys.as<float>(values.size());
                PHOTARA_SIFT_CHECK(cudaMemcpyAsync(
                    key_device, values.data(), values.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream));
                float* descriptor_device = descriptors.as<float>(
                    static_cast<std::size_t>(oriented_count) * 128);
                const int threads = oriented_count * 16;
                descriptor_kernel<<<(threads + 63) / 64, 64, 0, stream>>>(
                    key_device, oriented_count, gradient_magnitude,
                    gradient_orientation, w, h, 3.0F, descriptor_device);
                normalize_kernel<<<(oriented_count + 63) / 64, 64, 0, stream>>>(
                    descriptor_device, oriented_count);
                output.descriptors.resize(
                    static_cast<std::size_t>(oriented_count) * 128);
                PHOTARA_SIFT_CHECK(cudaMemcpyAsync(
                    output.descriptors.data(), descriptor_device,
                    output.descriptors.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream));
                PHOTARA_SIFT_CHECK(cudaStreamSynchronize(stream));
            };
            for (int level = 1; level < pyramid.level_num; ++level) {
                float* destination =
                    gauss[static_cast<std::size_t>(level & 1)].get<float>();
                const float* source =
                    gauss[static_cast<std::size_t>((level - 1) & 1)].get<float>();
                blur(destination, source, w, h,
                     pyramid.sigma_inc[static_cast<std::size_t>(level - 1)]);
                if (level == source_index) {
                    PHOTARA_SIFT_CHECK(cudaMemcpyAsync(
                        carry.get<float>(), destination,
                        static_cast<std::size_t>(shape.pixels) * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream));
                }
                if (level <= dog_levels)
                    process_level(level - 1, destination);
            }
        }
        PHOTARA_SIFT_CHECK(cudaGetLastError());
        if (options.maximum_features > 0) {
            std::size_t total = total_features;
            std::size_t index = 0;
            while (total > options.maximum_features && index < expanded.size()) {
                const std::size_t count = expanded[index].values.size() / 4;
                if (count > 0 && total - count > options.maximum_features) {
                    total -= count;
                    expanded[index].values.clear();
                    expanded[index].descriptors.clear();
                } else {
                    break;
                }
                ++index;
            }
            total_features = total;
        }
        if (total_features == 0) return result;

        const float octave_scale_base =
            plan.octave_min >= 0
                ? static_cast<float>(1 << plan.octave_min)
                : 1.0F / static_cast<float>(1 << -plan.octave_min);
        result.keypoints.reserve(total_features);
        result.descriptors.reserve(total_features * 128);
        for (const auto& level : expanded) {
            const float octave_scale =
                octave_scale_base *
                static_cast<float>(1 << level.slot->octave);
            for (std::size_t feature = 0; feature < level.values.size() / 4; ++feature) {
                const float x = level.values[feature * 4 + 0];
                const float y = level.values[feature * 4 + 1];
                const float scale = level.values[feature * 4 + 2];
                const float orientation = level.values[feature * 4 + 3];
                double mirrored =
                    std::fmod(kTwoPi - static_cast<double>(orientation), kTwoPi);
                if (mirrored < 0.0) mirrored += kTwoPi;
                result.keypoints.push_back(
                    {octave_scale * (x - 0.5F) + 0.5F,
                     octave_scale * (y - 0.5F) + 0.5F, octave_scale * scale,
                     static_cast<float>(mirrored), 0.0F});
            }
            result.descriptors.insert(
                result.descriptors.end(), level.descriptors.begin(),
                level.descriptors.end());
        }
        return result;
    }
};

bool sift_cuda_device_available(int device_index) {
    int devices = 0;
    return device_index >= 0 && cudaGetDeviceCount(&devices) == cudaSuccess &&
           device_index < devices;
}

SiftCudaEngine::SiftCudaEngine(const SiftGpuOptions& options)
    : impl_(std::make_unique<Impl>(options)) {}

SiftCudaEngine::~SiftCudaEngine() = default;

bool SiftCudaEngine::available() const noexcept {
    return impl_ && impl_->device_ready;
}

FeatureSet SiftCudaEngine::extract(
    std::span<const std::uint8_t> pixels, std::uint32_t width, std::uint32_t height,
    std::size_t row_stride) {
    return impl_->extract(pixels, width, height, row_stride);
}

}  // namespace photara::features
