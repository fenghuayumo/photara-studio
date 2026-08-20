#include "cuda_ops.hpp"
#include "fused_ssim.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <stdexcept>

namespace aetherscan::splat::detail {
namespace {

constexpr unsigned k_threads = 256;
constexpr unsigned k_reduced_adam_threads = 64;
constexpr unsigned k_geometry_summary_terms = 6;
// A 32x8 image-space CTA shares the integer reference tile and all fixed
// half-pixel samples needed by the 7x7 NCC patches of its 256 output pixels.
constexpr unsigned k_multi_view_block_x = 32;
constexpr unsigned k_multi_view_block_y = 8;
constexpr int k_ncc_reference_padding = 2;
constexpr int k_ncc_reference_tile_width =
    static_cast<int>(k_multi_view_block_x) +
    2 * k_ncc_reference_padding;
constexpr int k_ncc_reference_tile_height =
    static_cast<int>(k_multi_view_block_y) +
    2 * k_ncc_reference_padding;
constexpr int k_ncc_radius = 3;
constexpr int k_ncc_reference_sample_width =
    2 * static_cast<int>(k_multi_view_block_x) +
    2 * k_ncc_radius - 1;
constexpr int k_ncc_reference_sample_height =
    2 * static_cast<int>(k_multi_view_block_y) +
    2 * k_ncc_radius - 1;

void check_cuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
}

__device__ float sigmoid(const float value) {
    return 1.F / (1.F + expf(-value));
}

__global__ void geometry_distribution_summary_kernel(
    const float* log_scales, const float* opacity_logits,
    float* terms, const std::size_t count) {
    __shared__ float partial[k_geometry_summary_terms][k_threads];
    float local[k_geometry_summary_terms]{};
    for (std::size_t index =
             blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float opacity = sigmoid(opacity_logits[index]);
        const float sx = log_scales[3 * index];
        const float sy = log_scales[3 * index + 1];
        const float sz = log_scales[3 * index + 2];
        const float log_scale = (sx + sy + sz) / 3.F;
        const float anisotropy =
            fmaxf(sx, fmaxf(sy, sz)) - fminf(sx, fminf(sy, sz));
        local[0] += opacity;
        local[1] += opacity * opacity;
        local[2] += log_scale;
        local[3] += log_scale * log_scale;
        local[4] += anisotropy;
        local[5] += anisotropy * anisotropy;
    }
    for (unsigned term = 0; term < k_geometry_summary_terms; ++term)
        partial[term][threadIdx.x] = local[term];
    __syncthreads();
    for (unsigned stride = k_threads / 2; stride != 0; stride >>= 1U) {
        if (threadIdx.x < stride)
            for (unsigned term = 0;
                 term < k_geometry_summary_terms; ++term)
                partial[term][threadIdx.x] +=
                    partial[term][threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (unsigned term = 0; term < k_geometry_summary_terms; ++term)
            atomicAdd(terms + term, partial[term][0]);
}

__global__ void unpack_training_pixels_kernel(
    const unsigned* rgba, float* rgb, float* gray, float* mask,
    const std::size_t pixels) {
    const std::size_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    const unsigned value = rgba[pixel];
    constexpr float inverse_255 = 1.F / 255.F;
    const float red =
        static_cast<float>(value & 0xffU) * inverse_255;
    const float green =
        static_cast<float>((value >> 8U) & 0xffU) * inverse_255;
    const float blue =
        static_cast<float>((value >> 16U) & 0xffU) * inverse_255;
    rgb[pixel] = red;
    rgb[pixels + pixel] = green;
    rgb[2 * pixels + pixel] = blue;
    if (gray)
        gray[pixel] = 0.299F * red + 0.587F * green + 0.114F * blue;
    if (mask)
        mask[pixel] =
            static_cast<float>((value >> 24U) & 0xffU) * inverse_255;
}

__global__ void activate_kernel(
    const float* log_scales, const float* raw_quaternions,
    const float* opacity_logits, const float* filter_3d,
    float* scales, float* quaternions, float* opacities,
    const std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float determinant_ratio = 1.F;
    const float filter_squared = filter_3d != nullptr
        ? filter_3d[index] * filter_3d[index]
        : 0.F;
    for (int axis = 0; axis < 3; ++axis) {
        const std::size_t offset = 3 * index + axis;
        const float raw_scale = expf(log_scales[offset]);
        const float filtered_scale = sqrtf(
            raw_scale * raw_scale + filter_squared);
        scales[offset] = filtered_scale;
        determinant_ratio *= raw_scale / filtered_scale;
    }
    const float w = raw_quaternions[4 * index + 0];
    const float x = raw_quaternions[4 * index + 1];
    const float y = raw_quaternions[4 * index + 2];
    const float z = raw_quaternions[4 * index + 3];
    const float inverse_norm = rsqrtf(fmaxf(w * w + x * x + y * y + z * z, 1e-20F));
    quaternions[4 * index + 0] = w * inverse_norm;
    quaternions[4 * index + 1] = x * inverse_norm;
    quaternions[4 * index + 2] = y * inverse_norm;
    quaternions[4 * index + 3] = z * inverse_norm;
    opacities[index] =
        sigmoid(opacity_logits[index]) * determinant_ratio;
}

__global__ void bake_3d_filter_kernel(
    float* log_scales, float* opacity_logits, const float* filter_3d,
    const std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float filter_squared =
        filter_3d[index] * filter_3d[index];
    float determinant_ratio = 1.F;
    for (int axis = 0; axis < 3; ++axis) {
        const std::size_t offset = 3 * index + axis;
        const float raw_scale = expf(log_scales[offset]);
        const float filtered_scale = sqrtf(
            raw_scale * raw_scale + filter_squared);
        determinant_ratio *= raw_scale / filtered_scale;
        log_scales[offset] = logf(fmaxf(filtered_scale, 1e-20F));
    }
    const float filtered_opacity = fminf(fmaxf(
        sigmoid(opacity_logits[index]) * determinant_ratio,
        1e-12F), 1.F - 1e-12F);
    opacity_logits[index] =
        logf(filtered_opacity / (1.F - filtered_opacity));
}

__global__ void adc_plus_prune_kernel(
    const float* means, const float* log_scales,
    const float* quaternions, const float* opacity_logits,
    const float* sh, const std::size_t sh_stride,
    unsigned char* keep, unsigned char* hard_prune, float* opacities,
    const std::size_t count, const float minimum_opacity,
    const float maximum_bounds, const float center_x,
    const float center_y, const float center_z) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float opacity = sigmoid(opacity_logits[index]);
    opacities[index] = opacity;
    bool bad = !isfinite(opacity_logits[index]);
    float maximum_scale = 0.F;
    for (int axis = 0; axis < 3; ++axis) {
        const float mean = means[3 * index + axis];
        const float log_scale = log_scales[3 * index + axis];
        bad = bad || !isfinite(mean) || !isfinite(log_scale);
        maximum_scale = fmaxf(maximum_scale, expf(log_scale));
    }
    for (int component = 0; component < 4; ++component)
        bad = bad || !isfinite(quaternions[4 * index + component]);
    for (std::size_t component = 0; component < sh_stride; ++component)
        bad = bad || !isfinite(sh[index * sh_stride + component]);
    const bool outside =
        fabsf(means[3 * index] - center_x) > maximum_bounds ||
        fabsf(means[3 * index + 1] - center_y) > maximum_bounds ||
        fabsf(means[3 * index + 2] - center_z) > maximum_bounds;
    const bool hard = bad || outside || maximum_scale > maximum_bounds;
    hard_prune[index] = hard ? 1U : 0U;
    keep[index] = !hard && opacity >= minimum_opacity ? 1U : 0U;
}

__global__ void chain_gradient_kernel(
    const float* log_scales, const float* raw_quaternions,
    const float* opacity_logits, const float* filter_3d,
    const float* filtered_scales, const float* filtered_opacities,
    const float* grad_scales,
    const float* grad_quaternions, const float* grad_opacities,
    float* grad_log_scales, float* grad_raw_quaternions,
    float* grad_opacity_logits, const std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float filter_squared = filter_3d != nullptr
        ? filter_3d[index] * filter_3d[index]
        : 0.F;
    for (int axis = 0; axis < 3; ++axis) {
        const std::size_t offset = 3 * index + axis;
        const float raw_scale = expf(log_scales[offset]);
        const float filtered_scale = filtered_scales[offset];
        const float filtered_scale_squared =
            filtered_scale * filtered_scale;
        // d sqrt(s^2+f^2) / d log(s), plus the scale/opacity
        // cross-term introduced by density-preserving opacity compensation.
        grad_log_scales[offset] =
            grad_scales[offset] * raw_scale * raw_scale / filtered_scale +
            grad_opacities[index] * filtered_opacities[index] *
                filter_squared / filtered_scale_squared;
    }
    const float rw = raw_quaternions[4 * index + 0];
    const float rx = raw_quaternions[4 * index + 1];
    const float ry = raw_quaternions[4 * index + 2];
    const float rz = raw_quaternions[4 * index + 3];
    const float inverse_norm = rsqrtf(fmaxf(rw * rw + rx * rx + ry * ry + rz * rz, 1e-20F));
    const float q[4] = {rw * inverse_norm, rx * inverse_norm, ry * inverse_norm, rz * inverse_norm};
    const float dot = q[0] * grad_quaternions[4 * index + 0] +
                      q[1] * grad_quaternions[4 * index + 1] +
                      q[2] * grad_quaternions[4 * index + 2] +
                      q[3] * grad_quaternions[4 * index + 3];
    for (int component = 0; component < 4; ++component)
        grad_raw_quaternions[4 * index + component] = inverse_norm *
            (grad_quaternions[4 * index + component] - q[component] * dot);
    const float opacity = sigmoid(opacity_logits[index]);
    grad_opacity_logits[index] =
        grad_opacities[index] * filtered_opacities[index] * (1.F - opacity);
}

__global__ void compute_3d_filter_distance_kernel(
    const float* means, const float* cameras, const std::size_t count,
    const std::size_t camera_count, float* distances,
    unsigned* maximum_distance_bits,
    const bool all_camera_euclidean) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = means[3 * index];
    const float y = means[3 * index + 1];
    const float z = means[3 * index + 2];
    float minimum_distance = FLT_MAX;
    for (std::size_t view = 0; view < camera_count; ++view) {
        // 16 column-major world-to-camera values followed by fx, fy, W, H.
        const float* camera = cameras + 20 * view;
        const float camera_x = camera[0] * x + camera[4] * y +
                               camera[8] * z + camera[12];
        const float camera_y = camera[1] * x + camera[5] * y +
                               camera[9] * z + camera[13];
        const float camera_z = camera[2] * x + camera[6] * y +
                               camera[10] * z + camera[14];
        if (all_camera_euclidean) {
            // Brush's 3D filter is based on Euclidean distance to every
            // training camera. This keeps the floor continuous just outside
            // a view.
            minimum_distance = fminf(
                minimum_distance,
                sqrtf(camera_x * camera_x + camera_y * camera_y +
                      camera_z * camera_z));
        } else {
            if (!(camera_z > 0.2F)) continue;
            const float boundary_x = camera[18] / camera[16] * 0.575F;
            const float boundary_y = camera[19] / camera[17] * 0.575F;
            if (fabsf(camera_x / camera_z) > boundary_x ||
                fabsf(camera_y / camera_z) > boundary_y)
                continue;
            minimum_distance = fminf(minimum_distance, camera_z);
        }
    }
    distances[index] = minimum_distance;
    if (minimum_distance < FLT_MAX)
        atomicMax(maximum_distance_bits, __float_as_uint(minimum_distance));
}

__global__ void finalize_3d_filter_kernel(
    float* distances, const std::size_t count, const float maximum_distance,
    const float inverse_maximum_focal,
    const float minimum_scale_factor_sqrt) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float distance = distances[index] < FLT_MAX
        ? distances[index]
        : maximum_distance;
    distances[index] = distance * inverse_maximum_focal *
                       minimum_scale_factor_sqrt;
}

__device__ float sample_plane_bilinear(
    const float* image, const int width, const int height,
    const float u, const float v) {
    const int x0 = max(0, min(width - 1, static_cast<int>(floorf(u))));
    const int y0 = max(0, min(height - 1, static_cast<int>(floorf(v))));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float tx = u - floorf(u);
    const float ty = v - floorf(v);
    return (image[y0 * width + x0] * (1.F - tx) +
            image[y0 * width + x1] * tx) * (1.F - ty) +
           (image[y1 * width + x0] * (1.F - tx) +
            image[y1 * width + x1] * tx) * ty;
}

struct GrayBilinearGradient {
    float value;
    float du;
    float dv;
};

__device__ __forceinline__ GrayBilinearGradient
sample_gray_bilinear_gradient(
    const float* gray, const int width, const int height,
    const float u, const float v) {
    const float floor_u = floorf(u);
    const float floor_v = floorf(v);
    const int x0 = max(
        0, min(width - 1, static_cast<int>(floor_u)));
    const int y0 = max(
        0, min(height - 1, static_cast<int>(floor_v)));
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float tx = u - floor_u;
    const float ty = v - floor_v;
    const float c00 = gray[y0 * width + x0];
    const float c01 = gray[y0 * width + x1];
    const float c10 = gray[y1 * width + x0];
    const float c11 = gray[y1 * width + x1];
    return {
        (c00 * (1.F - tx) + c01 * tx) * (1.F - ty) +
            (c10 * (1.F - tx) + c11 * tx) * ty,
        (c01 - c00) * (1.F - ty) + (c11 - c10) * ty,
        (c10 - c00) * (1.F - tx) + (c11 - c01) * tx};
}

__device__ bool plane_warp_ncc(
    const float depth, float nx, float ny, float nz,
    const int center_x, const int center_y,
    const float* transform, const Camera reference,
    const Camera neighbour, const float* reference_gray,
    const int reference_gray_width,
    const int reference_sample_x, const int reference_sample_y,
    const float* neighbour_gray, float& ncc, float& grad_depth,
    float& grad_nx, float& grad_ny, float& grad_nz) {
    constexpr int radius = k_ncc_radius;
    constexpr float radius_scaled = 1.5F;
    constexpr int samples = 49;
    constexpr float inverse_samples = 1.F / samples;
    if (center_x - radius_scaled <= 0.F ||
        center_x + radius_scaled >= reference.width - 1 ||
        center_y - radius_scaled <= 0.F ||
        center_y + radius_scaled >= reference.height - 1)
        return false;
    const float normal_length = sqrtf(nx * nx + ny * ny + nz * nz);
    if (!(normal_length > 1e-8F)) return false;
    nx /= normal_length;
    ny /= normal_length;
    nz /= normal_length;
    const float qcx = (center_x - reference.cx) / reference.fx;
    const float qcy = (center_y - reference.cy) / reference.fy;
    const float distance = -(qcx * nx + qcy * ny + nz) * depth;
    if (!(fabsf(distance) > 1e-7F)) return false;
    float homography[9];
    for (int row = 0; row < 3; ++row) {
        const float translation = transform[9 + row];
        homography[3 * row] = transform[3 * row] - translation * nx / distance;
        homography[3 * row + 1] = transform[3 * row + 1] - translation * ny / distance;
        homography[3 * row + 2] = transform[3 * row + 2] - translation * nz / distance;
    }
    float sum_r = 0.F, sum_n = 0.F, sum_r2 = 0.F, sum_n2 = 0.F,
          sum_rn = 0.F;
    float3 derivative_sum = make_float3(0.F, 0.F, 0.F);
    float3 derivative_sum2 = make_float3(0.F, 0.F, 0.F);
    float3 derivative_cross = make_float3(0.F, 0.F, 0.F);
    const float aux_x = transform[9] / distance;
    const float aux_y = transform[10] / distance;
    const float aux_z = transform[11] / distance;
    for (int dv_i = -radius; dv_i <= radius; ++dv_i) {
        const float vr = center_y + 0.5F * dv_i;
        for (int du_i = -radius; du_i <= radius; ++du_i) {
            const float ur = center_x + 0.5F * du_i;
            const float qx = (ur - reference.cx) / reference.fx;
            const float qy = (vr - reference.cy) / reference.fy;
            const float hx = homography[0] * qx + homography[1] * qy + homography[2];
            const float hy = homography[3] * qx + homography[4] * qy + homography[5];
            const float hz = homography[6] * qx + homography[7] * qy + homography[8];
            if (!(hz > 1e-7F)) return false;
            const float un = neighbour.fx * hx / hz + neighbour.cx;
            const float vn = neighbour.fy * hy / hz + neighbour.cy;
            if (!(un - radius_scaled > 0.F &&
                  un + radius_scaled < neighbour.width - 1 &&
                  vn - radius_scaled > 0.F &&
                  vn + radius_scaled < neighbour.height - 1))
                return false;
            const float cr = reference_gray[
                (reference_sample_y + dv_i) * reference_gray_width +
                reference_sample_x + du_i];
            const GrayBilinearGradient neighbour_sample =
                sample_gray_bilinear_gradient(
                    neighbour_gray, static_cast<int>(neighbour.width),
                    static_cast<int>(neighbour.height), un, vn);
            const float cn = neighbour_sample.value;
            const float dc_du = neighbour_sample.du;
            const float dc_dv = neighbour_sample.dv;
            const float dc_dhx = dc_du * neighbour.fx / hz;
            const float dc_dhy = dc_dv * neighbour.fy / hz;
            const float dc_dhz =
                -(dc_du * (un - neighbour.cx) +
                  dc_dv * (vn - neighbour.cy)) / hz;
            const float factor = dc_dhx * aux_x + dc_dhy * aux_y +
                                 dc_dhz * aux_z;
            const float3 derivative = make_float3(qx * factor, qy * factor, factor);
            derivative_sum.x += derivative.x;
            derivative_sum.y += derivative.y;
            derivative_sum.z += derivative.z;
            derivative_sum2.x += 2.F * cn * derivative.x;
            derivative_sum2.y += 2.F * cn * derivative.y;
            derivative_sum2.z += 2.F * cn * derivative.z;
            derivative_cross.x += cr * derivative.x;
            derivative_cross.y += cr * derivative.y;
            derivative_cross.z += cr * derivative.z;
            sum_r += cr;
            sum_n += cn;
            sum_r2 += cr * cr;
            sum_n2 += cn * cn;
            sum_rn += cr * cn;
        }
    }
    const float cross = sum_rn - sum_r * sum_n * inverse_samples;
    const float variance_r = sum_r2 - sum_r * sum_r * inverse_samples;
    const float variance_n = sum_n2 - sum_n * sum_n * inverse_samples;
    if (!(variance_r > 5e-6F && variance_n > 5e-6F)) return false;
    const float denominator = variance_r * variance_n + 1e-8F;
    ncc = cross * cross / denominator;
    const float grad_cross = 2.F * cross / denominator;
    const float grad_variance_n = -ncc / (variance_n + 1e-8F);
    const float coefficient_sum =
        (-grad_cross * sum_r - 2.F * grad_variance_n * sum_n) *
        inverse_samples;
    const float3 derivative = make_float3(
        coefficient_sum * derivative_sum.x +
            grad_variance_n * derivative_sum2.x +
            grad_cross * derivative_cross.x,
        coefficient_sum * derivative_sum.y +
            grad_variance_n * derivative_sum2.y +
            grad_cross * derivative_cross.y,
        coefficient_sum * derivative_sum.z +
            grad_variance_n * derivative_sum2.z +
            grad_cross * derivative_cross.z);
    grad_nx = -derivative.x;
    grad_ny = -derivative.y;
    grad_nz = -derivative.z;
    const float grad_distance =
        (derivative.x * nx + derivative.y * ny + derivative.z * nz) /
        distance;
    grad_nx -= depth * grad_distance * qcx;
    grad_ny -= depth * grad_distance * qcy;
    grad_nz -= depth * grad_distance;
    grad_depth = -(qcx * nx + qcy * ny + nz) * grad_distance;
    // Python normalizes the selected raster normal before invoking the NCC
    // kernel, so its autograd path applies the normalization Jacobian.
    const float normal_dot_gradient =
        nx * grad_nx + ny * grad_ny + nz * grad_nz;
    grad_nx = (grad_nx - nx * normal_dot_gradient) / normal_length;
    grad_ny = (grad_ny - ny * normal_dot_gradient) / normal_length;
    grad_nz = (grad_nz - nz * normal_dot_gradient) / normal_length;
    return isfinite(ncc) && isfinite(grad_depth) && isfinite(grad_nx) &&
           isfinite(grad_ny) && isfinite(grad_nz);
}

__global__ void multi_view_raw_kernel(
    const float* reference_depth, const float* reference_normal,
    const float* reference_gray, const float* sampled_neighbour_points,
    const bool* sampled_inside, const float* neighbour_gray,
    const float* transform,
    const float* reference_mask, const float* neighbour_mask,
    const bool reference_has_mask, const bool neighbour_has_mask,
    const Camera reference, const Camera neighbour,
    const float pixel_noise_threshold, const bool robust_ncc,
    const bool enable_ncc, const bool count_geometry_candidates,
    const float ncc_lambda_reference, const float ncc_sharpness,
    const float ncc_min_weight, float* geo_grad_sampled,
    float* ncc_grad_depth, float* ncc_grad_normal, float* terms,
    const std::size_t pixels) {
    __shared__ float reference_tile[
        k_ncc_reference_tile_width * k_ncc_reference_tile_height];
    __shared__ float reference_samples[
        k_ncc_reference_sample_width *
        k_ncc_reference_sample_height];
    const int block_origin_x =
        static_cast<int>(blockIdx.x * k_multi_view_block_x);
    const int block_origin_y =
        static_cast<int>(blockIdx.y * k_multi_view_block_y);
    const unsigned thread_linear =
        threadIdx.y * k_multi_view_block_x + threadIdx.x;
    constexpr unsigned tile_pixels =
        k_ncc_reference_tile_width * k_ncc_reference_tile_height;
    constexpr unsigned block_threads =
        k_multi_view_block_x * k_multi_view_block_y;
    if (enable_ncc) {
        for (unsigned tile_pixel = thread_linear;
             tile_pixel < tile_pixels;
             tile_pixel += block_threads) {
            const int tile_x =
                static_cast<int>(tile_pixel % k_ncc_reference_tile_width);
            const int tile_y =
                static_cast<int>(tile_pixel / k_ncc_reference_tile_width);
            const int source_x = max(
                0, min(
                    static_cast<int>(reference.width) - 1,
                    block_origin_x + tile_x -
                        k_ncc_reference_padding));
            const int source_y = max(
                0, min(
                    static_cast<int>(reference.height) - 1,
                    block_origin_y + tile_y -
                        k_ncc_reference_padding));
            reference_tile[tile_pixel] =
                reference_gray[
                    static_cast<std::size_t>(source_y) *
                        reference.width +
                    static_cast<std::size_t>(source_x)];
        }
        __syncthreads();
        constexpr unsigned sample_pixels =
            k_ncc_reference_sample_width *
            k_ncc_reference_sample_height;
        for (unsigned sample_pixel = thread_linear;
             sample_pixel < sample_pixels;
             sample_pixel += block_threads) {
            const int sample_x = static_cast<int>(
                sample_pixel % k_ncc_reference_sample_width);
            const int sample_y = static_cast<int>(
                sample_pixel / k_ncc_reference_sample_width);
            const int x0 = (sample_x + 1) / 2;
            const int y0 = (sample_y + 1) / 2;
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            const float tx = (sample_x & 1) == 0 ? 0.5F : 0.F;
            const float ty = (sample_y & 1) == 0 ? 0.5F : 0.F;
            const float c00 =
                reference_tile[
                    y0 * k_ncc_reference_tile_width + x0];
            const float c01 =
                reference_tile[
                    y0 * k_ncc_reference_tile_width + x1];
            const float c10 =
                reference_tile[
                    y1 * k_ncc_reference_tile_width + x0];
            const float c11 =
                reference_tile[
                    y1 * k_ncc_reference_tile_width + x1];
            reference_samples[sample_pixel] =
                (c00 * (1.F - tx) + c01 * tx) * (1.F - ty) +
                (c10 * (1.F - tx) + c11 * tx) * ty;
        }
        __syncthreads();
    }
    const int x = block_origin_x + static_cast<int>(threadIdx.x);
    const int y = block_origin_y + static_cast<int>(threadIdx.y);
    if (x >= static_cast<int>(reference.width) ||
        y >= static_cast<int>(reference.height))
        return;
    const std::size_t pixel =
        static_cast<std::size_t>(y) * reference.width +
        static_cast<std::size_t>(x);
    if (pixel >= pixels) return;
    const float depth = reference_depth[pixel];
    if (!sampled_inside[pixel] || !(depth > 0.F) ||
        (reference_has_mask && reference_mask[pixel] <= 0.5F))
        return;
    if (neighbour_has_mask) {
        const float qx =
            (static_cast<float>(x) - reference.cx) / reference.fx * depth;
        const float qy =
            (static_cast<float>(y) - reference.cy) / reference.fy * depth;
        const float nx = transform[0] * qx + transform[1] * qy +
                         transform[2] * depth + transform[9];
        const float ny = transform[3] * qx + transform[4] * qy +
                         transform[5] * depth + transform[10];
        const float nz = transform[6] * qx + transform[7] * qy +
                         transform[8] * depth + transform[11];
        if (!(nz > 0.2F)) return;
        const int neighbour_x = static_cast<int>(lrintf(
            neighbour.fx * nx / nz + neighbour.cx));
        const int neighbour_y = static_cast<int>(lrintf(
            neighbour.fy * ny / nz + neighbour.cy));
        if (neighbour_x < 0 || neighbour_y < 0 ||
            neighbour_x >= static_cast<int>(neighbour.width) ||
            neighbour_y >= static_cast<int>(neighbour.height) ||
            neighbour_mask[
                static_cast<std::size_t>(neighbour_y) * neighbour.width +
                static_cast<std::size_t>(neighbour_x)] <= 0.5F)
            return;
    }
    const float sx = sampled_neighbour_points[3 * pixel];
    const float sy = sampled_neighbour_points[3 * pixel + 1];
    const float sz = sampled_neighbour_points[3 * pixel + 2];
    if (!(sz > 0.2F)) return;
    const float dx = sx - transform[9];
    const float dy = sy - transform[10];
    const float dz = sz - transform[11];
    const float rx = transform[0] * dx + transform[3] * dy + transform[6] * dz;
    const float ry = transform[1] * dx + transform[4] * dy + transform[7] * dz;
    const float rz = transform[2] * dx + transform[5] * dy + transform[8] * dz;
    if (!(rz > 0.2F)) return;
    if (count_geometry_candidates) {
        // All lanes that remain active here are valid round-trip candidates.
        // Count once per warp instead of serializing hundreds of thousands of
        // per-pixel atomics into the five-scalar loss buffer.
        const unsigned active_lanes = __activemask();
        const unsigned leader = static_cast<unsigned>(__ffs(active_lanes) - 1);
        if (threadIdx.x == leader)
            atomicAdd(
                terms + 4,
                static_cast<float>(__popc(active_lanes)));
    }
    const float projected_x = reference.fx * rx / rz + reference.cx;
    const float projected_y = reference.fy * ry / rz + reference.cy;
    const float du = projected_x - static_cast<float>(x);
    const float dv = projected_y - static_cast<float>(y);
    const float noise = sqrtf(du * du + dv * dv + 1e-12F);
    if (!(noise < pixel_noise_threshold)) return;
    const float weight = expf(-noise);
    const float inverse_noise = 1.F / noise;
    const float grad_rx = du * inverse_noise * reference.fx / rz;
    const float grad_ry = dv * inverse_noise * reference.fy / rz;
    const float grad_rz =
        -(du * reference.fx * rx + dv * reference.fy * ry) *
        inverse_noise / (rz * rz);
    geo_grad_sampled[3 * pixel] = weight *
        (transform[0] * grad_rx + transform[1] * grad_ry +
         transform[2] * grad_rz);
    geo_grad_sampled[3 * pixel + 1] = weight *
        (transform[3] * grad_rx + transform[4] * grad_ry +
         transform[5] * grad_rz);
    geo_grad_sampled[3 * pixel + 2] = weight *
        (transform[6] * grad_rx + transform[7] * grad_ry +
         transform[8] * grad_rz);
    atomicAdd(terms, weight * noise);
    atomicAdd(terms + 1, 1.F);

    if (!enable_ncc) return;
    const float nx = reference_normal[pixel];
    const float ny = reference_normal[pixels + pixel];
    const float nz = reference_normal[2 * pixels + pixel];
    float ncc{}, gd{}, gnx{}, gny{}, gnz{};
    if (!plane_warp_ncc(
            depth, nx, ny, nz, x, y, transform, reference, neighbour,
            reference_samples, k_ncc_reference_sample_width,
            2 * static_cast<int>(threadIdx.x) + k_ncc_radius,
            2 * static_cast<int>(threadIdx.y) + k_ncc_radius,
            neighbour_gray, ncc, gd, gnx, gny, gnz))
        return;
    const float error = fminf(fmaxf(1.F - ncc, 0.F), 2.F);
    if (!robust_ncc && error >= 0.9F) return;
    float confidence = robust_ncc
        ? 1.F / (1.F + expf(-(ncc_lambda_reference - error) *
                             ncc_sharpness))
        : 1.F;
    confidence = confidence * (1.F - ncc_min_weight) + ncc_min_weight;
    const float factor = -weight * confidence;
    ncc_grad_depth[pixel] = factor * gd;
    ncc_grad_normal[pixel] = factor * gnx;
    ncc_grad_normal[pixels + pixel] = factor * gny;
    ncc_grad_normal[2 * pixels + pixel] = factor * gnz;
    atomicAdd(terms + 2, weight * confidence * error);
    atomicAdd(terms + 3, 1.F);
}

__global__ void accumulate_multi_view_stability_kernel(
    const float* terms, float* accumulator) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    accumulator[0] += terms[1];
    accumulator[1] += terms[4];
    accumulator[2] += 1.F;
}

__global__ void add_multi_view_gradients_kernel(
    float* depth, float* normal, float* geo_sampled,
    const float* ncc_depth, const float* ncc_normal, const float* terms,
    const float geometry_weight, const float ncc_weight,
    const std::size_t pixels) {
    const std::size_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    const float geo_scale = terms[1] > 0.F ? geometry_weight / terms[1] : 0.F;
    const float ncc_scale = terms[3] > 0.F ? ncc_weight / terms[3] : 0.F;
    depth[pixel] += ncc_scale * ncc_depth[pixel];
    for (int axis = 0; axis < 3; ++axis)
        geo_sampled[3 * pixel + axis] *= geo_scale;
    for (int axis = 0; axis < 3; ++axis)
        normal[static_cast<std::size_t>(axis) * pixels + pixel] +=
            ncc_scale * ncc_normal[static_cast<std::size_t>(axis) * pixels + pixel];
}

__global__ void unproject_depth_to_world_kernel(
    const float* depth, float* points, const Camera camera,
    const std::size_t pixels) {
    const std::size_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    const float z = depth[pixel];
    const float x = (static_cast<float>(pixel % camera.width) - camera.cx) /
                    camera.fx * z;
    const float y = (static_cast<float>(pixel / camera.width) - camera.cy) /
                    camera.fy * z;
    const float dx = x - camera.world_to_camera[12];
    const float dy = y - camera.world_to_camera[13];
    const float dz = z - camera.world_to_camera[14];
    points[3 * pixel] = camera.world_to_camera[0] * dx +
                        camera.world_to_camera[1] * dy +
                        camera.world_to_camera[2] * dz;
    points[3 * pixel + 1] = camera.world_to_camera[4] * dx +
                            camera.world_to_camera[5] * dy +
                            camera.world_to_camera[6] * dz;
    points[3 * pixel + 2] = camera.world_to_camera[8] * dx +
                            camera.world_to_camera[9] * dy +
                            camera.world_to_camera[10] * dz;
}

__global__ void add_point_depth_gradients_kernel(
    const float* point_gradients, float* depth_gradients,
    const Camera camera, const std::size_t pixels) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < pixels) {
        const float qx = (static_cast<float>(index % camera.width) - camera.cx) /
                         camera.fx;
        const float qy = (static_cast<float>(index / camera.width) - camera.cy) /
                         camera.fy;
        const float wx = camera.world_to_camera[0] * qx +
                         camera.world_to_camera[1] * qy +
                         camera.world_to_camera[2];
        const float wy = camera.world_to_camera[4] * qx +
                         camera.world_to_camera[5] * qy +
                         camera.world_to_camera[6];
        const float wz = camera.world_to_camera[8] * qx +
                         camera.world_to_camera[9] * qy +
                         camera.world_to_camera[10];
        depth_gradients[index] += point_gradients[3 * index] * wx +
                                  point_gradients[3 * index + 1] * wy +
                                  point_gradients[3 * index + 2] * wz;
    }
}

__global__ void add_tensor_in_place_kernel(
    float* target, const float* added, const std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) target[index] += added[index];
}

__global__ void loss_kernel(
    const float* color, const float* alpha, const float* depth,
    const float* normal, const float* target_color,
    const float* target_depth, const float* target_normal,
    const float* mask, float* grad_color, float* grad_alpha,
    float* grad_depth, float* grad_normal, float* terms,
    const std::size_t pixels, const float photo_weight,
    const float depth_weight, const float normal_weight,
    const bool mask_enabled, const int alpha_mode,
    const float match_alpha_weight, const float l1_weight,
    const float geometry_epsilon) {
    const std::size_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    const float valid = mask_enabled ? mask[pixel] : 1.F;
    const float inverse_pixels = 1.F / static_cast<float>(pixels);
    float rgb_loss = 0.F;
    for (int channel = 0; channel < 3; ++channel) {
        const std::size_t offset = static_cast<std::size_t>(channel) * pixels + pixel;
        const float difference = color[offset] - target_color[offset];
        rgb_loss += fabsf(difference);
        grad_color[offset] = photo_weight * l1_weight * valid * inverse_pixels /
                             3.F * ((difference > 0.F) - (difference < 0.F));
    }
    if (terms)
        atomicAdd(
            terms + 0,
            photo_weight * l1_weight * valid * rgb_loss * inverse_pixels / 3.F);

    const bool has_depth = depth_weight > 0.F &&
        target_depth[pixel] > 0.F && depth[pixel] > 0.F;
    if (has_depth && valid > 0.F) {
        const float scale = fmaxf(target_depth[pixel], 1e-4F);
        const float difference = (depth[pixel] - target_depth[pixel]) / scale;
        const float robust = sqrtf(
            difference * difference + geometry_epsilon * geometry_epsilon);
        grad_depth[pixel] = depth_weight * inverse_pixels * difference /
                            (robust * scale);
        if (terms)
            atomicAdd(terms + 1, depth_weight * robust * inverse_pixels);
    }

    float tx = 0.F;
    float ty = 0.F;
    float tz = 0.F;
    float target_length2 = 0.F;
    if (normal_weight > 0.F) {
        tx = target_normal[pixel];
        ty = target_normal[pixels + pixel];
        tz = target_normal[2 * pixels + pixel];
        target_length2 = tx * tx + ty * ty + tz * tz;
    }
    if (target_length2 > 0.25F && valid > 0.F) {
        const float inverse_target_length = rsqrtf(target_length2);
        const float nx = normal[pixel];
        const float ny = normal[pixels + pixel];
        const float nz = normal[2 * pixels + pixel];
        const float dot = nx * tx * inverse_target_length +
                          ny * ty * inverse_target_length +
                          nz * tz * inverse_target_length;
        if (terms)
            atomicAdd(
                terms + 2,
                normal_weight * (1.F - dot) * inverse_pixels);
        grad_normal[pixel] = -normal_weight * tx * inverse_target_length * inverse_pixels;
        grad_normal[pixels + pixel] = -normal_weight * ty * inverse_target_length * inverse_pixels;
        grad_normal[2 * pixels + pixel] = -normal_weight * tz * inverse_target_length * inverse_pixels;
    }

    if (mask_enabled && alpha_mode == 0) {
        // pygsplat alpha_mode="masked": discourage any opacity outside the
        // foreground without forcing the foreground itself to be opaque.
        grad_alpha[pixel] = (1.F - valid) * inverse_pixels;
        if (terms)
            atomicAdd(
                terms + 3, alpha[pixel] * (1.F - valid) * inverse_pixels);
    } else if (mask_enabled && alpha_mode == 1 && match_alpha_weight > 0.F) {
        // pygsplat alpha_mode="transparent": full-image BCE(alpha, mask).
        constexpr float clamp_epsilon = 1e-7F;
        const float raw_prediction = alpha[pixel];
        const float prediction = fminf(
            fmaxf(raw_prediction, clamp_epsilon), 1.F - clamp_epsilon);
        // torch.clamp, used by pygsplat before BCE, has zero derivative outside
        // its interval. Continuing to differentiate the clamped value produces
        // enormous gradients at saturated pixels and destabilizes opacity,
        // refine-weight accumulation, and pruning.
        const bool inside_clamp =
            raw_prediction > clamp_epsilon &&
            raw_prediction < 1.F - clamp_epsilon;
        grad_alpha[pixel] = inside_clamp
            ? match_alpha_weight * inverse_pixels *
                  (prediction - valid) /
                  (prediction * (1.F - prediction))
            : 0.F;
        if (terms)
            atomicAdd(terms + 3, -match_alpha_weight * inverse_pixels *
                (valid * logf(prediction) +
                 (1.F - valid) * logf(1.F - prediction)));
    }
}

__device__ float3 subtract3(const float3 a, const float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 cross3(const float3 a, const float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ float dot3(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 scale3(const float3 value, const float scale) {
    return make_float3(value.x * scale, value.y * scale, value.z * scale);
}

__global__ void depth_normal_consistency_kernel(
    const float* depth, const float* normal, float* grad_depth,
    float* grad_normal, float* terms, const std::uint32_t width,
    const std::uint32_t height, const float fx, const float fy,
    const float cx, const float cy, const float weight) {
    const std::size_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    if (pixel >= pixels) return;
    const std::uint32_t x = static_cast<std::uint32_t>(pixel % width);
    const std::uint32_t y = static_cast<std::uint32_t>(pixel / width);
    if (x == 0 || y == 0 || x + 1 >= width || y + 1 >= height)
        return;

    const std::size_t top = pixel - width;
    const std::size_t bottom = pixel + width;
    const std::size_t left = pixel - 1;
    const std::size_t right = pixel + 1;
    if (!(depth[pixel] > 0.F && depth[top] > 0.F && depth[bottom] > 0.F &&
          depth[left] > 0.F && depth[right] > 0.F))
        return;

    const auto point = [&](const std::uint32_t px, const std::uint32_t py,
                           const float d) {
        return make_float3(
            (static_cast<float>(px) - cx) / fx * d,
            (static_cast<float>(py) - cy) / fy * d, d);
    };
    const float3 point_top = point(x, y - 1, depth[top]);
    const float3 point_bottom = point(x, y + 1, depth[bottom]);
    const float3 point_left = point(x - 1, y, depth[left]);
    const float3 point_right = point(x + 1, y, depth[right]);
    const float3 dy = subtract3(point_bottom, point_top);
    const float3 dx = subtract3(point_right, point_left);
    const float3 cross = cross3(dy, dx);
    const float length_squared = dot3(cross, cross);
    if (!(length_squared > 1e-20F) || !isfinite(length_squared)) return;
    const float inverse_length = rsqrtf(length_squared);
    const float3 depth_normal = scale3(cross, inverse_length);
    const float3 raster_normal = make_float3(
        normal[pixel], normal[pixels + pixel],
        normal[2 * pixels + pixel]);
    const float normalization = weight / static_cast<float>(pixels);
    const float loss = normalization *
        (1.F - dot3(raster_normal, depth_normal));
    if (terms) atomicAdd(terms + 2, loss);

    atomicAdd(grad_normal + pixel, -normalization * depth_normal.x);
    atomicAdd(
        grad_normal + pixels + pixel,
        -normalization * depth_normal.y);
    atomicAdd(
        grad_normal + 2 * pixels + pixel,
        -normalization * depth_normal.z);

    // Backpropagate through normalize(cross(dy, dx)), matching PyTorch's
    // autograd path in gggs_depth_to_normal.
    const float3 grad_unit = scale3(raster_normal, -normalization);
    const float projection = dot3(depth_normal, grad_unit);
    const float3 grad_cross = scale3(
        subtract3(grad_unit, scale3(depth_normal, projection)),
        inverse_length);
    const float3 grad_dy = cross3(dx, grad_cross);
    const float3 grad_dx = cross3(grad_cross, dy);
    const auto ray = [&](const std::uint32_t px, const std::uint32_t py) {
        return make_float3(
            (static_cast<float>(px) - cx) / fx,
            (static_cast<float>(py) - cy) / fy, 1.F);
    };
    atomicAdd(grad_depth + bottom, dot3(grad_dy, ray(x, y + 1)));
    atomicAdd(grad_depth + top, -dot3(grad_dy, ray(x, y - 1)));
    atomicAdd(grad_depth + right, dot3(grad_dx, ray(x + 1, y)));
    atomicAdd(grad_depth + left, -dot3(grad_dx, ray(x - 1, y)));
}

__global__ void normal_field_consistency_kernel(
    const float* depth, const float* oriented_normal, float* grad_depth,
    float* grad_oriented_normal, float* terms, const Camera camera,
    const float weight) {
    const std::size_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t pixels =
        static_cast<std::size_t>(camera.width) * camera.height;
    if (pixel >= pixels) return;
    const std::uint32_t x = static_cast<std::uint32_t>(pixel % camera.width);
    const std::uint32_t y = static_cast<std::uint32_t>(pixel / camera.width);
    if (x == 0 || y == 0 || x + 1 >= camera.width ||
        y + 1 >= camera.height)
        return;

    const std::size_t top = pixel - camera.width;
    const std::size_t bottom = pixel + camera.width;
    const std::size_t left = pixel - 1;
    const std::size_t right = pixel + 1;
    if (!(depth[pixel] > 0.F && depth[top] > 0.F && depth[bottom] > 0.F &&
          depth[left] > 0.F && depth[right] > 0.F))
        return;
    const auto point = [&](const std::uint32_t px, const std::uint32_t py,
                           const float d) {
        return make_float3(
            (static_cast<float>(px) - camera.cx) / camera.fx * d,
            (static_cast<float>(py) - camera.cy) / camera.fy * d, d);
    };
    const float3 dy = subtract3(
        point(x, y + 1, depth[bottom]), point(x, y - 1, depth[top]));
    const float3 dx = subtract3(
        point(x + 1, y, depth[right]), point(x - 1, y, depth[left]));
    const float3 cross = cross3(dy, dx);
    const float length_squared = dot3(cross, cross);
    if (!(length_squared > 1e-20F) || !isfinite(length_squared)) return;
    const float inverse_length = rsqrtf(length_squared);
    const float3 depth_normal = scale3(cross, inverse_length);

    const float3 normal_world = make_float3(
        oriented_normal[pixel], oriented_normal[pixels + pixel],
        oriented_normal[2 * pixels + pixel]);
    // Camera.world_to_camera is column-major. Apply the conventional R to
    // compare the learned world-space field against the camera-space depth
    // normal, exactly matching GaussianWrapping's world conversion.
    const float3 normal_camera = make_float3(
        camera.world_to_camera[0] * normal_world.x +
            camera.world_to_camera[4] * normal_world.y +
            camera.world_to_camera[8] * normal_world.z,
        camera.world_to_camera[1] * normal_world.x +
            camera.world_to_camera[5] * normal_world.y +
            camera.world_to_camera[9] * normal_world.z,
        camera.world_to_camera[2] * normal_world.x +
            camera.world_to_camera[6] * normal_world.y +
            camera.world_to_camera[10] * normal_world.z);
    const float normalization = weight / static_cast<float>(pixels);
    if (terms)
        atomicAdd(
            terms + 2,
            normalization * (1.F - dot3(normal_camera, depth_normal)));

    const float3 grad_camera = scale3(depth_normal, -normalization);
    const float3 grad_world = make_float3(
        camera.world_to_camera[0] * grad_camera.x +
            camera.world_to_camera[1] * grad_camera.y +
            camera.world_to_camera[2] * grad_camera.z,
        camera.world_to_camera[4] * grad_camera.x +
            camera.world_to_camera[5] * grad_camera.y +
            camera.world_to_camera[6] * grad_camera.z,
        camera.world_to_camera[8] * grad_camera.x +
            camera.world_to_camera[9] * grad_camera.y +
            camera.world_to_camera[10] * grad_camera.z);
    atomicAdd(grad_oriented_normal + pixel, grad_world.x);
    atomicAdd(grad_oriented_normal + pixels + pixel, grad_world.y);
    atomicAdd(grad_oriented_normal + 2 * pixels + pixel, grad_world.z);

    const float3 grad_unit = scale3(normal_camera, -normalization);
    const float projection = dot3(depth_normal, grad_unit);
    const float3 grad_cross = scale3(
        subtract3(grad_unit, scale3(depth_normal, projection)),
        inverse_length);
    const float3 grad_dy = cross3(dx, grad_cross);
    const float3 grad_dx = cross3(grad_cross, dy);
    const auto ray = [&](const std::uint32_t px, const std::uint32_t py) {
        return make_float3(
            (static_cast<float>(px) - camera.cx) / camera.fx,
            (static_cast<float>(py) - camera.cy) / camera.fy, 1.F);
    };
    atomicAdd(
        grad_depth + bottom, dot3(grad_dy, ray(x, y + 1)));
    atomicAdd(grad_depth + top, -dot3(grad_dy, ray(x, y - 1)));
    atomicAdd(grad_depth + right, dot3(grad_dx, ray(x + 1, y)));
    atomicAdd(grad_depth + left, -dot3(grad_dx, ray(x - 1, y)));
}

__global__ void normal_features_forward_kernel(
    const float* features, float* normals, const std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = features[4 * index];
    const float y = features[4 * index + 1];
    const float z = features[4 * index + 2];
    const float length = sqrtf(x * x + y * y + z * z);
    if (!(length > 1e-12F) || !isfinite(length)) return;
    const float sign = tanhf(features[4 * index + 3]);
    normals[3 * index] = sign * x / length;
    normals[3 * index + 1] = sign * y / length;
    normals[3 * index + 2] = sign * z / length;
}

__global__ void normal_features_backward_kernel(
    const float* features, const float* grad_normals, float* grad_features,
    const std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = features[4 * index];
    const float y = features[4 * index + 1];
    const float z = features[4 * index + 2];
    const float length = sqrtf(x * x + y * y + z * z);
    if (!(length > 1e-12F) || !isfinite(length)) return;
    const float inverse_length = 1.F / length;
    const float nx = x * inverse_length;
    const float ny = y * inverse_length;
    const float nz = z * inverse_length;
    const float gx = grad_normals[3 * index];
    const float gy = grad_normals[3 * index + 1];
    const float gz = grad_normals[3 * index + 2];
    const float tangent_projection = gx * nx + gy * ny + gz * nz;
    const float sign = tanhf(features[4 * index + 3]);
    const float direction_scale = sign * inverse_length;
    grad_features[4 * index] =
        direction_scale * (gx - nx * tangent_projection);
    grad_features[4 * index + 1] =
        direction_scale * (gy - ny * tangent_projection);
    grad_features[4 * index + 2] =
        direction_scale * (gz - nz * tangent_projection);
    grad_features[4 * index + 3] =
        (1.F - sign * sign) * tangent_projection;
}

__global__ void adam_kernel(
    float* parameter, const float* gradient, float* first, float* second,
    const std::size_t count, const float learning_rate,
    const float secondary_learning_rate, const std::size_t group_stride,
    const float beta1, const float beta2, const float correction1,
    const float correction2, const float epsilon,
    const float clamp_min, const float clamp_max) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float previous = parameter[index];
    const float grad = gradient[index];
    // A single degenerate projected Gaussian can occasionally produce a
    // non-finite gradient in the imported rasterizer.  Do not allow it to
    // poison the parameter and both Adam moments permanently.
    if (!isfinite(previous) || !isfinite(grad)) {
        first[index] = 0.F;
        second[index] = 0.F;
        parameter[index] = isfinite(previous)
            ? fminf(fmaxf(previous, clamp_min), clamp_max)
            : fminf(fmaxf(0.F, clamp_min), clamp_max);
        return;
    }
    const float m = beta1 * first[index] + (1.F - beta1) * grad;
    const float v = beta2 * second[index] + (1.F - beta2) * grad * grad;
    if (!isfinite(m) || !isfinite(v)) {
        first[index] = 0.F;
        second[index] = 0.F;
        parameter[index] = fminf(fmaxf(previous, clamp_min), clamp_max);
        return;
    }
    first[index] = m;
    second[index] = v;
    const float lr = group_stride != 0 && index % group_stride >= 3
                         ? secondary_learning_rate
                         : learning_rate;
    const float candidate = previous - lr * (m / correction1) /
                          (sqrtf(v / correction2) + epsilon);
    const float updated = isfinite(candidate) ? candidate : previous;
    parameter[index] = fminf(fmaxf(updated, clamp_min), clamp_max);
}

__global__ void adam_reduced_second_kernel(
    float* parameter, const float* gradient, float* first, float* second,
    const std::size_t row_count, const std::size_t full_row_stride,
    const std::size_t active_row_stride,
    const float learning_rate, const float secondary_learning_rate,
    const float beta1, const float beta2, const float correction1,
    const float correction2, const float epsilon) {
    const std::size_t row = blockIdx.x;
    if (row >= row_count) return;
    const unsigned lane = threadIdx.x;
    const std::size_t row_begin = row * full_row_stride;

    __shared__ float warp_square_sums[k_reduced_adam_threads / 32];
    __shared__ float denominator;
    __shared__ bool row_valid;

    float local_square_sum = 0.F;
    for (std::size_t column = lane; column < active_row_stride;
         column += blockDim.x) {
        const float value = gradient[row_begin + column];
        if (isfinite(value)) local_square_sum += value * value;
    }
    for (unsigned offset = 16; offset > 0; offset >>= 1)
        local_square_sum += __shfl_down_sync(
            0xffffffffU, local_square_sum, offset);
    if ((lane & 31U) == 0)
        warp_square_sums[lane >> 5U] = local_square_sum;
    __syncthreads();

    if (lane == 0) {
        const float square_sum =
            warp_square_sums[0] + warp_square_sums[1];
        const float grad_square_mean =
            square_sum / static_cast<float>(active_row_stride);
        const float v =
            beta2 * second[row] + (1.F - beta2) * grad_square_mean;
        row_valid = isfinite(v);
        if (row_valid) {
            second[row] = v;
            denominator = sqrtf(v / correction2) + epsilon;
            row_valid = isfinite(denominator) && denominator > 0.F;
        }
        if (!row_valid) {
            second[row] = 0.F;
            denominator = 1.F;
        }
    }
    __syncthreads();

    for (std::size_t column = lane; column < active_row_stride;
         column += blockDim.x) {
        const std::size_t index = row_begin + column;
        const float previous = parameter[index];
        const float grad = gradient[index];
        if (!row_valid) {
            first[index] = 0.F;
            continue;
        }
        if (!isfinite(previous) || !isfinite(grad)) {
            first[index] = 0.F;
            parameter[index] = isfinite(previous) ? previous : 0.F;
            continue;
        }
        const float m = beta1 * first[index] + (1.F - beta1) * grad;
        if (!isfinite(m)) {
            first[index] = 0.F;
            continue;
        }
        first[index] = m;
        const float lr = column >= 3
            ? secondary_learning_rate : learning_rate;
        const float candidate =
            previous - lr * (m / correction1) / denominator;
        parameter[index] = isfinite(candidate) ? candidate : previous;
    }
}

__global__ void adam_active_prefix_kernel(
    float* parameter, const float* gradient, float* first, float* second,
    const std::size_t active_count, const std::size_t full_row_stride,
    const std::size_t active_row_stride, const float learning_rate,
    const float secondary_learning_rate, const float beta1, const float beta2,
    const float correction1, const float correction2, const float epsilon) {
    const std::size_t active_index =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (active_index >= active_count) return;
    const std::size_t column = active_index % active_row_stride;
    const std::size_t index =
        active_index / active_row_stride * full_row_stride + column;
    const float previous = parameter[index];
    const float grad = gradient[index];
    if (!isfinite(previous) || !isfinite(grad)) {
        first[index] = 0.F;
        second[index] = 0.F;
        parameter[index] = isfinite(previous) ? previous : 0.F;
        return;
    }
    const float m = beta1 * first[index] + (1.F - beta1) * grad;
    const float v = beta2 * second[index] + (1.F - beta2) * grad * grad;
    if (!isfinite(m) || !isfinite(v)) {
        first[index] = 0.F;
        second[index] = 0.F;
        return;
    }
    first[index] = m;
    second[index] = v;
    const float lr =
        column >= 3 ? secondary_learning_rate : learning_rate;
    const float candidate = previous - lr * (m / correction1) /
        (sqrtf(v / correction2) + epsilon);
    parameter[index] = isfinite(candidate) ? candidate : previous;
}

__global__ void constrain_scale_ratio_kernel(
    float* log_scales, const std::size_t count, const float maximum_log_ratio) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float* values = log_scales + 3 * index;
    const float minimum = fminf(values[0], fminf(values[1], values[2]));
    const float maximum = fmaxf(values[0], fmaxf(values[1], values[2]));
    if (maximum - minimum <= maximum_log_ratio) return;
    const float midpoint = 0.5F * (minimum + maximum);
    const float half_range = 0.5F * maximum_log_ratio;
    for (int axis = 0; axis < 3; ++axis)
        values[axis] = fminf(fmaxf(values[axis], midpoint - half_range),
                             midpoint + half_range);
}

__global__ void accumulate_densification_kernel(
    const float* refine_weight, const float* visibility, const int* radii,
    float* gradient, float* count, float* max_screen_radius, float* priority,
    const std::size_t gaussian_count, const float inverse_resolution,
    const bool use_maximum, const bool require_contribution_visibility) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= gaussian_count || radii[index] <= 0 ||
        (require_contribution_visibility && visibility[index] <= 0.F))
        return;
    const float weight = isfinite(refine_weight[index])
        ? fmaxf(refine_weight[index], 0.F)
        : 0.F;
    if (use_maximum)
        gradient[index] = fmaxf(gradient[index], weight);
    else
        gradient[index] += weight;
    count[index] += 1.F;
    const float screen = radii[index] * inverse_resolution;
    max_screen_radius[index] = fmaxf(max_screen_radius[index], screen);
    priority[index] += weight * (1.F + screen);
}

__device__ void rotate_quaternion(
    const float* raw, const float x, const float y, const float z,
    float& out_x, float& out_y, float& out_z) {
    const float inverse_norm = rsqrtf(fmaxf(
        raw[0] * raw[0] + raw[1] * raw[1] +
        raw[2] * raw[2] + raw[3] * raw[3], 1e-20F));
    const float w = raw[0] * inverse_norm;
    const float qx = raw[1] * inverse_norm;
    const float qy = raw[2] * inverse_norm;
    const float qz = raw[3] * inverse_norm;
    const float tx = 2.F * (qy * z - qz * y);
    const float ty = 2.F * (qz * x - qx * z);
    const float tz = 2.F * (qx * y - qy * x);
    out_x = x + w * tx + (qy * tz - qz * ty);
    out_y = y + w * ty + (qz * tx - qx * tz);
    out_z = z + w * tz + (qx * ty - qy * tx);
}

__global__ void split_gaussians_kernel(
    float* parent_means, float* parent_log_scales,
    float* parent_opacity_logits, const float* parent_quaternions,
    float* child_means, float* child_log_scales,
    float* child_opacity_logits, const int* parent_indices,
    const float* random_samples, const float* screen_sizes,
    const std::size_t split_count, const int mode,
    const float minimum_opacity, const float split_at_screen_size) {
    const std::size_t child = blockIdx.x * blockDim.x + threadIdx.x;
    if (child >= split_count) return;
    const std::size_t parent = static_cast<std::size_t>(parent_indices[child]);
    const float* parent_quaternion = parent_quaternions + 4 * parent;
    float local[3]{};
    float log_scale_delta[3]{};
    if (mode == 4) {
        // Dense MVS already constrains the surface normal accurately. Split
        // only in the local tangent plane (local Z is initialized from the
        // fused-cloud normal) and preserve the normal-axis thickness.
        constexpr float tangent_offset = 0.5F;
        constexpr float tangent_scale = 0.7071067811865475F;
        local[0] = expf(parent_log_scales[3 * parent]) *
                   random_samples[3 * child] * tangent_offset;
        local[1] = expf(parent_log_scales[3 * parent + 1]) *
                   random_samples[3 * child + 1] * tangent_offset;
        local[2] = 0.F;
        log_scale_delta[0] = logf(tangent_scale);
        log_scale_delta[1] = logf(tangent_scale);
        log_scale_delta[2] = 0.F;
    } else if (mode == 3) {
        int largest = 0;
        if (parent_log_scales[3 * parent + 1] >
            parent_log_scales[3 * parent + largest]) largest = 1;
        if (parent_log_scales[3 * parent + 2] >
            parent_log_scales[3 * parent + largest]) largest = 2;
        for (int axis = 0; axis < 3; ++axis) {
            local[axis] = expf(parent_log_scales[3 * parent + axis]) *
                          random_samples[3 * child];
            log_scale_delta[axis] = axis == largest ? logf(0.5F) : 0.F;
        }
    } else if (mode == 2) {
        // Match brush-train's ADC+ covariance-aware split. The offset is
        // deterministic and anti-correlated, preserving the centroid. Axes
        // shrink in proportion to their covariance contribution; oversized
        // splats shrink harder so their largest on-screen extent reaches the
        // configured cap.
        float maximum_scale_squared = 0.F;
        float scale_squared[3]{};
        for (int axis = 0; axis < 3; ++axis) {
            const float scale = expf(parent_log_scales[3 * parent + axis]);
            scale_squared[axis] = scale * scale;
            maximum_scale_squared =
                fmaxf(maximum_scale_squared, scale_squared[axis]);
        }
        const float standard_k = rsqrtf(2.F);
        const float screen = screen_sizes != nullptr
            ? fmaxf(screen_sizes[child], 1e-6F)
            : 1e-6F;
        const float maximum_k = split_at_screen_size > 0.F
            ? fminf(standard_k, split_at_screen_size / screen)
            : standard_k;
        for (int axis = 0; axis < 3; ++axis) {
            const float ratio = scale_squared[axis] /
                fmaxf(maximum_scale_squared, 1e-30F);
            const float k = 1.F - ratio * (1.F - maximum_k);
            const float scale = sqrtf(scale_squared[axis]);
            local[axis] = sqrtf(fmaxf(1.F - k * k, 0.F)) * scale;
            log_scale_delta[axis] = logf(fmaxf(k, 1e-12F));
        }
    } else {
        const float scale_factor = 1.F / 1.6F;
        for (int axis = 0; axis < 3; ++axis) {
            local[axis] = expf(parent_log_scales[3 * parent + axis]) *
                          random_samples[3 * child + axis];
            log_scale_delta[axis] = logf(scale_factor);
        }
    }
    float offset_x{}, offset_y{}, offset_z{};
    rotate_quaternion(
        parent_quaternion, local[0], local[1], local[2],
        offset_x, offset_y, offset_z);
    const float offsets[3]{offset_x, offset_y, offset_z};
    for (int axis = 0; axis < 3; ++axis) {
        const float center = parent_means[3 * parent + axis];
        parent_means[3 * parent + axis] = center - offsets[axis];
        child_means[3 * child + axis] = center + offsets[axis];
        parent_log_scales[3 * parent + axis] += log_scale_delta[axis];
        child_log_scales[3 * child + axis] += log_scale_delta[axis];
    }
    const float opacity = sigmoid(parent_opacity_logits[parent]);
    const float opacity_floor = mode == 1 ? 1e-8F : minimum_opacity;
    const float opacity_power = mode == 2 ? rsqrtf(2.F) : 0.5F;
    const float revised = fminf(fmaxf(
        1.F - powf(fmaxf(1.F - opacity, 0.F), opacity_power),
        opacity_floor), 1.F - opacity_floor);
    const float revised_logit = logf(revised / (1.F - revised));
    parent_opacity_logits[parent] = revised_logit;
    child_opacity_logits[child] = revised_logit;
}

__global__ void adc_decay_kernel(
    float* log_scales, float* opacity_logits, const std::size_t count,
    const float opacity_decay, const float log_scale_decay) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float opacity = fminf(fmaxf(
        sigmoid(opacity_logits[index]) - opacity_decay, 1e-12F),
        1.F - 1e-12F);
    opacity_logits[index] = logf(opacity / (1.F - opacity));
    for (int axis = 0; axis < 3; ++axis)
        log_scales[3 * index + axis] += log_scale_decay;
}

__device__ std::uint32_t hash_u32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

__device__ float normal_sample(
    const std::uint32_t index, const std::uint32_t seed,
    const std::uint32_t axis) {
    const float u1 = (hash_u32(index * 3U + axis + seed * 17U) + 1.F) /
                     4294967297.F;
    const float u2 = (hash_u32(index * 7U + axis + seed * 29U) + 1.F) /
                     4294967297.F;
    return sqrtf(-2.F * logf(fmaxf(u1, 1e-12F))) *
           cosf(6.283185307179586F * u2);
}

__global__ void inject_adc_noise_kernel(
    float* means, const float* opacity_logits, const float* visibility,
    const std::size_t count, const float standard_deviation,
    const float maximum_noise, const unsigned seed) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || visibility[index] <= 0.F) return;
    const float inverse_opacity = 1.F - sigmoid(opacity_logits[index]);
    const float weight = powf(inverse_opacity, 150.F);
    for (std::uint32_t axis = 0; axis < 3; ++axis) {
        const float noise = fminf(fmaxf(
            normal_sample(static_cast<std::uint32_t>(index), seed, axis) *
                weight * standard_deviation,
            -maximum_noise), maximum_noise);
        means[3 * index + axis] += noise;
    }
}

__global__ void reset_opacity_kernel(
    float* opacity_logits, const std::size_t count,
    const float maximum_logit) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        opacity_logits[index] = fminf(opacity_logits[index], maximum_logit);
}

}  // namespace

DecodedTrainingPixels upload_packed_training_pixels(
    const std::vector<int>& rgba, const std::uint32_t width,
    const std::uint32_t height, const bool decode_mask,
    const bool decode_gray) {
    const std::size_t pixels =
        static_cast<std::size_t>(width) * height;
    if (rgba.size() != pixels)
        throw std::invalid_argument(
            "GGGS packed training image size does not match camera");
    const auto packed = tinytensor::Tensor::from_vector(
        rgba, {height, width}, tinytensor::Device::CUDA);
    DecodedTrainingPixels result{
        tinytensor::Tensor::empty(
            {std::size_t{3}, height, width},
            tinytensor::Device::CUDA),
        decode_gray
            ? tinytensor::Tensor::empty(
                  {height, width}, tinytensor::Device::CUDA)
            : tinytensor::Tensor::zeros(
                  {std::size_t{1}}, tinytensor::Device::CUDA),
        decode_mask
            ? tinytensor::Tensor::empty(
                  {height, width}, tinytensor::Device::CUDA)
            : tinytensor::Tensor::zeros(
                  {std::size_t{1}}, tinytensor::Device::CUDA)};
    if (pixels == 0) return result;
    unpack_training_pixels_kernel<<<
        (pixels + k_threads - 1) / k_threads, k_threads>>>(
        reinterpret_cast<const unsigned*>(packed.ptr<int>()),
        result.rgb.ptr<float>(),
        decode_gray ? result.gray.ptr<float>() : nullptr,
        decode_mask ? result.mask.ptr<float>() : nullptr,
        pixels);
    check_cuda(cudaGetLastError(), "unpack GGGS training pixels");
    return result;
}

ActivatedParameters activate_parameters(const GaussianModel& model) {
    const std::size_t count = model.size();
    ActivatedParameters result{
        tinytensor::Tensor::empty({count, 3}, tinytensor::Device::CUDA),
        tinytensor::Tensor::empty({count, 4}, tinytensor::Device::CUDA),
        tinytensor::Tensor::empty({count, 1}, tinytensor::Device::CUDA)};
    if (count == 0) return result;
    activate_kernel<<<(count + k_threads - 1) / k_threads, k_threads>>>(
        model.log_scales.ptr<float>(), model.quaternions.ptr<float>(),
        model.opacity_logits.ptr<float>(),
        model.filter_3d.is_valid() ? model.filter_3d.ptr<float>() : nullptr,
        result.scales.ptr<float>(),
        result.quaternions.ptr<float>(), result.opacities.ptr<float>(), count);
    check_cuda(cudaGetLastError(), "activate GGGS parameters");
    return result;
}

tinytensor::Tensor normal_features_to_normals(
    const tinytensor::Tensor& normal_features) {
    if (!normal_features.is_valid() ||
        normal_features.device() != tinytensor::Device::CUDA ||
        normal_features.dtype() != tinytensor::DataType::Float32 ||
        !normal_features.is_contiguous() ||
        normal_features.shape().rank() != 2 ||
        normal_features.shape()[1] != 4)
        throw std::invalid_argument(
            "GGGS normal features must be contiguous CUDA float32 [N,4]");
    const std::size_t count = normal_features.shape()[0];
    auto normals = tinytensor::Tensor::zeros(
        {count, std::size_t{3}}, tinytensor::Device::CUDA);
    if (count != 0)
        normal_features_forward_kernel<<<
            (count + k_threads - 1) / k_threads, k_threads>>>(
            normal_features.ptr<float>(), normals.ptr<float>(), count);
    check_cuda(cudaGetLastError(), "convert GGGS normal features");
    return normals;
}

tinytensor::Tensor normal_features_backward(
    const tinytensor::Tensor& normal_features,
    const tinytensor::Tensor& grad_normals) {
    if (!grad_normals.is_valid() || grad_normals.shape().rank() != 2 ||
        grad_normals.shape()[0] != normal_features.shape()[0] ||
        grad_normals.shape()[1] != 3)
        throw std::invalid_argument(
            "GGGS normal gradient must have shape [N,3]");
    const std::size_t count = normal_features.shape()[0];
    auto result = tinytensor::Tensor::zeros_like(normal_features);
    if (count != 0)
        normal_features_backward_kernel<<<
            (count + k_threads - 1) / k_threads, k_threads>>>(
            normal_features.ptr<float>(), grad_normals.ptr<float>(),
            result.ptr<float>(), count);
    check_cuda(cudaGetLastError(), "chain GGGS normal-feature gradients");
    return result;
}

void bake_3d_filter(GaussianModel& model) {
    if (!model.filter_3d.is_valid()) return;
    const std::size_t count = model.size();
    if (model.filter_3d.device() != tinytensor::Device::CUDA ||
        model.filter_3d.dtype() != tinytensor::DataType::Float32 ||
        model.filter_3d.shape().rank() != 2 ||
        model.filter_3d.shape()[0] != count ||
        model.filter_3d.shape()[1] != 1)
        throw std::invalid_argument(
            "GGGS 3D filter must be CUDA float32 shaped [N,1]");
    if (count != 0) {
        bake_3d_filter_kernel<<<
            (count + k_threads - 1) / k_threads, k_threads>>>(
            model.log_scales.ptr<float>(),
            model.opacity_logits.ptr<float>(),
            model.filter_3d.ptr<float>(), count);
        check_cuda(cudaGetLastError(), "bake GGGS 3D filter");
    }
    model.filter_3d = {};
}

AdcPlusPruneResult adc_plus_prune(
    const GaussianModel& model, const float minimum_opacity,
    const float maximum_bounds,
    const std::array<float, 3>& scene_center,
    const std::size_t maximum_count) {
    const std::size_t count = model.size();
    auto keep = tinytensor::Tensor::zeros_bool(
        {count}, tinytensor::Device::CUDA);
    auto hard = tinytensor::Tensor::zeros_bool(
        {count}, tinytensor::Device::CUDA);
    auto opacities = tinytensor::Tensor::empty(
        {count}, tinytensor::Device::CUDA);
    if (count != 0) {
        adc_plus_prune_kernel<<<
            (count + k_threads - 1) / k_threads, k_threads>>>(
            model.means.ptr<float>(), model.log_scales.ptr<float>(),
            model.quaternions.ptr<float>(),
            model.opacity_logits.ptr<float>(), model.sh.ptr<float>(),
            model.sh.numel() / count, keep.ptr<unsigned char>(),
            hard.ptr<unsigned char>(), opacities.ptr<float>(), count,
            minimum_opacity, maximum_bounds, scene_center[0],
            scene_center[1], scene_center[2]);
        check_cuda(cudaGetLastError(), "select ADC+ prune mask");
    }

    std::size_t retained = keep.count_nonzero();
    if ((retained == 0 && count != 0) ||
        retained > maximum_count) {
        std::vector<float> host_opacity(count);
        std::vector<unsigned char> host_keep(count);
        std::vector<unsigned char> host_hard(count);
        check_cuda(cudaMemcpy(
            host_opacity.data(), opacities.data_ptr(),
            count * sizeof(float), cudaMemcpyDeviceToHost),
            "download ADC+ fallback opacities");
        check_cuda(cudaMemcpy(
            host_keep.data(), keep.data_ptr(),
            count * sizeof(unsigned char), cudaMemcpyDeviceToHost),
            "download ADC+ fallback keep mask");
        check_cuda(cudaMemcpy(
            host_hard.data(), hard.data_ptr(),
            count * sizeof(unsigned char), cudaMemcpyDeviceToHost),
            "download ADC+ fallback hard-prune mask");
        if (retained == 0) {
            std::size_t best = count;
            float best_opacity = -1.F;
            for (std::size_t index = 0; index < count; ++index) {
                if (!host_hard[index] &&
                    host_opacity[index] > best_opacity) {
                    best = index;
                    best_opacity = host_opacity[index];
                }
            }
            if (best != count) {
                host_keep[best] = 1U;
                retained = 1;
            }
        }
        if (retained > maximum_count) {
            std::vector<std::pair<float, std::size_t>> rows;
            rows.reserve(retained);
            for (std::size_t index = 0; index < count; ++index)
                if (host_keep[index])
                    rows.emplace_back(host_opacity[index], index);
            std::sort(rows.begin(), rows.end());
            for (std::size_t index = 0;
                 index < retained - maximum_count; ++index)
                host_keep[rows[index].second] = 0U;
            retained = maximum_count;
        }
        check_cuda(cudaMemcpy(
            keep.data_ptr(), host_keep.data(),
            count * sizeof(unsigned char), cudaMemcpyHostToDevice),
            "upload ADC+ fallback keep mask");
    }
    auto keep_indices = keep.nonzero().squeeze(1).to(
        tinytensor::DataType::Int32);
    return {std::move(keep_indices), std::move(opacities),
            count - retained};
}

tinytensor::Tensor compute_3d_filter(
    const tinytensor::Tensor& means, const std::vector<Camera>& cameras,
    const float minimum_scale_factor,
    const bool all_camera_euclidean) {
    if (!means.is_valid() || means.device() != tinytensor::Device::CUDA ||
        means.dtype() != tinytensor::DataType::Float32 ||
        means.shape().rank() != 2 || means.shape()[1] != 3)
        throw std::invalid_argument(
            "GGGS 3D filter requires CUDA float32 means shaped [N,3]");
    if (cameras.empty())
        throw std::invalid_argument(
            "GGGS 3D filter requires at least one camera");
    const std::size_t count = means.shape()[0];
    if (count == 0)
        return tinytensor::Tensor::empty(
            {std::size_t{0}, std::size_t{1}}, tinytensor::Device::CUDA);
    std::vector<float> packed(cameras.size() * 20);
    float maximum_focal = 0.F;
    for (std::size_t view = 0; view < cameras.size(); ++view) {
        const Camera& camera = cameras[view];
        std::copy(
            camera.world_to_camera.begin(), camera.world_to_camera.end(),
            packed.begin() + static_cast<std::ptrdiff_t>(20 * view));
        packed[20 * view + 16] = camera.fx;
        packed[20 * view + 17] = camera.fy;
        packed[20 * view + 18] = static_cast<float>(camera.width);
        packed[20 * view + 19] = static_cast<float>(camera.height);
        maximum_focal = std::max(maximum_focal, camera.fx);
    }
    maximum_focal = std::max(maximum_focal, 1e-6F);
    const auto camera_tensor = tinytensor::Tensor::from_vector(
        packed, {cameras.size(), std::size_t{20}},
        tinytensor::Device::CUDA);
    auto result = tinytensor::Tensor::empty(
        {means.shape()[0], std::size_t{1}}, tinytensor::Device::CUDA);
    auto maximum_bits = tinytensor::Tensor::zeros(
        {std::size_t{1}}, tinytensor::Device::CUDA,
        tinytensor::DataType::Int32);
    compute_3d_filter_distance_kernel<<<
        (count + k_threads - 1) / k_threads, k_threads>>>(
            means.ptr<float>(), camera_tensor.ptr<float>(), count,
            cameras.size(), result.ptr<float>(),
            reinterpret_cast<unsigned*>(maximum_bits.data_ptr()),
            all_camera_euclidean);
    check_cuda(cudaGetLastError(), "compute GGGS 3D filter distances");
    unsigned maximum_distance_bits{};
    check_cuda(cudaMemcpy(
        &maximum_distance_bits, maximum_bits.data_ptr(), sizeof(unsigned),
        cudaMemcpyDeviceToHost), "download GGGS 3D filter maximum");
    float maximum_distance{};
    std::memcpy(
        &maximum_distance, &maximum_distance_bits, sizeof(maximum_distance));
    if (!(maximum_distance > 0.F) || !std::isfinite(maximum_distance))
        maximum_distance = 1.F;
    finalize_3d_filter_kernel<<<
        (count + k_threads - 1) / k_threads, k_threads>>>(
            result.ptr<float>(), count, maximum_distance,
            1.F / maximum_focal,
            std::sqrt(std::max(minimum_scale_factor, 0.F)));
    check_cuda(cudaGetLastError(), "finalize GGGS 3D filter");
    return result;
}

MultiViewLoss add_multi_view_loss(
    const tinytensor::Tensor& sampled_neighbour_points,
    const tinytensor::Tensor& sampled_inside,
    const RenderResult& reference_render,
    const TrainingView& reference,
    const TrainingView& neighbour,
    const TrainingOptions& options,
    LossGradients& gradients,
    tinytensor::Tensor& grad_sampled_points,
    const bool collect_scalar_terms,
    tinytensor::Tensor* stability_accumulator) {
    const std::size_t pixels =
        static_cast<std::size_t>(reference.camera.width) *
        reference.camera.height;
    if (pixels == 0 ||
        (options.multi_view_geo_weight <= 0.F &&
         options.multi_view_ncc_weight <= 0.F))
        return {};
    // Row-major reference-camera -> neighbour-camera rigid transform.
    float rr[9], rn[9], tr[3], tn[3];
    for (int row = 0; row < 3; ++row) {
        tr[row] = reference.camera.world_to_camera[12 + row];
        tn[row] = neighbour.camera.world_to_camera[12 + row];
        for (int column = 0; column < 3; ++column) {
            rr[3 * row + column] =
                reference.camera.world_to_camera[4 * column + row];
            rn[3 * row + column] =
                neighbour.camera.world_to_camera[4 * column + row];
        }
    }
    std::vector<float> transform(12, 0.F);
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            for (int k = 0; k < 3; ++k)
                transform[3 * row + column] +=
                    rn[3 * row + k] * rr[3 * column + k];
        }
        transform[9 + row] = tn[row];
        for (int column = 0; column < 3; ++column)
            transform[9 + row] -=
                transform[3 * row + column] * tr[column];
    }
    const auto transform_tensor = tinytensor::Tensor::from_vector(
        transform, {std::size_t{12}}, tinytensor::Device::CUDA);
    grad_sampled_points = tinytensor::Tensor::zeros_like(
        sampled_neighbour_points);
    auto ncc_depth = tinytensor::Tensor::zeros_like(
        reference_render.median_depth);
    auto ncc_normal = tinytensor::Tensor::zeros_like(reference_render.normal);
    auto terms = tinytensor::Tensor::zeros(
        {std::size_t{5}}, tinytensor::Device::CUDA);
    const bool enable_ncc = options.multi_view_ncc_weight > 0.F;
    const bool count_geometry_candidates =
        collect_scalar_terms || stability_accumulator != nullptr;
    const dim3 multi_view_block{
        k_multi_view_block_x, k_multi_view_block_y};
    const dim3 multi_view_grid{
        (reference.camera.width + k_multi_view_block_x - 1) /
            k_multi_view_block_x,
        (reference.camera.height + k_multi_view_block_y - 1) /
            k_multi_view_block_y};
    multi_view_raw_kernel<<<multi_view_grid, multi_view_block>>>(
        reference_render.median_depth.ptr<float>(),
        reference_render.normal.ptr<float>(),
        enable_ncc ? reference.gray.ptr<float>() : nullptr,
        sampled_neighbour_points.ptr<float>(), sampled_inside.ptr<bool>(),
        enable_ncc ? neighbour.gray.ptr<float>() : nullptr,
        transform_tensor.ptr<float>(),
        reference.has_mask ? reference.mask.ptr<float>() : nullptr,
        neighbour.has_mask ? neighbour.mask.ptr<float>() : nullptr,
        reference.has_mask, neighbour.has_mask,
        reference.camera, neighbour.camera,
        options.multi_view_pixel_noise_threshold,
        options.multi_view_robust_ncc,
        enable_ncc, count_geometry_candidates,
        options.multi_view_ncc_lambda_reference,
        options.multi_view_ncc_sharpness,
        std::clamp(options.multi_view_ncc_min_weight, 0.F, 1.F),
        grad_sampled_points.ptr<float>(), ncc_depth.ptr<float>(),
        ncc_normal.ptr<float>(), terms.ptr<float>(), pixels);
    check_cuda(cudaGetLastError(), "compute GGGS multi-view loss");
    if (stability_accumulator != nullptr) {
        if (stability_accumulator->device() !=
                tinytensor::Device::CUDA ||
            stability_accumulator->dtype() !=
                tinytensor::DataType::Float32 ||
            stability_accumulator->numel() != 3)
            throw std::invalid_argument(
                "GGGS stability accumulator must be CUDA float32[3]");
        accumulate_multi_view_stability_kernel<<<1, 1>>>(
            terms.ptr<float>(), stability_accumulator->ptr<float>());
        check_cuda(
            cudaGetLastError(),
            "accumulate GGGS multi-view stability terms");
    }
    add_multi_view_gradients_kernel<<<
        (pixels + k_threads - 1) / k_threads, k_threads>>>(
        gradients.depth.ptr<float>(), gradients.normal.ptr<float>(),
        grad_sampled_points.ptr<float>(), ncc_depth.ptr<float>(),
        ncc_normal.ptr<float>(), terms.ptr<float>(),
        options.multi_view_geo_weight, options.multi_view_ncc_weight, pixels);
    check_cuda(cudaGetLastError(), "accumulate GGGS multi-view gradients");
    if (!collect_scalar_terms) return {};
    const std::vector<float> values = terms.to_vector();
    MultiViewLoss result;
    result.geometry_pixels = static_cast<std::size_t>(values[1]);
    result.geometry_candidates = static_cast<std::size_t>(values[4]);
    result.ncc_pixels = static_cast<std::size_t>(values[3]);
    result.geometry = result.geometry_pixels != 0
        ? values[0] / values[1]
        : 0.F;
    result.ncc = result.ncc_pixels != 0 ? values[2] / values[3] : 0.F;
    return result;
}

GeometryDistributionSummary summarize_geometry_distribution(
    const GaussianModel& model) {
    GeometryDistributionSummary summary;
    const std::size_t count = model.size();
    if (count == 0) return summary;
    auto terms = tinytensor::Tensor::zeros(
        {static_cast<std::size_t>(k_geometry_summary_terms)},
        tinytensor::Device::CUDA);
    const std::size_t required_blocks =
        (count + k_threads - 1) / k_threads;
    const unsigned blocks = static_cast<unsigned>(
        std::min<std::size_t>(required_blocks, 1024));
    geometry_distribution_summary_kernel<<<blocks, k_threads>>>(
        model.log_scales.ptr<float>(),
        model.opacity_logits.ptr<float>(), terms.ptr<float>(), count);
    check_cuda(
        cudaGetLastError(), "summarize GGGS geometry distribution");
    const std::vector<float> values = terms.to_vector();
    const float inverse_count = 1.F / static_cast<float>(count);
    const auto moments = [&](const unsigned offset) {
        const float mean = values[offset] * inverse_count;
        const float second = values[offset + 1] * inverse_count;
        return std::array<float, 2>{
            mean, std::sqrt(std::max(second - mean * mean, 0.F))};
    };
    const auto opacity = moments(0);
    const auto scale = moments(2);
    const auto anisotropy = moments(4);
    summary.opacity_mean = opacity[0];
    summary.opacity_stddev = opacity[1];
    summary.log_scale_mean = scale[0];
    summary.log_scale_stddev = scale[1];
    summary.log_anisotropy_mean = anisotropy[0];
    summary.log_anisotropy_stddev = anisotropy[1];
    return summary;
}

tinytensor::Tensor unproject_depth_to_world(
    const tinytensor::Tensor& depth, const Camera& camera) {
    const std::size_t pixels =
        static_cast<std::size_t>(camera.width) * camera.height;
    if (depth.numel() != pixels)
        throw std::invalid_argument(
            "GGGS depth unprojection shape does not match camera");
    auto result = tinytensor::Tensor::empty(
        {pixels, std::size_t{3}}, tinytensor::Device::CUDA);
    unproject_depth_to_world_kernel<<<
        (pixels + k_threads - 1) / k_threads, k_threads>>>(
        depth.ptr<float>(), result.ptr<float>(), camera, pixels);
    check_cuda(cudaGetLastError(), "unproject GGGS reference depth");
    return result;
}

void add_sample_depth_point_gradients(
    const Camera& reference_camera,
    const tinytensor::Tensor& grad_world_points,
    LossGradients& image_gradients) {
    const std::size_t pixels = static_cast<std::size_t>(
        reference_camera.width) * reference_camera.height;
    add_point_depth_gradients_kernel<<<
        (pixels + k_threads - 1) / k_threads, k_threads>>>(
        grad_world_points.ptr<float>(), image_gradients.depth.ptr<float>(),
        reference_camera, pixels);
    check_cuda(cudaGetLastError(),
               "accumulate GGGS sample-depth point gradients");
}

void add_sample_depth_model_gradients(
    const DepthSampleGradients& sample_gradients,
    ModelGradients& model_gradients) {
    const auto add = [](tinytensor::Tensor& target,
                        const tinytensor::Tensor& source) {
        const std::size_t count = target.numel();
        add_tensor_in_place_kernel<<<
            (count + k_threads - 1) / k_threads, k_threads>>>(
            target.ptr<float>(), source.ptr<float>(), count);
    };
    add(model_gradients.means, sample_gradients.model.means);
    add(model_gradients.log_scales, sample_gradients.model.log_scales);
    add(model_gradients.quaternions, sample_gradients.model.quaternions);
    add(model_gradients.opacity_logits,
        sample_gradients.model.opacity_logits);
    check_cuda(cudaGetLastError(), "accumulate GGGS sample-depth gradients");
}

void add_model_gradients(
    const ModelGradients& source, ModelGradients& destination,
    const bool include_refine_weight) {
    const auto add = [](const tinytensor::Tensor& source_tensor,
                        tinytensor::Tensor& destination_tensor) {
        if (!source_tensor.is_valid()) return;
        if (!destination_tensor.is_valid() ||
            destination_tensor.shape() != source_tensor.shape())
            throw std::invalid_argument(
                "Cannot merge incompatible GGGS model gradients");
        const std::size_t count = destination_tensor.numel();
        add_tensor_in_place_kernel<<<
            (count + k_threads - 1) / k_threads, k_threads>>>(
            destination_tensor.ptr<float>(), source_tensor.ptr<float>(),
            count);
    };
    add(source.means, destination.means);
    add(source.log_scales, destination.log_scales);
    add(source.quaternions, destination.quaternions);
    add(source.opacity_logits, destination.opacity_logits);
    add(source.sh, destination.sh);
    if (include_refine_weight)
        add(source.refine_weight, destination.refine_weight);
    check_cuda(cudaGetLastError(), "merge GGGS model gradients");
}

void chain_parameter_gradients(
    const GaussianModel& model, const ActivatedParameters& activated,
    const tinytensor::Tensor& grad_scales,
    const tinytensor::Tensor& grad_quaternions,
    const tinytensor::Tensor& grad_opacities,
    ModelGradients& gradients) {
    const std::size_t count = model.size();
    gradients.log_scales = tinytensor::Tensor::zeros_like(model.log_scales);
    gradients.quaternions = tinytensor::Tensor::zeros_like(model.quaternions);
    gradients.opacity_logits = tinytensor::Tensor::zeros_like(model.opacity_logits);
    if (count == 0) return;
    chain_gradient_kernel<<<(count + k_threads - 1) / k_threads, k_threads>>>(
        model.log_scales.ptr<float>(), model.quaternions.ptr<float>(),
        model.opacity_logits.ptr<float>(),
        model.filter_3d.is_valid() ? model.filter_3d.ptr<float>() : nullptr,
        activated.scales.ptr<float>(), activated.opacities.ptr<float>(),
        grad_scales.ptr<float>(),
        grad_quaternions.ptr<float>(), grad_opacities.ptr<float>(),
        gradients.log_scales.ptr<float>(), gradients.quaternions.ptr<float>(),
        gradients.opacity_logits.ptr<float>(), count);
    check_cuda(cudaGetLastError(), "chain GGGS parameter gradients");
}

LossGradients compute_training_loss(
    const RenderResult& rendered, const TrainingView& target,
    const TrainingOptions& options, const bool collect_scalar_terms,
    const bool depth_normal_active) {
    const std::size_t pixels = static_cast<std::size_t>(target.camera.width) *
                               target.camera.height;
    LossGradients result{
        tinytensor::Tensor::zeros_like(rendered.color),
        tinytensor::Tensor::zeros_like(rendered.alpha),
        tinytensor::Tensor::zeros_like(rendered.median_depth),
        tinytensor::Tensor::zeros_like(rendered.normal)};
    tinytensor::Tensor terms;
    if (collect_scalar_terms)
        terms = tinytensor::Tensor::zeros({4}, tinytensor::Device::CUDA);
    const bool mask_enabled = options.use_mask && target.has_mask;
    const bool use_fused_photometric =
        target.camera.width > 10 && target.camera.height > 10;
    const float ssim_weight = std::clamp(options.ssim_weight, 0.F, 1.F);
    loss_kernel<<<(pixels + k_threads - 1) / k_threads, k_threads>>>(
        rendered.color.ptr<float>(), rendered.alpha.ptr<float>(),
        rendered.median_depth.ptr<float>(), rendered.normal.ptr<float>(),
        target.rgb.ptr<float>(), target.depth.ptr<float>(),
        target.normal.ptr<float>(), target.mask.ptr<float>(),
        result.color.ptr<float>(), result.alpha.ptr<float>(),
        result.depth.ptr<float>(), result.normal.ptr<float>(),
        collect_scalar_terms ? terms.ptr<float>() : nullptr,
        pixels, use_fused_photometric ? 0.F : options.photometric_weight,
        options.use_mvs_depth ? options.depth_weight : 0.F,
        options.use_mvs_normals ? options.normal_weight : 0.F,
        mask_enabled,
        options.alpha_mode == AlphaMode::masked ? 0 : 1,
        options.match_alpha_weight, 1.F,
        options.geometry_epsilon);
    check_cuda(cudaGetLastError(), "compute GGGS training loss");
    if (use_fused_photometric)
        fused_l1_ssim_loss(
            rendered.color, target.rgb, target.mask, mask_enabled,
            ssim_weight, options.photometric_weight, result.color,
            collect_scalar_terms ? terms.ptr<float>() : nullptr,
            target.camera.width, target.camera.height);
    if (depth_normal_active && options.depth_normal_weight > 0.F) {
        depth_normal_consistency_kernel<<<
            (pixels + k_threads - 1) / k_threads, k_threads>>>(
            rendered.median_depth.ptr<float>(), rendered.normal.ptr<float>(),
            result.depth.ptr<float>(), result.normal.ptr<float>(),
            collect_scalar_terms ? terms.ptr<float>() : nullptr,
            target.camera.width, target.camera.height, target.camera.fx,
            target.camera.fy, target.camera.cx, target.camera.cy,
            options.depth_normal_weight);
        check_cuda(
            cudaGetLastError(), "compute GGGS depth-normal consistency");
    }
    if (collect_scalar_terms) {
        std::array<float, 4> host{};
        check_cuda(cudaMemcpy(
            host.data(), terms.ptr<float>(), sizeof(host), cudaMemcpyDeviceToHost),
            "download GGGS loss");
        result.rgb = host[0];
        result.alpha_value = host[3];
        result.depth_value = host[1];
        result.normal_value = host[2];
        result.total = host[0] + host[1] + host[2] + host[3];
    }
    return result;
}

LossGradients compute_normal_field_loss(
    const RenderResult& rendered, const Camera& camera, const float weight,
    const bool collect_scalar_terms) {
    LossGradients result{
        tinytensor::Tensor::zeros_like(rendered.color),
        tinytensor::Tensor::zeros_like(rendered.alpha),
        tinytensor::Tensor::zeros_like(rendered.median_depth),
        tinytensor::Tensor::zeros_like(rendered.normal)};
    const std::size_t pixels =
        static_cast<std::size_t>(camera.width) * camera.height;
    tinytensor::Tensor terms;
    if (collect_scalar_terms)
        terms = tinytensor::Tensor::zeros({4}, tinytensor::Device::CUDA);
    if (pixels != 0 && weight > 0.F) {
        normal_field_consistency_kernel<<<
            (pixels + k_threads - 1) / k_threads, k_threads>>>(
            rendered.median_depth.ptr<float>(), rendered.color.ptr<float>(),
            result.depth.ptr<float>(), result.color.ptr<float>(),
            collect_scalar_terms ? terms.ptr<float>() : nullptr, camera,
            weight);
        check_cuda(cudaGetLastError(), "compute GGGS normal-field loss");
    }
    if (collect_scalar_terms) {
        std::array<float, 4> host{};
        check_cuda(
            cudaMemcpy(
                host.data(), terms.ptr<float>(), sizeof(host),
                cudaMemcpyDeviceToHost),
            "download GGGS normal-field loss");
        result.normal_value = host[2];
        result.total = host[2];
    }
    return result;
}

AdamState make_adam_state(const tinytensor::Tensor& parameter) {
    return {tinytensor::Tensor::zeros_like(parameter),
            tinytensor::Tensor::zeros_like(parameter)};
}

AdamState make_reduced_second_adam_state(
    const tinytensor::Tensor& parameter) {
    const auto dimensions = parameter.shape().dims();
    if (dimensions.empty())
        throw std::invalid_argument(
            "reduced-second Adam requires at least one parameter dimension");
    return {
        tinytensor::Tensor::zeros_like(parameter),
        tinytensor::Tensor::zeros(
            {dimensions.front()}, parameter.device())};
}

void adam_step(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, const float learning_rate, const unsigned step,
    const TrainingOptions& options, const std::size_t group_stride,
    const float secondary_learning_rate, const float clamp_min,
    const float clamp_max) {
    const std::size_t count = parameter.numel();
    if (count == 0) return;
    const float correction1 = 1.F - std::pow(options.beta1, static_cast<float>(step));
    const float correction2 = 1.F - std::pow(options.beta2, static_cast<float>(step));
    adam_kernel<<<(count + k_threads - 1) / k_threads, k_threads>>>(
        parameter.ptr<float>(), gradient.ptr<float>(), state.first.ptr<float>(),
        state.second.ptr<float>(), count, learning_rate,
        secondary_learning_rate, group_stride, options.beta1, options.beta2,
        correction1, correction2, options.adam_epsilon, clamp_min, clamp_max);
    check_cuda(cudaGetLastError(), "GGGS Adam update");
}

void adam_step_reduced_second(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, const float learning_rate, const unsigned step,
    const TrainingOptions& options, const std::size_t row_stride,
    const float secondary_learning_rate) {
    const std::size_t count = parameter.numel();
    if (count == 0) return;
    if (row_stride < 3 || count % row_stride != 0)
        throw std::invalid_argument("invalid reduced-second Adam row stride");
    const std::size_t row_count = count / row_stride;
    if (state.first.numel() != count || state.second.numel() != row_count)
        throw std::invalid_argument(
            "reduced-second Adam state has an incompatible shape");
    const float correction1 =
        1.F - std::pow(options.beta1, static_cast<float>(step));
    const float correction2 =
        1.F - std::pow(options.beta2, static_cast<float>(step));
    adam_reduced_second_kernel<<<
        row_count, k_reduced_adam_threads>>>(
        parameter.ptr<float>(), gradient.ptr<float>(), state.first.ptr<float>(),
        state.second.ptr<float>(), row_count, row_stride, row_stride,
        learning_rate, secondary_learning_rate, options.beta1, options.beta2,
        correction1, correction2, options.adam_epsilon);
    check_cuda(cudaGetLastError(), "GGGS reduced-second Adam update");
}

void adam_step_active_prefix(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, const float learning_rate, const unsigned step,
    const TrainingOptions& options, const std::size_t full_row_stride,
    const std::size_t active_row_stride,
    const float secondary_learning_rate) {
    const std::size_t count = parameter.numel();
    if (count == 0) return;
    if (full_row_stride == 0 || active_row_stride < 3 ||
        active_row_stride > full_row_stride ||
        count % full_row_stride != 0)
        throw std::invalid_argument("invalid active-prefix Adam row strides");
    const std::size_t active_count =
        count / full_row_stride * active_row_stride;
    const std::size_t row_count = count / full_row_stride;
    const float correction1 =
        1.F - std::pow(options.beta1, static_cast<float>(step));
    const float correction2 =
        1.F - std::pow(options.beta2, static_cast<float>(step));
    if (state.second.numel() == row_count) {
        if (state.first.numel() != count)
            throw std::invalid_argument(
                "active-prefix reduced-second Adam state has an "
                "incompatible shape");
        adam_reduced_second_kernel<<<
            row_count, k_reduced_adam_threads>>>(
            parameter.ptr<float>(), gradient.ptr<float>(),
            state.first.ptr<float>(), state.second.ptr<float>(), row_count,
            full_row_stride, active_row_stride, learning_rate,
            secondary_learning_rate, options.beta1, options.beta2,
            correction1, correction2, options.adam_epsilon);
        check_cuda(
            cudaGetLastError(),
            "GGGS active-prefix reduced-second Adam update");
        return;
    }
    if (state.first.numel() != count || state.second.numel() != count)
        throw std::invalid_argument(
            "active-prefix Adam state has an incompatible shape");
    adam_active_prefix_kernel<<<
        (active_count + k_threads - 1) / k_threads, k_threads>>>(
        parameter.ptr<float>(), gradient.ptr<float>(), state.first.ptr<float>(),
        state.second.ptr<float>(), active_count, full_row_stride,
        active_row_stride, learning_rate, secondary_learning_rate,
        options.beta1, options.beta2, correction1, correction2,
        options.adam_epsilon);
    check_cuda(cudaGetLastError(), "GGGS active-prefix Adam update");
}

void constrain_scale_ratio(
    tinytensor::Tensor& log_scales, const float maximum_ratio) {
    if (maximum_ratio <= 1.F || log_scales.numel() == 0) return;
    const std::size_t count = log_scales.numel() / 3;
    constrain_scale_ratio_kernel<<<
        (count + k_threads - 1) / k_threads, k_threads>>>(
        log_scales.ptr<float>(), count, std::log(maximum_ratio));
    check_cuda(cudaGetLastError(), "constrain GGGS scale ratio");
}

DensificationStats make_densification_stats(const std::size_t count) {
    return {
        tinytensor::Tensor::zeros({count}, tinytensor::Device::CUDA),
        tinytensor::Tensor::zeros({count}, tinytensor::Device::CUDA),
        tinytensor::Tensor::zeros({count}, tinytensor::Device::CUDA),
        tinytensor::Tensor::zeros({count}, tinytensor::Device::CUDA)};
}

void accumulate_densification_stats(
    const tinytensor::Tensor& refine_weight,
    const tinytensor::Tensor& visibility,
    const tinytensor::Tensor& radii,
    DensificationStats& stats,
    const std::uint32_t width,
    const std::uint32_t height,
    const bool use_maximum,
    const bool require_contribution_visibility) {
    const std::size_t count = refine_weight.numel();
    if (count == 0) return;
    const float inverse_resolution = 1.F /
        static_cast<float>(std::max<std::uint32_t>(1, std::min(width, height)));
    accumulate_densification_kernel<<<
        (count + k_threads - 1) / k_threads, k_threads>>>(
        refine_weight.ptr<float>(), visibility.ptr<float>(), radii.ptr<int>(),
        stats.gradient.ptr<float>(), stats.count.ptr<float>(),
        stats.max_screen_radius.ptr<float>(), stats.priority.ptr<float>(),
        count, inverse_resolution, use_maximum,
        require_contribution_visibility);
    check_cuda(cudaGetLastError(), "accumulate GGGS densification stats");
}

void split_gaussians(
    GaussianModel& parents,
    GaussianModel& children,
    const tinytensor::Tensor& parent_indices,
    const tinytensor::Tensor& random_samples,
    const tinytensor::Tensor& screen_sizes,
    const int mode,
    const float minimum_opacity,
    const float split_at_screen_size) {
    const std::size_t count = parent_indices.numel();
    if (count == 0) return;
    split_gaussians_kernel<<<
        (count + k_threads - 1) / k_threads, k_threads>>>(
        parents.means.ptr<float>(), parents.log_scales.ptr<float>(),
        parents.opacity_logits.ptr<float>(), parents.quaternions.ptr<float>(),
        children.means.ptr<float>(), children.log_scales.ptr<float>(),
        children.opacity_logits.ptr<float>(), parent_indices.ptr<int>(),
        random_samples.ptr<float>(),
        screen_sizes.is_valid() ? screen_sizes.ptr<float>() : nullptr,
        count, mode, std::clamp(minimum_opacity, 1e-8F, 0.49F),
        std::max(split_at_screen_size, 0.F));
    check_cuda(cudaGetLastError(), "split GGGS Gaussians");
}

void apply_adc_decay(
    GaussianModel& model, const float opacity_decay,
    const float scale_decay) {
    if (model.size() == 0) return;
    const float scale_factor = std::max(1.F - scale_decay, 1e-6F);
    adc_decay_kernel<<<
        (model.size() + k_threads - 1) / k_threads, k_threads>>>(
        model.log_scales.ptr<float>(), model.opacity_logits.ptr<float>(),
        model.size(), std::max(opacity_decay, 0.F), std::log(scale_factor));
    check_cuda(cudaGetLastError(), "decay ADC Gaussian parameters");
}

void inject_adc_noise(
    GaussianModel& model,
    const tinytensor::Tensor& visibility,
    const float standard_deviation,
    const float maximum_noise,
    const unsigned seed) {
    if (model.size() == 0 || standard_deviation <= 0.F) return;
    inject_adc_noise_kernel<<<
        (model.size() + k_threads - 1) / k_threads, k_threads>>>(
        model.means.ptr<float>(), model.opacity_logits.ptr<float>(),
        visibility.ptr<float>(), model.size(), standard_deviation,
        std::max(maximum_noise, 0.F), seed);
    check_cuda(cudaGetLastError(), "inject ADC exploration noise");
}

void reset_opacity(GaussianModel& model, const float maximum_opacity) {
    if (model.size() == 0) return;
    const float opacity = std::clamp(maximum_opacity, 1e-8F, 1.F - 1e-8F);
    reset_opacity_kernel<<<
        (model.size() + k_threads - 1) / k_threads, k_threads>>>(
        model.opacity_logits.ptr<float>(), model.size(),
        std::log(opacity / (1.F - opacity)));
    check_cuda(cudaGetLastError(), "reset GGGS opacity");
}

}  // namespace aetherscan::splat::detail
