#include "bilateral_grid.hpp"
#include "cuda_common.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace aetherscan::splat::detail {
namespace {

constexpr int k_affine_channels = 12;
constexpr float k_c2g_r = 0.299F;
constexpr float k_c2g_g = 0.587F;
constexpr float k_c2g_b = 0.114F;

// Identity affine [[1,0,0,0],[0,1,0,0],[0,0,1,0]] in the grid's channel order.
__constant__ float k_identity_affine[k_affine_channels] = {
    1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F, 0.F, 1.F, 0.F};

// Vector reduction of four consecutive grid coefficients. The 12 coefficients
// of one cell are 48 bytes, so every group of four starts on a 16-byte
// boundary. sm_90 and later reduce a float4 in one instruction; older targets
// fall back to four scalar reductions.
__device__ __forceinline__ void red_add4(
    float* __restrict__ destination, const float4 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile(
        "red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(destination),
        "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
        : "memory");
#else
    atomicAdd(destination + 0, value.x);
    atomicAdd(destination + 1, value.y);
    atomicAdd(destination + 2, value.z);
    atomicAdd(destination + 3, value.w);
#endif
}

__device__ __forceinline__ long long cell_offset(
    const int view, const int z, const int y, const int x, const int luma,
    const int grid_h, const int grid_w) {
    return (((((static_cast<long long>(view) * luma + z) * grid_h + y) *
                  grid_w +
              x) *
             k_affine_channels));
}

__global__ void bilateral_identity_kernel(
    float* grids, const long long cells) {
    const long long index = static_cast<long long>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= cells) return;
    const long long base = index * k_affine_channels;
    grids[base + 0] = 1.F;
    grids[base + 1] = 0.F;
    grids[base + 2] = 0.F;
    grids[base + 3] = 0.F;
    grids[base + 4] = 0.F;
    grids[base + 5] = 1.F;
    grids[base + 6] = 0.F;
    grids[base + 7] = 0.F;
    grids[base + 8] = 0.F;
    grids[base + 9] = 0.F;
    grids[base + 10] = 1.F;
    grids[base + 11] = 0.F;
}

// One block per (row, coefficient). The block optionally subtracts the row mean
// of that coefficient (the projection onto the affine-with-identity-mean
// subspace) and always bounds the coefficient's distance from identity: without
// the bound an Adam update on a table this large can diverge and produce
// non-finite renders.
__global__ void bilateral_constrain_kernel(
    float* grids, const int views, const int luma, const int grid_h,
    const int grid_w, const int project_mean, const float limit) {
    const int channel = blockIdx.x % k_affine_channels;
    const int view = blockIdx.x / k_affine_channels;
    if (view >= views) return;
    const long long cells = static_cast<long long>(luma) * grid_h * grid_w;
    if (cells == 0) return;
    float* base = grids +
        static_cast<long long>(view) * cells * k_affine_channels;
    const float identity = k_identity_affine[channel];
    if (!project_mean) {
        if (limit <= 0.F) return;
        for (long long cell = threadIdx.x; cell < cells; cell += blockDim.x) {
            float& value = base[cell * k_affine_channels + channel];
            value = fminf(fmaxf(value, identity - limit), identity + limit);
        }
        return;
    }
    float local = 0.F;
    for (long long cell = threadIdx.x; cell < cells; cell += blockDim.x)
        local += base[cell * k_affine_channels + channel];
    for (int offset = 16; offset > 0; offset >>= 1)
        local += __shfl_down_sync(0xffffffffU, local, offset);
    __shared__ float scratch[32];
    const unsigned warps = blockDim.x >> 5;
    if ((threadIdx.x & 31U) == 0U) scratch[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = threadIdx.x < static_cast<int>(warps)
            ? scratch[threadIdx.x]
            : 0.F;
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffU, value, offset);
        if (threadIdx.x == 0)
            scratch[0] = value / static_cast<float>(cells);
    }
    __syncthreads();
    const float offset = scratch[0] - identity;
    for (long long cell = threadIdx.x; cell < cells; cell += blockDim.x) {
        float& value = base[cell * k_affine_channels + channel];
        float updated = value - offset;
        if (limit > 0.F)
            updated = fminf(fmaxf(updated, identity - limit), identity + limit);
        value = updated;
    }
}

__global__ void bilateral_forward_kernel(
    const float* color, float* corrected, const float* grids, const int view,
    const int luma, const int grid_h, const int grid_w, const int height,
    const int width) {
    const int pixel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int pixels = height * width;
    if (pixel >= pixels) return;
    const int x = pixel % width;
    const int y = pixel / width;
    const float sr = color[pixel];
    const float sg = color[pixels + pixel];
    const float sb = color[2 * pixels + pixel];
    const float gx = width > 1 ? static_cast<float>(x) /
        static_cast<float>(width - 1) : 0.F;
    const float gy = height > 1 ? static_cast<float>(y) /
        static_cast<float>(height - 1) : 0.F;
    const float gz = fminf(fmaxf(k_c2g_r * sr + k_c2g_g * sg + k_c2g_b * sb,
        0.F), 1.F);
    const float fx_grid = gx * static_cast<float>(grid_w - 1);
    const float fy_grid = gy * static_cast<float>(grid_h - 1);
    const float fz_grid = gz * static_cast<float>(luma - 1);
    int x0 = static_cast<int>(floorf(fx_grid));
    int y0 = static_cast<int>(floorf(fy_grid));
    int z0 = static_cast<int>(floorf(fz_grid));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;
    x0 = min(max(x0, 0), grid_w - 1);
    x1 = min(max(x1, 0), grid_w - 1);
    y0 = min(max(y0, 0), grid_h - 1);
    y1 = min(max(y1, 0), grid_h - 1);
    z0 = min(max(z0, 0), luma - 1);
    z1 = min(max(z1, 0), luma - 1);
    const float fx = fx_grid - static_cast<float>(x0);
    const float fy = fy_grid - static_cast<float>(y0);
    const float fz = fz_grid - static_cast<float>(z0);
    const long long off000 =
        cell_offset(view, z0, y0, x0, luma, grid_h, grid_w);
    const long long off001 =
        cell_offset(view, z0, y0, x1, luma, grid_h, grid_w);
    const long long off010 =
        cell_offset(view, z0, y1, x0, luma, grid_h, grid_w);
    const long long off011 =
        cell_offset(view, z0, y1, x1, luma, grid_h, grid_w);
    const long long off100 =
        cell_offset(view, z1, y0, x0, luma, grid_h, grid_w);
    const long long off101 =
        cell_offset(view, z1, y0, x1, luma, grid_h, grid_w);
    const long long off110 =
        cell_offset(view, z1, y1, x0, luma, grid_h, grid_w);
    const long long off111 =
        cell_offset(view, z1, y1, x1, luma, grid_h, grid_w);
    float dr = 0.F;
    float dg = 0.F;
    float db = 0.F;
#pragma unroll
    for (int ci = 0; ci < k_affine_channels; ++ci) {
        const float c00 =
            grids[off000 + ci] * (1.F - fx) + grids[off001 + ci] * fx;
        const float c01 =
            grids[off010 + ci] * (1.F - fx) + grids[off011 + ci] * fx;
        const float c10 =
            grids[off100 + ci] * (1.F - fx) + grids[off101 + ci] * fx;
        const float c11 =
            grids[off110 + ci] * (1.F - fx) + grids[off111 + ci] * fx;
        const float c0 = c00 * (1.F - fy) + c01 * fy;
        const float c1 = c10 * (1.F - fy) + c11 * fy;
        const float val = c0 * (1.F - fz) + c1 * fz;
        const int si = ci % 4;
        const int di = ci / 4;
        const float coeff = si == 0 ? sr : si == 1 ? sg : si == 2 ? sb : 1.F;
        (di == 0 ? dr : di == 1 ? dg : db) += val * coeff;
    }
    corrected[pixel] = dr;
    corrected[pixels + pixel] = dg;
    corrected[2 * pixels + pixel] = db;
}

// One pixel of the grid-gradient scatter. The eight trilinear neighbours are
// visited once, and both the grid gradient and the input-colour gradient come
// out of that single pass.
template <bool k_vectorized>
__device__ __forceinline__ void bilateral_backward_pixel(
    const float* __restrict__ color, const float* __restrict__ grids,
    const float* __restrict__ output_grad, float* __restrict__ grid_grad,
    float* __restrict__ input_grad, const long long row_base, const int luma,
    const int grid_h, const int grid_w, const int height, const int width,
    const int pixel, const int pixels) {
    const int x = pixel % width;
    const int y = pixel / width;
    const float sr = color[pixel];
    const float sg = color[pixels + pixel];
    const float sb = color[2 * pixels + pixel];
    const float dr = output_grad[pixel];
    const float dg = output_grad[pixels + pixel];
    const float db = output_grad[2 * pixels + pixel];
    // The input gradient and every grid gradient are linear in the incoming
    // gradient, so a zero one -- a masked or background pixel, mostly --
    // contributes nothing anywhere.
    if (dr == 0.F && dg == 0.F && db == 0.F) {
        input_grad[pixel] = 0.F;
        input_grad[pixels + pixel] = 0.F;
        input_grad[2 * pixels + pixel] = 0.F;
        return;
    }
    const float gx = width > 1 ? static_cast<float>(x) /
        static_cast<float>(width - 1) : 0.F;
    const float gy = height > 1 ? static_cast<float>(y) /
        static_cast<float>(height - 1) : 0.F;
    const float gz_raw = k_c2g_r * sr + k_c2g_g * sg + k_c2g_b * sb;
    const bool gz_in_range = gz_raw >= 0.F && gz_raw <= 1.F;
    const float gz = fminf(fmaxf(gz_raw, 0.F), 1.F);
    const float fx_grid = gx * static_cast<float>(grid_w - 1);
    const float fy_grid = gy * static_cast<float>(grid_h - 1);
    const float fz_grid = gz * static_cast<float>(luma - 1);
    const int x0_raw = static_cast<int>(floorf(fx_grid));
    const int y0_raw = static_cast<int>(floorf(fy_grid));
    const int z0_raw = static_cast<int>(floorf(fz_grid));
    const float fx = fx_grid - static_cast<float>(x0_raw);
    const float fy = fy_grid - static_cast<float>(y0_raw);
    const float fz = fz_grid - static_cast<float>(z0_raw);
    const int x0 = min(max(x0_raw, 0), grid_w - 1);
    const int x1 = min(max(x0_raw + 1, 0), grid_w - 1);
    const int y0 = min(max(y0_raw, 0), grid_h - 1);
    const int y1 = min(max(y0_raw + 1, 0), grid_h - 1);
    const int z0 = min(max(z0_raw, 0), luma - 1);
    const int z1 = min(max(z0_raw + 1, 0), luma - 1);
    const float wx[2] = {1.F - fx, fx};
    const float wy[2] = {1.F - fy, fy};
    const float wz[2] = {1.F - fz, fz};
    const float luma_scale = static_cast<float>(luma - 1);
    float in_r = 0.F;
    float in_g = 0.F;
    float in_b = 0.F;
    float in_bias = 0.F;
    float gz_grad = 0.F;
#pragma unroll
    for (int cz = 0; cz < 2; ++cz) {
        const int z = cz == 0 ? z0 : z1;
#pragma unroll
        for (int cy = 0; cy < 2; ++cy) {
            const int cell_y = cy == 0 ? y0 : y1;
#pragma unroll
            for (int cx = 0; cx < 2; ++cx) {
                const int cell_x = cx == 0 ? x0 : x1;
                const int cell = (z * grid_h + cell_y) * grid_w + cell_x;
                const float* cell_in =
                    grids + row_base + cell * k_affine_channels;
                float* cell_out =
                    grid_grad + row_base + cell * k_affine_channels;
                float4 c0;
                float4 c1;
                float4 c2;
                if constexpr (k_vectorized) {
                    const float4* vector =
                        reinterpret_cast<const float4*>(cell_in);
                    c0 = vector[0];
                    c1 = vector[1];
                    c2 = vector[2];
                } else {
                    c0 = make_float4(
                        cell_in[0], cell_in[1], cell_in[2], cell_in[3]);
                    c1 = make_float4(
                        cell_in[4], cell_in[5], cell_in[6], cell_in[7]);
                    c2 = make_float4(
                        cell_in[8], cell_in[9], cell_in[10], cell_in[11]);
                }
                // VJP of the affine map: p[si] = sum_di c[di][si] * gout[di].
                const float p0 = c0.x * dr + c1.x * dg + c2.x * db;
                const float p1 = c0.y * dr + c1.y * dg + c2.y * db;
                const float p2 = c0.z * dr + c1.z * dg + c2.z * db;
                const float p3 = c0.w * dr + c1.w * dg + c2.w * db;
                const float weight = wx[cx] * wy[cy] * wz[cz];
                in_r += weight * p0;
                in_g += weight * p1;
                in_b += weight * p2;
                in_bias += weight * p3;
                // Luma is a clamped linear function of the input colour, so
                // its gradient only flows through the fractional z weight.
                gz_grad += wx[cx] * wy[cy] * (cz == 0 ? -1.F : 1.F) *
                    luma_scale * (p0 * sr + p1 * sg + p2 * sb + p3);
                const float4 g0 = make_float4(
                    weight * dr * sr, weight * dr * sg, weight * dr * sb,
                    weight * dr);
                const float4 g1 = make_float4(
                    weight * dg * sr, weight * dg * sg, weight * dg * sb,
                    weight * dg);
                const float4 g2 = make_float4(
                    weight * db * sr, weight * db * sg, weight * db * sb,
                    weight * db);
                if constexpr (k_vectorized) {
                    red_add4(cell_out, g0);
                    red_add4(cell_out + 4, g1);
                    red_add4(cell_out + 8, g2);
                } else {
                    atomicAdd(cell_out + 0, g0.x);
                    atomicAdd(cell_out + 1, g0.y);
                    atomicAdd(cell_out + 2, g0.z);
                    atomicAdd(cell_out + 3, g0.w);
                    atomicAdd(cell_out + 4, g1.x);
                    atomicAdd(cell_out + 5, g1.y);
                    atomicAdd(cell_out + 6, g1.z);
                    atomicAdd(cell_out + 7, g1.w);
                    atomicAdd(cell_out + 8, g2.x);
                    atomicAdd(cell_out + 9, g2.y);
                    atomicAdd(cell_out + 10, g2.z);
                    atomicAdd(cell_out + 11, g2.w);
                }
            }
        }
    }
    (void)in_bias;
    // The clamp on the luma coordinate is inactive inside [0, 1].
    if (!gz_in_range) gz_grad = 0.F;
    input_grad[pixel] = in_r + k_c2g_r * gz_grad;
    input_grad[pixels + pixel] = in_g + k_c2g_g * gz_grad;
    input_grad[2 * pixels + pixel] = in_b + k_c2g_b * gz_grad;
}

// Backward pass of the affine bilateral grid. One thread walks `k_pixels`
// pixels strided by the launch, which keeps several independent reduction
// streams in flight per warp: the kernel is limited by the latency of the grid
// reductions rather than by arithmetic, and four pixels per thread measured
// 1.8 ms faster than one on a 1728x1120 view.
//
// Reading a cell as three float4s and reducing its coefficients with float4
// reductions turns the previous 96 scalar loads plus 96 scalar atomicAdds per
// pixel into 24 vector loads plus 24 vector reductions. The previous version
// was 84% stalled on the L1 load/store queue ("LG throttle") at ~23 ms per
// 1728x1120 view; the arithmetic is unchanged.
//
// `k_vectorized` is false when the grid or its gradient is not 16-byte
// aligned, in which case the same math runs through scalar accesses.
template <bool k_vectorized, int k_pixels>
__global__ void bilateral_backward_kernel(
    const float* __restrict__ color, const float* __restrict__ grids,
    const float* __restrict__ output_grad, float* __restrict__ grid_grad,
    float* __restrict__ input_grad, const int view, const int luma,
    const int grid_h, const int grid_w, const int height, const int width) {
    const int pixels = height * width;
    const long long row_base = static_cast<long long>(view) *
        (luma * grid_h * grid_w) * k_affine_channels;
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    const int first = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
#pragma unroll
    for (int slot = 0; slot < k_pixels; ++slot) {
        const int pixel = first + slot * stride;
        if (pixel >= pixels) return;
        bilateral_backward_pixel<k_vectorized>(
            color, grids, output_grad, grid_grad, input_grad, row_base, luma,
            grid_h, grid_w, height, width, pixel, pixels);
    }
}

__global__ void bilateral_tv_backward_kernel(
    const float* grids, float* grads, const int views, const int luma,
    const int grid_h, const int grid_w, const float tv_weight) {
    const long long cells = static_cast<long long>(views) * luma * grid_h *
        grid_w;
    const long long cell =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= cells) return;
    long long tmp = cell;
    const int x = static_cast<int>(tmp % grid_w);
    tmp /= grid_w;
    const int y = static_cast<int>(tmp % grid_h);
    tmp /= grid_h;
    const int z = static_cast<int>(tmp % luma);
    tmp /= luma;
    const int n = static_cast<int>(tmp);
    const long long base = cell * k_affine_channels;
    const float nx = grid_w > 1
        ? tv_weight * 2.F /
            static_cast<float>(
                views * luma * grid_h * (grid_w - 1) * k_affine_channels)
        : 0.F;
    const float ny = grid_h > 1
        ? tv_weight * 2.F /
            static_cast<float>(
                views * luma * (grid_h - 1) * grid_w * k_affine_channels)
        : 0.F;
    const float nz = luma > 1
        ? tv_weight * 2.F /
            static_cast<float>(
                views * (luma - 1) * grid_h * grid_w * k_affine_channels)
        : 0.F;
    const long long sw = k_affine_channels;
    const long long sh = static_cast<long long>(grid_w) * k_affine_channels;
    const long long sl =
        static_cast<long long>(grid_h) * grid_w * k_affine_channels;
    (void)n;
#pragma unroll
    for (int c = 0; c < k_affine_channels; ++c) {
        const float val = grids[base + c];
        float g = 0.F;
        if (grid_w > 1 && x > 0) g += (val - grids[base + c - sw]) * nx;
        if (grid_w > 1 && x + 1 < grid_w)
            g += (val - grids[base + c + sw]) * nx;
        if (grid_h > 1 && y > 0) g += (val - grids[base + c - sh]) * ny;
        if (grid_h > 1 && y + 1 < grid_h)
            g += (val - grids[base + c + sh]) * ny;
        if (luma > 1 && z > 0) g += (val - grids[base + c - sl]) * nz;
        if (luma > 1 && z + 1 < luma) g += (val - grids[base + c + sl]) * nz;
        grads[base + c] += g;
    }
}

void launch_identity(float* grids, const long long cells) {
    if (cells <= 0) return;
    bilateral_identity_kernel<<<
        static_cast<unsigned>((cells + k_cuda_threads - 1) / k_cuda_threads),
        k_cuda_threads>>>(grids, cells);
    check_cuda(cudaGetLastError(), "initialize affine bilateral grid");
}

}  // namespace

BilateralGridState make_bilateral_grid_state(
    const std::size_t views, const TrainingOptions& options) {
    BilateralGridState state;
    if (views == 0) return state;
    state.grid_width = static_cast<int>(
        std::max(options.bilateral_grid_width, 1U));
    state.grid_height = static_cast<int>(
        std::max(options.bilateral_grid_height, 1U));
    state.luma = static_cast<int>(std::max(options.bilateral_grid_luma, 1U));
    state.shared = options.bilateral_grid_shared;
    const std::size_t rows = state.shared ? 1 : views;
    state.grids = tinytensor::Tensor::empty(
        {rows, static_cast<std::size_t>(state.luma),
            static_cast<std::size_t>(state.grid_height),
            static_cast<std::size_t>(state.grid_width),
            static_cast<std::size_t>(k_affine_channels)},
        tinytensor::Device::CUDA);
    const long long cells = static_cast<long long>(rows) * state.luma *
        state.grid_height * state.grid_width;
    launch_identity(state.grids.ptr<float>(), cells);
    state.gradient = tinytensor::Tensor::zeros_like(state.grids);
    state.adam = make_adam_state(state.grids);
    return state;
}

void apply_bilateral_grid(
    const tinytensor::Tensor& color, BilateralGridState& state,
    const std::size_t view) {
    if (color.numel() == 0) return;
    if (!state.is_valid())
        throw std::invalid_argument(
            "bilateral grid requires an initialized state");
    if (!state.shared && view >= state.grids.shape()[0])
        throw std::invalid_argument("bilateral grid view index is out of range");
    const int row = state.shared ? 0 : static_cast<int>(view);
    if (color.shape().rank() != 3 || color.shape()[0] != 3)
        throw std::invalid_argument("bilateral grid expects planar [3,H,W]");
    ensure_same_shape(state.output, color);
    const int height = static_cast<int>(color.shape()[1]);
    const int width = static_cast<int>(color.shape()[2]);
    const int pixels = height * width;
    if (pixels == 0) return;
    bilateral_forward_kernel<<<
        (pixels + k_cuda_threads - 1) / k_cuda_threads, k_cuda_threads>>>(
        color.ptr<float>(), state.output.ptr<float>(), state.grids.ptr<float>(),
        row, state.luma, state.grid_height, state.grid_width,
        height, width);
    check_cuda(cudaGetLastError(), "apply bilateral grid colour correction");
}

void backward_bilateral_grid(
    BilateralGridState& state, const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient, const std::size_t view) {
    if (color.numel() == 0) return;
    ensure_same_shape(state.input_grad, color);
    const int height = static_cast<int>(color.shape()[1]);
    const int width = static_cast<int>(color.shape()[2]);
    const int pixels = height * width;
    if (pixels == 0) return;
    state.gradient.zero_();
    if (!state.shared && view >= state.grids.shape()[0])
        throw std::invalid_argument("bilateral grid view index is out of range");
    const int row = state.shared ? 0 : static_cast<int>(view);
    // The vectorized pass reads and reduces whole cells, which needs both
    // buffers on 16-byte boundaries. Everything the state owns comes from the
    // CUDA pool, so this is a guard against a future allocator change rather
    // than a path that is expected to run.
    const auto aligned = [](const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U;
    };
    const bool vectorized = aligned(state.grids.ptr<float>()) &&
        aligned(state.gradient.ptr<float>());
    // One thread walks four pixels. The grid reductions, not the arithmetic,
    // set the kernel time, and four independent reduction streams per thread
    // hide far more of their latency than one does.
    constexpr int k_pixels_per_thread = 4;
    const unsigned threads = k_cuda_threads;
    const unsigned total =
        (static_cast<unsigned>(pixels) + k_pixels_per_thread - 1) /
        k_pixels_per_thread;
    const unsigned blocks = (total + threads - 1) / threads;
    if (vectorized) {
        bilateral_backward_kernel<true, k_pixels_per_thread>
            <<<blocks, threads>>>(
                color.ptr<float>(), state.grids.ptr<float>(),
                output_gradient.ptr<float>(), state.gradient.ptr<float>(),
                state.input_grad.ptr<float>(), row, state.luma,
                state.grid_height, state.grid_width, height, width);
    } else {
        bilateral_backward_kernel<false, 1><<<
            (static_cast<unsigned>(pixels) + threads - 1) / threads,
            threads>>>(
                color.ptr<float>(), state.grids.ptr<float>(),
                output_gradient.ptr<float>(), state.gradient.ptr<float>(),
                state.input_grad.ptr<float>(), row, state.luma,
                state.grid_height, state.grid_width, height, width);
    }
    check_cuda(cudaGetLastError(), "backward bilateral grid colour correction");
}

void step_bilateral_grid(
    BilateralGridState& state, const TrainingOptions& options,
    const unsigned iteration) {
    if (!state.is_valid()) return;
    if (options.bilateral_grid_tv_weight > 0.F) {
        const long long cells = static_cast<long long>(state.grids.shape()[0]) *
            state.luma * state.grid_height * state.grid_width;
        if (cells > 0)
            bilateral_tv_backward_kernel<<<
                static_cast<unsigned>(
                    (cells + k_cuda_threads - 1) / k_cuda_threads),
                k_cuda_threads>>>(
                state.grids.ptr<float>(), state.gradient.ptr<float>(),
                static_cast<int>(state.grids.shape()[0]), state.luma,
                state.grid_height, state.grid_width,
                options.bilateral_grid_tv_weight);
        check_cuda(cudaGetLastError(), "bilateral grid total-variation");
    }
    adam_step(
        state.grids, state.gradient, state.adam, options.bilateral_grid_lr,
        iteration, options);
    // A grid row carries 12 coefficients per cell over a 16x16x8 grid. An
    // unconstrained row mean absorbs the global colour mapping: the Gaussians
    // lose their colour and the exported (canonical) model renders wrong.
    // Projecting the mean back onto the identity affine keeps the grid a
    // spatial residual and leaves global exposure / white balance to the
    // anchored PPISP model. The bound keeps a divergent row from producing
    // non-finite renders.
    const int rows = static_cast<int>(state.grids.shape()[0]);
    if (rows <= 0) return;
    bilateral_constrain_kernel<<<
        static_cast<unsigned>(rows * k_affine_channels), k_cuda_threads>>>(
        state.grids.ptr<float>(), rows, state.luma, state.grid_height,
        state.grid_width, options.bilateral_grid_identity_projection ? 1 : 0,
        options.bilateral_grid_deviation_limit);
    check_cuda(
        cudaGetLastError(), "constrain bilateral grid coefficients");
}

}  // namespace aetherscan::splat::detail
