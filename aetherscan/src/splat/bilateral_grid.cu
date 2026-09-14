#include "bilateral_grid.hpp"
#include "cuda_common.hpp"

#include <algorithm>
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

// One block per (view, coefficient). The block reduces that coefficient over
// every cell of the view and subtracts its deviation from identity, which is
// the projection onto the affine-with-identity-mean subspace.
__global__ void bilateral_mean_projection_kernel(
    float* grids, const int views, const int luma, const int grid_h,
    const int grid_w) {
    const int channel = blockIdx.x % k_affine_channels;
    const int view = blockIdx.x / k_affine_channels;
    if (view >= views) return;
    const long long cells = static_cast<long long>(luma) * grid_h * grid_w;
    if (cells == 0) return;
    float* base = grids +
        static_cast<long long>(view) * cells * k_affine_channels;
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
    const float offset = scratch[0] - k_identity_affine[channel];
    if (offset == 0.F) return;
    for (long long cell = threadIdx.x; cell < cells; cell += blockDim.x)
        base[cell * k_affine_channels + channel] -= offset;
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

__global__ void bilateral_backward_kernel(
    const float* color, const float* grids, const float* output_grad,
    float* grid_grad, float* input_grad, const int view, const int luma,
    const int grid_h, const int grid_w, const int height, const int width) {
    const int pixel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int pixels = height * width;
    if (pixel >= pixels) return;
    const int x = pixel % width;
    const int y = pixel / width;
    const float sr = color[pixel];
    const float sg = color[pixels + pixel];
    const float sb = color[2 * pixels + pixel];
    const float dr = output_grad[pixel];
    const float dg = output_grad[pixels + pixel];
    const float db = output_grad[2 * pixels + pixel];
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
    const float f000 = (1.F - fx) * (1.F - fy) * (1.F - fz);
    const float f001 = fx * (1.F - fy) * (1.F - fz);
    const float f010 = (1.F - fx) * fy * (1.F - fz);
    const float f011 = fx * fy * (1.F - fz);
    const float f100 = (1.F - fx) * (1.F - fy) * fz;
    const float f101 = fx * (1.F - fy) * fz;
    const float f110 = (1.F - fx) * fy * fz;
    const float f111 = fx * fy * fz;
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
    float vr = 0.F;
    float vg = 0.F;
    float vb = 0.F;
#pragma unroll
    for (int ci = 0; ci < k_affine_channels; ++ci) {
        const int si = ci % 4;
        const int di = ci / 4;
        const float r_coeff = si == 0 ? sr : si == 1 ? sg : si == 2 ? sb : 1.F;
        const float gout = di == 0 ? dr : di == 1 ? dg : db;
        const float grad_weight = r_coeff * gout;
        atomicAdd(grid_grad + off000 + ci, f000 * grad_weight);
        atomicAdd(grid_grad + off001 + ci, f001 * grad_weight);
        atomicAdd(grid_grad + off010 + ci, f010 * grad_weight);
        atomicAdd(grid_grad + off011 + ci, f011 * grad_weight);
        atomicAdd(grid_grad + off100 + ci, f100 * grad_weight);
        atomicAdd(grid_grad + off101 + ci, f101 * grad_weight);
        atomicAdd(grid_grad + off110 + ci, f110 * grad_weight);
        atomicAdd(grid_grad + off111 + ci, f111 * grad_weight);
        if (si < 3) {
            const float val =
                (((grids[off000 + ci] * (1.F - fx) +
                      grids[off001 + ci] * fx) *
                         (1.F - fy) +
                     (grids[off010 + ci] * (1.F - fx) +
                         grids[off011 + ci] * fx) *
                         fy) *
                        (1.F - fz) +
                    ((grids[off100 + ci] * (1.F - fx) +
                         grids[off101 + ci] * fx) *
                            (1.F - fy) +
                        (grids[off110 + ci] * (1.F - fx) +
                            grids[off111 + ci] * fx) *
                            fy) *
                        fz);
            (si == 0 ? vr : si == 1 ? vg : vb) += val * gout;
        }
    }
    const long long corner_offs[8] = {
        off000, off001, off010, off011, off100, off101, off110, off111};
    const float dwdz[8] = {
        -(1.F - fx) * (1.F - fy), -fx * (1.F - fy), -(1.F - fx) * fy, -fx * fy,
        (1.F - fx) * (1.F - fy), fx * (1.F - fy), (1.F - fx) * fy, fx * fy};
    float gz_grad = 0.F;
#pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        float trilerp = 0.F;
#pragma unroll
        for (int ci = 0; ci < k_affine_channels; ++ci) {
            const float v = grids[corner_offs[corner] + ci];
            const int si = ci % 4;
            const int di = ci / 4;
            const float r_coeff =
                si == 0 ? sr : si == 1 ? sg : si == 2 ? sb : 1.F;
            const float gout = di == 0 ? dr : di == 1 ? dg : db;
            trilerp += v * r_coeff * gout;
        }
        gz_grad += dwdz[corner] * static_cast<float>(luma - 1) * trilerp;
    }
    if (!gz_in_range) gz_grad = 0.F;
    input_grad[pixel] = vr + k_c2g_r * gz_grad;
    input_grad[pixels + pixel] = vg + k_c2g_g * gz_grad;
    input_grad[2 * pixels + pixel] = vb + k_c2g_b * gz_grad;
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
    state.grids = tinytensor::Tensor::empty(
        {views, static_cast<std::size_t>(state.luma),
            static_cast<std::size_t>(state.grid_height),
            static_cast<std::size_t>(state.grid_width),
            static_cast<std::size_t>(k_affine_channels)},
        tinytensor::Device::CUDA);
    const long long cells = static_cast<long long>(views) * state.luma *
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
    if (view >= state.grids.shape()[0])
        throw std::invalid_argument("bilateral grid view index is out of range");
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
        static_cast<int>(view), state.luma, state.grid_height, state.grid_width,
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
    bilateral_backward_kernel<<<
        (pixels + k_cuda_threads - 1) / k_cuda_threads, k_cuda_threads>>>(
        color.ptr<float>(), state.grids.ptr<float>(),
        output_gradient.ptr<float>(), state.gradient.ptr<float>(),
        state.input_grad.ptr<float>(), static_cast<int>(view), state.luma,
        state.grid_height, state.grid_width, height, width);
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
    if (!options.bilateral_grid_identity_projection) return;
    // One view's grid carries 12 coefficients per cell over a 16x16x8 grid,
    // more free parameters than the view has pixels. An unconstrained per-view
    // mean would absorb the global colour mapping: the Gaussians lose their
    // colour and the exported (canonical) model renders wrong. Projecting the
    // mean back onto the identity affine keeps the grid a spatial residual and
    // leaves global exposure / white balance to the anchored PPISP model.
    const int views = static_cast<int>(state.grids.shape()[0]);
    if (views <= 0) return;
    bilateral_mean_projection_kernel<<<
        static_cast<unsigned>(views * k_affine_channels), k_cuda_threads>>>(
        state.grids.ptr<float>(), views, state.luma, state.grid_height,
        state.grid_width);
    check_cuda(
        cudaGetLastError(), "project bilateral grid mean onto identity");
}

}  // namespace aetherscan::splat::detail
