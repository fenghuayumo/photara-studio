// Ported from fused-ssim/ssim.cu (MIT, Copyright (c) 2024 Rahul Goel).
// The CUDA math and 11x11 Gaussian separable forward/backward kernels are kept
// intact; only the PyTorch allocation/dispatch wrapper is replaced by
// TinyTensor tensors and raw CUDA launch code.

#include "fused_ssim.hpp"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <stdexcept>
#include <string>
#include <cstdlib>

namespace cg = cooperative_groups;

namespace photara::splat::detail {
namespace {

__constant__ float c_gauss[11] = {
    0.001028380123898387F,
    0.0075987582094967365F,
    0.036000773310661316F,
    0.10936068743467331F,
    0.21300552785396576F,
    0.26601171493530273F,
    0.21300552785396576F,
    0.10936068743467331F,
    0.036000773310661316F,
    0.0075987582094967365F,
    0.001028380123898387F};

constexpr int k_block_x = 16;
constexpr int k_block_y = 16;
constexpr int k_halo = 5;
constexpr int k_halo2 = 10;
constexpr int k_shared_x = k_block_x + 2 * k_halo;
constexpr int k_shared_y = k_block_y + 2 * k_halo;
constexpr int k_conv_x = k_block_x;
constexpr int k_conv_y = k_shared_y;

void check_cuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
}

// Experimental opt-in: capture this allocation-free four-kernel suffix and
// update addresses/scalars each invocation. Launch on the caller's default
// stream so target uploads and loss initialization retain their ordering.
struct SsimGraph {
    cudaStream_t capture_stream{};
    cudaGraphExec_t executable{};
    SsimGraph() { check_cuda(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking), "create SSIM capture stream"); }
    ~SsimGraph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (capture_stream) cudaStreamDestroy(capture_stream);
    }
    void launch() {
        cudaGraph_t graph{};
        check_cuda(cudaStreamEndCapture(capture_stream, &graph), "end SSIM capture");
        if (executable) {
            cudaGraphExecUpdateResultInfo info{};
            const auto status = cudaGraphExecUpdate(executable, graph, &info);
            if (status == cudaErrorGraphExecUpdateFailure) {
                cudaGetLastError();
                cudaGraphExecDestroy(executable);
                executable = nullptr;
            } else if (status != cudaSuccess) {
                cudaGraphDestroy(graph);
                check_cuda(status, "update SSIM graph");
            }
        }
        if (!executable) {
            const auto status = cudaGraphInstantiate(&executable, graph, 0);
            cudaGraphDestroy(graph);
            check_cuda(status, "instantiate SSIM graph");
        } else cudaGraphDestroy(graph);
        check_cuda(cudaGraphLaunch(executable, nullptr), "launch SSIM graph");
    }
};

__device__ __forceinline__ float get_pixel(
    const float* image, const int batch, const int channel,
    const int y, const int x, const int channels, const int height,
    const int width) {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0.F;
    return image[batch * channels * height * width +
                 channel * height * width + y * width + x];
}

__global__ void apply_mask_kernel(
    const float* prediction, const float* target, const float* mask,
    float* masked_prediction, float* masked_target,
    const int channels, const int height, const int width,
    const bool mask_enabled) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t pixels = static_cast<std::size_t>(height) * width;
    const std::size_t count = static_cast<std::size_t>(channels) * pixels;
    if (index >= count) return;
    const float valid = mask_enabled ? mask[index % pixels] : 1.F;
    masked_prediction[index] = prediction[index] * valid;
    masked_target[index] = target[index] * valid;
}

// This is fusedl1ssim_lossCUDA from fused-ssim. It computes the complete
// Gaussian-windowed 11x11 SSIM statistics, the fused L1+SSIM map, and the
// partial derivatives required by the backward kernel.
__global__ void fused_l1_ssim_forward_kernel(
    const float ssim_weight, const int height, const int width,
    const int channels, const float c1, const float c2,
    const float* __restrict__ image1,
    const float* __restrict__ image2,
    float* __restrict__ loss_map,
    float* __restrict__ dm_dmu1,
    float* __restrict__ dm_dsigma1_sq,
    float* __restrict__ dm_dsigma12,
    float* __restrict__ scalar_terms,
    const float normalization) {
    auto block = cg::this_thread_block();
    const int batch = block.group_index().z;
    const int pixel_y = block.group_index().y * k_block_y + block.thread_index().y;
    const int pixel_x = block.group_index().x * k_block_x + block.thread_index().x;
    const int pixel_id = pixel_y * width + pixel_x;
    const int pixels = height * width;

    __shared__ float tile[k_shared_y][k_shared_x][2];
    __shared__ float horizontal[k_conv_y][k_conv_x][5];

    for (int channel = 0; channel < channels; ++channel) {
        const int tile_size = k_shared_y * k_shared_x;
        const int threads = k_block_x * k_block_y;
        const int steps = (tile_size + threads - 1) / threads;
        const int tile_start_y = block.group_index().y * k_block_y;
        const int tile_start_x = block.group_index().x * k_block_x;
        for (int step = 0; step < steps; ++step) {
            const int id = step * threads + block.thread_rank();
            if (id < tile_size) {
                const int local_y = id / k_shared_x;
                const int local_x = id % k_shared_x;
                const int y = tile_start_y + local_y - k_halo;
                const int x = tile_start_x + local_x - k_halo;
                tile[local_y][local_x][0] = get_pixel(
                    image1, batch, channel, y, x, channels, height, width);
                tile[local_y][local_x][1] = get_pixel(
                    image2, batch, channel, y, x, channels, height, width);
            }
        }
        block.sync();

        const float l1 = fabsf(
            tile[block.thread_index().y + k_halo]
                [block.thread_index().x + k_halo][0] -
            tile[block.thread_index().y + k_halo]
                [block.thread_index().x + k_halo][1]);

        {
            const int local_y = threadIdx.y;
            const int local_x = threadIdx.x + k_halo;
            float sum_x = 0.F, sum_x2 = 0.F, sum_y = 0.F;
            float sum_y2 = 0.F, sum_xy = 0.F;
#pragma unroll
            for (int d = 0; d <= k_halo2; ++d) {
                const float weight = c_gauss[d];
                const float x = tile[local_y][local_x + d - k_halo][0];
                const float y = tile[local_y][local_x + d - k_halo][1];
                sum_x += x * weight;
                sum_x2 += x * x * weight;
                sum_y += y * weight;
                sum_y2 += y * y * weight;
                sum_xy += x * y * weight;
            }
            horizontal[local_y][threadIdx.x][0] = sum_x;
            horizontal[local_y][threadIdx.x][1] = sum_x2;
            horizontal[local_y][threadIdx.x][2] = sum_y;
            horizontal[local_y][threadIdx.x][3] = sum_y2;
            horizontal[local_y][threadIdx.x][4] = sum_xy;

            const int second_y = local_y + k_block_y;
            if (second_y < k_conv_y) {
                sum_x = sum_x2 = sum_y = sum_y2 = sum_xy = 0.F;
#pragma unroll
                for (int d = 0; d <= k_halo2; ++d) {
                    const float weight = c_gauss[d];
                    const float x = tile[second_y][local_x + d - k_halo][0];
                    const float y = tile[second_y][local_x + d - k_halo][1];
                    sum_x += x * weight;
                    sum_x2 += x * x * weight;
                    sum_y += y * weight;
                    sum_y2 += y * y * weight;
                    sum_xy += x * y * weight;
                }
                horizontal[second_y][threadIdx.x][0] = sum_x;
                horizontal[second_y][threadIdx.x][1] = sum_x2;
                horizontal[second_y][threadIdx.x][2] = sum_y;
                horizontal[second_y][threadIdx.x][3] = sum_y2;
                horizontal[second_y][threadIdx.x][4] = sum_xy;
            }
        }
        block.sync();

        {
            const int local_y = threadIdx.y + k_halo;
            const int local_x = threadIdx.x;
            float out0 = 0.F, out1 = 0.F, out2 = 0.F, out3 = 0.F, out4 = 0.F;
#pragma unroll
            for (int d = 0; d <= k_halo2; ++d) {
                const float weight = c_gauss[d];
                const float* value = horizontal[local_y + d - k_halo][local_x];
                out0 += value[0] * weight;
                out1 += value[1] * weight;
                out2 += value[2] * weight;
                out3 += value[3] * weight;
                out4 += value[4] * weight;
            }
            if (pixel_x < width && pixel_y < height) {
                const float mu1 = out0;
                const float mu2 = out2;
                const float mu1_sq = mu1 * mu1;
                const float mu2_sq = mu2 * mu2;
                const float sigma1_sq = out1 - mu1_sq;
                const float sigma2_sq = out3 - mu2_sq;
                const float sigma12 = out4 - mu1 * mu2;
                const float a = mu1_sq + mu2_sq + c1;
                const float b = sigma1_sq + sigma2_sq + c2;
                const float c = 2.F * mu1 * mu2 + c1;
                const float d = 2.F * sigma12 + c2;
                const float ssim = c * d / (a * b);
                const int index = batch * channels * pixels + channel * pixels + pixel_id;
                const float loss_value =
                    ssim_weight * (1.F - ssim) + (1.F - ssim_weight) * l1;
                loss_map[index] = loss_value;
                // The valid-window reduction over the map used to be a fourth
                // kernel that re-read the whole image; the map is already here.
                if (scalar_terms != nullptr && pixel_x >= k_halo &&
                    pixel_x < width - k_halo && pixel_y >= k_halo &&
                    pixel_y < height - k_halo)
                    atomicAdd(scalar_terms, normalization * loss_value);
                dm_dmu1[index] =
                    (2.F * mu2 * d) / (a * b) -
                    (2.F * mu2 * c) / (a * b) -
                    (2.F * mu1 * c * d) / (a * a * b) +
                    (2.F * mu1 * c * d) / (a * b * b);
                dm_dsigma1_sq[index] = -c * d / (a * b * b);
                dm_dsigma12[index] = 2.F * c / (a * b);
            }
        }
        block.sync();
    }
}

// This is fusedl1ssim_loss_backwardCUDA from fused-ssim. The valid-window
// reduction is represented by a zero dL/dmap halo, exactly like the Python
// autograd wrapper's padding="valid" path.
__global__ void fused_l1_ssim_backward_kernel(
    const float ssim_weight, const int height, const int width,
    const int channels, const float* __restrict__ image1,
    const float* __restrict__ image2,
    const float normalization,
    float* __restrict__ dl_dimage1,
    const float* __restrict__ dm_dmu1,
    const float* __restrict__ dm_dsigma1_sq,
    const float* __restrict__ dm_dsigma12,
    const float* __restrict__ mask,
    const bool mask_enabled) {
    auto block = cg::this_thread_block();
    const int pixel_y = block.group_index().y * k_block_y + block.thread_index().y;
    const int pixel_x = block.group_index().x * k_block_x + block.thread_index().x;
    const int pixel_id = pixel_y * width + pixel_x;
    const int pixels = height * width;
    const int batch = block.group_index().z;

    __shared__ float data[k_shared_y][k_shared_x][3];
    __shared__ float scratch[k_conv_y][k_conv_x][3];

    for (int channel = 0; channel < channels; ++channel) {
        float pixel1 = 0.F, pixel2 = 0.F;
        if (pixel_x < width && pixel_y < height) {
            pixel1 = get_pixel(
                image1, batch, channel, pixel_y, pixel_x,
                channels, height, width);
            pixel2 = get_pixel(
                image2, batch, channel, pixel_y, pixel_x,
                channels, height, width);
        }
        const int start_y = block.group_index().y * k_block_y;
        const int start_x = block.group_index().x * k_block_x;
        const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        const int warp = thread_id / 32;
        const int lane = thread_id % 32;
        const int warps = (k_block_x * k_block_y + 31) / 32;
        for (int row = warp; row < k_shared_y; row += warps) {
            const int y = start_y + row - k_halo;
            for (int column = lane; column < k_shared_x; column += 32) {
                const int x = start_x + column - k_halo;
                const float chain = x >= k_halo && x < width - k_halo &&
                    y >= k_halo && y < height - k_halo ? normalization : 0.F;
                data[row][column][0] = -ssim_weight * get_pixel(
                    dm_dmu1, batch, channel, y, x, channels, height, width) * chain;
                data[row][column][1] = -ssim_weight * get_pixel(
                    dm_dsigma1_sq, batch, channel, y, x,
                    channels, height, width) * chain;
                data[row][column][2] = -ssim_weight * get_pixel(
                    dm_dsigma12, batch, channel, y, x,
                    channels, height, width) * chain;
            }
        }
        block.sync();

        const int local_y = threadIdx.y;
        const int local_x = threadIdx.x + k_halo;
        for (int pass = 0; pass < 2; ++pass) {
            const int y = local_y + pass * k_block_y;
            if (y < k_conv_y) {
                float sum0 = 0.F, sum1 = 0.F, sum2 = 0.F;
#pragma unroll
                for (int d = 0; d <= k_halo2; ++d) {
                    const float weight = c_gauss[d];
                    sum0 += data[y][local_x + d - k_halo][0] * weight;
                    sum1 += data[y][local_x + d - k_halo][1] * weight;
                    sum2 += data[y][local_x + d - k_halo][2] * weight;
                }
                scratch[y][threadIdx.x][0] = sum0;
                scratch[y][threadIdx.x][1] = sum1;
                scratch[y][threadIdx.x][2] = sum2;
            }
        }
        block.sync();

        if (pixel_x < width && pixel_y < height) {
            const int y = threadIdx.y + k_halo;
            const int x = threadIdx.x;
            float sum0 = 0.F, sum1 = 0.F, sum2 = 0.F;
#pragma unroll
            for (int d = 0; d <= k_halo2; ++d) {
                const float weight = c_gauss[d];
                sum0 += scratch[y + d - k_halo][x][0] * weight;
                sum1 += scratch[y + d - k_halo][x][1] * weight;
                sum2 += scratch[y + d - k_halo][x][2] * weight;
            }
            const int index = batch * channels * pixels + channel * pixels + pixel_id;
            const float chain = pixel_x >= k_halo && pixel_x < width - k_halo &&
                pixel_y >= k_halo && pixel_y < height - k_halo ? normalization : 0.F;
            const float sign = pixel1 == pixel2
                ? 0.F
                : copysignf(1.F, pixel1 - pixel2);
            // Outside the window the filtered terms are already zero, so the
            // mask multiply is the only thing the chained pass used to add.
            const float valid = mask_enabled ? mask[pixel_id] : 1.F;
            dl_dimage1[index] +=
                (sum0 + 2.F * pixel1 * sum1 + pixel2 * sum2 +
                 (1.F - ssim_weight) * sign * chain) *
                valid;
        }
        block.sync();
    }
}

}  // namespace

void fused_l1_ssim_loss(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    const bool mask_enabled,
    const float ssim_weight,
    const float photometric_weight,
    tinytensor::Tensor& gradient,
    float* scalar_terms,
    const std::uint32_t width,
    const std::uint32_t height) {
    if (width <= k_halo2 || height <= k_halo2)
        throw std::invalid_argument("fused SSIM requires width and height > 10");
    constexpr int channels = 3;
    const std::size_t count = static_cast<std::size_t>(channels) * width * height;
    const auto shape = prediction.shape();
    auto loss_map = tinytensor::Tensor::empty(shape, tinytensor::Device::CUDA);
    auto dm_dmu1 = tinytensor::Tensor::empty(shape, tinytensor::Device::CUDA);
    auto dm_dsigma1_sq = tinytensor::Tensor::empty(shape, tinytensor::Device::CUDA);
    auto dm_dsigma12 = tinytensor::Tensor::empty(shape, tinytensor::Device::CUDA);
    auto masked_prediction = tinytensor::Tensor::empty(shape, tinytensor::Device::CUDA);
    auto masked_target = tinytensor::Tensor::empty(shape, tinytensor::Device::CUDA);

    const char* graph_env = std::getenv("PHOTARA_SPLAT_SSIM_GRAPH");
    const bool use_graph = graph_env && graph_env[0] == '1';
    SsimGraph* graph = nullptr;
    cudaStream_t stream = nullptr;
    if (use_graph) {
        static thread_local SsimGraph state;
        graph = &state;
        stream = state.capture_stream;
        check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin SSIM capture");
    }

    // Without a mask the two image buffers would only ever hold a copy of the
    // prediction and the target, so the copy is skipped entirely.
    constexpr int threads = 256;
    const float* image1 = prediction.ptr<float>();
    const float* image2 = target.ptr<float>();
    if (mask_enabled) {
        apply_mask_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(
            prediction.ptr<float>(), target.ptr<float>(), mask.ptr<float>(),
            masked_prediction.ptr<float>(), masked_target.ptr<float>(),
            channels, static_cast<int>(height), static_cast<int>(width), true);
        image1 = masked_prediction.ptr<float>();
        image2 = masked_target.ptr<float>();
    }

    const dim3 block(k_block_x, k_block_y);
    const dim3 grid(
        (width + k_block_x - 1) / k_block_x,
        (height + k_block_y - 1) / k_block_y, 1);
    constexpr float c1 = 0.01F * 0.01F;
    constexpr float c2 = 0.03F * 0.03F;
    const float normalization = photometric_weight /
        static_cast<float>(channels * (width - k_halo2) * (height - k_halo2));
    fused_l1_ssim_forward_kernel<<<grid, block, 0, stream>>>(
        ssim_weight, static_cast<int>(height), static_cast<int>(width),
        channels, c1, c2, image1, image2, loss_map.ptr<float>(),
        dm_dmu1.ptr<float>(), dm_dsigma1_sq.ptr<float>(),
        dm_dsigma12.ptr<float>(), scalar_terms, normalization);

    // The valid-crop chain is a constant inside the crop and zero outside, and
    // the mask multiply folds into the same pass: the backward writes the
    // gradient the photometric loss consumes directly.
    fused_l1_ssim_backward_kernel<<<grid, block, 0, stream>>>(
        ssim_weight, static_cast<int>(height), static_cast<int>(width), channels,
        image1, image2, normalization, gradient.ptr<float>(),
        dm_dmu1.ptr<float>(), dm_dsigma1_sq.ptr<float>(),
        dm_dsigma12.ptr<float>(), mask.ptr<float>(), mask_enabled);
    check_cuda(cudaGetLastError(), "run fused L1+SSIM CUDA kernels");
    if (graph) graph->launch();
}

__global__ void reduce_ssim_metric_kernel(
    const float* loss_map, const float* mask, float* sum, float* count,
    const int channels, const int height, const int width,
    const bool mask_enabled) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t pixels = static_cast<std::size_t>(height) * width;
    const std::size_t n = static_cast<std::size_t>(channels) * pixels;
    if (index >= n) return;
    const int pixel = static_cast<int>(index % pixels);
    const int y = pixel / width;
    const int x = pixel % width;
    if (x < k_halo || x >= width - k_halo ||
        y < k_halo || y >= height - k_halo)
        return;
    if (mask_enabled && mask[pixel] <= 0.F) return;
    atomicAdd(sum, 1.F - loss_map[index]);
    atomicAdd(count, 1.F);
}

float fused_ssim_metric(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    const bool mask_enabled,
    const std::uint32_t width,
    const std::uint32_t height) {
    if (width <= k_halo2 || height <= k_halo2) return 0.F;
    constexpr int channels = 3;
    const std::size_t count = static_cast<std::size_t>(channels) * width * height;
    const auto shape = prediction.shape();
    auto loss_map = tinytensor::Tensor::empty(shape, tinytensor::Device::CUDA);
    auto dm_dmu1 = tinytensor::Tensor::empty(shape, tinytensor::Device::CUDA);
    auto dm_dsigma1_sq = tinytensor::Tensor::empty(shape, tinytensor::Device::CUDA);
    auto dm_dsigma12 = tinytensor::Tensor::empty(shape, tinytensor::Device::CUDA);
    auto masked_prediction =
        tinytensor::Tensor::empty(shape, tinytensor::Device::CUDA);
    auto masked_target = tinytensor::Tensor::empty(shape, tinytensor::Device::CUDA);
    auto sum = tinytensor::Tensor::zeros({1}, tinytensor::Device::CUDA);
    auto count_t = tinytensor::Tensor::zeros({1}, tinytensor::Device::CUDA);

    constexpr int threads = 256;
    apply_mask_kernel<<<(count + threads - 1) / threads, threads>>>(
        prediction.ptr<float>(), target.ptr<float>(),
        mask.is_valid() ? mask.ptr<float>() : nullptr,
        masked_prediction.ptr<float>(), masked_target.ptr<float>(), channels,
        static_cast<int>(height), static_cast<int>(width),
        mask_enabled && mask.is_valid());
    const float* image1 = mask_enabled && mask.is_valid()
        ? masked_prediction.ptr<float>()
        : prediction.ptr<float>();
    const float* image2 = mask_enabled && mask.is_valid()
        ? masked_target.ptr<float>()
        : target.ptr<float>();

    const dim3 block(k_block_x, k_block_y);
    const dim3 grid(
        (width + k_block_x - 1) / k_block_x,
        (height + k_block_y - 1) / k_block_y, 1);
    constexpr float c1 = 0.01F * 0.01F;
    constexpr float c2 = 0.03F * 0.03F;
    fused_l1_ssim_forward_kernel<<<grid, block>>>(
        1.F, static_cast<int>(height), static_cast<int>(width),
        channels, c1, c2, image1, image2, loss_map.ptr<float>(),
        dm_dmu1.ptr<float>(), dm_dsigma1_sq.ptr<float>(),
        dm_dsigma12.ptr<float>(), nullptr, 0.F);
    reduce_ssim_metric_kernel<<<(count + threads - 1) / threads, threads>>>(
        loss_map.ptr<float>(),
        mask.is_valid() ? mask.ptr<float>() : nullptr,
        sum.ptr<float>(), count_t.ptr<float>(),
        channels, static_cast<int>(height), static_cast<int>(width),
        mask_enabled && mask.is_valid());
    check_cuda(cudaGetLastError(), "run fused SSIM metric");
    const float host_sum = sum.to_vector()[0];
    const float host_count = count_t.to_vector()[0];
    return host_count <= 0.F ? 0.F : host_sum / host_count;
}

}  // namespace photara::splat::detail
