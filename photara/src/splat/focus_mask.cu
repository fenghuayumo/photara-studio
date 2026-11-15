#include "focus_mask_geometry.hpp"
#include "focus_mask.hpp"
#include "cuda_common.hpp"
#include "core/camera_projection.hpp"
#include "internal/cuda_stream_context.hpp"

#include <cuda_runtime.h>

#include <vector>

namespace photara::splat::detail {
namespace {

__global__ void focus_view_mask_kernel(const FocusMaskFrame frame, float* mask) {
    const unsigned pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned pixels = frame.width * frame.height;
    if (pixel >= pixels)
        return;
    const unsigned x = pixel % frame.width;
    const unsigned y = pixel / frame.width;
    mask[pixel] = focus_mask_value(frame, x, y);
}

} // namespace

tinytensor::Tensor focus_view_mask_cuda(const FocusMaskFrame& frame) {
    const std::size_t pixels = static_cast<std::size_t>(frame.width) * frame.height;
    auto mask = tinytensor::Tensor::empty({frame.height, frame.width}, tinytensor::Device::CUDA);
    if (pixels == 0)
        return mask;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((pixels + threads - 1) / threads);
    focus_view_mask_kernel<<<blocks, threads, 0, tinytensor::getCurrentCUDAStream()>>>(
        frame, mask.ptr<float>());
    check_cuda(cudaGetLastError(), "focus_view_mask");
    return mask;
}

} // namespace photara::splat::detail
