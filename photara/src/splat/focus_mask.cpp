#include "focus_mask.hpp"

#include "core/camera_projection.hpp"
#include "focus_mask_geometry.hpp"

#include <stdexcept>

#include <vector>

namespace photara::splat::detail {
tinytensor::Tensor focus_view_mask_cpu(
    const FocusMaskFrame& frame,
    const tinytensor::Device device) {
    const std::size_t pixels = static_cast<std::size_t>(frame.width) * frame.height;
    std::vector<float> mask(pixels, 0.F);
    for (unsigned y = 0; y < frame.height; ++y)
        for (unsigned x = 0; x < frame.width; ++x)
            mask[static_cast<std::size_t>(y) * frame.width + x] = focus_mask_value(frame, x, y);
    return tinytensor::Tensor::from_vector(mask, {frame.height, frame.width}, device);
}

#if !defined(PHOTARA_HAS_CUDA_SPLAT)
tinytensor::Tensor focus_view_mask_cuda(const FocusMaskFrame&) {
    throw std::runtime_error("CUDA focus mask backend is disabled");
}
#endif
} // namespace photara::splat::detail
