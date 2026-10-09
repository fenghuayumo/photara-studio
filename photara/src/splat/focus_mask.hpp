#pragma once

#include "internal/tensor_impl.hpp"

#include <cstdint>

namespace photara::splat::detail {

// Row-major camera_to_box matches the host Eigen product
// bounds.axes.transpose() * camera_to_world.
struct FocusMaskFrame {
    unsigned width = 0;
    unsigned height = 0;
    unsigned model = 0;
    float fx = 1.F;
    float fy = 1.F;
    float cx = 0.F;
    float cy = 0.F;
    float k1 = 0.F;
    float k2 = 0.F;
    float k3 = 0.F;
    float k4 = 0.F;
    float origin_x = 0.F;
    float origin_y = 0.F;
    float origin_z = 0.F;
    float camera_to_box[9] = {};
    float half_x = 0.F;
    float half_y = 0.F;
    float half_z = 0.F;
};

tinytensor::Tensor focus_view_mask_cuda(const FocusMaskFrame& frame);
tinytensor::Tensor focus_view_mask_cpu(
    const FocusMaskFrame& frame, tinytensor::Device device);
#if defined(TINYTENSOR_HAS_VULKAN)
tinytensor::Tensor focus_view_mask_vulkan(const FocusMaskFrame& frame);
#endif

}  // namespace photara::splat::detail
