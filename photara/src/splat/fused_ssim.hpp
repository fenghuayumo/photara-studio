#pragma once

#include "internal/tensor_impl.hpp"

#include <cstdint>

namespace photara::splat::detail {

// TinyTensor wrapper around the complete 11x11 fused L1+SSIM CUDA
// forward/backward implementation used by pygsplat. Images are CHW float32.
void fused_l1_ssim_loss(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    bool mask_enabled,
    float ssim_weight,
    float photometric_weight,
    tinytensor::Tensor& gradient,
    float* scalar_terms,
    std::uint32_t width,
    std::uint32_t height);

// Mean 11x11 Gaussian SSIM over valid interior pixels (and the mask, if
// enabled). Matches the training fused-SSIM window. Returns 0 if the image
// is too small for a valid window.
float fused_ssim_metric(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    bool mask_enabled,
    std::uint32_t width,
    std::uint32_t height);

}  // namespace photara::splat::detail
