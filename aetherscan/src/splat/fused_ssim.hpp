#pragma once

#include "internal/tensor_impl.hpp"

#include <cstdint>

namespace aetherscan::splat::detail {

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

}  // namespace aetherscan::splat::detail
