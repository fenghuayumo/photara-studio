#include "cuda_backend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace photara::splat::detail {
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

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (prediction.device() == tinytensor::Device::CUDA)
        return cuda_backend::fused_l1_ssim_loss(
            prediction, target, mask, mask_enabled, ssim_weight, photometric_weight, gradient,
            scalar_terms, width, height);
#endif
    throw std::runtime_error("fused_l1_ssim_loss: this operation requires the CUDA backend");
}

float fused_ssim_metric(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    const bool mask_enabled,
    const std::uint32_t width,
    const std::uint32_t height) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (prediction.device() == tinytensor::Device::CUDA)
        return cuda_backend::fused_ssim_metric(
            prediction, target, mask, mask_enabled, width, height);
#endif
    throw std::runtime_error("fused_ssim_metric: this operation requires the CUDA backend");
}

} // namespace photara::splat::detail
