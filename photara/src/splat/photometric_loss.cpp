#include "photometric_loss.hpp"

#include <stdexcept>

namespace photara::splat::detail {

void photometric_loss(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    const bool mask_enabled,
    tinytensor::Tensor& gradient,
    float* scalar_terms,
    const std::uint32_t width,
    const std::uint32_t height,
    const PhotometricLossOptions& options) {
    fused_l1_ssim_loss(
        prediction, target, mask, mask_enabled,
        options.ssim_weight, options.photometric_weight,
        gradient, scalar_terms, width, height);
}

splat_drender::vulkan::SplatDevicePhotometricOutput photometric_loss(
    splat_drender::vulkan::SplatRasterizer& rasterizer,
    const splat_drender::vulkan::SplatDeviceFrame& prediction,
    const splat_drender::vulkan::SplatBufferView& target,
    const PhotometricLossOptions& options,
    const splat_drender::vulkan::SplatBufferView& mask) {
    if (prediction.width == 0 || prediction.height == 0 ||
        prediction.color.buffer == VK_NULL_HANDLE)
        throw std::invalid_argument("Vulkan photometric loss requires a device render");
    return rasterizer.fused_l1_ssim_device(
        prediction.color, target, prediction.width, prediction.height,
        options.ssim_weight, options.photometric_weight, mask);
}

float photometric_loss_scalar(
    splat_drender::vulkan::SplatRasterizer& rasterizer,
    const splat_drender::vulkan::SplatDevicePhotometricOutput& output) {
    return rasterizer.read_photometric_loss(output);
}

}  // namespace photara::splat::detail
