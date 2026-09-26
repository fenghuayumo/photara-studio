#pragma once

#include "fused_ssim.hpp"
#include "splat_drender/vulkan_api.h"

#include <cstdint>

namespace photara::splat::detail {

// One backend-neutral description of the training objective. CUDA and Vulkan
// overloads below implement identical valid-window 11x11 L1+SSIM math.
struct PhotometricLossOptions {
    float ssim_weight = 0.2F;
    float photometric_weight = 1.0F;
};

// CUDA/TinyTensor training entry point. Inputs and gradient stay on CUDA;
// scalar_terms is the optional device-side reduction storage used by logging.
void photometric_loss(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    bool mask_enabled,
    tinytensor::Tensor& gradient,
    float* scalar_terms,
    std::uint32_t width,
    std::uint32_t height,
    const PhotometricLossOptions& options = {});

// Vulkan training entry point. The render, target, mask and returned gradient
// are all VkBuffer slices on the same device. No image/gradient readback occurs.
[[nodiscard]] splat_drender::vulkan::SplatDevicePhotometricOutput
photometric_loss(
    splat_drender::vulkan::SplatRasterizer& rasterizer,
    const splat_drender::vulkan::SplatDeviceFrame& prediction,
    const splat_drender::vulkan::SplatBufferView& target,
    const PhotometricLossOptions& options = {},
    const splat_drender::vulkan::SplatBufferView& mask = {});

// Vulkan logging path: reads one already-reduced float, never the image-sized
// loss map or gradient.
[[nodiscard]] float photometric_loss_scalar(
    splat_drender::vulkan::SplatRasterizer& rasterizer,
    const splat_drender::vulkan::SplatDevicePhotometricOutput& output);

}  // namespace photara::splat::detail
