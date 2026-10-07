#pragma once

#include "splat/rasterizer.hpp"

#include <span>

namespace splat_drender::vulkan {
struct SplatColorCorrectionCommand;
struct SplatDeviceMultiViewInput;
struct SplatDeviceMultiViewOutput;
}

namespace photara::splat::detail {

RenderResult vulkan_raster_forward(
    std::shared_ptr<void>& backend, const GaussianModel& model,
    const Camera& camera, const RasterizeOptions& options);

tinytensor::Tensor vulkan_pruning_scores(
    std::shared_ptr<void>& backend, const GaussianModel& model,
    const Camera& camera, RasterizeOptions options);

float vulkan_photometric_loss(
    const RenderResult& rendered, const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask, bool mask_enabled,
    float ssim_weight, float photometric_weight, bool read_loss_value,
    tinytensor::Tensor* color_gradient);

ModelGradients vulkan_raster_backward(
    const GaussianModel& model, const RenderResult& rendered,
    const tinytensor::Tensor& grad_color,
    const tinytensor::Tensor& grad_alpha,
    const tinytensor::Tensor& grad_depth,
    const tinytensor::Tensor& grad_normal,
    const tinytensor::Tensor& densify_map,
    const SHAdamUpdate* sh_adam,
    const StructureAdamUpdate* structure_adam = nullptr);

void vulkan_materialize_visibility(RenderResult& rendered);

DepthSampleResult vulkan_sample_depth(
    std::shared_ptr<void>& backend, const GaussianModel& model,
    const tinytensor::Tensor& points, const Camera& camera,
    const RasterizeOptions& options);
DepthSampleGradients vulkan_sample_depth_backward(
    const GaussianModel& model, const DepthSampleResult& sampled,
    const tinytensor::Tensor& gradient);
void vulkan_multi_view_loss(
    const RenderResult& rendered,
    const splat_drender::vulkan::SplatDeviceMultiViewInput& input,
    const splat_drender::vulkan::SplatDeviceMultiViewOutput& output);

// Reuse the rasterizer that owns the current training frame so colour
// correction does not construct a duplicate Vulkan pipeline set.
bool vulkan_dispatch_color_correction(
    std::span<const splat_drender::vulkan::SplatColorCorrectionCommand> commands);

}  // namespace photara::splat::detail
