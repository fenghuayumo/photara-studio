#pragma once

#include "splat/rasterizer.hpp"

namespace photara::splat::detail {

RenderResult vulkan_raster_forward(
    std::shared_ptr<void>& backend, const GaussianModel& model,
    const Camera& camera, const RasterizeOptions& options);

float vulkan_photometric_loss(
    const RenderResult& rendered, const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask, bool mask_enabled,
    float ssim_weight, float photometric_weight, bool read_loss_value);

ModelGradients vulkan_raster_backward(
    const GaussianModel& model, const RenderResult& rendered,
    const tinytensor::Tensor& grad_color,
    const tinytensor::Tensor& grad_alpha,
    const tinytensor::Tensor& grad_depth,
    const tinytensor::Tensor& grad_normal,
    const tinytensor::Tensor& densify_map,
    const SHAdamUpdate* sh_adam);

void vulkan_materialize_visibility(RenderResult& rendered);

}  // namespace photara::splat::detail
