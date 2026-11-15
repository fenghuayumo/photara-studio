#pragma once
#include "ops.hpp"
#include "splat/rasterizer.hpp"

namespace photara::splat::detail {
// The shared trainer owns scheduling; this boundary owns backend execution.
LossGradients compute_step_loss(
    const Rasterizer& rasterizer,
    const RenderResult& rendered,
    const RenderResult& loss_render,
    const TrainingView& target,
    const TrainingView& geometry_target,
    const TrainingOptions& options,
    bool report_progress,
    bool depth_normal_active,
    bool allocate_geometry_gradients,
    bool ai_prior_active,
    bool owned_color_gradient);
tinytensor::Tensor normal_features_from_smallest_axis(const GaussianModel& model);
struct ShAdamQuant;
void configure_training_raster(
    RasterizeOptions& raster,
    const TrainingOptions& options,
    const TrainingView& target,
    bool owned_color);
RenderResult render_normal_field(
    const Rasterizer& rasterizer,
    const GaussianModel& model,
    RenderResult& primary,
    const Camera& camera,
    RasterizeOptions options);
tinytensor::Tensor reference_color_gradient(
    const Rasterizer& rasterizer,
    const RenderResult& rendered,
    const RenderResult& teacher,
    const TrainingView& target,
    const TrainingOptions& options);
void step_training_sh(
    GaussianModel& model,
    ModelGradients& gradients,
    ShAdamQuant& state,
    unsigned active_degree,
    const TrainingOptions& options,
    unsigned iteration);
} // namespace photara::splat::detail
