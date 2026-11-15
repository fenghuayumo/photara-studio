#pragma once

// Internal Vulkan implementation contract, expressed in backend-neutral C++ types.
#include "ops.hpp"
#include "pam_ops.hpp"
#include "sh_adam_quant.hpp"
#include "formats_sog_xt.hpp"

#if defined(TINYTENSOR_HAS_VULKAN)
namespace photara::splat::detail::vulkan_backend {
std::array<float, 6> percentile_bounds(const tinytensor::Tensor& means, const float percentile);

tinytensor::Tensor normal_features_to_normals(const tinytensor::Tensor& normal_features);

tinytensor::Tensor normal_features_backward(
    const tinytensor::Tensor& normal_features,
    const tinytensor::Tensor& grad_normals);

PruneMasks prune_masks(
    const GaussianModel& model,
    const float minimum_opacity,
    const float maximum_bounds,
    const std::array<float, 3>& scene_center);

tinytensor::Tensor compute_3d_filter(
    const tinytensor::Tensor& means,
    const std::vector<Camera>& cameras,
    const float minimum_scale_factor);

MultiViewLoss add_multi_view_loss(
    const tinytensor::Tensor& sampled_neighbour_points,
    const tinytensor::Tensor& sampled_inside,
    const RenderResult& reference_render,
    const TrainingView& reference,
    const TrainingView& neighbour,
    const TrainingOptions& options,
    LossGradients& gradients,
    tinytensor::Tensor& grad_sampled_points,
    const bool collect_scalar_terms,
    tinytensor::Tensor* stability_accumulator);

GeometryDistributionSummary summarize_geometry_distribution(const GaussianModel& model);

tinytensor::Tensor unproject_depth_to_world(const tinytensor::Tensor& depth, const Camera& camera);

void add_sample_depth_point_gradients(
    const Camera& reference_camera,
    const tinytensor::Tensor& grad_world_points,
    LossGradients& image_gradients);

void add_sample_depth_model_gradients(
    const DepthSampleGradients& sample_gradients,
    ModelGradients& model_gradients);

LossGradients compute_training_loss(
    const RenderResult& rendered,
    const TrainingView& target,
    const TrainingOptions& options,
    const bool collect_scalar_terms,
    const bool depth_normal_active,
    const bool need_geometry_gradients,
    const bool ai_prior_active);

LossGradients compute_normal_field_loss(
    const RenderResult& rendered,
    const Camera& camera,
    const float weight,
    const bool collect_scalar_terms);

void zero_adam_rows(const tinytensor::Tensor& indices, const std::array<AdamState*, 6>& states);

void add_geometry_regularization(
    const GaussianModel& model,
    ModelGradients& gradients,
    const float opacity_weight,
    const float scale_weight);

tinytensor::Tensor panorama_densification_sizes(
    const GaussianModel& model,
    const std::array<float, 3>& camera_position,
    const float screen_threshold,
    const float scale_modifier);

void split_gaussians(
    GaussianModel& parents,
    GaussianModel& children,
    const tinytensor::Tensor& parent_indices,
    const tinytensor::Tensor& random_samples,
    const tinytensor::Tensor& screen_sizes,
    const SplitMode mode,
    const float minimum_opacity,
    const float split_at_screen_size,
    const float split_opacity_k);

void apply_shape_regularizers(
    GaussianModel& model,
    ModelGradients& gradients,
    const float erank_weight,
    const float erank_s3_weight,
    const float quat_weight,
    const float axis_ratio_weight,
    const float max_axis_ratio);

void apply_oversize_penalty(
    GaussianModel& model,
    const tinytensor::Tensor& radii,
    const float inverse_resolution,
    const float screen_limit,
    const float penalty,
    const float scales_lr);

void apply_adc_decay(GaussianModel& model, const float opacity_decay, const float scale_decay);

void clip_log_scale_by_screen(
    tinytensor::Tensor& log_scales,
    const tinytensor::Tensor& screens,
    const float screen_threshold,
    const float hardness);
} // namespace photara::splat::detail::vulkan_backend

namespace photara::splat::detail::vulkan_backend {
tinytensor::Tensor pam_field_gradients(
    const PamGpuField& field,
    const tinytensor::Tensor& points,
    unsigned neighbors);

void pam_refine_points(
    const PamGpuField& field,
    tinytensor::Tensor& points,
    const tinytensor::Tensor& occupancy,
    unsigned neighbors,
    float iso_value,
    float minimum_gradient_norm_squared,
    float step);
} // namespace photara::splat::detail::vulkan_backend

namespace photara::splat::detail::vulkan_backend {
void sh_adam_quant_select_rows(ShAdamQuant& state, const tinytensor::Tensor& indices);

void sh_adam_quant_zero_rows(ShAdamQuant& state, const tinytensor::Tensor& indices);

void sh_adam_quant_step(
    tinytensor::Tensor& parameter,
    const tinytensor::Tensor& gradient,
    ShAdamQuant& state,
    const int active_stride,
    const float learning_rate,
    const float rest_learning_rate,
    const float beta1,
    const float beta2,
    const float correction1,
    const float correction2,
    const float adam_epsilon,
    const float regularization_factor);
} // namespace photara::splat::detail::vulkan_backend

namespace photara::splat::detail::vulkan_backend {
GaussianModel sog_xt_training_forward(SogXtTrainingCodec& codec);

void sog_xt_training_backward(
    SogXtTrainingCodec& codec,
    const GaussianModel& decoded,
    ModelGradients& g,
    float tv,
    float qtv);
} // namespace photara::splat::detail::vulkan_backend

#endif
