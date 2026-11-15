#pragma once

// Internal CUDA implementation contract: plain C++ types, no CUDA SDK headers.
// Only shared C++ entry points dispatch to these optional implementations.
#include "ops.hpp"
#include "adc_backends.hpp"
#include "bilateral_grid.hpp"
#include "ppisp.hpp"
#include "fused_ssim.hpp"
#include "pam_ops.hpp"
#include "sh_adam_quant.hpp"
#include "formats_sog_xt.hpp"
#include "igs_selection_cuda.hpp"

#if defined(PHOTARA_HAS_CUDA_SPLAT)
namespace photara::splat::detail::cuda_backend {
DecodedTrainingPixels decode_packed_training_pixels(
    const tinytensor::Tensor& packed,
    const std::uint32_t width,
    const std::uint32_t height,
    const bool decode_mask,
    const bool decode_gray);

tinytensor::Tensor decode_packed_training_normals(
    const tinytensor::Tensor& packed,
    std::uint32_t width,
    std::uint32_t height);

std::array<float, 6> percentile_bounds(const tinytensor::Tensor& means, const float percentile);

ActivatedParameters activate_parameters(const GaussianModel& model);

tinytensor::Tensor normal_features_to_normals(const tinytensor::Tensor& normal_features);

tinytensor::Tensor normal_features_backward(
    const tinytensor::Tensor& normal_features,
    const tinytensor::Tensor& grad_normals);

void bake_3d_filter(GaussianModel& model);

AdcPlusPruneResult adc_plus_prune(
    const GaussianModel& model,
    const float minimum_opacity,
    const float maximum_bounds,
    const std::array<float, 3>& scene_center,
    const std::size_t maximum_count);

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

OpacityProgressStats summarize_opacity_progress(
    const tinytensor::Tensor& opacity_logits,
    const tinytensor::Tensor& opacity_gradients);

tinytensor::Tensor unproject_depth_to_world(const tinytensor::Tensor& depth, const Camera& camera);

void add_sample_depth_point_gradients(
    const Camera& reference_camera,
    const tinytensor::Tensor& grad_world_points,
    LossGradients& image_gradients);

void add_sample_depth_model_gradients(
    const DepthSampleGradients& sample_gradients,
    ModelGradients& model_gradients);

void add_model_gradients(
    const ModelGradients& source,
    ModelGradients& destination,
    const bool include_refine_weight);

void chain_parameter_gradients(
    const GaussianModel& model,
    const ActivatedParameters& activated,
    const tinytensor::Tensor& grad_scales,
    const tinytensor::Tensor& grad_quaternions,
    const tinytensor::Tensor& grad_opacities,
    ModelGradients& gradients);

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

void adam_step_cuda(
    tinytensor::Tensor& parameter,
    const tinytensor::Tensor& gradient,
    AdamState& state,
    const float learning_rate,
    const unsigned step,
    const TrainingOptions& options,
    const std::size_t group_stride,
    const float secondary_learning_rate,
    const float clamp_min,
    const float clamp_max);

void zero_adam_rows(const tinytensor::Tensor& indices, const std::array<AdamState*, 6>& states);

void adam_step_structure_cuda(
    GaussianModel& model,
    const ModelGradients& gradient,
    AdamState& means,
    AdamState& scales,
    AdamState& rotations,
    AdamState& opacity,
    float means_lr,
    unsigned step,
    const TrainingOptions& options,
    float minimum_log_scale,
    float maximum_log_scale);

void adam_step_reduced_second_cuda(
    tinytensor::Tensor& parameter,
    const tinytensor::Tensor& gradient,
    AdamState& state,
    const float learning_rate,
    const unsigned step,
    const TrainingOptions& options,
    const std::size_t row_stride,
    const float secondary_learning_rate);

void adam_step_active_prefix_cuda(
    tinytensor::Tensor& parameter,
    const tinytensor::Tensor& gradient,
    AdamState& state,
    const float learning_rate,
    const unsigned step,
    const TrainingOptions& options,
    const std::size_t full_row_stride,
    const std::size_t active_row_stride,
    const float secondary_learning_rate);

void add_sh_regularization(
    const tinytensor::Tensor& sh,
    tinytensor::Tensor& gradient,
    const float weight);

void add_geometry_regularization(
    const GaussianModel& model,
    ModelGradients& gradients,
    const float opacity_weight,
    const float scale_weight);

void accumulate_densification_stats_cuda(
    const tinytensor::Tensor& refine_weight,
    const tinytensor::Tensor& visibility,
    const tinytensor::Tensor& radii,
    DensificationStats& stats,
    const AdcStatsOptions& options,
    const tinytensor::Tensor& geometry_gradient,
    const tinytensor::Tensor& image_error,
    const tinytensor::Tensor& normalized_size);

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

void relocate_long_axis(
    GaussianModel& model,
    const tinytensor::Tensor& parent_indices,
    const tinytensor::Tensor& destination_indices,
    const float split_opacity_k);

void inject_emc_noise(
    GaussianModel& model,
    const tinytensor::Tensor& radii,
    const float scaler,
    const unsigned seed);

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

tinytensor::Tensor adc_plus_footprint_weights(
    const tinytensor::Tensor& gradients,
    const tinytensor::Tensor& screens);

void inject_adc_noise_cuda(
    GaussianModel& model,
    const tinytensor::Tensor& visibility,
    const tinytensor::Tensor& radii,
    const AdcNoiseOptions& options);

tinytensor::Tensor compute_ssim_cs_error_map(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    const bool mask_enabled,
    const float power);

tinytensor::Tensor scatter_error_map_to_gaussians(
    const tinytensor::Tensor& error_hw,
    const float* mean2d,
    const tinytensor::Tensor& radii,
    const tinytensor::Tensor& visibility);

tinytensor::Tensor densify_avg_scores(
    const tinytensor::Tensor& numerator,
    const tinytensor::Tensor& denominator);

tinytensor::Tensor densify_oversize_weights(
    const tinytensor::Tensor& scores,
    const tinytensor::Tensor& screens,
    const float screen_threshold,
    const float blend);

void clip_log_scale_by_screen(
    tinytensor::Tensor& log_scales,
    const tinytensor::Tensor& screens,
    const float screen_threshold,
    const float hardness);

void reset_opacity(GaussianModel& model, const float maximum_opacity);
} // namespace photara::splat::detail::cuda_backend

namespace photara::splat::detail::cuda_backend {
void pam_accumulate_occupancy_cuda(
    tinytensor::Tensor& occupancy,
    tinytensor::Tensor& observed,
    const tinytensor::Tensor& values,
    const tinytensor::Tensor& foreground,
    const tinytensor::Tensor& background,
    const tinytensor::Tensor& visibility);

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
} // namespace photara::splat::detail::cuda_backend

namespace photara::splat::detail::cuda_backend {
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
} // namespace photara::splat::detail::cuda_backend

namespace photara::splat::detail::cuda_backend {
BilateralGridState make_bilateral_grid_state(
    const std::size_t views,
    const TrainingOptions& options);

void apply_bilateral_grid(
    const tinytensor::Tensor& color,
    BilateralGridState& state,
    const std::size_t view,
    const bool wrap_horizontal);

void backward_bilateral_grid(
    BilateralGridState& state,
    const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient,
    const std::size_t view,
    const bool wrap_horizontal);

void step_bilateral_grid(
    BilateralGridState& state,
    const TrainingOptions& options,
    const unsigned iteration,
    const bool wrap_horizontal);
} // namespace photara::splat::detail::cuda_backend

namespace photara::splat::detail::cuda_backend {
PpispState make_ppisp_state(const std::size_t views, const TrainingOptions& options);

void apply_ppisp(
    const tinytensor::Tensor& color,
    PpispState& state,
    const Camera& camera,
    const std::size_t view);

void backward_ppisp(
    PpispState& state,
    const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient,
    const Camera& camera,
    const std::size_t view);

void step_ppisp(PpispState& state, const TrainingOptions& options, const unsigned iteration);
} // namespace photara::splat::detail::cuda_backend

namespace photara::splat::detail::cuda_backend {
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
    const std::uint32_t height);

float fused_ssim_metric(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    const bool mask_enabled,
    const std::uint32_t width,
    const std::uint32_t height);
} // namespace photara::splat::detail::cuda_backend

namespace photara::splat::densification::gpu_detail::cuda_backend {
tinytensor::Tensor igs_weighted_sample_cuda(
    const tinytensor::Tensor& weights,
    const std::size_t requested,
    const std::uint64_t seed,
    double* prepare_ms,
    double* sort_ms);
}

namespace photara::splat::detail::cuda_backend {
GaussianModel sog_xt_training_forward(SogXtTrainingCodec& codec);

void sog_xt_training_backward(
    SogXtTrainingCodec& codec,
    const GaussianModel& decoded,
    ModelGradients& g,
    float tv,
    float qtv);
} // namespace photara::splat::detail::cuda_backend

#endif
