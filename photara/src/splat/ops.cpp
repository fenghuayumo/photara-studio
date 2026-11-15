#include "cuda_backend.hpp"
#include "vulkan_backend.hpp"
#include "core/vram_profiler.hpp"

#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>

namespace photara::splat::detail {
DecodedTrainingPixels decode_packed_training_pixels(
    const tinytensor::Tensor& packed,
    const std::uint32_t width,
    const std::uint32_t height,
    const bool decode_mask,
    const bool decode_gray) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (packed.device() == tinytensor::Device::CUDA)
        return cuda_backend::decode_packed_training_pixels(
            packed, width, height, decode_mask, decode_gray);
#endif
    throw std::runtime_error(
        "decode_packed_training_pixels: this operation requires the CUDA backend");
}

tinytensor::Tensor decode_packed_training_normals(
    const tinytensor::Tensor& packed,
    std::uint32_t width,
    std::uint32_t height) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (packed.device() == tinytensor::Device::CUDA)
        return cuda_backend::decode_packed_training_normals(packed, width, height);
#endif
    throw std::runtime_error(
        "decode_packed_training_normals: this operation requires the CUDA backend");
}

std::array<float, 6> percentile_bounds(const tinytensor::Tensor& means, const float percentile) {
    if (!means.is_valid() ||
        (means.device() != tinytensor::Device::CUDA &&
         means.device() != tinytensor::Device::Vulkan) ||
        means.dtype() != tinytensor::DataType::Float32 || !means.is_contiguous() ||
        means.shape().rank() != 2 || means.shape()[1] != 3 ||
        means.shape()[0] > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument(
            "Percentile bounds require a device float32 [N,3] means tensor");

#if defined(TINYTENSOR_HAS_VULKAN)
    if (means.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::percentile_bounds(means, percentile);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (means.device() == tinytensor::Device::CUDA)
        return cuda_backend::percentile_bounds(means, percentile);
#endif
    throw std::runtime_error("percentile_bounds: requested backend is not enabled");
}

ActivatedParameters activate_parameters(const GaussianModel& model) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::activate_parameters(model);
#endif
    throw std::runtime_error("activate_parameters: this operation requires the CUDA backend");
}

tinytensor::Tensor normal_features_to_normals(const tinytensor::Tensor& normal_features) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (normal_features.is_valid() && normal_features.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::normal_features_to_normals(normal_features);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (normal_features.is_valid() && normal_features.device() == tinytensor::Device::CUDA)
        return cuda_backend::normal_features_to_normals(normal_features);
#endif
    throw std::runtime_error("normal_features_to_normals: requested backend is not enabled");
}

tinytensor::Tensor normal_features_backward(
    const tinytensor::Tensor& normal_features,
    const tinytensor::Tensor& grad_normals) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (normal_features.is_valid() && normal_features.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::normal_features_backward(normal_features, grad_normals);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (normal_features.is_valid() && normal_features.device() == tinytensor::Device::CUDA)
        return cuda_backend::normal_features_backward(normal_features, grad_normals);
#endif
    throw std::runtime_error("normal_features_backward: requested backend is not enabled");
}

void bake_3d_filter(GaussianModel& model) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::bake_3d_filter(model);
#endif
    throw std::runtime_error("bake_3d_filter: this operation requires the CUDA backend");
}

AdcPlusPruneResult adc_plus_prune(
    const GaussianModel& model,
    const float minimum_opacity,
    const float maximum_bounds,
    const std::array<float, 3>& scene_center,
    const std::size_t maximum_count) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::adc_plus_prune(
            model, minimum_opacity, maximum_bounds, scene_center, maximum_count);
#endif
    throw std::runtime_error("adc_plus_prune: this operation requires the CUDA backend");
}

PruneMasks prune_masks(
    const GaussianModel& model,
    const float minimum_opacity,
    const float maximum_bounds,
    const std::array<float, 3>& scene_center) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (model.means.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::prune_masks(model, minimum_opacity, maximum_bounds, scene_center);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::prune_masks(model, minimum_opacity, maximum_bounds, scene_center);
#endif
    throw std::runtime_error("prune_masks: requested backend is not enabled");
}

tinytensor::Tensor compute_3d_filter(
    const tinytensor::Tensor& means,
    const std::vector<Camera>& cameras,
    const float minimum_scale_factor) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (means.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::compute_3d_filter(means, cameras, minimum_scale_factor);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (means.device() == tinytensor::Device::CUDA)
        return cuda_backend::compute_3d_filter(means, cameras, minimum_scale_factor);
#endif
    throw std::runtime_error("compute_3d_filter: requested backend is not enabled");
}

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
    tinytensor::Tensor* stability_accumulator) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (reference_render.median_depth.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::add_multi_view_loss(
            sampled_neighbour_points, sampled_inside, reference_render, reference, neighbour,
            options, gradients, grad_sampled_points, collect_scalar_terms, stability_accumulator);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (reference_render.median_depth.device() == tinytensor::Device::CUDA)
        return cuda_backend::add_multi_view_loss(
            sampled_neighbour_points, sampled_inside, reference_render, reference, neighbour,
            options, gradients, grad_sampled_points, collect_scalar_terms, stability_accumulator);
#endif
    throw std::runtime_error("add_multi_view_loss: requested backend is not enabled");
}

GeometryDistributionSummary summarize_geometry_distribution(const GaussianModel& model) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (model.means.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::summarize_geometry_distribution(model);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::summarize_geometry_distribution(model);
#endif
    throw std::runtime_error("summarize_geometry_distribution: requested backend is not enabled");
}

OpacityProgressStats summarize_opacity_progress(
    const tinytensor::Tensor& opacity_logits,
    const tinytensor::Tensor& opacity_gradients) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (opacity_logits.device() == tinytensor::Device::CUDA)
        return cuda_backend::summarize_opacity_progress(opacity_logits, opacity_gradients);
#endif
    throw std::runtime_error(
        "summarize_opacity_progress: this operation requires the CUDA backend");
}

tinytensor::Tensor unproject_depth_to_world(const tinytensor::Tensor& depth, const Camera& camera) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (depth.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::unproject_depth_to_world(depth, camera);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (depth.device() == tinytensor::Device::CUDA)
        return cuda_backend::unproject_depth_to_world(depth, camera);
#endif
    throw std::runtime_error("unproject_depth_to_world: requested backend is not enabled");
}

void add_sample_depth_point_gradients(
    const Camera& reference_camera,
    const tinytensor::Tensor& grad_world_points,
    LossGradients& image_gradients) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (grad_world_points.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::add_sample_depth_point_gradients(
            reference_camera, grad_world_points, image_gradients);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (grad_world_points.device() == tinytensor::Device::CUDA)
        return cuda_backend::add_sample_depth_point_gradients(
            reference_camera, grad_world_points, image_gradients);
#endif
    throw std::runtime_error("add_sample_depth_point_gradients: requested backend is not enabled");
}

void add_sample_depth_model_gradients(
    const DepthSampleGradients& sample_gradients,
    ModelGradients& model_gradients) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (model_gradients.means.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::add_sample_depth_model_gradients(sample_gradients, model_gradients);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model_gradients.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::add_sample_depth_model_gradients(sample_gradients, model_gradients);
#endif
    throw std::runtime_error("add_sample_depth_model_gradients: requested backend is not enabled");
}

void add_model_gradients(
    const ModelGradients& source,
    ModelGradients& destination,
    const bool include_refine_weight) {
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (destination.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::add_model_gradients(source, destination, include_refine_weight);
#endif

    const auto add = [](const tinytensor::Tensor& source_tensor, tinytensor::Tensor& target) {
        if (!source_tensor.is_valid())
            return;
        if (!target.is_valid() || target.shape() != source_tensor.shape() ||
            target.device() != source_tensor.device())
            throw std::invalid_argument("Cannot merge incompatible model gradients");
        target.copy_from(target.add(source_tensor));
    };
    add(source.means, destination.means);
    add(source.log_scales, destination.log_scales);
    add(source.quaternions, destination.quaternions);
    add(source.opacity_logits, destination.opacity_logits);
    add(source.sh, destination.sh);
    if (include_refine_weight)
        add(source.refine_weight, destination.refine_weight);
}

void chain_parameter_gradients(
    const GaussianModel& model,
    const ActivatedParameters& activated,
    const tinytensor::Tensor& grad_scales,
    const tinytensor::Tensor& grad_quaternions,
    const tinytensor::Tensor& grad_opacities,
    ModelGradients& gradients) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::chain_parameter_gradients(
            model, activated, grad_scales, grad_quaternions, grad_opacities, gradients);
#endif
    throw std::runtime_error("chain_parameter_gradients: this operation requires the CUDA backend");
}

LossGradients compute_training_loss(
    const RenderResult& rendered,
    const TrainingView& target,
    const TrainingOptions& options,
    const bool collect_scalar_terms,
    const bool depth_normal_active,
    const bool need_geometry_gradients,
    const bool ai_prior_active) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (rendered.median_depth.device() == tinytensor::Device::Vulkan ||
        rendered.color.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::compute_training_loss(
            rendered, target, options, collect_scalar_terms, depth_normal_active,
            need_geometry_gradients, ai_prior_active);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (rendered.median_depth.device() == tinytensor::Device::CUDA ||
        rendered.color.device() == tinytensor::Device::CUDA)
        return cuda_backend::compute_training_loss(
            rendered, target, options, collect_scalar_terms, depth_normal_active,
            need_geometry_gradients, ai_prior_active);
#endif
    throw std::runtime_error("compute_training_loss: requested backend is not enabled");
}

LossGradients compute_normal_field_loss(
    const RenderResult& rendered,
    const Camera& camera,
    const float weight,
    const bool collect_scalar_terms) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (rendered.color.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::compute_normal_field_loss(
            rendered, camera, weight, collect_scalar_terms);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (rendered.color.device() == tinytensor::Device::CUDA)
        return cuda_backend::compute_normal_field_loss(
            rendered, camera, weight, collect_scalar_terms);
#endif
    throw std::runtime_error("compute_normal_field_loss: requested backend is not enabled");
}

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
    const float clamp_max) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (parameter.device() == tinytensor::Device::CUDA)
        return cuda_backend::adam_step_cuda(
            parameter, gradient, state, learning_rate, step, options, group_stride,
            secondary_learning_rate, clamp_min, clamp_max);
#endif
    throw std::runtime_error("adam_step_cuda: this operation requires the CUDA backend");
}

void zero_adam_rows(const tinytensor::Tensor& indices, const std::array<AdamState*, 6>& states) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (indices.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::zero_adam_rows(indices, states);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (indices.device() == tinytensor::Device::CUDA)
        return cuda_backend::zero_adam_rows(indices, states);
#endif
    throw std::runtime_error("zero_adam_rows: requested backend is not enabled");
}

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
    float maximum_log_scale) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::adam_step_structure_cuda(
            model, gradient, means, scales, rotations, opacity, means_lr, step, options,
            minimum_log_scale, maximum_log_scale);
#endif
    throw std::runtime_error("adam_step_structure_cuda: this operation requires the CUDA backend");
}

void adam_step_reduced_second_cuda(
    tinytensor::Tensor& parameter,
    const tinytensor::Tensor& gradient,
    AdamState& state,
    const float learning_rate,
    const unsigned step,
    const TrainingOptions& options,
    const std::size_t row_stride,
    const float secondary_learning_rate) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (parameter.device() == tinytensor::Device::CUDA)
        return cuda_backend::adam_step_reduced_second_cuda(
            parameter, gradient, state, learning_rate, step, options, row_stride,
            secondary_learning_rate);
#endif
    throw std::runtime_error(
        "adam_step_reduced_second_cuda: this operation requires the CUDA backend");
}

void adam_step_active_prefix_cuda(
    tinytensor::Tensor& parameter,
    const tinytensor::Tensor& gradient,
    AdamState& state,
    const float learning_rate,
    const unsigned step,
    const TrainingOptions& options,
    const std::size_t full_row_stride,
    const std::size_t active_row_stride,
    const float secondary_learning_rate) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (parameter.device() == tinytensor::Device::CUDA)
        return cuda_backend::adam_step_active_prefix_cuda(
            parameter, gradient, state, learning_rate, step, options, full_row_stride,
            active_row_stride, secondary_learning_rate);
#endif
    throw std::runtime_error(
        "adam_step_active_prefix_cuda: this operation requires the CUDA backend");
}

void add_sh_regularization(
    const tinytensor::Tensor& sh,
    tinytensor::Tensor& gradient,
    const float weight) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (sh.device() == tinytensor::Device::CUDA)
        return cuda_backend::add_sh_regularization(sh, gradient, weight);
#endif
    throw std::runtime_error("add_sh_regularization: this operation requires the CUDA backend");
}

void add_geometry_regularization(
    const GaussianModel& model,
    ModelGradients& gradients,
    const float opacity_weight,
    const float scale_weight) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (model.log_scales.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::add_geometry_regularization(
            model, gradients, opacity_weight, scale_weight);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.log_scales.device() == tinytensor::Device::CUDA)
        return cuda_backend::add_geometry_regularization(
            model, gradients, opacity_weight, scale_weight);
#endif
    throw std::runtime_error("add_geometry_regularization: requested backend is not enabled");
}

void accumulate_densification_stats_cuda(
    const tinytensor::Tensor& refine_weight,
    const tinytensor::Tensor& visibility,
    const tinytensor::Tensor& radii,
    DensificationStats& stats,
    const AdcStatsOptions& options,
    const tinytensor::Tensor& geometry_gradient,
    const tinytensor::Tensor& image_error,
    const tinytensor::Tensor& normalized_size) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (refine_weight.device() == tinytensor::Device::CUDA)
        return cuda_backend::accumulate_densification_stats_cuda(
            refine_weight, visibility, radii, stats, options, geometry_gradient, image_error,
            normalized_size);
#endif
    throw std::runtime_error(
        "accumulate_densification_stats_cuda: this operation requires the CUDA backend");
}

tinytensor::Tensor panorama_densification_sizes(
    const GaussianModel& model,
    const std::array<float, 3>& camera_position,
    const float screen_threshold,
    const float scale_modifier) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (model.means.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::panorama_densification_sizes(
            model, camera_position, screen_threshold, scale_modifier);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::panorama_densification_sizes(
            model, camera_position, screen_threshold, scale_modifier);
#endif
    throw std::runtime_error("panorama_densification_sizes: requested backend is not enabled");
}

void split_gaussians(
    GaussianModel& parents,
    GaussianModel& children,
    const tinytensor::Tensor& parent_indices,
    const tinytensor::Tensor& random_samples,
    const tinytensor::Tensor& screen_sizes,
    const SplitMode mode,
    const float minimum_opacity,
    const float split_at_screen_size,
    const float split_opacity_k) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (parents.means.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::split_gaussians(
            parents, children, parent_indices, random_samples, screen_sizes, mode, minimum_opacity,
            split_at_screen_size, split_opacity_k);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (parents.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::split_gaussians(
            parents, children, parent_indices, random_samples, screen_sizes, mode, minimum_opacity,
            split_at_screen_size, split_opacity_k);
#endif
    throw std::runtime_error("split_gaussians: requested backend is not enabled");
}

void relocate_long_axis(
    GaussianModel& model,
    const tinytensor::Tensor& parent_indices,
    const tinytensor::Tensor& destination_indices,
    const float split_opacity_k) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::relocate_long_axis(
            model, parent_indices, destination_indices, split_opacity_k);
#endif
    throw std::runtime_error("relocate_long_axis: this operation requires the CUDA backend");
}

void inject_emc_noise(
    GaussianModel& model,
    const tinytensor::Tensor& radii,
    const float scaler,
    const unsigned seed) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::inject_emc_noise(model, radii, scaler, seed);
#endif
    throw std::runtime_error("inject_emc_noise: this operation requires the CUDA backend");
}

void apply_shape_regularizers(
    GaussianModel& model,
    ModelGradients& gradients,
    const float erank_weight,
    const float erank_s3_weight,
    const float quat_weight,
    const float axis_ratio_weight,
    const float max_axis_ratio) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (model.log_scales.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::apply_shape_regularizers(
            model, gradients, erank_weight, erank_s3_weight, quat_weight, axis_ratio_weight,
            max_axis_ratio);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.log_scales.device() == tinytensor::Device::CUDA)
        return cuda_backend::apply_shape_regularizers(
            model, gradients, erank_weight, erank_s3_weight, quat_weight, axis_ratio_weight,
            max_axis_ratio);
#endif
    throw std::runtime_error("apply_shape_regularizers: requested backend is not enabled");
}

void apply_oversize_penalty(
    GaussianModel& model,
    const tinytensor::Tensor& radii,
    const float inverse_resolution,
    const float screen_limit,
    const float penalty,
    const float scales_lr) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (model.log_scales.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::apply_oversize_penalty(
            model, radii, inverse_resolution, screen_limit, penalty, scales_lr);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.log_scales.device() == tinytensor::Device::CUDA)
        return cuda_backend::apply_oversize_penalty(
            model, radii, inverse_resolution, screen_limit, penalty, scales_lr);
#endif
    throw std::runtime_error("apply_oversize_penalty: requested backend is not enabled");
}

void apply_adc_decay(GaussianModel& model, const float opacity_decay, const float scale_decay) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (model.means.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::apply_adc_decay(model, opacity_decay, scale_decay);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::apply_adc_decay(model, opacity_decay, scale_decay);
#endif
    throw std::runtime_error("apply_adc_decay: requested backend is not enabled");
}

tinytensor::Tensor adc_plus_footprint_weights(
    const tinytensor::Tensor& gradients,
    const tinytensor::Tensor& screens) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (gradients.device() == tinytensor::Device::CUDA)
        return cuda_backend::adc_plus_footprint_weights(gradients, screens);
#endif
    throw std::runtime_error(
        "adc_plus_footprint_weights: this operation requires the CUDA backend");
}

void inject_adc_noise_cuda(
    GaussianModel& model,
    const tinytensor::Tensor& visibility,
    const tinytensor::Tensor& radii,
    const AdcNoiseOptions& options) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::inject_adc_noise_cuda(model, visibility, radii, options);
#endif
    throw std::runtime_error("inject_adc_noise_cuda: this operation requires the CUDA backend");
}

tinytensor::Tensor compute_ssim_cs_error_map(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    const bool mask_enabled,
    const float power) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (prediction.device() == tinytensor::Device::CUDA)
        return cuda_backend::compute_ssim_cs_error_map(
            prediction, target, mask, mask_enabled, power);
#endif
    throw std::runtime_error("compute_ssim_cs_error_map: this operation requires the CUDA backend");
}

tinytensor::Tensor scatter_error_map_to_gaussians(
    const tinytensor::Tensor& error_hw,
    const float* mean2d,
    const tinytensor::Tensor& radii,
    const tinytensor::Tensor& visibility) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (error_hw.device() == tinytensor::Device::CUDA)
        return cuda_backend::scatter_error_map_to_gaussians(error_hw, mean2d, radii, visibility);
#endif
    throw std::runtime_error(
        "scatter_error_map_to_gaussians: this operation requires the CUDA backend");
}

tinytensor::Tensor densify_avg_scores(
    const tinytensor::Tensor& numerator,
    const tinytensor::Tensor& denominator) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (numerator.device() == tinytensor::Device::CUDA)
        return cuda_backend::densify_avg_scores(numerator, denominator);
#endif
    throw std::runtime_error("densify_avg_scores: this operation requires the CUDA backend");
}

tinytensor::Tensor densify_oversize_weights(
    const tinytensor::Tensor& scores,
    const tinytensor::Tensor& screens,
    const float screen_threshold,
    const float blend) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (scores.device() == tinytensor::Device::CUDA)
        return cuda_backend::densify_oversize_weights(scores, screens, screen_threshold, blend);
#endif
    throw std::runtime_error("densify_oversize_weights: this operation requires the CUDA backend");
}

void clip_log_scale_by_screen(
    tinytensor::Tensor& log_scales,
    const tinytensor::Tensor& screens,
    const float screen_threshold,
    const float hardness) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (log_scales.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::clip_log_scale_by_screen(
            log_scales, screens, screen_threshold, hardness);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (log_scales.device() == tinytensor::Device::CUDA)
        return cuda_backend::clip_log_scale_by_screen(
            log_scales, screens, screen_threshold, hardness);
#endif
    throw std::runtime_error("clip_log_scale_by_screen: requested backend is not enabled");
}

void reset_opacity(GaussianModel& model, const float maximum_opacity) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (model.means.device() == tinytensor::Device::CUDA)
        return cuda_backend::reset_opacity(model, maximum_opacity);
#endif
    throw std::runtime_error("reset_opacity: this operation requires the CUDA backend");
}

} // namespace photara::splat::detail
