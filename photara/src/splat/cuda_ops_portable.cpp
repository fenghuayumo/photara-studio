// Vulkan and backend-independent entry points. CUDA kernels stay in .cu files.
#include "cuda_ops.hpp"
#include "geometry_vulkan.hpp"
#include "adc_backends.hpp"
#include "rasterizer_vulkan.hpp"
#include "fused_ssim.hpp"
#include "optimizer_backends.hpp"
#include "sh_adam_quant.hpp"
#include "ppisp.hpp"
#include "bilateral_grid.hpp"
#include "pam_cuda.hpp"
#include "formats_sog_xt.hpp"
#include "igs_selection_cuda.hpp"
#include "core/vram_profiler.hpp"
#include "vulkan/ops.hpp"
#include "vulkan/backend.hpp"
#include "pam_vulkan.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <climits>
#include <cstring>
#include <stdexcept>

namespace photara::splat::detail {
DecodedTrainingPixels decode_packed_training_pixels(
    const tinytensor::Tensor& packed,
    const std::uint32_t width,
    const std::uint32_t height,
    const bool decode_mask,
    const bool decode_gray) {

    throw std::runtime_error("decode_packed_training_pixels: this operation requires the CUDA backend");

}

tinytensor::Tensor decode_packed_training_normals(
    const tinytensor::Tensor& packed, std::uint32_t width, std::uint32_t height) {

    throw std::runtime_error("decode_packed_training_normals: this operation requires the CUDA backend");

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
    if(means.numel()==0) return {-1.F,-1.F,-1.F,1.F,1.F,1.F};
    return tinytensor::vulkan::quantile_bounds(means,percentile);

}

ActivatedParameters activate_parameters(const GaussianModel& model) {

    throw std::runtime_error("activate_parameters: this operation requires the CUDA backend");

}

tinytensor::Tensor normal_features_to_normals(const tinytensor::Tensor& normal_features) {

    if (normal_features.is_valid() && normal_features.device() == tinytensor::Device::Vulkan)
        return normal_features_to_normals_vulkan(normal_features);
    throw std::runtime_error("normal_features_to_normals: this operation requires the CUDA backend");

}

tinytensor::Tensor normal_features_backward(
    const tinytensor::Tensor& normal_features,
    const tinytensor::Tensor& grad_normals) {

    if (normal_features.is_valid() && normal_features.device() == tinytensor::Device::Vulkan)
        return normal_features_backward_vulkan(normal_features, grad_normals);
    throw std::runtime_error("normal_features_backward: this operation requires the CUDA backend");

}

void bake_3d_filter(GaussianModel& model) {

    throw std::runtime_error("bake_3d_filter: this operation requires the CUDA backend");

}

AdcPlusPruneResult adc_plus_prune(
    const GaussianModel& model,
    const float minimum_opacity,
    const float maximum_bounds,
    const std::array<float, 3>& scene_center,
    const std::size_t maximum_count) {

    throw std::runtime_error("adc_plus_prune: this operation requires the CUDA backend");

}

PruneMasks prune_masks(
    const GaussianModel& model,
    const float minimum_opacity,
    const float maximum_bounds,
    const std::array<float, 3>& scene_center) {

    const std::size_t count = model.size();
    const auto device = model.means.device();
    PruneMasks masks{
        tinytensor::Tensor::zeros_bool({count}, device),
        tinytensor::Tensor::zeros_bool({count}, device),
        tinytensor::Tensor::empty({count}, device)};
    if (count == 0)
        return masks;
    if (device == tinytensor::Device::Vulkan) {
        // One fused dispatch, the twin of the CUDA kernel below. The first
        // version of this port composed ~30 TinyTensor ops; besides paying a
        // barrier per dispatch it also tripped the [N] x [N, 3] broadcast
        // outer-product quirk (a 41 GB request). Row-local logic belongs in
        // one shader.
        tinytensor::vulkan::prune_masks(
            model.means, model.log_scales, model.quaternions, model.opacity_logits, model.sh,
            masks.keep, masks.hard, masks.opacities, minimum_opacity, maximum_bounds,
            scene_center[0], scene_center[1], scene_center[2]);
        return masks;
    }
    throw std::runtime_error("prune_masks: this operation requires the CUDA backend");

}

tinytensor::Tensor compute_3d_filter(
    const tinytensor::Tensor& means,
    const std::vector<Camera>& cameras,
    const float minimum_scale_factor) {

    if (means.device() == tinytensor::Device::Vulkan)
        return compute_3d_filter_vulkan(means, cameras, minimum_scale_factor);
    throw std::runtime_error("compute_3d_filter: this operation requires the CUDA backend");

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

    const std::size_t pixels =
        static_cast<std::size_t>(reference.camera.width) * reference.camera.height;
    if (pixels == 0 ||
        (options.multi_view_geo_weight <= 0.F && options.multi_view_ncc_weight <= 0.F))
        return {};
    if (reference_render.median_depth.device() == tinytensor::Device::Vulkan)
        return add_multi_view_loss_vulkan(
            sampled_neighbour_points, sampled_inside, reference_render, reference, neighbour,
            options, gradients, grad_sampled_points, collect_scalar_terms, stability_accumulator);
    throw std::runtime_error("add_multi_view_loss: this operation requires the CUDA backend");

}

GeometryDistributionSummary summarize_geometry_distribution(const GaussianModel& model) {

    if (model.means.device() == tinytensor::Device::Vulkan)
        return summarize_geometry_distribution_vulkan(model);
    throw std::runtime_error("summarize_geometry_distribution: this operation requires the CUDA backend");

}

OpacityProgressStats summarize_opacity_progress(
    const tinytensor::Tensor& opacity_logits,
    const tinytensor::Tensor& opacity_gradients) {

    throw std::runtime_error("summarize_opacity_progress: this operation requires the CUDA backend");

}

tinytensor::Tensor unproject_depth_to_world(const tinytensor::Tensor& depth, const Camera& camera) {

    if (depth.device() == tinytensor::Device::Vulkan)
        return unproject_depth_to_world_vulkan(depth, camera);
    throw std::runtime_error("unproject_depth_to_world: this operation requires the CUDA backend");

}

void add_sample_depth_point_gradients(
    const Camera& reference_camera,
    const tinytensor::Tensor& grad_world_points,
    LossGradients& image_gradients) {

    if (grad_world_points.device() == tinytensor::Device::Vulkan) {
        add_sample_depth_point_gradients_vulkan(
            reference_camera, grad_world_points, image_gradients);
        return;
    }
    throw std::runtime_error("add_sample_depth_point_gradients: this operation requires the CUDA backend");

}

void add_sample_depth_model_gradients(
    const DepthSampleGradients& sample_gradients,
    ModelGradients& model_gradients) {

    if (model_gradients.means.device() == tinytensor::Device::Vulkan) {
        // Raster gradients are slices of a packed device allocation. Generic
        // Tensor view assignment uses host pointers; copy_from preserves the
        // slice and records a Vulkan copy, including its byte offset.
        model_gradients.means.copy_from(model_gradients.means.add(sample_gradients.model.means));
        model_gradients.log_scales.copy_from(
            model_gradients.log_scales.add(sample_gradients.model.log_scales));
        model_gradients.quaternions.copy_from(
            model_gradients.quaternions.add(sample_gradients.model.quaternions));
        model_gradients.opacity_logits.copy_from(
            model_gradients.opacity_logits.add(sample_gradients.model.opacity_logits));
        return;
    }
    throw std::runtime_error("add_sample_depth_model_gradients: this operation requires the CUDA backend");

}

void add_model_gradients(
    const ModelGradients& source,
    ModelGradients& destination,
    const bool include_refine_weight) {

    const auto add = [](const tinytensor::Tensor& source_tensor, tinytensor::Tensor& target) {
        if (!source_tensor.is_valid()) return;
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
    if (include_refine_weight) add(source.refine_weight, destination.refine_weight);

}

void chain_parameter_gradients(
    const GaussianModel& model,
    const ActivatedParameters& activated,
    const tinytensor::Tensor& grad_scales,
    const tinytensor::Tensor& grad_quaternions,
    const tinytensor::Tensor& grad_opacities,
    ModelGradients& gradients) {

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

    if (rendered.median_depth.device() == tinytensor::Device::Vulkan ||
        rendered.color.device() == tinytensor::Device::Vulkan) {
        LossGradients result;
        result.alpha = tinytensor::Tensor::zeros_like(rendered.alpha);
        if (options.photometric_weight > 0.F) {
            const bool mask = target.has_mask && (options.use_mask || target.mask_is_validity);
            result.rgb = vulkan_photometric_loss(
                rendered, target.rgb, target.mask, mask, options.ssim_weight,
                options.photometric_weight, collect_scalar_terms, &result.color);
            result.total = result.rgb;
        } else {
            result.color = tinytensor::Tensor::zeros_like(rendered.color);
        }
        add_alpha_mask_loss_vulkan(rendered, target, options, result, collect_scalar_terms);
        if (need_geometry_gradients)
            add_depth_normal_loss_vulkan(
                rendered, target.camera, depth_normal_active ? options.depth_normal_weight : 0.F,
                result, collect_scalar_terms);
        if (options.use_mvs_depth || options.use_mvs_normals)
            add_mvs_loss_vulkan(rendered,target,options,result,collect_scalar_terms);
        if (ai_prior_active && (options.lambda_d > 0.F || options.lambda_n > 0.F))
            add_prior_loss_vulkan(rendered, target, options, result, collect_scalar_terms);
        return result;
    }
    throw std::runtime_error("compute_training_loss: this operation requires the CUDA backend");

}

LossGradients compute_normal_field_loss(
    const RenderResult& rendered,
    const Camera& camera,
    const float weight,
    const bool collect_scalar_terms) {

    if (rendered.color.device() == tinytensor::Device::Vulkan)
        return compute_normal_field_loss_vulkan(rendered, camera, weight, collect_scalar_terms);
    throw std::runtime_error("compute_normal_field_loss: this operation requires the CUDA backend");

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

    throw std::runtime_error("adam_step_cuda: this operation requires the CUDA backend");

}

void zero_adam_rows(const tinytensor::Tensor& indices, const std::array<AdamState*, 6>& states) {

    if (indices.numel() == 0)
        return;
    if (indices.device() == tinytensor::Device::Vulkan) {
        for (AdamState* state : states) {
            if (!state)
                continue;
            if (state->first.is_valid())
                state->first.index_fill_(0, indices, 0.F);
            if (state->second.is_valid())
                state->second.index_fill_(0, indices, 0.F);
        }
        return;
    }
    throw std::runtime_error("zero_adam_rows: this operation requires the CUDA backend");

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

    throw std::runtime_error("adam_step_reduced_second_cuda: this operation requires the CUDA backend");

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

    throw std::runtime_error("adam_step_active_prefix_cuda: this operation requires the CUDA backend");

}

void add_sh_regularization(
    const tinytensor::Tensor& sh,
    tinytensor::Tensor& gradient,
    const float weight) {

    throw std::runtime_error("add_sh_regularization: this operation requires the CUDA backend");

}

void add_geometry_regularization(
    const GaussianModel& model,
    ModelGradients& gradients,
    const float opacity_weight,
    const float scale_weight) {

    const std::size_t count = model.size();
    if (count == 0 || (!(opacity_weight > 0.F) && !(scale_weight > 0.F)))
        return;
    if (model.log_scales.device() == tinytensor::Device::Vulkan) {
        if (opacity_weight > 0.F) {
            const auto invalid = model.opacity_logits.isfinite().logical_not();
            const auto logits = model.opacity_logits.masked_fill(invalid, 0.F);
            const auto alpha = logits.neg().exp().add(1.F).reciprocal();
            const auto prior =
                (alpha * alpha.neg().add(1.F) * (opacity_weight / static_cast<float>(count)))
                    .masked_fill(invalid, 0.F);
            gradients.opacity_logits.copy_from(gradients.opacity_logits.add(prior));
        }
        if (scale_weight > 0.F) {
            const auto invalid =
                model.log_scales.isfinite().logical_not().logical_or(model.log_scales.le(-40.F));
            const auto scales = model.log_scales.masked_fill(invalid, 0.F).exp();
            const auto prior = (scales * (scale_weight / (3.F * static_cast<float>(count))))
                                   .masked_fill(invalid, 0.F);
            gradients.log_scales.copy_from(gradients.log_scales.add(prior));
        }
        return;
    }
    throw std::runtime_error("add_geometry_regularization: this operation requires the CUDA backend");

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

    throw std::runtime_error("accumulate_densification_stats_cuda: this operation requires the CUDA backend");

}

tinytensor::Tensor panorama_densification_sizes(
    const GaussianModel& model,
    const std::array<float, 3>& camera_position,
    const float screen_threshold,
    const float scale_modifier) {

    const auto device = model.means.device();
    auto sizes = tinytensor::Tensor::empty({model.size()}, device);
    if (model.size() == 0)
        return sizes;
    constexpr float k_angular_threshold_radians = 30.F * std::numbers::pi_v<float> / 180.F;
    if (device == tinytensor::Device::Vulkan) {
        tinytensor::vulkan::panorama_sizes(
            model.means, model.log_scales, model.quaternions, sizes, camera_position[0],
            camera_position[1], camera_position[2], screen_threshold / k_angular_threshold_radians,
            scale_modifier);
        return sizes;
    }
    throw std::runtime_error("panorama_densification_sizes: this operation requires the CUDA backend");

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

    const std::size_t count = parent_indices.numel();
    if (count == 0)
        return;
    if (parents.means.device() == tinytensor::Device::Vulkan) {
        if (mode != SplitMode::igs_random)
            throw std::invalid_argument("Vulkan split primitive currently supports ADC-IGS only");
        // `children` already holds a copy of every selected parent row, so
        // the row math only needs the selection. The previous path downloaded
        // and re-uploaded every full-model tensor at each refinement; the
        // mutated parent rows now go back through a device-side scatter.
        auto parent_means = children.means.to_vector();
        auto parent_scales = children.log_scales.to_vector();
        auto parent_opacity = children.opacity_logits.to_vector();
        const auto parent_rotations = children.quaternions.to_vector();
        auto child_means = children.means.to_vector();
        auto child_scales = children.log_scales.to_vector();
        auto child_opacity = children.opacity_logits.to_vector();
        const auto indices = parent_indices.to_vector_int();
        const auto samples = random_samples.to_vector();
        for (std::size_t child = 0; child < count; ++child) {
            // Both sides are row-local copies of the same parent row.
            const std::size_t parent = child;
            int largest = 0;
            if (parent_scales[3 * parent + 1] > parent_scales[3 * parent + largest])
                largest = 1;
            if (parent_scales[3 * parent + 2] > parent_scales[3 * parent + largest])
                largest = 2;
            float local[3]{};
            for (int axis = 0; axis < 3; ++axis)
                local[axis] = std::exp(parent_scales[3 * parent + axis]) * samples[3 * child];
            const float* raw = parent_rotations.data() + 4 * parent;
            const float inverse_norm =
                1.F /
                std::sqrt(std::max(
                    raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3], 1e-20F));
            const float w = raw[0] * inverse_norm;
            const float qx = raw[1] * inverse_norm;
            const float qy = raw[2] * inverse_norm;
            const float qz = raw[3] * inverse_norm;
            const float tx = 2.F * (qy * local[2] - qz * local[1]);
            const float ty = 2.F * (qz * local[0] - qx * local[2]);
            const float tz = 2.F * (qx * local[1] - qy * local[0]);
            const float offset[3]{
                local[0] + w * tx + (qy * tz - qz * ty), local[1] + w * ty + (qz * tx - qx * tz),
                local[2] + w * tz + (qx * ty - qy * tx)};
            for (int axis = 0; axis < 3; ++axis) {
                const float center = parent_means[3 * parent + axis];
                parent_means[3 * parent + axis] = center - offset[axis];
                child_means[3 * child + axis] = center + offset[axis];
                if (axis == largest) {
                    parent_scales[3 * parent + axis] += std::log(0.5F);
                    child_scales[3 * child + axis] += std::log(0.5F);
                }
            }
            const float opacity = 1.F / (1.F + std::exp(-parent_opacity[parent]));
            const float revised = std::clamp(
                1.F - std::sqrt(std::max(1.F - opacity, 0.F)), minimum_opacity,
                1.F - minimum_opacity);
            const float logit = std::log(revised / (1.F - revised));
            parent_opacity[parent] = logit;
            child_opacity[child] = logit;
        }
        const auto device = tinytensor::Device::Vulkan;
        children.means =
            tinytensor::Tensor::from_vector(child_means, children.means.shape(), device);
        children.log_scales =
            tinytensor::Tensor::from_vector(child_scales, children.log_scales.shape(), device);
        children.opacity_logits =
            tinytensor::Tensor::from_vector(child_opacity, children.opacity_logits.shape(), device);
        tinytensor::vulkan::scatter(
            parents.means, parent_indices,
            tinytensor::Tensor::from_vector(parent_means, {count, std::size_t{3}}, device), 0,
            false);
        tinytensor::vulkan::scatter(
            parents.log_scales, parent_indices,
            tinytensor::Tensor::from_vector(parent_scales, {count, std::size_t{3}}, device), 0,
            false);
        tinytensor::vulkan::scatter(
            parents.opacity_logits, parent_indices,
            tinytensor::Tensor::from_vector(parent_opacity, {count}, device), 0, false);
        return;
    }
    throw std::runtime_error("split_gaussians: this operation requires the CUDA backend");

}

void relocate_long_axis(
    GaussianModel& model,
    const tinytensor::Tensor& parent_indices,
    const tinytensor::Tensor& destination_indices,
    const float split_opacity_k) {

    throw std::runtime_error("relocate_long_axis: this operation requires the CUDA backend");

}

void inject_emc_noise(
    GaussianModel& model,
    const tinytensor::Tensor& radii,
    const float scaler,
    const unsigned seed) {

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

    if (model.size() == 0)
        return;
    if (model.log_scales.device() == tinytensor::Device::Vulkan) {
        tinytensor::vulkan::apply_shape_regularizers(
            model.log_scales, model.quaternions, gradients.log_scales, gradients.quaternions,
            erank_weight, erank_s3_weight, quat_weight, axis_ratio_weight, max_axis_ratio);
        return;
    }
    throw std::runtime_error("apply_shape_regularizers: this operation requires the CUDA backend");

}

void apply_oversize_penalty(
    GaussianModel& model,
    const tinytensor::Tensor& radii,
    const float inverse_resolution,
    const float screen_limit,
    const float penalty,
    const float scales_lr) {

    if (model.size() == 0 || !radii.is_valid() || radii.numel() != model.size() ||
        !(screen_limit > 0.F) || !(penalty > 0.F))
        return;
    if (model.log_scales.device() == tinytensor::Device::Vulkan) {
        tinytensor::vulkan::apply_oversize_penalty(
            model.log_scales, radii, inverse_resolution, screen_limit, penalty, scales_lr);
        return;
    }
    throw std::runtime_error("apply_oversize_penalty: this operation requires the CUDA backend");

}

void apply_adc_decay(GaussianModel& model, const float opacity_decay, const float scale_decay) {

    if (model.size() == 0)
        return;
    const float scale_factor = std::max(1.F - scale_decay, 1e-6F);
    if (model.means.device() == tinytensor::Device::Vulkan) {
        auto opacity = model.opacity_logits.sigmoid()
                           .sub(std::max(opacity_decay, 0.F))
                           .clamp(1e-7F, 1.F - 1e-7F);
        model.opacity_logits = opacity.div(opacity.mul(-1.F).add(1.F)).log();
        model.log_scales = model.log_scales.add(std::log(scale_factor));
        return;
    }
    throw std::runtime_error("apply_adc_decay: this operation requires the CUDA backend");

}

tinytensor::Tensor adc_plus_footprint_weights(
    const tinytensor::Tensor& gradients,
    const tinytensor::Tensor& screens) {

    throw std::runtime_error("adc_plus_footprint_weights: this operation requires the CUDA backend");

}

void inject_adc_noise_cuda(
    GaussianModel& model,
    const tinytensor::Tensor& visibility,
    const tinytensor::Tensor& radii,
    const AdcNoiseOptions& options) {

    throw std::runtime_error("inject_adc_noise_cuda: this operation requires the CUDA backend");

}

tinytensor::Tensor compute_ssim_cs_error_map(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    const bool mask_enabled,
    const float power) {

    throw std::runtime_error("compute_ssim_cs_error_map: this operation requires the CUDA backend");

}

tinytensor::Tensor scatter_error_map_to_gaussians(
    const tinytensor::Tensor& error_hw,
    const float* mean2d,
    const tinytensor::Tensor& radii,
    const tinytensor::Tensor& visibility) {

    throw std::runtime_error("scatter_error_map_to_gaussians: this operation requires the CUDA backend");

}

tinytensor::Tensor densify_avg_scores(
    const tinytensor::Tensor& numerator,
    const tinytensor::Tensor& denominator) {

    throw std::runtime_error("densify_avg_scores: this operation requires the CUDA backend");

}

tinytensor::Tensor densify_oversize_weights(
    const tinytensor::Tensor& scores,
    const tinytensor::Tensor& screens,
    const float screen_threshold,
    const float blend) {

    throw std::runtime_error("densify_oversize_weights: this operation requires the CUDA backend");

}

void clip_log_scale_by_screen(
    tinytensor::Tensor& log_scales,
    const tinytensor::Tensor& screens,
    const float screen_threshold,
    const float hardness) {

    const std::size_t count = screens.numel();
    if (count == 0 || !(screen_threshold > 0.F))
        return;
    if (log_scales.device() == tinytensor::Device::Vulkan) {
        auto oversize = screens.div(screen_threshold).clamp_min(1.F);
        if (std::isfinite(hardness))
            oversize = oversize.clamp_max(hardness);
        log_scales = log_scales.sub(oversize.log().unsqueeze(1));
        return;
    }
    throw std::runtime_error("clip_log_scale_by_screen: this operation requires the CUDA backend");

}

void reset_opacity(GaussianModel& model, const float maximum_opacity) {

    throw std::runtime_error("reset_opacity: this operation requires the CUDA backend");

}

}
