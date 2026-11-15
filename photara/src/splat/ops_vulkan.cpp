#include "vulkan_backend.hpp"
#include "geometry_vulkan.hpp"
#include "rasterizer_vulkan.hpp"
#include "vulkan/ops.hpp"
#include "vulkan/backend.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <climits>
#include <stdexcept>

namespace photara::splat::detail {

std::array<float, 6> vulkan_backend::percentile_bounds(
    const tinytensor::Tensor& means,
    const float percentile) {

    if (means.numel() == 0)
        return {-1.F, -1.F, -1.F, 1.F, 1.F, 1.F};
    return tinytensor::vulkan::quantile_bounds(means, percentile);
}

tinytensor::Tensor vulkan_backend::normal_features_to_normals(
    const tinytensor::Tensor& normal_features) {

    return normal_features_to_normals_vulkan(normal_features);
}

tinytensor::Tensor vulkan_backend::normal_features_backward(
    const tinytensor::Tensor& normal_features,
    const tinytensor::Tensor& grad_normals) {

    return normal_features_backward_vulkan(normal_features, grad_normals);
}

PruneMasks vulkan_backend::prune_masks(
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

    // One fused dispatch, matching the CUDA implementation. The first
    // version of this port composed ~30 TinyTensor ops; besides paying a
    // barrier per dispatch it also tripped the [N] x [N, 3] broadcast
    // outer-product quirk (a 41 GB request). Row-local logic belongs in
    // one shader.
    tinytensor::vulkan::prune_masks(
        model.means, model.log_scales, model.quaternions, model.opacity_logits, model.sh,
        masks.keep, masks.hard, masks.opacities, minimum_opacity, maximum_bounds, scene_center[0],
        scene_center[1], scene_center[2]);
    return masks;
}

tinytensor::Tensor vulkan_backend::compute_3d_filter(
    const tinytensor::Tensor& means,
    const std::vector<Camera>& cameras,
    const float minimum_scale_factor) {

    return compute_3d_filter_vulkan(means, cameras, minimum_scale_factor);
}

MultiViewLoss vulkan_backend::add_multi_view_loss(
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

    return add_multi_view_loss_vulkan(
        sampled_neighbour_points, sampled_inside, reference_render, reference, neighbour, options,
        gradients, grad_sampled_points, collect_scalar_terms, stability_accumulator);
}

GeometryDistributionSummary vulkan_backend::summarize_geometry_distribution(
    const GaussianModel& model) {

    return summarize_geometry_distribution_vulkan(model);
}

tinytensor::Tensor vulkan_backend::unproject_depth_to_world(
    const tinytensor::Tensor& depth,
    const Camera& camera) {

    return unproject_depth_to_world_vulkan(depth, camera);
}

void vulkan_backend::add_sample_depth_point_gradients(
    const Camera& reference_camera,
    const tinytensor::Tensor& grad_world_points,
    LossGradients& image_gradients) {

    add_sample_depth_point_gradients_vulkan(reference_camera, grad_world_points, image_gradients);
    return;
}

void vulkan_backend::add_sample_depth_model_gradients(
    const DepthSampleGradients& sample_gradients,
    ModelGradients& model_gradients) {

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

LossGradients vulkan_backend::compute_training_loss(
    const RenderResult& rendered,
    const TrainingView& target,
    const TrainingOptions& options,
    const bool collect_scalar_terms,
    const bool depth_normal_active,
    const bool need_geometry_gradients,
    const bool ai_prior_active) {

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
        add_mvs_loss_vulkan(rendered, target, options, result, collect_scalar_terms);
    if (ai_prior_active && (options.lambda_d > 0.F || options.lambda_n > 0.F))
        add_prior_loss_vulkan(rendered, target, options, result, collect_scalar_terms);
    return result;
}

LossGradients vulkan_backend::compute_normal_field_loss(
    const RenderResult& rendered,
    const Camera& camera,
    const float weight,
    const bool collect_scalar_terms) {

    return compute_normal_field_loss_vulkan(rendered, camera, weight, collect_scalar_terms);
}

void vulkan_backend::zero_adam_rows(
    const tinytensor::Tensor& indices,
    const std::array<AdamState*, 6>& states) {

    if (indices.numel() == 0)
        return;

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

void vulkan_backend::add_geometry_regularization(
    const GaussianModel& model,
    ModelGradients& gradients,
    const float opacity_weight,
    const float scale_weight) {

    const std::size_t count = model.size();
    if (count == 0 || (!(opacity_weight > 0.F) && !(scale_weight > 0.F)))
        return;

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
        const auto prior =
            (scales * (scale_weight / (3.F * static_cast<float>(count)))).masked_fill(invalid, 0.F);
        gradients.log_scales.copy_from(gradients.log_scales.add(prior));
    }
    return;
}

tinytensor::Tensor vulkan_backend::panorama_densification_sizes(
    const GaussianModel& model,
    const std::array<float, 3>& camera_position,
    const float screen_threshold,
    const float scale_modifier) {

    const auto device = model.means.device();
    auto sizes = tinytensor::Tensor::empty({model.size()}, device);
    if (model.size() == 0)
        return sizes;
    constexpr float k_angular_threshold_radians = 30.F * std::numbers::pi_v<float> / 180.F;

    tinytensor::vulkan::panorama_sizes(
        model.means, model.log_scales, model.quaternions, sizes, camera_position[0],
        camera_position[1], camera_position[2], screen_threshold / k_angular_threshold_radians,
        scale_modifier);
    return sizes;
}

void vulkan_backend::split_gaussians(
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
            std::sqrt(
                std::max(
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
            1.F - std::sqrt(std::max(1.F - opacity, 0.F)), minimum_opacity, 1.F - minimum_opacity);
        const float logit = std::log(revised / (1.F - revised));
        parent_opacity[parent] = logit;
        child_opacity[child] = logit;
    }
    const auto device = tinytensor::Device::Vulkan;
    children.means = tinytensor::Tensor::from_vector(child_means, children.means.shape(), device);
    children.log_scales =
        tinytensor::Tensor::from_vector(child_scales, children.log_scales.shape(), device);
    children.opacity_logits =
        tinytensor::Tensor::from_vector(child_opacity, children.opacity_logits.shape(), device);
    tinytensor::vulkan::scatter(
        parents.means, parent_indices,
        tinytensor::Tensor::from_vector(parent_means, {count, std::size_t{3}}, device), 0, false);
    tinytensor::vulkan::scatter(
        parents.log_scales, parent_indices,
        tinytensor::Tensor::from_vector(parent_scales, {count, std::size_t{3}}, device), 0, false);
    tinytensor::vulkan::scatter(
        parents.opacity_logits, parent_indices,
        tinytensor::Tensor::from_vector(parent_opacity, {count}, device), 0, false);
    return;
}

void vulkan_backend::apply_shape_regularizers(
    GaussianModel& model,
    ModelGradients& gradients,
    const float erank_weight,
    const float erank_s3_weight,
    const float quat_weight,
    const float axis_ratio_weight,
    const float max_axis_ratio) {

    if (model.size() == 0)
        return;

    tinytensor::vulkan::apply_shape_regularizers(
        model.log_scales, model.quaternions, gradients.log_scales, gradients.quaternions,
        erank_weight, erank_s3_weight, quat_weight, axis_ratio_weight, max_axis_ratio);
    return;
}

void vulkan_backend::apply_oversize_penalty(
    GaussianModel& model,
    const tinytensor::Tensor& radii,
    const float inverse_resolution,
    const float screen_limit,
    const float penalty,
    const float scales_lr) {

    if (model.size() == 0 || !radii.is_valid() || radii.numel() != model.size() ||
        !(screen_limit > 0.F) || !(penalty > 0.F))
        return;

    tinytensor::vulkan::apply_oversize_penalty(
        model.log_scales, radii, inverse_resolution, screen_limit, penalty, scales_lr);
    return;
}

void vulkan_backend::apply_adc_decay(
    GaussianModel& model,
    const float opacity_decay,
    const float scale_decay) {

    if (model.size() == 0)
        return;
    const float scale_factor = std::max(1.F - scale_decay, 1e-6F);

    auto opacity =
        model.opacity_logits.sigmoid().sub(std::max(opacity_decay, 0.F)).clamp(1e-7F, 1.F - 1e-7F);
    model.opacity_logits = opacity.div(opacity.mul(-1.F).add(1.F)).log();
    model.log_scales = model.log_scales.add(std::log(scale_factor));
    return;
}

void vulkan_backend::clip_log_scale_by_screen(
    tinytensor::Tensor& log_scales,
    const tinytensor::Tensor& screens,
    const float screen_threshold,
    const float hardness) {

    const std::size_t count = screens.numel();
    if (count == 0 || !(screen_threshold > 0.F))
        return;

    auto oversize = screens.div(screen_threshold).clamp_min(1.F);
    if (std::isfinite(hardness))
        oversize = oversize.clamp_max(hardness);
    log_scales = log_scales.sub(oversize.log().unsqueeze(1));
    return;
}

} // namespace photara::splat::detail
