#include "training_dispatch.hpp"
#include "geometry_vulkan.hpp"
#include "training_dispatch_backends.hpp"
#include <Eigen/Geometry>
#include "sh_adam_quant.hpp"
#include <cmath>

namespace photara::splat::detail {
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
    bool owned_color_gradient) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (target.rgb.device() == tinytensor::Device::Vulkan)
        return compute_step_loss_vulkan(
            rasterizer, rendered, loss_render, target, geometry_target, options, report_progress,
            depth_normal_active, allocate_geometry_gradients, ai_prior_active,
            owned_color_gradient);
#endif
    return compute_training_loss(
        loss_render, geometry_target, options, report_progress, depth_normal_active,
        allocate_geometry_gradients, ai_prior_active);
}
tinytensor::Tensor normal_features_from_smallest_axis(const GaussianModel& model) {
#ifdef TINYTENSOR_HAS_VULKAN
    if (model.means.device() == tinytensor::Device::Vulkan)
        return normal_features_from_smallest_axis_vulkan(model);
#endif
    const ActivatedParameters activated = activate_parameters(model);
    const auto scales = activated.scales.to_vector();
    const auto quaternions = activated.quaternions.to_vector();
    std::vector<float> features(model.size() * 4, 0.F);
    for (std::size_t index = 0; index < model.size(); ++index) {
        int minimum_axis = 0;
        if (scales[3 * index + 1] < scales[3 * index])
            minimum_axis = 1;
        if (scales[3 * index + 2] < scales[3 * index + static_cast<std::size_t>(minimum_axis)])
            minimum_axis = 2;
        const Eigen::Quaternionf rotation(
            quaternions[4 * index], quaternions[4 * index + 1], quaternions[4 * index + 2],
            quaternions[4 * index + 3]);
        const Eigen::Vector3f direction = rotation.toRotationMatrix().col(minimum_axis);
        for (int axis = 0; axis < 3; ++axis)
            features[4 * index + static_cast<std::size_t>(axis)] = direction(axis);
        // GaussianWrapping resets the learnable orientation sign to zero.
        features[4 * index + 3] = 0.F;
    }
    return tinytensor::Tensor::from_vector(
        features, {model.size(), std::size_t{4}}, model.means.device());
}

void configure_training_raster(
    RasterizeOptions& raster_options,
    const TrainingOptions& options_,
    const TrainingView& target,
    bool owned_color) {
    const bool vulkan_backend = target.rgb.device() == tinytensor::Device::Vulkan;
    raster_options.copy_attachments = !vulkan_backend || owned_color;
    raster_options.copy_color_only = vulkan_backend && owned_color;
    raster_options.copy_alpha =
        vulkan_backend && options_.use_mask && target.has_mask && !target.mask_is_validity &&
        options_.alpha_mode == AlphaMode::transparent && options_.match_alpha_weight > 0.F;
    raster_options.defer_visibility = vulkan_backend;
}
RenderResult render_normal_field(
    const Rasterizer& rasterizer,
    const GaussianModel& model,
    RenderResult& primary,
    const Camera& camera,
    RasterizeOptions options) {
    if (model.means.device() == tinytensor::Device::Vulkan) {
        options.reuse_projection = true;
        return rasterizer.forward(model, camera, options);
    }
    return rasterizer.forward_colors(model, primary, options.colors_precomp);
}
tinytensor::Tensor reference_color_gradient(
    const Rasterizer& rasterizer,
    const RenderResult& rendered,
    const RenderResult& teacher,
    const TrainingView& target,
    const TrainingOptions& options) {
    if (target.rgb.device() == tinytensor::Device::Vulkan) {
        tinytensor::Tensor gradient;
        rasterizer.photometric_loss(
            rendered, teacher.color, target.mask,
            target.has_mask && (options.use_mask || target.mask_is_validity), options.ssim_weight,
            options.photometric_weight, false, &gradient);
        return gradient;
    }
    auto teacher_target = target;
    teacher_target.rgb = teacher.color;
    auto canonical_render = rendered;
    canonical_render.context.impl.reset();
    return compute_training_loss(
               canonical_render, teacher_target, options, false, false, false, false)
        .color;
}
void step_training_sh(
    GaussianModel& model,
    ModelGradients& gradients,
    ShAdamQuant& sh_quant,
    unsigned active_sh_degree,
    const TrainingOptions& options_,
    unsigned iteration) {
    const bool vulkan_backend = model.sh.device() == tinytensor::Device::Vulkan;
    const std::size_t full_sh_stride = model.sh.shape()[1] * 3;
    const float vulkan_sh_regularization =
        vulkan_backend && full_sh_stride > 3 && options_.sh_regularization_weight > 0.F
            ? 2.F * options_.sh_regularization_weight /
                  static_cast<float>(model.size() * (full_sh_stride - 3))
            : 0.F;
    if (!vulkan_backend)
        add_sh_regularization(model.sh, gradients.sh, options_.sh_regularization_weight);
    const std::size_t active_sh_stride =
        static_cast<std::size_t>(active_sh_degree + 1) * (active_sh_degree + 1) * 3;
    sh_adam_quant_step(
        model.sh, gradients.sh, sh_quant, static_cast<int>(active_sh_stride), options_.sh0_lr,
        options_.sh_rest_lr, options_.beta1, options_.beta2,
        1.F - std::pow(options_.beta1, static_cast<float>(iteration)),
        1.F - std::pow(options_.beta2, static_cast<float>(iteration)), options_.adam_epsilon,
        vulkan_sh_regularization);
}

} // namespace photara::splat::detail
