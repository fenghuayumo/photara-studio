#include "training_dispatch_backends.hpp"
#include "geometry_vulkan.hpp"

namespace photara::splat::detail {
LossGradients compute_step_loss_vulkan(
    const Rasterizer& rasterizer,
    const RenderResult& rendered,
    const RenderResult& loss_render,
    const TrainingView& target,
    const TrainingView& geometry_target,
    const TrainingOptions& options_,
    bool report_progress,
    bool depth_normal_active,
    bool allocate_geometry_gradients,
    bool ai_prior_active,
    bool owned_color_gradient) {
    LossGradients loss;
    const bool mask_enabled = target.has_mask && (options_.use_mask || target.mask_is_validity);
    if (owned_color_gradient)
        loss.color = loss_render.color;
    loss.total = loss.rgb = rasterizer.photometric_loss(
        loss_render, target.rgb, target.mask, mask_enabled, options_.ssim_weight,
        options_.photometric_weight, report_progress, owned_color_gradient ? &loss.color : nullptr);
    loss.alpha = rendered.alpha.is_valid() ? tinytensor::Tensor::zeros_like(rendered.alpha)
                                           : tinytensor::Tensor{};
    detail::add_alpha_mask_loss_vulkan(rendered, target, options_, loss, report_progress);
    if (allocate_geometry_gradients)
        detail::add_depth_normal_loss_vulkan(
            rendered, geometry_target.camera,
            depth_normal_active ? options_.depth_normal_weight : 0.F, loss, report_progress);
    if (options_.use_mvs_depth || options_.use_mvs_normals)
        detail::add_mvs_loss_vulkan(rendered, target, options_, loss, report_progress);
    if (ai_prior_active)
        detail::add_prior_loss_vulkan(rendered, target, options_, loss, report_progress);
    return loss;
}
} // namespace photara::splat::detail
