#include "optimizer_backends.hpp"

#include "vulkan/backend.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace photara::splat::detail {

void adam_step_vulkan(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, const float learning_rate, const unsigned step,
    const TrainingOptions& options, const std::size_t group_stride,
    const float secondary_learning_rate, const float clamp_min,
    const float clamp_max, const std::size_t active_row_stride,
    const float grouped_rest_regularization) {
    if (parameter.numel() == 0) return;
    if (state.first.numel() != parameter.numel() ||
        state.second.numel() != parameter.numel())
        throw std::invalid_argument(
            "Vulkan Adam requires full per-parameter moment state: parameter=" +
            std::to_string(parameter.numel()) + " first=" +
            std::to_string(state.first.numel()) + " second=" +
            std::to_string(state.second.numel()));
    tinytensor::vulkan::AdamStepOptions backend;
    backend.learning_rate = learning_rate;
    backend.secondary_learning_rate = secondary_learning_rate;
    backend.group_stride = static_cast<std::uint32_t>(group_stride);
    backend.active_row_stride =
        static_cast<std::uint32_t>(active_row_stride);
    backend.beta1 = options.beta1;
    backend.beta2 = options.beta2;
    backend.correction1 =
        1.F - std::pow(options.beta1, static_cast<float>(step));
    backend.correction2 =
        1.F - std::pow(options.beta2, static_cast<float>(step));
    backend.epsilon = options.adam_epsilon;
    backend.clamp_min = clamp_min;
    backend.clamp_max = clamp_max;
    backend.grouped_rest_regularization = grouped_rest_regularization;
    tinytensor::vulkan::adam_step(
        parameter, gradient, state.first, state.second, backend);
}

void adam_step_structure_vulkan(
    GaussianModel& model, const ModelGradients& gradient,
    AdamState& means, AdamState& scales, AdamState& rotations,
    AdamState& opacity, const float means_lr, const unsigned step,
    const TrainingOptions& options, const float minimum_log_scale,
    const float maximum_log_scale) {
    if (options.max_scale_ratio > 1.F)
        throw std::invalid_argument(
            "Vulkan structure Adam does not yet support max_scale_ratio");
    adam_step_vulkan(model.means, gradient.means, means, means_lr, step,
        options, 0, 0.F, -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
    adam_step_vulkan(model.log_scales, gradient.log_scales, scales,
        options.scales_lr, step, options, 0, 0.F, minimum_log_scale,
        maximum_log_scale);
    adam_step_vulkan(model.quaternions, gradient.quaternions, rotations,
        options.quaternions_lr, step, options, 0, 0.F,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
    adam_step_vulkan(model.opacity_logits, gradient.opacity_logits, opacity,
        options.opacities_lr, step, options, 0, 0.F, -12.F, 12.F);
}

}  // namespace photara::splat::detail
