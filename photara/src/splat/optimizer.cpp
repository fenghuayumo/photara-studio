#include "optimizer.hpp"
#include "optimizer_backends.hpp"

#include "core/vram_profiler.hpp"

#include <stdexcept>

namespace photara::splat::detail {

AdamState make_adam_state(const tinytensor::Tensor& parameter) {
    tinytensor::VramScope scope("optimizer.state");
    return {tinytensor::Tensor::zeros_like(parameter),
            tinytensor::Tensor::zeros_like(parameter)};
}

AdamState make_reduced_second_adam_state(
    const tinytensor::Tensor& parameter) {
    tinytensor::VramScope scope("optimizer.state.reduced_second");
    const auto dimensions = parameter.shape().dims();
    if (dimensions.empty())
        throw std::invalid_argument(
            "reduced-second Adam requires at least one parameter dimension");
    return {
        tinytensor::Tensor::zeros_like(parameter),
        tinytensor::Tensor::zeros(
            {dimensions.front()}, parameter.device())};
}

void adam_step(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, const float learning_rate, const unsigned step,
    const TrainingOptions& options, const std::size_t group_stride,
    const float secondary_learning_rate, const float clamp_min,
    const float clamp_max, const float grouped_rest_regularization) {
#ifdef TINYTENSOR_HAS_VULKAN
    if (parameter.device() == tinytensor::Device::Vulkan) {
        adam_step_vulkan(parameter, gradient, state, learning_rate, step,
            options, group_stride, secondary_learning_rate, clamp_min,
            clamp_max, 0, grouped_rest_regularization);
        return;
    }
#endif
    adam_step_cuda(parameter, gradient, state, learning_rate, step, options,
        group_stride, secondary_learning_rate, clamp_min, clamp_max);
}

void adam_step_structure(
    GaussianModel& model, const ModelGradients& gradient,
    AdamState& means, AdamState& scales, AdamState& rotations,
    AdamState& opacity, const float means_lr, const unsigned step,
    const TrainingOptions& options, const float minimum_log_scale,
    const float maximum_log_scale) {
#ifdef TINYTENSOR_HAS_VULKAN
    if (model.means.device() == tinytensor::Device::Vulkan) {
        adam_step_structure_vulkan(model, gradient, means, scales, rotations,
            opacity, means_lr, step, options, minimum_log_scale,
            maximum_log_scale);
        return;
    }
#endif
    adam_step_structure_cuda(model, gradient, means, scales, rotations,
        opacity, means_lr, step, options, minimum_log_scale,
        maximum_log_scale);
}

void adam_step_reduced_second(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, const float learning_rate, const unsigned step,
    const TrainingOptions& options, const std::size_t row_stride,
    const float secondary_learning_rate) {
    if (parameter.device() == tinytensor::Device::Vulkan)
        throw std::invalid_argument(
            "reduced-second Adam is not supported by the Vulkan backend");
    adam_step_reduced_second_cuda(parameter, gradient, state, learning_rate,
        step, options, row_stride, secondary_learning_rate);
}

void adam_step_active_prefix(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, const float learning_rate, const unsigned step,
    const TrainingOptions& options, const std::size_t full_row_stride,
    const std::size_t active_row_stride,
    const float secondary_learning_rate,
    const float grouped_rest_regularization) {
#ifdef TINYTENSOR_HAS_VULKAN
    if (parameter.device() == tinytensor::Device::Vulkan) {
        adam_step_vulkan(parameter, gradient, state, learning_rate, step,
            options, full_row_stride, secondary_learning_rate,
            -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(), active_row_stride,
            grouped_rest_regularization);
        return;
    }
#endif
    adam_step_active_prefix_cuda(parameter, gradient, state, learning_rate,
        step, options, full_row_stride, active_row_stride,
        secondary_learning_rate);
}

void constrain_scale_ratio(
    tinytensor::Tensor& log_scales, const float maximum_ratio) {
    if (log_scales.device() == tinytensor::Device::Vulkan) {
        if (maximum_ratio > 1.F)
            throw std::invalid_argument(
                "scale-ratio projection is not supported by Vulkan yet");
        return;
    }
    constrain_scale_ratio_cuda(log_scales, maximum_ratio);
}

}  // namespace photara::splat::detail
