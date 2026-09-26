#pragma once

#include "optimizer.hpp"

namespace photara::splat::detail {

void adam_step_cuda(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, float learning_rate, unsigned step,
    const TrainingOptions& options, std::size_t group_stride,
    float secondary_learning_rate, float clamp_min, float clamp_max);
void adam_step_structure_cuda(
    GaussianModel& model, const ModelGradients& gradient,
    AdamState& means, AdamState& scales, AdamState& rotations,
    AdamState& opacity, float means_lr, unsigned step,
    const TrainingOptions& options, float minimum_log_scale,
    float maximum_log_scale);
void adam_step_reduced_second_cuda(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, float learning_rate, unsigned step,
    const TrainingOptions& options, std::size_t row_stride,
    float secondary_learning_rate);
void adam_step_active_prefix_cuda(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, float learning_rate, unsigned step,
    const TrainingOptions& options, std::size_t full_row_stride,
    std::size_t active_row_stride, float secondary_learning_rate);
void constrain_scale_ratio_cuda(
    tinytensor::Tensor& log_scales, float maximum_ratio);

#ifdef TINYTENSOR_HAS_VULKAN
void adam_step_vulkan(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, float learning_rate, unsigned step,
    const TrainingOptions& options, std::size_t group_stride,
    float secondary_learning_rate, float clamp_min, float clamp_max,
    std::size_t active_row_stride = 0,
    float grouped_rest_regularization = 0.F);
void adam_step_structure_vulkan(
    GaussianModel& model, const ModelGradients& gradient,
    AdamState& means, AdamState& scales, AdamState& rotations,
    AdamState& opacity, float means_lr, unsigned step,
    const TrainingOptions& options, float minimum_log_scale,
    float maximum_log_scale);
#endif

}  // namespace photara::splat::detail
