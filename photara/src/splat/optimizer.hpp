#pragma once

#include "splat/options.hpp"
#include "splat/types.hpp"

#include <cstddef>
#include <limits>

namespace photara::splat::detail {

// Backend-neutral Adam state. The tensor device selects the CUDA or Vulkan
// implementation; callers use the same functions and training options.
struct AdamState {
    tinytensor::Tensor first;
    tinytensor::Tensor second;
};

AdamState make_adam_state(const tinytensor::Tensor& parameter);

// Keep a full first moment but only one second-moment scalar per row. This is
// the storage layout used by Brush's AdamScaled for SH coefficients.
AdamState make_reduced_second_adam_state(
    const tinytensor::Tensor& parameter);

void adam_step(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, float learning_rate, unsigned step,
    const TrainingOptions& options, std::size_t group_stride = 0,
    float secondary_learning_rate = 0.F,
    float clamp_min = -std::numeric_limits<float>::infinity(),
    float clamp_max = std::numeric_limits<float>::infinity(),
    float grouped_rest_regularization = 0.F);

// CUDA fuses the four structure groups into one kernel. Vulkan dispatches the
// same operation through device Adam while preserving the identical public API.
void adam_step_structure(GaussianModel& model, const ModelGradients& gradient,
    AdamState& means, AdamState& scales, AdamState& rotations, AdamState& opacity,
    float means_lr, unsigned step, const TrainingOptions& options,
    float minimum_log_scale, float maximum_log_scale);

void adam_step_reduced_second(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, float learning_rate, unsigned step,
    const TrainingOptions& options, std::size_t row_stride,
    float secondary_learning_rate);

// Leave inactive higher-order SH coefficients and both moments untouched.
void adam_step_active_prefix(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, float learning_rate, unsigned step,
    const TrainingOptions& options, std::size_t full_row_stride,
    std::size_t active_row_stride, float secondary_learning_rate,
    float grouped_rest_regularization = 0.F);

void constrain_scale_ratio(
    tinytensor::Tensor& log_scales, float maximum_ratio);

}  // namespace photara::splat::detail
