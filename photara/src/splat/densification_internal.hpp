#pragma once

// Strategy implementations are declared in their individual headers.
#include "densification_adc_plus.hpp"
#include "densification_igs.hpp"

#include <cstddef>
#include <random>

namespace photara::splat::densification::gpu_detail {

// Device-side plumbing shared by every strategy that keeps its decisions on
// the GPU: model compaction, split growth and weighted parent sampling. The
// ADC+ skeleton and the device port of the legacy ADC-IGS decisions both use
// these, so the two paths cannot drift apart.

GaussianModel select_model_rows(
    const GaussianModel& model, const tinytensor::Tensor& indices);

void append_model(GaussianModel& model, const GaussianModel& added);

void select_adam_rows(
    detail::AdamState& state, const tinytensor::Tensor& indices);

void append_zero_adam(detail::AdamState& state, std::size_t count);

void select_training_rows_gpu(
    GaussianModel& model, const tinytensor::Tensor& indices,
    const AdamStates& states);

// Sample `count` distinct rows of `weights` without replacement. This is the
// device form of the host Gumbel top-k rule: multinomial-without-replacement
// realises the same Plackett-Luce distribution. Returns Int32 row indices, and
// an empty tensor when nothing is samplable.
tinytensor::Tensor weighted_sample_without_replacement(
    const tinytensor::Tensor& weights, std::size_t count);

// Clone, split, zero the parent moments and append the children. `samples` is
// the per-parent random tensor of the split mode and may be empty, which the
// covariance operator expects (it stays on the major axis).
void grow_parents_gpu(
    GaussianModel& model, const tinytensor::Tensor& parents,
    const tinytensor::Tensor& screen_sizes, detail::SplitMode mode,
    const tinytensor::Tensor& samples, float minimum_opacity,
    float split_at_screen_size, bool keep_parent_adam,
    const AdamStates& states);

// Grow through SplitMode::igs_random. The random split offsets stay on the
// host (one normal per local axis, exactly like the host grow path that this
// replaces); everything downstream of them runs on the device.
void grow_igs_random_gpu(
    GaussianModel& model, const tinytensor::Tensor& parents,
    const tinytensor::Tensor& screen_sizes, const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states);

}  // namespace photara::splat::densification::gpu_detail
