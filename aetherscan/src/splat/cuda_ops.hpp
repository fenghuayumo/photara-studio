#pragma once

#include "splat/options.hpp"
#include "splat/types.hpp"

#include <limits>

namespace aetherscan::splat::detail {

struct ActivatedParameters {
    tinytensor::Tensor scales;
    tinytensor::Tensor quaternions;
    tinytensor::Tensor opacities;
};

struct LossGradients {
    tinytensor::Tensor color;
    tinytensor::Tensor alpha;
    tinytensor::Tensor depth;
    tinytensor::Tensor normal;
    float total{};
    float rgb{};
    float alpha_value{};
    float depth_value{};
    float normal_value{};
};

struct AdamState {
    tinytensor::Tensor first;
    tinytensor::Tensor second;
};

struct DensificationStats {
    tinytensor::Tensor gradient;
    tinytensor::Tensor count;
    tinytensor::Tensor max_screen_radius;
    tinytensor::Tensor priority;
};

ActivatedParameters activate_parameters(const GaussianModel& model);

void chain_parameter_gradients(
    const GaussianModel& model, const ActivatedParameters& activated,
    const tinytensor::Tensor& grad_scales,
    const tinytensor::Tensor& grad_quaternions,
    const tinytensor::Tensor& grad_opacities,
    ModelGradients& gradients);

LossGradients compute_training_loss(
    const RenderResult& rendered, const TrainingView& target,
    const TrainingOptions& options, bool collect_scalar_terms);

AdamState make_adam_state(const tinytensor::Tensor& parameter);

void adam_step(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, float learning_rate, unsigned step,
    const TrainingOptions& options, std::size_t group_stride = 0,
    float secondary_learning_rate = 0.F,
    float clamp_min = -std::numeric_limits<float>::infinity(),
    float clamp_max = std::numeric_limits<float>::infinity());

void constrain_scale_ratio(
    tinytensor::Tensor& log_scales, float maximum_ratio);

DensificationStats make_densification_stats(std::size_t count);

void accumulate_densification_stats(
    const tinytensor::Tensor& refine_weight,
    const tinytensor::Tensor& radii,
    DensificationStats& stats,
    std::uint32_t width,
    std::uint32_t height,
    bool use_maximum);

// Mutate selected parents and their already-cloned children in place.
// mode: 1=default split, 2=ADC+ split, 3=ADC-IGS split,
//       4=dense-MVS tangent-plane split.
void split_gaussians(
    GaussianModel& parents,
    GaussianModel& children,
    const tinytensor::Tensor& parent_indices,
    const tinytensor::Tensor& random_samples,
    int mode,
    float minimum_opacity);

void apply_adc_decay(
    GaussianModel& model, float opacity_decay, float scale_decay);

void inject_adc_noise(
    GaussianModel& model,
    const tinytensor::Tensor& radii,
    float standard_deviation,
    float maximum_noise,
    unsigned seed);

void reset_opacity(GaussianModel& model, float maximum_opacity);

}  // namespace aetherscan::splat::detail
