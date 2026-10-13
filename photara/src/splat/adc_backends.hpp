#pragma once

#include "cuda_ops.hpp"

namespace photara::splat::detail {

struct AdcStatsOptions {
    std::size_t count;
    float inverse_resolution;
    bool use_maximum, require_contribution_visibility;
    float step_score_power, oversize_screen_threshold;
    int view_index;
};
struct AdcNoiseOptions {
    float standard_deviation, maximum_noise;
    unsigned seed;
    bool revised;
};

void accumulate_densification_stats_cuda(
    const tinytensor::Tensor& refine_weight,
    const tinytensor::Tensor& visibility, const tinytensor::Tensor& radii,
    DensificationStats& stats, const AdcStatsOptions& options,
    const tinytensor::Tensor& geometry_gradient, const tinytensor::Tensor& image_error,
    const tinytensor::Tensor& normalized_size);
void inject_adc_noise_cuda(GaussianModel& model,
    const tinytensor::Tensor& visibility, const tinytensor::Tensor& radii,
    const AdcNoiseOptions& options);

void accumulate_densification_stats_vulkan(
    const tinytensor::Tensor& refine_weight,
    const tinytensor::Tensor& visibility, const tinytensor::Tensor& radii,
    DensificationStats& stats, const AdcStatsOptions& options,
    const tinytensor::Tensor& geometry_gradient,
    const tinytensor::Tensor& image_error, const tinytensor::Tensor& normalized_size);

void inject_adc_noise_vulkan(GaussianModel& model,
    const tinytensor::Tensor& visibility, const AdcNoiseOptions& options);

}  // namespace photara::splat::detail
