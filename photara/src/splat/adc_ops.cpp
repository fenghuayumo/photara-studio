#include "adc_backends.hpp"

#include <algorithm>
#include <stdexcept>

namespace photara::splat::detail {

DensificationStats make_densification_stats(std::size_t count, tinytensor::Device device) {
    using tinytensor::Tensor;
    return {Tensor::zeros({count}, device), Tensor::zeros({count}, device),
        Tensor::zeros({count}, device), Tensor::zeros({count}, device),
        Tensor::zeros({count}, device),
        Tensor::zeros({count}, device).to(tinytensor::DataType::Int32),
        Tensor::zeros({count}, device), Tensor::zeros({count}, device)};
}

// Both backends share the training-facing entry points and parameter rules.
// Their launchers only translate these options to CUDA arguments or Vulkan
// resources/push constants; each executes one kernel for the operation.
void accumulate_densification_stats(
    const tinytensor::Tensor& refine_weight, const tinytensor::Tensor& visibility,
    const tinytensor::Tensor& radii, DensificationStats& stats,
    std::uint32_t width, std::uint32_t height, bool use_maximum,
    bool require_contribution_visibility, float step_score_power,
    float oversize_screen_threshold, const tinytensor::Tensor& geometry_gradient,
    int view_index, const tinytensor::Tensor& image_error,
    const tinytensor::Tensor& normalized_size) {
    const auto count = refine_weight.numel();
    if (count == 0) return;
    const AdcStatsOptions options{count,
        1.F / static_cast<float>(std::max<std::uint32_t>(1, std::min(width, height))),
        use_maximum, require_contribution_visibility, step_score_power,
        oversize_screen_threshold, view_index};
    if (refine_weight.device() == tinytensor::Device::Vulkan) {
#ifdef TINYTENSOR_HAS_VULKAN
        accumulate_densification_stats_vulkan(refine_weight, visibility, radii,
            stats, options, geometry_gradient, image_error, normalized_size);
        // Distinct-view support is not consumed by ADC-IGS; unsupported Vulkan
        // strategies are rejected by Trainer before reaching this operation.
        return;
#else
        throw std::invalid_argument("Vulkan ADC backend is not built");
#endif
    }
    accumulate_densification_stats_cuda(refine_weight, visibility, radii,
        stats, options, geometry_gradient, image_error, normalized_size);
}

void inject_adc_noise(GaussianModel& model, const tinytensor::Tensor& visibility,
    float standard_deviation, float maximum_noise, unsigned seed,
    const tinytensor::Tensor& radii, bool revised) {
    if (model.size() == 0 || standard_deviation <= 0.F) return;
    const AdcNoiseOptions options{standard_deviation, std::max(maximum_noise, 0.F), seed, revised};
    if (model.means.device() == tinytensor::Device::Vulkan) {
#ifdef TINYTENSOR_HAS_VULKAN
        if (revised)
            throw std::invalid_argument("Vulkan ADC revised-noise primitive is not implemented");
        inject_adc_noise_vulkan(model, visibility, options);
        return;
#else
        throw std::invalid_argument("Vulkan ADC backend is not built");
#endif
    }
    inject_adc_noise_cuda(model, visibility, radii, options);
}
}  // namespace photara::splat::detail
