#include "cuda_backend.hpp"
#include "splat/device.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace photara::splat::detail {
PpispState make_ppisp_state(const std::size_t views, const TrainingOptions& options) {

#if defined(TINYTENSOR_HAS_VULKAN)
    if (training_device(options) == tinytensor::Device::Vulkan)
        return make_ppisp_state_vulkan(views, options);
#endif

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (training_device(options) == tinytensor::Device::CUDA)
        return cuda_backend::make_ppisp_state(views, options);
#endif
    throw std::runtime_error("make_ppisp_state: requested backend is not enabled");
}

void apply_ppisp(
    const tinytensor::Tensor& color,
    PpispState& state,
    const Camera& camera,
    const std::size_t view) {

#if defined(TINYTENSOR_HAS_VULKAN)
    if (state.parameters.device() == tinytensor::Device::Vulkan)
        return apply_ppisp_vulkan(color, state, camera, view);
#endif

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (state.parameters.device() == tinytensor::Device::CUDA)
        return cuda_backend::apply_ppisp(color, state, camera, view);
#endif
    throw std::runtime_error("apply_ppisp: requested backend is not enabled");
}

void backward_ppisp(
    PpispState& state,
    const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient,
    const Camera& camera,
    const std::size_t view) {

#if defined(TINYTENSOR_HAS_VULKAN)
    if (state.parameters.device() == tinytensor::Device::Vulkan)
        return backward_ppisp_vulkan(state, color, output_gradient, camera, view);
#endif

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (state.parameters.device() == tinytensor::Device::CUDA)
        return cuda_backend::backward_ppisp(state, color, output_gradient, camera, view);
#endif
    throw std::runtime_error("backward_ppisp: requested backend is not enabled");
}

void step_ppisp(PpispState& state, const TrainingOptions& options, const unsigned iteration) {

#if defined(TINYTENSOR_HAS_VULKAN)
    if (state.parameters.device() == tinytensor::Device::Vulkan)
        return step_ppisp_vulkan(state, options, iteration);
#endif

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (state.parameters.device() == tinytensor::Device::CUDA)
        return cuda_backend::step_ppisp(state, options, iteration);
#endif
    throw std::runtime_error("step_ppisp: requested backend is not enabled");
}

std::array<float, 2> ppisp_identity_deviation(const PpispState& state) {

    std::array<float, 2> result{0.F, 0.F};
    if (!state.is_valid() || state.num_params <= 0)
        return result;
    const std::vector<float> values = state.parameters.to_vector();
    const std::size_t views = values.size() / static_cast<std::size_t>(state.num_params);
    if (views == 0)
        return result;
    double total = 0.0;
    std::size_t count = 0;
    for (std::size_t view = 0; view < views; ++view) {
        const float* row = values.data() + view * state.num_params;
        const float deviation = std::abs(std::exp2(row[0]) - 1.F);
        total += deviation;
        ++count;
        result[1] = std::max(result[1], deviation);
    }
    if (count != 0)
        result[0] = static_cast<float>(total / count);
    return result;
}

} // namespace photara::splat::detail
