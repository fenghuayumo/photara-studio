#include "cuda_backend.hpp"
#include "splat/device.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace photara::splat::detail {
BilateralGridState make_bilateral_grid_state(
    const std::size_t views,
    const TrainingOptions& options) {

#if defined(TINYTENSOR_HAS_VULKAN)
    if (training_device(options) == tinytensor::Device::Vulkan)
        return make_bilateral_grid_state_vulkan(views, options);
#endif

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (training_device(options) == tinytensor::Device::CUDA)
        return cuda_backend::make_bilateral_grid_state(views, options);
#endif
    throw std::runtime_error("make_bilateral_grid_state: requested backend is not enabled");
}

void apply_bilateral_grid(
    const tinytensor::Tensor& color,
    BilateralGridState& state,
    const std::size_t view,
    const bool wrap_horizontal) {

#if defined(TINYTENSOR_HAS_VULKAN)
    if (state.grids.device() == tinytensor::Device::Vulkan)
        return apply_bilateral_grid_vulkan(color, state, view, wrap_horizontal);
#endif

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (state.grids.device() == tinytensor::Device::CUDA)
        return cuda_backend::apply_bilateral_grid(color, state, view, wrap_horizontal);
#endif
    throw std::runtime_error("apply_bilateral_grid: requested backend is not enabled");
}

void backward_bilateral_grid(
    BilateralGridState& state,
    const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient,
    const std::size_t view,
    const bool wrap_horizontal) {

#if defined(TINYTENSOR_HAS_VULKAN)
    if (state.grids.device() == tinytensor::Device::Vulkan)
        return backward_bilateral_grid_vulkan(state, color, output_gradient, view, wrap_horizontal);
#endif

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (state.grids.device() == tinytensor::Device::CUDA)
        return cuda_backend::backward_bilateral_grid(
            state, color, output_gradient, view, wrap_horizontal);
#endif
    throw std::runtime_error("backward_bilateral_grid: requested backend is not enabled");
}

void step_bilateral_grid(
    BilateralGridState& state,
    const TrainingOptions& options,
    const unsigned iteration,
    const bool wrap_horizontal) {

#if defined(TINYTENSOR_HAS_VULKAN)
    if (state.grids.device() == tinytensor::Device::Vulkan)
        return step_bilateral_grid_vulkan(state, options, iteration, wrap_horizontal);
#endif

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (state.grids.device() == tinytensor::Device::CUDA)
        return cuda_backend::step_bilateral_grid(state, options, iteration, wrap_horizontal);
#endif
    throw std::runtime_error("step_bilateral_grid: requested backend is not enabled");
}

} // namespace photara::splat::detail
