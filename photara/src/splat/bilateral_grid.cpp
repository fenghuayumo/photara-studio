#include "cuda_backend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace photara::splat::detail {
BilateralGridState make_bilateral_grid_state(
    const std::size_t views,
    const TrainingOptions& options) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return cuda_backend::make_bilateral_grid_state(views, options);
#else
    throw std::runtime_error("make_bilateral_grid_state: this operation requires the CUDA backend");
#endif
}

void apply_bilateral_grid(
    const tinytensor::Tensor& color,
    BilateralGridState& state,
    const std::size_t view,
    const bool wrap_horizontal) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return cuda_backend::apply_bilateral_grid(color, state, view, wrap_horizontal);
#else
    throw std::runtime_error("apply_bilateral_grid: this operation requires the CUDA backend");
#endif
}

void backward_bilateral_grid(
    BilateralGridState& state,
    const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient,
    const std::size_t view,
    const bool wrap_horizontal) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return cuda_backend::backward_bilateral_grid(
        state, color, output_gradient, view, wrap_horizontal);
#else
    throw std::runtime_error("backward_bilateral_grid: this operation requires the CUDA backend");
#endif
}

void step_bilateral_grid(
    BilateralGridState& state,
    const TrainingOptions& options,
    const unsigned iteration,
    const bool wrap_horizontal) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return cuda_backend::step_bilateral_grid(state, options, iteration, wrap_horizontal);
#else
    throw std::runtime_error("step_bilateral_grid: this operation requires the CUDA backend");
#endif
}

} // namespace photara::splat::detail
