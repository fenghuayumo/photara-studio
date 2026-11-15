#include "cuda_backend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace photara::splat::detail {
PpispState make_ppisp_state(const std::size_t views, const TrainingOptions& options) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return cuda_backend::make_ppisp_state(views, options);
#else
    throw std::runtime_error("make_ppisp_state: this operation requires the CUDA backend");
#endif
}

void apply_ppisp(
    const tinytensor::Tensor& color,
    PpispState& state,
    const Camera& camera,
    const std::size_t view) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return cuda_backend::apply_ppisp(color, state, camera, view);
#else
    throw std::runtime_error("apply_ppisp: this operation requires the CUDA backend");
#endif
}

void backward_ppisp(
    PpispState& state,
    const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient,
    const Camera& camera,
    const std::size_t view) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return cuda_backend::backward_ppisp(state, color, output_gradient, camera, view);
#else
    throw std::runtime_error("backward_ppisp: this operation requires the CUDA backend");
#endif
}

void step_ppisp(PpispState& state, const TrainingOptions& options, const unsigned iteration) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    return cuda_backend::step_ppisp(state, options, iteration);
#else
    throw std::runtime_error("step_ppisp: this operation requires the CUDA backend");
#endif
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
