// Vulkan and backend-independent entry points. CUDA kernels stay in .cu files.
#include "cuda_ops.hpp"
#include "geometry_vulkan.hpp"
#include "adc_backends.hpp"
#include "rasterizer_vulkan.hpp"
#include "fused_ssim.hpp"
#include "optimizer_backends.hpp"
#include "sh_adam_quant.hpp"
#include "ppisp.hpp"
#include "bilateral_grid.hpp"
#include "pam_cuda.hpp"
#include "formats_sog_xt.hpp"
#include "igs_selection_cuda.hpp"
#include "core/vram_profiler.hpp"
#include "vulkan/ops.hpp"
#include "vulkan/backend.hpp"
#include "pam_vulkan.hpp"
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>
#include <stdexcept>

namespace photara::splat::detail {
PpispState make_ppisp_state(const std::size_t views, const TrainingOptions& options) {

    throw std::runtime_error("make_ppisp_state: this operation requires the CUDA backend");

}

void apply_ppisp(
    const tinytensor::Tensor& color,
    PpispState& state,
    const Camera& camera,
    const std::size_t view) {

    throw std::runtime_error("apply_ppisp: this operation requires the CUDA backend");

}

void backward_ppisp(
    PpispState& state,
    const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient,
    const Camera& camera,
    const std::size_t view) {

    throw std::runtime_error("backward_ppisp: this operation requires the CUDA backend");

}

void step_ppisp(PpispState& state, const TrainingOptions& options, const unsigned iteration) {

    throw std::runtime_error("step_ppisp: this operation requires the CUDA backend");

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

}
