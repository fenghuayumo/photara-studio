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
BilateralGridState make_bilateral_grid_state(
    const std::size_t views,
    const TrainingOptions& options) {

    throw std::runtime_error("make_bilateral_grid_state: this operation requires the CUDA backend");

}

void apply_bilateral_grid(
    const tinytensor::Tensor& color,
    BilateralGridState& state,
    const std::size_t view,
    const bool wrap_horizontal) {

    throw std::runtime_error("apply_bilateral_grid: this operation requires the CUDA backend");

}

void backward_bilateral_grid(
    BilateralGridState& state,
    const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient,
    const std::size_t view,
    const bool wrap_horizontal) {

    throw std::runtime_error("backward_bilateral_grid: this operation requires the CUDA backend");

}

void step_bilateral_grid(
    BilateralGridState& state,
    const TrainingOptions& options,
    const unsigned iteration,
    const bool wrap_horizontal) {

    throw std::runtime_error("step_bilateral_grid: this operation requires the CUDA backend");

}

}
