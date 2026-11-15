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
void fused_l1_ssim_loss(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    const bool mask_enabled,
    const float ssim_weight,
    const float photometric_weight,
    tinytensor::Tensor& gradient,
    float* scalar_terms,
    const std::uint32_t width,
    const std::uint32_t height) {

    throw std::runtime_error("fused_l1_ssim_loss: this operation requires the CUDA backend");

}

float fused_ssim_metric(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    const bool mask_enabled,
    const std::uint32_t width,
    const std::uint32_t height) {

    throw std::runtime_error("fused_ssim_metric: this operation requires the CUDA backend");

}

}
