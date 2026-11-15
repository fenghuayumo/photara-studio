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
void pam_accumulate_occupancy_cuda(
    tinytensor::Tensor& occupancy,
    tinytensor::Tensor& observed,
    const tinytensor::Tensor& values,
    const tinytensor::Tensor& foreground,
    const tinytensor::Tensor& background,
    const tinytensor::Tensor& visibility) {

    throw std::runtime_error("pam_accumulate_occupancy_cuda: this operation requires the CUDA backend");

}

tinytensor::Tensor pam_field_gradients(
    const PamGpuField& field,
    const tinytensor::Tensor& points,
    unsigned neighbors) {

    if (points.device() == tinytensor::Device::Vulkan)
        return pam_field_gradients_vulkan(field, points, neighbors);
    throw std::runtime_error("pam_field_gradients: this operation requires the CUDA backend");

}

void pam_refine_points(
    const PamGpuField& field,
    tinytensor::Tensor& points,
    const tinytensor::Tensor& occupancy,
    unsigned neighbors,
    float iso_value,
    float minimum_gradient_norm_squared,
    float step) {

    if (points.device() == tinytensor::Device::Vulkan) {
        pam_refine_points_vulkan(
            field, points, occupancy, neighbors, iso_value, minimum_gradient_norm_squared, step);
        return;
    }
    throw std::runtime_error("pam_refine_points: this operation requires the CUDA backend");

}

}
