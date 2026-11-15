#include "cuda_backend.hpp"
#include "vulkan_backend.hpp"
#include "core/vram_profiler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace photara::splat::detail {
void pam_accumulate_occupancy_cuda(
    tinytensor::Tensor& occupancy,
    tinytensor::Tensor& observed,
    const tinytensor::Tensor& values,
    const tinytensor::Tensor& foreground,
    const tinytensor::Tensor& background,
    const tinytensor::Tensor& visibility) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (occupancy.device() == tinytensor::Device::CUDA)
        return cuda_backend::pam_accumulate_occupancy_cuda(
            occupancy, observed, values, foreground, background, visibility);
#endif
    throw std::runtime_error(
        "pam_accumulate_occupancy_cuda: this operation requires the CUDA backend");
}

tinytensor::Tensor pam_field_gradients(
    const PamGpuField& field,
    const tinytensor::Tensor& points,
    unsigned neighbors) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (points.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::pam_field_gradients(field, points, neighbors);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (points.device() == tinytensor::Device::CUDA)
        return cuda_backend::pam_field_gradients(field, points, neighbors);
#endif
    throw std::runtime_error("pam_field_gradients: requested backend is not enabled");
}

void pam_refine_points(
    const PamGpuField& field,
    tinytensor::Tensor& points,
    const tinytensor::Tensor& occupancy,
    unsigned neighbors,
    float iso_value,
    float minimum_gradient_norm_squared,
    float step) {
#if defined(TINYTENSOR_HAS_VULKAN)
    if (points.device() == tinytensor::Device::Vulkan)
        return vulkan_backend::pam_refine_points(
            field, points, occupancy, neighbors, iso_value, minimum_gradient_norm_squared, step);
#endif
#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (points.device() == tinytensor::Device::CUDA)
        return cuda_backend::pam_refine_points(
            field, points, occupancy, neighbors, iso_value, minimum_gradient_norm_squared, step);
#endif
    throw std::runtime_error("pam_refine_points: requested backend is not enabled");
}

} // namespace photara::splat::detail
