#include "vulkan_backend.hpp"
#include "pam_vulkan.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <climits>
#include <stdexcept>

namespace photara::splat::detail {

tinytensor::Tensor vulkan_backend::pam_field_gradients(
    const PamGpuField& field,
    const tinytensor::Tensor& points,
    unsigned neighbors) {

    return pam_field_gradients_vulkan(field, points, neighbors);
}

void vulkan_backend::pam_refine_points(
    const PamGpuField& field,
    tinytensor::Tensor& points,
    const tinytensor::Tensor& occupancy,
    unsigned neighbors,
    float iso_value,
    float minimum_gradient_norm_squared,
    float step) {

    pam_refine_points_vulkan(
        field, points, occupancy, neighbors, iso_value, minimum_gradient_norm_squared, step);
    return;
}

} // namespace photara::splat::detail
