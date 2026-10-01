#pragma once

#include "splat/types.hpp"

namespace photara::splat::detail {

// Same balanced KD tree and activated field used by the CPU reference.
// Nodes: [point, left, right, axis]. Field rows: mean(3), normal(3),
// inverse variance(3), column-major rotation(9), opacity(1).
struct PamGpuField {
    tinytensor::Tensor nodes;
    tinytensor::Tensor values;
    int root{-1};
};

inline constexpr unsigned pam_gpu_max_neighbors = 64;

PamGpuField make_pam_gpu_field(const GaussianModel& model);

// CPU reference used to validate exact search and anisotropic field math.
std::vector<float> pam_field_gradients_reference(
    const GaussianModel& model, const std::vector<float>& points, unsigned neighbors);

tinytensor::Tensor pam_field_gradients(
    const PamGpuField& field, const tinytensor::Tensor& points,
    unsigned neighbors);

void pam_refine_points(
    const PamGpuField& field, tinytensor::Tensor& points,
    const tinytensor::Tensor& occupancy, unsigned neighbors,
    float iso_value, float minimum_gradient_norm_squared, float step);

}  // namespace photara::splat::detail
