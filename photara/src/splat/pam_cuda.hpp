#pragma once

#include "splat/types.hpp"

namespace photara::splat::detail {

// Balanced KD tree and activated Gaussian field for device projection.
// Nodes: [point, left, right, axis]. Field rows: mean(3), normal(3),
// inverse variance(3), column-major rotation(9), opacity(1).
struct PamGpuField {
    tinytensor::Tensor nodes;
    tinytensor::Tensor values;
    int root{-1};
    tinytensor::Tensor bounds; // Per-node subtree bounds: minimum(3), maximum(3).
};

inline constexpr unsigned pam_gpu_max_neighbors = 64;

PamGpuField make_pam_gpu_field(const GaussianModel& model);

tinytensor::Tensor pam_field_gradients(
    const PamGpuField& field, const tinytensor::Tensor& points,
    unsigned neighbors);

void pam_refine_points(
    const PamGpuField& field, tinytensor::Tensor& points,
    const tinytensor::Tensor& occupancy, unsigned neighbors,
    float iso_value, float minimum_gradient_norm_squared, float step);

// Reduce a view into initialized occupancy/visibility accumulators. Foreground
// votes min occupancy; explicit background votes zero. Missing background means
// no background votes; missing visibility uses the foreground/frustum mask.
void pam_accumulate_occupancy_cuda(
    tinytensor::Tensor& occupancy, tinytensor::Tensor& observed,
    const tinytensor::Tensor& view_occupancy, const tinytensor::Tensor& foreground,
    const tinytensor::Tensor& background = {}, const tinytensor::Tensor& visibility = {});

}  // namespace photara::splat::detail
