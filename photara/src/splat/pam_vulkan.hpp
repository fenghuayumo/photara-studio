#pragma once
#include "pam_cuda.hpp"
namespace photara::splat::detail {
tinytensor::Tensor pam_field_gradients_vulkan(const PamGpuField&, const tinytensor::Tensor&,
                                              unsigned);
void pam_refine_points_vulkan(const PamGpuField&, tinytensor::Tensor&, const tinytensor::Tensor&,
                              unsigned, float, float, float);
} // namespace photara::splat::detail
