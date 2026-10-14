#pragma once
#include "splat/types.hpp"
#include <functional>
#include <vector>
namespace photara::splat::detail {
std::vector<float> lod_costs_vulkan(const std::vector<float> &values,
                                    const std::vector<std::int32_t> &pairs);
// Upload candidates once and retain their GPU buffers across voxel batches.
std::function<std::vector<float>(unsigned, unsigned)> prepare_voxel_field_vulkan(
    const std::vector<float> &values, const std::vector<std::int32_t> &offsets,
    const std::vector<std::int32_t> &indices, const std::array<unsigned, 3> &dims,
    const std::array<float, 3> &origin, float size, float threshold);
} // namespace photara::splat::detail
