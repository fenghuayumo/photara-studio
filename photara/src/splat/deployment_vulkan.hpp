#pragma once
#include "splat/types.hpp"
#include <vector>
namespace photara::splat::detail {
std::vector<float> lod_costs_vulkan(const std::vector<float> &values,
                                    const std::vector<std::int32_t> &pairs);
} // namespace photara::splat::detail
