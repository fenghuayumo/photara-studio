#pragma once
#include <cstdint>
#include <vector>

namespace photara::splat::detail {
struct SogPalette {
    std::vector<float> centroids;
    std::vector<std::uint32_t> labels;
};

// Hierarchical Lloyd clustering, including assignment and segmented centroid
// reduction, runs on Vulkan. Host code only compacts the final occupied palette.
SogPalette cluster_sog_vulkan(const std::vector<float>& points, unsigned count,
                              unsigned dimensions, unsigned max_palette = 65536);
} // namespace photara::splat::detail
