#pragma once
#include <cstdint>
#include <vector>

namespace photara::splat::detail {
struct SogPalette {
    std::vector<float> centroids;
    std::vector<std::uint32_t> labels;
};

// Hierarchical initialization followed by global Lloyd reassignment on Vulkan.
// The exact requested budget is preserved; bounded point batches share centers.
SogPalette cluster_sog_vulkan(
    const std::vector<float>& points,
    unsigned count,
    unsigned dimensions,
    unsigned max_palette = 65536);
} // namespace photara::splat::detail
