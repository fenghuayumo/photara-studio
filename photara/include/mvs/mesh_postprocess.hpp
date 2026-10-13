#pragma once

#include <cstdint>

namespace photara::mvs {

struct MeshPostprocessPreset {
    std::uint64_t target_faces;
    int smooth_iterations;
};

// Budgets for optional postprocessing, independent of extraction quality.
inline constexpr MeshPostprocessPreset mesh_postprocess_preset(int quality) {
    if (quality == 0) return {500'000, 0};
    if (quality == 2) return {2'000'000, 2};
    return {1'000'000, 1};
}

} // namespace photara::mvs
