#pragma once

#include "sfm/scene.hpp"

namespace photara::sfm {

struct PairWeightingOptions {
    unsigned min_inliers{15};
    double max_triplet_rotation_error_deg{5.0};
    float triplet_saturation{5.F};
    unsigned min_triplets_for_penalty{2};
    float max_inconsistent_triplet_ratio{0.5F};
    float inconsistent_triplet_scale{0.1F};
};

struct PairWeightingSummary {
    unsigned weighted_pairs{0};
    unsigned supported_pairs{0};
    unsigned inconsistent_pairs{0};
    std::uint64_t tested_triplets{0};
};

// Adds graph-local connectivity and three-view rotation-cycle reliability to
// the intrinsic two-view pair weights. Strongly inconsistent edges are
// down-weighted, not removed, so graph bridges remain available as fallbacks.
PairWeightingSummary compute_pair_weights(
    Scene& scene,
    const PairWeightingOptions& options = {});

}  // namespace photara::sfm
