#pragma once

#include "splat/types.hpp"

#include <cstddef>
#include <functional>

namespace photara::splat {

// Training-free NanoGS simplification. Opacity-prunes the tail, then
// repeatedly merges lowest-cost kNN pairs by mass-preserving moment matching
// until `keep_ratio` of the original count remains.
struct SimplifyOptions {
    float keep_ratio{0.8F};
    float min_opacity{0.10F};
    unsigned knn{16};
    // Fraction of remaining Gaussians that may merge in one pass.
    float merge_cap{0.5F};
    // Negative keeps the incoming SH degree.
    int sh_degree{-1};
    std::function<void(float progress)> progress;
    std::function<bool()> cancelled;
};

[[nodiscard]] GaussianModel simplify_gaussians(
    const GaussianModel& model, const SimplifyOptions& options = {});

}  // namespace photara::splat
