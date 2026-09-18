#pragma once

#include "densification.hpp"

#include <random>

namespace aetherscan::splat::densification {

// ADC-IGS refinement. Every decision - prune masks, candidate scores and
// weights, replacement and growth sampling, which parents split - is taken on
// the GPU from the per-step statistics: the recycle-fraction/best-row/cap
// pruning policy, the accumulated-gradient gate, opacity x edge-factor
// replacement sampling, forced oversized splits and the exact-alpha
// random-axis split. Only the sampled parent rows cross back to the host, so
// a refinement no longer stalls the GPU behind a model download and a host
// sort. Shared device plumbing lives in densification_internal.hpp.
class IgsStrategy {
public:
    RefinementCounts refine(
        GaussianModel& model, detail::DensificationStats& stats,
        unsigned iteration, float scene_extent,
        const mvs::Vec3f& scene_center, const TrainingOptions& options,
        std::mt19937& random, const AdamStates& states) const;
};

}  // namespace aetherscan::splat::densification
