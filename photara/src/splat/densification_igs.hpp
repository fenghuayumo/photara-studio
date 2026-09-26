#pragma once

#include "densification.hpp"

#include <random>

namespace photara::splat::densification {

struct IgsSelection {
    tinytensor::Tensor parents;
    std::size_t replacement{}, oversized{}, growth{};
};

// Allocate disjoint sets before their union, preserving replacement priority.
IgsSelection select_igs_parents(
    const tinytensor::Tensor& replacement_weights,
    const tinytensor::Tensor& oversize_scores,
    const tinytensor::Tensor& growth_weights,
    std::size_t replacement_slots, std::size_t desired_growth,
    std::size_t capacity, std::mt19937* random = nullptr);

// ADC-IGS refinement. Every decision - prune masks, candidate scores and
// weights, replacement and growth sampling, which parents split - is shared
// across the backends from the per-step statistics: the recycle-fraction/best-row/cap
// pruning policy, the accumulated-gradient gate, opacity x edge-factor
// replacement sampling, forced oversized splits and the exact-alpha
// random-axis split. Weighted sampling intentionally uses the common seeded
// host rule at the sparse refinement boundary so CUDA and Vulkan select the
// same parents. Shared device plumbing lives in densification_internal.hpp.
class IgsStrategy {
public:
    RefinementCounts refine(
        GaussianModel& model, detail::DensificationStats& stats,
        unsigned iteration, float scene_extent,
        const mvs::Vec3f& scene_center, const TrainingOptions& options,
        std::mt19937& random, const AdamStates& states) const;
};

}  // namespace photara::splat::densification
