#pragma once

#include "densification.hpp"

#include <random>

namespace photara::splat::densification {

struct IgsSelection {
    tinytensor::Tensor parents;
    std::size_t replacement{}, oversized{}, growth{};
};

// Optional per-stage wall time for select_igs_parents, filled only when the
// caller passes a pointer (the SPLAT_IGS_PROFILE run). Wall time is the right
// unit here: each stage ends in a host readback, so it carries the queue drain
// as well as the dispatch recording.
struct IgsSelectionProfile {
    double replacement_gumbel_ms{};
    double replacement_count_ms{};
    double replacement_sort_ms{};
    double oversized_rank_ms{};
    double growth_gumbel_ms{};
    double growth_count_ms{};
    double growth_sort_ms{};
    double compact_ms{};
};

// Allocate disjoint sets before their union, preserving replacement priority.
IgsSelection select_igs_parents(
    const tinytensor::Tensor& replacement_weights,
    const tinytensor::Tensor& oversize_scores,
    const tinytensor::Tensor& growth_weights,
    std::size_t replacement_slots, std::size_t desired_growth,
    std::size_t capacity, std::mt19937* random = nullptr,
    IgsSelectionProfile* profile = nullptr);

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
