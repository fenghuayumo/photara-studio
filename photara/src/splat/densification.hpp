#pragma once

#include "cuda_ops.hpp"
#include "mvs/types.hpp"
#include "splat/options.hpp"
#include "splat/types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <random>
#include <vector>

namespace photara::splat::densification {

struct RefinementCounts {
    std::size_t grown{};
    std::size_t pruned{};
};

struct StrategySchedule {
    unsigned start{};
    unsigned stop{};
    unsigned every{};
};

struct SceneGeometry {
    float scale{1.F};
    mvs::Vec3f center{mvs::Vec3f::Zero()};
    float maximum_extent{1.F};
};

using AdamStates = std::array<detail::AdamState*, 6>;

[[nodiscard]] SceneGeometry training_scene_geometry(
    const mvs::MvsScene& scene, bool dense_input);

// Brush derives its optimizer/refinement bounds from the middle 80% of the
// current Gaussian means. scale is BoundingBox::median_size(), while
// maximum_extent is the largest half-extent used by its out-of-bounds prune.
[[nodiscard]] SceneGeometry splat_scene_geometry(
    const std::vector<float>& xyz, float percentile = 0.8F);
[[nodiscard]] SceneGeometry splat_scene_geometry_cuda(
    const tinytensor::Tensor& means, float percentile = 0.8F);

[[nodiscard]] StrategySchedule strategy_schedule(
    const TrainingOptions& options);

[[nodiscard]] bool is_enabled(const TrainingOptions& options);

[[nodiscard]] bool is_refinement_iteration(
    unsigned iteration, const TrainingOptions& options);

// Shared by panorama ADC strategies; the hard model cap is unchanged.
[[nodiscard]] std::size_t panorama_progressive_growth_cap(
    std::size_t initial_count, unsigned iteration,
    const TrainingOptions& options);

[[nodiscard]] GaussianModel clone_model(const GaussianModel& model);

RefinementCounts refine_gaussians(
    GaussianModel& model, detail::DensificationStats& stats,
    unsigned iteration, float scene_extent,
    const mvs::Vec3f& scene_center, const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states);

// EMC (error-map MCMC) refinement: recycle dead rows in place through a
// long-axis split at error-sampled parents, then grow by a fixed multiplier
// of the live count, also error-sampled, with a reserved oversize share.
// Exposed for unit tests.
RefinementCounts refine_emc(
    GaussianModel& model, detail::DensificationStats& stats,
    unsigned iteration, const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states);

}  // namespace photara::splat::densification
