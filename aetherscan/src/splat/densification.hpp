#pragma once

#include "cuda_ops.hpp"
#include "mvs/types.hpp"
#include "splat/options.hpp"
#include "splat/types.hpp"

#include <array>
#include <cstddef>
#include <random>

namespace aetherscan::splat::densification {

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
};

using AdamStates = std::array<detail::AdamState*, 5>;

[[nodiscard]] SceneGeometry training_scene_geometry(
    const mvs::MvsScene& scene, bool dense_input);

[[nodiscard]] StrategySchedule strategy_schedule(
    const TrainingOptions& options);

[[nodiscard]] bool is_enabled(const TrainingOptions& options);

[[nodiscard]] bool is_refinement_iteration(
    unsigned iteration, const TrainingOptions& options);

[[nodiscard]] GaussianModel clone_model(const GaussianModel& model);

RefinementCounts refine_gaussians(
    GaussianModel& model, detail::DensificationStats& stats,
    unsigned iteration, float scene_extent,
    const mvs::Vec3f& scene_center, const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states);

}  // namespace aetherscan::splat::densification
