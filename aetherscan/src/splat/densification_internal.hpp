#pragma once

#include "densification.hpp"

namespace aetherscan::splat::densification::internal {

RefinementCounts refine_adc_plus_gpu(
    GaussianModel& model, detail::DensificationStats& stats,
    unsigned iteration, float scene_extent,
    const mvs::Vec3f& scene_center, const TrainingOptions& options,
    const AdamStates& states);

}  // namespace aetherscan::splat::densification::internal
