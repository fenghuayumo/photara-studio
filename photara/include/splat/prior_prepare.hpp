#pragma once
#include "mvs/types.hpp"
#include "splat/options.hpp"
#include <filesystem>

namespace photara::splat {
// Predict pinhole views directly; native fisheye/panorama use overlapping
// perspective predictions reprojected to the full training camera. Depth is
// aligned to observed SfM points per face. Only training views are cached.
// Valid caches are reused before loading MoGe.
void prepare_training_priors(
    const mvs::MvsScene& scene,
    const TrainingOptions& training,
    const std::filesystem::path& model,
    unsigned tokens = 1200);
} // namespace photara::splat
