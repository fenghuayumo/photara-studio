#pragma once
#include "mvs/types.hpp"
#include "splat/options.hpp"
#include <filesystem>

namespace photara::splat {
// Predict at the exact full training camera, align to its observed SfM points,
// and cache only training views. Valid caches are reused before loading MoGe.
void prepare_training_priors(const mvs::MvsScene& scene,
    const TrainingOptions& training, const std::filesystem::path& model,
    unsigned tokens = 1200);
}
