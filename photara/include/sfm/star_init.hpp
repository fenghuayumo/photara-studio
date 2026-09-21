#pragma once

#include "sfm/scene.hpp"

namespace photara::sfm {

struct StarInitConfig {
    unsigned min_views{4};
    unsigned max_views{36};
    unsigned min_tracks_per_view{50};
    float max_reproj_error{6.F};
    float min_angle_deg{1.F};
    unsigned min_initial_tracks{100};
};

Index select_reference_view(const Scene& scene);

// Star-configuration initialization (openMVS StarInitializer).
bool star_initialize(Scene& scene, const StarInitConfig& config = {});

}  // namespace photara::sfm
