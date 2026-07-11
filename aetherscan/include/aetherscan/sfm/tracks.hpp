#pragma once

#include "aetherscan/sfm/scene.hpp"

#include <utility>

namespace aetherscan::sfm {

// Union-find track building from geometrically verified pair matches (openMVS BuildTracks).
void build_tracks(Scene& scene, float min_pair_weight = 0.F);

// Filter triangulated tracks by reprojection / angle / depth bounds.
// Returns {mean_reproj_px, mean_reproj_deg}.
std::pair<float, float> filter_tracks(
    Scene& scene,
    float max_reproj_error_px = 3.F,
    float min_angle_deg = 2.F,
    float mult_depth_near = 0.05F,
    float mult_depth_far = 20.F);

}  // namespace aetherscan::sfm
