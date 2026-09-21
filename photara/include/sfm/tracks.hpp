#pragma once

#include "sfm/scene.hpp"

#include <utility>

namespace photara::sfm {

// Union-find track building from geometrically verified pair matches (openMVS BuildTracks).
void build_tracks(Scene& scene, float min_pair_weight = 0.F);
void rebuild_track_index(Scene& scene);

// Filter triangulated tracks by reprojection / angle / depth bounds.
// Returns {mean_reproj_px, mean_reproj_deg}.
std::pair<float, float> filter_tracks(
    Scene& scene,
    float max_reproj_error_px = 3.F,
    float min_angle_deg = 2.F,
    float mult_depth_near = 0.05F,
    float mult_depth_far = 20.F);

// Incremental filtering for tracks affected by newly registered/locally
// optimized images. Global median-depth clipping is intentionally deferred to
// the next full filter because untouched tracks have unchanged geometry.
std::pair<float, float> filter_tracks(
    Scene& scene,
    const std::vector<Index>& track_ids,
    float max_reproj_error_px = 3.F,
    float min_angle_deg = 2.F);

}  // namespace photara::sfm
