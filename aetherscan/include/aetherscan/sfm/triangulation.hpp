#pragma once

#include "aetherscan/sfm/scene.hpp"

namespace aetherscan::sfm {

// Skew-symmetric multi-view triangulation (openMVS TriangulateSkewLLS).
// Works on unit bearings; marks inliers at the front of track.observations.
unsigned triangulate_track(
    Track& track,
    const Scene& scene,
    float reproj_threshold_px = 4.F,
    float min_angle_deg = 2.F,
    unsigned min_inliers = 2);

unsigned triangulate_tracks(
    Scene& scene,
    bool outliers_only = false,
    float reproj_threshold_px = 4.F,
    float min_angle_deg = 2.F);

float track_min_ray_angle_deg(const Track& track, const Scene& scene);

}  // namespace aetherscan::sfm
