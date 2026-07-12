#pragma once

#include "sfm/scene.hpp"

namespace aetherscan::sfm {

struct TriangulationOptions {
    float reproj_threshold_px{4.F};
    float min_angle_deg{2.F};
    unsigned min_inliers{2};
    // Below this observation count, skip RANSAC and use linear LLS only.
    unsigned min_observations_for_ransac{4};
    unsigned ransac_iterations{64};
    unsigned ransac_seed{123};
    bool use_lo_ransac{true};
    bool refine_nonlinear{true};
    unsigned refine_iterations{10};
    // After a successful consensus set, peel remaining registered outliers into
    // new tracks and triangulate them (bounded recursion depth).
    bool split_tracks{true};
    unsigned max_splits_per_track{3};
};

// Skew-symmetric multi-view triangulation with optional LO-RANSAC and
// nonlinear N-view refinement. Marks inliers at the front of track.observations.
unsigned triangulate_track(
    Track& track,
    const Scene& scene,
    const TriangulationOptions& options = {});

unsigned triangulate_track(
    Track& track,
    const Scene& scene,
    float reproj_threshold_px,
    float min_angle_deg,
    unsigned min_inliers = 2);

unsigned triangulate_tracks(
    Scene& scene,
    bool outliers_only = false,
    const TriangulationOptions& options = {});

unsigned triangulate_tracks(
    Scene& scene,
    bool outliers_only,
    float reproj_threshold_px,
    float min_angle_deg);

float track_min_ray_angle_deg(const Track& track, const Scene& scene);

}  // namespace aetherscan::sfm
