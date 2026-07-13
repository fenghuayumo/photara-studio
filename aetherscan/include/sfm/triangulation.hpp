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
    // Fast split gate: first absorb registered observations supported by the
    // parent's current 3D point, then require a small alternate consensus
    // before launching LO-RANSAC for the remaining registered outliers.
    bool use_fast_split_gate{true};
    unsigned split_gate_min_support{3};
    unsigned split_gate_max_observations{16};
    unsigned split_gate_max_pairs{64};
    float split_gate_threshold_multiplier{1.5F};
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

// Incremental variant. Only tracks whose geometry can have changed are
// revisited; recursive splitting is also restricted to this candidate set.
unsigned triangulate_tracks(
    Scene& scene,
    const std::vector<Index>& track_ids,
    bool outliers_only,
    const TriangulationOptions& options = {});

unsigned triangulate_tracks(
    Scene& scene,
    bool outliers_only,
    float reproj_threshold_px,
    float min_angle_deg);

unsigned triangulate_tracks(
    Scene& scene,
    const std::vector<Index>& track_ids,
    bool outliers_only,
    float reproj_threshold_px,
    float min_angle_deg);

float track_min_ray_angle_deg(const Track& track, const Scene& scene);

}  // namespace aetherscan::sfm
