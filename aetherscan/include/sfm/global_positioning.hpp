#pragma once

#include "sfm/scene.hpp"

namespace aetherscan::sfm {

enum class GlobalPositioningConstraint {
    only_points,
    only_cameras,
    points_and_cameras_balanced,
    points_and_cameras,
};

struct GlobalPositioningOptions {
    unsigned min_views_per_track{3};
    // Kept for source compatibility; positioning is a single Ceres solve.
    unsigned max_irls_iterations{8};
    unsigned max_num_iterations{200};
    double function_tolerance{1e-5};
    double huber_threshold{0.1};
    unsigned random_seed{123};
    bool generate_random_positions{true};
    bool generate_random_points{true};
    bool generate_scales{true};
    bool optimize_positions{true};
    bool optimize_points{true};
    bool optimize_scales{true};
    GlobalPositioningConstraint constraint{
        GlobalPositioningConstraint::only_points};
    double constraint_reweight_scale{1.0};
};

struct GlobalPositioningSummary {
    bool success{false};
    unsigned positioned_images{0};
    unsigned positioned_tracks{0};
    unsigned observations{0};
    unsigned iterations{0};
    double final_residual{0.0};
};

// Fixed-rotation global positioning. Jointly solves camera centers, 3D points,
// and one positive-depth scale per observation from X - C = scale * world_ray.
GlobalPositioningSummary solve_global_positions(
    Scene& scene,
    const GlobalPositioningOptions& options = {});

}  // namespace aetherscan::sfm
