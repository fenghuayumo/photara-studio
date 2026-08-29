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
    // Prefer longer tracks; 4–5 cuts short noisy constraints on real scenes.
    unsigned min_views_per_track{4};
    // Coverage-aware selection scales with scene size while retaining a hard
    // ceiling for predictable solve time.
    unsigned min_tracks_for_positioning{2500};
    unsigned tracks_per_registered_image{20};
    unsigned max_tracks_for_positioning{20000};
    unsigned coverage_grid_size{4};
    unsigned max_irls_iterations{8};
    unsigned irls_inner_iterations{8};
    double irls_tuning_constant{4.685};
    double irls_min_weight{1e-3};
    unsigned irls_quarantine_after{2};
    double irls_weight_convergence{1e-2};
    unsigned max_num_iterations{100};
    // Hard wall-clock budget per Ceres attempt (seconds). 0 disables.
    double max_solver_time_sec{45.0};
    double function_tolerance{1e-5};
    double huber_threshold{0.1};
    unsigned random_seed{123};
    bool generate_random_positions{true};
    bool generate_random_points{true};
    bool generate_scales{true};
    bool optimize_positions{true};
    bool optimize_points{true};
    bool optimize_scales{true};
    // After camera-only warm-start, initialize points by multi-ray midpoints
    // instead of uniform random noise (much better conditioned).
    bool ray_initialize_points{true};
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
    unsigned irls_iterations{0};
    unsigned downweighted_constraints{0};
    unsigned quarantined_constraints{0};
    double final_residual{0.0};
    double median_residual{0.0};
    double p90_residual{0.0};
};

// Fixed-rotation global positioning. Jointly solves camera centers, 3D points,
// and one positive-depth scale per observation from X - C = scale * world_ray.
GlobalPositioningSummary solve_global_positions(
    Scene& scene,
    const GlobalPositioningOptions& options = {});

}  // namespace aetherscan::sfm
