#pragma once

#include "sfm/scene.hpp"

namespace photara::sfm {

enum class GlobalPositioningConstraint {
    only_points,
    only_cameras,
    points_and_cameras_balanced,
    points_and_cameras,
};

// automatic prefers a GPU: CUDA when prefer_cuda and a device are present,
// otherwise Vulkan, then CPU. Explicit vulkan/cuda do not fall through to
// the other GPU.
enum class PositioningBackend {
    automatic,
    cpu,
    cuda,
    vulkan,
};

struct GlobalPositioningOptions {
    // Dense camera Schur system for the fixed-rotation bearing solve.
    // Camera-only initialization and unsupported sizes retain the CPU path.
    // prefer_cuda applies only when backend is automatic.
    bool prefer_cuda{true};
    PositioningBackend backend{PositioningBackend::automatic};
    // Prefer longer tracks; 4–5 cuts short noisy constraints on real scenes
    // and avoids unconstrained short tracks in the selected subset.
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
    // Shared time budget per positioning attempt. Ceres checks between steps,
    // so a step may overrun it. 0 disables this shared limit, not Solve's cap.
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
    // Initialize camera centers from robust relative-translation constraints
    // before point-bearing refinement. This avoids random point-only local
    // minima on large weakly connected scenes.
    bool camera_warm_start_first{true};
    unsigned camera_warm_start_min_images{512};
    GlobalPositioningConstraint constraint{
        GlobalPositioningConstraint::only_points};
    double constraint_reweight_scale{1.0};
    // Experimental, opt-in prior for ordered captures with a known uniform
    // circular trajectory. It projects centers onto an equal-angle circle and
    // locks them during final BA. A complete orbit alone does not establish
    // uniform motion: keep measured geometry free by default, including for
    // handheld or nonuniform turntable sequences.
    bool regularize_complete_orbits{false};
};

enum class GlobalPositioningTermination {
    not_run, irls_converged, iteration_limit, time_limit, stalled, failed,
    camera_only_fallback,
};

inline const char* positioning_termination_name(GlobalPositioningTermination termination) {
    switch (termination) {
    case GlobalPositioningTermination::not_run: return "not_run";
    case GlobalPositioningTermination::irls_converged: return "irls_converged";
    case GlobalPositioningTermination::iteration_limit: return "iteration_limit";
    case GlobalPositioningTermination::time_limit: return "time_limit";
    case GlobalPositioningTermination::stalled: return "stalled";
    case GlobalPositioningTermination::failed: return "failed";
    case GlobalPositioningTermination::camera_only_fallback: return "camera_only_fallback";
    }
    return "unknown";
}

// success describes a usable solution, not convergence. Termination describes
// the final positioning attempt; CPU counters exclude evaluation iteration zero.
struct GlobalPositioningSummary {
    bool success{false};
    GlobalPositioningTermination termination{GlobalPositioningTermination::not_run};
    unsigned cpu_accepted_updates{};
    unsigned cpu_time_limited_passes{};
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

}  // namespace photara::sfm
