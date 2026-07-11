#pragma once

#include "aetherscan/sfm/scene.hpp"

namespace aetherscan::sfm {

struct GlobalPositioningOptions {
    unsigned min_views_per_track{3};
    unsigned max_irls_iterations{8};
    unsigned max_linear_iterations{300};
    double linear_tolerance{1e-7};
    double huber_threshold{0.1};
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
