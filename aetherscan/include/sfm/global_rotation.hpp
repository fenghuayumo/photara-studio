#pragma once

#include "sfm/scene.hpp"

namespace aetherscan::sfm {

struct GlobalRotationOptions {
    unsigned max_l1_iterations{5};
    unsigned max_irls_iterations{100};
    double step_convergence_threshold{1e-3};
    double irls_sigma_deg{5.0};
    double max_relative_rotation_error_deg{12.0};
    bool use_pair_weights{true};
};

struct GlobalRotationSummary {
    bool success{false};
    unsigned estimated_images{0};
    unsigned used_pairs{0};
    unsigned filtered_pairs{0};
    unsigned iterations{0};
    Index fixed_image{k_invalid};
};

// OpenMVS/GLOMAP-style global rotation averaging:
// maximum-spanning-tree initialization followed by tangent-space robust IRLS.
GlobalRotationSummary estimate_global_rotations(
    Scene& scene,
    const GlobalRotationOptions& options = {});

}  // namespace aetherscan::sfm
