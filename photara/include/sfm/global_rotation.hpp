#pragma once

#include "sfm/scene.hpp"

namespace photara::sfm {

struct GlobalRotationOptions {
    enum class WeightType {
        geman_mcclure,
        half_norm,
    };

    unsigned max_l1_iterations{5};
    unsigned max_irls_iterations{100};
    double step_convergence_threshold{1e-3};
    double irls_sigma_deg{5.0};
    double max_relative_rotation_error_deg{12.0};
    // Match OpenMVS/GLOMAP: use the composite view-graph weight so cycle
    // support and connectivity can override raw inlier count on repetitive
    // low-parallax walkthroughs.
    bool use_pair_weights{true};
    // Homography-dominant pairs carry ambiguous rotation on weak-parallax
    // scenes and should not seed the global rotation backbone.
    bool reject_planar_pairs{false};
    WeightType weight_type{WeightType::geman_mcclure};
};

struct GlobalRotationSummary {
    bool success{false};
    unsigned estimated_images{0};
    unsigned used_pairs{0};
    unsigned filtered_pairs{0};
    unsigned iterations{0};
    Index fixed_image{k_invalid};
};

// OpenMVS/GLOMAP global rotation averaging: maximum-spanning-tree
// initialization, LAD/ADMM L1 minimization, then robust IRLS.
GlobalRotationSummary estimate_global_rotations(
    Scene& scene,
    const GlobalRotationOptions& options = {});

}  // namespace photara::sfm
