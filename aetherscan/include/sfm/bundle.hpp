#pragma once

#include "ba/optimizer.hpp"
#include "sfm/scene.hpp"

namespace aetherscan::sfm {

struct BundleOptions {
    ba::OptimizerOptions optimizer{};
    bool optimize_points{true};
    // Write optimized shared intrinsics back into Scene::cameras.
    bool write_intrinsics{true};
    // If non-empty, only these image IDs have free poses; others are fixed
    // by excluding them from the BA problem (held constant outside).
    std::vector<Index> free_image_ids;
    // If true and free_image_ids empty, optimize all registered images.
    bool optimize_all_registered{true};
};

struct BundleSummary {
    bool success{false};
    ba::OptimizerSummary optimizer;
    unsigned num_cameras{0};
    unsigned num_points{0};
    unsigned num_observations{0};
};

// Builds a BA problem from triangulated inlier tracks and runs aetherscan::ba.
BundleSummary run_bundle_adjustment(Scene& scene, const BundleOptions& options = {});

}  // namespace aetherscan::sfm
