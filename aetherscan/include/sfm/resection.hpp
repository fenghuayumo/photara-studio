#pragma once

#include "sfm/bundle.hpp"
#include "sfm/geometry.hpp"
#include "sfm/scene.hpp"

#include <array>
#include <functional>

namespace aetherscan::sfm {

struct ResectionConfig {
    unsigned min_correspondences{15};
    unsigned min_inliers{12};
    unsigned max_local_window{25};
    unsigned local_ba_every{10};
    // Parallel PnP wave size against a fixed triangulation snapshot.
    unsigned max_pose_wave{8};
    std::array<unsigned, 3> full_ba_every{25, 50, 100};
    float ratio_correspondences{0.3F};
    float avg_inliers_ratio_force_ba{0.6F};
    float max_reproj_error{4.F};
    float min_angle_deg{1.F};
    float mult_depth_near{0.05F};
    float mult_depth_far{20.F};
    AbsolutePoseOptions ransac{};
    ba::OptimizerOptions local_ba{};
    ba::OptimizerOptions full_ba{};
    // Called after a consistent registration/triangulation update.
    std::function<void(const Scene&)> checkpoint_callback;
    unsigned checkpoint_interval{1};

    ResectionConfig() {
        ransac.max_reproj_error_px = 4.0;
        ransac.confidence = 0.999;
        ransac.max_iterations = 10000;
        ransac.min_iterations = 100;
        ransac.min_inliers = 12;
        local_ba.maximum_iterations = 20;
        local_ba.huber_delta = 2.0;
        local_ba.optimize_focal = true;
        full_ba.maximum_iterations = 40;
        full_ba.huber_delta = 2.0;
        full_ba.optimize_focal = true;
        full_ba.optimize_distortion = true;
    }
};

// Incremental PnP registration (openMVS Resection).
unsigned register_images(Scene& scene, const ResectionConfig& config = {});

}  // namespace aetherscan::sfm
