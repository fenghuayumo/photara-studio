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
    // Scheduled full BA starts with a cheap probe and only spends the
    // remaining iteration budget while the probe tail is still improving.
    // Weak-registration rescue BA bypasses the probe and always runs fully.
    unsigned periodic_full_ba_probe_iterations{10};
    unsigned periodic_full_ba_tail_window{4};
    double periodic_full_ba_tail_relative_improvement{1e-3};
    // Continue the final full BA only while its tail is still making useful
    // progress. This recovers hard scenes without paying 100 iterations on
    // captures that converge in the default 40.
    unsigned final_ba_additional_iterations{60};
    unsigned final_ba_tail_window{8};
    double final_ba_tail_relative_improvement{1e-3};
    // Do not turn a short burst of weak registrations into a full-BA storm.
    unsigned min_force_full_ba_samples{5};
    unsigned min_force_full_ba_interval{10};
    float ratio_correspondences{0.3F};
    float avg_inliers_ratio_force_ba{0.6F};
    float min_inlier_ratio{0.35F};
    unsigned inlier_grid_size{4};
    unsigned min_inlier_grid_cells{4};
    // Strong, spatially distributed 2D-3D support is more reliable than the
    // translation direction of low-parallax two-view edges.
    float consistency_bypass_inlier_ratio{0.5F};
    float min_consistency_pair_weight{3.F};
    unsigned min_rotation_consistency_neighbors{2};
    float max_median_rotation_error_deg{12.F};
    unsigned min_translation_consistency_neighbors{2};
    float max_median_translation_error_deg{45.F};
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
        // Match openMVS local BA: keep intrinsics fixed while resecting.
        local_ba.optimize_focal = false;
        local_ba.optimize_distortion = false;
        full_ba.maximum_iterations = 40;
        full_ba.huber_delta = 2.0;
        full_ba.optimize_focal = true;
        full_ba.optimize_aspect_ratio = true;
        full_ba.optimize_distortion = true;
    }
};

// Incremental PnP registration (openMVS Resection).
unsigned register_images(Scene& scene, const ResectionConfig& config = {});

}  // namespace aetherscan::sfm
