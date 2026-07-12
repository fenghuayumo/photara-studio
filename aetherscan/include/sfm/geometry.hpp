#pragma once

#include "sfm/scene.hpp"
#include "sfm/types.hpp"

#include <optional>
#include <vector>

namespace aetherscan::sfm {

struct RelativePoseResult {
    bool success{false};
    Pose3D pose;
    Mat3 E{Mat3::Zero()};
    Mat3 F{Mat3::Zero()};
    std::optional<Mat3> H;
    std::vector<char> inlier_mask;
    unsigned num_ransac_inliers{0};  // Before cheirality/reprojection/angle filtering.
    unsigned num_inliers{0};
    unsigned num_homography_inliers{0};
    float mean_ray_angle{0.F};  // radians
    float weight_spatial{0.F};
    // H_inliers / E_inliers after FilterMatches; high => low parallax / planar.
    float homography_ratio{0.F};
    bool degenerate_planar{false};
};

struct AbsolutePoseResult {
    bool success{false};
    Pose3D pose;
    std::vector<char> inlier_mask;
    unsigned num_inliers{0};
};

struct RelativePoseOptions {
    double max_epipolar_error_px{4.0};
    // FilterMatches angular reprojection threshold (pixels → radians per camera).
    double max_reproj_error_px{6.0};
    double min_ray_angle_deg{0.5};
    // Matches closer than this (px) to an epipole are rejected; 0 disables.
    double epipole_filter_px{0.0};
    double confidence{0.999};
    unsigned max_iterations{10000};
    unsigned min_iterations{100};
    unsigned min_inliers{30};

    // openMVS GeometricFilter branches:
    //   trusted intrinsics → Essential / relative pose
    //   force_fundamental / !trust → Fundamental (+ optional E decomposition)
    //   force_shared_focal → shared-focal relative pose
    bool force_fundamental{false};
    bool force_shared_focal{false};
    bool decompose_fundamental{true};

    // Homography degeneracy: if H_inliers/E_inliers >= ratio, mark planar.
    bool estimate_homography{true};
    double homography_degeneracy_ratio{0.80};
    // Init weight multiplier when planar-degenerate (tracks still keep the pair).
    float degenerate_weight_scale{0.05F};
};

struct AbsolutePoseOptions {
    double max_reproj_error_px{4.0};
    double confidence{0.999};
    unsigned max_iterations{2000};
    unsigned min_iterations{100};
    unsigned min_inliers{12};
};

// PoseLib GeometricFilter-style relative pose (E / F / shared-focal) + FilterMatches
// + optional H degeneracy test. Falls back to 8-point Essential without PoseLib.
RelativePoseResult estimate_relative_pose(
    const std::vector<Vec2>& pixels1,
    const std::vector<Vec2>& pixels2,
    const PinholeCamera& camera1,
    const PinholeCamera& camera2,
    const RelativePoseOptions& options = {});

// Bearing-vector absolute pose (PoseLib EPnP when available).
AbsolutePoseResult estimate_absolute_pose(
    const std::vector<Vec3>& bearings,
    const std::vector<Vec3>& points_world,
    const PinholeCamera& camera,
    const AbsolutePoseOptions& options = {});

Mat3 essential_from_pose(const Pose3D& pose);
Pose3D pose_from_essential_rt(const Mat3& R, const Vec3& t);

float compute_spatial_weight(
    const std::vector<Vec2>& pixels,
    std::uint32_t width,
    std::uint32_t height,
    int grid = 8);

}  // namespace aetherscan::sfm
