#pragma once

#include "sfm/scene.hpp"
#include "sfm/types.hpp"

#include <vector>

namespace aetherscan::sfm {

struct RelativePoseResult {
    bool success{false};
    Pose3D pose;
    Mat3 E{Mat3::Zero()};
    Mat3 F{Mat3::Zero()};
    std::vector<char> inlier_mask;
    unsigned num_inliers{0};
    float mean_ray_angle{0.F};
    float weight_spatial{0.F};
};

struct AbsolutePoseResult {
    bool success{false};
    Pose3D pose;
    std::vector<char> inlier_mask;
    unsigned num_inliers{0};
};

struct RelativePoseOptions {
    double max_epipolar_error_px{4.0};
    double min_ray_angle_deg{1.0};
    double confidence{0.999};
    unsigned max_iterations{2000};
    unsigned min_inliers{30};
};

struct AbsolutePoseOptions {
    double max_reproj_error_px{4.0};
    double confidence{0.999};
    unsigned max_iterations{2000};
    unsigned min_iterations{100};
    unsigned min_inliers{12};
};

// Calibrated essential-matrix RANSAC + cheirality pose selection.
RelativePoseResult estimate_relative_pose(
    const std::vector<Vec2>& pixels1,
    const std::vector<Vec2>& pixels2,
    const PinholeCamera& camera1,
    const PinholeCamera& camera2,
    const RelativePoseOptions& options = {});

// Bearing-vector absolute pose (DLT-PnP + angular RANSAC).
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
