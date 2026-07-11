#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aetherscan::geometry {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Mat34 = Eigen::Matrix<double, 3, 4>;

/// Rotation matrix from Rodrigues axis-angle vector.
Mat3 rodrigues(const Vec3& rvec);

/// Axis-angle vector from rotation matrix.
Vec3 inverse_rodrigues(const Mat3& R);

/// 8-point essential matrix on normalized image coordinates; projects to
/// essential manifold via SVD with singular values (1, 1, 0).
Mat3 essential_from_normalized_8point(
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    const std::vector<std::size_t>& sample);

/// Classic 8-point fundamental matrix (pixel coordinates), rank-2 enforced.
Mat3 fundamental_from_8point(
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    const std::vector<std::size_t>& sample);

/// Direct Linear Transform homography from (at least) 4 point correspondences.
Mat3 homography_from_4point(
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    const std::vector<std::size_t>& sample);

struct RansacResult {
    Mat3 model = Mat3::Zero();
    std::vector<std::uint8_t> inliers;
    std::size_t inlier_count{0};
    bool valid{false};
};

RansacResult estimate_essential_ransac(
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    double threshold_normalized,
    double confidence = 0.999,
    std::size_t max_iters = 10000);

RansacResult estimate_fundamental_ransac(
    const std::vector<Vec2>& p1_pixels,
    const std::vector<Vec2>& p2_pixels,
    double threshold_pixels,
    double confidence = 0.999,
    std::size_t max_iters = 10000);

RansacResult estimate_homography_ransac(
    const std::vector<Vec2>& p1_pixels,
    const std::vector<Vec2>& p2_pixels,
    double threshold_pixels,
    double confidence = 0.999,
    std::size_t max_iters = 10000);

struct RelativePose {
    Mat3 R = Mat3::Identity();
    Vec3 t = Vec3::Zero();
    std::size_t cheirality_count{0};
    bool valid{false};
};

/// Decompose E into four poses and pick the best by positive-depth count.
/// Updates inlier_mask in place to clear points that fail cheirality when
/// a valid pose is found (mask size must match p1/p2 if non-empty).
RelativePose recover_pose(
    const Mat3& E,
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    std::vector<std::uint8_t>& inlier_mask);

struct AbsolutePose {
    Mat3 R = Mat3::Identity();
    Vec3 t = Vec3::Zero();  ///< world to camera: Xc = R Xw + t
    bool valid{false};
};

/// DLT PnP + RANSAC for >= 6 correspondences.
/// If K is provided (fx,fy,cx,cy all > 0 via non-null), image_points are pixels
/// and threshold is in pixels; otherwise image_points are already normalized
/// and threshold is interpreted in the same units.
AbsolutePose estimate_pnp_ransac(
    const std::vector<Vec3>& object_points,
    const std::vector<Vec2>& image_points,
    double fx,
    double fy,
    double cx,
    double cy,
    double threshold_pixels,
    double confidence = 0.999,
    std::size_t max_iters = 10000,
    bool use_pixel_and_K = true);

/// Levenberg-Marquardt refinement of PnP with Rodrigues parameterization.
AbsolutePose refine_pnp_lm(
    const AbsolutePose& initial,
    const std::vector<Vec3>& object_points,
    const std::vector<Vec2>& image_points,
    double fx,
    double fy,
    double cx,
    double cy,
    const std::vector<std::uint8_t>* inlier_mask = nullptr,
    std::size_t max_iters = 50);

Vec2 project_point(
    const Mat3& R,
    const Vec3& t,
    const Vec3& X,
    double fx,
    double fy,
    double cx,
    double cy);

/// Homogeneous multi-view DLT triangulation via JacobiSVD.
bool triangulate_dlt(
    const std::vector<Mat34>& projections,
    const std::vector<Vec2>& observations,
    Vec3& X);

}  // namespace aetherscan::geometry
