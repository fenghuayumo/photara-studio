#include "sfm/geometry.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#if defined(AETHERSCAN_HAS_POSELIB)
#include <PoseLib/poselib.h>
#endif

namespace aetherscan::sfm {
namespace {

constexpr double k_pi = 3.14159265358979323846;

Mat3 skew(const Vec3& v) {
    Mat3 m;
    m << 0, -v.z(), v.y(),
        v.z(), 0, -v.x(),
        -v.y(), v.x(), 0;
    return m;
}

Mat3 normalize_essential(const Mat3& E) {
    Eigen::JacobiSVD<Mat3> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Vec3 singular = svd.singularValues();
    const double s = 0.5 * (singular(0) + singular(1));
    singular << s, s, 0.0;
    return svd.matrixU() * singular.asDiagonal() * svd.matrixV().transpose();
}

bool triangulate_midpoint(
    const Mat3& R,
    const Vec3& C,
    const Vec3& b1,
    const Vec3& b2,
    Vec3& X) {
    const Vec3 d1 = b1.normalized();
    const Vec3 d2 = (R.transpose() * b2).normalized();
    const Mat3 A = Mat3::Identity() - d1 * d1.transpose() + Mat3::Identity() -
                   d2 * d2.transpose();
    const Vec3 rhs = (Mat3::Identity() - d2 * d2.transpose()) * C;
    Eigen::LDLT<Mat3> ldlt(A);
    if (ldlt.info() != Eigen::Success) return false;
    X = ldlt.solve(rhs);
    return X.allFinite();
}

void decompose_essential(const Mat3& E, Mat3 Rs[2], Vec3& t) {
    Eigen::JacobiSVD<Mat3> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd.matrixU();
    Mat3 V = svd.matrixV();
    if (U.determinant() < 0) U.col(2) *= -1.0;
    if (V.determinant() < 0) V.col(2) *= -1.0;
    Mat3 W;
    W << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    Rs[0] = U * W * V.transpose();
    Rs[1] = U * W.transpose() * V.transpose();
    if (Rs[0].determinant() < 0) Rs[0] *= -1.0;
    if (Rs[1].determinant() < 0) Rs[1] *= -1.0;
    t = U.col(2);
    t.normalize();
}

unsigned score_pose_cheirality(
    const Mat3& R,
    const Vec3& t,
    const std::vector<Vec3>& x1,
    const std::vector<Vec3>& x2,
    const std::vector<char>& mask,
    double min_cos_angle) {
    const Vec3 C = -R.transpose() * t;
    unsigned good = 0;
    for (std::size_t i = 0; i < x1.size(); ++i) {
        if (!mask[i]) continue;
        Vec3 X;
        if (!triangulate_midpoint(R, C, x1[i], x2[i], X)) continue;
        const Vec3 X1 = X;
        const Vec3 X2 = R * (X - C);
        if (X1.z() <= 0 || X2.z() <= 0) continue;
        const double cos_a = X1.normalized().dot((X - C).normalized());
        if (cos_a > min_cos_angle) continue;
        ++good;
    }
    return good;
}

bool recover_pose_from_essential(
    const Mat3& E_raw,
    const std::vector<Vec3>& bearings1,
    const std::vector<Vec3>& bearings2,
    const std::vector<char>& seed_mask,
    const double min_cos_angle,
    const unsigned min_inliers,
    Pose3D& pose,
    std::vector<char>& refined_mask) {
    const Mat3 E = normalize_essential(E_raw);
    Mat3 Rs[2];
    Vec3 t;
    decompose_essential(E, Rs, t);
    Mat3 best_R = Rs[0];
    Vec3 best_t = t;
    unsigned best = 0;
    for (int ri = 0; ri < 2; ++ri) {
        for (double sign : {1.0, -1.0}) {
            const unsigned score =
                score_pose_cheirality(Rs[ri], sign * t, bearings1, bearings2, seed_mask, min_cos_angle);
            if (score > best) {
                best = score;
                best_R = Rs[ri];
                best_t = sign * t;
            }
        }
    }
    if (best < min_inliers) return false;
    pose = pose_from_essential_rt(best_R, best_t);
    refined_mask = seed_mask;
    return true;
}

// openMVS ImagePair::FilterMatches: epipole + angular reprojection + triangulation angle.
unsigned filter_matches(
    const std::vector<Vec2>& pixels1,
    const std::vector<Vec2>& pixels2,
    const PinholeCamera& camera1,
    const PinholeCamera& camera2,
    const Pose3D& pose,
    const RelativePoseOptions& options,
    std::vector<char>& mask,
    float& mean_ray_angle,
    std::vector<Vec2>* inlier_pixels) {
    const double min_cos_angle =
        std::cos(options.min_ray_angle_deg * k_pi / 180.0);
    const double cos_reproj =
        options.max_reproj_error_px > 0.0
            ? std::cos(
                  0.5 *
                  (camera1.pixel_error_to_angular(options.max_reproj_error_px) +
                   camera2.pixel_error_to_angular(options.max_reproj_error_px)))
            : -1.0;

    Vec3 epipole1 = Vec3::Zero();
    Vec3 epipole2 = Vec3::Zero();
    double cos_epipole = 2.0;
    if (options.epipole_filter_px > 0.0) {
        cos_epipole = std::cos(
            0.5 *
            (camera1.pixel_error_to_angular(options.epipole_filter_px) +
             camera2.pixel_error_to_angular(options.epipole_filter_px)));
        if (pose.C.norm() > 1e-12) epipole1 = pose.C.normalized();
        const Vec3 t = pose.translation();
        if (t.norm() > 1e-12) epipole2 = t.normalized();
    }

    double angle_weight_sum = 0.0;
    double cos_angle_accum = 0.0;
    unsigned num_inliers = 0;
    if (inlier_pixels) inlier_pixels->clear();

    for (std::size_t i = 0; i < pixels1.size(); ++i) {
        if (!mask[i]) continue;
        const Vec3 b1 = camera1.unproject_normalized(pixels1[i]);
        const Vec3 b2 = camera2.unproject_normalized(pixels2[i]);

        if (options.epipole_filter_px > 0.0 &&
            (b1.dot(epipole1) > cos_epipole || b2.dot(epipole2) > cos_epipole)) {
            mask[i] = 0;
            continue;
        }

        Vec3 X;
        if (!triangulate_midpoint(pose.R, pose.C, b1, b2, X)) {
            mask[i] = 0;
            continue;
        }
        const double n1 = X.norm();
        if (n1 < 1e-12) {
            mask[i] = 0;
            continue;
        }
        const Vec3 b1_proj = X / n1;
        const double cos_err1 = b1.dot(b1_proj);
        if (cos_err1 <= 0.0) {
            mask[i] = 0;
            continue;
        }

        const Vec3 X2 = pose.transform_world_to_camera(X);
        const double n2 = X2.norm();
        if (n2 < 1e-12) {
            mask[i] = 0;
            continue;
        }
        const Vec3 b2_proj = X2 / n2;
        const double cos_err2 = b2.dot(b2_proj);
        if (cos_err2 <= 0.0) {
            mask[i] = 0;
            continue;
        }

        if (options.max_reproj_error_px > 0.0 &&
            (cos_err1 < cos_reproj || cos_err2 < cos_reproj)) {
            mask[i] = 0;
            continue;
        }

        const Vec3 v1 = X;
        const Vec3 v2 = X - pose.C;
        const double cos_angle =
            v1.normalized().dot(v2.normalized());
        if (options.min_ray_angle_deg > 0.0 && cos_angle > min_cos_angle) {
            mask[i] = 0;
            continue;
        }

        const float one_minus_cos =
            static_cast<float>(1.0 - std::min(cos_err1, cos_err2));
        const float w = 1.F / std::max(one_minus_cos, 1e-10F);
        cos_angle_accum += cos_angle * w;
        angle_weight_sum += w;
        ++num_inliers;
        if (inlier_pixels) inlier_pixels->push_back(pixels1[i]);
    }

    mean_ray_angle =
        num_inliers > 0 && angle_weight_sum > 0.0
            ? static_cast<float>(
                  std::acos(std::clamp(cos_angle_accum / angle_weight_sum, -1.0, 1.0)))
            : 0.F;
    return num_inliers;
}

#if defined(AETHERSCAN_HAS_POSELIB)
bool has_distortion(const PinholeCamera& camera) {
    return camera.k1 != 0.0 || camera.k2 != 0.0 ||
           camera.p1 != 0.0 || camera.p2 != 0.0;
}

poselib::Camera to_poselib_camera(const PinholeCamera& camera) {
    if (has_distortion(camera)) {
        return poselib::Camera(
            "OPENCV",
            {camera.fx, camera.fy, camera.cx, camera.cy,
             camera.k1, camera.k2, camera.p1, camera.p2},
            static_cast<int>(camera.width),
            static_cast<int>(camera.height));
    }
    return poselib::Camera(
        "PINHOLE",
        {camera.fx, camera.fy, camera.cx, camera.cy},
        static_cast<int>(camera.width),
        static_cast<int>(camera.height));
}

void fill_poselib_points(
    const std::vector<Vec2>& pixels1,
    const std::vector<Vec2>& pixels2,
    std::vector<poselib::Point2D>& pts1,
    std::vector<poselib::Point2D>& pts2) {
    pts1.resize(pixels1.size());
    pts2.resize(pixels2.size());
    for (std::size_t i = 0; i < pixels1.size(); ++i) {
        pts1[i] = pixels1[i];
        pts2[i] = pixels2[i];
    }
}

void undistort_poselib_points(
    const std::vector<Vec2>& pixels,
    const PinholeCamera& camera,
    std::vector<poselib::Point2D>& points) {
    points.resize(pixels.size());
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        const Vec3 ray = camera.unproject(pixels[i]);
        points[i] = {
            camera.fx * ray.x() / ray.z() + camera.cx,
            camera.fy * ray.y() / ray.z() + camera.cy};
    }
}

bool estimate_with_poselib(
    const std::vector<Vec2>& pixels1,
    const std::vector<Vec2>& pixels2,
    const PinholeCamera& camera1,
    const PinholeCamera& camera2,
    const RelativePoseOptions& options,
    RelativePoseResult& result) {
    std::vector<poselib::Point2D> pts1, pts2;
    fill_poselib_points(pixels1, pixels2, pts1, pts2);
    std::vector<poselib::Point2D> pinhole_pts1 = pts1;
    std::vector<poselib::Point2D> pinhole_pts2 = pts2;
    if (has_distortion(camera1))
        undistort_poselib_points(pixels1, camera1, pinhole_pts1);
    if (has_distortion(camera2))
        undistort_poselib_points(pixels2, camera2, pinhole_pts2);

    poselib::RansacOptions ransac;
    ransac.max_iterations = options.max_iterations;
    ransac.min_iterations = options.min_iterations;
    ransac.success_prob = options.confidence;
    ransac.max_epipolar_error = options.max_epipolar_error_px;
    ransac.max_reproj_error = options.max_reproj_error_px;

    poselib::BundleOptions bundle;
    bundle.max_iterations = 50;
    bundle.loss_type = poselib::BundleOptions::LossType::HUBER;
    bundle.loss_scale = options.max_epipolar_error_px;

    std::vector<char> inliers;
    Pose3D pose;
    Mat3 E = Mat3::Zero();
    Mat3 F = Mat3::Zero();
    bool have_pose = false;

    const bool shared_camera =
        camera1.width == camera2.width && camera1.height == camera2.height &&
        std::abs(camera1.fx - camera2.fx) < 1e-6 &&
        std::abs(camera1.fy - camera2.fy) < 1e-6 &&
        std::abs(camera1.cx - camera2.cx) < 1e-6 &&
        std::abs(camera1.cy - camera2.cy) < 1e-6 &&
        std::abs(camera1.k1 - camera2.k1) < 1e-12 &&
        std::abs(camera1.k2 - camera2.k2) < 1e-12 &&
        std::abs(camera1.p1 - camera2.p1) < 1e-12 &&
        std::abs(camera1.p2 - camera2.p2) < 1e-12;

    const bool use_shared_focal =
        options.force_shared_focal ||
        (!camera1.trust_intrinsics && !camera2.trust_intrinsics && shared_camera);
    const bool use_calibrated =
        !options.force_fundamental && !use_shared_focal &&
        camera1.trust_intrinsics && camera2.trust_intrinsics;

    if (use_shared_focal) {
        poselib::ImagePair image_pair;
        const poselib::Point2D pp(camera1.cx, camera1.cy);
        const poselib::RansacStats stats = poselib::estimate_shared_focal_relative_pose(
            pinhole_pts1, pinhole_pts2, pp, ransac, bundle, &image_pair, &inliers);
        if (stats.num_inliers < options.min_inliers) return false;
        pose.set_from_rt(image_pair.pose.R(), image_pair.pose.t);
        E = essential_from_pose(pose);
        // Approximate F with the estimated shared focal camera.
        const double f = image_pair.camera1.focal();
        Mat3 K = Mat3::Identity();
        K(0, 0) = K(1, 1) = f;
        K(0, 2) = camera1.cx;
        K(1, 2) = camera1.cy;
        F = K.transpose().inverse() * E * K.inverse();
        have_pose = true;
    } else if (use_calibrated) {
        poselib::CameraPose pl_pose;
        const poselib::RansacStats stats = poselib::estimate_relative_pose(
            pts1, pts2, to_poselib_camera(camera1), to_poselib_camera(camera2),
            ransac, bundle, &pl_pose, &inliers);
        if (stats.num_inliers < options.min_inliers) return false;
        pose.set_from_rt(pl_pose.R(), pl_pose.t);
        E = essential_from_pose(pose);
        F = camera2.K().transpose().inverse() * E * camera1.K().inverse();
        have_pose = true;
    } else {
        // Uncalibrated: Fundamental matrix.
        const poselib::RansacStats stats =
            poselib::estimate_fundamental(
                pinhole_pts1, pinhole_pts2, ransac, bundle, &F, &inliers);
        if (stats.num_inliers < options.min_inliers) return false;

        if (options.decompose_fundamental &&
            (camera1.trust_intrinsics || options.force_fundamental)) {
            E = normalize_essential(
                camera2.K().transpose() * F * camera1.K());
            std::vector<Vec3> b1(pixels1.size()), b2(pixels2.size());
            for (std::size_t i = 0; i < pixels1.size(); ++i) {
                b1[i] = camera1.unproject(pixels1[i]);
                b2[i] = camera2.unproject(pixels2[i]);
            }
            const double min_cos =
                std::cos(options.min_ray_angle_deg * k_pi / 180.0);
            std::vector<char> refined;
            if (!recover_pose_from_essential(
                    E, b1, b2, inliers, min_cos, options.min_inliers, pose, refined)) {
                // Keep F-only pair without pose — not useful for incremental init.
                return false;
            }
            inliers = std::move(refined);
            E = essential_from_pose(pose);
            have_pose = true;
        } else {
            return false;
        }
    }

    if (!have_pose) return false;

    result.num_ransac_inliers =
        static_cast<unsigned>(std::count(inliers.begin(), inliers.end(), char{1}));
    float mean_angle = 0.F;
    std::vector<Vec2> inlier_pixels;
    const unsigned filtered = filter_matches(
        pixels1, pixels2, camera1, camera2, pose, options, inliers, mean_angle,
        &inlier_pixels);
    result.num_inliers = filtered;
    if (filtered < options.min_inliers) return false;

    result.success = true;
    result.pose = pose;
    result.E = E;
    result.F = F;
    result.inlier_mask = std::move(inliers);
    result.mean_ray_angle = mean_angle;
    result.weight_spatial =
        compute_spatial_weight(inlier_pixels, camera1.width, camera1.height);

    if (options.estimate_homography) {
        Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
        std::vector<char> h_inliers;
        poselib::RansacOptions h_ransac = ransac;
        h_ransac.max_reproj_error = options.max_reproj_error_px;
        const poselib::RansacStats h_stats = poselib::estimate_homography(
            pinhole_pts1, pinhole_pts2, h_ransac, bundle, &H, &h_inliers);
        // Count H inliers among geometric (E/F filtered) inliers only.
        unsigned h_among_e = 0;
        for (std::size_t i = 0; i < result.inlier_mask.size(); ++i) {
            if (result.inlier_mask[i] && i < h_inliers.size() && h_inliers[i]) ++h_among_e;
        }
        result.num_homography_inliers = h_among_e;
        result.H = H;
        result.homography_ratio =
            result.num_inliers > 0
                ? static_cast<float>(h_among_e) / static_cast<float>(result.num_inliers)
                : 0.F;
        // Also consider raw H/E ratio from RANSAC counts as fallback signal.
        const float raw_ratio =
            result.num_inliers > 0
                ? static_cast<float>(h_stats.num_inliers) /
                      static_cast<float>(result.num_inliers)
                : 0.F;
        result.homography_ratio = std::max(result.homography_ratio, std::min(raw_ratio, 1.F));
        result.degenerate_planar =
            result.homography_ratio >= static_cast<float>(options.homography_degeneracy_ratio);
    }
    return true;
}
#endif

#if !defined(AETHERSCAN_HAS_POSELIB)
Mat3 estimate_essential_8pt(
    const std::vector<Vec3>& x1,
    const std::vector<Vec3>& x2,
    const std::vector<unsigned>& sample) {
    Eigen::Matrix<double, Eigen::Dynamic, 9> A(sample.size(), 9);
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(sample.size()); ++i) {
        const Vec3& a = x1[sample[static_cast<std::size_t>(i)]];
        const Vec3& b = x2[sample[static_cast<std::size_t>(i)]];
        A.row(i) << b.x() * a.x(), b.x() * a.y(), b.x() * a.z(),
            b.y() * a.x(), b.y() * a.y(), b.y() * a.z(),
            b.z() * a.x(), b.z() * a.y(), b.z() * a.z();
    }
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
        A, Eigen::ComputeFullV);
    Eigen::Matrix<double, 9, 1> e = svd.matrixV().col(8);
    Mat3 E;
    E << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);
    return normalize_essential(E);
}

double sampson_error(const Mat3& E, const Vec3& x1, const Vec3& x2) {
    const Vec3 Ex1 = E * x1;
    const Vec3 Etx2 = E.transpose() * x2;
    const double x2tEx1 = x2.dot(Ex1);
    const double denom =
        Ex1.head<2>().squaredNorm() + Etx2.head<2>().squaredNorm();
    if (denom < 1e-12) return 1e6;
    return std::abs(x2tEx1) / std::sqrt(denom);
}

unsigned ransac_iterations(double inlier_ratio, double confidence, unsigned sample_size) {
    if (inlier_ratio <= 0.0) return 100000;
    const double w = std::pow(inlier_ratio, sample_size);
    if (w >= 1.0) return 1;
    const double num = std::log(std::max(1e-12, 1.0 - confidence));
    const double den = std::log(std::max(1e-12, 1.0 - w));
    if (den >= 0.0) return 100000;
    return static_cast<unsigned>(std::min(100000.0, std::ceil(num / den)));
}
#endif

bool bearing_to_plane(const Vec3& bearing, double& u, double& v) {
    const Vec3 b = bearing.normalized();
    if (std::abs(b.z()) < 1e-8) return false;
    u = b.x() / b.z();
    v = b.y() / b.z();
    return std::isfinite(u) && std::isfinite(v);
}

bool solve_dlt_pnp(
    const std::vector<Vec3>& bearings,
    const std::vector<Vec3>& points,
    const std::vector<unsigned>& sample,
    Mat3& R,
    Vec3& t) {
    if (sample.size() < 6) return false;
    Eigen::MatrixXd A(2 * static_cast<Eigen::Index>(sample.size()), 12);
    A.setZero();
    Eigen::Index row = 0;
    for (unsigned idx : sample) {
        double u = 0, v = 0;
        if (!bearing_to_plane(bearings[idx], u, v)) continue;
        const Vec3& X = points[idx];
        A.row(row++) << X.x(), X.y(), X.z(), 1, 0, 0, 0, 0, -u * X.x(), -u * X.y(),
            -u * X.z(), -u;
        A.row(row++) << 0, 0, 0, 0, X.x(), X.y(), X.z(), 1, -v * X.x(), -v * X.y(),
            -v * X.z(), -v;
    }
    if (row < 12) return false;
    A.conservativeResize(row, 12);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::Matrix<double, 12, 1> p = svd.matrixV().col(11);
    Mat34 P;
    P.row(0) = p.segment<4>(0);
    P.row(1) = p.segment<4>(4);
    P.row(2) = p.segment<4>(8);

    Mat3 M = P.leftCols<3>();
    Eigen::JacobiSVD<Mat3> svdM(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svdM.matrixU() * svdM.matrixV().transpose();
    if (R.determinant() < 0) {
        Mat3 U = svdM.matrixU();
        U.col(2) *= -1.0;
        R = U * svdM.matrixV().transpose();
    }
    const double scale = (svdM.singularValues()(0) + svdM.singularValues()(1) +
                          svdM.singularValues()(2)) /
                         3.0;
    if (std::abs(scale) < 1e-12) return false;
    t = P.col(3) / scale;

    unsigned front = 0;
    for (unsigned idx : sample) {
        if ((R * points[idx] + t).z() > 0) ++front;
    }
    if (front * 2 < sample.size()) {
        R *= -1.0;
        t *= -1.0;
        if (R.determinant() < 0) R.col(2) *= -1.0;
    }
    return R.allFinite() && t.allFinite() && std::abs(R.determinant() - 1.0) < 1e-2;
}

}  // namespace

Mat3 essential_from_pose(const Pose3D& pose) {
    return skew(pose.translation()) * pose.R;
}

Pose3D pose_from_essential_rt(const Mat3& R, const Vec3& t) {
    Pose3D pose;
    pose.set_from_rt(R, t.normalized());
    return pose;
}

float compute_spatial_weight(
    const std::vector<Vec2>& pixels,
    const std::uint32_t width,
    const std::uint32_t height,
    const int grid) {
    if (pixels.empty() || width == 0 || height == 0 || grid <= 0) return 0.F;
    std::vector<char> occupied(static_cast<std::size_t>(grid * grid), 0);
    unsigned filled = 0;
    for (const Vec2& p : pixels) {
        const int gx = std::clamp(
            static_cast<int>(p.x() / width * grid), 0, grid - 1);
        const int gy = std::clamp(
            static_cast<int>(p.y() / height * grid), 0, grid - 1);
        const std::size_t idx = static_cast<std::size_t>(gy * grid + gx);
        if (!occupied[idx]) {
            occupied[idx] = 1;
            ++filled;
        }
    }
    return static_cast<float>(filled) / static_cast<float>(grid * grid);
}

RelativePoseResult estimate_relative_pose(
    const std::vector<Vec2>& pixels1,
    const std::vector<Vec2>& pixels2,
    const PinholeCamera& camera1,
    const PinholeCamera& camera2,
    const RelativePoseOptions& options) {
    RelativePoseResult result;
    if (pixels1.size() != pixels2.size() || pixels1.size() < 8) return result;

#if defined(AETHERSCAN_HAS_POSELIB)
    if (estimate_with_poselib(pixels1, pixels2, camera1, camera2, options, result))
        return result;
    return result;
#else
    std::vector<Vec3> x1(pixels1.size()), x2(pixels2.size());
    for (std::size_t i = 0; i < pixels1.size(); ++i) {
        x1[i] = camera1.unproject(pixels1[i]);
        x2[i] = camera2.unproject(pixels2[i]);
    }

    const double focal = 0.5 * (camera1.focal() + camera2.focal());
    const double thresh = options.max_epipolar_error_px / std::max(focal, 1.0);
    const double min_cos =
        std::cos(options.min_ray_angle_deg * k_pi / 180.0);

    std::mt19937 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, pixels1.size() - 1);
    std::vector<char> best_mask(pixels1.size(), 0);
    Mat3 best_E = Mat3::Zero();
    unsigned best_inliers = 0;
    unsigned max_iters = options.max_iterations;

    for (unsigned iter = 0; iter < max_iters; ++iter) {
        std::vector<unsigned> sample;
        sample.reserve(8);
        while (sample.size() < 8) {
            const unsigned idx = static_cast<unsigned>(dist(rng));
            if (std::find(sample.begin(), sample.end(), idx) == sample.end())
                sample.push_back(idx);
        }
        const Mat3 E = estimate_essential_8pt(x1, x2, sample);
        std::vector<char> mask(pixels1.size(), 0);
        unsigned inliers = 0;
        for (std::size_t i = 0; i < pixels1.size(); ++i) {
            if (sampson_error(E, x1[i], x2[i]) < thresh) {
                mask[i] = 1;
                ++inliers;
            }
        }
        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_mask = mask;
            best_E = E;
            const double ratio =
                static_cast<double>(inliers) / static_cast<double>(pixels1.size());
            max_iters = std::max(100u, ransac_iterations(ratio, options.confidence, 8));
            max_iters = std::min(max_iters, options.max_iterations);
        }
    }

    if (best_inliers < options.min_inliers) return result;

    std::vector<unsigned> inlier_ids;
    for (unsigned i = 0; i < best_mask.size(); ++i) {
        if (best_mask[i]) inlier_ids.push_back(i);
    }
    if (inlier_ids.size() >= 8) best_E = estimate_essential_8pt(x1, x2, inlier_ids);

    Pose3D pose;
    std::vector<char> refined = best_mask;
    if (!recover_pose_from_essential(
            best_E, x1, x2, best_mask, min_cos, options.min_inliers, pose, refined))
        return result;

    result.num_ransac_inliers =
        static_cast<unsigned>(std::count(refined.begin(), refined.end(), char{1}));
    float mean_angle = 0.F;
    std::vector<Vec2> inlier_pixels;
    const unsigned filtered = filter_matches(
        pixels1, pixels2, camera1, camera2, pose, options, refined, mean_angle,
        &inlier_pixels);
    result.num_inliers = filtered;
    if (filtered < options.min_inliers) return result;

    result.success = true;
    result.pose = pose;
    result.E = essential_from_pose(pose);
    result.F = camera2.K().transpose().inverse() * result.E * camera1.K().inverse();
    result.inlier_mask = std::move(refined);
    result.mean_ray_angle = mean_angle;
    result.weight_spatial =
        compute_spatial_weight(inlier_pixels, camera1.width, camera1.height);
    return result;
#endif
}

AbsolutePoseResult estimate_absolute_pose(
    const std::vector<Vec3>& bearings,
    const std::vector<Vec3>& points_world,
    const PinholeCamera& camera,
    const AbsolutePoseOptions& options) {
    AbsolutePoseResult result;
    if (bearings.size() != points_world.size() || bearings.size() < 4) return result;

#if defined(AETHERSCAN_HAS_POSELIB)
    std::vector<poselib::Point2D> pixels;
    std::vector<poselib::Point3D> points;
    pixels.reserve(bearings.size());
    points.reserve(points_world.size());
    for (std::size_t i = 0; i < bearings.size(); ++i) {
        const Vec3 bearing = bearings[i].normalized();
        if (bearing.z() <= 1e-8) return result;
        pixels.emplace_back(
            camera.fx * bearing.x() / bearing.z() + camera.cx,
            camera.fy * bearing.y() / bearing.z() + camera.cy);
        points.emplace_back(points_world[i]);
    }

    const poselib::Camera pose_camera(
        "PINHOLE",
        {camera.fx, camera.fy, camera.cx, camera.cy},
        static_cast<int>(camera.width),
        static_cast<int>(camera.height));
    poselib::RansacOptions ransac;
    ransac.max_iterations = options.max_iterations;
    ransac.min_iterations = options.min_iterations;
    ransac.success_prob = options.confidence;
    ransac.max_reproj_error = options.max_reproj_error_px;
    poselib::BundleOptions bundle;
    bundle.max_iterations = 50;
    bundle.loss_type = poselib::BundleOptions::LossType::HUBER;
    bundle.loss_scale = options.max_reproj_error_px;

    poselib::CameraPose pose;
    std::vector<char> inliers;
    const poselib::RansacStats stats = poselib::estimate_absolute_pose(
        pixels, points, pose_camera, ransac, bundle, &pose, &inliers);
    if (stats.num_inliers < options.min_inliers) return result;

    result.success = true;
    result.pose.set_from_rt(pose.R(), pose.t);
    result.inlier_mask = std::move(inliers);
    result.num_inliers = static_cast<unsigned>(stats.num_inliers);
    return result;
#else
    const double cos_thresh = std::cos(camera.pixel_error_to_angular(options.max_reproj_error_px));
    std::mt19937 rng(7);
    std::uniform_int_distribution<std::size_t> dist(0, bearings.size() - 1);

    std::vector<char> best_mask(bearings.size(), 0);
    Mat3 best_R = Mat3::Identity();
    Vec3 best_t = Vec3::Zero();
    unsigned best_inliers = 0;
    unsigned max_iters = options.max_iterations;

    constexpr unsigned k_sample_size = 6;
    for (unsigned iter = 0; iter < max_iters; ++iter) {
        std::vector<unsigned> sample;
        while (sample.size() < k_sample_size) {
            const unsigned idx = static_cast<unsigned>(dist(rng));
            if (std::find(sample.begin(), sample.end(), idx) == sample.end())
                sample.push_back(idx);
        }
        Mat3 R;
        Vec3 t;
        if (!solve_dlt_pnp(bearings, points_world, sample, R, t)) continue;

        std::vector<char> mask(bearings.size(), 0);
        unsigned inliers = 0;
        for (std::size_t i = 0; i < bearings.size(); ++i) {
            const Vec3 Xc = R * points_world[i] + t;
            if (Xc.z() <= 1e-8) continue;
            const double c = bearings[i].normalized().dot(Xc.normalized());
            if (c >= cos_thresh) {
                mask[i] = 1;
                ++inliers;
            }
        }
        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_mask = mask;
            best_R = R;
            best_t = t;
            const double ratio =
                static_cast<double>(inliers) / static_cast<double>(bearings.size());
            max_iters = std::max(
                options.min_iterations,
                ransac_iterations(ratio, options.confidence, k_sample_size));
            max_iters = std::min(max_iters, options.max_iterations);
        }
    }

    if (best_inliers < options.min_inliers) return result;

    std::vector<unsigned> inlier_ids;
    for (unsigned i = 0; i < best_mask.size(); ++i) {
        if (best_mask[i]) inlier_ids.push_back(i);
    }
    Mat3 R = best_R;
    Vec3 t = best_t;
    if (inlier_ids.size() >= 6) solve_dlt_pnp(bearings, points_world, inlier_ids, R, t);

    unsigned final_inliers = 0;
    for (std::size_t i = 0; i < bearings.size(); ++i) {
        const Vec3 Xc = R * points_world[i] + t;
        if (Xc.norm() < 1e-12) {
            best_mask[i] = 0;
            continue;
        }
        const double c = bearings[i].normalized().dot(Xc.normalized());
        if (c >= cos_thresh) {
            best_mask[i] = 1;
            ++final_inliers;
        } else {
            best_mask[i] = 0;
        }
    }
    if (final_inliers < options.min_inliers) return result;

    result.success = true;
    result.pose.set_from_rt(R, t);
    result.inlier_mask = std::move(best_mask);
    result.num_inliers = final_inliers;
    return result;
#endif
}

}  // namespace aetherscan::sfm
