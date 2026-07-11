#include "aetherscan/sfm/geometry.hpp"

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

bool triangulate_midpoint(
    const Mat3& R,
    const Vec3& C,
    const Vec3& b1,
    const Vec3& b2,
    Vec3& X) {
    // Camera1 at origin, Camera2 at C with rotation R (world->cam2 relative).
    // Rays: origin + a*b1 and C + b*(R^T b2)
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
        const double cos_a =
            (X1.normalized()).dot(((X - C).normalized()));
        if (cos_a > min_cos_angle) continue;  // angle too small
        ++good;
    }
    return good;
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
    if (sample.size() < 6) return false;  // DLT PnP needs >=6 for stability
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

    // Ensure points are in front of the camera.
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

    // Refine E on inliers
    std::vector<unsigned> inlier_ids;
    inlier_ids.reserve(best_inliers);
    for (unsigned i = 0; i < best_mask.size(); ++i) {
        if (best_mask[i]) inlier_ids.push_back(i);
    }
    if (inlier_ids.size() >= 8) best_E = estimate_essential_8pt(x1, x2, inlier_ids);

    Mat3 Rs[2];
    Vec3 t;
    decompose_essential(best_E, Rs, t);

    Mat3 best_R = Rs[0];
    Vec3 best_t = t;
    unsigned best_cheirality = 0;
    for (int ri = 0; ri < 2; ++ri) {
        for (double sign : {1.0, -1.0}) {
            const Vec3 tt = sign * t;
            const unsigned score =
                score_pose_cheirality(Rs[ri], tt, x1, x2, best_mask, min_cos);
            if (score > best_cheirality) {
                best_cheirality = score;
                best_R = Rs[ri];
                best_t = tt;
            }
        }
    }
    if (best_cheirality < options.min_inliers) return result;

    // Recompute inlier mask with angular reprojection of chosen pose
    const Pose3D pose = pose_from_essential_rt(best_R, best_t);
    const double cos_thresh =
        std::cos(0.5 * (camera1.pixel_error_to_angular(options.max_epipolar_error_px) +
                        camera2.pixel_error_to_angular(options.max_epipolar_error_px)));
    std::vector<char> final_mask(pixels1.size(), 0);
    unsigned final_inliers = 0;
    double angle_sum = 0.0;
    std::vector<Vec2> inlier_pixels;
    for (std::size_t i = 0; i < pixels1.size(); ++i) {
        if (!best_mask[i]) continue;
        Vec3 X;
        if (!triangulate_midpoint(pose.R, pose.C, x1[i].normalized(), x2[i].normalized(), X))
            continue;
        const Vec3 X1 = X;
        const Vec3 X2 = pose.transform_world_to_camera(X);
        if (X1.z() <= 0 || X2.z() <= 0) continue;
        const double c1 = x1[i].normalized().dot(X1.normalized());
        const double c2 = x2[i].normalized().dot(X2.normalized());
        if (c1 < cos_thresh || c2 < cos_thresh) continue;
        const double cos_a = X1.normalized().dot((X - pose.C).normalized());
        if (cos_a > min_cos) continue;
        final_mask[i] = 1;
        ++final_inliers;
        angle_sum += std::acos(std::clamp(cos_a, -1.0, 1.0));
        inlier_pixels.push_back(pixels1[i]);
    }
    if (final_inliers < options.min_inliers) return result;

    result.success = true;
    result.pose = pose;
    result.E = essential_from_pose(pose);
    result.F = camera2.K().transpose().inverse() * result.E * camera1.K().inverse();
    result.inlier_mask = std::move(final_mask);
    result.num_inliers = final_inliers;
    result.mean_ray_angle =
        static_cast<float>(angle_sum / static_cast<double>(final_inliers));
    result.weight_spatial = compute_spatial_weight(
        inlier_pixels, camera1.width, camera1.height);
    return result;
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

    // Refine on all inliers
    std::vector<unsigned> inlier_ids;
    for (unsigned i = 0; i < best_mask.size(); ++i) {
        if (best_mask[i]) inlier_ids.push_back(i);
    }
    Mat3 R = best_R;
    Vec3 t = best_t;
    if (inlier_ids.size() >= 6) solve_dlt_pnp(bearings, points_world, inlier_ids, R, t);

    // Final inlier recount
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
