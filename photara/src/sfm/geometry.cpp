#include "sfm/geometry.hpp"
#include "core/logging.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

#if defined(PHOTARA_HAS_POSELIB)
#include <PoseLib/poselib.h>
#endif

namespace photara::sfm {
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


#if defined(PHOTARA_HAS_POSELIB)
bool has_distortion(const PinholeCamera& camera) {
    return camera.model == CameraModel::opencv_fisheye ||
           camera.k1 != 0.0 || camera.k2 != 0.0 ||
           camera.p1 != 0.0 || camera.p2 != 0.0;
}

poselib::Camera to_poselib_camera(const PinholeCamera& camera) {
    if (has_distortion(camera)) {
        return poselib::Camera(
            camera.model == CameraModel::opencv_fisheye ? "OPENCV_FISHEYE" : "OPENCV",
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

double shared_focal_essential_score(
    const Mat3& F, const double cx, const double cy, const double focal) {
    Mat3 K = Mat3::Identity();
    K(0, 0) = K(1, 1) = focal;
    K(0, 2) = cx;
    K(1, 2) = cy;
    const Mat3 E = K.transpose() * F * K;
    const Eigen::JacobiSVD<Mat3> svd(E);
    const auto singular = svd.singularValues();
    const double scale = std::max(
        singular[0] * singular[0] + singular[1] * singular[1], 1e-30);
    const double equal =
        (singular[0] - singular[1]) * (singular[0] - singular[1]) / scale;
    const double rank = singular[2] * singular[2] /
        std::max(singular[1] * singular[1], 1e-30);
    return equal + rank;
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

// Unit-norm symmetric Sampson distance of the epipolar constraint in bearing
// space: r = (b2^T E b1) / sqrt(|E b1|^2 + |E^T b2|^2). In the small-error
// limit it is the perpendicular angular distance to the epipolar great circles
// and reduces to the planar Sampson for pinhole bearings (b.z = 1), so a single
// angular threshold serves every central model. Unlike the plane form it stays
// meaningful for back-hemisphere equirectangular features, where normalizing by
// z is impossible.
double sampson_error_bearing(const Mat3& E, const Vec3& x1, const Vec3& x2) {
    const Vec3 Ex1 = E * x1;
    const Vec3 Etx2 = E.transpose() * x2;
    const double denominator = Ex1.squaredNorm() + Etx2.squaredNorm();
    if (denominator < 1e-18) return 1e6;
    return std::abs(x2.dot(Ex1)) / std::sqrt(denominator);
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

// Intersection depths of the two observed rays (midpoint solution). Unlike the
// pinhole z>0 test these depths stay valid for rays anywhere on the sphere, so
// they are the cheirality test used for equirectangular pairs.
bool triangulate_ray_depths(
    const Mat3& R,
    const Vec3& C,
    const Vec3& b1,
    const Vec3& b2,
    Vec3& X,
    double& depth1,
    double& depth2) {
    const Vec3 d1 = b1.normalized();
    const Vec3 d2 = (R.transpose() * b2).normalized();
    if (!d1.allFinite() || !d2.allFinite()) return false;
    const double cosine = d1.dot(d2);
    const double denominator = 1.0 - cosine * cosine;
    if (denominator < 1e-12) return false;
    const double ac = d1.dot(C);
    const double bc = d2.dot(C);
    depth1 = (ac - cosine * bc) / denominator;
    depth2 = (cosine * ac - bc) / denominator;
    X = 0.5 * (depth1 * d1 + C + depth2 * d2);
    return X.allFinite();
}

unsigned score_pose_cheirality_bearings(
    const Mat3& R,
    const Vec3& t,
    const std::vector<Vec3>& x1,
    const std::vector<Vec3>& x2,
    const std::vector<char>& mask) {
    const Vec3 C = -R.transpose() * t;
    unsigned good = 0;
    for (std::size_t i = 0; i < x1.size(); ++i) {
        if (!mask[i]) continue;
        Vec3 X;
        double depth1 = 0.0, depth2 = 0.0;
        if (!triangulate_ray_depths(R, C, x1[i], x2[i], X, depth1, depth2))
            continue;
        if (depth1 <= 0.0 || depth2 <= 0.0) continue;
        ++good;
    }
    return good;
}

// Bearing-native equivalent of recover_pose_from_essential: the four (R, +-t)
// decompositions of E have identical Sampson scores, so cheirality has to pick
// one, but the test must be ray based to keep back-hemisphere observations.
bool recover_pose_from_essential_bearings(
    const Mat3& E_raw,
    const std::vector<Vec3>& bearings1,
    const std::vector<Vec3>& bearings2,
    const std::vector<char>& seed_mask,
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
            const unsigned score = score_pose_cheirality_bearings(
                Rs[ri], sign * t, bearings1, bearings2, seed_mask);
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

// Levenberg-Marquardt refinement of the relative pose by direct minimization of
// the bearing Sampson error. PoseLib has no spherical camera model, so the
// refinement is done here: R <- Exp(w) R and t <- normalize(t + B d) with the
// two tangent directions B spanning the translation direction. A finite
// difference Jacobian keeps the (small) parameter count honest without
// hand-deriving the second-order epipolar terms.
void refine_relative_pose_bearings(
    const std::vector<Vec3>& bearings1,
    const std::vector<Vec3>& bearings2,
    const std::vector<char>& mask,
    Pose3D& pose) {
    Vec3 t = pose.translation().normalized();
    if (!t.allFinite() || t.norm() < 1e-12) return;
    Mat3 R = pose.R;

    Vec3 a = Vec3::UnitX();
    if (std::abs(t.dot(a)) > 0.9) a = Vec3::UnitY();
    const Vec3 b1 = t.cross(a).normalized();
    const Vec3 b2 = t.cross(b1).normalized();

    auto essential_of = [](const Mat3& rotation, const Vec3& translation) {
        return skew(translation) * rotation;
    };
    auto cost_of = [&](const Mat3& rotation, const Vec3& translation) {
        const Mat3 E = essential_of(rotation, translation);
        double cost = 0.0;
        unsigned count = 0;
        for (std::size_t i = 0; i < bearings1.size(); ++i) {
            if (!mask[i]) continue;
            const double error = sampson_error_bearing(E, bearings1[i], bearings2[i]);
            if (!std::isfinite(error)) return std::numeric_limits<double>::infinity();
            cost += error * error;
            ++count;
        }
        return count > 0 ? cost / static_cast<double>(count)
                         : std::numeric_limits<double>::infinity();
    };

    double cost = cost_of(R, t);
    if (!std::isfinite(cost)) return;
    double lambda = 1e-4;
    constexpr double k_step = 1e-6;
    for (int iteration = 0; iteration < 20; ++iteration) {
        // Five parameters: 3 rotation tangent + 2 translation tangent.
        Eigen::Matrix<double, Eigen::Dynamic, 5> J;
        std::vector<std::size_t> rows;
        rows.reserve(bearings1.size());
        for (std::size_t i = 0; i < bearings1.size(); ++i)
            if (mask[i]) rows.push_back(i);
        if (rows.size() < 6) return;
        J.resize(static_cast<Eigen::Index>(rows.size()), 5);
        const Mat3 E = essential_of(R, t);
        for (std::size_t r = 0; r < rows.size(); ++r) {
            const Vec3& x1 = bearings1[rows[r]];
            const Vec3& x2 = bearings2[rows[r]];
            for (int p = 0; p < 5; ++p) {
                Mat3 dR = R;
                Vec3 dt = t;
                if (p < 3) {
                    Vec3 omega = Vec3::Zero();
                    omega(p) = k_step;
                    dR = (Eigen::AngleAxisd(k_step, omega.normalized())
                              .toRotationMatrix()) * R;
                } else {
                    const Vec3 basis = (p == 3) ? b1 : b2;
                    dt = (t + k_step * basis).normalized();
                }
                const double plus = sampson_error_bearing(
                    essential_of(dR, dt), x1, x2);
                const double base = sampson_error_bearing(E, x1, x2);
                J(static_cast<Eigen::Index>(r), p) = (plus - base) / k_step;
            }
        }
        Eigen::Matrix<double, 5, 5> H = J.transpose() * J;
        Eigen::Matrix<double, 5, 1> g = Eigen::Matrix<double, 5, 1>::Zero();
        {
            Eigen::Matrix<double, Eigen::Dynamic, 1> residual(
                static_cast<Eigen::Index>(rows.size()));
            for (std::size_t r = 0; r < rows.size(); ++r) {
                residual(static_cast<Eigen::Index>(r)) = sampson_error_bearing(
                    E, bearings1[rows[r]], bearings2[rows[r]]);
            }
            g = J.transpose() * residual;
        }
        bool accepted = false;
        for (int attempt = 0; attempt < 4 && !accepted; ++attempt) {
            Eigen::Matrix<double, 5, 5> damped = H;
            for (int d = 0; d < 5; ++d) damped(d, d) += lambda * (1.0 + H(d, d));
            Eigen::LDLT<Eigen::Matrix<double, 5, 5>> ldlt(damped);
            if (ldlt.info() != Eigen::Success) return;
            const Eigen::Matrix<double, 5, 1> step = ldlt.solve(g);
            if (!step.allFinite()) return;
            const Vec3 omega(step(0), step(1), step(2));
            const double angle = omega.norm();
            Mat3 candidate_R = R;
            if (angle > 1e-12) {
                candidate_R = Eigen::AngleAxisd(angle, omega / angle)
                                  .toRotationMatrix() * R;
            }
            const Vec3 candidate_t =
                (t + step(3) * b1 + step(4) * b2).normalized();
            const double candidate_cost = cost_of(candidate_R, candidate_t);
            if (std::isfinite(candidate_cost) && candidate_cost < cost) {
                R = candidate_R;
                t = candidate_t;
                cost = candidate_cost;
                lambda = std::max(1e-10, lambda * 0.3);
                accepted = true;
                if (step.norm() < 1e-9) return;
            } else {
                lambda *= 10.0;
            }
        }
        if (!accepted) break;
    }
    pose = pose_from_essential_rt(R, t);
}

// Equirectangular pixel grids are equal-solid-angle over the sphere, not equal
// area on the image, so the pinhole grid coverage score overweights the poles.
// Bin by azimuth and sin(elevation) instead; azimuth binning wraps the seam.
float compute_spherical_spatial_weight(
    const std::vector<Vec2>& pixels,
    const std::uint32_t width,
    const std::uint32_t height,
    const int grid) {
    if (pixels.empty() || width == 0 || height == 0 || grid <= 0) return 0.F;
    std::vector<char> occupied(static_cast<std::size_t>(grid * grid), 0);
    unsigned filled = 0;
    for (const Vec2& p : pixels) {
        const CameraRay ray = unproject_equirectangular_camera(
            p.x(), p.y(), static_cast<int>(width), static_cast<int>(height));
        if (!ray.valid) continue;
        const double azimuth = std::atan2(ray.x, ray.z);
        const int gx = std::clamp(
            static_cast<int>((azimuth + k_pi) / (2.0 * k_pi) * grid), 0, grid - 1);
        const int gy = std::clamp(
            static_cast<int>((ray.y + 1.0) * 0.5 * grid), 0, grid - 1);
        const std::size_t idx = static_cast<std::size_t>(gy * grid + gx);
        if (!occupied[idx]) {
            occupied[idx] = 1;
            ++filled;
        }
    }
    return static_cast<float>(filled) / static_cast<float>(grid * grid);
}

// Pure rotation between two panoramic views explains the matches without any
// baseline, which is the equirectangular analogue of the pinhole homography
// degeneracy test (a planar H has no meaning in the spherical chart).
bool equirect_rotation_degeneracy(
    const std::vector<Vec3>& bearings1,
    const std::vector<Vec3>& bearings2,
    const std::vector<char>& mask,
    const double angle_threshold,
    const double ratio_threshold,
    float& ratio_out) {
    ratio_out = 0.F;
    Mat3 covariance = Mat3::Zero();
    unsigned count = 0;
    for (std::size_t i = 0; i < bearings1.size(); ++i) {
        if (!mask[i]) continue;
        covariance.noalias() += bearings2[i] * bearings1[i].transpose();
        ++count;
    }
    if (count < 8) return false;
    Eigen::JacobiSVD<Mat3> svd(
        covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0.0) {
        Mat3 V = svd.matrixV();
        V.col(2) *= -1.0;
        R = svd.matrixU() * V.transpose();
    }
    unsigned agreeing = 0;
    for (std::size_t i = 0; i < bearings1.size(); ++i) {
        if (!mask[i]) continue;
        const Vec3 rotated = R * bearings1[i];
        const double angle = bearing_angle(
            rotated.x(), rotated.y(), rotated.z(),
            bearings2[i].x(), bearings2[i].y(), bearings2[i].z());
        if (std::isfinite(angle) && angle <= angle_threshold) ++agreeing;
    }
    ratio_out = static_cast<float>(agreeing) / static_cast<float>(count);
    return ratio_out >= static_cast<float>(ratio_threshold);
}

// Bearing-native two-view estimation, used whenever a pair involves an
// equirectangular camera. PoseLib exposes no spherical camera model, but the
// epipolar constraint depends only on the *ray* through a pixel, so a bearing
// can be negated into the front hemisphere without changing the geometry. That
// makes the (x/z, y/z) parameterization - exactly a focal-1 pinhole camera with
// the principal point at the origin, in PoseLib's own terms - available for
// every observation, and the robust estimator can be the same LO-RANSAC +
// minimal-solver machinery the perspective paths use (an 8-point DLT is far
// weaker on the small-baseline pairs a panorama capture produces).
//
// PoseLib then sees only front-hemisphere rays, so its own cheirality choice is
// meaningless for back-hemisphere content: the decomposition is re-run here on
// the original bearings, where cheirality is the depth sign along each observed
// ray, and the pose is refined against the true angular residual.
bool estimate_with_bearings(
    const std::vector<Vec2>& pixels1,
    const std::vector<Vec2>& pixels2,
    const PinholeCamera& camera1,
    const PinholeCamera& camera2,
    const RelativePoseOptions& options,
    RelativePoseResult& result) {
    if (pixels1.size() != pixels2.size() || pixels1.size() < 8) return false;

    std::vector<Vec3> x1(pixels1.size()), x2(pixels2.size());
    for (std::size_t i = 0; i < pixels1.size(); ++i) {
        x1[i] = camera1.unproject_normalized(pixels1[i]);
        x2[i] = camera2.unproject_normalized(pixels2[i]);
        if (!x1[i].allFinite() || !x2[i].allFinite()) return false;
    }

    // The Sampson-on-sphere residual is an angle, so the pixel threshold of the
    // pair is converted per camera and averaged (same convention as filtering).
    const double angle_threshold =
        0.5 * (camera1.pixel_error_to_angular(options.max_epipolar_error_px) +
               camera2.pixel_error_to_angular(options.max_epipolar_error_px));
    if (!(angle_threshold > 0.0)) return false;

    // Canonical front-hemisphere form. |z| near zero is the chart's equator
    // band (the left/right extremes of the image), where the plane
    // parameterization diverges; those rays carry no extra geometric
    // information that the rest of the sphere does not, so they are excluded
    // from the linearized estimate and re-enter the angular filtering below.
    constexpr double k_equator_z = 1e-3;
    std::vector<char> usable(pixels1.size(), 0);
    std::vector<unsigned> usable_ids;
    usable_ids.reserve(pixels1.size());
    std::vector<Vec3> proxy1, proxy2;
    proxy1.reserve(pixels1.size());
    proxy2.reserve(pixels1.size());
    for (std::size_t i = 0; i < pixels1.size(); ++i) {
        if (std::abs(x1[i].z()) < k_equator_z || std::abs(x2[i].z()) < k_equator_z)
            continue;
        usable[i] = 1;
        usable_ids.push_back(static_cast<unsigned>(i));
        const Vec3 b1 = x1[i].z() > 0.0 ? x1[i] : -x1[i];
        const Vec3 b2 = x2[i].z() > 0.0 ? x2[i] : -x2[i];
        proxy1.emplace_back(b1.x() / b1.z(), b1.y() / b1.z(), 1.0);
        proxy2.emplace_back(b2.x() / b2.z(), b2.y() / b2.z(), 1.0);
    }
    if (usable_ids.size() < 8) return false;

    Mat3 best_E = Mat3::Zero();
    unsigned best_inliers = 0;
    std::vector<char> kept_mask(usable_ids.size(), 0);

    // Minimal solver. PoseLib's 5-point solver is used on the proxy rays (the
    // plane parameterization of a front-hemisphere ray - exactly the normalized
    // coordinates a calibrated solver expects), while the *score* is always the
    // bearing Sampson error on the true rays: it is an angle, so it neither
    // wraps at the seam nor blows up off axis, and it treats the two
    // hemispheres alike. Scoring PoseLib's own way would instead measure the
    // error in plane units, which at 80 degrees off axis is ~30x stricter.
    std::mt19937 rng(1337);
    auto score_model = [&](const Mat3& E) {
        std::vector<char> mask(usable_ids.size(), 0);
        unsigned inliers = 0;
        for (std::size_t k = 0; k < usable_ids.size(); ++k) {
            const std::size_t i = usable_ids[k];
            if (sampson_error_bearing(E, x1[i], x2[i]) < angle_threshold) {
                mask[k] = 1;
                ++inliers;
            }
        }
        return std::make_pair(inliers, std::move(mask));
    };
    const std::size_t pool = usable_ids.size();
    std::uniform_int_distribution<std::size_t> dist(0, pool - 1);
#if defined(PHOTARA_HAS_POSELIB)
    constexpr std::size_t k_minimal = 5;
#else
    constexpr std::size_t k_minimal = 8;
#endif
    auto sample_minimal = [&](std::vector<unsigned>& sample,
                              const std::vector<std::size_t>* restrict,
                              const std::size_t restrict_size) {
        sample.clear();
        while (sample.size() < k_minimal) {
            const std::size_t drawn = restrict
                ? (*restrict)[dist(rng) % restrict_size]
                : dist(rng);
            if (std::find(sample.begin(), sample.end(), drawn) == sample.end())
                sample.push_back(static_cast<unsigned>(drawn));
        }
    };
    auto evaluate_sample = [&](const std::vector<unsigned>& sample) {
        std::vector<poselib::Point3D> p1, p2;
        p1.reserve(k_minimal);
        p2.reserve(k_minimal);
        for (const unsigned index : sample) {
            p1.push_back(proxy1[index]);
            p2.push_back(proxy2[index]);
        }
        std::vector<Mat3> essentials;
#if defined(PHOTARA_HAS_POSELIB)
        poselib::relpose_5pt(p1, p2, &essentials);
#else
        (void)p1;
        (void)p2;
        essentials.push_back(estimate_essential_8pt(x1, x2, sample));
#endif
        for (const Mat3& E : essentials) {
            if (!E.allFinite()) continue;
            auto [inliers, mask] = score_model(E);
            if (inliers > best_inliers) {
                best_inliers = inliers;
                best_E = E;
                kept_mask = std::move(mask);
            }
        }
    };

    unsigned max_iters = options.max_iterations;
    std::vector<unsigned> sample;
    for (unsigned iter = 0; iter < max_iters; ++iter) {
        sample_minimal(sample, nullptr, 0);
        evaluate_sample(sample);
        if (best_inliers > 0) {
            const double ratio = static_cast<double>(best_inliers) /
                                 static_cast<double>(pool);
            max_iters = std::min(
                options.max_iterations,
                std::max(options.min_iterations,
                         ransac_iterations(
                             ratio, options.confidence,
                             static_cast<unsigned>(k_minimal))));
        }
    }
    if (best_inliers < options.min_inliers) return false;

    // Local optimization: resample minimal sets from the current inliers and
    // keep any model that scores better. The final pose accuracy comes from the
    // bearing LM refinement below, so a handful of rounds is enough.
    std::vector<std::size_t> inlier_pool;
    for (std::size_t k = 0; k < kept_mask.size(); ++k)
        if (kept_mask[k]) inlier_pool.push_back(k);
    if (inlier_pool.size() >= k_minimal) {
        for (int round = 0; round < 12; ++round) {
            sample_minimal(sample, &inlier_pool, inlier_pool.size());
            evaluate_sample(sample);
        }
    }

    // Bearing-native decomposition: PoseLib picked the translation sign for the
    // front-hemisphere proxy rays, which is meaningless for a panorama. Score
    // the four (R, +-t) candidates with ray depths on the true bearings.
    Pose3D pose;
    std::vector<char> seed(pixels1.size(), 0);
    for (std::size_t k = 0; k < usable_ids.size(); ++k)
        if (kept_mask[k]) seed[usable_ids[k]] = 1;
    std::vector<char> refined = seed;
    if (!recover_pose_from_essential_bearings(
            best_E, x1, x2, seed, options.min_inliers, pose, refined))
        return false;
    refine_relative_pose_bearings(x1, x2, refined, pose);
    best_E = essential_from_pose(pose);

    result.num_ransac_inliers = best_inliers;
    float mean_angle = 0.F;
    std::vector<Vec2> inlier_pixels;
    const unsigned filtered = filter_matches(
        pixels1, pixels2, camera1, camera2, pose, options, refined, mean_angle,
        &inlier_pixels);
    if (filtered < options.min_inliers) return false;

    result.success = true;
    result.pose = pose;
    result.E = best_E;
    // F has no meaning for a spherical chart; leave it empty so downstream
    // consumers take the bearing path instead of a pixel-space epipolar one.
    result.F = Mat3::Zero();
    result.estimated_focal.reset();
    result.inlier_mask = std::move(refined);
    result.num_inliers = filtered;
    result.mean_ray_angle = mean_angle;
    result.weight_spatial = camera1.is_equirectangular()
        ? compute_spherical_spatial_weight(
              inlier_pixels, camera1.width, camera1.height, 8)
        : compute_spatial_weight(inlier_pixels, camera1.width, camera1.height);

    if (options.estimate_homography) {
        // A planar homography is not defined in the spherical chart; the
        // equivalent degeneracy is a pure rotation explaining the matches.
        const double degeneracy_angle =
            0.5 * (camera1.pixel_error_to_angular(options.max_reproj_error_px) +
                   camera2.pixel_error_to_angular(options.max_reproj_error_px));
        float ratio = 0.F;
        result.degenerate_planar = equirect_rotation_degeneracy(
            x1, x2, result.inlier_mask, degeneracy_angle,
            options.homography_degeneracy_ratio, ratio);
        result.homography_ratio = ratio;
    }
    return true;
}

std::optional<double> estimate_pair_focal_for_diagnostics(
    const Mat3& F, const PinholeCamera& camera) {
    const double image_scale =
        static_cast<double>(std::max(camera.width, camera.height));
    const double log_min = std::log(std::max(0.25 * image_scale, 1.0));
    const double log_max = std::log(std::max(4.0 * image_scale, 2.0));
    constexpr int samples = 80;
    int best_index = 0;
    double best_score = std::numeric_limits<double>::infinity();
    for (int index = 0; index < samples; ++index) {
        const double alpha =
            static_cast<double>(index) / static_cast<double>(samples - 1);
        const double candidate = log_min + alpha * (log_max - log_min);
        const double score = shared_focal_essential_score(
            F, camera.cx, camera.cy, std::exp(candidate));
        if (score < best_score) {
            best_score = score;
            best_index = index;
        }
    }
    if (best_index <= 1 || best_index >= samples - 2)
        return std::nullopt;
    const double step = (log_max - log_min) / (samples - 1);
    double left = std::max(log_min, log_min + (best_index - 1) * step);
    double right = std::min(log_max, log_min + (best_index + 1) * step);
    for (int iteration = 0; iteration < 32; ++iteration) {
        const double first = (2.0 * left + right) / 3.0;
        const double second = (left + 2.0 * right) / 3.0;
        if (shared_focal_essential_score(
                F, camera.cx, camera.cy, std::exp(first)) <=
            shared_focal_essential_score(
                F, camera.cx, camera.cy, std::exp(second)))
            right = second;
        else
            left = first;
    }
    const double focal = std::exp(0.5 * (left + right));
    return std::isfinite(focal) ? std::optional<double>{focal} : std::nullopt;
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
    PinholeCamera filter_camera1 = camera1;
    PinholeCamera filter_camera2 = camera2;
    bool have_pose = false;
    // With unknown intrinsics the pose recovered from F uses only the current
    // focal prior.  Keep the F-RANSAC support until the graph has calibrated
    // that focal; otherwise the provisional cheirality/reprojection pass can
    // irreversibly discard correct matches before the calibrated rerun.
    std::vector<char> deferred_filter_inliers;

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

    const bool fisheye = camera1.model == CameraModel::opencv_fisheye ||
                         camera2.model == CameraModel::opencv_fisheye;
    const bool use_shared_focal = !fisheye && (
        options.force_shared_focal ||
        (!camera1.trust_intrinsics && !camera2.trust_intrinsics && shared_camera));
    const bool use_calibrated =
        !options.force_fundamental && !use_shared_focal &&
        ((camera1.trust_intrinsics && camera2.trust_intrinsics) || fisheye);

    if (use_shared_focal) {
        // Estimate F first, then let the frontend's Fetzer view-graph solve
        // calibrate the shared focal from all pairs together.  Decompose with
        // the current prior for this provisional pass; relative poses are
        // recomputed after the global focal update (as in openMVS).
        ransac.real_focal_check = true;
        const poselib::RansacStats stats =
            poselib::estimate_fundamental(
                pinhole_pts1, pinhole_pts2, ransac, bundle, &F, &inliers);
        if (stats.num_inliers < options.min_inliers) return false;
        deferred_filter_inliers = inliers;
        result.estimated_focal =
            estimate_pair_focal_for_diagnostics(F, camera1);
        E = normalize_essential(
            filter_camera2.K().transpose() * F * filter_camera1.K());
        std::vector<Vec3> b1(pixels1.size()), b2(pixels2.size());
        for (std::size_t i = 0; i < pixels1.size(); ++i) {
            b1[i] = filter_camera1.unproject(pixels1[i]);
            b2[i] = filter_camera2.unproject(pixels2[i]);
        }
        const double min_cos =
            std::cos(options.min_ray_angle_deg * k_pi / 180.0);
        std::vector<char> refined;
        if (!recover_pose_from_essential(
                E, b1, b2, inliers, min_cos, options.min_inliers, pose,
                refined))
            return false;
        inliers = std::move(refined);
        E = essential_from_pose(pose);
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

    const std::vector<char>& ransac_support =
        deferred_filter_inliers.empty() ? inliers : deferred_filter_inliers;
    result.num_ransac_inliers = static_cast<unsigned>(std::count(
        ransac_support.begin(), ransac_support.end(), char{1}));
    float mean_angle = 0.F;
    std::vector<Vec2> inlier_pixels;
    if (deferred_filter_inliers.empty()) {
        const unsigned filtered = filter_matches(
            pixels1, pixels2, filter_camera1, filter_camera2, pose, options,
            inliers, mean_angle, &inlier_pixels);
        result.num_inliers = filtered;
        if (filtered < options.min_inliers) return false;
    } else {
        // Compute provisional angle diagnostics from the current focal, but do
        // not use them to shrink the correspondence set.  The frontend reruns
        // relative pose estimation and strict filtering after graph focal
        // calibration, matching openMVS' preservation of all pair matches.
        std::vector<char> diagnostic_mask = deferred_filter_inliers;
        filter_matches(
            pixels1, pixels2, filter_camera1, filter_camera2, pose, options,
            diagnostic_mask, mean_angle, nullptr);
        inliers = std::move(deferred_filter_inliers);
        result.num_inliers = result.num_ransac_inliers;
        inlier_pixels.reserve(result.num_inliers);
        for (std::size_t index = 0; index < inliers.size(); ++index)
            if (inliers[index]) inlier_pixels.push_back(pixels1[index]);
    }

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

// Linear absolute pose from bearing observations. The constraint
//   b x (R X + t) = 0
// is linear in the 12 entries of P = [R | t] and, unlike the plane form
// (x, y) = (X/Z, Y/Z), it does not require the observation to lie in the front
// hemisphere: it constrains the 3D ray, so equirectangular features behind the
// camera are as valid as those in front. Cheirality is the depth sign along the
// ray, checked after the decomposition.
bool solve_dlt_pnp_bearings(
    const std::vector<Vec3>& bearings,
    const std::vector<Vec3>& points,
    const std::vector<unsigned>& sample,
    Mat3& R,
    Vec3& t) {
    if (sample.size() < 6) return false;
    // Hartley conditioning: the linear system is solved in a normalized frame
    // (centroid at the origin, mean radius sqrt(3)) and mapped back afterwards.
    Vec3 centroid = Vec3::Zero();
    for (const unsigned index : sample) centroid += points[index];
    centroid /= static_cast<double>(sample.size());
    double mean_radius = 0.0;
    for (const unsigned index : sample) mean_radius += (points[index] - centroid).norm();
    mean_radius /= static_cast<double>(sample.size());
    if (!(mean_radius > 1e-12) || !std::isfinite(mean_radius)) {
        centroid = Vec3::Zero();
        mean_radius = 1.0;
    }
    const double scale = 1.7320508075688772 / mean_radius;

    Eigen::MatrixXd A(3 * static_cast<Eigen::Index>(sample.size()), 12);
    Eigen::Index row = 0;
    for (const unsigned index : sample) {
        const Vec3 b = bearings[index].normalized();
        if (!b.allFinite()) continue;
        const Vec3 X = (points[index] - centroid) * scale;
        // [b]_x rows: b x (M X + t) = 0 expanded into three linear equations.
        const double rows_[3][3] = {
            {0.0, -b.z(), b.y()},
            {b.z(), 0.0, -b.x()},
            {-b.y(), b.x(), 0.0}};
        for (int k = 0; k < 3; ++k) {
            for (int column = 0; column < 3; ++column) {
                A(row, column) = rows_[k][column] * X.x();
                A(row, 3 + column) = rows_[k][column] * X.y();
                A(row, 6 + column) = rows_[k][column] * X.z();
            }
            A(row, 9) = rows_[k][0];
            A(row, 10) = rows_[k][1];
            A(row, 11) = rows_[k][2];
            ++row;
        }
    }
    if (row < 11) return false;
    A.conservativeResize(row, 12);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::Matrix<double, 12, 1> p = svd.matrixV().col(11);
    // The unknowns are ordered column by column of M (see the row layout above),
    // followed by the translation.
    Mat3 M_tilde;
    M_tilde.col(0) = p.segment<3>(0);
    M_tilde.col(1) = p.segment<3>(3);
    M_tilde.col(2) = p.segment<3>(6);
    const Vec3 t_tilde = p.segment<3>(9);

    // Undo X_tilde = scale * (X - centroid): the pose in the original frame is
    // M = scale * M_tilde and t = t_tilde - scale * M_tilde * centroid.
    const Mat3 M = scale * M_tilde;
    const Vec3 t_raw = t_tilde - scale * M_tilde * centroid;
    Eigen::JacobiSVD<Mat3> svd_m(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd_m.matrixU() * svd_m.matrixV().transpose();
    if (R.determinant() < 0) {
        Mat3 U = svd_m.matrixU();
        U.col(2) *= -1.0;
        R = U * svd_m.matrixV().transpose();
    }
    const double metric =
        (svd_m.singularValues()(0) + svd_m.singularValues()(1) +
         svd_m.singularValues()(2)) / 3.0;
    if (!(std::abs(metric) > 1e-12)) return false;
    t = t_raw / metric;

    // Ray cheirality: most observations must lie along their own bearing.
    unsigned front = 0;
    for (const unsigned index : sample) {
        if ((R * points[index] + t).dot(bearings[index].normalized()) > 0.0)
            ++front;
    }
    if (front * 2 < sample.size()) {
        R = -R;
        t = -t;
    }
    return R.allFinite() && t.allFinite() &&
           std::abs(R.determinant() - 1.0) < 1e-2;
}

// Local refinement of the absolute pose against the observed unit bearings.
// The cost is the chord distance between the predicted and observed directions,
// which is smooth everywhere on the sphere (no acos singularity at zero error
// and no z>0 restriction). Finite differences keep the six parameters honest.
void refine_absolute_pose_bearings(
    const std::vector<Vec3>& bearings,
    const std::vector<Vec3>& points,
    const std::vector<char>& mask,
    Mat3& R,
    Vec3& t) {
    std::vector<std::size_t> rows;
    rows.reserve(bearings.size());
    for (std::size_t i = 0; i < bearings.size(); ++i)
        if (mask[i]) rows.push_back(i);
    if (rows.size() < 6) return;

    auto cost_of = [&](const Mat3& rotation, const Vec3& translation) {
        double cost = 0.0;
        for (const std::size_t index : rows) {
            const Vec3 predicted = rotation * points[index] + translation;
            const double length = predicted.norm();
            if (!(length > 1e-12) || !std::isfinite(length)) {
                return std::numeric_limits<double>::infinity();
            }
            cost += (predicted / length - bearings[index]).squaredNorm();
        }
        return cost / static_cast<double>(rows.size());
    };

    double cost = cost_of(R, t);
    if (!std::isfinite(cost)) return;
    double lambda = 1e-4;
    constexpr double k_step = 1e-6;
    const Eigen::Index row_count = static_cast<Eigen::Index>(rows.size());
    for (int iteration = 0; iteration < 20; ++iteration) {
        // 3 residual components per correspondence (the unit-direction error),
        // 6 parameters (3 rotation tangent + 3 translation).
        Eigen::MatrixXd J(3 * row_count, 6);
        Eigen::VectorXd residual(3 * row_count);
        J.setZero();
        for (Eigen::Index r = 0; r < row_count; ++r) {
            const Vec3& X = points[rows[static_cast<std::size_t>(r)]];
            const Vec3& b = bearings[rows[static_cast<std::size_t>(r)]];
            const Vec3 predicted = R * X + t;
            const double length = predicted.norm();
            if (!(length > 1e-12)) return;
            const Vec3 base = predicted / length;
            residual.segment<3>(3 * r) = base - b;
            for (int p = 0; p < 6; ++p) {
                Mat3 dR = R;
                Vec3 dt = t;
                if (p < 3) {
                    Vec3 omega = Vec3::Zero();
                    omega(p) = k_step;
                    dR = Eigen::AngleAxisd(k_step, omega.normalized())
                             .toRotationMatrix() * R;
                } else {
                    dt(p - 3) += k_step;
                }
                const Vec3 perturbed = dR * X + dt;
                const double perturbed_length = perturbed.norm();
                if (!(perturbed_length > 1e-12)) return;
                J.block<3, 1>(3 * r, p) =
                    (perturbed / perturbed_length - base) / k_step;
            }
        }
        const Eigen::Matrix<double, 6, 6> H = J.transpose() * J;
        const Eigen::Matrix<double, 6, 1> g = J.transpose() * residual;
        bool accepted = false;
        for (int attempt = 0; attempt < 4 && !accepted; ++attempt) {
            Eigen::Matrix<double, 6, 6> damped = H;
            for (int d = 0; d < 6; ++d) damped(d, d) += lambda * (1.0 + H(d, d));
            Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(damped);
            if (ldlt.info() != Eigen::Success) return;
            const Eigen::Matrix<double, 6, 1> step = ldlt.solve(g);
            if (!step.allFinite()) return;
            const Vec3 omega = step.head<3>();
            const double angle = omega.norm();
            Mat3 candidate_R = R;
            if (angle > 1e-12) {
                candidate_R = Eigen::AngleAxisd(angle, omega / angle)
                                  .toRotationMatrix() * R;
            }
            const Vec3 candidate_t = t + step.tail<3>();
            const double candidate_cost = cost_of(candidate_R, candidate_t);
            if (std::isfinite(candidate_cost) && candidate_cost < cost) {
                R = candidate_R;
                t = candidate_t;
                cost = candidate_cost;
                lambda = std::max(1e-10, lambda * 0.3);
                accepted = true;
            } else {
                lambda *= 10.0;
            }
        }
        if (!accepted) break;
    }
}

}  // namespace

namespace {

// RANSAC + local refinement of the absolute pose from bearing observations.
// Used for equirectangular cameras and as the PoseLib-free fallback: sampling
// uses the ray-form DLT, scoring and refinement use the sphere-safe chord/angle
// residual, and cheirality is the depth sign along each observed ray.
AbsolutePoseResult estimate_absolute_pose_bearing_native(
    const std::vector<Vec3>& bearings,
    const std::vector<Vec3>& points_world,
    const PinholeCamera& camera,
    const AbsolutePoseOptions& options) {
    AbsolutePoseResult result;
    if (bearings.size() != points_world.size() || bearings.size() < 6)
        return result;

    std::vector<Vec3> rays(bearings.size());
    for (std::size_t i = 0; i < bearings.size(); ++i) {
        rays[i] = bearings[i].normalized();
        if (!rays[i].allFinite()) return result;
    }
    const double cos_threshold =
        std::cos(camera.pixel_error_to_angular(options.max_reproj_error_px));

    std::mt19937 rng(7);
    std::uniform_int_distribution<std::size_t> dist(0, rays.size() - 1);
    std::vector<char> best_mask(rays.size(), 0);
    Mat3 best_R = Mat3::Identity();
    Vec3 best_t = Vec3::Zero();
    unsigned best_inliers = 0;
    unsigned max_iters = options.max_iterations;
    constexpr unsigned k_sample_size = 6;
    for (unsigned iter = 0; iter < max_iters; ++iter) {
        std::vector<unsigned> sample;
        sample.reserve(k_sample_size);
        while (sample.size() < k_sample_size) {
            const unsigned index = static_cast<unsigned>(dist(rng));
            if (std::find(sample.begin(), sample.end(), index) == sample.end())
                sample.push_back(index);
        }
        Mat3 R;
        Vec3 t;
        if (!solve_dlt_pnp_bearings(rays, points_world, sample, R, t)) continue;
        std::vector<char> mask(rays.size(), 0);
        unsigned inliers = 0;
        for (std::size_t i = 0; i < rays.size(); ++i) {
            const Vec3 Xc = R * points_world[i] + t;
            const double length = Xc.norm();
            if (!(length > 1e-12)) continue;
            if (rays[i].dot(Xc / length) >= cos_threshold) {
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
                static_cast<double>(inliers) / static_cast<double>(rays.size());
            max_iters = std::max(
                options.min_iterations,
                ransac_iterations(ratio, options.confidence, k_sample_size));
            max_iters = std::min(max_iters, options.max_iterations);
        }
    }
    if (best_inliers < options.min_inliers) return result;

    std::vector<unsigned> inlier_ids;
    for (unsigned i = 0; i < best_mask.size(); ++i)
        if (best_mask[i]) inlier_ids.push_back(i);
    Mat3 R = best_R;
    Vec3 t = best_t;
    if (inlier_ids.size() >= k_sample_size)
        solve_dlt_pnp_bearings(rays, points_world, inlier_ids, R, t);
    refine_absolute_pose_bearings(rays, points_world, best_mask, R, t);

    // Re-score after refinement so the reported support matches the final pose.
    unsigned final_inliers = 0;
    for (std::size_t i = 0; i < rays.size(); ++i) {
        const Vec3 Xc = R * points_world[i] + t;
        const double length = Xc.norm();
        const bool inlier =
            length > 1e-12 && rays[i].dot(Xc / length) >= cos_threshold;
        best_mask[i] = inlier ? 1 : 0;
        if (inlier) ++final_inliers;
    }
    if (final_inliers < options.min_inliers) return result;
    if (inlier_ids.size() >= k_sample_size) {
        refine_absolute_pose_bearings(rays, points_world, best_mask, R, t);
    }

    result.success = true;
    result.pose.set_from_rt(R, t);
    result.inlier_mask = std::move(best_mask);
    result.num_inliers = final_inliers;
    return result;
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

float compute_spatial_weight(
    const std::vector<Vec2>& pixels,
    const PinholeCamera& camera,
    const int grid) {
    if (camera.is_equirectangular())
        return compute_spherical_spatial_weight(
            pixels, camera.width, camera.height, grid);
    return compute_spatial_weight(pixels, camera.width, camera.height, grid);
}

RelativePoseResult estimate_relative_pose(
    const std::vector<Vec2>& pixels1,
    const std::vector<Vec2>& pixels2,
    const PinholeCamera& camera1,
    const PinholeCamera& camera2,
    const RelativePoseOptions& options) {
    RelativePoseResult result;
    if (pixels1.size() != pixels2.size() || pixels1.size() < 8) return result;

    // A pair that contains a full-sphere camera cannot be scored in pixels or
    // normalized by z; it takes the bearing path in both build configurations.
    const bool bearing_pair =
        camera1.is_equirectangular() || camera2.is_equirectangular();

#if defined(PHOTARA_HAS_POSELIB)
    if (bearing_pair) {
        estimate_with_bearings(
            pixels1, pixels2, camera1, camera2, options, result);
        return result;
    }
    if (estimate_with_poselib(pixels1, pixels2, camera1, camera2, options, result))
        return result;
    return result;
#else
    if (bearing_pair && estimate_with_bearings(
            pixels1, pixels2, camera1, camera2, options, result))
        return result;
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

    // Equirectangular cameras observe the full sphere, so their resection is
    // estimated natively on bearings (PoseLib has no spherical camera model and
    // its calibrated path rejects z <= 0 observations outright).
    if (camera.is_equirectangular()) {
        return estimate_absolute_pose_bearing_native(
            bearings, points_world, camera, options);
    }

#if defined(PHOTARA_HAS_POSELIB)
    std::vector<poselib::Point2D> pixels;
    std::vector<poselib::Point3D> points;
    pixels.reserve(bearings.size());
    points.reserve(points_world.size());
    for (std::size_t i = 0; i < bearings.size(); ++i) {
        const Vec3 bearing = bearings[i].normalized();
        if (bearing.z() <= 1e-8) return result;
        if (camera.model == CameraModel::opencv_fisheye)
            pixels.emplace_back(camera.project(bearing));
        else
            pixels.emplace_back(
                camera.fx * bearing.x() / bearing.z() + camera.cx,
                camera.fy * bearing.y() / bearing.z() + camera.cy);
        points.emplace_back(points_world[i]);
    }

    const poselib::Camera pose_camera = camera.model == CameraModel::opencv_fisheye
        ? to_poselib_camera(camera) : poselib::Camera(
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

}  // namespace photara::sfm
