#pragma once

#include "core/camera_projection.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>
#include <limits>

namespace aetherscan::sfm {

using Index = std::uint32_t;
inline constexpr Index k_invalid = std::numeric_limits<Index>::max();

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Mat34 = Eigen::Matrix<double, 3, 4>;
using Mat4 = Eigen::Matrix4d;
using Quat = Eigen::Quaterniond;

// OpenMVS / OpenMVG convention: P = K R [I | -C]
// R: world -> camera, C: camera center in world.
struct Pose3D {
    Mat3 R{Mat3::Identity()};
    Vec3 C{Vec3::Zero()};

    [[nodiscard]] static Pose3D identity() { return {}; }

    [[nodiscard]] Vec3 translation() const { return -R * C; }

    void set_from_rt(const Mat3& rotation, const Vec3& t) {
        R = rotation;
        C = -rotation.transpose() * t;
    }

    [[nodiscard]] Vec3 transform_world_to_camera(const Vec3& X) const {
        return R * (X - C);
    }

    [[nodiscard]] Vec3 transform_camera_to_world(const Vec3& X) const {
        return R.transpose() * X + C;
    }

    [[nodiscard]] Pose3D inverse() const {
        // openMVS: Inverse() -> (R^T, -(R C)); applying as W2C yields original C2W.
        return {R.transpose(), -(R * C)};
    }

    // Compose: result = this * other  (World -> other -> this)
    [[nodiscard]] Pose3D operator*(const Pose3D& other) const {
        Pose3D out;
        out.R = R * other.R;
        out.C = other.C + other.R.transpose() * C;
        return out;
    }

    // Relative pose such that result * other = *this
    [[nodiscard]] Pose3D operator/(const Pose3D& other) const {
        Pose3D out;
        out.R = R * other.R.transpose();
        out.C = other.R * (C - other.C);
        return out;
    }

    [[nodiscard]] Quat quaternion() const { return Quat(R).normalized(); }
};

// Historical type name retained for source compatibility; model selects projection.
struct PinholeCamera {
    Index id{k_invalid};
    std::uint32_t width{};
    std::uint32_t height{};
    double fx{1.0};
    double fy{1.0};
    double cx{};
    double cy{};
    double k1{};
    double k2{};
    double p1{};
    double p2{};
    // Stable focal anchor for BA priors/bounds. Zero means initialize from fx/fy.
    double focal_prior{};
    bool trust_intrinsics{true};
    CameraModel model{CameraModel::pinhole};
    // For opencv_fisheye, p1/p2 store angular k3/k4 (not tangential terms).

    [[nodiscard]] Mat3 K() const {
        Mat3 matrix = Mat3::Identity();
        matrix(0, 0) = fx;
        matrix(1, 1) = fy;
        matrix(0, 2) = cx;
        matrix(1, 2) = cy;
        return matrix;
    }

    [[nodiscard]] double focal() const { return 0.5 * (fx + fy); }

    [[nodiscard]] bool is_equirectangular() const {
        return model == CameraModel::equirectangular;
    }

    // Equirectangular cameras carry no free intrinsics: the projection is fully
    // determined by the image size (fx = width / 2pi, fy = height / pi, the
    // principal point at the image centre). Callers that estimate or optimize
    // focal length / distortion must skip these cameras.
    void set_equirectangular_intrinsics() {
        model = CameraModel::equirectangular;
        fx = equirect_pixel_scale_x(static_cast<int>(width));
        fy = equirect_pixel_scale_y(static_cast<int>(height));
        cx = 0.5 * static_cast<double>(width);
        cy = 0.5 * static_cast<double>(height);
        k1 = k2 = p1 = p2 = 0.0;
        focal_prior = fx;
        trust_intrinsics = true;
    }

    // Pixel -> unit bearing in camera space, with model-specific inverse
    // distortion. Pinhole and fisheye return a point on the z=1 normalized
    // plane (z > 0); equirectangular returns the true unit bearing over the
    // full sphere, whose z is negative behind the camera.
    [[nodiscard]] Vec3 unproject(const Vec2& pixel) const {
        if (model == CameraModel::equirectangular) {
            const CameraRay ray = unproject_equirectangular_camera(
                pixel.x(), pixel.y(), static_cast<int>(width),
                static_cast<int>(height));
            if (!ray.valid) return Vec3::Constant(std::numeric_limits<double>::quiet_NaN());
            return Vec3(ray.x, ray.y, ray.z);
        }
        double x = (pixel.x() - cx) / fx;
        double y = (pixel.y() - cy) / fy;
        if (model == CameraModel::opencv_fisheye) {
            const double radius = std::hypot(x, y);
            if (radius < 1e-12) return {x, y, 1.0};
            // Safeguarded inverse on the forward hemisphere; invalid pixels
            // outside a monotonic calibration return NaN for existing filters.
            double lo = 0.0, hi = 1.5707963267948966 - 1e-8;
            auto distorted_angle = [&](double theta) {
                const double t2 = theta * theta;
                return theta * (1 + t2*(k1+t2*(k2+t2*(p1+t2*p2))));
            };
            if (radius >= distorted_angle(hi))
                return Vec3::Constant(std::numeric_limits<double>::quiet_NaN());
            double theta = radius < hi ? radius : 0.5*hi;
            for (int i = 0; i < 50; ++i) {
                const double error = distorted_angle(theta)-radius;
                if (std::abs(error) < 1e-13) break;
                if (error < 0) lo = theta; else hi = theta;
                const double t2 = theta*theta;
                const double derivative = 1+t2*(3*k1+t2*(5*k2+t2*(7*p1+t2*9*p2)));
                const double next = theta-error/derivative;
                theta = derivative > 0 && next > lo && next < hi ? next : 0.5*(lo+hi);
            }
            const double scale = std::tan(theta)/radius;
            return {x*scale, y*scale, 1.0};
        }
        if (k1 != 0.0 || k2 != 0.0 || p1 != 0.0 || p2 != 0.0) {
            const double xd = x;
            const double yd = y;
            for (int iter = 0; iter < 5; ++iter) {
                const double r2 = x * x + y * y;
                const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
                const double x_distorted =
                    x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
                const double y_distorted =
                    y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
                // Additive correction toward observed distorted coords.
                x += xd - x_distorted;
                y += yd - y_distorted;
            }
        }
        return {x, y, 1.0};
    }

    [[nodiscard]] Vec3 unproject_normalized(const Vec2& pixel) const {
        return unproject(pixel).normalized();
    }

    [[nodiscard]] Vec2 project(const Vec3& camera_point) const {
        if (model == CameraModel::equirectangular) {
            const CameraPixel projected = project_equirectangular_camera(
                camera_point.x(), camera_point.y(), camera_point.z(),
                static_cast<int>(width), static_cast<int>(height));
            return {projected.u, projected.v};
        }
        const double inv_z = 1.0 / camera_point.z();
        const double x = camera_point.x() * inv_z;
        const double y = camera_point.y() * inv_z;
        const auto projected = project_camera_plane(model, x, y, k1, k2, p1, p2);
        return {fx * projected.x + cx, fy * projected.y + cy};
    }

    [[nodiscard]] bool project_checked(const Vec3& camera_point, Vec2& pixel) const {
        if (model == CameraModel::equirectangular) {
            // No cheirality for a full-sphere camera: only the degenerate case
            // of a point at the optical centre is unobservable.
            if (!camera_point.allFinite() ||
                camera_point.norm() <= 1e-12) return false;
            pixel = project(camera_point);
            return std::isfinite(pixel.x()) && std::isfinite(pixel.y());
        }
        if (camera_point.z() <= 1e-8) return false;
        pixel = project(camera_point);
        return std::isfinite(pixel.x()) && std::isfinite(pixel.y());
    }

    [[nodiscard]] double pixel_error_to_angular(const double pixel_error) const {
        if (model == CameraModel::equirectangular) {
            // One pixel spans 2 pi / width radians along the azimuth, but the
            // chart samples the sphere far more coarsely than a rectilinear lens
            // samples the image plane: a 2048 px panorama is 0.176 deg/px, where
            // a 50 mm pinhole frame of the same width is ~0.03 deg/px. Every
            // pixel-space threshold in the pipeline (epipolar, reprojection,
            // track filtering, resection RANSAC) would therefore be several
            // times looser in angle for a panorama. The scale below restores a
            // comparable angular precision so that "N pixels" means the same
            // geometry on both camera types.
            if (width == 0) return 0.0;
            return pixel_error * 2.0 * k_pi / static_cast<double>(width) *
                   k_equirect_threshold_scale;
        }
        return std::atan2(pixel_error, focal());
    }

    // Tangent-plane reprojection residual (pixels) and its 2x3 Jacobian with
    // respect to the camera-space point. For pinhole and fisheye this is the
    // ordinary pixel difference; for equirectangular it is the seam- and
    // pole-safe angular residual. The point is reported in camera space on
    // purpose: the world-to-camera derivative is model independent.
    struct LocalResidual {
        Vec2 residual{Vec2::Zero()};
        Eigen::Matrix<double, 2, 3> jacobian{Eigen::Matrix<double, 2, 3>::Zero()};
        bool valid{false};
    };

    [[nodiscard]] LocalResidual local_reprojection(
        const Vec3& camera_point, const Vec2& observed) const {
        LocalResidual out;
        if (model == CameraModel::equirectangular) {
            const EquirectTangentBasis basis = equirect_tangent_basis(
                observed.x(), observed.y(), fx, fy, cx, cy);
            const EquirectLocalReprojection local =
                equirect_local_reprojection(
                    camera_point.x(), camera_point.y(), camera_point.z(), basis);
            if (!local.valid) return out;
            out.residual = {local.residual_x, local.residual_y};
            out.jacobian << local.j00, local.j01, local.j02,
                            local.j10, local.j11, local.j12;
            out.valid = true;
            return out;
        }
        Vec2 projected;
        if (!project_checked(camera_point, projected)) return out;
        const double inv_z = 1.0 / camera_point.z();
        out.residual = projected - observed;
        // Distortion derivatives are deliberately ignored (the residual still
        // uses the full projection), matching the historical refinement.
        out.jacobian << fx * inv_z, 0.0, -fx * camera_point.x() * inv_z * inv_z,
            0.0, fy * inv_z, -fy * camera_point.y() * inv_z * inv_z;
        out.valid = true;
        return out;
    }

    // Angular distance between a camera-space point and an observed pixel,
    // expressed in pixels on this camera's image. Valid for every model, so
    // filtering code can share one threshold.
    [[nodiscard]] double angular_error_px(
        const Vec3& camera_point, const Vec2& observed) const {
        const Vec3 predicted = camera_point.normalized();
        const Vec3 observation = unproject_normalized(observed);
        if (!predicted.allFinite() || !observation.allFinite()) {
            return std::numeric_limits<double>::infinity();
        }
        const double angle = bearing_angle(
            predicted.x(), predicted.y(), predicted.z(),
            observation.x(), observation.y(), observation.z());
        if (!std::isfinite(angle)) {
            return std::numeric_limits<double>::infinity();
        }
        return angle / std::max(focal(), 1e-12);
    }
};

struct FeatureMatch {
    Index query{};
    Index train{};
};

struct Observation {
    Index image_id{k_invalid};
    Index feature_id{k_invalid};
};

}  // namespace aetherscan::sfm
