#pragma once

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
    bool trust_intrinsics{true};

    [[nodiscard]] Mat3 K() const {
        Mat3 matrix = Mat3::Identity();
        matrix(0, 0) = fx;
        matrix(1, 1) = fy;
        matrix(0, 2) = cx;
        matrix(1, 2) = cy;
        return matrix;
    }

    [[nodiscard]] double focal() const { return 0.5 * (fx + fy); }

    // Pixel -> normalized plane (z=1). Iteratively undo Brown distortion when present.
    [[nodiscard]] Vec3 unproject(const Vec2& pixel) const {
        double x = (pixel.x() - cx) / fx;
        double y = (pixel.y() - cy) / fy;
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
        const double inv_z = 1.0 / camera_point.z();
        const double x = camera_point.x() * inv_z;
        const double y = camera_point.y() * inv_z;
        const double r2 = x * x + y * y;
        const double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
        const double distorted_x =
            x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        const double distorted_y =
            y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        return {fx * distorted_x + cx, fy * distorted_y + cy};
    }

    [[nodiscard]] bool project_checked(const Vec3& camera_point, Vec2& pixel) const {
        if (camera_point.z() <= 1e-8) return false;
        pixel = project(camera_point);
        return std::isfinite(pixel.x()) && std::isfinite(pixel.y());
    }

    [[nodiscard]] double pixel_error_to_angular(const double pixel_error) const {
        return std::atan2(pixel_error, focal());
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
