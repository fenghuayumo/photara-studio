#pragma once

#include "ba/linearizer.hpp"

#include <cmath>

#if defined(__CUDACC__)
#define PHOTARA_HD __host__ __device__
#define PHOTARA_FORCEINLINE __forceinline__
#else
#define PHOTARA_HD
#define PHOTARA_FORCEINLINE inline
#endif

namespace photara::ba::detail {

PHOTARA_HD PHOTARA_FORCEINLINE void linearize_observation(
    const Pose& pose,
    const PinholeIntrinsics& intrinsics,
    const Point3& point,
    const double observed_x,
    const double observed_y,
    const double observation_weight,
    const LinearizerOptions& options,
    LinearizedObservation& result) {
    result.residual[0] = 0.0;
    result.residual[1] = 0.0;
    for (int i = 0; i < 12; ++i) {
        result.pose_jacobian[i] = 0.0;
    }
    for (int i = 0; i < 6; ++i) {
        result.point_jacobian[i] = 0.0;
    }
    for (int i = 0; i < static_cast<int>(2 * k_max_intrinsic_params); ++i) {
        result.intrinsic_jacobian[i] = 0.0;
    }
    result.robust_weight = 0.0;
    result.valid = false;

    const double qw = pose.qw;
    const double qx = pose.qx;
    const double qy = pose.qy;
    const double qz = pose.qz;

    const double r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
    const double r01 = 2.0 * (qx * qy - qw * qz);
    const double r02 = 2.0 * (qx * qz + qw * qy);
    const double r10 = 2.0 * (qx * qy + qw * qz);
    const double r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
    const double r12 = 2.0 * (qy * qz - qw * qx);
    const double r20 = 2.0 * (qx * qz - qw * qy);
    const double r21 = 2.0 * (qy * qz + qw * qx);
    const double r22 = 1.0 - 2.0 * (qx * qx + qy * qy);

    const double dx = point.x - pose.cx;
    const double dy = point.y - pose.cy;
    const double dz = point.z - pose.cz;
    const double px = r00 * dx + r01 * dy + r02 * dz;
    const double py = r10 * dx + r11 * dy + r12 * dz;
    const double pz = r20 * dx + r21 * dy + r22 * dz;
    // Equirectangular cameras see the whole sphere, so the usual "point in
    // front of the camera" invariant does not exist: only a point at the optical
    // centre is unobservable.
    const bool equirect = uses_bearing_projection(intrinsics.model);
    if (equirect) {
        const double length2 = px * px + py * py + pz * pz;
        if (!(length2 > options.minimum_depth * options.minimum_depth) ||
            !::isfinite(length2)) {
            return;
        }
    } else if (!(pz > options.minimum_depth) || !::isfinite(pz)) {
        return;
    }

    // Residual and its 2x3 Jacobian with respect to the camera-space point.
    // Pinhole/fisheye use the pixel difference; equirectangular uses the
    // tangent-plane (azimuth/elevation scaled) residual, which is the correct
    // metric everywhere on the sphere and stays in pixels for the Huber weight.
    double raw_rx = 0.0;
    double raw_ry = 0.0;
    double j00 = 0.0, j01 = 0.0, j02 = 0.0;
    double j10 = 0.0, j11 = 0.0, j12 = 0.0;
    double x_distorted = 0.0;
    double y_distorted = 0.0;
    CameraProjection projection{};
    if (equirect) {
        const EquirectTangentBasis basis = equirect_tangent_basis(
            observed_x, observed_y, intrinsics.fx, intrinsics.fy, intrinsics.cx,
            intrinsics.cy);
        const EquirectLocalReprojection local =
            equirect_local_reprojection(px, py, pz, basis);
        if (!local.valid) return;
        raw_rx = local.residual_x;
        raw_ry = local.residual_y;
        j00 = local.j00;
        j01 = local.j01;
        j02 = local.j02;
        j10 = local.j10;
        j11 = local.j11;
        j12 = local.j12;
    } else {
        const double inv_z = 1.0 / pz;
        const double xn = px * inv_z;
        const double yn = py * inv_z;
        projection = project_camera_plane(intrinsics.model, xn, yn,
            intrinsics.k1, intrinsics.k2, intrinsics.p1, intrinsics.p2);
        x_distorted = projection.x;
        y_distorted = projection.y;
        const double projected_x = intrinsics.fx * x_distorted + intrinsics.cx;
        const double projected_y = intrinsics.fy * y_distorted + intrinsics.cy;
        raw_rx = projected_x - observed_x;
        raw_ry = projected_y - observed_y;
        j00 = intrinsics.fx * projection.xx * inv_z;
        j01 = intrinsics.fx * projection.xy * inv_z;
        j02 = -(j00 * px + j01 * py) * inv_z;
        j10 = intrinsics.fy * projection.yx * inv_z;
        j11 = intrinsics.fy * projection.yy * inv_z;
        j12 = -(j10 * px + j11 * py) * inv_z;
    }
    if (!::isfinite(raw_rx) || !::isfinite(raw_ry)) {
        return;
    }

    double robust_weight = observation_weight;
    if (options.huber_delta > 0.0) {
        const double norm = ::sqrt(raw_rx * raw_rx + raw_ry * raw_ry);
        if (norm > options.huber_delta) {
            robust_weight *= options.huber_delta / norm;
        }
    }
    const double scale = ::sqrt(robust_weight);
    result.residual[0] = scale * raw_rx;
    result.residual[1] = scale * raw_ry;
    result.robust_weight = robust_weight;

    // d p_camera / d delta_rotation for R' = Exp(delta_rotation) R.
    const double rotation_jacobian[9] = {
        0.0, pz, -py,
        -pz, 0.0, px,
        py, -px, 0.0};
    const double center_jacobian[9] = {
        -r00, -r01, -r02,
        -r10, -r11, -r12,
        -r20, -r21, -r22};
    const double point_jacobian[9] = {
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22};
    const double projection_jacobian[6] = {j00, j01, j02, j10, j11, j12};

    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 3; ++column) {
            double rotation_value = 0.0;
            double center_value = 0.0;
            double point_value = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double projection = projection_jacobian[row * 3 + k];
                rotation_value += projection * rotation_jacobian[k * 3 + column];
                center_value += projection * center_jacobian[k * 3 + column];
                point_value += projection * point_jacobian[k * 3 + column];
            }
            result.pose_jacobian[row * 6 + column] = scale * rotation_value;
            result.pose_jacobian[row * 6 + 3 + column] = scale * center_value;
            result.point_jacobian[row * 3 + column] = scale * point_value;
        }
    }

    // Shared intrinsic Jacobians (param order matches intrinsic_dof()).
    // An equirectangular camera has no free intrinsics: the projection is fixed
    // by the image size, so the group carries no gradient and stays frozen.
    if (equirect) {
        result.valid = true;
        return;
    }
    int intrinsic_column = 0;
    if (options.optimize_focal) {
        if (options.optimize_aspect_ratio) {
            result.intrinsic_jacobian[intrinsic_column] = scale * x_distorted;
            result.intrinsic_jacobian[
                k_max_intrinsic_params + intrinsic_column] = 0.0;
            ++intrinsic_column;
            result.intrinsic_jacobian[intrinsic_column] = 0.0;
            result.intrinsic_jacobian[
                k_max_intrinsic_params + intrinsic_column] =
                scale * y_distorted;
            ++intrinsic_column;
        } else {
            result.intrinsic_jacobian[intrinsic_column] = scale * x_distorted;
            result.intrinsic_jacobian[
                k_max_intrinsic_params + intrinsic_column] =
                scale * y_distorted;
            ++intrinsic_column;
        }
    }
    if (options.optimize_principal_point) {
        result.intrinsic_jacobian[intrinsic_column] = scale;
        result.intrinsic_jacobian[k_max_intrinsic_params + intrinsic_column] = 0.0;
        ++intrinsic_column;
        result.intrinsic_jacobian[intrinsic_column] = 0.0;
        result.intrinsic_jacobian[k_max_intrinsic_params + intrinsic_column] = scale;
        ++intrinsic_column;
    }
    if (options.optimize_distortion) {
        for (int i = 0; i < 4; ++i, ++intrinsic_column) {
            result.intrinsic_jacobian[intrinsic_column] =
                scale * intrinsics.fx * projection.dx[i];
            result.intrinsic_jacobian[k_max_intrinsic_params + intrinsic_column] =
                scale * intrinsics.fy * projection.dy[i];
        }
    }
    (void)intrinsic_column;
    result.valid = true;
}

}  // namespace photara::ba::detail

#undef PHOTARA_HD
#undef PHOTARA_FORCEINLINE
