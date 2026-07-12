#pragma once

#include "ba/linearizer.hpp"

#include <cmath>

#if defined(__CUDACC__)
#define AETHERSCAN_HD __host__ __device__
#define AETHERSCAN_FORCEINLINE __forceinline__
#else
#define AETHERSCAN_HD
#define AETHERSCAN_FORCEINLINE inline
#endif

namespace aetherscan::ba::detail {

AETHERSCAN_HD AETHERSCAN_FORCEINLINE void linearize_observation(
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
    if (!(pz > options.minimum_depth) || !::isfinite(pz)) {
        return;
    }

    const double inv_z = 1.0 / pz;
    const double xn = px * inv_z;
    const double yn = py * inv_z;
    const double r2 = xn * xn + yn * yn;
    const double radial = 1.0 + intrinsics.k1 * r2 + intrinsics.k2 * r2 * r2;
    const double x_distorted =
        xn * radial + 2.0 * intrinsics.p1 * xn * yn +
        intrinsics.p2 * (r2 + 2.0 * xn * xn);
    const double y_distorted =
        yn * radial + intrinsics.p1 * (r2 + 2.0 * yn * yn) +
        2.0 * intrinsics.p2 * xn * yn;

    const double projected_x = intrinsics.fx * x_distorted + intrinsics.cx;
    const double projected_y = intrinsics.fy * y_distorted + intrinsics.cy;
    const double raw_rx = projected_x - observed_x;
    const double raw_ry = projected_y - observed_y;
    if (!::isfinite(raw_rx) || !::isfinite(raw_ry)) {
        return;
    }

    const double radial_slope = intrinsics.k1 + 2.0 * intrinsics.k2 * r2;
    const double dradial_dx = 2.0 * xn * radial_slope;
    const double dradial_dy = 2.0 * yn * radial_slope;
    const double dxd_dx = radial + xn * dradial_dx +
                          2.0 * intrinsics.p1 * yn + 6.0 * intrinsics.p2 * xn;
    const double dxd_dy = xn * dradial_dy +
                          2.0 * intrinsics.p1 * xn + 2.0 * intrinsics.p2 * yn;
    const double dyd_dx = yn * dradial_dx +
                          2.0 * intrinsics.p1 * xn + 2.0 * intrinsics.p2 * yn;
    const double dyd_dy = radial + yn * dradial_dy +
                          6.0 * intrinsics.p1 * yn + 2.0 * intrinsics.p2 * xn;

    const double j00 = intrinsics.fx * dxd_dx * inv_z;
    const double j01 = intrinsics.fx * dxd_dy * inv_z;
    const double j02 = -(j00 * px + j01 * py) * inv_z;
    const double j10 = intrinsics.fy * dyd_dx * inv_z;
    const double j11 = intrinsics.fy * dyd_dy * inv_z;
    const double j12 = -(j10 * px + j11 * py) * inv_z;

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
    int intrinsic_column = 0;
    if (options.optimize_focal) {
        result.intrinsic_jacobian[intrinsic_column] = scale * x_distorted;
        result.intrinsic_jacobian[k_max_intrinsic_params + intrinsic_column] =
            scale * y_distorted;
        ++intrinsic_column;
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
        const double fx = intrinsics.fx;
        const double fy = intrinsics.fy;
        result.intrinsic_jacobian[intrinsic_column] = scale * fx * xn * r2;
        result.intrinsic_jacobian[k_max_intrinsic_params + intrinsic_column] =
            scale * fy * yn * r2;
        ++intrinsic_column;
        result.intrinsic_jacobian[intrinsic_column] = scale * fx * xn * r2 * r2;
        result.intrinsic_jacobian[k_max_intrinsic_params + intrinsic_column] =
            scale * fy * yn * r2 * r2;
        ++intrinsic_column;
        result.intrinsic_jacobian[intrinsic_column] = scale * fx * (2.0 * xn * yn);
        result.intrinsic_jacobian[k_max_intrinsic_params + intrinsic_column] =
            scale * fy * (r2 + 2.0 * yn * yn);
        ++intrinsic_column;
        result.intrinsic_jacobian[intrinsic_column] = scale * fx * (r2 + 2.0 * xn * xn);
        result.intrinsic_jacobian[k_max_intrinsic_params + intrinsic_column] =
            scale * fy * (2.0 * xn * yn);
        ++intrinsic_column;
    }
    (void)intrinsic_column;
    result.valid = true;
}

}  // namespace aetherscan::ba::detail

#undef AETHERSCAN_HD
#undef AETHERSCAN_FORCEINLINE
