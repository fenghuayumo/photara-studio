#include "ba_common.hlsli"
#include "ba_trig.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer poses;
[[vk::binding(1, 0)]] RWByteAddressBuffer intrinsics;
[[vk::binding(2, 0)]] RWByteAddressBuffer pose_group;
[[vk::binding(3, 0)]] RWByteAddressBuffer tracks;
[[vk::binding(4, 0)]] RWByteAddressBuffer cameras;
[[vk::binding(5, 0)]] RWByteAddressBuffer track_ids;
[[vk::binding(6, 0)]] RWStructuredBuffer<double> observed_x;
[[vk::binding(7, 0)]] RWStructuredBuffer<double> observed_y;
[[vk::binding(8, 0)]] RWStructuredBuffer<double> weights;
[[vk::binding(9, 0)]] RWByteAddressBuffer output;

[[vk::push_constant]] ConstantBuffer<Push> pc;

struct Plane {
    double x, y, xx, xy, yx, yy;
    double dx[4];
    double dy[4];
};

Plane project_plane(uint model, double x, double y, double k1, double k2, double p1, double p2) {
    Plane outp;
    outp.x = 0.0;
    outp.y = 0.0;
    outp.xx = 0.0;
    outp.xy = 0.0;
    outp.yx = 0.0;
    outp.yy = 0.0;
    [unroll] for (uint i = 0u; i < 4u; ++i) {
        outp.dx[i] = 0.0;
        outp.dy[i] = 0.0;
    }
    double r2 = x * x + y * y;
    if (model == 1u) {
        double r = sqrt_d(r2);
        double theta = atan_d(r);
        double t2 = theta * theta;
        double poly = 1.0 + t2 * (k1 + t2 * (k2 + t2 * (p1 + t2 * p2)));
        double s = r > 1e-8 ? theta * poly / r : 1.0;
        double derivative = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * p1 + t2 * 9.0 * p2)));
        double slope = r > 1e-8 ? (derivative / (1.0 + r2) - s) / r2 : 2.0 * (k1 - 1.0 / 3.0);
        outp.x = x * s;
        outp.y = y * s;
        outp.xx = s + x * x * slope;
        outp.xy = x * y * slope;
        outp.yx = outp.xy;
        outp.yy = s + y * y * slope;
        double power = t2;
        [unroll] for (uint i = 0u; i < 4u; ++i) {
            double factor = r > 1e-8 ? theta * power / r : 0.0;
            outp.dx[i] = x * factor;
            outp.dy[i] = y * factor;
            power *= t2;
        }
    } else {
        double radial = 1.0 + k1 * r2 + k2 * r2 * r2;
        double slope = 2.0 * (k1 + 2.0 * k2 * r2);
        outp.x = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        outp.y = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
        outp.xx = radial + x * x * slope + 2.0 * p1 * y + 6.0 * p2 * x;
        outp.xy = x * y * slope + 2.0 * p1 * x + 2.0 * p2 * y;
        outp.yx = outp.xy;
        outp.yy = radial + y * y * slope + 6.0 * p1 * y + 2.0 * p2 * x;
        outp.dx[0] = x * r2;
        outp.dx[1] = x * r2 * r2;
        outp.dx[2] = 2.0 * x * y;
        outp.dx[3] = r2 + 2.0 * x * x;
        outp.dy[0] = y * r2;
        outp.dy[1] = y * r2 * r2;
        outp.dy[2] = r2 + 2.0 * y * y;
        outp.dy[3] = 2.0 * x * y;
    }
    return outp;
}

void clear_observation(uint observation) {
    [unroll] for (uint element = 0u; element < 37u; ++element)
        store_lin(output, observation, element, 0.0);
    store_lin_valid(output, observation, false);
}

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint stride = max(pc.groups, 1u) * 256u;
    double huber = unpack(pc.huber_lo, pc.huber_hi);
    double minimum_depth = unpack(pc.depth_lo, pc.depth_hi);
    bool opt_focal = (pc.flags & kOptFocal) != 0u;
    bool opt_aspect = (pc.flags & kOptAspect) != 0u;
    bool opt_principal = (pc.flags & kOptPrincipal) != 0u;
    bool opt_distortion = (pc.flags & kOptDistortion) != 0u;
    for (uint observation = group_id.x * 256u + group_thread; observation < pc.n;
         observation += stride) {
        clear_observation(observation);
        uint camera = load_uint(cameras, observation);
        uint track = load_uint(track_ids, observation);
        uint group = load_uint(pose_group, camera);
        uint pose_byte = camera * kPoseBytes;
        double qw = load_at(poses, pose_byte);
        double qx = load_at(poses, pose_byte + 8u);
        double qy = load_at(poses, pose_byte + 16u);
        double qz = load_at(poses, pose_byte + 24u);
        double cx = load_at(poses, pose_byte + 32u);
        double cy = load_at(poses, pose_byte + 40u);
        double cz = load_at(poses, pose_byte + 48u);
        uint intr_byte = group * kIntrinsicBytes;
        double fx = load_at(intrinsics, intr_byte);
        double fy = load_at(intrinsics, intr_byte + 8u);
        double icx = load_at(intrinsics, intr_byte + 16u);
        double icy = load_at(intrinsics, intr_byte + 24u);
        double k1 = load_at(intrinsics, intr_byte + 32u);
        double k2 = load_at(intrinsics, intr_byte + 40u);
        double p1 = load_at(intrinsics, intr_byte + 48u);
        double p2 = load_at(intrinsics, intr_byte + 56u);
        uint model = intrinsics.Load(intr_byte + 64u);
        uint track_byte = track * kPointBytes;
        double wx = load_at(tracks, track_byte);
        double wy = load_at(tracks, track_byte + 8u);
        double wz = load_at(tracks, track_byte + 16u);
        double r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
        double r01 = 2.0 * (qx * qy - qw * qz);
        double r02 = 2.0 * (qx * qz + qw * qy);
        double r10 = 2.0 * (qx * qy + qw * qz);
        double r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
        double r12 = 2.0 * (qy * qz - qw * qx);
        double r20 = 2.0 * (qx * qz - qw * qy);
        double r21 = 2.0 * (qy * qz + qw * qx);
        double r22 = 1.0 - 2.0 * (qx * qx + qy * qy);
        double dx = wx - cx;
        double dy = wy - cy;
        double dz = wz - cz;
        double px = r00 * dx + r01 * dy + r02 * dz;
        double py = r10 * dx + r11 * dy + r12 * dz;
        double pz = r20 * dx + r21 * dy + r22 * dz;
        bool equirect = model == 3u;
        if (equirect) {
            double length2 = px * px + py * py + pz * pz;
            if (!(length2 > minimum_depth * minimum_depth) || !isfinite_d(length2)) continue;
        } else if (!(pz > minimum_depth) || !isfinite_d(pz)) {
            continue;
        }
        double raw_x = 0.0;
        double raw_y = 0.0;
        double j00 = 0.0, j01 = 0.0, j02 = 0.0;
        double j10 = 0.0, j11 = 0.0, j12 = 0.0;
        double x_distorted = 0.0;
        double y_distorted = 0.0;
        Plane plane;
        plane.x = 0.0;
        plane.y = 0.0;
        plane.xx = 0.0;
        plane.xy = 0.0;
        plane.yx = 0.0;
        plane.yy = 0.0;
        [unroll] for (uint c = 0u; c < 4u; ++c) {
            plane.dx[c] = 0.0;
            plane.dy[c] = 0.0;
        }
        if (equirect) {
            if (!(fx > 1e-12) || !(fy > 1e-12)) continue;
            double observed_u = observed_x[observation];
            double observed_v = observed_y[observation];
            double azimuth = (observed_u - icx) / fx;
            double elevation = (observed_v - icy) / fy;
            double sin_lon = sin_d(azimuth);
            double cos_lon = cos_d(azimuth);
            double sin_lat = sin_d(elevation);
            double cos_lat = cos_d(elevation);
            double t1x = cos_lon;
            double t1y = 0.0;
            double t1z = -sin_lon;
            double t2x = -sin_lat * sin_lon;
            double t2y = cos_lat;
            double t2z = -sin_lat * cos_lon;
            double length = sqrt_d(px * px + py * py + pz * pz);
            if (!(length > 1e-12) || !isfinite_d(length)) continue;
            double inv_length = 1.0 / length;
            double bx = px * inv_length;
            double by = py * inv_length;
            double bz = pz * inv_length;
            double d00 = (1.0 - bx * bx) * inv_length;
            double d01 = -bx * by * inv_length;
            double d02 = -bx * bz * inv_length;
            double d10 = -by * bx * inv_length;
            double d11 = (1.0 - by * by) * inv_length;
            double d12 = -by * bz * inv_length;
            double d20 = -bz * bx * inv_length;
            double d21 = -bz * by * inv_length;
            double d22 = (1.0 - bz * bz) * inv_length;
            raw_x = fx * (bx * t1x + by * t1y + bz * t1z);
            raw_y = fy * (bx * t2x + by * t2y + bz * t2z);
            j00 = fx * (t1x * d00 + t1y * d10 + t1z * d20);
            j01 = fx * (t1x * d01 + t1y * d11 + t1z * d21);
            j02 = fx * (t1x * d02 + t1y * d12 + t1z * d22);
            j10 = fy * (t2x * d00 + t2y * d10 + t2z * d20);
            j11 = fy * (t2x * d01 + t2y * d11 + t2z * d21);
            j12 = fy * (t2x * d02 + t2y * d12 + t2z * d22);
            if (!isfinite_d(raw_x) || !isfinite_d(raw_y) || !isfinite_d(j00) ||
                !isfinite_d(j01) || !isfinite_d(j02) || !isfinite_d(j10) ||
                !isfinite_d(j11) || !isfinite_d(j12))
                continue;
        } else {
            double inv_z = 1.0 / pz;
            double xn = px * inv_z;
            double yn = py * inv_z;
            plane = project_plane(model, xn, yn, k1, k2, p1, p2);
            x_distorted = plane.x;
            y_distorted = plane.y;
            double projected_x = fx * x_distorted + icx;
            double projected_y = fy * y_distorted + icy;
            raw_x = projected_x - observed_x[observation];
            raw_y = projected_y - observed_y[observation];
            j00 = fx * plane.xx * inv_z;
            j01 = fx * plane.xy * inv_z;
            j02 = -(j00 * px + j01 * py) * inv_z;
            j10 = fy * plane.yx * inv_z;
            j11 = fy * plane.yy * inv_z;
            j12 = -(j10 * px + j11 * py) * inv_z;
        }
        if (!isfinite_d(raw_x) || !isfinite_d(raw_y)) continue;
        double robust = weights[observation];
        if (huber > 0.0) {
            double norm = sqrt_d(raw_x * raw_x + raw_y * raw_y);
            if (norm > huber) robust *= huber / norm;
        }
        double scale = sqrt_d(robust);
        store_lin(output, observation, 0u, scale * raw_x);
        store_lin(output, observation, 1u, scale * raw_y);
        store_lin(output, observation, 36u, robust);
        double projection_j[6] = {j00, j01, j02, j10, j11, j12};
        double rotation_j[9] = {0.0, pz, -py, -pz, 0.0, px, py, -px, 0.0};
        double center_j[9] = {-r00, -r01, -r02, -r10, -r11, -r12, -r20, -r21, -r22};
        double track_j[9] = {r00, r01, r02, r10, r11, r12, r20, r21, r22};
        [unroll] for (uint row = 0u; row < 2u; ++row) {
            [unroll] for (uint column = 0u; column < 3u; ++column) {
                double rotation_value = 0.0;
                double center_value = 0.0;
                double track_value = 0.0;
                [unroll] for (uint k = 0u; k < 3u; ++k) {
                    double projection = projection_j[row * 3u + k];
                    rotation_value += projection * rotation_j[k * 3u + column];
                    center_value += projection * center_j[k * 3u + column];
                    track_value += projection * track_j[k * 3u + column];
                }
                store_lin(output, observation, 2u + row * 6u + column, scale * rotation_value);
                store_lin(output, observation, 2u + row * 6u + 3u + column, scale * center_value);
                store_lin(output, observation, 14u + row * 3u + column, scale * track_value);
            }
        }
        if (!equirect) {
            uint column = 0u;
            if (opt_focal) {
                if (opt_aspect) {
                    store_lin(output, observation, 20u + column, scale * x_distorted);
                    store_lin(output, observation, 28u + column, 0.0);
                    ++column;
                    store_lin(output, observation, 20u + column, 0.0);
                    store_lin(output, observation, 28u + column, scale * y_distorted);
                    ++column;
                } else {
                    store_lin(output, observation, 20u + column, scale * x_distorted);
                    store_lin(output, observation, 28u + column, scale * y_distorted);
                    ++column;
                }
            }
            if (opt_principal) {
                store_lin(output, observation, 20u + column, scale);
                store_lin(output, observation, 28u + column, 0.0);
                ++column;
                store_lin(output, observation, 20u + column, 0.0);
                store_lin(output, observation, 28u + column, scale);
                ++column;
            }
            if (opt_distortion) {
                [unroll] for (uint coeff = 0u; coeff < 4u; ++coeff) {
                    store_lin(output, observation, 20u + column,
                              scale * fx * plane.dx[coeff]);
                    store_lin(output, observation, 28u + column,
                              scale * fy * plane.dy[coeff]);
                    ++column;
                }
            }
        }
        store_lin_valid(output, observation, true);
    }
}
