#include "ba_common.hlsli"
#include "ba_trig.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer poses;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> step;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    bool fix_first = (pc.flags & kFixPose) != 0u;
    bool opt_rot = (pc.flags & kOptRot) != 0u;
    bool opt_trans = (pc.flags & kOptTrans) != 0u;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint camera = group_id.x * 256u + group_thread; camera < pc.n; camera += stride) {
        if (fix_first && camera == 0u) continue;
        uint byte = camera * kPoseBytes;
        double s0 = step[camera * 6u];
        double s1 = step[camera * 6u + 1u];
        double s2 = step[camera * 6u + 2u];
        if (opt_rot) {
            double qw = load_at(poses, byte);
            double qx = load_at(poses, byte + 8u);
            double qy = load_at(poses, byte + 16u);
            double qz = load_at(poses, byte + 24u);
            double angle = sqrt_d(s0 * s0 + s1 * s1 + s2 * s2);
            double dw = 1.0;
            double dx = 0.5 * s0;
            double dy = 0.5 * s1;
            double dz = 0.5 * s2;
            if (angle > 1e-12) {
                dw = cos_d(0.5 * angle);
                double k = sin_d(0.5 * angle) / angle;
                dx = k * s0;
                dy = k * s1;
                dz = k * s2;
            }
            double nw = dw * qw - dx * qx - dy * qy - dz * qz;
            double nx = dw * qx + dx * qw + dy * qz - dz * qy;
            double ny = dw * qy - dx * qz + dy * qw + dz * qx;
            double nz = dw * qz + dx * qy - dy * qx + dz * qw;
            double inv = 1.0 / sqrt_d(nw * nw + nx * nx + ny * ny + nz * nz);
            store_at(poses, byte, nw * inv);
            store_at(poses, byte + 8u, nx * inv);
            store_at(poses, byte + 16u, ny * inv);
            store_at(poses, byte + 24u, nz * inv);
        }
        if (opt_trans) {
            store_at(poses, byte + 32u, load_at(poses, byte + 32u) + step[camera * 6u + 3u]);
            store_at(poses, byte + 40u, load_at(poses, byte + 40u) + step[camera * 6u + 4u]);
            store_at(poses, byte + 48u, load_at(poses, byte + 48u) + step[camera * 6u + 5u]);
        }
    }
}
