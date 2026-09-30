#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer intrinsics;
[[vk::binding(1, 0)]] RWByteAddressBuffer initial;
[[vk::binding(2, 0)]] RWByteAddressBuffer constant;
[[vk::binding(3, 0)]] RWStructuredBuffer<double> step;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    double min_ratio = unpack(pc.min_f_lo, pc.min_f_hi);
    double max_ratio = unpack(pc.max_f_lo, pc.max_f_hi);
    uint dof = pc.dof;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint group = group_id.x * 256u + group_thread; group < pc.n; group += stride) {
        if (load_uint(constant, group) != 0u) continue;
        uint byte = group * kIntrinsicBytes;
        uint index = 0u;
        uint sbase = pc.camera_values + group * dof;
        if ((pc.flags & kOptFocal) != 0u) {
            double fx = load_at(intrinsics, byte) + step[sbase + index];
            ++index;
            double fy;
            if ((pc.flags & kOptAspect) != 0u) {
                fy = load_at(intrinsics, byte + 8u) + step[sbase + index];
                ++index;
            } else {
                fy = load_at(intrinsics, byte + 8u) + step[sbase + index - 1u];
            }
            double fx0 = fmax_d(load_at(initial, byte), 1.0);
            double fy0 = fmax_d(load_at(initial, byte + 8u), 1.0);
            double min_x = fmax_d(1.0, fx0 * min_ratio);
            double max_x = fx0 * max_ratio;
            double min_y = fmax_d(1.0, fy0 * min_ratio);
            double max_y = fy0 * max_ratio;
            store_at(intrinsics, byte, fmin_d(fmax_d(fx, min_x), max_x));
            store_at(intrinsics, byte + 8u, fmin_d(fmax_d(fy, min_y), max_y));
        }
        if ((pc.flags & kOptPrincipal) != 0u) {
            store_at(intrinsics, byte + 16u, load_at(intrinsics, byte + 16u) + step[sbase + index]);
            ++index;
            store_at(intrinsics, byte + 24u, load_at(intrinsics, byte + 24u) + step[sbase + index]);
            ++index;
        }
        if ((pc.flags & kOptDistortion) != 0u) {
            store_at(intrinsics, byte + 32u, load_at(intrinsics, byte + 32u) + step[sbase + index]);
            ++index;
            store_at(intrinsics, byte + 40u, load_at(intrinsics, byte + 40u) + step[sbase + index]);
            ++index;
            store_at(intrinsics, byte + 48u, load_at(intrinsics, byte + 48u) + step[sbase + index]);
            ++index;
            store_at(intrinsics, byte + 56u, load_at(intrinsics, byte + 56u) + step[sbase + index]);
        }
    }
}
