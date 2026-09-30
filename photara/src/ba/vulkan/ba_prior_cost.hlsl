#include "ba_common.hlsli"
#include "ba_atomic.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer intrinsics;
[[vk::binding(1, 0)]] RWByteAddressBuffer initial;
[[vk::binding(2, 0)]] RWByteAddressBuffer constant;
[[vk::binding(3, 0)]] RWStructuredBuffer<double> total;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(1, 1, 1)]
void main() {
    double prior = unpack(pc.prior_lo, pc.prior_hi);
    if ((pc.flags & kOptFocal) == 0u || !(prior > 0.0)) return;
    double observations = unpack(pc.obs_lo, pc.obs_hi);
    bool aspect = (pc.flags & kOptAspect) != 0u;
    double local = 0.0;
    for (uint group = 0u; group < pc.n; ++group) {
        if (load_uint(constant, group) != 0u) continue;
        uint byte = group * kIntrinsicBytes;
        if (aspect) {
            double fx0 = fmax_d(load_at(initial, byte), 1.0);
            double fy0 = fmax_d(load_at(initial, byte + 8u), 1.0);
            double rx = (load_at(intrinsics, byte) - fx0) / fx0;
            double ry = (load_at(intrinsics, byte + 8u) - fy0) / fy0;
            local += 0.25 * prior * observations * (rx * rx + ry * ry);
        } else {
            double fx0 = load_at(initial, byte);
            double fy0 = load_at(initial, byte + 8u);
            double f0 = fmax_d(0.5 * (fx0 + fy0), 1.0);
            double focal = 0.5 * (load_at(intrinsics, byte) + load_at(intrinsics, byte + 8u));
            double r = (focal - f0) / f0;
            local += 0.5 * prior * observations * r * r;
        }
    }
    atomic_add_f64(total[0], 1u, 0u, local);
}
