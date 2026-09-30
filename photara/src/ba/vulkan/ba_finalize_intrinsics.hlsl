#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<double> intrinsic_h;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> intrinsic_b;
[[vk::binding(2, 0)]] RWByteAddressBuffer intrinsics;
[[vk::binding(3, 0)]] RWByteAddressBuffer initial;
[[vk::binding(4, 0)]] RWByteAddressBuffer constant;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    double damping = unpack(pc.damp_lo, pc.damp_hi);
    double prior = unpack(pc.prior_lo, pc.prior_hi);
    double observations = unpack(pc.obs_lo, pc.obs_hi);
    uint dof = pc.dof;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint group = group_id.x * 256u + group_thread; group < pc.n; group += stride) {
        uint base = group * dof * dof;
        uint rhs = group * dof;
        for (uint diagonal = 0u; diagonal < dof; ++diagonal) {
            uint index = base + diagonal * dof + diagonal;
            double value = intrinsic_h[index];
            intrinsic_h[index] = value + damping * (value + 1.0);
        }
        if (load_uint(constant, group) != 0u) {
            for (uint element = 0u; element < dof * dof; ++element) intrinsic_h[base + element] = 0.0;
            for (uint diagonal = 0u; diagonal < dof; ++diagonal) {
                intrinsic_h[base + diagonal * dof + diagonal] = 1.0;
                intrinsic_b[rhs + diagonal] = 0.0;
            }
            continue;
        }
        if ((pc.flags & kOptFocal) == 0u || !(prior > 0.0)) continue;
        uint byte = group * kIntrinsicBytes;
        double fx = load_at(intrinsics, byte);
        double fy = load_at(intrinsics, byte + 8u);
        double fx0 = fmax_d(load_at(initial, byte), 1.0);
        double fy0 = fmax_d(load_at(initial, byte + 8u), 1.0);
        if ((pc.flags & kOptAspect) != 0u) {
            double weight_x = 0.5 * prior * observations / (fx0 * fx0);
            double weight_y = 0.5 * prior * observations / (fy0 * fy0);
            intrinsic_h[base] += weight_x;
            intrinsic_h[base + dof + 1u] += weight_y;
            intrinsic_b[rhs] -= weight_x * (fx - fx0);
            intrinsic_b[rhs + 1u] -= weight_y * (fy - fy0);
        } else {
            double f0 = fmax_d(0.5 * (fx0 + fy0), 1.0);
            double focal = 0.5 * (fx + fy);
            double weight = prior * observations / (f0 * f0);
            intrinsic_h[base] += weight;
            intrinsic_b[rhs] -= weight * (focal - f0);
        }
    }
}
