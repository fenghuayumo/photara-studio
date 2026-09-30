#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<double> hessian;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> rhs;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    double damping = unpack(pc.damp_lo, pc.damp_hi);
    bool fix_first = (pc.flags & kFixPoint) != 0u;
    bool optimize = (pc.flags & kOptPoints) != 0u;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint track = group_id.x * 256u + group_thread; track < pc.n; track += stride) {
        uint base = track * 9u;
        if (!optimize || (fix_first && track == 0u)) {
            [unroll] for (uint k = 0u; k < 9u; ++k) hessian[base + k] = 0.0;
            [unroll] for (uint k = 0u; k < 3u; ++k) rhs[track * 3u + k] = 0.0;
            continue;
        }
        [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
            uint index = base + axis * 3u + axis;
            double diagonal = hessian[index];
            hessian[index] = diagonal + damping * (diagonal + 1.0);
        }
        double a = hessian[base];
        double b = hessian[base + 1u];
        double c = hessian[base + 2u];
        double d = hessian[base + 4u];
        double e = hessian[base + 5u];
        double f = hessian[base + 8u];
        double c00 = d * f - e * e;
        double c01 = c * e - b * f;
        double c02 = b * e - c * d;
        double c11 = a * f - c * c;
        double c12 = b * c - a * e;
        double c22 = a * d - b * b;
        double determinant = a * c00 + b * c01 + c * c02;
        if (!(determinant > 1e-30) || !isfinite_d(determinant)) {
            [unroll] for (uint k = 0u; k < 9u; ++k) hessian[base + k] = 0.0;
            continue;
        }
        double s = 1.0 / determinant;
        hessian[base] = c00 * s;
        hessian[base + 1u] = c01 * s;
        hessian[base + 2u] = c02 * s;
        hessian[base + 3u] = hessian[base + 1u];
        hessian[base + 4u] = c11 * s;
        hessian[base + 5u] = c12 * s;
        hessian[base + 6u] = hessian[base + 2u];
        hessian[base + 7u] = hessian[base + 5u];
        hessian[base + 8u] = c22 * s;
    }
}
