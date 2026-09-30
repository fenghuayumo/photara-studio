#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer costs;
[[vk::binding(1, 0)]] RWByteAddressBuffer cameras;
[[vk::binding(2, 0)]] RWByteAddressBuffer total;
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared double part[256];

[numthreads(256, 1, 1)]
void main(uint group_thread : SV_GroupIndex) {
    double value = 0.0;
    for (uint i = group_thread; i < pc.n; i += 256u) {
        value += load_double(costs, i);
    }
    part[group_thread] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint span = 128u; span > 0u; span >>= 1u) {
        if (group_thread < span) part[group_thread] += part[group_thread + span];
        GroupMemoryBarrierWithGroupSync();
    }
    if (group_thread == 0u) {
        double distance2 = 0.0;
        [unroll] for (uint k = 0u; k < 3u; ++k) {
            double delta = load_double(cameras, pc.p0 * 3u + k) - load_double(cameras, pc.p1 * 3u + k);
            distance2 += delta * delta;
        }
        double residual = sqrt_d(distance2) - unpack(pc.baseline_lo, pc.baseline_hi);
        store_double(total, 0u, part[0] + 0.5 * residual * residual);
    }
}
