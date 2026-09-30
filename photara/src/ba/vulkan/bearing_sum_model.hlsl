#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer values;
[[vk::binding(1, 0)]] RWByteAddressBuffer cameras;
[[vk::binding(2, 0)]] RWByteAddressBuffer step;
[[vk::binding(3, 0)]] RWByteAddressBuffer total;
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared double part[256];

[numthreads(256, 1, 1)]
void main(uint group_thread : SV_GroupIndex) {
    double value = 0.0;
    for (uint i = group_thread; i < pc.n; i += 256u) value += load_double(values, i);
    part[group_thread] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint span = 128u; span > 0u; span >>= 1u) {
        if (group_thread < span) part[group_thread] += part[group_thread + span];
        GroupMemoryBarrierWithGroupSync();
    }
    if (group_thread == 0u) {
        double u0 = load_double(cameras, pc.p1 * 3u) - load_double(cameras, pc.p0 * 3u);
        double u1 = load_double(cameras, pc.p1 * 3u + 1u) - load_double(cameras, pc.p0 * 3u + 1u);
        double u2 = load_double(cameras, pc.p1 * 3u + 2u) - load_double(cameras, pc.p0 * 3u + 2u);
        double length = sqrt_d(fmax_d(u0 * u0 + u1 * u1 + u2 * u2, 0.0));
        double inv = 1.0 / fmax_d(length, 1.0e-10);
        double delta = 0.0;
        delta += u0 * inv * (load_double(step, pc.p1 * 3u) - load_double(step, pc.p0 * 3u));
        delta += u1 * inv * (load_double(step, pc.p1 * 3u + 1u) - load_double(step, pc.p0 * 3u + 1u));
        delta += u2 * inv * (load_double(step, pc.p1 * 3u + 2u) - load_double(step, pc.p0 * 3u + 2u));
        double baseline = unpack(pc.baseline_lo, pc.baseline_hi);
        store_double(total, 0u, part[0] - (length - baseline) * delta - 0.5 * delta * delta);
    }
}
