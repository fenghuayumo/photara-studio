#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer lin;
[[vk::binding(1, 0)]] RWByteAddressBuffer offsets;
[[vk::binding(2, 0)]] RWByteAddressBuffer indices;
[[vk::binding(3, 0)]] RWByteAddressBuffer cameras;
[[vk::binding(4, 0)]] RWByteAddressBuffer scale;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint step = max(pc.groups, 1u) * 256u;
    for (uint i = id.x; i < pc.n; i += step) {
        double diagonal0 = 0.0;
        double diagonal1 = 0.0;
        double diagonal2 = 0.0;
        uint begin = load_uint(offsets, i);
        uint end = load_uint(offsets, i + 1u);
        for (uint j = begin; j < end; ++j) {
            uint obs = load_uint(indices, j);
            diagonal0 += load_double(lin, obs * 12u);
            diagonal1 += load_double(lin, obs * 12u + 4u);
            diagonal2 += load_double(lin, obs * 12u + 8u);
        }
        if ((pc.flags & 2u) != 0u && (i == pc.p0 || i == pc.p1)) {
            double u0 = load_double(cameras, pc.p1 * 3u) - load_double(cameras, pc.p0 * 3u);
            double u1 = load_double(cameras, pc.p1 * 3u + 1u) - load_double(cameras, pc.p0 * 3u + 1u);
            double u2 = load_double(cameras, pc.p1 * 3u + 2u) - load_double(cameras, pc.p0 * 3u + 2u);
            double length2 = fmax_d(u0 * u0 + u1 * u1 + u2 * u2, 1.0e-20);
            diagonal0 += u0 * u0 / length2;
            diagonal1 += u1 * u1 / length2;
            diagonal2 += u2 * u2 / length2;
        }
        double s0 = 1.0 + sqrt_d(diagonal0);
        double s1 = 1.0 + sqrt_d(diagonal1);
        double s2 = 1.0 + sqrt_d(diagonal2);
        store_double(scale, i * 3u, s0 * s0);
        store_double(scale, i * 3u + 1u, s1 * s1);
        store_double(scale, i * 3u + 2u, s2 * s2);
    }
}
