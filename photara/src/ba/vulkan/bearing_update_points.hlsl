#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer lin;
[[vk::binding(1, 0)]] RWByteAddressBuffer observations;
[[vk::binding(2, 0)]] RWByteAddressBuffer offsets;
[[vk::binding(3, 0)]] RWByteAddressBuffer indices;
[[vk::binding(4, 0)]] RWByteAddressBuffer inverse;
[[vk::binding(5, 0)]] RWByteAddressBuffer rhs;
[[vk::binding(6, 0)]] RWByteAddressBuffer step;
[[vk::binding(7, 0)]] RWByteAddressBuffer points;
[[vk::binding(8, 0)]] RWByteAddressBuffer candidate;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint step_count = max(pc.groups, 1u) * 256u;
    for (uint p = id.x; p < pc.n; p += step_count) {
        double b0 = load_double(rhs, p * 3u);
        double b1 = load_double(rhs, p * 3u + 1u);
        double b2 = load_double(rhs, p * 3u + 2u);
        uint begin = load_uint(offsets, p);
        uint end = load_uint(offsets, p + 1u);
        for (uint j = begin; j < end; ++j) {
            uint obs = load_uint(indices, j);
            uint camera = observations.Load(obs * 32u);
            double s0 = load_double(step, camera * 3u);
            double s1 = load_double(step, camera * 3u + 1u);
            double s2 = load_double(step, camera * 3u + 2u);
            uint base = obs * 12u;
            b0 += load_double(lin, base) * s0 + load_double(lin, base + 1u) * s1 + load_double(lin, base + 2u) * s2;
            b1 += load_double(lin, base + 3u) * s0 + load_double(lin, base + 4u) * s1 + load_double(lin, base + 5u) * s2;
            b2 += load_double(lin, base + 6u) * s0 + load_double(lin, base + 7u) * s1 + load_double(lin, base + 8u) * s2;
        }
        [unroll] for (uint row = 0u; row < 3u; ++row) {
            double value = load_double(points, p * 3u + row);
            [unroll] for (uint k = 0u; k < 3u; ++k) {
                double bk = k == 0u ? b0 : (k == 1u ? b1 : b2);
                value += load_double(inverse, p * 9u + row * 3u + k) * bk;
            }
            store_double(candidate, p * 3u + row, value);
        }
    }
}
