#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer lin;
[[vk::binding(1, 0)]] RWByteAddressBuffer observations;
[[vk::binding(2, 0)]] RWByteAddressBuffer step;
[[vk::binding(3, 0)]] RWByteAddressBuffer points;
[[vk::binding(4, 0)]] RWByteAddressBuffer candidate;
[[vk::binding(5, 0)]] RWByteAddressBuffer values;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint i = id.x; i < pc.n; i += stride) {
        uint camera = observations.Load(i * 32u);
        uint track = observations.Load(i * 32u + 4u);
        double v0 = load_double(candidate, track * 3u) - load_double(points, track * 3u) - load_double(step, camera * 3u);
        double v1 = load_double(candidate, track * 3u + 1u) - load_double(points, track * 3u + 1u) - load_double(step, camera * 3u + 1u);
        double v2 = load_double(candidate, track * 3u + 2u) - load_double(points, track * 3u + 2u) - load_double(step, camera * 3u + 2u);
        double delta[3] = {v0, v1, v2};
        double value = 0.0;
        [unroll] for (uint row = 0u; row < 3u; ++row) {
            value += load_double(lin, i * 12u + 9u + row) * delta[row];
            [unroll] for (uint col = 0u; col < 3u; ++col) {
                value -= 0.5 * delta[row] * load_double(lin, i * 12u + row * 3u + col) * delta[col];
            }
        }
        store_double(values, i, value);
    }
}
