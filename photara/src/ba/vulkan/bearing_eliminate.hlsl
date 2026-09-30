#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer lin;
[[vk::binding(1, 0)]] RWByteAddressBuffer observations;
[[vk::binding(2, 0)]] RWByteAddressBuffer inverse;
[[vk::binding(3, 0)]] RWByteAddressBuffer eliminated;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint step = max(pc.groups, 1u) * 256u;
    for (uint i = id.x; i < pc.n; i += step) {
        uint track = observations.Load(i * 32u + 4u);
        [unroll] for (uint row = 0u; row < 3u; ++row) {
            [unroll] for (uint col = 0u; col < 3u; ++col) {
                double value = 0.0;
                [unroll] for (uint k = 0u; k < 3u; ++k) {
                    value += load_double(lin, i * 12u + row * 3u + k) *
                             load_double(inverse, track * 9u + k * 3u + col);
                }
                store_double(eliminated, i * 9u + row * 3u + col, value);
            }
        }
    }
}
