#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<double> inverse;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> rhs;
[[vk::binding(2, 0)]] RWStructuredBuffer<double> reduced;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint track = group_id.x * 256u + group_thread; track < pc.n; track += stride) {
        uint h = track * 9u;
        uint b = track * 3u;
        [unroll] for (uint row = 0u; row < 3u; ++row) {
            reduced[b + row] = inverse[h + row * 3u] * rhs[b] +
                               inverse[h + row * 3u + 1u] * rhs[b + 1u] +
                               inverse[h + row * 3u + 2u] * rhs[b + 2u];
        }
    }
}
