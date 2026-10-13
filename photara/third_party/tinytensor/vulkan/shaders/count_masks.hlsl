#include "common.hlsli"
struct PushConstants { uint count, offset0, offset1, offset2; };
[[vk::binding(0, 0)]] RWByteAddressBuffer mask0;
[[vk::binding(1, 0)]] RWByteAddressBuffer mask1;
[[vk::binding(2, 0)]] RWByteAddressBuffer mask2;
[[vk::binding(3, 0)]] RWByteAddressBuffer counts;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
groupshared uint sums[TT_GROUP_SIZE];
[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint index = group.x * TT_GROUP_SIZE + lane;
    uint value = 0u;
    if (index < pc.count) {
        if (group.y == 0u) value = load_u8(mask0, pc.offset0 + index);
        else if (group.y == 1u) value = load_u8(mask1, pc.offset1 + index);
        else value = load_u8(mask2, pc.offset2 + index);
    }
    sums[lane] = value != 0u ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = TT_GROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u && sums[0] != 0u) counts.InterlockedAdd(group.y * 4u, sums[0]);
}
