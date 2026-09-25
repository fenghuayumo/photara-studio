#include "splat_math.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<uint> data;

groupshared uint scratch[256];

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint n = pc.u0;
    uint mode = pc.u1;
    uint in_offset = pc.u2;
    uint out_offset = pc.u3;
    uint sums_offset = pc.u4;

    if (mode == 1) {
        uint index = group_id.x * 256 + group_thread;
        if (index >= n) return;
        uint block = index >> 8;
        if (block == 0) return;
        data[out_offset + index] += data[sums_offset + block - 1];
        return;
    }

    uint index = group_id.x * 256 + group_thread;
    uint value = index < n ? data[in_offset + index] : 0;
    scratch[group_thread] = value;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint offset = 1; offset < 256; offset <<= 1) {
        uint addend = group_thread >= offset ? scratch[group_thread - offset] : 0;
        GroupMemoryBarrierWithGroupSync();
        if (group_thread >= offset) scratch[group_thread] += addend;
        GroupMemoryBarrierWithGroupSync();
    }
    if (index < n) data[out_offset + index] = scratch[group_thread];
    if (group_thread == 255) data[sums_offset + group_id.x] = scratch[255];
}
