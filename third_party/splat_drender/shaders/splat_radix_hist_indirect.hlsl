#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> key_lo;
[[vk::binding(1, 0)]] StructuredBuffer<uint> key_hi;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> hist;
[[vk::binding(3, 0)]] StructuredBuffer<uint> control;

uint extract_digit(uint lo, uint hi, uint shift) {
    uint word = shift < 32u ? lo : hi;
    uint local_shift = shift < 32u ? shift : shift - 32u;
    return (word >> local_shift) & 255u;
}

groupshared uint bins[256];

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint n = control[pc.u5];
    uint num_blocks = pc.u1;
    uint tid = group_thread;
    bins[tid] = 0u;
    GroupMemoryBarrierWithGroupSync();
    uint start = group_id.x * 1024u;
    [loop] for (uint i = 0u; i < 4u; ++i) {
        uint index = start + i * 256u + tid;
        if (control[2] == 0u && index < n) {
            uint hi = pc.u3 != 0u ? key_hi[index] : 0u;
            InterlockedAdd(bins[extract_digit(key_lo[index], hi, pc.u2)], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    hist[tid * num_blocks + group_id.x] = bins[tid];
}
