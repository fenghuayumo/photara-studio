#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> key_lo;
[[vk::binding(1, 0)]] StructuredBuffer<uint> key_hi;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> hist;

uint extract_digit(uint lo, uint hi, uint shift) {
    uint word = shift < 32 ? lo : hi;
    uint local_shift = shift < 32 ? shift : shift - 32;
    return (word >> local_shift) & 255u;
}

groupshared uint bins[256];

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint n = pc.u0;
    uint num_blocks = pc.u1;
    uint shift = pc.u2;
    uint key_is_64 = pc.u3;
    uint tid = group_thread;
    // One thread per key into a shared histogram: the previous "one thread per
    // digit" shape re-read the whole 1024-key block once per digit, so every
    // key was loaded 256 times.
    bins[tid] = 0;
    GroupMemoryBarrierWithGroupSync();
    uint start = group_id.x * 1024;
    [loop] for (uint i = 0; i < 4; ++i) {
        uint index = start + i * 256u + tid;
        if (index < n) {
            uint hi = key_is_64 != 0 ? key_hi[index] : 0;
            InterlockedAdd(bins[extract_digit(key_lo[index], hi, shift)], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    hist[tid * num_blocks + group_id.x] = bins[tid];
}
