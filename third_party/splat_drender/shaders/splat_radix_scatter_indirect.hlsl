#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> in_lo;
[[vk::binding(1, 0)]] StructuredBuffer<uint> in_hi;
[[vk::binding(2, 0)]] StructuredBuffer<uint> in_val;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> out_lo;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> out_hi;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> out_val;
[[vk::binding(6, 0)]] StructuredBuffer<uint> hist;
[[vk::binding(7, 0)]] StructuredBuffer<uint> hist_scan;
[[vk::binding(8, 0)]] StructuredBuffer<uint> control;

uint extract_digit(uint lo, uint hi, uint shift) {
    uint word = shift < 32u ? lo : hi;
    uint local_shift = shift < 32u ? shift : shift - 32u;
    return (word >> local_shift) & 255u;
}

groupshared uint digits[256];
groupshared uint chunk_base[256];
groupshared uint chunk_count[256];

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    if (control[2] != 0u) return;
    uint n = control[pc.u5];
    uint num_blocks = pc.u1;
    uint tid = group_thread;
    chunk_base[tid] = 0u;
    GroupMemoryBarrierWithGroupSync();
    [loop] for (uint chunk = 0u; chunk < 4u; ++chunk) {
        uint index = group_id.x * 1024u + chunk * 256u + tid;
        bool valid = index < n;
        uint lo = valid ? in_lo[index] : 0u;
        uint hi = valid && pc.u3 != 0u ? in_hi[index] : 0u;
        uint value = valid ? in_val[index] : 0u;
        uint digit = valid ? extract_digit(lo, hi, pc.u2) : 0u;
        digits[tid] = valid ? digit : 0xffffffffu;
        GroupMemoryBarrierWithGroupSync();
        uint rank = 0u;
        if (valid) {
            [loop] for (uint earlier = 0u; earlier < tid; ++earlier)
                rank += digits[earlier] == digit ? 1u : 0u;
            uint hist_index = digit * num_blocks + group_id.x;
            uint exclusive = hist_scan[pc.u4 + hist_index] - hist[hist_index];
            uint dest = exclusive + chunk_base[digit] + rank;
            out_lo[dest] = lo;
            if (pc.u3 != 0u) out_hi[dest] = hi;
            out_val[dest] = value;
        }
        GroupMemoryBarrierWithGroupSync();
        chunk_count[tid] = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (valid) InterlockedAdd(chunk_count[digit], 1u);
        GroupMemoryBarrierWithGroupSync();
        chunk_base[tid] += chunk_count[tid];
        GroupMemoryBarrierWithGroupSync();
    }
}
