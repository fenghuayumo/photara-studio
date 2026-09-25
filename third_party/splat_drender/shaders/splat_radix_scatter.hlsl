#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> in_lo;
[[vk::binding(1, 0)]] StructuredBuffer<uint> in_hi;
[[vk::binding(2, 0)]] StructuredBuffer<uint> in_val;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> out_lo;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> out_hi;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> out_val;
[[vk::binding(6, 0)]] StructuredBuffer<uint> hist;
[[vk::binding(7, 0)]] StructuredBuffer<uint> hist_scan;

uint extract_digit(uint lo, uint hi, uint shift) {
    uint word = shift < 32 ? lo : hi;
    uint local_shift = shift < 32 ? shift : shift - 32;
    return (word >> local_shift) & 255u;
}

groupshared uint digits[256];
groupshared uint chunk_base[256];
groupshared uint chunk_count[256];

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint n = pc.u0;
    uint num_blocks = pc.u1;
    uint shift = pc.u2;
    uint key_is_64 = pc.u3;
    uint tid = group_thread;
    chunk_base[tid] = 0;
    GroupMemoryBarrierWithGroupSync();

    [loop] for (uint chunk = 0; chunk < 4; ++chunk) {
        uint index = group_id.x * 1024 + chunk * 256 + tid;
        bool valid = index < n;
        uint lo = 0;
        uint hi = 0;
        uint value = 0;
        uint digit = 0;
        if (valid) {
            lo = in_lo[index];
            hi = key_is_64 != 0 ? in_hi[index] : 0;
            value = in_val[index];
            digit = extract_digit(lo, hi, shift);
        }
        // 0xFFFFFFFF never matches a real digit, so a tail block's empty lanes
        // cannot inflate the rank of a valid key that shares its digit.
        digits[tid] = valid ? digit : 0xFFFFFFFFu;
        GroupMemoryBarrierWithGroupSync();

        uint rank = 0;
        if (valid) {
            [loop] for (uint earlier = 0; earlier < tid; ++earlier) {
                rank += digits[earlier] == digit ? 1u : 0u;
            }
            uint hist_index = digit * num_blocks + group_id.x;
            uint exclusive = hist_scan[pc.u4 + hist_index] - hist[hist_index];
            uint dest = exclusive + chunk_base[digit] + rank;
            out_lo[dest] = lo;
            if (key_is_64 != 0) out_hi[dest] = hi;
            out_val[dest] = value;
        }
        GroupMemoryBarrierWithGroupSync();
        chunk_count[tid] = 0;
        GroupMemoryBarrierWithGroupSync();
        // One atomic per key instead of a serial 256-iteration loop in lane 0.
        if (valid) InterlockedAdd(chunk_count[digit], 1u);
        GroupMemoryBarrierWithGroupSync();
        chunk_base[tid] += chunk_count[tid];
        GroupMemoryBarrierWithGroupSync();
    }
}
