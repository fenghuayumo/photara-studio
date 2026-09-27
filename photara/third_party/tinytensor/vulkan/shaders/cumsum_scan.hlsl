#include "common.hlsli"

// Inclusive scan of contiguous Float32 rows.
// mode 0: each group scans one 1024-element chunk and writes the chunk total.
// mode 1: each group turns one row of chunk totals (<=1024) into exclusive prefixes.
// mode 2: add those prefixes back onto the chunk scans.

struct PushConstants
{
    uint mode;
    uint rows;
    uint length;
    uint num_chunks;
    uint src_offset;
    uint dst_offset;
    uint sums_offset;
    uint chunk;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer dst;
[[vk::binding(2, 0)]] RWByteAddressBuffer sums;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

groupshared float totals[TT_GROUP_SIZE];

void scan_thread_totals(uint lane, inout float running)
{
    totals[lane] = running;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint offset = 1u; offset < TT_GROUP_SIZE; offset <<= 1u)
    {
        float addend = 0.0f;
        if (lane >= offset) addend = totals[lane - offset];
        GroupMemoryBarrierWithGroupSync();
        running += addend;
        totals[lane] = running;
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID)
{
    const uint lane = group_thread.x;
    if (pc.mode == 2u)
    {
        const uint index = dtid.x;
        const uint total = pc.rows * pc.length;
        if (index >= total) return;
        const uint row = index / pc.length;
        const uint col = index - row * pc.length;
        const uint chunk_index = col / pc.chunk;
        float value = asfloat(src.Load(pc.src_offset + index * 4u));
        if (chunk_index != 0u)
        {
            const float prefix = asfloat(sums.Load(pc.sums_offset + (row * pc.num_chunks + chunk_index) * 4u));
            value += prefix;
        }
        dst.Store(pc.dst_offset + index * 4u, asuint(value));
        return;
    }

    const uint row = pc.mode == 0u ? group_id.x / pc.num_chunks : group_id.x;
    const uint chunk_index = pc.mode == 0u ? group_id.x - row * pc.num_chunks : 0u;
    if (row >= pc.rows) return;
    const uint span = pc.mode == 0u ? pc.chunk : pc.length;
    const uint begin = pc.mode == 0u ? chunk_index * pc.chunk : 0u;
    const uint base = pc.mode == 0u ? row * pc.length + begin : row * pc.length;

    float original[4];
    float values[4];
    uint valid = 0u;
    float running = 0.0f;
    [unroll]
    for (uint item = 0u; item < 4u; ++item)
    {
        const uint local = lane * 4u + item;
        const uint column = begin + local;
        original[item] = 0.0f;
        values[item] = 0.0f;
        if (local < span && column < pc.length)
        {
            const float loaded = asfloat(src.Load(pc.src_offset + (base + local) * 4u));
            original[item] = loaded;
            running += loaded;
            values[item] = running;
            valid = item + 1u;
        }
    }
    scan_thread_totals(lane, running);
    const float prefix = lane == 0u ? 0.0f : totals[lane - 1u];
    if (valid != 0u)
    {
        [unroll]
        for (uint item = 0u; item < 4u; ++item)
        {
            if (item < valid)
            {
                const float inclusive = values[item] + prefix;
                const float stored = pc.mode == 1u ? inclusive - original[item] : inclusive;
                dst.Store(pc.dst_offset + (base + lane * 4u + item) * 4u, asuint(stored));
            }
        }
    }
    if (pc.mode == 0u && lane == 0u)
    {
        sums.Store(pc.sums_offset + (row * pc.num_chunks + chunk_index) * 4u, asuint(totals[TT_GROUP_SIZE - 1u]));
    }
}
