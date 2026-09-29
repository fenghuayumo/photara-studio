#include "sift_common.hlsli"

// Translation of the CUDA `tiles` kernel: one workgroup per 32x32 descriptor
// tile keeps the tile in groupshared memory, computes the integer dot-product
// matrix, and reduces per-row and (optionally) per-column top-two matches.
// Partials use the CUDA layout rows[x_tile * nq + query] and
// cols[y_tile * nt + train], stored as three int32 words (dot, index, second).
struct PushConstants
{
    uint query_count;
    uint train_count;
    uint mutual;
    uint pad0;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer query;
[[vk::binding(1, 0)]] RWByteAddressBuffer train;
[[vk::binding(2, 0)]] RWByteAddressBuffer row_partials;
[[vk::binding(3, 0)]] RWByteAddressBuffer col_partials;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

groupshared uint query_tile[32][32]; // 32 rows x 128 bytes (32 uint)
groupshared uint train_tile[32][32];
groupshared int dots[32][32];

uint load_word(RWByteAddressBuffer descriptors, uint row, uint word, uint count)
{
    if (row >= count) return 0u;
    return descriptors.Load((row * 32u + word) * 4u);
}

uint byte_dot(uint a, uint b)
{
    return ((a & 0xffu) * (b & 0xffu)) +
           (((a >> 8) & 0xffu) * ((b >> 8) & 0xffu)) +
           (((a >> 16) & 0xffu) * ((b >> 16) & 0xffu)) +
           (((a >> 24) & 0xffu) * ((b >> 24) & 0xffu));
}

void insert(int value, int index, inout int3 best)
{
    if (value > best.x)
    {
        best.z = best.x;
        best.x = value;
        best.y = index;
    }
    else
    {
        best.z = max(best.z, value);
    }
}

void store_partial(RWByteAddressBuffer buffer, uint index, int3 value)
{
    buffer.Store3(index * 12u, uint3(asuint(value.x), asuint(value.y), asuint(value.z)));
}

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint query_base = gid.y * 32u;
    const uint train_base = gid.x * 32u;

    for (uint e = tid; e < 1024u; e += 256u)
    {
        const uint r = e >> 5u;
        const uint k = e & 31u;
        query_tile[r][k] = load_word(query, query_base + r, k, pc.query_count);
        train_tile[r][k] = load_word(train, train_base + r, k, pc.train_count);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint e = tid; e < 1024u; e += 256u)
    {
        const uint r = e >> 5u;
        const uint c = e & 31u;
        uint dot = 0u;
        [unroll]
        for (uint k = 0u; k < 32u; ++k)
        {
            dot += byte_dot(query_tile[r][k], train_tile[c][k]);
        }
        dots[r][c] = int(dot);
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid < 32u && query_base + tid < pc.query_count)
    {
        int3 best = int3(0, -1, 0);
        for (uint c = 0u; c < 32u && train_base + c < pc.train_count; ++c)
        {
            insert(dots[tid][c], int(train_base + c), best);
        }
        store_partial(row_partials,
                      (train_base / 32u) * pc.query_count + query_base + tid, best);
    }

    if (pc.mutual != 0u && tid >= 32u && tid < 64u)
    {
        const uint c = tid - 32u;
        if (train_base + c < pc.train_count)
        {
            int3 best = int3(0, -1, 0);
            for (uint r = 0u; r < 32u && query_base + r < pc.query_count; ++r)
            {
                insert(dots[r][c], int(query_base + r), best);
            }
            store_partial(col_partials,
                          (query_base / 32u) * pc.train_count + train_base + c, best);
        }
    }
}
