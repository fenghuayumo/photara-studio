#include "sift_common.hlsli"

// Translation of the CUDA `tiles` kernel: one workgroup per descriptor tile
// keeps the tile in groupshared memory, computes the integer dot-product
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

// Query rows per tile. Taller tiles amortize the per-tile work, but cost
// shared memory, so the host picks the tallest tile the device can host
// (see Context::descriptor_tile_plan). Measured on a 12000x12000 descriptor
// pair: 64 rows halve the 32-row cost (22.0 -> 11.3 ms) and 128 rows halve
// it again (-> 6.5 ms). The 160-row variant is the largest multiple of 32
// that fits a 48 KiB shared-memory limit and cuts another ~7% on the 72-image
// 1080x1920 capture benchmark.
#ifndef TILE_ROWS
#define TILE_ROWS 32
#endif

groupshared uint query_tile[TILE_ROWS][32]; // TILE_ROWS x 128 bytes
groupshared uint train_tile[32][32];
groupshared int dots[TILE_ROWS][32];

uint load_word(RWByteAddressBuffer descriptors, uint row, uint word, uint count)
{
    if (row >= count) return 0u;
    return descriptors.Load((row * 32u + word) * 4u);
}

uint byte_dot(uint a, uint b)
{
#if defined(PHOTARA_MATCH_DP4A)
    // One instruction per four bytes; the integer result is identical to the
    // scalar expansion below.
    return dot4add_u8packed(a, b, 0u);
#else
    return ((a & 0xffu) * (b & 0xffu)) +
           (((a >> 8) & 0xffu) * ((b >> 8) & 0xffu)) +
           (((a >> 16) & 0xffu) * ((b >> 16) & 0xffu)) +
           (((a >> 24) & 0xffu) * ((b >> 24) & 0xffu));
#endif
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
    const uint query_base = gid.y * TILE_ROWS;
    const uint train_base = gid.x * 32u;

    for (uint e = tid; e < TILE_ROWS * 32u; e += 256u)
    {
        const uint r = e >> 5u;
        const uint k = e & 31u;
        query_tile[r][k] = load_word(query, query_base + r, k, pc.query_count);
    }
    for (uint e = tid; e < 1024u; e += 256u)
    {
        const uint r = e >> 5u;
        const uint k = e & 31u;
        train_tile[r][k] = load_word(train, train_base + r, k, pc.train_count);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint e = tid; e < TILE_ROWS * 32u; e += 256u)
    {
        const uint r = e >> 5u;
        const uint c = e & 31u;
        uint dot = 0u;
#if defined(PHOTARA_MATCH_DP4A)
        [unroll]
        for (uint k = 0u; k < 32u; ++k)
        {
            dot = dot4add_u8packed(query_tile[r][k], train_tile[c][k], dot);
        }
#else
        [unroll]
        for (uint k = 0u; k < 32u; ++k)
        {
            dot += byte_dot(query_tile[r][k], train_tile[c][k]);
        }
#endif
        dots[r][c] = int(dot);
    }
    GroupMemoryBarrierWithGroupSync();

    // Row partials keep one entry per (32-column train block, query row): the
    // tile height only changes how many rows one workgroup covers.
    if (tid < TILE_ROWS && query_base + tid < pc.query_count)
    {
        int3 best = int3(0, -1, 0);
        for (uint c = 0u; c < 32u && train_base + c < pc.train_count; ++c)
        {
            insert(dots[tid][c], int(train_base + c), best);
        }
        store_partial(row_partials,
                      (train_base / 32u) * pc.query_count + query_base + tid, best);
    }

    // Column partials stay per 32 query rows, so each half of a tall tile
    // reduces its own rows.
    if (pc.mutual != 0u && tid < TILE_ROWS / 32u * 32u)
    {
        const uint half = tid >> 5u;
        const uint c = tid & 31u;
        const uint row0 = query_base + half * 32u;
        if (train_base + c < pc.train_count)
        {
            int3 best = int3(0, -1, 0);
            for (uint r = 0u; r < 32u && row0 + r < pc.query_count; ++r)
            {
                insert(dots[half * 32u + r][c], int(row0 + r), best);
            }
            store_partial(col_partials,
                          (row0 / 32u) * pc.train_count + train_base + c, best);
        }
    }
}
