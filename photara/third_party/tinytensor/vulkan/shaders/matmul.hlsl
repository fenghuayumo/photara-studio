#include "common.hlsli"

struct PushConstants
{
    uint m;
    uint n;
    uint k;
    uint batch;
    uint a_offset;
    uint b_offset;
    uint c_offset;
    uint a_row_stride;
    uint a_col_stride;
    uint b_row_stride;
    uint b_col_stride;
    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_c;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer a_buf;
[[vk::binding(1, 0)]] RWByteAddressBuffer b_buf;
[[vk::binding(2, 0)]] RWByteAddressBuffer c_buf;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint kTile = 16;

groupshared float tile_a[kTile][kTile];
groupshared float tile_b[kTile][kTile];

[numthreads(kTile, kTile, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    const uint batch = gid.z;
    if (batch >= pc.batch)
    {
        return;
    }
    const uint row = gid.y * kTile + gtid.y;
    const uint col = gid.x * kTile + gtid.x;
    float acc = 0.0f;
    const uint tiles = (pc.k + kTile - 1u) / kTile;
    for (uint tile = 0u; tile < tiles; ++tile)
    {
        const uint a_col = tile * kTile + gtid.x;
        const uint b_row = tile * kTile + gtid.y;
        const uint a_row = gid.y * kTile + gtid.y;
        const uint b_col = gid.x * kTile + gtid.x;
        float a_val = 0.0f;
        float b_val = 0.0f;
        if (a_row < pc.m && a_col < pc.k)
        {
            const uint elem = batch * pc.batch_stride_a + a_row * pc.a_row_stride + a_col * pc.a_col_stride;
            a_val = asfloat(a_buf.Load(pc.a_offset + elem * 4u));
        }
        if (b_row < pc.k && b_col < pc.n)
        {
            const uint elem = batch * pc.batch_stride_b + b_row * pc.b_row_stride + b_col * pc.b_col_stride;
            b_val = asfloat(b_buf.Load(pc.b_offset + elem * 4u));
        }
        tile_a[gtid.y][gtid.x] = a_val;
        tile_b[gtid.y][gtid.x] = b_val;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint i = 0u; i < kTile; ++i)
        {
            acc += tile_a[gtid.y][i] * tile_b[i][gtid.x];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < pc.m && col < pc.n)
    {
        const uint elem = batch * pc.batch_stride_c + row * pc.n + col;
        c_buf.Store(pc.c_offset + elem * 4u, asuint(acc));
    }
}
