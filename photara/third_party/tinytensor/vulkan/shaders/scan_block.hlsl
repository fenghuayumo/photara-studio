#include "common.hlsli"

struct PushConstants
{
    uint count;
    uint is_float;
    uint src_offset;
    uint dst_offset;
    uint sums_offset;
    uint exclusive;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer input_buf;
[[vk::binding(1, 0)]] RWByteAddressBuffer output_buf;
[[vk::binding(2, 0)]] RWByteAddressBuffer block_sums;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

groupshared uint shared_u[TT_GROUP_SIZE];
groupshared float shared_f[TT_GROUP_SIZE];

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint gidx : SV_GroupIndex, uint3 gid : SV_GroupID)
{
    const uint index = dtid.x;
    const bool in_range = index < pc.count;

    if (pc.is_float != 0)
    {
        const float value = in_range
            ? asfloat(input_buf.Load(pc.src_offset + index * 4u))
            : 0.0f;
        shared_f[gidx] = value;
        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint stride = 1; stride < TT_GROUP_SIZE; stride <<= 1)
        {
            float addend = 0.0f;
            if (gidx >= stride)
            {
                addend = shared_f[gidx - stride];
            }
            GroupMemoryBarrierWithGroupSync();
            shared_f[gidx] += addend;
            GroupMemoryBarrierWithGroupSync();
        }

        const float inclusive = shared_f[gidx];
        const float written = pc.exclusive != 0 ? inclusive - value : inclusive;
        if (in_range)
        {
            output_buf.Store(pc.dst_offset + index * 4u, asuint(written));
        }
        if (gidx == TT_GROUP_SIZE - 1)
        {
            block_sums.Store(pc.sums_offset + gid.x * 4u, asuint(inclusive));
        }
        return;
    }

    const uint value = in_range ? input_buf.Load(pc.src_offset + index * 4u) : 0u;
    shared_u[gidx] = value;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 1; stride < TT_GROUP_SIZE; stride <<= 1)
    {
        uint addend = 0u;
        if (gidx >= stride)
        {
            addend = shared_u[gidx - stride];
        }
        GroupMemoryBarrierWithGroupSync();
        shared_u[gidx] += addend;
        GroupMemoryBarrierWithGroupSync();
    }

    const uint inclusive = shared_u[gidx];
    const uint written = pc.exclusive != 0 ? inclusive - value : inclusive;
    if (in_range)
    {
        output_buf.Store(pc.dst_offset + index * 4u, written);
    }
    if (gidx == TT_GROUP_SIZE - 1)
    {
        block_sums.Store(pc.sums_offset + gid.x * 4u, inclusive);
    }
}
