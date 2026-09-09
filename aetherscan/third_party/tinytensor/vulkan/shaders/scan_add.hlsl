#include "common.hlsli"

struct PushConstants
{
    uint count;
    uint is_float;
    uint data_offset;
    uint sums_offset;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer data;
[[vk::binding(1, 0)]] RWByteAddressBuffer block_sums;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint3 gid : SV_GroupID)
{
    const uint index = dtid.x;
    if (index >= pc.count || gid.x == 0)
    {
        return;
    }

    if (pc.is_float != 0)
    {
        const float prefix = asfloat(block_sums.Load(pc.sums_offset + gid.x * 4u));
        const uint addr = pc.data_offset + index * 4u;
        data.Store(addr, asuint(asfloat(data.Load(addr)) + prefix));
        return;
    }

    const uint prefix = block_sums.Load(pc.sums_offset + gid.x * 4u);
    const uint addr = pc.data_offset + index * 4u;
    data.Store(addr, data.Load(addr) + prefix);
}
