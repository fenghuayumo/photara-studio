#include "common.hlsli"

struct PushConstants
{
    uint count;
    uint dtype;
    uint src_offset;
    uint dst_offset;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer flags;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count)
    {
        return;
    }

    const float value = load_as_float(src, pc.src_offset + index * dtype_size(pc.dtype), pc.dtype);
    flags.Store(pc.dst_offset + index * 4u, value != 0.0f ? 1u : 0u);
}
