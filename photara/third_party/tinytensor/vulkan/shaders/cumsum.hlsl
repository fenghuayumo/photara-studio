#include "common.hlsli"

struct PushConstants
{
    uint outer;
    uint dim_size;
    uint inner;
    uint src_offset;
    uint dst_offset;
    uint dtype;
    uint elem_size;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer dst;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint inner = max(pc.inner, 1u);
    const uint lines = pc.outer * inner;
    const uint line_index = dtid.x;
    if (line_index >= lines)
    {
        return;
    }
    const uint o = line_index / inner;
    const uint j = line_index % inner;
    float acc = 0.0f;
    for (uint i = 0u; i < pc.dim_size; ++i)
    {
        const uint elem = (o * pc.dim_size + i) * inner + j;
        acc += load_as_float(src, pc.src_offset + elem * pc.elem_size, pc.dtype);
        store_from_float(dst, pc.dst_offset + elem * pc.elem_size, pc.dtype, acc);
    }
}
