#include "common.hlsli"

struct PushConstants
{
    uint count;
    uint rank;
    uint elem_size;
    uint src_offset_elems;
    uint dst_offset_elems;
    uint src_shape0;
    uint src_shape1;
    uint src_shape2;
    uint src_shape3;
    uint src_stride0;
    uint src_stride1;
    uint src_stride2;
    uint src_stride3;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer dst;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

uint shape_at(uint axis)
{
    if (axis == 0)
    {
        return pc.src_shape0;
    }
    if (axis == 1)
    {
        return pc.src_shape1;
    }
    if (axis == 2)
    {
        return pc.src_shape2;
    }
    return pc.src_shape3;
}

uint stride_at(uint axis)
{
    if (axis == 0)
    {
        return pc.src_stride0;
    }
    if (axis == 1)
    {
        return pc.src_stride1;
    }
    if (axis == 2)
    {
        return pc.src_stride2;
    }
    return pc.src_stride3;
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count)
    {
        return;
    }

    uint remaining = index;
    uint src_elem = pc.src_offset_elems;
    // Decode row-major linear index against the destination (contiguous) shape,
    // which matches the source logical shape.
    for (int axis = int(pc.rank) - 1; axis >= 0; --axis)
    {
        const uint dim = max(shape_at(uint(axis)), 1u);
        const uint coord = remaining % dim;
        remaining /= dim;
        src_elem += coord * stride_at(uint(axis));
    }

    const uint src_off = src_elem * pc.elem_size;
    const uint dst_off = (pc.dst_offset_elems + index) * pc.elem_size;
    copy_elem(dst, dst_off, src, src_off, pc.elem_size);
}
