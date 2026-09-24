#include "common.hlsli"

struct PushConstants
{
    uint count;
    uint rank;
    uint elem_size;
    uint src_offset_elems;
    uint dst_offset_elems;
    uint mode;
    uint fill_bits;
    uint dtype;
    uint src_shape0;
    uint src_shape1;
    uint src_shape2;
    uint src_shape3;
    uint src_shape4;
    uint src_shape5;
    uint src_shape6;
    uint src_shape7;
    uint src_stride0;
    uint src_stride1;
    uint src_stride2;
    uint src_stride3;
    uint src_stride4;
    uint src_stride5;
    uint src_stride6;
    uint src_stride7;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer dst;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint kModeContiguous = 0;
static const uint kModeScatter = 1;
static const uint kModeFill = 2;

uint shape_at(uint axis)
{
    if (axis == 0) return pc.src_shape0;
    if (axis == 1) return pc.src_shape1;
    if (axis == 2) return pc.src_shape2;
    if (axis == 3) return pc.src_shape3;
    if (axis == 4) return pc.src_shape4;
    if (axis == 5) return pc.src_shape5;
    if (axis == 6) return pc.src_shape6;
    return pc.src_shape7;
}

uint stride_at(uint axis)
{
    if (axis == 0) return pc.src_stride0;
    if (axis == 1) return pc.src_stride1;
    if (axis == 2) return pc.src_stride2;
    if (axis == 3) return pc.src_stride3;
    if (axis == 4) return pc.src_stride4;
    if (axis == 5) return pc.src_stride5;
    if (axis == 6) return pc.src_stride6;
    return pc.src_stride7;
}

uint strided_elem(uint index)
{
    uint remaining = index;
    uint elem = pc.src_offset_elems;
    for (int axis = int(pc.rank) - 1; axis >= 0; --axis)
    {
        const uint dim = max(shape_at(uint(axis)), 1u);
        const uint coord = remaining % dim;
        remaining /= dim;
        elem += coord * stride_at(uint(axis));
    }
    return elem;
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count)
    {
        return;
    }

    if (pc.mode == kModeFill)
    {
        const uint dst_elem = strided_elem(index);
        store_from_float(dst, dst_elem * pc.elem_size, pc.dtype, asfloat(pc.fill_bits));
        return;
    }

    if (pc.mode == kModeScatter)
    {
        const uint dst_elem = strided_elem(index);
        copy_elem(
            dst,
            dst_elem * pc.elem_size,
            src,
            (pc.dst_offset_elems + index) * pc.elem_size,
            pc.elem_size);
        return;
    }

    const uint src_elem = strided_elem(index);
    copy_elem(
        dst,
        (pc.dst_offset_elems + index) * pc.elem_size,
        src,
        src_elem * pc.elem_size,
        pc.elem_size);
}
