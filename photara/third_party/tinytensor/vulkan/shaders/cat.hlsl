#include "common.hlsli"

struct PushConstants
{
    uint count;
    uint outer;
    uint inner;
    uint n_tensors;
    uint elem_size;
    uint dst_offset;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer packed;
[[vk::binding(1, 0)]] RWByteAddressBuffer dst;
[[vk::binding(2, 0)]] RWByteAddressBuffer meta;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count)
    {
        return;
    }

    const uint inner = max(pc.inner, 1u);
    uint axis_total = 0u;
    for (uint t = 0u; t < pc.n_tensors; ++t)
    {
        axis_total += meta.Load(t * 8u + 4u);
    }
    axis_total = max(axis_total, 1u);
    const uint rem = index / inner;
    const uint local_inner = index % inner;
    const uint axis_coord = rem % axis_total;
    const uint outer_index = rem / axis_total;

    uint base = 0u;
    for (uint source = 0u; source < pc.n_tensors; ++source)
    {
        const uint axis_size = meta.Load(source * 8u + 4u);
        if (axis_coord < base + axis_size)
        {
            const uint src_off = meta.Load(source * 8u);
            const uint local_axis = axis_coord - base;
            const uint src_elem = (outer_index * max(axis_size, 1u) + local_axis) * inner + local_inner;
            copy_elem(
                dst,
                pc.dst_offset + index * pc.elem_size,
                packed,
                src_off + src_elem * pc.elem_size,
                pc.elem_size);
            return;
        }
        base += axis_size;
    }
}
