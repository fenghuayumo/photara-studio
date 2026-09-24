#include "common.hlsli"

struct PushConstants
{
    uint count;
    uint ndim;
    uint flags_offset;
    uint scan_offset;
    uint out_offset;
    uint shape0;
    uint shape1;
    uint shape2;
    uint shape3;
    uint shape4;
    uint shape5;
    uint shape6;
    uint shape7;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer flags;
[[vk::binding(1, 0)]] RWByteAddressBuffer scan;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_indices;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

uint shape_at(uint axis)
{
    if (axis == 0) return max(pc.shape0, 1u);
    if (axis == 1) return max(pc.shape1, 1u);
    if (axis == 2) return max(pc.shape2, 1u);
    if (axis == 3) return max(pc.shape3, 1u);
    if (axis == 4) return max(pc.shape4, 1u);
    if (axis == 5) return max(pc.shape5, 1u);
    if (axis == 6) return max(pc.shape6, 1u);
    return max(pc.shape7, 1u);
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count)
    {
        return;
    }
    if (flags.Load(pc.flags_offset + index * 4u) == 0u)
    {
        return;
    }

    const uint slot = scan.Load(pc.scan_offset + index * 4u);
    const uint ndim = min(max(pc.ndim, 1u), 8u);
    if (ndim == 1)
    {
        store_i64_from_i32(out_indices, pc.out_offset + slot * 8u, int(index));
        return;
    }

    uint remaining = index;
    uint coords[8];
    for (int axis = int(ndim) - 1; axis >= 0; --axis)
    {
        const uint dim = shape_at(uint(axis));
        coords[axis] = remaining % dim;
        remaining /= dim;
    }
    for (uint axis = 0; axis < ndim; ++axis)
    {
        store_i64_from_i32(
            out_indices,
            pc.out_offset + (slot * ndim + axis) * 8u,
            int(coords[axis]));
    }
}
