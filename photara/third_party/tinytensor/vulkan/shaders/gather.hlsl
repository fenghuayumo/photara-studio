#include "common.hlsli"

struct PushConstants
{
    uint rank;
    uint dim;
    uint count;
    uint mode;
    uint elem_size;
    uint src_offset;
    uint dst_offset;
    uint idx_offset;
    uint idx_is_i64;
    uint idx_rank;
    uint src_shape[8];
    uint idx_shape[8];
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer indices;
[[vk::binding(2, 0)]] RWByteAddressBuffer dst;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

int read_index(uint i)
{
    if (pc.idx_is_i64 != 0)
    {
        return i64_as_i32(load_u64(indices, pc.idx_offset + i * 8u));
    }
    return load_i32(indices, pc.idx_offset + i * 4u);
}

int map_index(int sel, int dim)
{
    if (pc.mode == 1)
    {
        return clamp(sel, 0, dim - 1);
    }
    if (pc.mode == 2)
    {
        int wrapped = sel % dim;
        if (wrapped < 0)
        {
            wrapped += dim;
        }
        return wrapped;
    }
    if (sel < 0)
    {
        sel += dim;
    }
    return sel;
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count || pc.rank == 0u || pc.rank > 8u || pc.dim >= pc.rank)
    {
        return;
    }

    uint coord[8];
    uint remaining = index;
    [unroll]
    for (int axis = 7; axis >= 0; --axis)
    {
        coord[axis] = 0u;
        if (uint(axis) < pc.idx_rank)
        {
            const uint dim = max(pc.idx_shape[axis], 1u);
            coord[axis] = remaining % dim;
            remaining /= dim;
        }
    }

    const int src_dim = int(pc.src_shape[pc.dim]);
    const int sel = map_index(read_index(index), src_dim);
    const uint dst_off = pc.dst_offset + index * pc.elem_size;
    if (sel < 0 || sel >= src_dim)
    {
        return;
    }
    coord[pc.dim] = uint(sel);

    uint src_index = 0u;
    uint stride = 1u;
    [unroll]
    for (int axis = 7; axis >= 0; --axis)
    {
        if (uint(axis) < pc.rank)
        {
            if (coord[axis] >= pc.src_shape[axis])
            {
                return;
            }
            src_index += coord[axis] * stride;
            stride *= pc.src_shape[axis];
        }
    }
    copy_elem(dst, dst_off, src, pc.src_offset + src_index * pc.elem_size, pc.elem_size);
}
