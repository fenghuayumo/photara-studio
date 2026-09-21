#include "common.hlsli"

struct PushConstants
{
    uint outer;
    uint dim_size;
    uint inner;
    uint n_indices;
    uint elem_size;
    uint src_offset;
    uint dst_offset;
    uint idx_offset;
    uint idx_is_i64;
    uint mode; // 0 assert/skip, 1 clamp, 2 wrap
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

int map_index(int sel)
{
    const int dim = int(pc.dim_size);
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
    const uint total = pc.outer * pc.n_indices * pc.inner;
    const uint index = dtid.x;
    if (index >= total)
    {
        return;
    }

    const uint inner = max(pc.inner, 1u);
    const uint n_indices = max(pc.n_indices, 1u);
    const uint j = index % inner;
    const uint tmp = index / inner;
    const uint i = tmp % n_indices;
    const uint o = tmp / n_indices;

    const int sel = map_index(read_index(i));
    if (sel < 0 || sel >= int(pc.dim_size))
    {
        return;
    }

    const uint src_elem = (o * pc.dim_size + uint(sel)) * inner + j;
    const uint dst_elem = (o * n_indices + i) * inner + j;
    copy_elem(
        dst,
        pc.dst_offset + dst_elem * pc.elem_size,
        src,
        pc.src_offset + src_elem * pc.elem_size,
        pc.elem_size);
}
