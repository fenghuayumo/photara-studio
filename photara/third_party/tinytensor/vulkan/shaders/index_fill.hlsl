#include "common.hlsli"

struct PushConstants
{
    uint n_indices;
    uint dim_size;
    uint elem_size;
    uint dst_offset;
    uint idx_offset;
    uint idx_is_i64;
    uint dtype;
    uint value_bits;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer dst;
[[vk::binding(1, 0)]] RWByteAddressBuffer indices;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

int read_index(uint i)
{
    if (pc.idx_is_i64 != 0)
    {
        return i64_as_i32(load_u64(indices, pc.idx_offset + i * 8u));
    }
    return load_i32(indices, pc.idx_offset + i * 4u);
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.n_indices)
    {
        return;
    }

    int sel = read_index(index);
    if (sel < 0)
    {
        sel += int(pc.dim_size);
    }
    if (sel < 0 || sel >= int(pc.dim_size))
    {
        return;
    }

    store_from_float(
        dst,
        pc.dst_offset + uint(sel) * pc.elem_size,
        pc.dtype,
        asfloat(pc.value_bits));
}
