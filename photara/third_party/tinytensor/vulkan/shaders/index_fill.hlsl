#include "common.hlsli"

struct PushConstants
{
    uint n_indices;
    uint outer;
    uint dim_size;
    uint inner;
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
    const uint inner = max(pc.inner, 1u);
    const uint n_indices = max(pc.n_indices, 1u);
    const uint total = max(pc.outer, 1u) * n_indices * inner;
    const uint index = dtid.x;
    if (index >= total)
    {
        return;
    }

    const uint j = index % inner;
    const uint tmp = index / inner;
    const uint i = tmp % n_indices;
    const uint o = tmp / n_indices;

    int sel = read_index(i);
    if (sel < 0)
    {
        sel += int(pc.dim_size);
    }
    if (sel < 0 || sel >= int(pc.dim_size))
    {
        return;
    }

    const uint dst_elem = (o * pc.dim_size + uint(sel)) * inner + j;
    store_from_float(
        dst,
        pc.dst_offset + dst_elem * pc.elem_size,
        pc.dtype,
        asfloat(pc.value_bits));
}
