#include "common.hlsli"

struct PushConstants
{
    uint op;
    uint outer;
    uint dim_size;
    uint inner;
    uint n_indices;
    uint elem_size;
    uint src_offset;
    uint dst_offset;
    uint idx_offset;
    uint idx_is_i64;
    uint dtype;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer indices;
[[vk::binding(2, 0)]] RWByteAddressBuffer dst;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint kOpReplace = 0;
static const uint kOpAdd = 1;

int read_index(uint i)
{
    if (pc.idx_is_i64 != 0)
    {
        return i64_as_i32(load_u64(indices, pc.idx_offset + i * 8u));
    }
    return load_i32(indices, pc.idx_offset + i * 4u);
}

void atomic_add_f32(uint addr, float delta)
{
    uint old = dst.Load(addr);
    [loop]
    while (true)
    {
        const uint assumed = old;
        uint previous = 0u;
        dst.InterlockedCompareExchange(addr, assumed, asuint(asfloat(assumed) + delta), previous);
        if (previous == assumed)
        {
            return;
        }
        old = previous;
    }
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
    if (sel < 0) sel += int(pc.dim_size);
    if (sel < 0 || sel >= int(pc.dim_size)) return;

    const uint src_elem = (o * n_indices + i) * inner + j;
    const uint dst_elem = (o * pc.dim_size + uint(sel)) * inner + j;
    const uint src_addr = pc.src_offset + src_elem * pc.elem_size;
    const uint dst_addr = pc.dst_offset + dst_elem * pc.elem_size;
    if (pc.op == kOpAdd && pc.dtype == kDtypeFloat32)
    {
        atomic_add_f32(dst_addr, load_f32(src, src_addr));
        return;
    }
    if (pc.op == kOpAdd && pc.dtype == kDtypeInt32)
    {
        int old = asint(dst.Load(dst_addr));
        [loop]
        while (true)
        {
            const int assumed = old;
            uint previous = 0u;
            dst.InterlockedCompareExchange(dst_addr, asuint(assumed), asuint(assumed + load_i32(src, src_addr)), previous);
            if (asint(previous) == assumed) return;
            old = asint(previous);
        }
    }
    copy_elem(dst, dst_addr, src, src_addr, pc.elem_size);
}
