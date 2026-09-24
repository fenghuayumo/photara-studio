#include "common.hlsli"

struct PushConstants
{
    uint count;
    uint elem_size;
    uint dtype;
    uint src_offset;
    uint flags_offset;
    uint scan_offset;
    uint dst_offset;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer flags;
[[vk::binding(2, 0)]] RWByteAddressBuffer scan;
[[vk::binding(3, 0)]] RWByteAddressBuffer dst;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count) return;
    if (flags.Load(pc.flags_offset + index * 4u) == 0u) return;
    const uint slot = scan.Load(pc.scan_offset + index * 4u);
    copy_elem(
        dst,
        pc.dst_offset + slot * pc.elem_size,
        src,
        pc.src_offset + index * pc.elem_size,
        pc.elem_size);
}
