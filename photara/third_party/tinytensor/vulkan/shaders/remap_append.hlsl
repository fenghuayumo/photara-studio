#include "common.hlsli"
struct PushConstants {
    uint count, row_width, retained, elem_size;
    uint src_offset, indices_offset, added_offset, dst_offset;
    uint has_indices, has_added;
};
[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer indices;
[[vk::binding(2, 0)]] RWByteAddressBuffer added;
[[vk::binding(3, 0)]] RWByteAddressBuffer dst;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    const uint index = dtid.x;
    if (index >= pc.count) return;
    const uint row = index / pc.row_width;
    const uint component = index % pc.row_width;
    const uint output_offset = pc.dst_offset + index * pc.elem_size;
    if (row < pc.retained) {
        const uint source_row = pc.has_indices != 0u ? indices.Load(pc.indices_offset + row * 4u) : row;
        copy_elem(dst, output_offset, src, pc.src_offset + (source_row * pc.row_width + component) * pc.elem_size, pc.elem_size);
    } else if (pc.has_added != 0u) {
        copy_elem(dst, output_offset, added, pc.added_offset + ((row - pc.retained) * pc.row_width + component) * pc.elem_size, pc.elem_size);
    } else {
        if (pc.elem_size == 1u) store_u8(dst, output_offset, 0u);
        else if (pc.elem_size == 2u) store_u16(dst, output_offset, 0u);
        else if (pc.elem_size == 8u) dst.Store2(output_offset, uint2(0u, 0u));
        else dst.Store(output_offset, 0u);
    }
}
