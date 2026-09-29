#include "sift_common.hlsli"

// ConvertByteToFloat: grayscale u8 -> float in [0, 1] with the same row
// compaction SiftGPU applies (the input width is truncated to a multiple of
// four, dropping the trailing columns). One thread converts four consecutive
// destination pixels of one row.
struct PushConstants
{
    uint pixels;
    uint source_stride;
    uint destination_stride;
    uint pad0;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer input_bytes;
[[vk::binding(1, 0)]] RWByteAddressBuffer output_floats;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(128, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint col = dtid.x * 4u;
    const uint row = dtid.y;
    if (col >= pc.destination_stride) return;
    [unroll]
    for (uint i = 0u; i < 4u; ++i)
    {
        const uint index = row * pc.destination_stride + col + i;
        if (index >= pc.pixels) break;
        const uint source = row * pc.source_stride + col + i;
        const uint word = input_bytes.Load(source & ~3u);
        const uint value = (word >> ((source & 3u) * 8u)) & 0xffu;
        store_f32(output_floats, index * 4u, float(value) * (1.0f / 255.0f));
    }
}
