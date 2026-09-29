#include "sift_common.hlsli"

// DownsampleKernel: point-sampled decimation by `scale` (a power of two);
// scale = 1 turns the pass into a padded row copy.
struct PushConstants
{
    uint src_width;
    uint dst_width;
    uint scale;
    uint pad0;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer source;
[[vk::binding(1, 0)]] RWByteAddressBuffer destination;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint col = dtid.x;
    if (col >= pc.dst_width) return;
    const uint src_col = min(col * pc.scale, pc.src_width - 1u);
    const uint src_row = dtid.y * pc.scale;
    store_f32(destination, (dtid.y * pc.dst_width + col) * 4u,
              load_f32(source, (src_row * pc.src_width + src_col) * 4u));
}
