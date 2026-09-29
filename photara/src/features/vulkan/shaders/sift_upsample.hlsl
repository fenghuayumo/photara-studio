#include "sift_common.hlsli"

// UpsampleKernel<1>: 2x bilinear upsampling with the reference kernel's raw
// linear addressing. Reads use the *source* row stride, so the last column of a
// row blends with the first pixel of the next row, and reads past the end of
// the image return zero (tex1Dfetch semantics). SiftGPU's own border reads are
// undefined, which is why its keypoints there are not run-to-run reproducible;
// zero is what its texture fetch returns and what its results match on
// reproducible inputs.
struct PushConstants
{
    uint src_width;
    uint src_height;
    uint dst_width;
    uint pad0;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer source;
[[vk::binding(1, 0)]] RWByteAddressBuffer destination;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(128, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint x = dtid.x;
    const uint y = dtid.y;
    if (x >= pc.src_width) return;

    const uint count = pc.src_width * pc.src_height;
    const uint row = y >> 1u;
    const uint index = row * pc.src_width + x;
    const uint dst_index = y * pc.dst_width + x * 2u;
    if ((y & 1u) != 0u)
    {
        const float v11 = fetch_f32(source, index, count);
        const float v12 = fetch_f32(source, index + 1u, count);
        const float v21 = fetch_f32(source, index + pc.src_width, count);
        const float v22 = fetch_f32(source, index + pc.src_width + 1u, count);
        // CUDA: v1 = v21 * w1 + w2 * v11, v2 = v22 * w1 + w2 * v12 (w1 = w2).
        const float first = v21 * 0.5f + 0.5f * v11;
        const float second = v22 * 0.5f + 0.5f * v12;
        store_f32(destination, dst_index * 4u, first);
        store_f32(destination, (dst_index + 1u) * 4u,
                  first * 0.5f + second * 0.5f);
    }
    else
    {
        const float v1 = fetch_f32(source, index, count);
        const float v2 = fetch_f32(source, index + 1u, count);
        store_f32(destination, dst_index * 4u, v1);
        store_f32(destination, (dst_index + 1u) * 4u, v1 * 0.5f + v2 * 0.5f);
    }
}
