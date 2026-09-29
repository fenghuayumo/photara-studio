#include "sift_common.hlsli"

// InitHist: pack per-pixel keypoint flags into int4 (four consecutive key
// columns per histogram element). Only interior pixels can hold keypoints.
struct PushConstants
{
    uint key_width;
    uint key_height;
    uint hist_width;
    uint pad0;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer keys;
[[vk::binding(1, 0)]] RWByteAddressBuffer hist;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(128, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint row = dtid.y;
    const uint col = dtid.x;
    if (row >= pc.key_height || col >= pc.hist_width) return;

    int4 value = int4(0, 0, 0, 0);
    if (row > 0u && row < pc.key_height - 1u)
    {
        [unroll]
        for (uint i = 0u; i < 4u; ++i)
        {
            const uint scol = col * 4u + i;
            const float4 temp = load_f4(keys, (row * pc.key_width + scol) * 16u);
            value[int(i)] = (scol < pc.key_width - 1u && scol > 0u && temp.x != 0.0f)
                                ? 1
                                : 0;
        }
    }
    store_i4(hist, (row * pc.hist_width + col) * 16u, value);
}
