#include "sift_common.hlsli"

// ReduceHist: 4:1 reduction of the packed int4 histogram pyramid.
struct PushConstants
{
    uint in_width;
    uint out_width;
    uint out_height;
    uint pad0;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer input_hist;
[[vk::binding(1, 0)]] RWByteAddressBuffer output_hist;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(128, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint row = dtid.y;
    const uint col = dtid.x;
    if (row >= pc.out_height || col >= pc.out_width) return;

    int4 value = int4(0, 0, 0, 0);
    [unroll]
    for (uint i = 0u; i < 4u; ++i)
    {
        const uint scol = col * 4u + i;
        if (scol < pc.in_width)
        {
            const int4 temp = load_i4(input_hist, (row * pc.in_width + scol) * 16u);
            value[int(i)] += temp.x + temp.y + temp.z + temp.w;
        }
    }
    store_i4(output_hist, (row * pc.out_width + col) * 16u, value);
}
