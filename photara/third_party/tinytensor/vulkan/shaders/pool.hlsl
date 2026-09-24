#include "common.hlsli"

struct PushConstants
{
    uint op;
    uint n;
    uint c;
    uint h_in;
    uint w_in;
    uint h_out;
    uint w_out;
    uint kernel;
    uint stride;
    uint padding;
    uint src_offset;
    uint dst_offset;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer dst;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint kOpMaxPool = 0;
static const uint kOpAdaptiveAvg = 1;

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint total = pc.n * pc.c * pc.h_out * pc.w_out;
    const uint index = dtid.x;
    if (index >= total)
    {
        return;
    }
    const uint w_out = index % pc.w_out;
    uint tmp = index / pc.w_out;
    const uint h_out = tmp % pc.h_out;
    tmp /= pc.h_out;
    const uint c = tmp % pc.c;
    const uint n = tmp / pc.c;

    if (pc.op == kOpMaxPool)
    {
        const int h_start = int(h_out * pc.stride) - int(pc.padding);
        const int w_start = int(w_out * pc.stride) - int(pc.padding);
        float best = -3.4e38f;
        for (uint kh = 0u; kh < pc.kernel; ++kh)
        {
            const int h = h_start + int(kh);
            if (h < 0 || h >= int(pc.h_in)) continue;
            for (uint kw = 0u; kw < pc.kernel; ++kw)
            {
                const int w = w_start + int(kw);
                if (w < 0 || w >= int(pc.w_in)) continue;
                const uint elem = ((n * pc.c + c) * pc.h_in + uint(h)) * pc.w_in + uint(w);
                best = max(best, asfloat(src.Load(pc.src_offset + elem * 4u)));
            }
        }
        dst.Store(pc.dst_offset + index * 4u, asuint(best));
        return;
    }

    const int h0 = int((h_out * pc.h_in) / pc.h_out);
    const int h1 = int(((h_out + 1u) * pc.h_in + pc.h_out - 1u) / pc.h_out);
    const int w0 = int((w_out * pc.w_in) / pc.w_out);
    const int w1 = int(((w_out + 1u) * pc.w_in + pc.w_out - 1u) / pc.w_out);
    float sum = 0.0f;
    int count = 0;
    for (int h = h0; h < h1; ++h)
    {
        for (int w = w0; w < w1; ++w)
        {
            const uint elem = ((n * pc.c + c) * pc.h_in + uint(h)) * pc.w_in + uint(w);
            sum += asfloat(src.Load(pc.src_offset + elem * 4u));
            count += 1;
        }
    }
    const float mean = count > 0 ? sum / float(count) : 0.0f;
    dst.Store(pc.dst_offset + index * 4u, asuint(mean));
}
