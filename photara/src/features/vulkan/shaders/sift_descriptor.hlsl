#include "sift_common.hlsli"

// ComputeDescriptor: 16 threads per keypoint, one 4x4 spatial block each.
// Gradient samples are trilinearly accumulated into 8 orientation bins with
// the Gaussian window of the CUDA kernel; normalization is a separate pass.
struct PushConstants
{
    uint count;
    uint width;
    uint height;
    float window_factor;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer list; // float4 keypoints
[[vk::binding(1, 0)]] RWByteAddressBuffer gradient; // float2 per pixel
[[vk::binding(2, 0)]] RWByteAddressBuffer descriptors; // 128 floats per keypoint
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const float kRadiansPerPiOver4 = 4.0f / 3.14159265358979323846f;

float2 sample_gradient(float x, float y)
{
    const float fx = x - 0.5f;
    const float fy = y - 0.5f;
    const int ix = int(floor(fx));
    const int iy = int(floor(fy));
    const float tx = fx - float(ix);
    const float ty = fy - float(iy);
    const int x0 = clampi(ix, 0, int(pc.width) - 1);
    const int x1 = clampi(ix + 1, 0, int(pc.width) - 1);
    const int y0 = clampi(iy, 0, int(pc.height) - 1);
    const int y1 = clampi(iy + 1, 0, int(pc.height) - 1);
    const float2 a = load_f2(gradient, uint(y0 * int(pc.width) + x0) * 8u);
    const float2 b = load_f2(gradient, uint(y0 * int(pc.width) + x1) * 8u);
    const float2 c = load_f2(gradient, uint(y1 * int(pc.width) + x0) * 8u);
    const float2 d = load_f2(gradient, uint(y1 * int(pc.width) + x1) * 8u);
    return lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
}

[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    const uint fidx = index >> 4u;
    if (fidx >= pc.count) return;

    const float4 key = load_f4(list, fidx * 16u);
    const uint bidx = index & 0xfu;
    const float ix = float(bidx & 0x3u);
    const float iy = float(bidx >> 2u);

    const float spt = abs(key.z * pc.window_factor);
    float s, c;
    sincos(key.w, s, c);
    const float anglef =
        key.w > 3.14159265358979323846f
            ? key.w - 6.28318530717958647692f
            : key.w;
    const float cspt = c * spt;
    const float sspt = s * spt;
    const float crspt = c / spt;
    const float srspt = s / spt;

    const float2 offsetpt = float2(ix - 1.5f, iy - 1.5f);
    const float2 pt = float2(cspt * offsetpt.x - sspt * offsetpt.y + key.x,
                             cspt * offsetpt.y + sspt * offsetpt.x + key.y);
    const float bsz = abs(cspt) + abs(sspt);
    const float xmin = max(1.5f, floor(pt.x - bsz) + 0.5f);
    const float ymin = max(1.5f, floor(pt.y - bsz) + 0.5f);
    const float xmax = min(float(pc.width) - 1.5f, floor(pt.x + bsz) + 0.5f);
    const float ymax = min(float(pc.height) - 1.5f, floor(pt.y + bsz) + 0.5f);

    float des[9];
    [unroll]
    for (uint i = 0u; i < 9u; ++i) des[i] = 0.0f;

    for (float y = ymin; y <= ymax; y += 1.0f)
    {
        for (float x = xmin; x <= xmax; x += 1.0f)
        {
            const float dx = x - pt.x;
            const float dy = y - pt.y;
            const float nx = crspt * dx + srspt * dy;
            const float ny = crspt * dy - srspt * dx;
            const float nxn = abs(nx);
            const float nyn = abs(ny);
            if (nxn < 1.0f && nyn < 1.0f)
            {
                const float2 cc = sample_gradient(x, y);
                const float dnx = nx + offsetpt.x;
                const float dny = ny + offsetpt.y;
                const float ww = exp(-0.125f * (dnx * dnx + dny * dny));
                const float wx = 1.0f - nxn;
                const float wy = 1.0f - nyn;
                const float weight = ww * wx * wy * cc.x;
                float theta = (anglef - cc.y) * kRadiansPerPiOver4;
                if (theta < 0.0f) theta += 8.0f;
                const float fo = floor(theta);
                const uint fbin = uint(fo);
                const float weight1 = fo + 1.0f - theta;
                const float weight2 = theta - fo;
                [unroll]
                for (uint k = 0u; k < 8u; ++k)
                {
                    if (k == fbin)
                    {
                        des[k] += weight1 * weight;
                        des[k + 1u] += weight2 * weight;
                    }
                }
            }
        }
    }
    des[0] += des[8];

    const uint out_base = index * 8u * 4u;
    store_f4(descriptors, out_base, float4(des[0], des[1], des[2], des[3]));
    store_f4(descriptors, out_base + 16u, float4(des[4], des[5], des[6], des[7]));
}
