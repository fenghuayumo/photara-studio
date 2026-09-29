#include "sift_common.hlsli"

// ComputeOrientation: build the 36-bin Gaussian-weighted orientation
// histogram, smooth it, and pack up to two dominant orientations into the
// keypoint's fourth float (two uint16 values), exactly like the CUDA kernel
// with existing_keypoint=0 / subpixel=1 / keepsign=0.
struct PushConstants
{
    uint list_length;
    uint width;
    uint height;
    uint num_orientation;
    float sigma;
    float sigma_step;
    float gaussian_factor;
    float sample_factor;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer list; // int4 in, float4 out (in place)
[[vk::binding(1, 0)]] RWByteAddressBuffer keys; // float4 per pixel: (sign, dx, dy, ds)
[[vk::binding(2, 0)]] RWByteAddressBuffer gradient; // float2 per pixel
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const float kTenDegreesPerRadius = 5.7295779513082320876798154814105;
static const float kRadiusPerTenDegrees = 1.0f / 5.7295779513082320876798154814105;

float2 sample_gradient(float x, float y)
{
    // tex2D with clamp addressing and unnormalized coordinates: texel centers
    // sit at integer + 0.5, so the fetch position is shifted by half a texel.
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
    if (index >= pc.list_length) return;

    const int4 ikey = load_i4(list, index * 16u);
    float4 key = float4(float(ikey.x) + 0.5f, float(ikey.y) + 0.5f, pc.sigma, 0.0f);
    const float4 offset = load_f4(keys, uint(ikey.y * int(pc.width) + ikey.x) * 16u);
    key.x += offset.y;
    key.y += offset.z;
    key.z *= pow(pc.sigma_step, offset.w);

    if (pc.num_orientation == 0u)
    {
        store_f4(list, index * 16u, float4(key.xyz, 0.0f));
        return;
    }

    float vote[37];
    const float gsigma = key.z * pc.gaussian_factor;
    const float win = abs(key.z) * pc.sample_factor;
    const float dist_threshold = win * win + 0.5f;
    const float factor = -0.5f / (gsigma * gsigma);
    const float xmin = max(1.5f, floor(key.x - win) + 0.5f);
    const float ymin = max(1.5f, floor(key.y - win) + 0.5f);
    const float xmax = min(float(pc.width) - 1.5f, floor(key.x + win) + 0.5f);
    const float ymax = min(float(pc.height) - 1.5f, floor(key.y + win) + 0.5f);

    [unroll]
    for (uint i = 0u; i < 36u; ++i) vote[i] = 0.0f;

    for (float y = ymin; y <= ymax; y += 1.0f)
    {
        for (float x = xmin; x <= xmax; x += 1.0f)
        {
            const float dx = x - key.x;
            const float dy = y - key.y;
            const float sq_dist = dx * dx + dy * dy;
            if (sq_dist >= dist_threshold) continue;
            const float2 got = sample_gradient(x, y);
            const float weight = got.x * exp(sq_dist * factor);
            int oidx = int(floor(got.y * kTenDegreesPerRadius));
            if (oidx < 0) oidx += 36;
            vote[oidx] += weight;
        }
    }

    // Six box-smoothing passes over the circular histogram.
    const float one_third = 1.0f / 3.0f;
    [unroll]
    for (uint pass = 0u; pass < 6u; ++pass)
    {
        vote[36] = vote[0];
        float previous = vote[35];
        [unroll]
        for (uint j = 0u; j < 36u; ++j)
        {
            const float temp = one_third * (previous + vote[j] + vote[j + 1u]);
            previous = vote[j];
            vote[j] = temp;
        }
    }

    vote[36] = vote[0];

    if (pc.num_orientation == 1u)
    {
        // Single dominant orientation: CUDA takes the global vote maximum in
        // this branch. The orientation is still stored in the packed two-slot
        // layout so the host expansion stays identical (the second slot is the
        // 65535 sentinel).
        int index_max = 0;
        float max_vote = vote[0];
        [unroll]
        for (uint i = 1u; i < 36u; ++i)
        {
            index_max = vote[i] > max_vote ? int(i) : index_max;
            max_vote = max(max_vote, vote[i]);
        }
        const float pre = vote[index_max == 0 ? 35 : index_max - 1];
        const float next = vote[index_max + 1];
        const float off = 0.5f * ((next - pre) / (max_vote + max_vote - next - pre));
        float fraction = (float(index_max) + 0.5f + off) / 36.0f;
        if (fraction < 0.0f) fraction += 1.0f;
        const uint packed = (65535u << 16) | (uint(floor(fraction * 65535.0f)) & 0xffffu);
        store_f4(list, index * 16u, float4(key.xyz, asfloat(packed)));
        return;
    }

    float max_vote = vote[0];
    [unroll]
    for (uint i = 1u; i < 36u; ++i) max_vote = max(max_vote, vote[i]);

    const float vote_threshold = max_vote * 0.8f;
    float previous = vote[35];
    float max_rot0 = 0.0f, max_rot1 = 0.0f;
    float max_vot0 = 0.0f, max_vot1 = 0.0f;
    int ocount = 0;
    [unroll]
    for (uint i = 0u; i < 36u; ++i)
    {
        const float next = vote[i + 1u];
        if (vote[i] > vote_threshold && vote[i] > previous && vote[i] > next)
        {
            const float di = 0.5f * ((next - previous) /
                                     (vote[i] + vote[i] - next - previous));
            const float rot = float(i) + di + 0.5f;
            const float weight = vote[i];
            if (weight > max_vot1)
            {
                if (weight > max_vot0)
                {
                    max_vot1 = max_vot0;
                    max_rot1 = max_rot0;
                    max_vot0 = weight;
                    max_rot0 = rot;
                }
                else
                {
                    max_vot1 = weight;
                    max_rot1 = rot;
                }
                ocount++;
            }
        }
        previous = vote[i];
    }

    float fr1 = max_rot0 / 36.0f;
    if (fr1 < 0.0f) fr1 += 1.0f;
    uint us1 = ocount == 0 ? 65535u : uint(floor(fr1 * 65535.0f));
    uint us2 = 65535u;
    if (ocount > 1)
    {
        float fr2 = max_rot1 / 36.0f;
        if (fr2 < 0.0f) fr2 += 1.0f;
        us2 = uint(floor(fr2 * 65535.0f));
    }
    store_f4(list, index * 16u, float4(key.xyz, asfloat((us2 << 16) | us1)));
}
