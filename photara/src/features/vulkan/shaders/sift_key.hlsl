#include "sift_common.hlsli"

// ComputeKEY: DoG extrema detection with edge suppression and SiftGPU's
// partial-pivot sub-pixel/sub-scale solve. Output is one float4 per pixel:
// (sign 0/+1/-1, dx, dy, ds); rejected pixels store zeroes.
struct PushConstants
{
    uint width;
    uint height;
    float dog_threshold0;
    float dog_threshold;
    float edge_threshold;
    uint pad0;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer dog_previous;
[[vk::binding(1, 0)]] RWByteAddressBuffer dog_current;
[[vk::binding(2, 0)]] RWByteAddressBuffer dog_next;
[[vk::binding(3, 0)]] RWByteAddressBuffer key_output;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

float center(RWByteAddressBuffer buf, uint index)
{
    return load_f32(buf, index * 4u);
}

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint row = dtid.y;
    const uint col = dtid.x;
    if (row == 0u || col == 0u || row >= pc.height - 1u || col >= pc.width - 1u)
        return;

    const uint index = row * pc.width + col;
    const int3 rows = int3(int(index) - int(pc.width), int(index),
                           int(index) + int(pc.width));

    float data[3][3];
    float datap[3][3];
    float datan[3][3];
    float dx = 0.0f, dy = 0.0f, ds = 0.0f;
    bool offset_test_passed = true;
    float result = 0.0f;

    const float v = load_f32(dog_current, uint(rows.y) * 4u);
    if (abs(v) <= pc.dog_threshold0) return;

    data[1][1] = v;
    data[1][0] = load_f32(dog_current, uint(rows.y - 1) * 4u);
    data[1][2] = load_f32(dog_current, uint(rows.y + 1) * 4u);
    float nmax = max(data[1][0], data[1][2]);
    float nmin = min(data[1][0], data[1][2]);
    if (v <= nmax && v >= nmin) return;

    // Rows above and below are checked with the READ_CMP_DOG_DATA macro
    // semantics: every neighbor must be strictly smaller (maximum) or larger
    // (minimum) than the center. The center row was already compared above.
    [unroll]
    for (uint rr = 0u; rr < 2u; ++rr)
    {
        const uint r = rr == 0u ? 0u : 2u;
        const int base = r == 0u ? rows.x : rows.z;
        [unroll]
        for (uint c = 0; c < 3u; ++c)
        {
            data[r][c] = load_f32(dog_current, uint(base + int(c) - 1) * 4u);
        }
        if (v > nmax)
        {
            nmax = max(nmax, max(max(data[r][0], data[r][1]), data[r][2]));
            if (v < nmax) return;
        }
        else
        {
            nmin = min(nmin, min(min(data[r][0], data[r][1]), data[r][2]));
            if (v > nmin) return;
        }
    }

    // Edge suppression from the 2x2 Hessian of the center level.
    const float vx2 = v * 2.0f;
    const float fxx = data[1][0] + data[1][2] - vx2;
    const float fyy = data[0][1] + data[2][1] - vx2;
    const float fxy = 0.25f * (data[2][2] + data[0][0] - data[2][0] - data[0][2]);
    const float det = fxx * fyy - fxy * fxy;
    const float trace_sq = (fxx + fyy) * (fxx + fyy);
    if (det <= 0.0f || trace_sq > pc.edge_threshold * det) return;

    // The previous and next DoG levels take part in the extremum test exactly
    // like the CUDA READ_CMP_DOG_DATA macro: each three-value group extends the
    // running max/min and rejects as soon as the center is no longer the
    // strict extremum.
    [unroll]
    for (uint r = 0; r < 3u; ++r)
    {
        const int basep = r == 0u ? rows.x : (r == 1u ? rows.y : rows.z);
        [unroll]
        for (uint c = 0; c < 3u; ++c)
            datap[r][c] = load_f32(dog_previous, uint(basep + int(c) - 1) * 4u);
        if (v > nmax)
        {
            nmax = max(nmax, max(max(datap[r][0], datap[r][1]), datap[r][2]));
            if (v < nmax) return;
        }
        else
        {
            nmin = min(nmin, min(min(datap[r][0], datap[r][1]), datap[r][2]));
            if (v > nmin) return;
        }
    }
    [unroll]
    for (uint r = 0; r < 3u; ++r)
    {
        const int basen = r == 0u ? rows.x : (r == 1u ? rows.y : rows.z);
        [unroll]
        for (uint c = 0; c < 3u; ++c)
            datan[r][c] = load_f32(dog_next, uint(basen + int(c) - 1) * 4u);
        if (v > nmax)
        {
            nmax = max(nmax, max(max(datan[r][0], datan[r][1]), datan[r][2]));
            if (v < nmax) return;
        }
        else
        {
            nmin = min(nmin, min(min(datan[r][0], datan[r][1]), datan[r][2]));
            if (v > nmin) return;
        }
    }

    // Sub-pixel localization: solve H * (dx, dy, ds) = -grad by the same
    // sign-normalized Gaussian elimination the CUDA kernel performs.
    const float fx = 0.5f * (data[1][2] - data[1][0]);
    const float fy = 0.5f * (data[2][1] - data[0][1]);
    const float fs = 0.5f * (datan[1][1] - datap[1][1]);

    const float fss = datan[1][1] + datap[1][1] - vx2;
    const float fxs = 0.25f * (datan[1][2] + datap[1][0] - datan[1][0] - datap[1][2]);
    const float fys = 0.25f * (datan[2][1] + datap[0][1] - datan[0][1] - datap[2][1]);

    float4 A0 = fxx > 0.0f ? float4(fxx, fxy, fxs, -fx)
                           : float4(-fxx, -fxy, -fxs, fx);
    float4 A1 = fxy > 0.0f ? float4(fxy, fyy, fys, -fy)
                           : float4(-fxy, -fyy, -fys, fy);
    float4 A2 = fxs > 0.0f ? float4(fxs, fys, fss, -fs)
                           : float4(-fxs, -fys, -fss, fs);
    const float maxa = max(max(A0.x, A1.x), A2.x);
    if (maxa >= 1e-10f)
    {
        if (maxa == A1.x)
        {
            const float4 temp = A1;
            A1 = A0;
            A0 = temp;
        }
        else if (maxa == A2.x)
        {
            const float4 temp = A2;
            A2 = A0;
            A0 = temp;
        }
        A0.y /= A0.x;
        A0.z /= A0.x;
        A0.w /= A0.x;
        A1.y -= A1.x * A0.y;
        A1.z -= A1.x * A0.z;
        A1.w -= A1.x * A0.w;
        A2.y -= A2.x * A0.y;
        A2.z -= A2.x * A0.z;
        A2.w -= A2.x * A0.w;
        if (abs(A2.y) > abs(A1.y))
        {
            const float4 temp = A2;
            A2 = A1;
            A1 = temp;
        }
        if (abs(A1.y) >= 1e-10f)
        {
            A1.z /= A1.y;
            A1.w /= A1.y;
            A2.z -= A2.y * A1.z;
            A2.w -= A2.y * A1.w;
            if (abs(A2.z) >= 1e-10f)
            {
                ds = A2.w / A2.z;
                dy = A1.w - ds * A1.z;
                dx = A0.w - ds * A0.z - dy * A0.y;

                offset_test_passed =
                    abs(data[1][1] + 0.5f * (dx * fx + dy * fy + ds * fs)) >
                        pc.dog_threshold &&
                    abs(ds) < 1.0f && abs(dx) < 1.0f && abs(dy) < 1.0f;
            }
        }
    }

    if (offset_test_passed) result = v > nmax ? 1.0f : -1.0f;
    store_f4(key_output, index * 16u, float4(result, dx, dy, ds));
}
