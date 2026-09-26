#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] StructuredBuffer<float> camera;
[[vk::binding(4, 0)]] StructuredBuffer<float> points;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> sample_f;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> sample_u;

[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint point_count = pc.u0;
    uint i = dispatch_id.x;
    if (i >= point_count) return;
    uint width = pc.u1;
    uint height = pc.u2;
    uint tiles_x = pc.u3;
    uint mode = pc.u4;
    float fx = asfloat(pc.u6);
    float fy = asfloat(pc.u7);
    float cx = asfloat(pc.u8);
    float cy = asfloat(pc.u9);
    float k1 = asfloat(pc.u10);
    float k2 = asfloat(pc.u11);
    float k3 = asfloat(pc.u12);
    float k4 = asfloat(pc.u13);
    float bracket = asfloat(pc.u14);
    float tolerance = asfloat(pc.u15);
    uint fbase = 3u * point_count;
    sample_f[3u * i] = 0.0f;
    sample_f[3u * i + 1u] = 0.0f;
    sample_f[3u * i + 2u] = 0.0f;
    sample_f[fbase + i] = 0.0f;
    sample_u[i] = 0u;
    sample_u[point_count + i] = 0u;
    float3 world = float3(points[3u * i], points[3u * i + 1u], points[3u * i + 2u]);
    float3 t = xform_point(
        world,
        camera[0], camera[1], camera[2],
        camera[4], camera[5], camera[6],
        camera[8], camera[9], camera[10],
        camera[12], camera[13], camera[14]);
    if (mode == kModeEquirect) {
        if (!(dot(t, t) > 1.0e-12f)) return;
    } else if (!(t.z > (mode == kModeFisheye ? 1.0e-6f : 0.2f))) {
        return;
    }
    Projection pr = project_camera(
        t, mode, int(width), int(height), fx, fy, cx, cy, k1, k2, k3, k4);
    if (!pr.valid || pr.pixel_x < 0.0f || pr.pixel_x > float(width - 1u) ||
        pr.pixel_y < 0.0f || pr.pixel_y > float(height - 1u)) {
        return;
    }
    uint tx = min(tiles_x - 1u, uint(max(0.0f, floor((pr.pixel_x + 0.5f) / 16.0f))));
    uint tiles_y = (height + 15u) / 16u;
    uint ty = min(tiles_y - 1u, uint(max(0.0f, floor((pr.pixel_y + 0.5f) / 16.0f))));
    uint begin = ranges[2u * (ty * tiles_x + tx)];
    uint end = ranges[2u * (ty * tiles_x + tx) + 1u];
    float T = 1.0f;
    float seed = 0.0f;
    uint last = 0u;
    uint contributor = 0u;
    [loop] for (uint at = begin; at < end; ++at) {
        ++contributor;
        uint g = instances[at];
        float2 d = gauss_f[8u * g].xy - float2(pr.pixel_x, pr.pixel_y);
        float4 co = gauss_f[8u * g + 1u];
        float power = gaussian_power(co, d.x, d.y);
        if (power > 0.0f) continue;
        float alpha = min(kAlphaClip, co.w * exp(power));
        if (alpha < kAlphaFloor) continue;
        float test_t = T * (1.0f - alpha);
        if (test_t < kTransmittanceFloor) break;
        float4 rp = gauss_f[8u * g + 3u];
        float peak = rp.x * d.x + rp.y * d.y + rp.z;
        if (T > 0.5f) seed = peak;
        T = test_t;
        last = contributor;
    }
    bool in_range = T <= kDepthMinTransmittance;
    float window = bracket > 0.0f ? bracket : 200.0f;
    float lo = max(seed - window, 0.0f);
    float hi = max(seed + window, 0.0f);
    float probes[9];
    [unroll] for (uint s = 0u; s < 9u; ++s) probes[s] = 1.0f;
    [loop] for (uint refine = 0u; refine < 8u; ++refine) {
        bool first = refine == 0u;
        uint sb = first ? 0u : 1u;
        uint se = first ? 9u : 8u;
        [unroll] for (uint sr = sb; sr < se; ++sr) probes[sr] = 1.0f;
        float interval = (hi - lo) / 8.0f;
        uint used = 0u;
        [loop] for (uint at2 = begin; at2 < end && used < last; ++at2) {
            ++used;
            uint g = instances[at2];
            float2 d = gauss_f[8u * g].xy - float2(pr.pixel_x, pr.pixel_y);
            float4 co = gauss_f[8u * g + 1u];
            float power = gaussian_power(co, d.x, d.y);
            if (power > 0.0f) continue;
            float alpha = min(kAlphaClip, co.w * exp(power));
            if (alpha < kAlphaFloor) continue;
            float4 rp = gauss_f[8u * g + 3u];
            float peak = rp.x * d.x + rp.y * d.y + rp.z;
            [unroll] for (uint s2 = sb; s2 < se; ++s2) {
                float ts = lo + interval * float(s2);
                float td = (ts - peak) * rp.w;
                float ge = rp.w > 0.0f ? exp(-0.5f * td * td) : 0.0f;
                float om = 1.0f - alpha * ge;
                probes[s2] *= (ts > peak ? (1.0f - alpha) : om) * rsqrt(om);
            }
        }
        if (first) in_range = in_range && probes[0] >= 0.5f && probes[8] <= 0.5f;
        uint sid = 0u;
        [unroll] for (uint s3 = 1u; s3 < 8u; ++s3) {
            if (probes[s3] >= 0.5f) sid = s3;
        }
        float nlo = lo + float(sid) * interval;
        float nhi = lo + float(sid + 1u) * interval;
        probes[0] = probes[sid];
        probes[8] = probes[sid + 1u];
        lo = nlo;
        hi = nhi;
        if (tolerance > 0.0f && hi - lo <= tolerance) break;
    }
    float weight = saturate((probes[0] - 0.5f) / (probes[0] - probes[8]));
    float depth = in_range ? weight * hi + (1.0f - weight) * lo : 0.0f;
    float3 ray = pixel_unit_ray(
        pr.pixel_x, pr.pixel_y, mode, fx, fy, cx, cy, k1, k2, k3, k4);
    float3 camera_point = ray * depth;
    sample_f[3u * i] = camera_point.x;
    sample_f[3u * i + 1u] = camera_point.y;
    sample_f[3u * i + 2u] = camera_point.z;
    sample_f[fbase + i] = depth;
    sample_u[i] = last;
    sample_u[point_count + i] = in_range ? 1u : 0u;
}
