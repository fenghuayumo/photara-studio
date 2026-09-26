#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] StructuredBuffer<float> camera;
[[vk::binding(4, 0)]] StructuredBuffer<float> points;
[[vk::binding(5, 0)]] StructuredBuffer<float> sample_f;
[[vk::binding(6, 0)]] StructuredBuffer<uint> sample_u;
[[vk::binding(7, 0)]] StructuredBuffer<float> loss_points;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> blend_grad;
[[vk::binding(9, 0)]] RWStructuredBuffer<float> point_grad;

void atomic_add_f32(uint index, float value) {
    if (value == 0.0f || isnan(value) || isinf(value)) return;
    uint expected = blend_grad[index];
    [loop] for (;;) {
        uint desired = asuint(asfloat(expected) + value);
        uint observed;
        InterlockedCompareExchange(blend_grad[index], expected, desired, observed);
        if (observed == expected) return;
        expected = observed;
    }
}

[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint point_count = pc.u0;
    uint i = dispatch_id.x;
    if (i >= point_count) return;
    uint width = pc.u1;
    uint height = pc.u2;
    uint tiles_x = pc.u3;
    uint mode = pc.u4;
    uint count = pc.u5;
    float fx = asfloat(pc.u6);
    float fy = asfloat(pc.u7);
    float cx = asfloat(pc.u8);
    float cy = asfloat(pc.u9);
    float k1 = asfloat(pc.u10);
    float k2 = asfloat(pc.u11);
    float k3 = asfloat(pc.u12);
    float k4 = asfloat(pc.u13);
    point_grad[3u * i] = 0.0f;
    point_grad[3u * i + 1u] = 0.0f;
    point_grad[3u * i + 2u] = 0.0f;
    uint last = sample_u[i];
    if (last == 0u || sample_u[point_count + i] == 0u) return;
    float3 world = float3(points[3u * i], points[3u * i + 1u], points[3u * i + 2u]);
    float3 t = xform_point(
        world,
        camera[0], camera[1], camera[2],
        camera[4], camera[5], camera[6],
        camera[8], camera[9], camera[10],
        camera[12], camera[13], camera[14]);
    Projection pr = project_camera(
        t, mode, int(width), int(height), fx, fy, cx, cy, k1, k2, k3, k4);
    if (!pr.valid) return;
    uint tx = min(tiles_x - 1u, uint(max(0.0f, floor((pr.pixel_x + 0.5f) / 16.0f))));
    uint ty = min(
        (height + 15u) / 16u - 1u,
        uint(max(0.0f, floor((pr.pixel_y + 0.5f) / 16.0f))));
    uint begin = ranges[2u * (ty * tiles_x + tx)];
    uint end = ranges[2u * (ty * tiles_x + tx) + 1u];
    last = min(last, end - begin);
    float median = sample_f[3u * point_count + i];
    float3 dl = float3(
        loss_points[3u * i], loss_points[3u * i + 1u], loss_points[3u * i + 2u]);
    float3 ray = pixel_unit_ray(
        pr.pixel_x, pr.pixel_y, mode, fx, fy, cx, cy, k1, k2, k3, k4);
    float d_depth = dot(ray, dl);
    float2 dxy = 0.0f;
    if (mode == kModePinhole || mode == kModeOrtho) {
        float2 nf = float2((pr.pixel_x - cx) / fx, (pr.pixel_y - cy) / fy);
        float rln = rsqrt(dot(nf, nf) + 1.0f);
        float rln2 = 1.0f / (dot(nf, nf) + 1.0f);
        float depth = median * rln;
        float aux = (dl.x * nf.x + dl.y * nf.y + dl.z) * rln2;
        dxy = float2(
            (dl.x - aux * nf.x) * depth / fx,
            (dl.y - aux * nf.y) * depth / fy);
    }
    float dT = 0.0f;
    [loop] for (uint p = 0u; p < last; ++p) {
        uint g = instances[begin + p];
        float2 d = gauss_f[7u * g].xy - float2(pr.pixel_x, pr.pixel_y);
        float4 co = gauss_f[7u * g + 1u];
        float power = gaussian_power(co, d.x, d.y);
        if (power > 0.0f) continue;
        float alpha = min(kAlphaClip, co.w * exp(power));
        if (alpha < kAlphaFloor) continue;
        float4 rp = gauss_f[7u * g + 3u];
        float peak = rp.x * d.x + rp.y * d.y + rp.z;
        float td = (median - peak) * rp.w;
        float gt = alpha * exp(-0.5f * td * td);
        dT += -0.25f * gt / (1.0f - gt) * abs(td) * rp.w;
    }
    float scale = d_depth / max(-dT, 1.0e-7f);
    [loop] for (uint rev = last; rev > 0u; --rev) {
        uint g = instances[begin + rev - 1u];
        float2 d = gauss_f[7u * g].xy - float2(pr.pixel_x, pr.pixel_y);
        float4 co = gauss_f[7u * g + 1u];
        float power = gaussian_power(co, d.x, d.y);
        if (power > 0.0f) continue;
        float G = exp(power);
        float raw = co.w * G;
        if (raw < kAlphaFloor) continue;
        float alpha = min(kAlphaClip, raw);
        float4 rp = gauss_f[7u * g + 3u];
        float peak = rp.x * d.x + rp.y * d.y + rp.z;
        float td = (median - peak) * rp.w;
        float ge = exp(-0.5f * td * td);
        float gt = alpha * ge;
        float dgt = scale * 0.25f / (1.0f - gt);
        dgt = median > peak ? dgt : -dgt;
        dgt = rp.w > 0.0f ? dgt : 0.0f;
        float dopa = dgt * ge - scale * (td > 0.0f ? 0.5f / (1.0f - alpha) : 0.0f);
        float ddelta = -dgt * gt * td;
        float dpeak = -ddelta * rp.w;
        if (mode == kModeFisheye && raw >= kAlphaClip) dopa = 0.0f;
        float dG = co.w * dopa;
        float gdx = G * d.x;
        float gdy = G * d.y;
        float ddx = dG * (-gdx * co.x - gdy * co.y) + dpeak * rp.x;
        float ddy = dG * (-gdy * co.z - gdx * co.y) + dpeak * rp.y;
        dxy -= float2(ddx, ddy);
        uint mb = 3u * g;
        uint cb = 3u * count + 4u * g;
        uint pb = 10u * count + 4u * g;
        atomic_add_f32(mb, ddx);
        atomic_add_f32(mb + 1u, ddy);
        atomic_add_f32(cb, -0.5f * gdx * d.x * dG);
        atomic_add_f32(cb + 1u, -0.5f * gdx * d.y * dG);
        atomic_add_f32(cb + 2u, -0.5f * gdy * d.y * dG);
        atomic_add_f32(cb + 3u, G * dopa);
        atomic_add_f32(pb, dpeak * d.x);
        atomic_add_f32(pb + 1u, dpeak * d.y);
        atomic_add_f32(pb + 2u, dpeak);
        atomic_add_f32(pb + 3u, ddelta * (median - peak));
    }
    float3 gc = float3(
        pr.du0 * dxy.x + pr.dv0 * dxy.y,
        pr.du1 * dxy.x + pr.dv1 * dxy.y,
        pr.du2 * dxy.x + pr.dv2 * dxy.y);
    float3 gw = float3(
        camera[0] * gc.x + camera[1] * gc.y + camera[2] * gc.z,
        camera[4] * gc.x + camera[5] * gc.y + camera[6] * gc.z,
        camera[8] * gc.x + camera[9] * gc.y + camera[10] * gc.z);
    point_grad[3u * i] = gw.x;
    point_grad[3u * i + 1u] = gw.y;
    point_grad[3u * i + 2u] = gw.z;
}
