#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] StructuredBuffer<float> out_f;
[[vk::binding(4, 0)]] StructuredBuffer<uint> out_u;
[[vk::binding(5, 0)]] StructuredBuffer<float> loss_depth;
// x is the ray-length median used by blending backward; y is
// dL/d(median) divided by -dT/d(median), matching the CUDA PixelState::dL_dmt.
[[vk::binding(6, 0)]] RWStructuredBuffer<float2> median_state;

[numthreads(16, 16, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread : SV_GroupThreadID) {
    uint width = pc.u0;
    uint height = pc.u1;
    uint tiles_x = pc.u2;
    uint mode = pc.u3;
    int wrap_width = int(pc.u4);
    uint gaussian_count = pc.u5;
    uint pixel_count = pc.u6;
    float fx = asfloat(pc.u7);
    float fy = asfloat(pc.u8);
    float cx = asfloat(pc.u9);
    float cy = asfloat(pc.u10);
    float k1 = asfloat(pc.u11);
    float k2 = asfloat(pc.u12);
    float k3 = asfloat(pc.u13);
    float k4 = asfloat(pc.u14);

    uint px = group_id.x * 16u + group_thread.x;
    uint py = group_id.y * 16u + group_thread.y;
    if (px >= width || py >= height) return;
    uint pixel = py * width + px;
    float ray_z = pixel_ray_z(float(px), float(py), mode, fx, fy, cx, cy, k1, k2, k3, k4);
    float median = out_f[7u * pixel_count + pixel] / max(ray_z, 1.0e-8f);
    float upstream = loss_depth[pixel] * ray_z;
    median_state[pixel] = float2(median, 0.0f);

    uint last = out_u[gaussian_count + pixel];
    if (median == 0.0f || last == 0u || upstream == 0.0f) return;
    uint tile = group_id.y * tiles_x + group_id.x;
    uint begin = ranges[2u * tile];
    uint finish = min(ranges[2u * tile + 1u], begin + last);
    float dT_dmedian = 0.0f;
    [loop] for (uint at = begin; at < finish; ++at) {
        uint gaussian = instances[at];
        float2 mean = gauss_f[8u * gaussian].xy;
        float4 conic = gauss_f[8u * gaussian + 1u];
        float2 delta = float2(
            wrap_dx(mean.x - float(px), wrap_width, mode), mean.y - float(py));
        float power = gaussian_power(conic, delta.x, delta.y);
        if (power > 0.0f) continue;
        float alpha = min(kAlphaClip, conic.w * exp(power));
        if (alpha < kAlphaFloor) continue;
        float4 plane = gauss_f[8u * gaussian + 3u];
        float peak = plane.x * delta.x + plane.y * delta.y + plane.z;
        float td = (median - peak) * plane.w;
        float gt = alpha * exp(-0.5f * td * td);
        dT_dmedian += -0.25f * gt / (1.0f - gt) * abs(td) * plane.w;
    }
    median_state[pixel].y = upstream / max(-dT_dmedian, 1.0e-7f);
}
