#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<float> means;
[[vk::binding(1, 0)]] StructuredBuffer<float> opacities;
[[vk::binding(2, 0)]] StructuredBuffer<float> scales;
[[vk::binding(3, 0)]] StructuredBuffer<float> rotations;
[[vk::binding(4, 0)]] StructuredBuffer<float> covariances;
[[vk::binding(5, 0)]] StructuredBuffer<float> color_src;
[[vk::binding(6, 0)]] StructuredBuffer<float> camera;
[[vk::binding(7, 0)]] RWStructuredBuffer<float4> gauss_f;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> gauss_u;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> key_lo;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> key_hi;
[[vk::binding(11, 0)]] RWStructuredBuffer<uint> values;

#include "splat_enum.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint count = pc.u0;
    uint index = dispatch_id.x;
    if (index >= count) return;

    gauss_u[index] = 0;
    gauss_u[count + index] = 0;
    gauss_u[3 * count + index] = 0;

    uint mode = pc.u16;
    int width = int(pc.u4);
    int height = int(pc.u5);
    int grid_x = int(pc.u6);
    int grid_y = int(pc.u7);
    float fx = asfloat(pc.u8);
    float fy = asfloat(pc.u9);
    float cx = asfloat(pc.u10);
    float cy = asfloat(pc.u11);
    float k1 = asfloat(pc.u12);
    float k2 = asfloat(pc.u13);
    float k3 = asfloat(pc.u14);
    float k4 = asfloat(pc.u15);
    float kernel = asfloat(pc.u17);
    float scale_modifier = asfloat(pc.u18);
    int wrap_width = int(pc.u19);
    uint flags = pc.u3;
    bool has_sh = (flags & 1u) != 0;
    bool has_scales = (flags & 2u) != 0;

    float3 mean = float3(means[3 * index], means[3 * index + 1], means[3 * index + 2]);
    float3 t = xform_point(mean, camera[0], camera[1], camera[2], camera[4], camera[5], camera[6],
                           camera[8], camera[9], camera[10], camera[12], camera[13], camera[14]);
    if (!camera_visible(t, mode)) return;

    float3 scale = 0.0f;
    float4 rotation = float4(1.0f, 0.0f, 0.0f, 0.0f);
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f, c5 = 0.0f;
    if (has_scales) {
        scale = float3(scales[3 * index], scales[3 * index + 1], scales[3 * index + 2]);
        rotation = float4(rotations[4 * index], rotations[4 * index + 1], rotations[4 * index + 2], rotations[4 * index + 3]);
    } else {
        uint base = 6 * index;
        c0 = covariances[base];
        c1 = covariances[base + 1];
        c2 = covariances[base + 2];
        c3 = covariances[base + 3];
        c4 = covariances[base + 4];
        c5 = covariances[base + 5];
    }

    SplatGeom geom = project_splat(
        mean, camera[0], camera[1], camera[2], camera[4], camera[5], camera[6],
        camera[8], camera[9], camera[10], camera[12], camera[13], camera[14],
        mode, width, height, fx, fy, cx, cy, k1, k2, k3, k4, kernel, scale_modifier,
        has_scales, scale, rotation, c0, c1, c2, c3, c4, c5);
    if (!geom.ok) return;

    Projection projected;
    if (mode == kModeOrtho) projected = project_ortho(t, fx, fy, cx, cy);
    else projected = project_camera(t, mode, width, height, fx, fy, cx, cy, k1, k2, k3, k4);
    if (!projected.valid) return;

    float det = geom.cov0 * geom.cov2 - geom.cov1 * geom.cov1;
    if (det == 0.0f) return;
    float det_inv = 1.0f / det;
    float4 conic = float4(geom.cov2 * det_inv, -geom.cov1 * det_inv, geom.cov0 * det_inv,
                          opacities[index] * geom.coef);
    float mid = 0.5f * (geom.cov0 + geom.cov2);
    float root = sqrt(max(0.1f, mid * mid - det));
    float radius = ceil(3.0f * sqrt(max(mid + root, mid - root)));
    uint touched = enumerate_tiles(float2(projected.pixel_x, projected.pixel_y), conic, grid_x, grid_y,
                                   index, 0, wrap_width, 0, 0);
    if (touched == 0) return;

    float clamped0 = 0.0f;
    float clamped1 = 0.0f;
    float clamped2 = 0.0f;
    float3 rgb;
    if (has_sh) {
        uint bases = pc.u2;
        rgb = evaluate_sh(pc.u1, mean, float3(camera[16], camera[17], camera[18]), color_src, index * bases * 3,
                          clamped0, clamped1, clamped2);
    } else {
        rgb = float3(color_src[3 * index], color_src[3 * index + 1], color_src[3 * index + 2]);
    }

    float depth = sqrt(dot(t, t));
    uint depth_key = asuint(depth);
    gauss_f[index * 8 + 0] = float4(projected.pixel_x, projected.pixel_y, asfloat(depth_key), asfloat(uint(radius)));
    gauss_f[index * 8 + 1] = conic;
    gauss_f[index * 8 + 2] = float4(rgb, 0.0f);
    gauss_f[index * 8 + 3] = geom.ray_plane;
    gauss_f[index * 8 + 4] = float4(geom.normal, 0.0f);
    float threshold = 2.0f * log(conic.w / kAlphaFloor);
    float ex = sqrt(max(0.0f, threshold * geom.cov0));
    float ey = sqrt(max(0.0f, threshold * geom.cov2));
    uint x0 = uint(max(0.0f, min(65535.0f, floor(projected.pixel_x - ex))));
    uint x1 = uint(max(0.0f, min(65535.0f, floor(projected.pixel_x + ex) + 1.0f)));
    uint y0 = uint(max(0.0f, min(65535.0f, floor(projected.pixel_y - ey))));
    uint y1 = uint(max(0.0f, min(65535.0f, floor(projected.pixel_y + ey) + 1.0f)));
    gauss_f[index * 8 + 5] = float4(asfloat(x0), asfloat(x1), asfloat(y0), asfloat(y1));
    gauss_f[index * 8 + 6] = float4(clamped0, clamped1, clamped2, 0.0f);
    gauss_f[index * 8 + 7] = float4(geom.cov0, geom.cov1, geom.cov2, geom.coef);

    gauss_u[index] = touched;
    gauss_u[count + index] = 1;
    gauss_u[2 * count + index] = depth_key;
    gauss_u[3 * count + index] = uint(radius);
}
