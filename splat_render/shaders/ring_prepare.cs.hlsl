#include "ring_math.hlsl"

cbuffer FrameBlock : register(b0) {
    GutFrame frame;
};

StructuredBuffer<float4> centers : register(t1);
StructuredBuffer<float4> scales : register(t2);
StructuredBuffer<float4> rotations : register(t3);
ByteAddressBuffer harmonics : register(t4);
RWStructuredBuffer<float4> quads : register(u5);
StructuredBuffer<uint> sorted : register(t8);

struct PushConstants {
    uint count;
    uint groups;
    uint shift;
    uint parity;
};
[[vk::push_constant]]
PushConstants push;

void store_quad(
    uint slot, float4 color, float4 center, float4 scale, float4 row0, float4 row1,
    float4 row2, float4 box) {
    uint base = slot * 7;
    quads[base + 0] = color;
    quads[base + 1] = center;
    quads[base + 2] = scale;
    quads[base + 3] = row0;
    quads[base + 4] = row1;
    quads[base + 5] = row2;
    quads[base + 6] = box;
}

void store_hidden(uint slot) {
    store_quad(slot, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint slot = tid.x;
    if (slot >= push.count) return;

    uint id = sorted[slot];
    float4 center_opacity = centers[id];
    float opacity = center_opacity.w;
    if (opacity <= 1.0 / 255.0) {
        store_hidden(slot);
        return;
    }
    float3 center = center_opacity.xyz;
    float3 scale = max(scales[id].xyz, 1e-8);
    float4 q = rotations[id];
    float qn = length(q);
    if (qn < 1e-8) q = float4(1.0, 0.0, 0.0, 0.0);
    else q /= qn;
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float rx = r * x;
    float ry = r * y;
    float rz = r * z;
    float3 row0 = float3(1.0 - 2.0 * (yy + zz), 2.0 * (xy + rz), 2.0 * (xz - ry));
    float3 row1 = float3(2.0 * (xy - rz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + rx));
    float3 row2 = float3(2.0 * (xz + ry), 2.0 * (yz - rx), 1.0 - 2.0 * (xx + yy));

    float2 pixel;
    if (!project_pixel(frame, center, pixel)) {
        store_hidden(slot);
        return;
    }
    float2 axis1;
    float2 axis2;
    if (!project_axes(frame, center, row0, row1, row2, scale, pixel, axis1, axis2)) {
        store_hidden(slot);
        return;
    }
    float2 resolution = max(frame.viewport.xy, float2(1.0, 1.0));
    float2 extent = abs(axis1) + abs(axis2);
    if (pixel.x + extent.x < 0.0 || pixel.y + extent.y < 0.0 ||
        pixel.x - extent.x > resolution.x || pixel.y - extent.y > resolution.y) {
        store_hidden(slot);
        return;
    }

    float3 rgb = sh_color(
        harmonics, id, center, frame.meta.x, frame.meta.z, frame.eye.xyz);
    // axis1 lives in (scale.w, row0.w), axis2 in (row1.w, row2.w), pixel in box.xy.
    store_quad(
        slot, float4(rgb, opacity), float4(center, 1.0), float4(scale, axis1.x),
        float4(row0, axis1.y), float4(row1, axis2.x), float4(row2, axis2.y),
        float4(pixel, extent));
}
