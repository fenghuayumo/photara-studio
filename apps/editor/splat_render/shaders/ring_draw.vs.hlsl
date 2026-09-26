#include "ring_math.hlsl"

cbuffer FrameBlock : register(b0) {
    GutFrame frame;
};

StructuredBuffer<float4> quads : register(t5);

struct VsOut {
    float4 position : SV_Position;
    nointerpolation float4 color : COLOR0;
    float2 uv : TEXCOORD0;
};

static const float2 k_corners[6] = {
    float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
    float2(-1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0)};

VsOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
    VsOut output;
    uint base = iid * 7;
    float4 center = quads[base + 1];
    float2 axis1 = float2(quads[base + 2].w, quads[base + 3].w);
    float2 axis2 = float2(quads[base + 4].w, quads[base + 5].w);
    output.color = quads[base];
    output.uv = k_corners[vid];
    if (center.w < 0.5 || dot(axis1, axis1) + dot(axis2, axis2) < 1e-4) {
        output.position = float4(2.0, 2.0, 0.0, 1.0);
        return output;
    }
    float2 resolution = max(frame.viewport.xy, float2(1.0, 1.0));
    float2 pixel = quads[base + 6].xy + output.uv.x * axis1 + output.uv.y * axis2;
    float2 ndc = float2(
        pixel.x / resolution.x * 2.0 - 1.0,
        pixel.y / resolution.y * 2.0 - 1.0);
    output.position = float4(ndc, 0.5, 1.0);
    return output;
}
