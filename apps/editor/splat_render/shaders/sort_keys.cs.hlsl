#include "ring_math.hlsl"

cbuffer FrameBlock : register(b0) {
    SplatFrame frame;
};

StructuredBuffer<float4> centers : register(t1);
RWStructuredBuffer<uint> keys0 : register(u6);
RWStructuredBuffer<uint> vals0 : register(u8);

struct PushConstants {
    uint count;
    uint groups;
    uint shift;
    uint parity;
};
[[vk::push_constant]]
PushConstants push;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.x;
    if (index >= push.count) return;
    float3 camera = transform_point(frame, centers[index].xyz);
    uint key = 0xffffffffu;
    if (camera_in_front(frame, camera)) {
        // Match the renderer radial order; camera-Z produced sheet artifacts
        // in the ori_img 30k-step mesh comparison.
        // float depth = camera.z; // Original 3DGS camera-Z order.
        float depth = length(camera);
        key = ~depth_sort_key(depth);
    }
    keys0[index] = key;
    vals0[index] = index;
}
