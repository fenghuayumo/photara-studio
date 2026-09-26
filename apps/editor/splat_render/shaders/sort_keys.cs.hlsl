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
        float depth = camera_model(frame) == k_model_equirect ? length(camera) : camera.z;
        key = ~depth_sort_key(depth);
    }
    keys0[index] = key;
    vals0[index] = index;
}
