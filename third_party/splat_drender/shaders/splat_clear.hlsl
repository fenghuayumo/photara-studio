#include "splat_math.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<float> values;

[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    if (dispatch_id.x < pc.u0) values[dispatch_id.x] = 0.0f;
}
