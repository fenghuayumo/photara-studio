#include "common.hlsli"

struct PushConstants {
    uint count;
    float camera_x;
    float camera_y;
    float camera_z;
    float angular_normalization;
    float scale_modifier;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer means;
[[vk::binding(1, 0)]] RWByteAddressBuffer log_scales;
[[vk::binding(2, 0)]] RWByteAddressBuffer quaternions;
[[vk::binding(3, 0)]] RWByteAddressBuffer sizes;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

float3 rotate_axis(float4 q, float3 axis) {
    float3 t = 2.0f * cross(q.yzw, axis);
    return axis + q.x * t + cross(q.yzw, t);
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;
    if (i >= pc.count) return;
    float3 offset = float3(
        load_f32(means, (3u*i) * 4u) - pc.camera_x,
        load_f32(means, (3u*i+1u) * 4u) - pc.camera_y,
        load_f32(means, (3u*i+2u) * 4u) - pc.camera_z);
    float distance = length(offset);
    if (!(distance > 1.0e-8f) || !isfinite(distance)) {
        store_f32(sizes, i*4u, 0.0f);
        return;
    }
    float3 n = offset / distance;
    float3 u = abs(n.z) < 0.9f ? float3(-n.y, n.x, 0.0f)
                                : float3(0.0f, -n.z, n.y);
    u = normalize(u);
    float3 v = cross(n, u);
    float4 q = float4(
        load_f32(quaternions, (4u*i) * 4u),
        load_f32(quaternions, (4u*i+1u) * 4u),
        load_f32(quaternions, (4u*i+2u) * 4u),
        load_f32(quaternions, (4u*i+3u) * 4u));
    q *= rsqrt(max(dot(q, q), 1.0e-20f));
    float a = 0.0f, b = 0.0f, c = 0.0f;
    [unroll]
    for (uint axis = 0u; axis < 3u; ++axis) {
        float3 rotated = rotate_axis(q, axis == 0u ? float3(1,0,0)
                                    : axis == 1u ? float3(0,1,0) : float3(0,0,1));
        float du = dot(rotated, u), dv = dot(rotated, v);
        float scale = exp(2.0f * load_f32(log_scales, (3u*i+axis)*4u));
        a += scale*du*du;
        b += scale*du*dv;
        c += scale*dv*dv;
    }
    float lambda = 0.5f * (a+c+sqrt((a-c)*(a-c)+4.0f*b*b));
    float angle = atan2(3.0f * pc.scale_modifier * sqrt(max(lambda, 0.0f)), distance);
    store_f32(sizes, i*4u, isfinite(angle) ? angle*pc.angular_normalization : 0.0f);
}
