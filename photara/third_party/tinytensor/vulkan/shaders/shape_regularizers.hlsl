// Same shape-gradient math as apply_shape_regularizers_kernel (CUDA).
[[vk::binding(0, 0)]] StructuredBuffer<float> log_scales;
[[vk::binding(1, 0)]] StructuredBuffer<float> quaternions;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> scale_gradients;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> quaternion_gradients;
struct Push {
    uint count;
    float erank_weight, erank_s3_weight, quat_weight;
    float axis_ratio_weight, max_axis_ratio;
    uint scale_offset, quat_offset, scale_grad_offset, quat_grad_offset;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;
#include "shape_regularizers_math.hlsli"
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    const uint index = dtid.x;
    if (index >= pc.count) return;
    uint s = pc.scale_offset + 3u * index, q = pc.quat_offset + 4u * index;
    uint gs = pc.scale_grad_offset + 3u * index, gq = pc.quat_grad_offset + 4u * index;
    float3 raw_scales = float3(log_scales[s], log_scales[s+1u], log_scales[s+2u]);
    float4 raw_quat = float4(quaternions[q], quaternions[q+1u], quaternions[q+2u], quaternions[q+3u]);
    float3 sg = float3(scale_gradients[gs], scale_gradients[gs+1u], scale_gradients[gs+2u]);
    float4 qg = float4(quaternion_gradients[gq], quaternion_gradients[gq+1u],
                     quaternion_gradients[gq+2u], quaternion_gradients[gq+3u]);
    shape_regularizer_gradients(raw_scales, raw_quat, sg, qg,
        pc.erank_weight, pc.erank_s3_weight, pc.quat_weight, pc.axis_ratio_weight, pc.max_axis_ratio);
    [unroll] for (uint a = 0u; a < 3u; ++a) scale_gradients[gs+a] = sg[a];
    [unroll] for (uint a = 0u; a < 4u; ++a) quaternion_gradients[gq+a] = qg[a];
}
