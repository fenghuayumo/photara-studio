[[vk::binding(19, 0)]] RWStructuredBuffer<float> means_m;
[[vk::binding(20, 0)]] RWStructuredBuffer<float> means_v;
[[vk::binding(21, 0)]] RWStructuredBuffer<float> scales_m;
[[vk::binding(22, 0)]] RWStructuredBuffer<float> scales_v;
[[vk::binding(23, 0)]] RWStructuredBuffer<float> rotations_m;
[[vk::binding(24, 0)]] RWStructuredBuffer<float> rotations_v;
[[vk::binding(25, 0)]] RWStructuredBuffer<float> opacity_m;
[[vk::binding(26, 0)]] RWStructuredBuffer<float> opacity_v;
[[vk::binding(27, 0)]] StructuredBuffer<float> structure_settings;

#include "../../../photara/third_party/tinytensor/vulkan/shaders/shape_regularizers_math.hlsli"

float structure_step(float previous, float grad, inout float m, inout float v, float lr, float lo,
                     float hi) {
    if (!isfinite(previous) || !isfinite(grad)) {
        m = v = 0.0f;
        return clamp(isfinite(previous) ? previous : 0.0f, lo, hi);
    }
    m = structure_settings[4] * m + (1.0f - structure_settings[4]) * grad;
    v = structure_settings[5] * v + (1.0f - structure_settings[5]) * grad * grad;
    if (!isfinite(m) || !isfinite(v)) {
        m = v = 0.0f;
        return clamp(previous, lo, hi);
    }
    float candidate = previous - lr * (m / structure_settings[6]) /
                                     (sqrt(v / structure_settings[7]) + structure_settings[8]);
    return clamp(isfinite(candidate) ? candidate : previous, lo, hi);
}

void update_structure_row(uint row) {
    float lo = structure_settings[9], hi = structure_settings[10];
    float unlimited = asfloat(0x7f800000u);
    // Read the old row before updating any component: the shape prior couples
    // all scales and all quaternion components, including culled Gaussians.
    float3 raw_scales = float3(raw_log_scales[3u*row], raw_log_scales[3u*row+1u],
                               raw_log_scales[3u*row+2u]);
    float4 raw_quat = float4(raw_rotations[4u*row], raw_rotations[4u*row+1u],
                            raw_rotations[4u*row+2u], raw_rotations[4u*row+3u]);
    float3 sg = float3(model_grad[11u], model_grad[12u], model_grad[13u]);
    float4 qg = float4(model_grad[14u], model_grad[15u], model_grad[16u], model_grad[17u]);
    [unroll] for (uint a = 0u; a < 3u; ++a)
        if (structure_settings[12] > 0.0f && isfinite(raw_scales[a]) && raw_scales[a] > -40.0f)
            sg[a] += structure_settings[12] * exp(raw_scales[a]);
    shape_regularizer_gradients(raw_scales, raw_quat, sg, qg,
        structure_settings[13], structure_settings[14], structure_settings[15],
        structure_settings[16], structure_settings[17]);
    [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
        uint i = 3u * row + axis;
        float m = means_m[i], v = means_v[i];
        means[i] = structure_step(means[i], model_grad[axis], m, v, structure_settings[0],
                                  -unlimited, unlimited);
        means_m[i] = m;
        means_v[i] = v;
        float previous = raw_scales[axis];
        float grad = sg[axis];
        m = scales_m[i];
        v = scales_v[i];
        raw_log_scales[i] = structure_step(previous, grad, m, v, structure_settings[1], lo, hi);
        scales_m[i] = m;
        scales_v[i] = v;
    }
    [unroll] for (uint axis = 0u; axis < 4u; ++axis) {
        uint i = 4u * row + axis;
        float m = rotations_m[i], v = rotations_v[i];
        raw_rotations[i] = structure_step(raw_quat[axis], qg[axis], m, v,
                                          structure_settings[2], -unlimited, unlimited);
        rotations_m[i] = m;
        rotations_v[i] = v;
    }
    float previous = opacity_logits[row], grad = model_grad[18];
    if (structure_settings[11] > 0.0f && isfinite(previous)) {
        float alpha = 1.0f / (1.0f + exp(-previous));
        grad += structure_settings[11] * alpha * (1.0f - alpha);
    }
    float m = opacity_m[row], v = opacity_v[row];
    opacity_logits[row] =
        structure_step(previous, grad, m, v, structure_settings[3], -12.0f, 12.0f);
    opacity_m[row] = m;
    opacity_v[row] = v;
    structure_output[row] = blend_grad[17u * pc.u0 + row];
    if (pc.u17 != 0u)
        structure_output[pc.u0 + row] = grad;
}
