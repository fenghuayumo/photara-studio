[[vk::binding(19, 0)]] RWStructuredBuffer<float> means_m;
[[vk::binding(20, 0)]] RWStructuredBuffer<float> means_v;
[[vk::binding(21, 0)]] RWStructuredBuffer<float> scales_m;
[[vk::binding(22, 0)]] RWStructuredBuffer<float> scales_v;
[[vk::binding(23, 0)]] RWStructuredBuffer<float> rotations_m;
[[vk::binding(24, 0)]] RWStructuredBuffer<float> rotations_v;
[[vk::binding(25, 0)]] RWStructuredBuffer<float> opacity_m;
[[vk::binding(26, 0)]] RWStructuredBuffer<float> opacity_v;
[[vk::binding(27, 0)]] StructuredBuffer<float> structure_settings;

float structure_step(float previous, float grad, inout float m, inout float v,
                     float lr, float lo, float hi) {
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
    [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
        uint i = 3u * row + axis;
        float m = means_m[i], v = means_v[i];
        means[i] = structure_step(means[i], model_grad[axis], m, v,
            structure_settings[0], -unlimited, unlimited);
        means_m[i] = m; means_v[i] = v;
        float previous = raw_log_scales[i];
        float grad = model_grad[11u + axis];
        if (structure_settings[12] > 0.0f && isfinite(previous) && previous > -40.0f)
            grad += structure_settings[12] * exp(previous);
        m = scales_m[i]; v = scales_v[i];
        raw_log_scales[i] = structure_step(previous, grad, m, v,
            structure_settings[1], lo, hi);
        scales_m[i] = m; scales_v[i] = v;
    }
    [unroll] for (uint axis = 0u; axis < 4u; ++axis) {
        uint i = 4u * row + axis;
        float m = rotations_m[i], v = rotations_v[i];
        raw_rotations[i] = structure_step(raw_rotations[i], model_grad[14u + axis], m, v,
            structure_settings[2], -unlimited, unlimited);
        rotations_m[i] = m; rotations_v[i] = v;
    }
    float previous = opacity_logits[row], grad = model_grad[18];
    if (structure_settings[11] > 0.0f && isfinite(previous)) {
        float alpha = 1.0f / (1.0f + exp(-previous));
        grad += structure_settings[11] * alpha * (1.0f - alpha);
    }
    float m = opacity_m[row], v = opacity_v[row];
    opacity_logits[row] = structure_step(previous, grad, m, v,
        structure_settings[3], -12.0f, 12.0f);
    opacity_m[row] = m; opacity_v[row] = v;
    structure_output[row] = blend_grad[17u * pc.u0 + row];
    if (pc.u17 != 0u) structure_output[pc.u0 + row] = grad;
}
