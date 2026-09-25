// In-place Float32 Adam update for training backends.  Parameter and moment
// state stay resident in Vulkan buffers; one dispatch replaces the chain of
// elementwise temporaries that a tensor-level expression would create.

struct PushConstants
{
    uint count;
    uint parameter_offset;
    uint gradient_offset;
    uint first_offset;
    uint second_offset;
    float learning_rate;
    float secondary_learning_rate;
    uint group_stride;
    float beta1;
    float beta2;
    float correction1;
    float correction2;
    float epsilon;
    float clamp_min;
    float clamp_max;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer parameter;
[[vk::binding(1, 0)]] ByteAddressBuffer gradient;
[[vk::binding(2, 0)]] RWByteAddressBuffer first;
[[vk::binding(3, 0)]] RWByteAddressBuffer second;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count) return;
    const uint p_address = pc.parameter_offset + index * 4u;
    const uint g_address = pc.gradient_offset + index * 4u;
    const uint m_address = pc.first_offset + index * 4u;
    const uint v_address = pc.second_offset + index * 4u;
    const float previous = asfloat(parameter.Load(p_address));
    const float grad = asfloat(gradient.Load(g_address));
    if (!isfinite(previous) || !isfinite(grad))
    {
        first.Store(m_address, asuint(0.0f));
        second.Store(v_address, asuint(0.0f));
        const float safe_previous = isfinite(previous) ? previous : 0.0f;
        parameter.Store(p_address, asuint(clamp(safe_previous, pc.clamp_min, pc.clamp_max)));
        return;
    }

    const float m = pc.beta1 * asfloat(first.Load(m_address)) + (1.0f - pc.beta1) * grad;
    const float v = pc.beta2 * asfloat(second.Load(v_address)) +
                    (1.0f - pc.beta2) * grad * grad;
    if (!isfinite(m) || !isfinite(v))
    {
        first.Store(m_address, asuint(0.0f));
        second.Store(v_address, asuint(0.0f));
        parameter.Store(p_address, asuint(clamp(previous, pc.clamp_min, pc.clamp_max)));
        return;
    }

    first.Store(m_address, asuint(m));
    second.Store(v_address, asuint(v));
    const float lr = pc.group_stride != 0u && index % pc.group_stride >= 3u
        ? pc.secondary_learning_rate : pc.learning_rate;
    const float candidate = previous - lr * (m / pc.correction1) /
                            (sqrt(v / pc.correction2) + pc.epsilon);
    const float updated = isfinite(candidate) ? candidate : previous;
    parameter.Store(p_address, asuint(clamp(updated, pc.clamp_min, pc.clamp_max)));
}
