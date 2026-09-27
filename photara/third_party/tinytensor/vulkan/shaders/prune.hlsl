#include "common.hlsli"

// One thread per Gaussian row: the fused Vulkan twin of the CUDA
// adc_plus_prune_kernel. Every tensor is row-major contiguous with
// dim0 == count; strides travel as push constants so the shader stays
// independent of the rank the host happens to use for [N], [N, 3] or
// [N, 16, 3] storage.
struct PushConstants
{
    uint count;
    uint mean_stride;
    uint scale_stride;
    uint rotation_stride;
    uint opacity_stride;
    uint sh_stride;
    float minimum_opacity;
    float maximum_bounds;
    float center_x;
    float center_y;
    float center_z;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer means;
[[vk::binding(1, 0)]] RWByteAddressBuffer log_scales;
[[vk::binding(2, 0)]] RWByteAddressBuffer quaternions;
[[vk::binding(3, 0)]] RWByteAddressBuffer opacity_logits;
[[vk::binding(4, 0)]] RWByteAddressBuffer sh;
[[vk::binding(5, 0)]] RWByteAddressBuffer keep;
[[vk::binding(6, 0)]] RWByteAddressBuffer hard;
[[vk::binding(7, 0)]] RWByteAddressBuffer opacities;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint row = dtid.x;
    if (row >= pc.count) return;

    const float logit = load_f32(opacity_logits, row * pc.opacity_stride * 4u);
    const float opacity = 1.0f / (1.0f + exp(-logit));
    store_f32(opacities, row * 4u, opacity);

    bool bad = !isfinite(logit);
    bool outside = false;
    float maximum_scale = 0.0f;
    const float center[3] = {pc.center_x, pc.center_y, pc.center_z};
    [unroll]
    for (uint axis = 0; axis < 3u; ++axis)
    {
        const float mean = load_f32(means, (row * pc.mean_stride + axis) * 4u);
        const float log_scale = load_f32(
            log_scales, (row * pc.scale_stride + axis) * 4u);
        bad = bad || !isfinite(mean) || !isfinite(log_scale);
        if (isfinite(log_scale))
        {
            maximum_scale = max(maximum_scale, exp(log_scale));
        }
        outside = outside || abs(mean - center[axis]) > pc.maximum_bounds;
    }
    [unroll]
    for (uint component = 0; component < 4u; ++component)
    {
        bad = bad || !isfinite(load_f32(
            quaternions, (row * pc.rotation_stride + component) * 4u));
    }
    [loop]
    for (uint component = 0; component < pc.sh_stride; ++component)
    {
        bad = bad || !isfinite(load_f32(
            sh, (row * pc.sh_stride + component) * 4u));
    }

    const bool hard_row =
        bad || outside || maximum_scale > pc.maximum_bounds;
    store_u8(hard, row, hard_row ? 1u : 0u);
    store_u8(keep, row,
        !hard_row && opacity >= pc.minimum_opacity ? 1u : 0u);
}
