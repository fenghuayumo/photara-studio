#include "sift_common.hlsli"

// ComputeDOG: difference of Gaussians plus the gradient/orientation texture
// (0.5 * |grad|, atan2(dy, dx)). Gradient fetches are raw like the CUDA
// tex1Dfetch reads: index +/- 1 crosses row borders, out-of-range reads zero.
struct PushConstants
{
    uint width;
    uint height;
    uint write_gradient;
    uint pad0;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer level_current;
[[vk::binding(1, 0)]] RWByteAddressBuffer level_previous;
[[vk::binding(2, 0)]] RWByteAddressBuffer dog;
[[vk::binding(3, 0)]] RWByteAddressBuffer gradient;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(16, 16, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint x = dtid.x;
    const uint y = dtid.y;
    if (x >= pc.width || y >= pc.height) return;
    const uint count = pc.width * pc.height;
    const uint index = y * pc.width + x;

    const float previous = load_f32(level_previous, index * 4u);
    const float current = load_f32(level_current, index * 4u);
    store_f32(dog, index * 4u, current - previous);

    if (pc.write_gradient == 0u) return;

    const float vxn = fetch_f32(level_current, index + 1u, count);
    const float vxp = fetch_f32(level_current, index - 1u, count);
    const float vyp = fetch_f32(level_current, index - pc.width, count);
    const float vyn = fetch_f32(level_current, index + pc.width, count);
    const float dx = vxn - vxp;
    const float dy = vyn - vyp;
    const float grd = 0.5f * sqrt(dx * dx + dy * dy);
    const float rot = grd == 0.0f ? 0.0f : atan2(dy, dx);
    store_f2(gradient, index * 8u, float2(grd, rot));
}
