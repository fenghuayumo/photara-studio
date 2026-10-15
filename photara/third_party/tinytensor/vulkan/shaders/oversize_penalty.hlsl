[[vk::binding(0, 0)]] RWStructuredBuffer<float> log_scales;
[[vk::binding(1, 0)]] StructuredBuffer<int> radii;
struct Push {
    uint count;
    float inverse_resolution, screen_limit, penalty, scales_lr;
    uint scale_offset, radius_offset;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;
[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    const uint index = dtid.x;
    if (index >= pc.count || radii[pc.radius_offset + index] <= 0) return;
    const float screen = float(radii[pc.radius_offset + index]) * pc.inverse_resolution;
    if (!(screen > pc.screen_limit) || !(pc.penalty > 0.F)) return;
    const float push = pc.penalty * log2(screen / pc.screen_limit);
    const float largest = max(log_scales[pc.scale_offset + 3 * index],
        max(log_scales[pc.scale_offset + 3 * index + 1], log_scales[pc.scale_offset + 3 * index + 2]));
    for (int axis = 0; axis < 3; ++axis) {
        const float share = exp(2.F * (log_scales[pc.scale_offset + 3 * index + axis] - largest));
        log_scales[pc.scale_offset + 3 * index + axis] -= pc.scales_lr * push * share;
    }
}
