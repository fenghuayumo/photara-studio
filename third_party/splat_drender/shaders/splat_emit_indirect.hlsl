#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(1, 0)]] StructuredBuffer<uint> gauss_u;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> key_lo;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> key_hi;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> values;
[[vk::binding(5, 0)]] StructuredBuffer<uint> sorted_ids;
[[vk::binding(6, 0)]] StructuredBuffer<uint> compact;
[[vk::binding(7, 0)]] StructuredBuffer<uint> control;

#include "splat_enum.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    if (control[2] != 0u) return;
    uint mode = pc.u1;
    uint gaussian_count = pc.u5;
    uint count = mode == 0u ? gaussian_count : control[1];
    uint index = dispatch_id.x;
    if (index >= count) return;
    uint gaussian_slots = pc.u6;

    // Depth-key compaction is launched over all Gaussians. The two remaining
    // modes are dispatched indirectly from the visible count.
    if (mode == 0u) {
        if (gauss_u[gaussian_count + index] == 0u) return;
        uint pos = gauss_u[5u * gaussian_count + index] - 1u;
        key_lo[pos] = gauss_u[2u * gaussian_count + index];
        values[pos] = index;
        return;
    }
    if (mode == 1u) {
        values[index] = gauss_u[sorted_ids[index]];
        return;
    }

    uint gaussian = sorted_ids[index];
    uint exclusive = index == 0u ? 0u : compact[index - 1u];
    float4 mean_depth = gauss_f[gaussian * gaussian_slots];
    float4 conic = gauss_f[gaussian * gaussian_slots + 1u];
    uint depth = gauss_u[2u * gaussian_count + gaussian];
    enumerate_tiles(
        mean_depth.xy, conic, int(pc.u2), int(pc.u3), gaussian, exclusive,
        int(pc.u4), depth, 1u);
}
