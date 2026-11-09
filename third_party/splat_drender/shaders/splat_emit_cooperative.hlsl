#include "splat_math.hlsli"
[[vk::binding(0, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(1, 0)]] StructuredBuffer<uint> gauss_u;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> key_lo;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> key_hi;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> values;
[[vk::binding(5, 0)]] StructuredBuffer<uint> sorted_ids;
[[vk::binding(6, 0)]] StructuredBuffer<uint> compact;
[[vk::binding(7, 0)]] StructuredBuffer<uint> control;

// A 32-lane team preserves each Gaussian's existing tile enumeration order,
// but writes each row in parallel. No atomics or subgroup-size assumption.
static uint emit_lane;
#define SPLAT_EMIT_COOPERATIVE 1
#include "splat_enum.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint index = id.x / 32u;
    if (control[2] != 0u) return;
    emit_lane = id.x & 31u;
    // The dispatch is capped by maxComputeWorkGroupCount[0]. Grid-striding
    // preserves coverage on devices with a smaller dispatch limit.
    uint stride = control[19] * 8u;
    [loop] for (; index < control[1]; index += stride) {
        uint gaussian = sorted_ids[index];
        uint exclusive = index == 0u ? 0u : compact[index - 1u];
        float4 mean_depth = gauss_f[gaussian * pc.u6];
        float4 conic = gauss_f[gaussian * pc.u6 + 1u];
        uint depth = gauss_u[2u * pc.u5 + gaussian];
        enumerate_tiles(mean_depth.xy, conic, int(pc.u2), int(pc.u3), gaussian,
                        exclusive, 0, depth, 1u);
    }
}
