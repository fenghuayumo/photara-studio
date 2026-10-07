#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] StructuredBuffer<float> out_f;
[[vk::binding(4, 0)]] StructuredBuffer<uint> out_u;
#if defined(SPLAT_PRUNING_ATOMIC)
[[vk::binding(5, 0)]] RWStructuredBuffer<float> scores;
[[vk::ext_extension("SPV_EXT_shader_atomic_float_add")]]
[[vk::ext_capability(6033)]]
[[vk::ext_instruction(6035)]]
float op_atomic_f_add([[vk::ext_reference]] float destination,
    uint scope, uint semantics, float value);
#else
// Portable Vulkan 1.2 accumulation, including devices without float atomics.
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> scores;
#endif

groupshared uint sid[256];
groupshared float2 smean[256];
groupshared float4 sconic[256];
groupshared float scolour[256];
#if !defined(SPLAT_PRUNING_SUBGROUP)
groupshared float reduction[256];
#endif

void add_score(uint gaussian, float value) {
    if (value == 0.0f) return;
#if defined(SPLAT_PRUNING_ATOMIC)
    op_atomic_f_add(scores[gaussian], 1u, 0u, value);
#else
    uint expected = scores[gaussian];
    [loop] for (;;) {
        uint observed;
        InterlockedCompareExchange(scores[gaussian], expected,
            asuint(asfloat(expected) + value), observed);
        if (observed == expected) return;
        expected = observed;
    }
#endif
}

[numthreads(16, 16, 1)]
void main(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID, uint rank : SV_GroupIndex) {
    uint width = pc.u0, height = pc.u1, tiles_x = pc.u2;
    uint mode = pc.u3, count = pc.u5, pixels = pc.u6, slots = pc.u7;
    int wrap_width = int(pc.u4);
    uint x = group_id.x * 16u + thread_id.x;
    uint y = group_id.y * 16u + thread_id.y;
    bool inside = x < width && y < height;
    uint pixel = y * width + x;
    uint tile = group_id.y * tiles_x + group_id.x;
    uint begin = ranges[2u * tile], end = ranges[2u * tile + 1u];
    uint last = inside ? out_u[count + pixel] : 0u;
    uint limit = min(end - begin, out_u[count + pixels + tile]);
    // The rendered image includes the final background. Removing the prefix
    // leaves the weighted colour behind the current accepted contributor.
    float residual = inside ? out_f[pixel] + out_f[pixels + pixel] + out_f[2u * pixels + pixel] : 0.0f;
    float transmittance = 1.0f;
    [loop] for (uint base = 0u; base < limit; base += 256u) {
        // Stage each primitive once per tile, coalescing the draw-list reads.
        GroupMemoryBarrierWithGroupSync();
        if (base + rank < limit) {
            uint g = instances[begin + base + rank];
            sid[rank] = g;
            smean[rank] = gauss_f[g * slots].xy;
            sconic[rank] = gauss_f[g * slots + 1u];
            float3 rgb = gauss_f[g * slots + 2u].xyz;
            scolour[rank] = rgb.x + rgb.y + rgb.z;
        }
        GroupMemoryBarrierWithGroupSync();
        [loop] for (uint j = 0u; j < min(256u, limit - base); ++j) {
            float value = 0.0f;
            if (base + j < last) {
                float2 mean = smean[j];
                float2 delta = float2(wrap_dx(mean.x - float(x), wrap_width, mode), mean.y - float(y));
                float4 conic = sconic[j];
                float power = gaussian_power(conic, delta.x, delta.y);
                if (power <= 0.0f) {
                    float alpha = min(kAlphaClip, conic.w * exp(power));
                    if (alpha >= kAlphaFloor) {
                        float colour = scolour[j];
                        residual -= transmittance * alpha * colour;
                        float derivative = conic.w * (transmittance * colour - residual / (1.0f - alpha));
                        value = derivative * derivative;
                        transmittance *= 1.0f - alpha;
                    }
                }
            }
#if defined(SPLAT_PRUNING_SUBGROUP)
            // Identical per-pixel recurrence to CUDA, but only one global
            // atomic per subgroup/Gaussian instead of one per pixel.
            float total = WaveActiveSum(value);
            if (WaveIsFirstLane()) add_score(sid[j], total);
#else
            reduction[rank] = value;
            GroupMemoryBarrierWithGroupSync();
            [unroll] for (uint stride = 128u; stride > 0u; stride >>= 1u) {
                if (rank < stride) reduction[rank] += reduction[rank + stride];
                GroupMemoryBarrierWithGroupSync();
            }
            if (rank == 0u) add_score(sid[j], reduction[0]);
            GroupMemoryBarrierWithGroupSync();
#endif
        }
    }
}
