#include "common.hlsli"

// Same codec and row semantics as splat_drender/sh_adam_quant.cuh.
// A fixed 32-thread workgroup owns a row, independent of subgroup width.
struct PushConstants {
    uint rows, stride, active;
    uint parameter_offset, gradient_offset, first_offset, packed_offset, bounds_offset;
    float dc_lr, rest_lr, beta1, beta2, correction1, correction2, epsilon, regularization;
    uint groups_x;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer parameter;
[[vk::binding(1, 0)]] RWByteAddressBuffer gradient;
[[vk::binding(2, 0)]] RWByteAddressBuffer first;
[[vk::binding(3, 0)]] RWByteAddressBuffer packed;
[[vk::binding(4, 0)]] RWByteAddressBuffer bounds;

static const float codec_epsilon = 1e-15f;
groupshared float row_log[64];
groupshared uint row_half[64];
groupshared uint row_code[64];
groupshared float dc_min[32], dc_max[32], rest_min[32], rest_max[32];
groupshared float4 old_bounds;

float second_from_log(float s) {
    // HLSL has no expm1. Preserve small decoded values without cancellation.
    float e = abs(s) < 1e-4f ? s * (1.0f + s * (0.5f + s / 6.0f)) : exp(s) - 1.0f;
    float root = codec_epsilon * e;
    return root * root;
}
float log_from_second(float v) {
    float x = sqrt(max(v, 0.0f)) / codec_epsilon;
    return x < 1e-4f ? x * (1.0f - x * (0.5f - x / 3.0f)) : log(1.0f + x);
}
float normalized(float m, float v) {
    if (pc.correction1 <= 0.0f || pc.correction2 <= 0.0f) return 0.0f;
    return (m / pc.correction1) / (sqrt(max(v, 0.0f) / pc.correction2) + pc.epsilon);
}
void update_column(uint row, uint col) {
    uint index = row * pc.stride + col;
    float lo = col < 3u ? old_bounds.x : old_bounds.z;
    float hi = col < 3u ? old_bounds.y : old_bounds.w;
    float s = lo + (hi - lo) * (float(load_u8(packed, pc.packed_offset + index)) / 255.0f);
    float v = second_from_log(s);
    float u = f16tof32(load_u16(first, pc.first_offset + index * 2u));
    float previous_c1 = pc.beta1 > 0.0f ? max((pc.correction1 - (1.0f - pc.beta1)) / pc.beta1, 0.0f) : 0.0f;
    float previous_c2 = pc.beta2 > 0.0f ? max((pc.correction2 - (1.0f - pc.beta2)) / pc.beta2, 0.0f) : 0.0f;
    float m = previous_c1 > 0.0f
        ? u * ((previous_c2 > 0.0f ? sqrt(max(v, 0.0f) / previous_c2) : 0.0f) + pc.epsilon) * previous_c1
        : 0.0f;
    if (col >= pc.active) {
        if (!isfinite(m)) m = 0.0f;
        if (!isfinite(s)) { s = 0.0f; m = 0.0f; }
        u = normalized(m, second_from_log(s));
        if (!isfinite(u)) u = 0.0f;
    } else {
        float g = load_f32(gradient, pc.gradient_offset + index * 4u);
        float previous = load_f32(parameter, pc.parameter_offset + index * 4u);
        if (pc.regularization > 0.0f && col >= 3u && isfinite(previous))
            g += pc.regularization * previous;
        if (!isfinite(previous) || !isfinite(g)) {
            u = 0.0f; s = 0.0f;
            store_f32(parameter, pc.parameter_offset + index * 4u, isfinite(previous) ? previous : 0.0f);
        } else {
            m = pc.beta1 * m + (1.0f - pc.beta1) * g;
            v = pc.beta2 * v + (1.0f - pc.beta2) * g * g;
            if (!isfinite(m) || !isfinite(v)) { u = 0.0f; s = 0.0f; }
            else {
                u = normalized(m, v);
                float candidate = previous - (col < 3u ? pc.dc_lr : pc.rest_lr) * u;
                store_f32(parameter, pc.parameter_offset + index * 4u, isfinite(candidate) ? candidate : previous);
                s = log_from_second(v);
                if (!isfinite(s)) s = 0.0f;
            }
        }
    }
    row_log[col] = s;
    // f32tof16 uses round-to-nearest-even, matching CUDA __float2half_rn.
    row_half[col] = f32tof16(u);
}
uint encode(float s, float lo, float hi) {
    // CUDA roundf is halfway-away-from-zero; the code domain is nonnegative.
    float q = 255.0f * (s - lo) / max(hi - lo, codec_epsilon);
    return uint(clamp(floor(q + 0.5f), 0.0f, 255.0f));
}

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    uint row = gid.x + gid.y * pc.groups_x;
    if (row >= pc.rows) return;
    if (lane == 0u) {
        old_bounds = asfloat(bounds.Load4(pc.bounds_offset + row * 16u));
        if (!isfinite(old_bounds.x) || !isfinite(old_bounds.y)) old_bounds.xy = 0.0f;
        if (!isfinite(old_bounds.z) || !isfinite(old_bounds.w)) old_bounds.zw = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < pc.stride) update_column(row, lane);
    if (lane + 32u < pc.stride) update_column(row, lane + 32u);
    GroupMemoryBarrierWithGroupSync();
    dc_min[lane] = 1e30f; dc_max[lane] = -1e30f;
    rest_min[lane] = 1e30f; rest_max[lane] = -1e30f;
    for (uint col = lane; col < pc.stride; col += 32u) {
        float s = row_log[col];
        if (col < 3u) { dc_min[lane] = min(dc_min[lane], s); dc_max[lane] = max(dc_max[lane], s); }
        else { rest_min[lane] = min(rest_min[lane], s); rest_max[lane] = max(rest_max[lane], s); }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        if (lane < offset) {
            dc_min[lane] = min(dc_min[lane], dc_min[lane + offset]);
            dc_max[lane] = max(dc_max[lane], dc_max[lane + offset]);
            rest_min[lane] = min(rest_min[lane], rest_min[lane + offset]);
            rest_max[lane] = max(rest_max[lane], rest_max[lane + offset]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float4 next_bounds = float4(dc_min[0], dc_max[0],
        pc.stride <= 3u ? 0.0f : rest_min[0], pc.stride <= 3u ? 0.0f : rest_max[0]);
    if (lane == 0u) bounds.Store4(pc.bounds_offset + row * 16u, asuint(next_bounds));
    for (uint col = lane; col < pc.stride; col += 32u)
        row_code[col] = col < 3u ? encode(row_log[col], next_bounds.x, next_bounds.y)
                                : encode(row_log[col], next_bounds.z, next_bounds.w);
    GroupMemoryBarrierWithGroupSync();
    uint base = row * pc.stride;
    // Aligned rows (including degree 1/3) use exclusive full-word stores.
    // Odd strides use masked atomic stores, preserving adjacent rows' bits.
    if (((pc.stride * 2u | pc.first_offset) & 3u) == 0u) {
        if (lane < pc.stride / 2u)
            first.Store(pc.first_offset + base * 2u + lane * 4u,
                row_half[lane * 2u] | (row_half[lane * 2u + 1u] << 16u));
    } else {
        for (uint col = lane; col < pc.stride; col += 32u)
            store_u16(first, pc.first_offset + (base + col) * 2u, row_half[col]);
    }
    if (((pc.stride | pc.packed_offset) & 3u) == 0u) {
        if (lane < pc.stride / 4u)
            packed.Store(pc.packed_offset + base + lane * 4u,
                row_code[lane * 4u] | (row_code[lane * 4u + 1u] << 8u) |
                (row_code[lane * 4u + 2u] << 16u) | (row_code[lane * 4u + 3u] << 24u));
    } else {
        for (uint col = lane; col < pc.stride; col += 32u)
            store_u8(packed, pc.packed_offset + base + col, row_code[col]);
    }
}
