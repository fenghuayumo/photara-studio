// Per-invocation SH gradient and codec state; no device SH gradient allocation.
[[vk::binding(15, 0)]] RWByteAddressBuffer parameter;
[[vk::binding(16, 0)]] RWByteAddressBuffer first;
[[vk::binding(17, 0)]] RWByteAddressBuffer packed;
[[vk::binding(18, 0)]] RWByteAddressBuffer bounds;
struct AdamConstants {
    uint stride, active, parameter_offset, first_offset, packed_offset, bounds_offset;
    float dc_lr, rest_lr, beta1, beta2, correction1, correction2, epsilon, regularization;
};
static AdamConstants adam;
static float sh_gradient[48], row_log[48];
static uint row_half[48];
static float4 old_bounds;
static const float codec_epsilon = 1e-15f;
uint aligned4(uint byte_offset)
{
    return byte_offset & ~3u;
}

uint byte_shift(uint byte_offset)
{
    return (byte_offset & 3u) * 8u;
}

uint load_u8(RWByteAddressBuffer buf, uint byte_offset)
{
    const uint word = buf.Load(aligned4(byte_offset));
    return (word >> byte_shift(byte_offset)) & 0xFFu;
}

void store_u8(RWByteAddressBuffer buf, uint byte_offset, uint value)
{
    const uint shift = byte_shift(byte_offset);
    const uint mask = 0xFFu << shift;
    const uint aligned = aligned4(byte_offset);
    buf.InterlockedAnd(aligned, ~mask);
    buf.InterlockedOr(aligned, (value & 0xFFu) << shift);
}

uint load_u16(RWByteAddressBuffer buf, uint byte_offset)
{
    const uint word = buf.Load(aligned4(byte_offset));
    return (word >> byte_shift(byte_offset)) & 0xFFFFu;
}

void store_u16(RWByteAddressBuffer buf, uint byte_offset, uint value)
{
    const uint shift = byte_shift(byte_offset);
    const uint mask = 0xFFFFu << shift;
    const uint aligned = aligned4(byte_offset);
    buf.InterlockedAnd(aligned, ~mask);
    buf.InterlockedOr(aligned, (value & 0xFFFFu) << shift);
}

uint load_u32(RWByteAddressBuffer buf, uint byte_offset)
{
    return buf.Load(byte_offset);
}

void store_u32(RWByteAddressBuffer buf, uint byte_offset, uint value)
{
    buf.Store(byte_offset, value);
}

float load_f32(RWByteAddressBuffer buf, uint byte_offset)
{
    return asfloat(buf.Load(byte_offset));
}

void store_f32(RWByteAddressBuffer buf, uint byte_offset, float value)
{
    buf.Store(byte_offset, asuint(value));
}

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
    if (adam.correction1 <= 0.0f || adam.correction2 <= 0.0f) return 0.0f;
    return (m / adam.correction1) / (sqrt(max(v, 0.0f) / adam.correction2) + adam.epsilon);
}
void update_column(uint row, uint col) {
    uint index = row * adam.stride + col;
    float lo = col < 3u ? old_bounds.x : old_bounds.z;
    float hi = col < 3u ? old_bounds.y : old_bounds.w;
    float s = lo + (hi - lo) * (float(load_u8(packed, adam.packed_offset + index)) / 255.0f);
    float v = second_from_log(s);
    float u = f16tof32(load_u16(first, adam.first_offset + index * 2u));
    float previous_c1 = adam.beta1 > 0.0f ? max((adam.correction1 - (1.0f - adam.beta1)) / adam.beta1, 0.0f) : 0.0f;
    float previous_c2 = adam.beta2 > 0.0f ? max((adam.correction2 - (1.0f - adam.beta2)) / adam.beta2, 0.0f) : 0.0f;
    float m = previous_c1 > 0.0f
        ? u * ((previous_c2 > 0.0f ? sqrt(max(v, 0.0f) / previous_c2) : 0.0f) + adam.epsilon) * previous_c1
        : 0.0f;
    if (col >= adam.active) {
        if (!isfinite(m)) m = 0.0f;
        if (!isfinite(s)) { s = 0.0f; m = 0.0f; }
        u = normalized(m, second_from_log(s));
        if (!isfinite(u)) u = 0.0f;
    } else {
        float g = sh_gradient[col];
        float previous = load_f32(parameter, adam.parameter_offset + index * 4u);
        if (adam.regularization > 0.0f && col >= 3u && isfinite(previous))
            g += adam.regularization * previous;
        if (!isfinite(previous) || !isfinite(g)) {
            u = 0.0f; s = 0.0f;
            store_f32(parameter, adam.parameter_offset + index * 4u, isfinite(previous) ? previous : 0.0f);
        } else {
            m = adam.beta1 * m + (1.0f - adam.beta1) * g;
            v = adam.beta2 * v + (1.0f - adam.beta2) * g * g;
            if (!isfinite(m) || !isfinite(v)) { u = 0.0f; s = 0.0f; }
            else {
                u = normalized(m, v);
                float candidate = previous - (col < 3u ? adam.dc_lr : adam.rest_lr) * u;
                store_f32(parameter, adam.parameter_offset + index * 4u, isfinite(candidate) ? candidate : previous);
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


void update_sh_row(uint row) {
    adam.parameter_offset = adam.first_offset = adam.packed_offset = adam.bounds_offset = 0u;
    adam.stride = pc.u6 * 3u;
    adam.active = (pc.u5 + 1u) * (pc.u5 + 1u) * 3u;
    adam.dc_lr = pc.dc_lr; adam.rest_lr = pc.rest_lr;
    adam.beta1 = pc.beta1; adam.beta2 = pc.beta2;
    adam.correction1 = pc.correction1; adam.correction2 = pc.correction2;
    adam.epsilon = pc.epsilon; adam.regularization = pc.regularization;
    old_bounds = asfloat(bounds.Load4(row * 16u));
    if (!isfinite(old_bounds.x) || !isfinite(old_bounds.y)) old_bounds.xy = 0.0f;
    if (!isfinite(old_bounds.z) || !isfinite(old_bounds.w)) old_bounds.zw = 0.0f;
    float4 next_bounds = float4(1e30f, -1e30f, 1e30f, -1e30f);
    for (uint col = 0u; col < adam.stride; ++col) {
        update_column(row, col);
        if (col < 3u) { next_bounds.x = min(next_bounds.x, row_log[col]); next_bounds.y = max(next_bounds.y, row_log[col]); }
        else { next_bounds.z = min(next_bounds.z, row_log[col]); next_bounds.w = max(next_bounds.w, row_log[col]); }
    }
    if (adam.stride <= 3u) next_bounds.zw = 0.0f;
    bounds.Store4(row * 16u, asuint(next_bounds));
    uint base = row * adam.stride;
    // Whole-word writes for aligned rows; masked atomics preserve adjacent
    // rows for degree 0/2 strides, which share packed boundary words.
    for (uint col = 0u; col < adam.stride; ++col) {
        uint q = col < 3u ? encode(row_log[col], next_bounds.x, next_bounds.y)
                         : encode(row_log[col], next_bounds.z, next_bounds.w);
        if ((adam.stride & 3u) != 0u) store_u8(packed, base + col, q);
        else if ((col & 3u) == 0u) {
            uint word = q;
            for (uint j = 1u; j < 4u; ++j) {
                uint c = col + j;
                word |= (c < 3u ? encode(row_log[c], next_bounds.x, next_bounds.y)
                               : encode(row_log[c], next_bounds.z, next_bounds.w)) << (j * 8u);
            }
            packed.Store(base + col, word);
        }
        if ((adam.stride & 1u) != 0u) store_u16(first, (base + col) * 2u, row_half[col]);
        else if ((col & 1u) == 0u)
            first.Store((base + col) * 2u, row_half[col] | (row_half[col + 1u] << 16u));
    }
}
