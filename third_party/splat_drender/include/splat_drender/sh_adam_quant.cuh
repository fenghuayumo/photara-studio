// Hybrid Adam state for one SH row.
//
// Store the normalized Adam update
//   u = (m / correction1) / (sqrt(v / correction2) + adam_epsilon)
// in FP16. The second moment uses
//   log_s = log1p(sqrt(g2) / 1e-15)
// with endpoint-exact uint8 codes. Columns [0, 3) (DC RGB) and [3, stride)
// (non-DC) each own (log_s_min, log_s_max). The 1e-15 here is the codec
// constant. The Adam step still uses the caller's epsilon.
//
// One block per Gaussian, not a flat 256-wide block. Densify remaps whole
// rows, and 48 SH coefficients do not fill a 256-cell block. A warp owns
// one row: lane L holds columns L and L+32 (storage stride <= 64).
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace splat_drender {

inline constexpr float kShAdamCodecEps = 1e-15f;
inline constexpr float kShAdamQMax = 255.f;
inline constexpr float kShAdamInvQMax = 1.f / 255.f;

__device__ __forceinline__ float sh_adam_dequant(float code, float lo, float hi) {
    return lo + (hi - lo) * (code * kShAdamInvQMax);
}

__device__ __forceinline__ std::uint8_t sh_adam_enquant(
    float value, float lo, float hi) {
    const float range = fmaxf(hi - lo, kShAdamCodecEps);
    const float code = fminf(
        fmaxf(roundf(kShAdamQMax * (value - lo) / range), 0.f), kShAdamQMax);
    return static_cast<std::uint8_t>(code);
}

__device__ __forceinline__ float sh_adam_second_to_log(float g2) {
    const float sqrt_g2 = sqrtf(fmaxf(g2, 0.f));
    return log1pf(sqrt_g2 * (1.f / kShAdamCodecEps));
}

__device__ __forceinline__ float sh_adam_log_to_second(float log_s) {
    const float sqrt_g2 = kShAdamCodecEps * expm1f(log_s);
    return sqrt_g2 * sqrt_g2;
}

// Recover the previous raw first moment from the stored normalized Adam
// update. For c_t = 1 - beta^t, c_{t-1} follows directly from
// c_t = (1 - beta) + beta*c_{t-1}; no iteration counter is required.
__device__ __forceinline__ float sh_adam_previous_correction(
    float correction, float beta) {
    return beta > 0.f
        ? fmaxf((correction - (1.f - beta)) / beta, 0.f)
        : 0.f;
}

__device__ __forceinline__ float sh_adam_normalized_to_first(
    float u, float v, float previous_correction1,
    float previous_correction2, float adam_epsilon) {
    if (previous_correction1 <= 0.f) return 0.f;
    const float sqrt_v_hat = previous_correction2 > 0.f
        ? sqrtf(fmaxf(v, 0.f) / previous_correction2) : 0.f;
    return u * (sqrt_v_hat + adam_epsilon) * previous_correction1;
}

__device__ __forceinline__ float sh_adam_first_to_normalized(
    float m, float v, float correction1, float correction2,
    float adam_epsilon) {
    if (correction1 <= 0.f || correction2 <= 0.f) return 0.f;
    return (m / correction1) /
        (sqrtf(fmaxf(v, 0.f) / correction2) + adam_epsilon);
}

__device__ __forceinline__ float sh_adam_warp_min(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fminf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

__device__ __forceinline__ float sh_adam_warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

// Decode one coefficient, Adam-update it when it is inside the active prefix,
// and return the log-second primitive that must be re-encoded.
__device__ __forceinline__ void sh_adam_quant_column(
    int col, int active, float* parameter, const float* gradient,
    __half* first, const std::uint8_t* packed, float bound_s_min,
    float bound_s_max, float dc_lr, float rest_lr, float beta1, float beta2,
    float correction1, float correction2, float adam_epsilon,
    float regularization, float& log_s) {
    log_s = sh_adam_dequant(
        static_cast<float>(packed[col]), bound_s_min, bound_s_max);
    const float g2 = sh_adam_log_to_second(log_s);
    const float stored_first = __half2float(first[col]);
    const float previous_correction1 = sh_adam_previous_correction(
        correction1, beta1);
    const float previous_correction2 = sh_adam_previous_correction(
        correction2, beta2);
    float g1 = sh_adam_normalized_to_first(
        stored_first, g2, previous_correction1,
        previous_correction2, adam_epsilon);
    if (col >= active) {
        if (!isfinite(g1)) g1 = 0.f;
        if (!isfinite(log_s)) {
            log_s = 0.f;
            g1 = 0.f;
        }
        const float encoded_first = sh_adam_first_to_normalized(
            g1, sh_adam_log_to_second(log_s), correction1,
            correction2, adam_epsilon);
        first[col] = __float2half_rn(
            isfinite(encoded_first) ? encoded_first : 0.f);
        return;
    }
    float grad = gradient[col];
    const float previous = parameter[col];
    if (regularization > 0.f && col >= 3 && isfinite(previous))
        grad += regularization * previous;
    if (!isfinite(previous) || !isfinite(grad)) {
        first[col] = __float2half(0.f);
        log_s = 0.f;
        parameter[col] = isfinite(previous) ? previous : 0.f;
        return;
    }
    const float m = beta1 * g1 + (1.f - beta1) * grad;
    const float v = beta2 * g2 + (1.f - beta2) * grad * grad;
    if (!isfinite(m) || !isfinite(v)) {
        first[col] = __float2half(0.f);
        log_s = 0.f;
        return;
    }
    const float learning_rate = col >= 3 ? rest_lr : dc_lr;
    const float normalized = sh_adam_first_to_normalized(
        m, v, correction1, correction2, adam_epsilon);
    const float candidate = previous - learning_rate * normalized;
    parameter[col] = isfinite(candidate) ? candidate : previous;
    first[col] = __float2half_rn(normalized);
    log_s = sh_adam_second_to_log(v);
    if (!isfinite(log_s)) log_s = 0.f;
}

// Every lane of the warp must call this together. `gradient` holds at least
// `active` coefficients; columns [active, stride) keep their decoded moments
// and are re-encoded under the new bounds. `regularization` is added to
// non-DC active gradients before the Adam step (0 when the caller already
// wrote that term into `gradient`).
__device__ __forceinline__ void sh_adam_quant_warp_row(
    int stride, int active, float* __restrict__ parameter,
    const float* __restrict__ gradient, __half* __restrict__ first,
    std::uint8_t* __restrict__ packed, float* __restrict__ bounds,
    float dc_lr, float rest_lr, float beta1, float beta2, float correction1,
    float correction2, float adam_epsilon, float regularization) {
    const int lane = threadIdx.x & 31;
    float dc_s_min = 0.f, dc_s_max = 0.f;
    float rest_s_min = 0.f, rest_s_max = 0.f;
    if (lane == 0) {
        dc_s_min = bounds[0];
        dc_s_max = bounds[1];
        rest_s_min = bounds[2];
        rest_s_max = bounds[3];
        if (!isfinite(dc_s_min) || !isfinite(dc_s_max)) {
            dc_s_min = 0.f;
            dc_s_max = 0.f;
        }
        if (!isfinite(rest_s_min) || !isfinite(rest_s_max)) {
            rest_s_min = 0.f;
            rest_s_max = 0.f;
        }
    }
    dc_s_min = __shfl_sync(0xffffffffu, dc_s_min, 0);
    dc_s_max = __shfl_sync(0xffffffffu, dc_s_max, 0);
    rest_s_min = __shfl_sync(0xffffffffu, rest_s_min, 0);
    rest_s_max = __shfl_sync(0xffffffffu, rest_s_max, 0);

    const int col0 = lane;
    const int col1 = lane + 32;
    const bool own0 = col0 < stride;
    const bool own1 = col1 < stride;
    float s0 = 0.f, s1 = 0.f;
    if (own0) {
        const bool dc = col0 < 3;
        sh_adam_quant_column(
            col0, active, parameter, gradient, first, packed,
            dc ? dc_s_min : rest_s_min, dc ? dc_s_max : rest_s_max, dc_lr,
            rest_lr, beta1, beta2, correction1, correction2, adam_epsilon,
            regularization, s0);
    }
    if (own1) {
        const bool dc = col1 < 3;
        sh_adam_quant_column(
            col1, active, parameter, gradient, first, packed,
            dc ? dc_s_min : rest_s_min, dc ? dc_s_max : rest_s_max, dc_lr,
            rest_lr, beta1, beta2, correction1, correction2, adam_epsilon,
            regularization, s1);
    }

    const bool dc0 = own0 && col0 < 3;
    const bool dc1 = own1 && col1 < 3;
    float new_dc_s_min = dc0 ? s0 : (dc1 ? s1 : 1e30f);
    float new_dc_s_max = dc0 ? s0 : (dc1 ? s1 : -1e30f);
    if (dc0 && dc1) {
        new_dc_s_min = fminf(s0, s1);
        new_dc_s_max = fmaxf(s0, s1);
    }
    const bool rest0 = own0 && col0 >= 3;
    const bool rest1 = own1 && col1 >= 3;
    float new_rest_s_min = rest0 ? s0 : (rest1 ? s1 : 1e30f);
    float new_rest_s_max = rest0 ? s0 : (rest1 ? s1 : -1e30f);
    if (rest0 && rest1) {
        new_rest_s_min = fminf(s0, s1);
        new_rest_s_max = fmaxf(s0, s1);
    }
    new_dc_s_min = sh_adam_warp_min(new_dc_s_min);
    new_dc_s_max = sh_adam_warp_max(new_dc_s_max);
    new_rest_s_min = sh_adam_warp_min(new_rest_s_min);
    new_rest_s_max = sh_adam_warp_max(new_rest_s_max);
    if (stride <= 3) {
        new_rest_s_min = 0.f;
        new_rest_s_max = 0.f;
    }
    if (lane == 0) {
        bounds[0] = new_dc_s_min;
        bounds[1] = new_dc_s_max;
        bounds[2] = new_rest_s_min;
        bounds[3] = new_rest_s_max;
    }
    if (own0) {
        const bool dc = col0 < 3;
        packed[col0] = sh_adam_enquant(
            s0, dc ? new_dc_s_min : new_rest_s_min,
            dc ? new_dc_s_max : new_rest_s_max);
    }
    if (own1) {
        const bool dc = col1 < 3;
        packed[col1] = sh_adam_enquant(
            s1, dc ? new_dc_s_min : new_rest_s_min,
            dc ? new_dc_s_max : new_rest_s_max);
    }
}

}  // namespace splat_drender
