#include "ppisp.hpp"
#include "cuda_common.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace aetherscan::splat::detail {
namespace {

constexpr int k_max_params = 36;
// Entries of the colour homography, and of its pull-back vector.
constexpr int k_affine_pullback = 9;
constexpr float k_ln2 = 0.69314718056F;
// Step of the device-side central difference that builds the colour-homography
// Jacobian. At 1e-3 the float32 round-off of `compute_homography` dominated the
// estimate (measured ~3% against a host-side finite difference); 1e-2 moves the
// balance to the truncation side, where the residual is under 1%.
constexpr float k_color_eps = 1e-2F;

__device__ __forceinline__ void mat3_mul(
    const float a[9], const float b[9], float o[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            o[3 * i + j] = a[3 * i] * b[j] + a[3 * i + 1] * b[3 + j] +
                a[3 * i + 2] * b[6 + j];
}

__device__ __forceinline__ float3 mat3_mul_vec(const float m[9], const float3 v) {
    return make_float3(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z);
}

__device__ __forceinline__ float3 cross3(const float3 a, const float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ void compute_homography(const float* c, float H[9]) {
    const float2 bd = make_float2(
        0.0480542F * c[0] + -0.0043631F * c[1],
        -0.0043631F * c[0] + 0.0481283F * c[1]);
    const float2 rd = make_float2(
        0.0580570F * c[2] + -0.0179872F * c[3],
        -0.0179872F * c[2] + 0.0431061F * c[3]);
    const float2 gd = make_float2(
        0.0433336F * c[4] + -0.0180537F * c[5],
        -0.0180537F * c[4] + 0.0580500F * c[5]);
    const float2 nd = make_float2(
        0.0128369F * c[6] + -0.0034654F * c[7],
        -0.0034654F * c[6] + 0.0128158F * c[7]);
    float T[9] = {
        bd.x, 1.F + rd.x, gd.x, bd.y, rd.y, 1.F + gd.y, 1.F, 1.F, 1.F};
    const float gx = 1.F / 3.F + nd.x;
    const float gy = 1.F / 3.F + nd.y;
    float skew[9] = {0.F, -1.F, gy, 1.F, 0.F, -gx, -gy, gx, 0.F};
    float M[9];
    mat3_mul(skew, T, M);
    const float3 r0 = make_float3(M[0], M[1], M[2]);
    const float3 r1 = make_float3(M[3], M[4], M[5]);
    const float3 r2 = make_float3(M[6], M[7], M[8]);
    float3 lambda = cross3(r0, r1);
    if (lambda.x * lambda.x + lambda.y * lambda.y + lambda.z * lambda.z <
        1.0e-20F) {
        lambda = cross3(r0, r2);
        if (lambda.x * lambda.x + lambda.y * lambda.y + lambda.z * lambda.z <
            1.0e-20F)
            lambda = cross3(r1, r2);
    }
    float TD[9];
    for (int i = 0; i < 3; ++i) {
        TD[3 * i + 0] = T[3 * i + 0] * lambda.x;
        TD[3 * i + 1] = T[3 * i + 1] * lambda.y;
        TD[3 * i + 2] = T[3 * i + 2] * lambda.z;
    }
    const float Sinv[9] = {-1.F, -1.F, 1.F, 1.F, 0.F, 0.F, 0.F, 1.F, 0.F};
    mat3_mul(TD, Sinv, H);
    const float s = H[8];
    if (fabsf(s) > 1.0e-20F) {
        const float inv = 1.F / s;
        for (int i = 0; i < 9; ++i) H[i] *= inv;
    }
}

__device__ void homography_jacobian(
    const float* color, float H[9], float dH[8][9]) {
    compute_homography(color, H);
    float perturbed[8];
    for (int i = 0; i < 8; ++i) perturbed[i] = color[i];
    for (int i = 0; i < 8; ++i) {
        float plus[9];
        float minus[9];
        perturbed[i] = color[i] + k_color_eps;
        compute_homography(perturbed, plus);
        perturbed[i] = color[i] - k_color_eps;
        compute_homography(perturbed, minus);
        perturbed[i] = color[i];
        for (int k = 0; k < 9; ++k)
            dH[i][k] = (plus[k] - minus[k]) / (2.F * k_color_eps);
    }
}

__device__ float3 apply_color(const float3 rgb, const float H[9]) {
    const float intensity = rgb.x + rgb.y + rgb.z;
    const float3 rgi_in = make_float3(rgb.x, rgb.y, intensity);
    const float3 rgi_out = mat3_mul_vec(H, rgi_in);
    const float z =
        fmaxf(rgi_out.z, 1.0e-4F * fabsf(intensity) + 1.0e-8F);
    const float norm = intensity / z;
    const float out_r = rgi_out.x * norm;
    const float out_g = rgi_out.y * norm;
    return make_float3(out_r, out_g, intensity - out_r - out_g);
}

// VJP of the colour homography. The chain is unchanged; what used to be a
// per-pixel contraction with the eight-by-nine Jacobian now accumulates the
// nine-element pull-back, and the contraction happens once per view in
// ppisp_colour_grad_kernel. The Jacobian is the same for every pixel of a
// view, so contracting per pixel spent 72 fused multiplies per pixel on work
// that belongs to the view.
__device__ void apply_color_vjp_pullback(
    const float3 rgb, const float H[9], const float3 d_out, float3& d_rgb,
    float colour_pullback[k_affine_pullback]) {
    const float intensity = rgb.x + rgb.y + rgb.z;
    const float3 rgi_in = make_float3(rgb.x, rgb.y, intensity);
    const float3 rgi_out = mat3_mul_vec(H, rgi_in);
    const float zmin = 1.0e-4F * fabsf(intensity) + 1.0e-8F;
    const float z = fmaxf(rgi_out.z, zmin);
    const float norm = intensity / z;
    const float d_or = d_out.x - d_out.z;
    const float d_og = d_out.y - d_out.z;
    float d_int = d_out.z;
    const float d_rgi_x = d_or * norm;
    const float d_rgi_y = d_og * norm;
    const float d_norm = d_or * rgi_out.x + d_og * rgi_out.y;
    d_int += d_norm / z;
    const float d_z = -d_norm * intensity / (z * z);
    float d_rgi_z = 0.F;
    if (rgi_out.z > zmin) d_rgi_z += d_z;
    else d_int += d_z * (intensity >= 0.F ? 1.0e-4F : -1.0e-4F);
    const float accumulated[9] = {
        d_rgi_x * rgi_in.x, d_rgi_x * rgi_in.y, d_rgi_x * rgi_in.z,
        d_rgi_y * rgi_in.x, d_rgi_y * rgi_in.y, d_rgi_y * rgi_in.z,
        d_rgi_z * rgi_in.x, d_rgi_z * rgi_in.y, d_rgi_z * rgi_in.z};
#pragma unroll
    for (int k = 0; k < k_affine_pullback; ++k)
        colour_pullback[k] += accumulated[k];
    const float d_r = H[0] * d_rgi_x + H[3] * d_rgi_y + H[6] * d_rgi_z;
    const float d_g = H[1] * d_rgi_x + H[4] * d_rgi_y + H[7] * d_rgi_z;
    d_int += H[2] * d_rgi_x + H[5] * d_rgi_y + H[8] * d_rgi_z;
    d_rgb.x += d_r + d_int;
    d_rgb.y += d_g + d_int;
    d_rgb.z += d_int;
}

__device__ float3 apply_vignetting(
    const float3 rgb, const float2 pix, const float2 center, const float2 size,
    const float* p) {
    const float max_res = fmaxf(size.x, size.y);
    const float2 uv = make_float2(
        (pix.x - center.x) / max_res, (pix.y - center.y) / max_res);
    float out[3] = {rgb.x, rgb.y, rgb.z};
    for (int i = 0; i < 3; ++i) {
        const float dx = uv.x - p[5 * i + 0];
        const float dy = uv.y - p[5 * i + 1];
        const float r2 = dx * dx + dy * dy;
        const float r4 = r2 * r2;
        const float r6 = r4 * r2;
        float falloff =
            p[5 * i + 4] * r6 + p[5 * i + 3] * r4 + p[5 * i + 2] * r2 + 1.F;
        falloff = fminf(fmaxf(falloff, 0.F), 1.F);
        out[i] *= falloff;
    }
    return make_float3(out[0], out[1], out[2]);
}

__device__ void apply_vignetting_vjp(
    const float3 rgb, const float2 pix, const float2 center, const float2 size,
    const float* p, const float3 d_out, float3& d_rgb, float* d_p) {
    const float max_res = fmaxf(size.x, size.y);
    const float2 uv = make_float2(
        (pix.x - center.x) / max_res, (pix.y - center.y) / max_res);
    const float in[3] = {rgb.x, rgb.y, rgb.z};
    const float dout[3] = {d_out.x, d_out.y, d_out.z};
    float din[3] = {0.F, 0.F, 0.F};
    for (int i = 0; i < 3; ++i) {
        const float cx = p[5 * i + 0];
        const float cy = p[5 * i + 1];
        const float a0 = p[5 * i + 2];
        const float a1 = p[5 * i + 3];
        const float a2 = p[5 * i + 4];
        const float dx = uv.x - cx;
        const float dy = uv.y - cy;
        const float r2 = dx * dx + dy * dy;
        const float r4 = r2 * r2;
        const float r6 = r4 * r2;
        const float raw = a2 * r6 + a1 * r4 + a0 * r2 + 1.F;
        const float falloff = fminf(fmaxf(raw, 0.F), 1.F);
        din[i] += dout[i] * falloff;
        if (raw > 0.F && raw < 1.F) {
            const float d_raw = dout[i] * in[i];
            d_p[5 * i + 2] += d_raw * r2;
            d_p[5 * i + 3] += d_raw * r4;
            d_p[5 * i + 4] += d_raw * r6;
            const float d_r2 = d_raw * (a0 + 2.F * a1 * r2 + 3.F * a2 * r4);
            d_p[5 * i + 0] += d_r2 * (-2.F * dx);
            d_p[5 * i + 1] += d_r2 * (-2.F * dy);
        }
    }
    d_rgb.x += din[0];
    d_rgb.y += din[1];
    d_rgb.z += din[2];
}

__device__ float softplus(const float x) {
    return x > 20.F ? x : logf(1.F + expf(x));
}

__device__ float softplus_gradient(const float x) {
    return x > 20.F ? 1.F : 1.F / (1.F + expf(-x));
}

__device__ float crf_channel(
    float x, const float toe_raw, const float shoulder_raw,
    const float gamma_raw, const float center_raw) {
    x = fminf(fmaxf(x, 0.F), 1.F);
    const float toe = 0.3F + softplus(toe_raw);
    const float shoulder = 0.3F + softplus(shoulder_raw);
    const float gamma = 0.1F + softplus(gamma_raw);
    const float center = fminf(
        fmaxf(1.F / (1.F + expf(-center_raw)), 1.0e-4F), 1.F - 1.0e-4F);
    const float lerp_val = (1.F - center) * toe + center * shoulder;
    const float a = (shoulder * center) / fmaxf(lerp_val, 1.0e-8F);
    const float b = 1.F - a;
    float y;
    if (x <= center)
        y = a * powf(x / center, toe);
    else
        y = 1.F - b * powf((1.F - x) / (1.F - center), shoulder);
    return powf(fmaxf(y, 0.F), gamma);
}

__device__ float3 apply_crf(const float3 rgb, const float* p) {
    return make_float3(
        crf_channel(rgb.x, p[0], p[1], p[2], p[3]),
        crf_channel(rgb.y, p[4], p[5], p[6], p[7]),
        crf_channel(rgb.z, p[8], p[9], p[10], p[11]));
}

// Value and derivatives of one tone-curve channel. `crf_channel` is evaluated
// term by term so the analytic VJP reproduces it exactly; the finite-difference
// VJP this replaces needed fifteen full curve evaluations per pixel.
__device__ void crf_channel_grad(
    const float x_raw, const float toe_raw, const float shoulder_raw,
    const float gamma_raw, const float center_raw, float& value, float& dy_dx,
    float dy_dparameter[4]) {
    const float toe = 0.3F + softplus(toe_raw);
    const float shoulder = 0.3F + softplus(shoulder_raw);
    const float gamma = 0.1F + softplus(gamma_raw);
    const float center_unclamped = 1.F / (1.F + expf(-center_raw));
    const bool center_clamped =
        center_unclamped < 1.0e-4F || center_unclamped > 1.F - 1.0e-4F;
    const float center =
        fminf(fmaxf(center_unclamped, 1.0e-4F), 1.F - 1.0e-4F);
    const float lerp_value = (1.F - center) * toe + center * shoulder;
    const float denominator = fmaxf(lerp_value, 1.0e-8F);
    const float a = (shoulder * center) / denominator;
    const float b = 1.F - a;
    const float inverse_squared = 1.F / (denominator * denominator);
    const float da_dtoe =
        -shoulder * center * (1.F - center) * inverse_squared;
    const float da_dshoulder =
        center * (denominator - shoulder * center) * inverse_squared;
    const float da_dcenter = (shoulder * denominator -
        shoulder * center * (shoulder - toe)) * inverse_squared;

    const bool x_in_range = x_raw >= 0.F && x_raw <= 1.F;
    const float x = fminf(fmaxf(x_raw, 0.F), 1.F);
    float y = 0.F;
    float dy_dx_local = 0.F;
    float dy_dtoe = 0.F;
    float dy_dshoulder = 0.F;
    float dy_dcenter = 0.F;
    if (x <= center) {
        const float u = x / center;
        const float p = powf(u, toe);
        y = a * p;
        dy_dx_local = a * toe * powf(u, toe - 1.F) / center;
        dy_dtoe = da_dtoe * p + a * p * (u > 0.F ? logf(u) : 0.F);
        dy_dshoulder = da_dshoulder * p;
        dy_dcenter = da_dcenter * p - a * toe * p / center;
    } else {
        const float v = (1.F - x) / (1.F - center);
        const float q = powf(v, shoulder);
        y = 1.F - b * q;
        dy_dx_local = b * shoulder * powf(v, shoulder - 1.F) / (1.F - center);
        dy_dtoe = da_dtoe * q;
        dy_dshoulder = da_dshoulder * q - b * q * (v > 0.F ? logf(v) : 0.F);
        dy_dcenter =
            da_dcenter * q - b * q * shoulder / (1.F - center);
    }
    const float base = fmaxf(y, 0.F);
    value = powf(base, gamma);
    const bool positive = base > 0.F;
    const float dvalue_dy = positive ? gamma * value / base : 0.F;
    const float dvalue_dgamma = positive ? value * logf(base) : 0.F;
    dy_dx = x_in_range ? dvalue_dy * dy_dx_local : 0.F;
    dy_dparameter[0] = dvalue_dy * dy_dtoe * softplus_gradient(toe_raw);
    dy_dparameter[1] = dvalue_dy * dy_dshoulder * softplus_gradient(shoulder_raw);
    dy_dparameter[2] = dvalue_dgamma * softplus_gradient(gamma_raw);
    dy_dparameter[3] = dvalue_dy * dy_dcenter *
        (center_clamped ? 0.F : center * (1.F - center));
}

__device__ void apply_crf_vjp(
    const float3 rgb, const float* p, const float3 d_out, float3& d_rgb,
    float* d_p) {
    float value = 0.F;
    float dy_dx = 0.F;
    float dy_dparameter[4] = {0.F, 0.F, 0.F, 0.F};
    crf_channel_grad(
        rgb.x, p[0], p[1], p[2], p[3], value, dy_dx, dy_dparameter);
    d_rgb.x += d_out.x * dy_dx;
    for (int i = 0; i < 4; ++i) d_p[i] += d_out.x * dy_dparameter[i];
    crf_channel_grad(
        rgb.y, p[4], p[5], p[6], p[7], value, dy_dx, dy_dparameter);
    d_rgb.y += d_out.y * dy_dx;
    for (int i = 0; i < 4; ++i) d_p[4 + i] += d_out.y * dy_dparameter[i];
    crf_channel_grad(
        rgb.z, p[8], p[9], p[10], p[11], value, dy_dx, dy_dparameter);
    d_rgb.z += d_out.z * dy_dx;
    for (int i = 0; i < 4; ++i) d_p[8 + i] += d_out.z * dy_dparameter[i];
}

constexpr int color_offset(const int num_params) {
    return num_params == 9 ? 1 : 16;
}

// The parameter layout is a compile-time property: the per-pixel kernels are
// templated on it so the vignetting / tone-curve branches disappear and the
// per-parameter reduction only walks the parameters the layout owns. The
// runtime version executed the widest layout's loop for every layout.
template <int k_params>
constexpr int layout_color_offset() {
    return color_offset(k_params);
}

// Instantiate the per-pixel kernels once per layout and pick at the call site.
template <typename Launcher>
void dispatch_ppisp_layout(const int num_params, const Launcher& launcher) {
    switch (num_params) {
    case 9:
        launcher(std::integral_constant<int, 9>{});
        break;
    case 24:
        launcher(std::integral_constant<int, 24>{});
        break;
    case 36:
        launcher(std::integral_constant<int, 36>{});
        break;
    default:
        throw std::invalid_argument("unsupported PPISP parameter layout");
    }
}

template <int k_params>
__device__ float3 apply_ppisp_pixel(
    float3 rgb, const float2 pix, const float2 center, const float2 size,
    const float* params, const bool clamp_output, const float H[9]) {
    const float gain = exp2f(params[0]);
    rgb = make_float3(rgb.x * gain, rgb.y * gain, rgb.z * gain);
    if constexpr (k_params >= 24)
        rgb = apply_vignetting(rgb, pix, center, size, params + 1);
    rgb = apply_color(rgb, H);
    if constexpr (k_params == 36)
        rgb = apply_crf(rgb, params + 24);
    if (clamp_output) {
        rgb.x = fminf(fmaxf(rgb.x, 0.F), 1.F);
        rgb.y = fminf(fmaxf(rgb.y, 0.F), 1.F);
        rgb.z = fminf(fmaxf(rgb.z, 0.F), 1.F);
    }
    return rgb;
}

template <int k_params>
__global__ void ppisp_forward_kernel(
    const float* color, float* corrected, const float* params, const int view,
    const int height, const int width, const float cx, const float cy,
    const bool clamp_output) {
    __shared__ float shared_params[k_params];
    __shared__ float H[9];
    if (threadIdx.x < k_params)
        shared_params[threadIdx.x] = params[view * k_params + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0)
        compute_homography(shared_params + layout_color_offset<k_params>(), H);
    __syncthreads();
    const int pixel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int pixels = height * width;
    if (pixel >= pixels) return;
    const int x = pixel % width;
    const int y = pixel / width;
    const float3 rgb = make_float3(
        color[pixel], color[pixels + pixel], color[2 * pixels + pixel]);
    const float3 out = apply_ppisp_pixel<k_params>(
        rgb, make_float2(static_cast<float>(x), static_cast<float>(y)),
        make_float2(cx, cy),
        make_float2(static_cast<float>(width), static_cast<float>(height)),
        shared_params, clamp_output, H);
    corrected[pixel] = out.x;
    corrected[pixels + pixel] = out.y;
    corrected[2 * pixels + pixel] = out.z;
}

// Per-pixel pull-back of the PPISP chain. Colour and output gradient arrive per
// pixel; parameter contributions accumulate into `local`, while the colour
// homography accumulates the nine-element pull-back vector that the view-level
// contraction turns into eight parameter gradients afterwards.
template <int k_params>
__device__ __forceinline__ void ppisp_pixel_vjp(
    const float3 rgb, const float3 d_out_in, const float2 pix,
    const float2 center, const float2 size, const float* shared_params,
    const float H[9], const bool clamp_output, float* local,
    float* colour_pullback, float3& d_in) {
    if (d_out_in.x == 0.F && d_out_in.y == 0.F && d_out_in.z == 0.F) return;
    float3 d_out = d_out_in;
    const float gain = exp2f(shared_params[0]);
    const float3 exposed =
        make_float3(rgb.x * gain, rgb.y * gain, rgb.z * gain);
    const float3 vig = k_params >= 24
        ? apply_vignetting(exposed, pix, center, size, shared_params + 1)
        : exposed;
    const float3 coloured = apply_color(vig, H);
    float3 pre_clamp = coloured;
    if constexpr (k_params == 36)
        pre_clamp = apply_crf(coloured, shared_params + 24);
    if (clamp_output) {
        if (pre_clamp.x <= 0.F || pre_clamp.x >= 1.F) d_out.x = 0.F;
        if (pre_clamp.y <= 0.F || pre_clamp.y >= 1.F) d_out.y = 0.F;
        if (pre_clamp.z <= 0.F || pre_clamp.z >= 1.F) d_out.z = 0.F;
    }
    float3 d_stage = make_float3(0.F, 0.F, 0.F);
    if constexpr (k_params == 36) {
        apply_crf_vjp(
            coloured, shared_params + 24, d_out, d_stage, local + 24);
        d_out = d_stage;
        d_stage = make_float3(0.F, 0.F, 0.F);
    }
    apply_color_vjp_pullback(vig, H, d_out, d_stage, colour_pullback);
    d_out = d_stage;
    d_stage = make_float3(0.F, 0.F, 0.F);
    if constexpr (k_params >= 24) {
        apply_vignetting_vjp(
            exposed, pix, center, size, shared_params + 1, d_out, d_stage,
            local + 1);
        d_out = d_stage;
    }
    d_in.x += d_out.x * gain;
    d_in.y += d_out.y * gain;
    d_in.z += d_out.z * gain;
    local[0] += (d_out.x * exposed.x + d_out.y * exposed.y +
                    d_out.z * exposed.z) *
        k_ln2;
}

// Backward pass of PPISP. One thread owns a run of `k_pixels` consecutive
// pixels loaded as float4, so a warp keeps four times as many bytes in flight:
// the per-pixel version spent 56% of its warp cycles on long-scoreboard stalls
// (global loads) and 26% on short-scoreboard ones (shared and shuffle
// results). The run's parameter contributions are summed in registers before
// the warp reduction, which cuts that reduction by the run length as well.
template <int k_params>
__global__ void ppisp_backward_kernel(
    const float* __restrict__ color, const float* __restrict__ params,
    const float* __restrict__ output_grad, float* __restrict__ input_grad,
    float* __restrict__ param_grad, float* __restrict__ colour_pullback_view,
    const int view, const int height, const int width, const float cx,
    const float cy, const bool clamp_output) {
    constexpr int k_pixels = 4;
    constexpr int k_color = layout_color_offset<k_params>();
    __shared__ float shared_params[k_params];
    __shared__ float H[9];
    if (threadIdx.x < k_params)
        shared_params[threadIdx.x] = params[view * k_params + threadIdx.x];
    __syncthreads();
    // The eight-by-nine Jacobian is gone from this kernel: the pull-back it
    // used to contract is now summed per view.
    if (threadIdx.x == 0)
        compute_homography(shared_params + k_color, H);
    __syncthreads();

    const int pixels = height * width;
    const int first =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) * k_pixels;
    float local[k_params]{};
    float pullback[k_affine_pullback]{};
    if (first < pixels) {
        float3 rgb[k_pixels];
        float3 d_out[k_pixels];
        float3 d_in[k_pixels];
        const bool vector_ok = (pixels % k_pixels) == 0;
#pragma unroll
        for (int slot = 0; slot < k_pixels; ++slot) {
            rgb[slot] = make_float3(0.F, 0.F, 0.F);
            d_out[slot] = make_float3(0.F, 0.F, 0.F);
            d_in[slot] = make_float3(0.F, 0.F, 0.F);
        }
        if (vector_ok) {
            float4 colour[3];
            float4 gradient[3];
#pragma unroll
            for (int channel = 0; channel < 3; ++channel) {
                colour[channel] = *reinterpret_cast<const float4*>(
                    color + channel * pixels + first);
                gradient[channel] = *reinterpret_cast<const float4*>(
                    output_grad + channel * pixels + first);
            }
#pragma unroll
            for (int slot = 0; slot < k_pixels; ++slot) {
                const float* colour_slot =
                    reinterpret_cast<const float*>(colour);
                const float* gradient_slot =
                    reinterpret_cast<const float*>(gradient);
                rgb[slot] = make_float3(
                    colour_slot[slot], colour_slot[k_pixels + slot],
                    colour_slot[2 * k_pixels + slot]);
                d_out[slot] = make_float3(
                    gradient_slot[slot], gradient_slot[k_pixels + slot],
                    gradient_slot[2 * k_pixels + slot]);
            }
        } else {
#pragma unroll
            for (int slot = 0; slot < k_pixels; ++slot) {
                const int pixel = first + slot;
                if (pixel >= pixels) break;
                rgb[slot] = make_float3(
                    color[pixel], color[pixels + pixel],
                    color[2 * pixels + pixel]);
                d_out[slot] = make_float3(
                    output_grad[pixel], output_grad[pixels + pixel],
                    output_grad[2 * pixels + pixel]);
            }
        }
        const float2 center = make_float2(cx, cy);
        const float2 size =
            make_float2(static_cast<float>(width), static_cast<float>(height));
#pragma unroll
        for (int slot = 0; slot < k_pixels; ++slot) {
            const int pixel = first + slot;
            if (pixel >= pixels) break;
            ppisp_pixel_vjp<k_params>(
                rgb[slot], d_out[slot],
                make_float2(
                    static_cast<float>(pixel % width),
                    static_cast<float>(pixel / width)),
                center, size, shared_params, H, clamp_output, local, pullback,
                d_in[slot]);
        }
#pragma unroll
        for (int slot = 0; slot < k_pixels; ++slot) {
            const int pixel = first + slot;
            if (pixel >= pixels) break;
            input_grad[pixel] = d_in[slot].x;
            input_grad[pixels + pixel] = d_in[slot].y;
            input_grad[2 * pixels + pixel] = d_in[slot].z;
        }
    }

    // The colour pull-back is shared by the whole view, so it lands in its own
    // buffer; the eight colour gradients come out of a single contraction per
    // view in ppisp_colour_grad_kernel.
#pragma unroll
    for (int i = 0; i < k_params + k_affine_pullback; ++i) {
        float value = i < k_params ? local[i] : pullback[i - k_params];
        value = isfinite(value) ? value : 0.F;
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffU, value, offset);
        if ((threadIdx.x & 31U) != 0U || value == 0.F) continue;
        if (i < k_params)
            atomicAdd(param_grad + view * k_params + i, value);
        else
            atomicAdd(colour_pullback_view + i - k_params, value);
    }
}

// Contraction of one view's colour pull-back into its eight parameter
// gradients: d_p[i] = sum_k dH[i][k] * pullback[k]. It runs over the whole
// view table but only the view that just trained has a non-zero pull-back.
template <int k_params>
__global__ void ppisp_colour_grad_kernel(
    const float* __restrict__ params, const float* __restrict__ pullback,
    float* __restrict__ param_grad, const int views) {
    constexpr int k_color = layout_color_offset<k_params>();
    const int view = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (view >= views) return;
    const float* row = pullback + view * k_affine_pullback;
    float magnitude = 0.F;
#pragma unroll
    for (int k = 0; k < k_affine_pullback; ++k)
        magnitude += fabsf(row[k]);
    if (magnitude == 0.F) return;
    float H[9];
    float dH[8][9];
    homography_jacobian(params + view * k_params + k_color, H, dH);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float value = 0.F;
#pragma unroll
        for (int k = 0; k < 9; ++k) value += dH[i][k] * row[k];
        param_grad[view * k_params + k_color + i] = value;
    }
    (void)H;
}

__device__ void zca_color(const float* latent, float zca[8]) {
    zca[0] = 0.0480542F * latent[0] + -0.0043631F * latent[1];
    zca[1] = -0.0043631F * latent[0] + 0.0481283F * latent[1];
    zca[2] = 0.0580570F * latent[2] + -0.0179872F * latent[3];
    zca[3] = -0.0179872F * latent[2] + 0.0431061F * latent[3];
    zca[4] = 0.0433336F * latent[4] + -0.0180537F * latent[5];
    zca[5] = -0.0180537F * latent[4] + 0.0580500F * latent[5];
    zca[6] = 0.0128369F * latent[6] + -0.0034654F * latent[7];
    zca[7] = -0.0034654F * latent[6] + 0.0128158F * latent[7];
}

__global__ void ppisp_reg_sum_kernel(
    const float* params, float* sums, const int views, const int num_params) {
    const int view = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (view >= views) return;
    const float* row = params + view * num_params;
    atomicAdd(sums + 0, row[0]);
    float zca[8];
    zca_color(row + color_offset(num_params), zca);
    for (int i = 0; i < 8; ++i) atomicAdd(sums + 1 + i, zca[i]);
}

__device__ float smooth_l1_grad(const float x, const float beta) {
    if (fabsf(x) < beta) return x / beta;
    return x > 0.F ? 1.F : -1.F;
}

__global__ void ppisp_reg_grad_kernel(
    const float* params, float* grads, const float* sums, const int views,
    const int num_params, const float exposure_weight, const float color_weight,
    const float vig_center_weight, const float vig_non_pos_weight,
    const float vig_var_weight, const float crf_var_weight) {
    const int view = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (view >= views) return;
    const float inv_n = 1.F / static_cast<float>(views);
    const float* row = params + view * num_params;
    float* grad = grads + view * num_params;
    const float mean_e = sums[0] * inv_n;
    grad[0] += exposure_weight * smooth_l1_grad(mean_e, 0.1F);
    const int color = color_offset(num_params);
    float d_zca[8]{};
    for (int i = 0; i < 8; ++i)
        d_zca[i] = color_weight / 8.F *
            smooth_l1_grad(sums[1 + i] * inv_n, 0.005F);
    float d_lat[8];
    zca_color(d_zca, d_lat);
    for (int i = 0; i < 8; ++i) grad[color + i] += d_lat[i];
    if (num_params < 24) return;
    const float center_scale = vig_center_weight / (3.F * views);
    const float relu_scale = vig_non_pos_weight / (9.F * views);
    const float var_scale = vig_var_weight / (5.F * views);
    for (int axis = 0; axis < 2; ++axis) {
        float values[3];
        for (int c = 0; c < 3; ++c) values[c] = row[1 + 5 * c + axis];
        const float mean = (values[0] + values[1] + values[2]) / 3.F;
        for (int c = 0; c < 3; ++c) {
            const float v = values[c];
            grad[1 + 5 * c + axis] += 2.F * v * center_scale +
                (2.F / 3.F) * (v - mean) * var_scale;
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        float values[3];
        for (int c = 0; c < 3; ++c) values[c] = row[1 + 5 * c + 2 + axis];
        const float mean = (values[0] + values[1] + values[2]) / 3.F;
        for (int c = 0; c < 3; ++c) {
            const float v = values[c];
            if (v > 0.F) grad[1 + 5 * c + 2 + axis] += relu_scale;
            grad[1 + 5 * c + 2 + axis] += (2.F / 3.F) * (v - mean) * var_scale;
        }
    }
    if (num_params < 36) return;
    const float crf_scale = crf_var_weight / (4.F * views);
    for (int axis = 0; axis < 4; ++axis) {
        float values[3];
        for (int c = 0; c < 3; ++c) values[c] = row[24 + 4 * c + axis];
        const float mean = (values[0] + values[1] + values[2]) / 3.F;
        for (int c = 0; c < 3; ++c)
            grad[24 + 4 * c + axis] +=
                (2.F / 3.F) * (values[c] - mean) * crf_scale;
    }
}

__global__ void ppisp_original_init_kernel(float* params, const int views) {
    const int view = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (view >= views) return;
    constexpr float a = 0.013658988289535046F;
    constexpr float b = 0.37816452980041504F;
    float* row = params + view * 36;
    for (int i = 0; i < 24; ++i) row[i] = 0.F;
    for (int c = 0; c < 3; ++c) {
        row[24 + 4 * c + 0] = a;
        row[24 + 4 * c + 1] = a;
        row[24 + 4 * c + 2] = b;
        row[24 + 4 * c + 3] = 0.F;
    }
}

}  // namespace

PpispState make_ppisp_state(
    const std::size_t views, const TrainingOptions& options) {
    PpispState state;
    if (views == 0) return state;
    state.type = options.ppisp_type;
    state.num_params = ppisp_parameter_count(options.ppisp_type);
    state.clamp_output = options.ppisp_clamp_output;
    const std::size_t params = static_cast<std::size_t>(state.num_params);
    if (state.type == PpispParamType::original) {
        state.parameters = tinytensor::Tensor::empty(
            {views, params}, tinytensor::Device::CUDA);
        ppisp_original_init_kernel<<<
            static_cast<unsigned>(
                (views + k_cuda_threads - 1) / k_cuda_threads),
            k_cuda_threads>>>(
            state.parameters.ptr<float>(), static_cast<int>(views));
        check_cuda(cudaGetLastError(), "initialize original PPISP parameters");
    } else {
        state.parameters = tinytensor::Tensor::zeros(
            {views, params}, tinytensor::Device::CUDA);
    }
    state.gradient = tinytensor::Tensor::zeros_like(state.parameters);
    state.adam = make_adam_state(state.parameters);
    state.raw_sums = tinytensor::Tensor::zeros(
        {std::size_t{9}}, tinytensor::Device::CUDA);
    state.colour_pullback = tinytensor::Tensor::zeros(
        {views, std::size_t{k_affine_pullback}}, tinytensor::Device::CUDA);
    return state;
}

void apply_ppisp(
    const tinytensor::Tensor& color, PpispState& state, const Camera& camera,
    const std::size_t view) {
    if (color.numel() == 0) return;
    if (!state.is_valid())
        throw std::invalid_argument("PPISP requires an initialized state");
    if (view >= state.parameters.shape()[0])
        throw std::invalid_argument("PPISP view index is out of range");
    if (color.shape().rank() != 3 || color.shape()[0] != 3)
        throw std::invalid_argument("PPISP expects planar [3,H,W]");
    ensure_same_shape(state.output, color);
    const int height = static_cast<int>(color.shape()[1]);
    const int width = static_cast<int>(color.shape()[2]);
    const int pixels = height * width;
    if (pixels == 0) return;
    const unsigned blocks =
        (pixels + k_cuda_threads - 1) / k_cuda_threads;
    const auto launch = [&](auto layout) {
        ppisp_forward_kernel<decltype(layout)::value>
            <<<blocks, k_cuda_threads>>>(
                color.ptr<float>(), state.output.ptr<float>(),
                state.parameters.ptr<float>(), static_cast<int>(view), height,
                width, camera.cx, camera.cy, state.clamp_output);
    };
    dispatch_ppisp_layout(state.num_params, launch);
    check_cuda(cudaGetLastError(), "apply PPISP colour correction");
}

void backward_ppisp(
    PpispState& state, const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient, const Camera& camera,
    const std::size_t view) {
    if (color.numel() == 0) return;
    ensure_same_shape(state.input_grad, color);
    const int height = static_cast<int>(color.shape()[1]);
    const int width = static_cast<int>(color.shape()[2]);
    const int pixels = height * width;
    if (pixels == 0) return;
    state.gradient.zero_();
    state.colour_pullback.zero_();
    const unsigned threads = k_cuda_threads;
    const unsigned blocks =
        (static_cast<unsigned>(pixels) + 4 * threads - 1) / (4 * threads);
    const auto launch = [&](auto layout) {
        constexpr int k_params = decltype(layout)::value;
        ppisp_backward_kernel<k_params><<<blocks, threads>>>(
            color.ptr<float>(), state.parameters.ptr<float>(),
            output_gradient.ptr<float>(), state.input_grad.ptr<float>(),
            state.gradient.ptr<float>(), state.colour_pullback.ptr<float>(),
            static_cast<int>(view), height, width, camera.cx, camera.cy,
            state.clamp_output);
        const unsigned views =
            static_cast<unsigned>(state.parameters.shape()[0]);
        ppisp_colour_grad_kernel<k_params><<<
            (views + threads - 1) / threads, threads>>>(
            state.parameters.ptr<float>(), state.colour_pullback.ptr<float>(),
            state.gradient.ptr<float>(), static_cast<int>(views));
    };
    dispatch_ppisp_layout(state.num_params, launch);
    check_cuda(cudaGetLastError(), "backward PPISP colour correction");
}

void step_ppisp(
    PpispState& state, const TrainingOptions& options,
    const unsigned iteration) {
    if (!state.is_valid()) return;
    const int views = static_cast<int>(state.parameters.shape()[0]);
    if (views <= 0) return;
    state.raw_sums.zero_();
    ppisp_reg_sum_kernel<<<
        (views + k_cuda_threads - 1) / k_cuda_threads, k_cuda_threads>>>(
        state.parameters.ptr<float>(), state.raw_sums.ptr<float>(), views,
        state.num_params);
    check_cuda(cudaGetLastError(), "reduce PPISP regularization");
    ppisp_reg_grad_kernel<<<
        (views + k_cuda_threads - 1) / k_cuda_threads, k_cuda_threads>>>(
        state.parameters.ptr<float>(), state.gradient.ptr<float>(),
        state.raw_sums.ptr<float>(), views, state.num_params,
        options.ppisp_reg_exposure_mean, options.ppisp_reg_color_mean,
        options.ppisp_reg_vig_center, options.ppisp_reg_vig_non_pos,
        options.ppisp_reg_vig_channel_var, options.ppisp_reg_crf_channel_var);
    check_cuda(cudaGetLastError(), "PPISP regularization gradients");
    adam_step(
        state.parameters, state.gradient, state.adam, options.ppisp_lr,
        iteration, options);
}

std::array<float, 2> ppisp_identity_deviation(const PpispState& state) {
    std::array<float, 2> result{0.F, 0.F};
    if (!state.is_valid() || state.num_params <= 0) return result;
    const std::vector<float> values = state.parameters.to_vector();
    const std::size_t views =
        values.size() / static_cast<std::size_t>(state.num_params);
    if (views == 0) return result;
    double total = 0.0;
    std::size_t count = 0;
    for (std::size_t view = 0; view < views; ++view) {
        const float* row = values.data() + view * state.num_params;
        const float deviation = std::abs(std::exp2(row[0]) - 1.F);
        total += deviation;
        ++count;
        result[1] = std::max(result[1], deviation);
    }
    if (count != 0) result[0] = static_cast<float>(total / count);
    return result;
}

}  // namespace aetherscan::splat::detail
