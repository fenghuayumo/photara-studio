#include "ppisp.hpp"
#include "cuda_common.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace aetherscan::splat::detail {
namespace {

constexpr int k_max_params = 36;
constexpr float k_ln2 = 0.69314718056F;
constexpr float k_color_eps = 1e-3F;

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

__device__ void apply_color_vjp(
    const float3 rgb, const float H[9], const float dH[8][9],
    const float3 d_out, float3& d_rgb, float d_color[8]) {
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
    float dH_acc[9] = {
        d_rgi_x * rgi_in.x, d_rgi_x * rgi_in.y, d_rgi_x * rgi_in.z,
        d_rgi_y * rgi_in.x, d_rgi_y * rgi_in.y, d_rgi_y * rgi_in.z,
        d_rgi_z * rgi_in.x, d_rgi_z * rgi_in.y, d_rgi_z * rgi_in.z};
    const float d_r = H[0] * d_rgi_x + H[3] * d_rgi_y + H[6] * d_rgi_z;
    const float d_g = H[1] * d_rgi_x + H[4] * d_rgi_y + H[7] * d_rgi_z;
    d_int += H[2] * d_rgi_x + H[5] * d_rgi_y + H[8] * d_rgi_z;
    d_rgb.x += d_r + d_int;
    d_rgb.y += d_g + d_int;
    d_rgb.z += d_int;
    for (int i = 0; i < 8; ++i) {
        float sum = 0.F;
        for (int k = 0; k < 9; ++k) sum += dH_acc[k] * dH[i][k];
        d_color[i] += sum;
    }
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

__device__ void apply_crf_vjp(
    const float3 rgb, const float* p, const float3 d_out, float3& d_rgb,
    float* d_p) {
    const float eps = 1.0e-3F;
    const float3 y = apply_crf(rgb, p);
    (void)y;
    for (int c = 0; c < 3; ++c) {
        float3 plus = rgb;
        float3 minus = rgb;
        const float value = c == 0 ? rgb.x : c == 1 ? rgb.y : rgb.z;
        (c == 0 ? plus.x : c == 1 ? plus.y : plus.z) = value + eps;
        (c == 0 ? minus.x : c == 1 ? minus.y : minus.z) = value - eps;
        const float3 yp = apply_crf(plus, p);
        const float3 ym = apply_crf(minus, p);
        const float dyc = ((c == 0 ? yp.x : c == 1 ? yp.y : yp.z) -
                              (c == 0 ? ym.x : c == 1 ? ym.y : ym.z)) /
            (2.F * eps);
        (c == 0 ? d_rgb.x : c == 1 ? d_rgb.y : d_rgb.z) +=
            (c == 0 ? d_out.x : c == 1 ? d_out.y : d_out.z) * dyc;
    }
    for (int i = 0; i < 12; ++i) {
        float plus[12];
        float minus[12];
        for (int k = 0; k < 12; ++k) {
            plus[k] = p[k];
            minus[k] = p[k];
        }
        plus[i] += eps;
        minus[i] -= eps;
        const float3 yp = apply_crf(rgb, plus);
        const float3 ym = apply_crf(rgb, minus);
        d_p[i] += (d_out.x * (yp.x - ym.x) + d_out.y * (yp.y - ym.y) +
                      d_out.z * (yp.z - ym.z)) /
            (2.F * eps);
    }
}

__device__ int color_offset(const int num_params) {
    return num_params == 9 ? 1 : 16;
}

__device__ float3 apply_ppisp_pixel(
    float3 rgb, const float2 pix, const float2 center, const float2 size,
    const float* params, const int num_params, const bool clamp_output,
    const float H[9]) {
    const float gain = exp2f(params[0]);
    rgb = make_float3(rgb.x * gain, rgb.y * gain, rgb.z * gain);
    if (num_params >= 24) rgb = apply_vignetting(rgb, pix, center, size, params + 1);
    rgb = apply_color(rgb, H);
    if (num_params == 36) rgb = apply_crf(rgb, params + 24);
    if (clamp_output) {
        rgb.x = fminf(fmaxf(rgb.x, 0.F), 1.F);
        rgb.y = fminf(fmaxf(rgb.y, 0.F), 1.F);
        rgb.z = fminf(fmaxf(rgb.z, 0.F), 1.F);
    }
    return rgb;
}

__global__ void ppisp_forward_kernel(
    const float* color, float* corrected, const float* params,
    const int view, const int num_params, const int height, const int width,
    const float cx, const float cy, const bool clamp_output) {
    __shared__ float shared_params[k_max_params];
    __shared__ float H[9];
    if (threadIdx.x < num_params)
        shared_params[threadIdx.x] = params[view * num_params + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0)
        compute_homography(shared_params + color_offset(num_params), H);
    __syncthreads();
    const int pixel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int pixels = height * width;
    if (pixel >= pixels) return;
    const int x = pixel % width;
    const int y = pixel / width;
    const float3 rgb = make_float3(
        color[pixel], color[pixels + pixel], color[2 * pixels + pixel]);
    const float3 out = apply_ppisp_pixel(
        rgb, make_float2(static_cast<float>(x), static_cast<float>(y)),
        make_float2(cx, cy),
        make_float2(static_cast<float>(width), static_cast<float>(height)),
        shared_params, num_params, clamp_output, H);
    corrected[pixel] = out.x;
    corrected[pixels + pixel] = out.y;
    corrected[2 * pixels + pixel] = out.z;
}

__global__ void ppisp_backward_kernel(
    const float* color, const float* params, const float* output_grad,
    float* input_grad, float* param_grad, const int view, const int num_params,
    const int height, const int width, const float cx, const float cy,
    const bool clamp_output) {
    __shared__ float shared_params[k_max_params];
    __shared__ float H[9];
    __shared__ float dH[8][9];
    if (threadIdx.x < num_params)
        shared_params[threadIdx.x] = params[view * num_params + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0)
        homography_jacobian(shared_params + color_offset(num_params), H, dH);
    __syncthreads();
    const int pixel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int pixels = height * width;
    const bool inside = pixel < pixels;
    float local[k_max_params]{};
    float3 d_in = make_float3(0.F, 0.F, 0.F);
    if (inside) {
        const int x = pixel % width;
        const int y = pixel / width;
        float3 rgb = make_float3(
            color[pixel], color[pixels + pixel], color[2 * pixels + pixel]);
        float3 d_out = make_float3(
            output_grad[pixel], output_grad[pixels + pixel],
            output_grad[2 * pixels + pixel]);
        if (d_out.x != 0.F || d_out.y != 0.F || d_out.z != 0.F) {
            const float2 pix =
                make_float2(static_cast<float>(x), static_cast<float>(y));
            const float2 center = make_float2(cx, cy);
            const float2 size = make_float2(
                static_cast<float>(width), static_cast<float>(height));
            const float gain = exp2f(shared_params[0]);
            const float3 exposed =
                make_float3(rgb.x * gain, rgb.y * gain, rgb.z * gain);
            const float3 vig = num_params >= 24
                ? apply_vignetting(exposed, pix, center, size, shared_params + 1)
                : exposed;
            const float3 coloured = apply_color(vig, H);
            float3 pre_clamp = coloured;
            if (num_params == 36) pre_clamp = apply_crf(coloured, shared_params + 24);
            if (clamp_output) {
                if (pre_clamp.x <= 0.F || pre_clamp.x >= 1.F) d_out.x = 0.F;
                if (pre_clamp.y <= 0.F || pre_clamp.y >= 1.F) d_out.y = 0.F;
                if (pre_clamp.z <= 0.F || pre_clamp.z >= 1.F) d_out.z = 0.F;
            }
            float3 d_stage = make_float3(0.F, 0.F, 0.F);
            if (num_params == 36) {
                apply_crf_vjp(
                    coloured, shared_params + 24, d_out, d_stage, local + 24);
                d_out = d_stage;
                d_stage = make_float3(0.F, 0.F, 0.F);
            }
            apply_color_vjp(
                vig, H, dH, d_out, d_stage, local + color_offset(num_params));
            d_out = d_stage;
            d_stage = make_float3(0.F, 0.F, 0.F);
            if (num_params >= 24) {
                apply_vignetting_vjp(
                    exposed, pix, center, size, shared_params + 1, d_out,
                    d_stage, local + 1);
                d_out = d_stage;
            }
            d_in.x += d_out.x * gain;
            d_in.y += d_out.y * gain;
            d_in.z += d_out.z * gain;
            local[0] += (d_out.x * exposed.x + d_out.y * exposed.y +
                            d_out.z * exposed.z) *
                k_ln2;
        }
        input_grad[pixel] = d_in.x;
        input_grad[pixels + pixel] = d_in.y;
        input_grad[2 * pixels + pixel] = d_in.z;
    }
    for (int i = 0; i < num_params; ++i) {
        float value = isfinite(local[i]) ? local[i] : 0.F;
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffU, value, offset);
        if ((threadIdx.x & 31U) == 0U && value != 0.F)
            atomicAdd(param_grad + view * num_params + i, value);
    }
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
    ppisp_forward_kernel<<<
        (pixels + k_cuda_threads - 1) / k_cuda_threads, k_cuda_threads>>>(
        color.ptr<float>(), state.output.ptr<float>(),
        state.parameters.ptr<float>(), static_cast<int>(view), state.num_params,
        height, width, camera.cx, camera.cy, state.clamp_output);
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
    ppisp_backward_kernel<<<
        (pixels + k_cuda_threads - 1) / k_cuda_threads, k_cuda_threads>>>(
        color.ptr<float>(), state.parameters.ptr<float>(),
        output_gradient.ptr<float>(), state.input_grad.ptr<float>(),
        state.gradient.ptr<float>(), static_cast<int>(view), state.num_params,
        height, width, camera.cx, camera.cy, state.clamp_output);
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
