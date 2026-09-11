// Real spherical harmonics color evaluation, degrees 0..3.
#pragma once

#include "matrix.cuh"

namespace splat_drender::sh {

SD_D2 constexpr float kC0 = 0.28209479177387814f;
SD_D2 constexpr float kC1 = 0.4886025119029199f;
SD_D2 constexpr float kC2[5] = {
    1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
    -1.0925484305920792f, 0.5462742152960396f};
SD_D2 constexpr float kC3[7] = {
    -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
    0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
    -0.5900435899266435f};

// Evaluates the view-dependent RGB color of Gaussian `idx`.
// `clamped` receives per-channel flags for the max(result, 0) nonlinearity.
SD_D2 inline float3 evaluate(int idx, int degree, int bases,
                             const float3* means, float3 campos,
                             const float* coeffs, bool* clamped) {
    const float3 pos = means[idx];
    float3 dir = make_float3(pos.x - campos.x, pos.y - campos.y, pos.z - campos.z);
    const float inv = rnorm3df(dir.x, dir.y, dir.z);
    dir = make_float3(dir.x * inv, dir.y * inv, dir.z * inv);

    const float3* c = reinterpret_cast<const float3*>(coeffs) + idx * bases;
    float3 result = make_float3(kC0 * c[0].x, kC0 * c[0].y, kC0 * c[0].z);

    const float x = dir.x, y = dir.y, z = dir.z;
    if (degree > 0) {
        result.x += -kC1 * y * c[1].x + kC1 * z * c[2].x - kC1 * x * c[3].x;
        result.y += -kC1 * y * c[1].y + kC1 * z * c[2].y - kC1 * x * c[3].y;
        result.z += -kC1 * y * c[1].z + kC1 * z * c[2].z - kC1 * x * c[3].z;
        if (degree > 1) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            const float q2[5] = {xy, yz, 2.f * zz - xx - yy, xz, xx - yy};
#pragma unroll
            for (int i = 0; i < 5; ++i) {
                result.x += kC2[i] * q2[i] * c[4 + i].x;
                result.y += kC2[i] * q2[i] * c[4 + i].y;
                result.z += kC2[i] * q2[i] * c[4 + i].z;
            }
            if (degree > 2) {
                const float q3[7] = {
                    y * (3.f * xx - yy), xy * z, y * (4.f * zz - xx - yy),
                    z * (2.f * zz - 3.f * xx - 3.f * yy),
                    x * (4.f * zz - xx - yy), z * (xx - yy),
                    x * (xx - 3.f * yy)};
#pragma unroll
                for (int i = 0; i < 7; ++i) {
                    result.x += kC3[i] * q3[i] * c[9 + i].x;
                    result.y += kC3[i] * q3[i] * c[9 + i].y;
                    result.z += kC3[i] * q3[i] * c[9 + i].z;
                }
            }
        }
    }

    result.x += 0.5f;
    result.y += 0.5f;
    result.z += 0.5f;
    clamped[3 * idx + 0] = result.x < 0.f;
    clamped[3 * idx + 1] = result.y < 0.f;
    clamped[3 * idx + 2] = result.z < 0.f;
    return make_float3(fmaxf(result.x, 0.f), fmaxf(result.y, 0.f),
                       fmaxf(result.z, 0.f));
}

// Backward of evaluate(): accumulates dL/d(sh coefficients) and the
// view-direction part of dL/d(mean3d).
struct BackwardIO {
    float3 grad_mean;      // accumulated in/out
    float3* grad_sh;       // [N, bases, 3]
};

SD_D2 inline void backward(int idx, int degree, int bases,
                           const float3* means, float3 campos,
                           const float* coeffs, const bool* clamped,
                           const float3* grad_color, BackwardIO& io) {
    const float3 pos = means[idx];
    const float3 dir_orig = make_float3(pos.x - campos.x, pos.y - campos.y,
                                        pos.z - campos.z);
    const float inv = rnorm3df(dir_orig.x, dir_orig.y, dir_orig.z);
    const float3 dir = make_float3(dir_orig.x * inv, dir_orig.y * inv,
                                   dir_orig.z * inv);

    const float3* c = reinterpret_cast<const float3*>(coeffs) + idx * bases;
    float3 dl = grad_color[idx];
    dl.x *= clamped[3 * idx + 0] ? 0.f : 1.f;
    dl.y *= clamped[3 * idx + 1] ? 0.f : 1.f;
    dl.z *= clamped[3 * idx + 2] ? 0.f : 1.f;

    float3 dxyz[3] = {make_float3(0.f, 0.f, 0.f), make_float3(0.f, 0.f, 0.f),
                      make_float3(0.f, 0.f, 0.f)};
    const float x = dir.x, y = dir.y, z = dir.z;
    float3* g = reinterpret_cast<float3*>(io.grad_sh) + idx * bases;

    g[0].x += kC0 * dl.x;
    g[0].y += kC0 * dl.y;
    g[0].z += kC0 * dl.z;
    if (degree > 0) {
        const float e[3] = {-kC1 * y, kC1 * z, -kC1 * x};
        g[1].x += e[0] * dl.x; g[1].y += e[0] * dl.y; g[1].z += e[0] * dl.z;
        g[2].x += e[1] * dl.x; g[2].y += e[1] * dl.y; g[2].z += e[1] * dl.z;
        g[3].x += e[2] * dl.x; g[3].y += e[2] * dl.y; g[3].z += e[2] * dl.z;
        dxyz[0] = make_float3(-kC1 * c[3].x, -kC1 * c[3].y, -kC1 * c[3].z);
        dxyz[1] = make_float3(-kC1 * c[1].x, -kC1 * c[1].y, -kC1 * c[1].z);
        dxyz[2] = make_float3(kC1 * c[2].x, kC1 * c[2].y, kC1 * c[2].z);
        if (degree > 1) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            const float q2[5] = {xy, yz, 2.f * zz - xx - yy, xz, xx - yy};
            const float dq2_dx[5] = {y, 0.f, -2.f * x, z, 2.f * x};
            const float dq2_dy[5] = {x, z, -2.f * y, 0.f, -2.f * y};
            const float dq2_dz[5] = {0.f, y, 4.f * z, x, 0.f};
#pragma unroll
            for (int i = 0; i < 5; ++i) {
                g[4 + i].x += kC2[i] * q2[i] * dl.x;
                g[4 + i].y += kC2[i] * q2[i] * dl.y;
                g[4 + i].z += kC2[i] * q2[i] * dl.z;
                dxyz[0].x += kC2[i] * dq2_dx[i] * c[4 + i].x;
                dxyz[0].y += kC2[i] * dq2_dx[i] * c[4 + i].y;
                dxyz[0].z += kC2[i] * dq2_dx[i] * c[4 + i].z;
                dxyz[1].x += kC2[i] * dq2_dy[i] * c[4 + i].x;
                dxyz[1].y += kC2[i] * dq2_dy[i] * c[4 + i].y;
                dxyz[1].z += kC2[i] * dq2_dy[i] * c[4 + i].z;
                dxyz[2].x += kC2[i] * dq2_dz[i] * c[4 + i].x;
                dxyz[2].y += kC2[i] * dq2_dz[i] * c[4 + i].y;
                dxyz[2].z += kC2[i] * dq2_dz[i] * c[4 + i].z;
            }
            if (degree > 2) {
                const float q3[7] = {
                    y * (3.f * xx - yy), xy * z, y * (4.f * zz - xx - yy),
                    z * (2.f * zz - 3.f * xx - 3.f * yy),
                    x * (4.f * zz - xx - yy), z * (xx - yy),
                    x * (xx - 3.f * yy)};
                const float dq3_dx[7] = {
                    6.f * xy, yz, -2.f * xy, -6.f * xz,
                    -3.f * xx + 4.f * zz - yy, 2.f * xz, 3.f * (xx - yy)};
                const float dq3_dy[7] = {
                    3.f * (xx - yy), xz, -3.f * yy + 4.f * zz - xx, -6.f * yz,
                    -2.f * xy, -2.f * yz, -6.f * xy};
                const float dq3_dz[7] = {
                    0.f, xy, 8.f * yz, 3.f * (2.f * zz - xx - yy),
                    8.f * xz, xx - yy, 0.f};
#pragma unroll
                for (int i = 0; i < 7; ++i) {
                    g[9 + i].x += kC3[i] * q3[i] * dl.x;
                    g[9 + i].y += kC3[i] * q3[i] * dl.y;
                    g[9 + i].z += kC3[i] * q3[i] * dl.z;
                    dxyz[0].x += kC3[i] * dq3_dx[i] * c[9 + i].x;
                    dxyz[0].y += kC3[i] * dq3_dx[i] * c[9 + i].y;
                    dxyz[0].z += kC3[i] * dq3_dx[i] * c[9 + i].z;
                    dxyz[1].x += kC3[i] * dq3_dy[i] * c[9 + i].x;
                    dxyz[1].y += kC3[i] * dq3_dy[i] * c[9 + i].y;
                    dxyz[1].z += kC3[i] * dq3_dy[i] * c[9 + i].z;
                    dxyz[2].x += kC3[i] * dq3_dz[i] * c[9 + i].x;
                    dxyz[2].y += kC3[i] * dq3_dz[i] * c[9 + i].y;
                    dxyz[2].z += kC3[i] * dq3_dz[i] * c[9 + i].z;
                }
            }
        }
    }

    const float3 dl_dir = make_float3(
        mat::dot3(dxyz[0], dl), mat::dot3(dxyz[1], dl), mat::dot3(dxyz[2], dl));
    const float3 dm = mat::dnormalize(dir_orig, dl_dir);
    io.grad_mean.x += dm.x;
    io.grad_mean.y += dm.y;
    io.grad_mean.z += dm.z;
}

}  // namespace splat_drender::sh
