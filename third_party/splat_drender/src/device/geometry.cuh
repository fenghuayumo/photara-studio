// EWA splat projection geometry, forward and backward.
//
// Forward: world Gaussian (mean, scale/rotation or covariance) -> projected
// 2D covariance, conic, anti-aliased opacity coefficient, peak-depth ray
// plane and footprint normal.
//
// Backward: per-Gaussian gradients of the loss w.r.t. conic/opacity, ray
// plane and normal, propagated to mean3d, opacity, scales, rotations (or
// raw covariance).
//
// Formulas translated from column-major notation use Mat3::at(col,row).
#pragma once

#include "fisheye.cuh"
#include "matrix.cuh"
#include "sh.cuh"
#include "splat_drender/camera.h"

namespace splat_drender::geo {

// Fix the contraction order in each pixel traversal. With thin splats the terms
// nearly cancel; compiler CSE in backward can otherwise change alpha even
// though forward and backward contain the same source expression.
SD_D2 inline float gaussian_power(float4 conic, float dx, float dy) {
    const float xx = __fmul_rn(conic.x, dx);
    const float yy = __fmul_rn(__fmul_rn(conic.z, dy), dy);
    const float xy = __fmul_rn(__fmul_rn(conic.y, dx), dy);
    // Match the reference forward PTX: fuse the x-square sum, then round
    // the half and cross-term subtraction separately.
    return __fsub_rn(__fmul_rn(-0.5f, __fmaf_rn(dx, xx, yy)), xy);
}

struct Splat {
    float2 mean2d{};
    float cov2d[3]{};       // xx, xy, yy (kernel-inflated)
    float conic[3]{};       // inverse of cov2d
    float coef = 1.f;       // sqrt(det0/det1) opacity compensation
    float4 ray_plane{};     // a,b: peak-depth gradient; c: |t|; w: rsigma
    float3 normal{};        // normalized footprint normal, world space
    int radius = 0;
    bool well_conditioned = false;
};

// Cyclic Jacobi eigensolver for symmetric 3x3 matrices (used only with
// precomputed world covariances).
SD_D2 inline void sym_eig3(mat::Mat3 a, float eig[3], mat::Mat3& vecs) {
    vecs = mat::Mat3::diag(1.f, 1.f, 1.f);
    for (int sweep = 0; sweep < 16; ++sweep) {
        float off = fabsf(a.m[0][1]) + fabsf(a.m[0][2]) + fabsf(a.m[1][2]);
        if (off < 1.0e-12f) break;
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (fabsf(a.m[p][q]) < 1.0e-15f) continue;
                const float theta = 0.5f * (a.m[q][q] - a.m[p][p]) / a.m[p][q];
                const float t = copysignf(1.f / (fabsf(theta) + sqrtf(theta * theta + 1.f)), theta);
                const float c = 1.f / sqrtf(t * t + 1.f);
                const float s = t * c;
                for (int k = 0; k < 3; ++k) {
                    const float akp = a.m[k][p], akq = a.m[k][q];
                    a.m[k][p] = c * akp - s * akq;
                    a.m[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    const float apk = a.m[p][k], aqk = a.m[q][k];
                    a.m[p][k] = c * apk - s * aqk;
                    a.m[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    const float vkp = vecs.m[k][p], vkq = vecs.m[k][q];
                    vecs.m[k][p] = c * vkp - s * vkq;
                    vecs.m[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    eig[0] = a.m[0][0];
    eig[1] = a.m[1][1];
    eig[2] = a.m[2][2];
}

namespace detail {

SD_D2 inline int min_index3(const float v[3]) {
    int i = 0;
    if (v[1] < v[i]) i = 1;
    if (v[2] < v[i]) i = 2;
    return i;
}

// GLM stores these as column-major constructors; the visual matrices are
// row-major here. nJ maps a ray-space normal into camera space; nJ_inv
// maps the inverse-covariance plane into pixel-ray coordinates.
SD_D2 inline mat::Mat3 peak_nJ(float3 t, float l) {
    mat::Mat3 J{};
    const float iz = 1.f / t.z;
    J.m[0][0] = iz;
    J.m[0][2] = t.x / l;
    J.m[1][1] = iz;
    J.m[1][2] = t.y / l;
    J.m[2][0] = -t.x * iz * iz;
    J.m[2][1] = -t.y * iz * iz;
    J.m[2][2] = t.z / l;
    return J;
}

SD_D2 inline mat::Mat3 peak_nJ_inv(float u, float v) {
    mat::Mat3 M{};
    const float u2 = u * u, v2 = v * v, uv = u * v;
    M.m[0][0] = v2 + 1.f;
    M.m[0][1] = -uv;
    M.m[0][2] = -u;
    M.m[1][0] = -uv;
    M.m[1][1] = u2 + 1.f;
    M.m[1][2] = -v;
    return M;
}

}  // namespace detail

// See header comment. `tan_fov` args are only used by the pinhole branch.
SD_D2 inline bool project_splat(
    float3 mean, const float* view, const CameraIntrinsics& K, int width,
    int height, float kernel_size, float scale_modifier,
    const float3* scales, const float4* rotations, const float* cov6,
    Splat& out) {
    using mat::Mat3;
    float3 t = mat::xform_point(mean, view);
    const float tc = norm3df(t.x, t.y, t.z);

    float u = 0.f, v = 0.f;
    Mat3 J{};
    float x_mul = 1.f, y_mul = 1.f;
    if (is_pinhole(K.mode)) {
        const float tan_fovx = float(width) / (2.f * K.fx);
        const float tan_fovy = float(height) / (2.f * K.fy);
        const float limx = 1.3f * tan_fovx, limy = 1.3f * tan_fovy;
        u = t.x / t.z;
        v = t.y / t.z;
        t.x = fminf(limx, fmaxf(-limx, u)) * t.z;
        t.y = fminf(limy, fmaxf(-limy, v)) * t.z;
        u = t.x / t.z;
        v = t.y / t.z;
        J.m[0][0] = K.fx / t.z; J.m[0][1] = 0.f;            J.m[0][2] = 0.f;
        J.m[1][0] = 0.f;         J.m[1][1] = K.fy / t.z;    J.m[1][2] = 0.f;
        J.m[2][0] = -(K.fx * t.x) / (t.z * t.z);
        J.m[2][1] = -(K.fy * t.y) / (t.z * t.z);
        J.m[2][2] = 0.f;
    } else {
        const Projection p = project(t, K, width, height);
        if (!p.valid) return false;
        J.m[0][0] = p.J.du[0]; J.m[0][1] = p.J.dv[0];
        J.m[1][0] = p.J.du[1]; J.m[1][1] = p.J.dv[1];
        J.m[2][0] = p.J.du[2]; J.m[2][1] = p.J.dv[2];
        u = t.x / fmaxf(t.z, 1.0e-6f);
        v = t.y / fmaxf(t.z, 1.0e-6f);
    }

    const Mat3 W = mat::world_rotation_transposed(view);
    const Mat3 T = W * J;

    Mat3 cov, Vrk, Vrk_inv, cov_cam_inv;
    bool well_conditioned;
    if (scales) {
        const float3 s = *scales;
        const float3 sl = make_float3(scale_modifier * s.x, scale_modifier * s.y,
                                      scale_modifier * s.z);
        const Mat3 S = Mat3::diag(sl.x, sl.y, sl.z);
        const Mat3 S_inv = Mat3::diag(__frcp_rn(sl.x), __frcp_rn(sl.y), __frcp_rn(sl.z));
        const Mat3 R = mat::quat_rotation(*rotations);
        const Mat3 SR = S * R;
        const Mat3 M = SR * T;
        cov = mat::transpose(M) * M;
        Vrk = mat::transpose(SR) * SR;
        const Mat3 M_inv = S_inv * R * W;
        cov_cam_inv = mat::transpose(M_inv) * M_inv;
        const Mat3 M_inv2 = S_inv * R;
        Vrk_inv = mat::transpose(M_inv2) * M_inv2;
        well_conditioned = true;
    } else {
        Vrk = Mat3{};  // filled below from the 6 unique entries
        Vrk.m[0][0] = cov6[0]; Vrk.m[0][1] = cov6[1]; Vrk.m[0][2] = cov6[2];
        Vrk.m[1][0] = cov6[1]; Vrk.m[1][1] = cov6[3]; Vrk.m[1][2] = cov6[4];
        Vrk.m[2][0] = cov6[2]; Vrk.m[2][1] = cov6[4]; Vrk.m[2][2] = cov6[5];
        cov = mat::transpose(T) * mat::transpose(Vrk) * T;

        float eig[3];
        Mat3 vecs;
        sym_eig3(Vrk, eig, vecs);
        const int mi = detail::min_index3(eig);
        well_conditioned = eig[mi] > 1.0e-8f;
        if (well_conditioned) {
            const Mat3 diag = Mat3::diag(1.f / eig[0], 1.f / eig[1], 1.f / eig[2]);
            Vrk_inv = vecs * diag * mat::transpose(vecs);
        } else {
            // Degenerate direction: eigenvector of the minimum eigenvalue.
            float3 evec = make_float3(vecs.at(mi, 0), vecs.at(mi, 1), vecs.at(mi, 2));
            Vrk_inv = mat::outer(evec, evec);
        }
        cov_cam_inv = mat::transpose(W) * Vrk_inv * W;
    }

    if (is_fisheye(K.mode)) {
        const float t_arr[3] = {t.x, t.y, t.z};
        const float cov_arr[6] = {Vrk.m[0][0], Vrk.m[0][1], Vrk.m[0][2],
                                  Vrk.m[1][1], Vrk.m[1][2], Vrk.m[2][2]};
        const auto native = fisheye::evaluate(t_arr, cov_arr, W, K, kernel_size);
        for (int i = 0; i < 3; ++i) {
            out.cov2d[i] = native.cov[i];
            if (!isfinite(native.cov[i]) || !isfinite(native.normal[i])) return false;
        }
        for (int i = 0; i < 4; ++i)
            if (!isfinite(native.plane[i])) return false;
        out.normal = make_float3(native.normal[0], native.normal[1], native.normal[2]);
        out.ray_plane = make_float4(native.plane[0], native.plane[1],
                                    native.plane[2], native.plane[3]);
        out.coef = native.coefficient;
        return isfinite(out.coef) &&
               out.cov2d[0] * out.cov2d[2] > out.cov2d[1] * out.cov2d[1];
    }

    out.cov2d[0] = cov.m[0][0] + kernel_size;
    out.cov2d[1] = cov.m[0][1];
    out.cov2d[2] = cov.m[1][1] + kernel_size;
    const float det0 = fmaxf(1.0e-6f, cov.m[0][0] * cov.m[1][1] - cov.m[0][1] * cov.m[0][1]);
    const float det1 = fmaxf(1.0e-6f, out.cov2d[0] * out.cov2d[2] - out.cov2d[1] * out.cov2d[1]);
    out.coef = sqrtf(det0 / det1);

    const float3 uvh = make_float3(u, v, 1.f);
    const float3 uvh_m = cov_cam_inv * uvh;
    {
        const float u2 = u * u, v2 = v * v;
        const float l = norm3df(t.x, t.y, t.z);
        const Mat3 nJ_inv = detail::peak_nJ_inv(u, v);
        const float vb = mat::dot3(uvh_m, uvh);
        const float ray_len2 = u2 + v2 + 1.f;
        const float factor_normal = l / ray_len2;
        const float3 plane = nJ_inv * (uvh_m / vb);
        const float rsigmat = well_conditioned ? sqrtf(vb / ray_len2) : 0.f;

        out.ray_plane = make_float4(plane.x * factor_normal / K.fx,
                                    plane.y * factor_normal / K.fy, tc, rsigmat);
        const float3 ray_normal = make_float3(-plane.x * factor_normal,
                                              -plane.y * factor_normal, -1.f);
        const Mat3 nJ = detail::peak_nJ(t, l);
        const float3 cam_normal = nJ * ray_normal;
        const float inv = rnorm3df(cam_normal.x, cam_normal.y, cam_normal.z);
        out.normal = make_float3(cam_normal.x * inv, cam_normal.y * inv,
                                 cam_normal.z * inv);
    }
    return well_conditioned;
}

}  // namespace splat_drender::geo

namespace splat_drender::geo {

// Backward of project_splat() for one Gaussian plus the mean2d->mean3d
// contribution. Accumulates into grad_mean/grad_opacity; produces
// grad_scale/grad_rot or grad_cov6 depending on the parameterization.
struct SplatBackward {
    // Blending gradients for this Gaussian (from the tile backward pass).
    float4 d_conic{};      // xyz conic, w opacity
    float4 d_ray_plane{};  // raw; x,y divided by focal internally
    float3 d_normal{};
    float2 d_mean2d{};

    // Parameterization.
    float3 mean{};
    const float* view = nullptr;
    CameraIntrinsics K{};
    int width = 0, height = 0;
    float kernel_size = 0.f;
    float scale_modifier = 1.f;
    const float3* scale = nullptr;
    const float4* rotation = nullptr;
    const float* cov6 = nullptr;
    float opacity = 0.f;  // activated opacity (fisheye coefficient gradient)

    // Outputs.
    float3 grad_mean{};
    float3 fish_dual_mean{};  // debug: position part of the dual loss
    float grad_opacity = 0.f;
    float3 grad_scale{};
    float4 grad_rotation{};
    float grad_cov6[6]{};
};

namespace detail {

// dSigma -> d(scale, rotation) for Sigma = (S R)^T (S R).
SD_D2 inline void cov3d_backward(float3 scale_local, float4 rot,
                                  const mat::Mat3& R,
                                  const float dL_dsigma[6], float3 dL_dr,
                                  unsigned min_id, float3& grad_scale,
                                  float4& grad_rotation) {
    using mat::Mat3;
    const Mat3 S = Mat3::diag(scale_local.x, scale_local.y, scale_local.z);
    const Mat3 M = S * R;

    const float dL_dsig[3][3] = {
        {dL_dsigma[0], 0.5f * dL_dsigma[1], 0.5f * dL_dsigma[2]},
        {0.5f * dL_dsigma[1], dL_dsigma[3], 0.5f * dL_dsigma[4]},
        {0.5f * dL_dsigma[2], 0.5f * dL_dsigma[4], dL_dsigma[5]}};
    Mat3 dL_dM{};
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < 3; ++k)
                acc += 2.f * M.m[i][k] * dL_dsig[k][j];
            dL_dM.m[i][j] = acc;
        }

    // dL_dMt[a][b] in glm indexing == at(a, b) here; build the transposed
    // matrix once and use at() so the glm formulas translate verbatim.
    const Mat3 dL_dMt_raw = mat::transpose(dL_dM);
    Mat3 dL_dMt = dL_dMt_raw;

    // dL_dscale_i = <row_i(R), col_i(dL_dMt)> from the UNSCALED matrix
    // (S is already folded into dL_dM); the per-column scaling below is
    // only for the quaternion chain, matching the reference order.
    grad_scale.x = R.m[0][0] * dL_dMt.m[0][0] + R.m[0][1] * dL_dMt.m[1][0] +
                   R.m[0][2] * dL_dMt.m[2][0];
    grad_scale.y = R.m[1][0] * dL_dMt.m[0][1] + R.m[1][1] * dL_dMt.m[1][1] +
                   R.m[1][2] * dL_dMt.m[2][1];
    grad_scale.z = R.m[2][0] * dL_dMt.m[0][2] + R.m[2][1] * dL_dMt.m[1][2] +
                   R.m[2][2] * dL_dMt.m[2][2];

    // glm "dL_dMt[i] *= scale_i" scales column i (quaternion chain input).
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        dL_dMt.m[r][0] *= scale_local.x;
        dL_dMt.m[r][1] *= scale_local.y;
        dL_dMt.m[r][2] *= scale_local.z;
    }

    // Degenerate-scale rotation gradient lands on the smallest axis.
#pragma unroll
    for (int r = 0; r < 3; ++r) dL_dMt.m[r][min_id] += dL_dr.x * (r == 0) + dL_dr.y * (r == 1) + dL_dr.z * (r == 2);

    const float r_ = rot.x, x = rot.y, y = rot.z, z = rot.w;
    const float a01 = dL_dMt.at(0, 1), a10 = dL_dMt.at(1, 0);
    const float a02 = dL_dMt.at(0, 2), a20 = dL_dMt.at(2, 0);
    const float a12 = dL_dMt.at(1, 2), a21 = dL_dMt.at(2, 1);
    const float a00 = dL_dMt.at(0, 0), a11 = dL_dMt.at(1, 1), a22 = dL_dMt.at(2, 2);
    grad_rotation.x = 2.f * z * (a01 - a10) + 2.f * y * (a20 - a02) + 2.f * x * (a12 - a21);
    grad_rotation.y = 2.f * y * (a10 + a01) + 2.f * z * (a20 + a02) + 2.f * r_ * (a12 - a21) - 4.f * x * (a22 + a11);
    grad_rotation.z = 2.f * x * (a10 + a01) + 2.f * r_ * (a20 - a02) + 2.f * z * (a12 + a21) - 4.f * y * (a22 + a00);
    grad_rotation.w = 2.f * r_ * (a01 - a10) + 2.f * x * (a20 + a02) + 2.f * y * (a12 + a21) - 4.f * z * (a11 + a00);
}

}  // namespace detail

// Adds the dL/d(pixel center) -> dL/d(mean3d) contribution through the
// camera projection Jacobian. Applies to every camera model.
SD_D2 inline void add_mean2d_gradient(SplatBackward& io) {
    float3 gm;
    if (is_pinhole(io.K.mode)) {
        const float3 pv = mat::xform_point(io.mean, io.view);
        const float rz = 1.f / (pv.z + 1.0e-7f);
        const float sx = pv.x * rz, sy = pv.y * rz;
        gm.x = (io.K.fx * (io.view[0] - sx * io.view[2]) * io.d_mean2d.x +
                io.K.fy * (io.view[1] - sy * io.view[2]) * io.d_mean2d.y) * rz;
        gm.y = (io.K.fx * (io.view[4] - sx * io.view[6]) * io.d_mean2d.x +
                io.K.fy * (io.view[5] - sy * io.view[6]) * io.d_mean2d.y) * rz;
        gm.z = (io.K.fx * (io.view[8] - sx * io.view[10]) * io.d_mean2d.x +
                io.K.fy * (io.view[9] - sy * io.view[10]) * io.d_mean2d.y) * rz;
    } else {
        const float3 pv = mat::xform_point(io.mean, io.view);
        const Projection p = project(pv, io.K, io.width, io.height);
        if (p.valid) {
            const float gcx = p.J.du[0] * io.d_mean2d.x + p.J.dv[0] * io.d_mean2d.y;
            const float gcy = p.J.du[1] * io.d_mean2d.x + p.J.dv[1] * io.d_mean2d.y;
            const float gcz = p.J.du[2] * io.d_mean2d.x + p.J.dv[2] * io.d_mean2d.y;
            gm.x = io.view[0] * gcx + io.view[1] * gcy + io.view[2] * gcz;
            gm.y = io.view[4] * gcx + io.view[5] * gcy + io.view[6] * gcz;
            gm.z = io.view[8] * gcx + io.view[9] * gcy + io.view[10] * gcz;
        } else {
            gm = make_float3(0.f, 0.f, 0.f);
        }
    }
    io.grad_mean.x += gm.x;
    io.grad_mean.y += gm.y;
    io.grad_mean.z += gm.z;
}

SD_D2 inline void splat_backward(SplatBackward& io) {
    using mat::Mat3;
    const float h_x = io.K.fx, h_y = io.K.fy;
    // The generic path stores the plane gradient pre-divided by the focal
    // lengths (forward stores plane.x/y * factor / f); the native fisheye
    // geometry consumes the raw blended gradient instead.
    float4 d_ray_plane = io.d_ray_plane;
    d_ray_plane.x /= h_x;
    d_ray_plane.y /= h_y;
    const float4& d_ray_plane_raw = io.d_ray_plane;
    const float dL_dtc = d_ray_plane.z;
    const float dL_drsigma = d_ray_plane.w;

    float3 t = mat::xform_point(io.mean, io.view);
    const float rtc = rnorm3df(t.x, t.y, t.z);
    const float3 dL_dt_tc = make_float3(t.x * rtc * dL_dtc, t.y * rtc * dL_dtc,
                                        t.z * rtc * dL_dtc);

    float u, v;
    float x_mul = 1.f, y_mul = 1.f;
    Mat3 J{};
    if (is_pinhole(io.K.mode)) {
        const float tan_fovx = float(io.width) / (2.f * h_x);
        const float tan_fovy = float(io.height) / (2.f * h_y);
        const float limx = 1.3f * tan_fovx, limy = 1.3f * tan_fovy;
        u = t.x / t.z;
        v = t.y / t.z;
        const float uu = u, vv = v;
        t.x = fminf(limx, fmaxf(-limx, u)) * t.z;
        t.y = fminf(limy, fmaxf(-limy, v)) * t.z;
        x_mul = (uu < -limx || uu > limx) ? 0.f : 1.f;
        y_mul = (vv < -limy || vv > limy) ? 0.f : 1.f;
        u = t.x / t.z;
        v = t.y / t.z;
        J.m[0][0] = h_x / t.z; J.m[0][1] = 0.f; J.m[0][2] = 0.f;
        J.m[1][0] = 0.f; J.m[1][1] = h_y / t.z; J.m[1][2] = 0.f;
        J.m[2][0] = -(h_x * t.x) / (t.z * t.z);
        J.m[2][1] = -(h_y * t.y) / (t.z * t.z);
        J.m[2][2] = 0.f;
    } else {
        const Projection p = project(t, io.K, io.width, io.height);
        if (!p.valid) return;
        J.m[0][0] = p.J.du[0]; J.m[0][1] = p.J.du[1]; J.m[0][2] = p.J.du[2];
        J.m[1][0] = p.J.dv[0]; J.m[1][1] = p.J.dv[1]; J.m[1][2] = p.J.dv[2];
        u = t.x / fmaxf(t.z, 1.0e-6f);
        v = t.y / fmaxf(t.z, 1.0e-6f);
    }

    const Mat3 W = mat::world_rotation_transposed(io.view);
    const Mat3 T = W * J;

    Mat3 cov2D, cov_cam_inv, Vrk, Vrk_inv;
    Mat3 eig_vecs{};
    float eig_val[3]{};
    Mat3 R{};
    float3 scale_local{};
    float4 rot{};
    bool well_conditioned;
    unsigned min_id = 0;

    if (io.scale) {
        const float3 s = *io.scale;
        scale_local = make_float3(io.scale_modifier * s.x, io.scale_modifier * s.y,
                                  io.scale_modifier * s.z);
        const Mat3 S = Mat3::diag(scale_local.x, scale_local.y, scale_local.z);
        const Mat3 S_inv = Mat3::diag(__frcp_rn(scale_local.x),
                                      __frcp_rn(scale_local.y),
                                      __frcp_rn(scale_local.z));
        rot = *io.rotation;
        R = mat::quat_rotation(rot);
        const Mat3 SR = S * R;
        const Mat3 M = SR * T;
        cov2D = mat::transpose(M) * M;
        Vrk = mat::transpose(SR) * SR;
        Mat3 M_inv = S_inv * R * W;
        cov_cam_inv = mat::transpose(M_inv) * M_inv;
        M_inv = S_inv * R;
        Vrk_inv = mat::transpose(M_inv) * M_inv;
        well_conditioned = true;
        const float sl[3] = {scale_local.x, scale_local.y, scale_local.z};
        min_id = unsigned(detail::min_index3(sl));
    } else {
        Vrk = Mat3{};
        Vrk.m[0][0] = io.cov6[0]; Vrk.m[0][1] = io.cov6[1]; Vrk.m[0][2] = io.cov6[2];
        Vrk.m[1][0] = io.cov6[1]; Vrk.m[1][1] = io.cov6[3]; Vrk.m[1][2] = io.cov6[4];
        Vrk.m[2][0] = io.cov6[2]; Vrk.m[2][1] = io.cov6[4]; Vrk.m[2][2] = io.cov6[5];
        cov2D = mat::transpose(T) * mat::transpose(Vrk) * T;
        sym_eig3(Vrk, eig_val, eig_vecs);
        min_id = unsigned(detail::min_index3(eig_val));
        well_conditioned = eig_val[min_id] > 1.0e-8f;
        if (well_conditioned) {
            const Mat3 d = Mat3::diag(1.f / eig_val[0], 1.f / eig_val[1],
                                      1.f / eig_val[2]);
            Vrk_inv = eig_vecs * d * mat::transpose(eig_vecs);
        } else {
            const float3 evec = make_float3(eig_vecs.at(min_id, 0),
                                            eig_vecs.at(min_id, 1),
                                            eig_vecs.at(min_id, 2));
            Vrk_inv = mat::outer(evec, evec);
        }
        cov_cam_inv = mat::transpose(W) * Vrk_inv * W;
    }

    if (is_fisheye(io.K.mode)) {
        // Forward-mode duals differentiate the exact native expressions.
        const fisheye::Dual position[3] = {
            fisheye::Dual::variable(t.x, 0), fisheye::Dual::variable(t.y, 1),
            fisheye::Dual::variable(t.z, 2)};
        const fisheye::Dual covariance[6] = {
            fisheye::Dual::variable(Vrk.m[0][0], 3),
            fisheye::Dual::variable(Vrk.m[0][1], 4),
            fisheye::Dual::variable(Vrk.m[0][2], 5),
            fisheye::Dual::variable(Vrk.m[1][1], 6),
            fisheye::Dual::variable(Vrk.m[1][2], 7),
            fisheye::Dual::variable(Vrk.m[2][2], 8)};
        const auto native = fisheye::evaluate(position, covariance, W, io.K,
                                               io.kernel_size);
        // The raster kernel stores half of the off-diagonal conic gradient.
        fisheye::Dual loss =
            native.conic[0] * fisheye::Dual(io.d_conic.x) +
            native.conic[1] * fisheye::Dual(2.f * io.d_conic.y) +
            native.conic[2] * fisheye::Dual(io.d_conic.z) +
            native.coefficient * fisheye::Dual(io.opacity * io.d_conic.w);
        loss = loss + native.plane[0] * fisheye::Dual(d_ray_plane_raw.x) +
               native.plane[1] * fisheye::Dual(d_ray_plane_raw.y) +
               native.plane[2] * fisheye::Dual(d_ray_plane_raw.z) +
               native.plane[3] * fisheye::Dual(d_ray_plane_raw.w) +
               native.normal[0] * fisheye::Dual(io.d_normal.x) +
               native.normal[1] * fisheye::Dual(io.d_normal.y) +
               native.normal[2] * fisheye::Dual(io.d_normal.z);
        const float3 gm = mat::xform_dir_transpose(
            make_float3(loss.d[0], loss.d[1], loss.d[2]), io.view);
        io.grad_mean.x += gm.x;
        io.grad_mean.y += gm.y;
        io.grad_mean.z += gm.z;
        io.grad_opacity = io.d_conic.w * native.coefficient.v;
        float gc[6];
        for (int i = 0; i < 6; ++i) gc[i] = loss.d[i + 3];
        if (io.scale) {
            detail::cov3d_backward(scale_local, rot, R, gc,
                                   make_float3(0.f, 0.f, 0.f), min_id,
                                   io.grad_scale, io.grad_rotation);
            io.grad_scale = io.grad_scale * io.scale_modifier;
        } else {
            for (int i = 0; i < 6; ++i) io.grad_cov6[i] = gc[i];
        }
        io.fish_dual_mean = make_float3(loss.d[0], loss.d[1], loss.d[2]);
        add_mean2d_gradient(io);
        return;
    }

    // ---- generic (pinhole & equirect) backward chain ----
    const float det_0 = fmaxf(1.0e-6f, cov2D.m[0][0] * cov2D.m[1][1] -
                                           cov2D.m[0][1] * cov2D.m[0][1]);
    const float det_1 = fmaxf(1.0e-6f,
                              (cov2D.m[0][0] + io.kernel_size) *
                                      (cov2D.m[1][1] + io.kernel_size) -
                                  cov2D.m[0][1] * cov2D.m[0][1]);
    const float coef = sqrtf(det_0 / det_1);

    const float3 uvh = make_float3(u, v, 1.f);
    const float3 uvh_m = cov_cam_inv * uvh;
    const float u2 = u * u, v2 = v * v, uv = u * v;

    Mat3 dL_dVrk{};
    float3 dL_dr(0.f, 0.f, 0.f);
    float dL_du = 0.f, dL_dv = 0.f, dL_dz = 0.f;
    // An unused depth/normal branch is exactly zero. Evaluating its inverse
    // covariance algebra anyway can produce 0 * NaN for very thin splats,
    // contaminating otherwise valid RGB derivatives (also in the reference).
    if (io.d_normal.x != 0.f || io.d_normal.y != 0.f || io.d_normal.z != 0.f ||
        d_ray_plane.x != 0.f || d_ray_plane.y != 0.f ||
        d_ray_plane.z != 0.f || d_ray_plane.w != 0.f) {
        const float vb = mat::dot3(uvh_m, uvh);
        const float l = norm3df(t.x, t.y, t.z);
        const Mat3 nJ = detail::peak_nJ(t, l);
        const Mat3 nJ_inv = detail::peak_nJ_inv(u, v);
        const float clamp_vb = fmaxf(vb, 1.0e-7f);
        const float ray_len2 = u2 + v2 + 1.f;
        const float ray_len_inv = rsqrtf(ray_len2);
        const float factor_normal = l / ray_len2;
        const float3 uvh_m_vb = make_float3(uvh_m.x / clamp_vb, uvh_m.y / clamp_vb,
                                            uvh_m.z / clamp_vb);
        const float3 plane = nJ_inv * uvh_m_vb;
        const float3 ray_normal = make_float3(-plane.x * factor_normal,
                                              -plane.y * factor_normal, -1.f);
        const float3 cam_normal = nJ * ray_normal;
        const float inv = rnorm3df(cam_normal.x, cam_normal.y, cam_normal.z);
        const float3 nrm = make_float3(cam_normal.x * inv, cam_normal.y * inv,
                                       cam_normal.z * inv);
        // The reference evaluates the normalization-Jacobian factor on the
        // already-normalized vector (so it is 1, not 1/|cam_normal|).
        // Replicate that exactly: training dynamics were tuned with it.
        const float rlv = rnorm3df(nrm.x, nrm.y, nrm.z);
        const float ndl = mat::dot3(nrm, io.d_normal);
        const float3 dL_dcam_normal = make_float3(
            (io.d_normal.x - nrm.x * ndl) * rlv,
            (io.d_normal.y - nrm.y * ndl) * rlv,
            (io.d_normal.z - nrm.z * ndl) * rlv);
        const float3 dL_dray_normal = mat::transpose(nJ) * dL_dcam_normal;
        const Mat3 dL_dnJ = mat::outer(dL_dcam_normal, ray_normal);
        const float dL_dfactor =
            plane.x * (-dL_dray_normal.x + d_ray_plane.x) +
            plane.y * (-dL_dray_normal.y + d_ray_plane.y);

        const float dL_dplane_x = (-dL_dray_normal.x + d_ray_plane.x) * factor_normal;
        const float dL_dplane_y = (-dL_dray_normal.y + d_ray_plane.y) * factor_normal;
        const float3 dL_dplane_append = make_float3(dL_dplane_x, dL_dplane_y, 0.f);

        const float aux = dL_dplane_x * plane.x + dL_dplane_y * plane.y;
        const float3 W_uvh = W * uvh;

        const float3 tmp = mat::transpose(nJ_inv) * dL_dplane_append;
        const float3 numerator = make_float3(
            (cov_cam_inv.m[0][0] * tmp.x + cov_cam_inv.m[0][1] * tmp.y +
             cov_cam_inv.m[0][2] * tmp.z) / clamp_vb,
            (cov_cam_inv.m[1][0] * tmp.x + cov_cam_inv.m[1][1] * tmp.y +
             cov_cam_inv.m[1][2] * tmp.z) / clamp_vb,
            (cov_cam_inv.m[2][0] * tmp.x + cov_cam_inv.m[2][1] * tmp.y +
             cov_cam_inv.m[2][2] * tmp.z) / clamp_vb);
        const float3 dL_duvh_plane = make_float3(
            2.f * (-aux) * uvh_m_vb.x + numerator.x,
            2.f * (-aux) * uvh_m_vb.y + numerator.y,
            2.f * (-aux) * uvh_m_vb.z + numerator.z);

        const float rsigmat = sqrtf(vb / ray_len2);
        const float dL_dlen2_x2 = -dL_drsigma * rsigmat / ray_len2;
        const float dL_du_sigma = dL_dlen2_x2 * u;
        const float dL_dv_sigma = dL_dlen2_x2 * v;

        const float aux_nJ = (-dL_dnJ.at(2, 0) * u - dL_dnJ.at(2, 1) * v -
                              dL_dnJ.at(2, 2)) / ray_len2 * ray_len_inv;
        const float dL_du_nJ = -dL_dnJ.at(0, 2) / t.z + dL_dnJ.at(2, 0) * ray_len_inv +
                               aux_nJ * u;
        const float dL_dv_nJ = -dL_dnJ.at(1, 2) / t.z + dL_dnJ.at(2, 1) * ray_len_inv +
                               aux_nJ * v;
        const float dL_dz_nJ = (dL_dnJ.at(0, 0) + dL_dnJ.at(1, 1) -
                                dL_dnJ.at(0, 2) * u - dL_dnJ.at(1, 2) * v) /
                               (-t.z * t.z);

        const Mat3 dL_dnJ_inv = mat::outer(dL_dplane_append, uvh_m_vb);
        const float dL_du_plane =
            dL_duvh_plane.x + (dL_dnJ_inv.at(0, 1) + dL_dnJ_inv.at(1, 0)) * (-v) +
            2.f * dL_dnJ_inv.at(1, 1) * u - dL_dnJ_inv.at(2, 0);
        const float dL_dv_plane =
            dL_duvh_plane.y + (dL_dnJ_inv.at(0, 1) + dL_dnJ_inv.at(1, 0)) * (-u) +
            2.f * dL_dnJ_inv.at(0, 0) * v - dL_dnJ_inv.at(2, 1);

        const float aux_factor = dL_dfactor * (-t.z / ray_len2 * ray_len_inv);
        const float dL_du_factor = aux_factor * u;
        const float dL_dv_factor = aux_factor * v;
        const float dL_dz_factor = dL_dfactor * ray_len_inv;

        dL_du = dL_du_nJ + dL_du_plane + dL_du_factor + dL_du_sigma;
        dL_dv = dL_dv_nJ + dL_dv_plane + dL_dv_factor + dL_dv_sigma;
        dL_dz = dL_dz_nJ + dL_dz_factor;

        const float dL_dvb_xvb = -aux + dL_drsigma * 0.5f * rsigmat;
        if (well_conditioned) {
            const float3 rhs1 = W * tmp;  // W * transpose(nJ_inv) * dL_dplane
            const float3 rhs2 = make_float3(W_uvh.x * dL_dvb_xvb,
                                            W_uvh.y * dL_dvb_xvb,
                                            W_uvh.z * dL_dvb_xvb);
            const float3 rhs = make_float3(rhs1.x + rhs2.x, rhs1.y + rhs2.y,
                                           rhs1.z + rhs2.z);
            const float3 lhs = Vrk_inv * W_uvh;
            const float3 vr = Vrk_inv * rhs;
            dL_dVrk = mat::outer(lhs, make_float3(-vr.x, -vr.y, -vr.z));
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j) dL_dVrk.m[i][j] /= vb;
            dL_dr = make_float3(0.f, 0.f, 0.f);
        } else {
            dL_dVrk = Mat3{};
            const float3 nJ_inv_plane = mat::transpose(nJ_inv) * dL_dplane_append;
            const float3 wcol = W * nJ_inv_plane;
            Mat3 dL_dVrk_inv{};
            {
                const float3 b = make_float3(W_uvh.x * dL_dvb_xvb + wcol.x,
                                             W_uvh.y * dL_dvb_xvb + wcol.y,
                                             W_uvh.z * dL_dvb_xvb + wcol.z);
                dL_dVrk_inv = mat::outer(W_uvh, b);
#pragma unroll
                for (int i = 0; i < 3; ++i)
#pragma unroll
                    for (int j = 0; j < 3; ++j) dL_dVrk_inv.m[i][j] /= vb;
            }
            if (io.scale) {
                const float3 evec = make_float3(R.at(min_id, 0), R.at(min_id, 1),
                                                R.at(min_id, 2));
                const Mat3 sym = dL_dVrk_inv + mat::transpose(dL_dVrk_inv);
                dL_dr = sym * evec;
            } else {
                const float3 evec = make_float3(eig_vecs.at(min_id, 0),
                                                eig_vecs.at(min_id, 1),
                                                eig_vecs.at(min_id, 2));
                const Mat3 sym = dL_dVrk_inv + mat::transpose(dL_dVrk_inv);
                const float3 dL_dv = sym * evec;
                for (int j = 1; j < 3; ++j) {
                    const int k = (j + int(min_id)) % 3;
                    const float3 vk = make_float3(eig_vecs.at(k, 0), eig_vecs.at(k, 1),
                                                  eig_vecs.at(k, 2));
                    const float scale =
                        mat::dot3(vk, dL_dv) /
                        fminf(eig_val[min_id] - eig_val[k], -1.0e-7f);
                    const Mat3 add = mat::outer(
                        make_float3(vk.x * scale, vk.y * scale, vk.z * scale), evec);
#pragma unroll
                    for (int a = 0; a < 3; ++a)
#pragma unroll
                        for (int b = 0; b < 3; ++b) dL_dVrk.m[a][b] += add.m[a][b];
                }
            }
        }
    }

    // Conic (inverse covariance) backward.
    const float a = cov2D.m[0][0] + io.kernel_size;
    const float b = cov2D.m[0][1];
    const float c = cov2D.m[1][1] + io.kernel_size;
    const float denom = a * c - b * b;
    const float denom2inv = 1.f / (denom * denom + 1.0e-7f);
    float dL_da = 0.f, dL_db = 0.f, dL_dc = 0.f;
    float dL_dcov_local[6];
    if (denom2inv != 0.f) {
        dL_da = denom2inv * (-c * c * io.d_conic.x + 2.f * b * c * io.d_conic.y +
                             (denom - a * c) * io.d_conic.z);
        dL_dc = denom2inv * (-a * a * io.d_conic.z + 2.f * a * b * io.d_conic.y +
                             (denom - a * c) * io.d_conic.x);
        dL_db = denom2inv * 2.f * (b * c * io.d_conic.x -
                                   (denom + 2.f * b * b) * io.d_conic.y +
                                   a * b * io.d_conic.z);

        io.grad_opacity = io.d_conic.w * coef;
        // The Mip opacity compensation stays detached from the covariance
        // gradients exactly like the reference: its full analytical
        // derivative is numerically unstable for tiny/elongated splats and
        // drives degenerate scale growth. The compensated forward opacity
        // and the opacity gradient above remain intact.

        dL_dcov_local[0] = T.at(0, 0) * T.at(0, 0) * dL_da +
                           T.at(0, 0) * T.at(1, 0) * dL_db +
                           T.at(1, 0) * T.at(1, 0) * dL_dc;
        dL_dcov_local[3] = T.at(0, 1) * T.at(0, 1) * dL_da +
                           T.at(0, 1) * T.at(1, 1) * dL_db +
                           T.at(1, 1) * T.at(1, 1) * dL_dc;
        dL_dcov_local[5] = T.at(0, 2) * T.at(0, 2) * dL_da +
                           T.at(0, 2) * T.at(1, 2) * dL_db +
                           T.at(1, 2) * T.at(1, 2) * dL_dc;
        dL_dcov_local[1] = 2.f * T.at(0, 0) * T.at(0, 1) * dL_da +
                           (T.at(0, 0) * T.at(1, 1) + T.at(0, 1) * T.at(1, 0)) * dL_db +
                           2.f * T.at(1, 0) * T.at(1, 1) * dL_dc;
        dL_dcov_local[2] = 2.f * T.at(0, 0) * T.at(0, 2) * dL_da +
                           (T.at(0, 0) * T.at(1, 2) + T.at(0, 2) * T.at(1, 0)) * dL_db +
                           2.f * T.at(1, 0) * T.at(1, 2) * dL_dc;
        dL_dcov_local[4] = 2.f * T.at(0, 2) * T.at(0, 1) * dL_da +
                           (T.at(0, 1) * T.at(1, 2) + T.at(0, 2) * T.at(1, 1)) * dL_db +
                           2.f * T.at(1, 1) * T.at(1, 2) * dL_dc;
    } else {
        for (int i = 0; i < 6; ++i) dL_dcov_local[i] = 0.f;
    }
    dL_dcov_local[0] += dL_dVrk.m[0][0];
    dL_dcov_local[3] += dL_dVrk.m[1][1];
    dL_dcov_local[5] += dL_dVrk.m[2][2];
    dL_dcov_local[1] += dL_dVrk.m[0][1] + dL_dVrk.m[1][0];
    dL_dcov_local[2] += dL_dVrk.m[0][2] + dL_dVrk.m[2][0];
    dL_dcov_local[4] += dL_dVrk.m[1][2] + dL_dVrk.m[2][1];

    if (io.scale) {
        detail::cov3d_backward(scale_local, rot, R, dL_dcov_local, dL_dr, min_id,
                               io.grad_scale, io.grad_rotation);
        io.grad_scale = io.grad_scale * io.scale_modifier;
    } else {
        for (int i = 0; i < 6; ++i) io.grad_cov6[i] = dL_dcov_local[i];
    }

    const float dL_dT00 = 2.f * (T.at(0, 0) * Vrk.at(0, 0) + T.at(0, 1) * Vrk.at(0, 1) +
                                 T.at(0, 2) * Vrk.at(0, 2)) * dL_da +
                          (T.at(1, 0) * Vrk.at(0, 0) + T.at(1, 1) * Vrk.at(0, 1) +
                           T.at(1, 2) * Vrk.at(0, 2)) * dL_db;
    const float dL_dT01 = 2.f * (T.at(0, 0) * Vrk.at(1, 0) + T.at(0, 1) * Vrk.at(1, 1) +
                                 T.at(0, 2) * Vrk.at(1, 2)) * dL_da +
                          (T.at(1, 0) * Vrk.at(1, 0) + T.at(1, 1) * Vrk.at(1, 1) +
                           T.at(1, 2) * Vrk.at(1, 2)) * dL_db;
    const float dL_dT02 = 2.f * (T.at(0, 0) * Vrk.at(2, 0) + T.at(0, 1) * Vrk.at(2, 1) +
                                 T.at(0, 2) * Vrk.at(2, 2)) * dL_da +
                          (T.at(1, 0) * Vrk.at(2, 0) + T.at(1, 1) * Vrk.at(2, 1) +
                           T.at(1, 2) * Vrk.at(2, 2)) * dL_db;
    const float dL_dT10 = 2.f * (T.at(1, 0) * Vrk.at(0, 0) + T.at(1, 1) * Vrk.at(0, 1) +
                                 T.at(1, 2) * Vrk.at(0, 2)) * dL_dc +
                          (T.at(0, 0) * Vrk.at(0, 0) + T.at(0, 1) * Vrk.at(0, 1) +
                           T.at(0, 2) * Vrk.at(0, 2)) * dL_db;
    const float dL_dT11 = 2.f * (T.at(1,0) * Vrk.at(1, 0) + T.at(1, 1) * Vrk.at(1, 1) +
                                 T.at(1, 2) * Vrk.at(1, 2)) * dL_dc +
                          (T.at(0, 0) * Vrk.at(1, 0) + T.at(0, 1) * Vrk.at(1, 1) +
                           T.at(0, 2) * Vrk.at(1, 2)) * dL_db;
    const float dL_dT12 = 2.f * (T.at(1, 0) * Vrk.at(2, 0) + T.at(1, 1) * Vrk.at(2, 1) +
                                 T.at(1, 2) * Vrk.at(2, 2)) * dL_dc +
                          (T.at(0, 0) * Vrk.at(2, 0) + T.at(0, 1) * Vrk.at(2, 1) +
                           T.at(0, 2) * Vrk.at(2, 2)) * dL_db;

    const float dL_dJ00 = W.at(0, 0) * dL_dT00 + W.at(0, 1) * dL_dT01 + W.at(0, 2) * dL_dT02;
    const float dL_dJ02 = W.at(2, 0) * dL_dT00 + W.at(2, 1) * dL_dT01 + W.at(2, 2) * dL_dT02;
    const float dL_dJ11 = W.at(1, 0) * dL_dT10 + W.at(1, 1) * dL_dT11 + W.at(1, 2) * dL_dT12;
    const float dL_dJ12 = W.at(2, 0) * dL_dT10 + W.at(2, 1) * dL_dT11 + W.at(2, 2) * dL_dT12;

    const float tz = 1.f / t.z;
    const float tz2 = tz * tz;
    const float tz3 = tz2 * tz;
    const float dL_dtx = x_mul * (-h_x * tz2 * dL_dJ02 + dL_du * tz);
    const float dL_dty = y_mul * (-h_y * tz2 * dL_dJ12 + dL_dv * tz);
    const float dL_dtz =
        -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 +
        ((1.f + x_mul) * h_x * t.x) * tz3 * dL_dJ02 +
        ((1.f + y_mul) * h_y * t.y) * tz3 * dL_dJ12 -
        (x_mul * dL_du * t.x + y_mul * dL_dv * t.y) * tz2 + dL_dz;

    const float3 gm = mat::xform_dir_transpose(
        make_float3(dL_dtx + dL_dt_tc.x, dL_dty + dL_dt_tc.y, dL_dtz + dL_dt_tc.z),
        io.view);
    io.grad_mean.x += gm.x;
    io.grad_mean.y += gm.y;
    io.grad_mean.z += gm.z;

    add_mean2d_gradient(io);
}

}  // namespace splat_drender::geo
