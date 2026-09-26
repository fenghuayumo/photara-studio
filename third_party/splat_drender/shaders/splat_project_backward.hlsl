#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<float> means;
[[vk::binding(1, 0)]] StructuredBuffer<float> opacities;
[[vk::binding(2, 0)]] StructuredBuffer<float> scales;
[[vk::binding(3, 0)]] StructuredBuffer<float> rotations;
[[vk::binding(4, 0)]] StructuredBuffer<float> covariances;
[[vk::binding(5, 0)]] StructuredBuffer<float> color_src;
[[vk::binding(6, 0)]] StructuredBuffer<float> camera;
[[vk::binding(7, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(8, 0)]] StructuredBuffer<uint> gauss_u;
[[vk::binding(9, 0)]] StructuredBuffer<float> blend_grad;
[[vk::binding(10, 0)]] StructuredBuffer<float> raw_log_scales;
[[vk::binding(11, 0)]] StructuredBuffer<float> raw_rotations;
[[vk::binding(12, 0)]] StructuredBuffer<float> opacity_logits;
[[vk::binding(13, 0)]] StructuredBuffer<float> filter_3d;
[[vk::binding(14, 0)]] RWStructuredBuffer<float> model_grad;

struct D3 {
    float v;
    float3 d;
};

D3 dc(float v) { D3 r; r.v = v; r.d = 0.0f; return r; }
D3 dv(float v, uint axis) {
    D3 r = dc(v);
    if (axis == 0u) r.d.x = 1.0f;
    else if (axis == 1u) r.d.y = 1.0f;
    else r.d.z = 1.0f;
    return r;
}
D3 da(D3 a, D3 b) { D3 r; r.v = a.v + b.v; r.d = a.d + b.d; return r; }
D3 ds(D3 a, D3 b) { D3 r; r.v = a.v - b.v; r.d = a.d - b.d; return r; }
D3 dn(D3 a) { D3 r; r.v = -a.v; r.d = -a.d; return r; }
D3 dm(D3 a, D3 b) { D3 r; r.v = a.v * b.v; r.d = a.d * b.v + a.v * b.d; return r; }
D3 dd(D3 a, D3 b) { D3 r; r.v = a.v / b.v; r.d = (a.d - r.v * b.d) / b.v; return r; }
D3 dq(D3 a) { D3 r; r.v = sqrt(a.v); r.d = a.d * (0.5f / r.v); return r; }
D3 datan2(D3 y, D3 x) {
    D3 r;
    r.v = atan2(y.v, x.v);
    r.d = (x.v * y.d - y.v * x.d) / (x.v * x.v + y.v * y.v);
    return r;
}
D3 dclamp(D3 a, float lo, float hi) {
    if (a.v < lo) return dc(lo);
    if (a.v > hi) return dc(hi);
    return a;
}

struct DualProjection {
    D3 j00, j01, j02;
    D3 j10, j11, j12;
    bool valid;
};

DualProjection projection_jacobian_dual(
    float3 t, uint mode, int width, int height, float fx, float fy,
    float k1, float k2, float k3, float k4) {
    DualProjection p;
    p.j00 = p.j01 = p.j02 = p.j10 = p.j11 = p.j12 = dc(0.0f);
    p.valid = true;
    D3 x = dv(t.x, 0u), y = dv(t.y, 1u), z = dv(t.z, 2u);
    if (mode == kModeOrtho) {
        p.j00 = dc(fx);
        p.j11 = dc(fy);
        return p;
    }
    if (mode == kModePinhole) {
        float limx = 1.3f * float(width) / (2.0f * fx);
        float limy = 1.3f * float(height) / (2.0f * fy);
        D3 u = dclamp(dd(x, z), -limx, limx);
        D3 v = dclamp(dd(y, z), -limy, limy);
        D3 tx = dm(u, z), ty = dm(v, z);
        D3 iz = dd(dc(1.0f), z);
        p.j00 = dm(dc(fx), iz);
        p.j11 = dm(dc(fy), iz);
        p.j02 = dn(dm(dm(dc(fx), tx), dm(iz, iz)));
        p.j12 = dn(dm(dm(dc(fy), ty), dm(iz, iz)));
        return p;
    }
    if (mode == kModeEquirect) {
        D3 h = dq(da(dm(x, x), dm(z, z)));
        D3 l2 = da(da(dm(x, x), dm(y, y)), dm(z, z));
        D3 h2 = dm(h, h);
        D3 su = dc(float(width) / (2.0f * kPi));
        D3 sv = dc(float(height) / kPi);
        p.j00 = dm(su, dd(z, h2));
        p.j01 = dc(0.0f);
        p.j02 = dn(dm(su, dd(x, h2)));
        p.j10 = dn(dm(sv, dd(dm(y, x), dm(h, l2))));
        p.j11 = dm(sv, dd(h, l2));
        p.j12 = dn(dm(sv, dd(dm(y, z), dm(h, l2))));
        p.valid = h.v >= 1.0e-5f;
        return p;
    }

    D3 r2 = da(dm(x, x), dm(y, y));
    D3 l2 = da(r2, dm(z, z));
    if (r2.v < 1.0e-6f * z.v * z.v) {
        D3 iz = dd(dc(1.0f), z);
        D3 q = dm(r2, dm(iz, iz));
        float aa = k1 - 1.0f / 3.0f;
        float bb = k2 - k1 + 1.0f / 5.0f;
        D3 s = dm(iz, da(dc(1.0f), da(dm(dc(aa), q), dm(dc(bb), dm(q, q)))));
        D3 deriv = dm(dc(2.0f), dm(dm(dm(iz, iz), iz), da(dc(aa), dm(dc(2.0f * bb), q))));
        D3 dz = dn(dm(dm(iz, iz), da(dc(1.0f), da(dm(dc(3.0f * aa), q), dm(dc(5.0f * bb), dm(q, q))))));
        p.j00 = dm(dc(fx), da(s, dm(dm(x, x), deriv)));
        p.j01 = dm(dc(fx), dm(dm(x, y), deriv));
        p.j02 = dm(dc(fx), dm(x, dz));
        p.j10 = dm(dc(fy), dm(dm(x, y), deriv));
        p.j11 = dm(dc(fy), da(s, dm(dm(y, y), deriv)));
        p.j12 = dm(dc(fy), dm(y, dz));
    } else {
        D3 r = dq(r2);
        D3 theta = datan2(r, z);
        D3 q = dm(theta, theta);
        D3 td = dm(theta, da(dc(1.0f), dm(q, da(
            dc(k1), dm(q, da(dc(k2), dm(q, da(dc(k3), dm(q, dc(k4))))))))));
        D3 dt = da(dc(1.0f), dm(q, da(
            dc(3.0f * k1), dm(q, da(dc(5.0f * k2), dm(q, da(
                dc(7.0f * k3), dm(q, dc(9.0f * k4)))))))));
        D3 s = dd(td, r);
        D3 deriv = dd(ds(dd(dm(dt, z), l2), s), r2);
        p.j00 = dm(dc(fx), da(s, dm(dm(x, x), deriv)));
        p.j01 = dm(dc(fx), dm(dm(x, y), deriv));
        p.j02 = dn(dm(dc(fx), dd(dm(dt, x), l2)));
        p.j10 = dm(dc(fy), dm(dm(x, y), deriv));
        p.j11 = dm(dc(fy), da(s, dm(dm(y, y), deriv)));
        p.j12 = dn(dm(dc(fy), dd(dm(dt, y), l2)));
    }
    return p;
}

Mat3 mat_add(Mat3 a, Mat3 b) {
    Mat3 r;
    [unroll] for (int i = 0; i < 3; ++i)
        [unroll] for (int j = 0; j < 3; ++j) r.m[i][j] = a.m[i][j] + b.m[i][j];
    return r;
}

void cov3d_backward(
    float3 scale_local, float4 rot, Mat3 R, float g0, float g1, float g2,
    float g3, float g4, float g5, out float3 grad_scale, out float4 grad_rotation) {
    Mat3 M = mat_mul(mat_diag(scale_local.x, scale_local.y, scale_local.z), R);
    Mat3 G = mat_zero();
    G.m[0][0] = g0; G.m[0][1] = 0.5f * g1; G.m[0][2] = 0.5f * g2;
    G.m[1][0] = 0.5f * g1; G.m[1][1] = g3; G.m[1][2] = 0.5f * g4;
    G.m[2][0] = 0.5f * g2; G.m[2][1] = 0.5f * g4; G.m[2][2] = g5;
    Mat3 dM = mat_mul(M, G);
    [unroll] for (int i = 0; i < 3; ++i)
        [unroll] for (int j = 0; j < 3; ++j) dM.m[i][j] *= 2.0f;
    Mat3 A = mat_transpose(dM);
    grad_scale = float3(
        R.m[0][0] * A.m[0][0] + R.m[0][1] * A.m[1][0] + R.m[0][2] * A.m[2][0],
        R.m[1][0] * A.m[0][1] + R.m[1][1] * A.m[1][1] + R.m[1][2] * A.m[2][1],
        R.m[2][0] * A.m[0][2] + R.m[2][1] * A.m[1][2] + R.m[2][2] * A.m[2][2]);
    [unroll] for (int row = 0; row < 3; ++row) {
        A.m[row][0] *= scale_local.x;
        A.m[row][1] *= scale_local.y;
        A.m[row][2] *= scale_local.z;
    }
    float w = rot.x, x = rot.y, y = rot.z, z = rot.w;
    float a01 = mat_at(A, 0, 1), a10 = mat_at(A, 1, 0);
    float a02 = mat_at(A, 0, 2), a20 = mat_at(A, 2, 0);
    float a12 = mat_at(A, 1, 2), a21 = mat_at(A, 2, 1);
    float a00 = mat_at(A, 0, 0), a11 = mat_at(A, 1, 1), a22 = mat_at(A, 2, 2);
    grad_rotation = float4(
        2.0f * z * (a01 - a10) + 2.0f * y * (a20 - a02) + 2.0f * x * (a12 - a21),
        2.0f * y * (a10 + a01) + 2.0f * z * (a20 + a02) + 2.0f * w * (a12 - a21) - 4.0f * x * (a22 + a11),
        2.0f * x * (a10 + a01) + 2.0f * w * (a20 - a02) + 2.0f * z * (a12 + a21) - 4.0f * y * (a22 + a00),
        2.0f * w * (a01 - a10) + 2.0f * x * (a20 + a02) + 2.0f * y * (a12 + a21) - 4.0f * z * (a11 + a00));
}

static const float C0 = 0.28209479177387814f;
static const float C1 = 0.4886025119029199f;
static const float C2[5] = {
    1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
    -1.0925484305920792f, 0.5462742152960396f};
static const float C3[7] = {
    -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
    0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
    -0.5900435899266435f};

float3 sh_backward(uint index, uint degree, uint bases, float3 mean, float3 center,
                   float3 color_grad, float3 clamped, uint feature_base) {
    float3 original = mean - center;
    float inv = rsqrt(dot(original, original));
    float3 dir = original * inv;
    float3 dl = color_grad * float3(clamped.x == 0.0f, clamped.y == 0.0f, clamped.z == 0.0f);
    uint source = index * bases * 3u;
    model_grad[feature_base + source] = C0 * dl.x;
    model_grad[feature_base + source + 1u] = C0 * dl.y;
    model_grad[feature_base + source + 2u] = C0 * dl.z;
    float3 dx = 0.0f, dy = 0.0f, dz = 0.0f;
    float x = dir.x, y = dir.y, z = dir.z;
    if (degree > 0u) {
        float e[3] = {-C1 * y, C1 * z, -C1 * x};
        [unroll] for (uint b = 0u; b < 3u; ++b) {
            uint o = feature_base + source + (b + 1u) * 3u;
            model_grad[o] = e[b] * dl.x;
            model_grad[o + 1u] = e[b] * dl.y;
            model_grad[o + 2u] = e[b] * dl.z;
        }
        float3 c1 = float3(color_src[source + 3u], color_src[source + 4u], color_src[source + 5u]);
        float3 c2v = float3(color_src[source + 6u], color_src[source + 7u], color_src[source + 8u]);
        float3 c3v = float3(color_src[source + 9u], color_src[source + 10u], color_src[source + 11u]);
        dx = -C1 * c3v;
        dy = -C1 * c1;
        dz = C1 * c2v;
        if (degree > 1u) {
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float q[5] = {x * y, y * z, 2.0f * zz - xx - yy, x * z, xx - yy};
            float qx[5] = {y, 0.0f, -2.0f * x, z, 2.0f * x};
            float qy[5] = {x, z, -2.0f * y, 0.0f, -2.0f * y};
            float qz[5] = {0.0f, y, 4.0f * z, x, 0.0f};
            [unroll] for (uint j = 0u; j < 5u; ++j) {
                uint so = source + (4u + j) * 3u;
                uint go = feature_base + so;
                float3 c = float3(color_src[so], color_src[so + 1u], color_src[so + 2u]);
                model_grad[go] = C2[j] * q[j] * dl.x;
                model_grad[go + 1u] = C2[j] * q[j] * dl.y;
                model_grad[go + 2u] = C2[j] * q[j] * dl.z;
                dx += C2[j] * qx[j] * c;
                dy += C2[j] * qy[j] * c;
                dz += C2[j] * qz[j] * c;
            }
            if (degree > 2u) {
                float q3[7] = {
                    y * (3.0f * xx - yy), x * y * z, y * (4.0f * zz - xx - yy),
                    z * (2.0f * zz - 3.0f * xx - 3.0f * yy), x * (4.0f * zz - xx - yy),
                    z * (xx - yy), x * (xx - 3.0f * yy)};
                float qx3[7] = {
                    6.0f * x * y, y * z, -2.0f * x * y, -6.0f * x * z,
                    -3.0f * xx + 4.0f * zz - yy, 2.0f * x * z, 3.0f * (xx - yy)};
                float qy3[7] = {
                    3.0f * (xx - yy), x * z, -3.0f * yy + 4.0f * zz - xx, -6.0f * y * z,
                    -2.0f * x * y, -2.0f * y * z, -6.0f * x * y};
                float qz3[7] = {
                    0.0f, x * y, 8.0f * y * z, 3.0f * (2.0f * zz - xx - yy),
                    8.0f * x * z, xx - yy, 0.0f};
                [unroll] for (uint j3 = 0u; j3 < 7u; ++j3) {
                    uint so3 = source + (9u + j3) * 3u;
                    uint go3 = feature_base + so3;
                    float3 cc = float3(
                        color_src[so3], color_src[so3 + 1u], color_src[so3 + 2u]);
                    model_grad[go3] = C3[j3] * q3[j3] * dl.x;
                    model_grad[go3 + 1u] = C3[j3] * q3[j3] * dl.y;
                    model_grad[go3 + 2u] = C3[j3] * q3[j3] * dl.z;
                    dx += C3[j3] * qx3[j3] * cc;
                    dy += C3[j3] * qy3[j3] * cc;
                    dz += C3[j3] * qz3[j3] * cc;
                }
            }
        }
    }
    float3 direction_grad = float3(dot(dx, dl), dot(dy, dl), dot(dz, dl));
    return (direction_grad - dir * dot(dir, direction_grad)) * inv;
}

SplatGeom geometry_eval(
    float3 mean, bool has_scales, float3 scale, float4 rotation,
    float c0, float c1, float c2, float c3, float c4, float c5,
    uint mode, int width, int height, float fx, float fy, float cx, float cy,
    float k1, float k2, float k3, float k4, float kernel,
    float scale_modifier) {
    return project_splat(
        mean, camera[0], camera[1], camera[2], camera[4], camera[5], camera[6],
        camera[8], camera[9], camera[10], camera[12], camera[13], camera[14],
        mode, width, height, fx, fy, cx, cy, k1, k2, k3, k4, kernel,
        scale_modifier, has_scales, scale, rotation, c0, c1, c2, c3, c4, c5);
}

float geometry_loss(SplatGeom value, float4 d_plane, float3 d_normal) {
    if (!value.ok) return 0.0f;
    return dot(value.ray_plane, d_plane) + dot(value.normal, d_normal);
}

void generic_geometry_backward(
    float3 t, Mat3 W, Mat3 V, bool has_scales, float3 scale_local,
    float4 d_plane_raw, float3 d_normal, float fx, float fy,
    out Mat3 dV, out float3 dt) {
    dV = mat_zero();
    dt = 0.0f;
    float4 d_plane = d_plane_raw;
    d_plane.x /= fx;
    d_plane.y /= fy;
    float inv_len = rsqrt(dot(t, t));
    float3 dt_center = t * inv_len * d_plane.z;
    float u = t.x / max(t.z, 1.0e-6f);
    float v = t.y / max(t.z, 1.0e-6f);
    float3 uvh = float3(u, v, 1.0f);

    Mat3 V_inv = mat_zero();
    bool well = true;
    if (has_scales) {
        Mat3 Sinv = mat_diag(1.0f / scale_local.x, 1.0f / scale_local.y, 1.0f / scale_local.z);
        // V = R^T S^2 R, hence V^-1 = (S^-1 R)^T(S^-1 R).
        // Recovering R from V is not stable, so callers replace this matrix
        // below for the scale representation.
    } else {
        Mat3 eig_in = V;
        float e0, e1, e2;
        Mat3 vecs;
        sym_eig3(eig_in, e0, e1, e2, vecs);
        well = min(e0, min(e1, e2)) > 1.0e-8f;
        if (well) {
            V_inv = mat_mul(
                mat_mul(vecs, mat_diag(1.0f / e0, 1.0f / e1, 1.0f / e2)),
                mat_transpose(vecs));
        }
    }
    // The scale path supplies V as positive definite. Its inverse can be
    // formed directly; this also avoids quaternion convention ambiguity.
    if (has_scales) {
        float a = V.m[0][0];
        float b = V.m[0][1];
        float c = V.m[0][2];
        float d = V.m[1][1];
        float e = V.m[1][2];
        float f = V.m[2][2];
        float det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
        V_inv.m[0][0] = (d * f - e * e) / det;
        V_inv.m[0][1] = (c * e - b * f) / det;
        V_inv.m[0][2] = (b * e - c * d) / det;
        V_inv.m[1][0] = V_inv.m[0][1];
        V_inv.m[1][1] = (a * f - c * c) / det;
        V_inv.m[1][2] = (b * c - a * e) / det;
        V_inv.m[2][0] = V_inv.m[0][2];
        V_inv.m[2][1] = V_inv.m[1][2];
        V_inv.m[2][2] = (a * d - b * b) / det;
    }
    Mat3 cov_cam_inv = mat_mul(mat_mul(mat_transpose(W), V_inv), W);
    float3 uvh_m = mat_mul_vec(cov_cam_inv, uvh);
    float vb = dot(uvh_m, uvh);
    float clamp_vb = max(vb, 1.0e-7f);
    float ray_len2 = u * u + v * v + 1.0f;
    float ray_len_inv = rsqrt(ray_len2);
    float length_t = sqrt(dot(t, t));
    float factor = length_t / ray_len2;
    Mat3 nJinv = mat_zero();
    nJinv.m[0][0] = v * v + 1.0f;
    nJinv.m[0][1] = -u * v;
    nJinv.m[0][2] = -u;
    nJinv.m[1][0] = -u * v;
    nJinv.m[1][1] = u * u + 1.0f;
    nJinv.m[1][2] = -v;
    float3 uvh_m_vb = uvh_m / clamp_vb;
    float3 plane = mat_mul_vec(nJinv, uvh_m_vb);
    float3 ray_normal = float3(-plane.x * factor, -plane.y * factor, -1.0f);
    Mat3 nJ = mat_zero();
    float iz = 1.0f / t.z;
    nJ.m[0][0] = iz;
    nJ.m[0][2] = t.x / length_t;
    nJ.m[1][1] = iz;
    nJ.m[1][2] = t.y / length_t;
    nJ.m[2][0] = -t.x * iz * iz;
    nJ.m[2][1] = -t.y * iz * iz;
    nJ.m[2][2] = t.z / length_t;
    float3 cam_normal = mat_mul_vec(nJ, ray_normal);
    float3 nrm = cam_normal * rsqrt(dot(cam_normal, cam_normal));
    // Deliberately match the CUDA/reference convention: the normalization
    // Jacobian is evaluated on the already-normalized vector (factor one).
    float3 d_cam_normal = d_normal - nrm * dot(nrm, d_normal);
    float3 d_ray_normal = mat_mul_vec(mat_transpose(nJ), d_cam_normal);
    Mat3 d_nJ = mat_outer(d_cam_normal, ray_normal);
    float d_factor =
        plane.x * (-d_ray_normal.x + d_plane.x) +
        plane.y * (-d_ray_normal.y + d_plane.y);
    float d_plane_x = (-d_ray_normal.x + d_plane.x) * factor;
    float d_plane_y = (-d_ray_normal.y + d_plane.y) * factor;
    float3 d_plane3 = float3(d_plane_x, d_plane_y, 0.0f);
    float aux = d_plane_x * plane.x + d_plane_y * plane.y;
    float3 W_uvh = mat_mul_vec(W, uvh);
    float3 tmp = mat_mul_vec(mat_transpose(nJinv), d_plane3);
    float3 numerator = mat_mul_vec(cov_cam_inv, tmp) / clamp_vb;
    float3 d_uvh_plane = -2.0f * aux * uvh_m_vb + numerator;
    float rsigma = sqrt(vb / ray_len2);
    float d_len2_x2 = -d_plane.w * rsigma / ray_len2;
    float d_u_sigma = d_len2_x2 * u;
    float d_v_sigma = d_len2_x2 * v;
    float aux_nJ =
        (-mat_at(d_nJ, 2, 0) * u - mat_at(d_nJ, 2, 1) * v - mat_at(d_nJ, 2, 2)) /
        ray_len2 * ray_len_inv;
    float d_u_nJ =
        -mat_at(d_nJ, 0, 2) / t.z + mat_at(d_nJ, 2, 0) * ray_len_inv + aux_nJ * u;
    float d_v_nJ =
        -mat_at(d_nJ, 1, 2) / t.z + mat_at(d_nJ, 2, 1) * ray_len_inv + aux_nJ * v;
    float d_z_nJ =
        (mat_at(d_nJ, 0, 0) + mat_at(d_nJ, 1, 1) -
         mat_at(d_nJ, 0, 2) * u - mat_at(d_nJ, 1, 2) * v) /
        (-t.z * t.z);
    Mat3 d_nJinv = mat_outer(d_plane3, uvh_m_vb);
    float d_u_plane =
        d_uvh_plane.x + (mat_at(d_nJinv, 0, 1) + mat_at(d_nJinv, 1, 0)) * (-v) +
        2.0f * mat_at(d_nJinv, 1, 1) * u - mat_at(d_nJinv, 2, 0);
    float d_v_plane =
        d_uvh_plane.y + (mat_at(d_nJinv, 0, 1) + mat_at(d_nJinv, 1, 0)) * (-u) +
        2.0f * mat_at(d_nJinv, 0, 0) * v - mat_at(d_nJinv, 2, 1);
    float aux_factor = d_factor * (-t.z / ray_len2 * ray_len_inv);
    float d_u = d_u_nJ + d_u_plane + aux_factor * u + d_u_sigma;
    float d_v = d_v_nJ + d_v_plane + aux_factor * v + d_v_sigma;
    float d_z = d_z_nJ + d_factor * ray_len_inv;
    float d_vb_xvb = -aux + d_plane.w * 0.5f * rsigma;
    if (well) {
        float3 rhs = mat_mul_vec(W, tmp) + W_uvh * d_vb_xvb;
        float3 lhs = mat_mul_vec(V_inv, W_uvh);
        float3 vr = mat_mul_vec(V_inv, rhs);
        dV = mat_outer(lhs, -vr);
        [unroll] for (int rr = 0; rr < 3; ++rr)
            [unroll] for (int cc = 0; cc < 3; ++cc) dV.m[rr][cc] /= vb;
    }
    float rz = 1.0f / t.z;
    dt = float3(d_u * rz, d_v * rz, -(d_u * t.x + d_v * t.y) * rz * rz + d_z) + dt_center;
}

[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint count = pc.u0;
    uint index = dispatch_id.x;
    if (index >= count) return;
    uint flags = pc.u1;
    uint mode = pc.u2;
    bool has_sh = (flags & 1u) != 0u;
    bool has_scales = (flags & 2u) != 0u;
    bool raw_chain = (flags & 4u) != 0u;
    // Device-resident training consumes raw parameter gradients only. Its
    // packed output omits the six covariance slots that only apply to models
    // without scale/rotation parameters. The host/debug path keeps the stable
    // legacy layout so its download adapter remains unchanged.
    bool compact_raw_gradients = (flags & 8u) != 0u;
    uint width = pc.u3;
    uint height = pc.u4;
    uint degree = pc.u5;
    uint bases = pc.u6;
    float fx = asfloat(pc.u7);
    float fy = asfloat(pc.u8);
    float cx = asfloat(pc.u9);
    float cy = asfloat(pc.u10);
    float k1 = asfloat(pc.u11);
    float k2 = asfloat(pc.u12);
    float k3 = asfloat(pc.u13);
    float k4 = asfloat(pc.u14);
    float kernel = asfloat(pc.u15);
    float scale_modifier = asfloat(pc.u16);
    uint feature_count = has_sh ? count * bases * 3u : count * 3u;
    uint feature_base = 3u * count;
    uint opacity_base = feature_base + feature_count;
    uint scale_base = opacity_base + count;
    uint rotation_base = scale_base + 3u * count;
    uint covariance_base = rotation_base + 4u * count;
    uint log_scale_base = covariance_base +
        ((compact_raw_gradients && has_scales) ? 0u : 6u * count);
    uint raw_rotation_base = log_scale_base + 3u * count;
    uint logit_base = raw_rotation_base + 4u * count;
    uint refine_base = logit_base + count;

    [unroll] for (uint component = 0u; component < 3u; ++component)
        model_grad[3u * index + component] = 0.0f;
    uint feature_components = has_sh ? bases * 3u : 3u;
    for (uint feature = 0u; feature < feature_components; ++feature)
        model_grad[feature_base + index * feature_components + feature] = 0.0f;
    model_grad[opacity_base + index] = 0.0f;
    [unroll] for (uint scale_component = 0u; scale_component < 3u; ++scale_component) {
        model_grad[scale_base + 3u * index + scale_component] = 0.0f;
        model_grad[log_scale_base + 3u * index + scale_component] = 0.0f;
    }
    [unroll] for (uint rotation_component = 0u; rotation_component < 4u; ++rotation_component) {
        model_grad[rotation_base + 4u * index + rotation_component] = 0.0f;
        model_grad[raw_rotation_base + 4u * index + rotation_component] = 0.0f;
    }
    if (!(compact_raw_gradients && has_scales)) {
        [unroll] for (uint covariance_component = 0u; covariance_component < 6u; ++covariance_component)
            model_grad[covariance_base + 6u * index + covariance_component] = 0.0f;
    }
    model_grad[logit_base + index] = 0.0f;
    model_grad[refine_base + index] = blend_grad[17u * count + index];
    if (gauss_u[3u * count + index] == 0u) return;

    float3 mean = float3(means[3u * index], means[3u * index + 1u], means[3u * index + 2u]);
    float3 t = xform_point(
        mean,
        camera[0], camera[1], camera[2],
        camera[4], camera[5], camera[6],
        camera[8], camera[9], camera[10],
        camera[12], camera[13], camera[14]);
    Mat3 W = world_rotation_transposed(
        camera[0], camera[1], camera[2],
        camera[4], camera[5], camera[6],
        camera[8], camera[9], camera[10]);
    DualProjection dual = projection_jacobian_dual(
        t, mode, int(width), int(height), fx, fy, k1, k2, k3, k4);
    if (!dual.valid) return;
    Mat3 J = mat_zero();
    J.m[0][0] = dual.j00.v;
    J.m[0][1] = dual.j10.v;
    J.m[1][0] = dual.j01.v;
    J.m[1][1] = dual.j11.v;
    J.m[2][0] = dual.j02.v;
    J.m[2][1] = dual.j12.v;
    Mat3 T = mat_mul(W, J);
    Mat3 V = mat_zero();
    Mat3 R = mat_zero();
    float3 scale = 0.0f;
    float3 scale_local = 0.0f;
    float4 rotation = 0.0f;
    float activated_opacity = opacities[index];
    float cv0 = 0.0f, cv1 = 0.0f, cv2 = 0.0f, cv3 = 0.0f, cv4 = 0.0f, cv5 = 0.0f;
    if (has_scales) {
        if (raw_chain) {
            float f2 = filter_3d[index] * filter_3d[index];
            float determinant_ratio = 1.0f;
            [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
                float raw_scale = exp(raw_log_scales[3u * index + axis]);
                scale[axis] = sqrt(raw_scale * raw_scale + f2);
                determinant_ratio *= raw_scale / scale[axis];
            }
            rotation = float4(
                raw_rotations[4u * index], raw_rotations[4u * index + 1u],
                raw_rotations[4u * index + 2u], raw_rotations[4u * index + 3u]);
            rotation *= rsqrt(max(dot(rotation, rotation), 1.0e-20f));
            activated_opacity =
                (1.0f / (1.0f + exp(-opacity_logits[index]))) * determinant_ratio;
        } else {
            scale = float3(scales[3u * index], scales[3u * index + 1u], scales[3u * index + 2u]);
            rotation = float4(
                rotations[4u * index], rotations[4u * index + 1u],
                rotations[4u * index + 2u], rotations[4u * index + 3u]);
        }
        scale_local = scale_modifier * scale;
        R = quat_rotation(rotation);
        Mat3 SR = mat_mul(mat_diag(scale_local.x, scale_local.y, scale_local.z), R);
        V = mat_mul(mat_transpose(SR), SR);
    } else {
        uint cb = 6u * index;
        cv0 = covariances[cb];
        cv1 = covariances[cb + 1u];
        cv2 = covariances[cb + 2u];
        cv3 = covariances[cb + 3u];
        cv4 = covariances[cb + 4u];
        cv5 = covariances[cb + 5u];
        V.m[0][0] = cv0;
        V.m[0][1] = cv1;
        V.m[0][2] = cv2;
        V.m[1][0] = cv1;
        V.m[1][1] = cv3;
        V.m[1][2] = cv4;
        V.m[2][0] = cv2;
        V.m[2][1] = cv4;
        V.m[2][2] = cv5;
    }
    Mat3 cov = mat_mul(mat_transpose(T), mat_mul(V, T));
    uint conic_base = 3u * count + 4u * index;
    float4 gc = float4(
        blend_grad[conic_base], blend_grad[conic_base + 1u],
        blend_grad[conic_base + 2u], blend_grad[conic_base + 3u]);
    uint plane_grad_base = 10u * count + 4u * index;
    uint normal_grad_base = 14u * count + 3u * index;
    float4 gplane = float4(
        blend_grad[plane_grad_base], blend_grad[plane_grad_base + 1u],
        blend_grad[plane_grad_base + 2u], blend_grad[plane_grad_base + 3u]);
    float3 gnormal = float3(
        blend_grad[normal_grad_base],
        blend_grad[normal_grad_base + 1u],
        blend_grad[normal_grad_base + 2u]);
    float a0 = cov.m[0][0];
    float b = cov.m[0][1];
    float c0v = cov.m[1][1];
    float a = a0 + kernel;
    float c = c0v + kernel;
    float denom = a * c - b * b;
    float denom2inv = 1.0f / (denom * denom + 1.0e-7f);
    float ga = denom2inv * (-c * c * gc.x + 2.0f * b * c * gc.y + (denom - a * c) * gc.z);
    float gcc = denom2inv * (-a * a * gc.z + 2.0f * a * b * gc.y + (denom - a * c) * gc.x);
    float gb = denom2inv * 2.0f * (b * c * gc.x - (denom + 2.0f * b * b) * gc.y + a * b * gc.z);
    float det0 = max(1.0e-6f, a0 * c0v - b * b);
    float det1 = max(1.0e-6f, denom);
    float coef = sqrt(det0 / det1);
    if (mode == kModeFisheye) {
        float lc = activated_opacity * gc.w;
        if (a0 * c0v - b * b > 1.0e-6f) {
            float q = lc * 0.5f * coef / det0;
            ga += q * c0v;
            gcc += q * a0;
            gb -= q * 2.0f * b;
        }
        if (denom > 1.0e-6f) {
            float q = -lc * 0.5f * coef / det1;
            ga += q * c;
            gcc += q * a;
            gb -= q * 2.0f * b;
        }
    }
    model_grad[opacity_base + index] = gc.w * coef;
    Mat3 G = mat_zero();
    G.m[0][0] = ga;
    G.m[0][1] = 0.5f * gb;
    G.m[1][0] = 0.5f * gb;
    G.m[1][1] = gcc;
    Mat3 dV = mat_mul(mat_mul(T, G), mat_transpose(T));
    float3 geometry_dt = 0.0f;
    if (mode != kModeFisheye && (any(gplane != 0.0f) || any(gnormal != 0.0f))) {
        Mat3 geometry_dV;
        generic_geometry_backward(
            t, W, V, has_scales, scale_local, gplane, gnormal, fx, fy,
            geometry_dV, geometry_dt);
        dV = mat_add(dV, geometry_dV);
    }
    float gv0 = dV.m[0][0];
    float gv1 = dV.m[0][1] + dV.m[1][0];
    float gv2 = dV.m[0][2] + dV.m[2][0];
    float gv3 = dV.m[1][1];
    float gv4 = dV.m[1][2] + dV.m[2][1];
    float gv5 = dV.m[2][2];
    if (has_scales) {
        float3 gs;
        float4 gr;
        cov3d_backward(scale_local, rotation, R, gv0, gv1, gv2, gv3, gv4, gv5, gs, gr);
        gs *= scale_modifier;
        model_grad[scale_base + 3u * index] = gs.x;
        model_grad[scale_base + 3u * index + 1u] = gs.y;
        model_grad[scale_base + 3u * index + 2u] = gs.z;
        [unroll] for (uint q = 0u; q < 4u; ++q)
            model_grad[rotation_base + 4u * index + q] = gr[q];
    } else {
        uint o = covariance_base + 6u * index;
        model_grad[o] = gv0;
        model_grad[o + 1u] = gv1;
        model_grad[o + 2u] = gv2;
        model_grad[o + 3u] = gv3;
        model_grad[o + 4u] = gv4;
        model_grad[o + 5u] = gv5;
    }
    Mat3 dT = mat_mul(mat_mul(V, T), G);
    [unroll] for (int rr = 0; rr < 3; ++rr)
        [unroll] for (int cc = 0; cc < 3; ++cc) dT.m[rr][cc] *= 2.0f;
    Mat3 dJ = mat_mul(mat_transpose(W), dT);
    float3 gt =
        dJ.m[0][0] * dual.j00.d + dJ.m[1][0] * dual.j01.d + dJ.m[2][0] * dual.j02.d +
        dJ.m[0][1] * dual.j10.d + dJ.m[1][1] * dual.j11.d + dJ.m[2][1] * dual.j12.d +
        geometry_dt;
    uint mean2base = 3u * index;
    float2 gm2 = float2(blend_grad[mean2base], blend_grad[mean2base + 1u]);
    Projection projected;
    if (mode == kModeOrtho) projected = project_ortho(t, fx, fy, cx, cy);
    else projected = project_camera(t, mode, int(width), int(height), fx, fy, cx, cy, k1, k2, k3, k4);
    if (mode == kModePinhole) {
        float rz = 1.0f / (t.z + 1.0e-7f);
        float sx = t.x * rz;
        float sy = t.y * rz;
        gt += float3(
            fx * rz * gm2.x, fy * rz * gm2.y, -(fx * sx * gm2.x + fy * sy * gm2.y) * rz);
    } else if (projected.valid) {
        gt += float3(
            projected.du0 * gm2.x + projected.dv0 * gm2.y,
            projected.du1 * gm2.x + projected.dv1 * gm2.y,
            projected.du2 * gm2.x + projected.dv2 * gm2.y);
    }
    float3 gm = float3(
        camera[0] * gt.x + camera[1] * gt.y + camera[2] * gt.z,
        camera[4] * gt.x + camera[5] * gt.y + camera[6] * gt.z,
        camera[8] * gt.x + camera[9] * gt.y + camera[10] * gt.z);
    uint color_grad_base = 7u * count + 3u * index;
    float3 gcolor = float3(
        blend_grad[color_grad_base],
        blend_grad[color_grad_base + 1u],
        blend_grad[color_grad_base + 2u]);
    if (has_sh) {
        gm += sh_backward(
            index, degree, bases, mean,
            float3(camera[16], camera[17], camera[18]),
            gcolor, gauss_f[8u * index + 6u].xyz, feature_base);
    } else {
        uint fo = feature_base + 3u * index;
        model_grad[fo] = gcolor.x;
        model_grad[fo + 1u] = gcolor.y;
        model_grad[fo + 2u] = gcolor.z;
    }
    model_grad[3u * index] = gm.x;
    model_grad[3u * index + 1u] = gm.y;
    model_grad[3u * index + 2u] = gm.z;
    if (mode == kModeFisheye && (any(gplane != 0.0f) || any(gnormal != 0.0f))) {
        [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
            float eps = 1.0e-3f * max(1.0f, abs(mean[axis]));
            float3 mp = mean;
            float3 mm = mean;
            mp[axis] += eps;
            mm[axis] -= eps;
            SplatGeom gp = geometry_eval(
                mp, has_scales, scale, rotation, cv0, cv1, cv2, cv3, cv4, cv5,
                mode, int(width), int(height), fx, fy, cx, cy, k1, k2, k3, k4,
                kernel, scale_modifier);
            SplatGeom gn = geometry_eval(
                mm, has_scales, scale, rotation, cv0, cv1, cv2, cv3, cv4, cv5,
                mode, int(width), int(height), fx, fy, cx, cy, k1, k2, k3, k4,
                kernel, scale_modifier);
            model_grad[3u * index + axis] +=
                (geometry_loss(gp, gplane, gnormal) - geometry_loss(gn, gplane, gnormal)) /
                (2.0f * eps);
        }
        if (has_scales) {
            [unroll] for (uint axis2 = 0u; axis2 < 3u; ++axis2) {
                float eps = 1.0e-3f * max(0.1f, abs(scale[axis2]));
                float3 sp = scale;
                float3 sm = scale;
                sp[axis2] += eps;
                sm[axis2] -= eps;
                SplatGeom gp = geometry_eval(
                    mean, true, sp, rotation, cv0, cv1, cv2, cv3, cv4, cv5,
                    mode, int(width), int(height), fx, fy, cx, cy, k1, k2, k3, k4,
                    kernel, scale_modifier);
                SplatGeom gn = geometry_eval(
                    mean, true, sm, rotation, cv0, cv1, cv2, cv3, cv4, cv5,
                    mode, int(width), int(height), fx, fy, cx, cy, k1, k2, k3, k4,
                    kernel, scale_modifier);
                model_grad[scale_base + 3u * index + axis2] +=
                    (geometry_loss(gp, gplane, gnormal) - geometry_loss(gn, gplane, gnormal)) /
                    (2.0f * eps);
            }
            [unroll] for (uint qi = 0u; qi < 4u; ++qi) {
                float eps = 1.0e-3f;
                float4 qp = rotation;
                float4 qm = rotation;
                qp[qi] += eps;
                qm[qi] -= eps;
                SplatGeom gp = geometry_eval(
                    mean, true, scale, qp, cv0, cv1, cv2, cv3, cv4, cv5,
                    mode, int(width), int(height), fx, fy, cx, cy, k1, k2, k3, k4,
                    kernel, scale_modifier);
                SplatGeom gn = geometry_eval(
                    mean, true, scale, qm, cv0, cv1, cv2, cv3, cv4, cv5,
                    mode, int(width), int(height), fx, fy, cx, cy, k1, k2, k3, k4,
                    kernel, scale_modifier);
                model_grad[rotation_base + 4u * index + qi] +=
                    (geometry_loss(gp, gplane, gnormal) - geometry_loss(gn, gplane, gnormal)) /
                    (2.0f * eps);
            }
        } else {
            float cv[6] = {cv0, cv1, cv2, cv3, cv4, cv5};
            [unroll] for (uint ci = 0u; ci < 6u; ++ci) {
                float eps = 1.0e-3f * max(0.01f, abs(cv[ci]));
                float cp[6] = {cv0, cv1, cv2, cv3, cv4, cv5};
                float cm[6] = {cv0, cv1, cv2, cv3, cv4, cv5};
                cp[ci] += eps;
                cm[ci] -= eps;
                SplatGeom gp = geometry_eval(
                    mean, false, scale, rotation, cp[0], cp[1], cp[2], cp[3], cp[4], cp[5],
                    mode, int(width), int(height), fx, fy, cx, cy, k1, k2, k3, k4,
                    kernel, scale_modifier);
                SplatGeom gn = geometry_eval(
                    mean, false, scale, rotation, cm[0], cm[1], cm[2], cm[3], cm[4], cm[5],
                    mode, int(width), int(height), fx, fy, cx, cy, k1, k2, k3, k4,
                    kernel, scale_modifier);
                model_grad[covariance_base + 6u * index + ci] +=
                    (geometry_loss(gp, gplane, gnormal) - geometry_loss(gn, gplane, gnormal)) /
                    (2.0f * eps);
            }
        }
    }
    if (raw_chain && has_scales) {
        float f = filter_3d[index];
        float f2 = f * f;
        float gop = model_grad[opacity_base + index];
        [unroll] for (uint ax = 0u; ax < 3u; ++ax) {
            uint o = 3u * index + ax;
            float raw = exp(raw_log_scales[o]);
            float fs = scale[ax];
            model_grad[log_scale_base + o] =
                model_grad[scale_base + o] * raw * raw / fs +
                gop * activated_opacity * f2 / (fs * fs);
        }
        float4 rawq = float4(
            raw_rotations[4u * index], raw_rotations[4u * index + 1u],
            raw_rotations[4u * index + 2u], raw_rotations[4u * index + 3u]);
        float rinv = rsqrt(max(dot(rawq, rawq), 1.0e-20f));
        float4 qn = rawq * rinv;
        float4 gq = float4(
            model_grad[rotation_base + 4u * index],
            model_grad[rotation_base + 4u * index + 1u],
            model_grad[rotation_base + 4u * index + 2u],
            model_grad[rotation_base + 4u * index + 3u]);
        float qdot = dot(qn, gq);
        float4 grq = rinv * (gq - qn * qdot);
        [unroll] for (uint qi = 0u; qi < 4u; ++qi)
            model_grad[raw_rotation_base + 4u * index + qi] = grq[qi];
        float opa = 1.0f / (1.0f + exp(-opacity_logits[index]));
        model_grad[logit_base + index] =
            gop * activated_opacity * (1.0f - opa);
    }
}
