// Native fisheye splat geometry with forward-mode differentiation.
//
// One templated evaluation produces the projected covariance, conic,
// anti-aliased opacity coefficient, depth ray-plane and footprint normal
// for both the forward pass (float) and the backward pass (Dual numbers
// over camera position + world covariance, 9 partials).
#pragma once

#include "matrix.cuh"
#include "splat_drender/camera.h"

namespace splat_drender::fisheye {

struct Dual {
    float v = 0.f;
    float d[9]{};

    Dual() = default;
    SD_D2 Dual(float value) : v(value) {}
    SD_D2 static Dual variable(float value, int idx) {
        Dual x(value);
        x.d[idx] = 1.f;
        return x;
    }
};

SD_D2 inline Dual operator+(Dual a, Dual b) {
    Dual c(a.v + b.v);
#pragma unroll
    for (int i = 0; i < 9; ++i) c.d[i] = a.d[i] + b.d[i];
    return c;
}
SD_D2 inline Dual operator-(Dual a, Dual b) {
    Dual c(a.v - b.v);
#pragma unroll
    for (int i = 0; i < 9; ++i) c.d[i] = a.d[i] - b.d[i];
    return c;
}
SD_D2 inline Dual operator-(Dual a) { return Dual(0.f) - a; }
SD_D2 inline Dual operator*(Dual a, Dual b) {
    Dual c(a.v * b.v);
#pragma unroll
    for (int i = 0; i < 9; ++i) c.d[i] = a.d[i] * b.v + a.v * b.d[i];
    return c;
}
SD_D2 inline Dual operator*(float k, Dual a) { return Dual(k) * a; }
SD_D2 inline Dual operator/(Dual a, Dual b) {
    Dual c(a.v / b.v);
#pragma unroll
    for (int i = 0; i < 9; ++i) c.d[i] = (a.d[i] - c.v * b.d[i]) / b.v;
    return c;
}

SD_D2 inline float scalar(float a) { return a; }
SD_D2 inline float scalar(const Dual& a) { return a.v; }
SD_D2 inline float sqrt_(float a) { return sqrtf(a); }
SD_D2 inline Dual sqrt_(const Dual& a) {
    Dual c(sqrtf(a.v));
#pragma unroll
    for (int i = 0; i < 9; ++i) c.d[i] = a.d[i] * 0.5f / c.v;
    return c;
}
SD_D2 inline float atan2_(float y, float x) { return atan2f(y, x); }
SD_D2 inline Dual atan2_(const Dual& y, const Dual& x) {
    Dual c(atan2f(y.v, x.v));
#pragma unroll
    for (int i = 0; i < 9; ++i)
        c.d[i] = (x.v * y.d[i] - y.v * x.d[i]) / (x.v * x.v + y.v * y.v);
    return c;
}
template <class S>
SD_D2 S clamp_min(S a, float lo) { return scalar(a) > lo ? a : S(lo); }

template <class S>
struct Result {
    S cov[3], conic[3], coefficient, plane[4], normal[3];
};

// `t` = camera-space Gaussian mean, `covariance` = world upper-triangular
// covariance, `W` = world-from-camera rotation (rows). Independent
// variables for the Dual instantiation: t (0..2), covariance (3..8).
template <class S>
SD_D2 Result<S> evaluate(const S* t, const S* covariance, const mat::Mat3& W,
                         const CameraIntrinsics& K, float kernel) {
    Result<S> out;
    const S x = t[0], y = t[1], z = t[2];
    const S r2 = x * x + y * y, l2 = r2 + z * z, length = sqrt_(l2);

    S j00, j01, j02, j10, j11, j12;
    if (scalar(r2) < 1.0e-6f * scalar(z * z)) {
        // Taylor branch on the optical axis; preserves 1st/2nd derivatives.
        const S iz = S(1) / z, q = r2 * iz * iz;
        const float a = K.k1 - 1.f / 3.f;
        const float b = K.k2 - K.k1 + 1.f / 5.f;
        const S s = iz * (S(1) + S(a) * q + S(b) * q * q);
        const S ds = S(2) * iz * iz * iz * (S(a) + S(2 * b) * q);
        const S dz = -iz * iz * (S(1) + S(3 * a) * q + S(5 * b) * q * q);
        j00 = S(K.fx) * (s + x * x * ds); j01 = S(K.fx) * x * y * ds; j02 = S(K.fx) * x * dz;
        j10 = S(K.fy) * x * y * ds;       j11 = S(K.fy) * (s + y * y * ds); j12 = S(K.fy) * y * dz;
    } else {
        const S r = sqrt_(r2), theta = atan2_(r, z), q = theta * theta;
        const S td = theta * (S(1) + q * (S(K.k1) + q * (S(K.k2) + q * (S(K.k3) + q * S(K.k4)))));
        const S dt = S(1) + q * (S(3 * K.k1) + q * (S(5 * K.k2) + q * (S(7 * K.k3) + q * S(9 * K.k4))));
        const S s = td / r, ds = (dt * z / l2 - s) / r2;
        j00 = S(K.fx) * (s + x * x * ds); j01 = S(K.fx) * x * y * ds; j02 = -S(K.fx) * dt * x / l2;
        j10 = S(K.fy) * x * y * ds;       j11 = S(K.fy) * (s + y * y * ds); j12 = -S(K.fy) * dt * y / l2;
    }

    // Camera-space covariance C = W^T Sigma W expanded over 6 entries.
    S C[3][3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            C[i][j] = covariance[0] * S(W.at(i, 0) * W.at(j, 0)) +
                      covariance[1] * S(W.at(i, 0) * W.at(j, 1) + W.at(i, 1) * W.at(j, 0)) +
                      covariance[2] * S(W.at(i, 0) * W.at(j, 2) + W.at(i, 2) * W.at(j, 0)) +
                      covariance[3] * S(W.at(i, 1) * W.at(j, 1)) +
                      covariance[4] * S(W.at(i, 1) * W.at(j, 2) + W.at(i, 2) * W.at(j, 1)) +
                      covariance[5] * S(W.at(i, 2) * W.at(j, 2));
        }
    }

    S a(0), b(0), c(0);
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const S j0 = (i == 0) ? j00 : ((i == 1) ? j01 : j02);
            const S j1 = (i == 0) ? j10 : ((i == 1) ? j11 : j12);
            a = a + j0 * C[i][j] * ((j == 0) ? j00 : ((j == 1) ? j01 : j02));
            b = b + j0 * C[i][j] * ((j == 0) ? j10 : ((j == 1) ? j11 : j12));
            c = c + j1 * C[i][j] * ((j == 0) ? j10 : ((j == 1) ? j11 : j12));
        }

    const S det0 = clamp_min(a * c - b * b, 1.0e-6f);
    a = a + S(kernel);
    c = c + S(kernel);
    const S det = a * c - b * b;
    out.cov[0] = a;
    out.cov[1] = b;
    out.cov[2] = c;
    out.conic[0] = c / det;
    out.conic[1] = -b / det;
    out.conic[2] = a / det;
    out.coefficient = sqrt_(det0 / clamp_min(det, 1.0e-6f));

    // Precision matrix from the adjugate of C drives the footprint normal
    // and the ray-plane (peak depth along the projected ray).
    S A[3][3];
    A[0][0] = C[1][1] * C[2][2] - C[1][2] * C[2][1];
    A[0][1] = C[0][2] * C[2][1] - C[0][1] * C[2][2];
    A[0][2] = C[0][1] * C[1][2] - C[0][2] * C[1][1];
    A[1][0] = A[0][1];
    A[1][1] = C[0][0] * C[2][2] - C[0][2] * C[2][0];
    A[1][2] = C[0][2] * C[1][0] - C[0][0] * C[1][2];
    A[2][0] = A[0][2];
    A[2][1] = A[1][2];
    A[2][2] = C[0][0] * C[1][1] - C[0][1] * C[1][0];
    const S determinant = C[0][0] * A[0][0] + C[0][1] * A[0][1] + C[0][2] * A[0][2];

    S n[3] = {x / length, y / length, z / length}, qv[3], vb(0), qlen(0);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        qv[i] = S(0);
#pragma unroll
        for (int j = 0; j < 3; ++j) qv[i] = qv[i] + A[i][j] * n[j] / determinant;
        vb = vb + n[i] * qv[i];
        qlen = qlen + qv[i] * qv[i];
    }
    qlen = sqrt_(qlen);
#pragma unroll
    for (int i = 0; i < 3; ++i) out.normal[i] = -qv[i] / qlen;

    // J annihilates the center ray, so J^+ (J J^T)^-1 inverts it on the
    // tangent plane: the radial peak-depth slope is (J^+)^T (C^-1 n) /
    // (n^T C^-1 n); ray-length factors cancel.
    S g00(0), g01(0), g11(0), qj0(0), qj1(0);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const S j0 = (i == 0) ? j00 : ((i == 1) ? j01 : j02);
        const S j1 = (i == 0) ? j10 : ((i == 1) ? j11 : j12);
        g00 = g00 + j0 * j0;
        g01 = g01 + j0 * j1;
        g11 = g11 + j1 * j1;
        qj0 = qj0 + qv[i] * j0;
        qj1 = qj1 + qv[i] * j1;
    }
    const S gd = g00 * g11 - g01 * g01;
    out.plane[0] = (qj0 * g11 - qj1 * g01) / (vb * gd);
    out.plane[1] = (qj1 * g00 - qj0 * g01) / (vb * gd);
    out.plane[2] = length;
    out.plane[3] = sqrt_(vb);
    return out;
}

}  // namespace splat_drender::fisheye
