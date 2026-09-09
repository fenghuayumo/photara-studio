#pragma once
#include "camera_model.h"

// Forward-mode differentiation keeps the nonlinear projection, covariance and
// ray geometry on exactly the same expression graph in both CUDA passes.
// Independent variables: camera-space mean (3), world covariance (6).
namespace fisheye_geometry {
struct Dual {
    float v;
    float d[9];
    __device__ Dual(float value = 0.f) : v(value), d{} {}
    __device__ static Dual variable(float value, int index) {
        Dual a(value);
        a.d[index] = 1.f;
        return a;
    }
};
__device__ inline Dual operator+(const Dual &a, const Dual &b) {
    Dual c(a.v + b.v);
    for (int i = 0; i < 9; ++i)
        c.d[i] = a.d[i] + b.d[i];
    return c;
}
__device__ inline Dual operator-(const Dual &a, const Dual &b) {
    Dual c(a.v - b.v);
    for (int i = 0; i < 9; ++i)
        c.d[i] = a.d[i] - b.d[i];
    return c;
}
__device__ inline Dual operator-(const Dual &a) { return Dual(0) - a; }
__device__ inline Dual operator*(const Dual &a, const Dual &b) {
    Dual c(a.v * b.v);
    for (int i = 0; i < 9; ++i)
        c.d[i] = a.d[i] * b.v + a.v * b.d[i];
    return c;
}
__device__ inline Dual operator/(const Dual &a, const Dual &b) {
    Dual c(a.v / b.v);
    for (int i = 0; i < 9; ++i)
        c.d[i] = (a.d[i] - c.v * b.d[i]) / b.v;
    return c;
}
__device__ inline float value(float a) { return a; }
__device__ inline float value(const Dual &a) { return a.v; }
__device__ inline float root(float a) { return sqrtf(a); }
__device__ inline Dual root(const Dual &a) {
    Dual c(sqrtf(a.v));
    for (int i = 0; i < 9; ++i)
        c.d[i] = a.d[i] * 0.5f / c.v;
    return c;
}
__device__ inline float angle(float y, float x) { return atan2f(y, x); }
__device__ inline Dual angle(const Dual &y, const Dual &x) {
    Dual c(atan2f(y.v, x.v));
    for (int i = 0; i < 9; ++i)
        c.d[i] = (x.v * y.d[i] - y.v * x.d[i]) / (x.v * x.v + y.v * y.v);
    return c;
}
template <class S> __device__ S floor_value(S a, float lo) { return value(a) > lo ? a : S(lo); }
template <class S> struct Result {
    S cov[3], conic[3], coefficient, plane[4], normal[3];
};

// W is the transpose of the world-to-camera rotation (GLM convention).
template <class S>
__device__ Result<S> evaluate(const S *t, const S *covariance, const glm::mat3 &W,
                              const RasterIntrinsics &K, float kernel) {
    Result<S> out;
    const S x = t[0], y = t[1], z = t[2];
    const S r2 = x * x + y * y, l2 = r2 + z * z, length = root(l2);
    S J[2][3];
    // Taylor expansion avoids 0/0 and cancellation at the optical axis;
    // unlike a constant center branch it preserves first/second derivatives.
    if (value(r2) < 1e-6f * value(z * z)) {
        const S iz = S(1) / z, q = r2 * iz * iz;
        const float a = K.k1 - 1.f / 3.f;
        const float b = K.k2 - K.k1 + 1.f / 5.f;
        const S s = iz * (S(1) + S(a) * q + S(b) * q * q);
        const S ds = S(2) * iz * iz * iz * (S(a) + S(2 * b) * q);
        const S dz = -iz * iz * (S(1) + S(3 * a) * q + S(5 * b) * q * q);
        J[0][0] = S(K.focal_x) * (s + x * x * ds);
        J[0][1] = S(K.focal_x) * x * y * ds;
        J[0][2] = S(K.focal_x) * x * dz;
        J[1][0] = S(K.focal_y) * x * y * ds;
        J[1][1] = S(K.focal_y) * (s + y * y * ds);
        J[1][2] = S(K.focal_y) * y * dz;
    } else {
        const S r = root(r2), theta = angle(r, z), q = theta * theta;
        const S td = theta * (S(1) + q * (S(K.k1) + q * (S(K.k2) + q * (S(K.k3) + q * S(K.k4)))));
        const S dt =
            S(1) + q * (S(3 * K.k1) + q * (S(5 * K.k2) + q * (S(7 * K.k3) + q * S(9 * K.k4))));
        const S s = td / r, ds = (dt * z / l2 - s) / r2;
        J[0][0] = S(K.focal_x) * (s + x * x * ds);
        J[0][1] = S(K.focal_x) * x * y * ds;
        J[0][2] = -S(K.focal_x) * dt * x / l2;
        J[1][0] = S(K.focal_y) * x * y * ds;
        J[1][1] = S(K.focal_y) * (s + y * y * ds);
        J[1][2] = -S(K.focal_y) * dt * y / l2;
    }
    // Expand the six unique covariance entries explicitly. This also avoids
    // nested, dynamically indexed aggregate accumulation in CUDA AD code.
    S C[3][3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            C[i][j] = covariance[0] * S(W[i][0] * W[j][0]) +
                      covariance[1] * S(W[i][0] * W[j][1] + W[i][1] * W[j][0]) +
                      covariance[2] * S(W[i][0] * W[j][2] + W[i][2] * W[j][0]) +
                      covariance[3] * S(W[i][1] * W[j][1]) +
                      covariance[4] * S(W[i][1] * W[j][2] + W[i][2] * W[j][1]) +
                      covariance[5] * S(W[i][2] * W[j][2]);
        }
    }
    S a(0), b(0), c(0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            a = a + J[0][i] * C[i][j] * J[0][j];
            b = b + J[0][i] * C[i][j] * J[1][j];
            c = c + J[1][i] * C[i][j] * J[1][j];
        }
    const S det0 = floor_value(a * c - b * b, 1e-6f);
    a = a + S(kernel);
    c = c + S(kernel);
    const S det = a * c - b * b;
    out.cov[0] = a;
    out.cov[1] = b;
    out.cov[2] = c;
    out.conic[0] = c / det;
    out.conic[1] = -b / det;
    out.conic[2] = a / det;
    out.coefficient = root(det0 / floor_value(det, 1e-6f));

    // Precision matrix from the adjugate of camera-space covariance.
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
    for (int i = 0; i < 3; ++i) {
        qv[i] = S(0);
        for (int j = 0; j < 3; ++j)
            qv[i] = qv[i] + A[i][j] * n[j] / determinant;
        vb = vb + n[i] * qv[i];
        qlen = qlen + qv[i] * qv[i];
    }
    qlen = root(qlen);
    for (int i = 0; i < 3; ++i)
        out.normal[i] = -qv[i] / qlen;
    // J maps camera displacements to pixels and annihilates the center ray.
    // J^T (J J^T)^-1 is therefore the inverse on its tangent plane.
    // For d = projected_mean - pixel, the radial peak slope is
    // (J^+)^T (C^-1 n) / (n^T C^-1 n). The ray-length factors cancel.
    S g00(0), g01(0), g11(0), qj0(0), qj1(0);
    for (int i = 0; i < 3; ++i) {
        g00 = g00 + J[0][i] * J[0][i];
        g01 = g01 + J[0][i] * J[1][i];
        g11 = g11 + J[1][i] * J[1][i];
        qj0 = qj0 + qv[i] * J[0][i];
        qj1 = qj1 + qv[i] * J[1][i];
    }
    const S gd = g00 * g11 - g01 * g01;
    out.plane[0] = (qj0 * g11 - qj1 * g01) / (vb * gd);
    out.plane[1] = (qj1 * g00 - qj0 * g01) / (vb * gd);
    out.plane[2] = length;
    out.plane[3] = root(vb);
    return out;
}
} // namespace fisheye_geometry
