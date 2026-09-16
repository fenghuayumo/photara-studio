// Native equirectangular projection derivatives for the splat backward pass.
//
// The generic (pinhole-shaped) backward chain differentiates the projected
// covariance through the perspective Jacobian identities: only four of the
// six entries of d(pixel)/d(camera point) are considered live, and their
// position derivatives are the perspective ones. A full panorama is
// different. Its longitude/latitude mapping
//
//   u = (atan2(x, z) / 2pi + 1/2) * width
//   v = (atan2(y, sqrt(x^2 + z^2)) / pi + 1/2) * height
//
// has all six entries live (dv/dx is not zero), and the second derivatives
// of u and v do not match the perspective formulas at all, which is what
// makes the analytic dL/dmean disagree with finite differences.
//
// The projection is therefore differentiated with nested forward-mode dual
// numbers: one evaluation returns both Jacobian rows and their position
// derivatives, so the backward pass can chain dL/dJ to dL/d(mean) exactly.
#pragma once

#include "matrix.cuh"
#include "splat_drender/camera.h"

#include <type_traits>

namespace splat_drender::equirect {

// Forward-mode dual over a fixed 3-vector. Instantiating the value type with
// another Dual yields the second derivatives of the same source expression.
template <class S>
struct Dual {
    S v{};
    S d[3]{};

    Dual() = default;
    // Arithmetic-only, so the copy constructor is never shadowed and nested
    // constants (float -> Dual<float> -> Dual<Dual<float>>) build in one step.
    template <class T, class = typename std::enable_if<
                           std::is_arithmetic<T>::value>::type>
    SD_D2 Dual(T value) : v(S(value)) {}
};

// Variables carry unit first derivatives so every expression above them also
// produces the derivative chain through that axis.
SD_D2 inline Dual<Dual<float>> make_variable(float value, int index) {
    Dual<Dual<float>> x;
    x.v = Dual<float>(value);
    x.v.d[index] = 1.f;
    x.d[index] = Dual<float>(1.f);
    return x;
}

SD_D2 inline float sqrt_(float a) { return sqrtf(a); }
SD_D2 inline float atan2_(float y, float x) { return atan2f(y, x); }

template <class S>
SD_D2 inline Dual<S> operator+(const Dual<S>& a, const Dual<S>& b) {
    Dual<S> c;
    c.v = a.v + b.v;
#pragma unroll
    for (int i = 0; i < 3; ++i) c.d[i] = a.d[i] + b.d[i];
    return c;
}

template <class S>
SD_D2 inline Dual<S> operator-(const Dual<S>& a, const Dual<S>& b) {
    Dual<S> c;
    c.v = a.v - b.v;
#pragma unroll
    for (int i = 0; i < 3; ++i) c.d[i] = a.d[i] - b.d[i];
    return c;
}

template <class S>
SD_D2 inline Dual<S> operator-(const Dual<S>& a) {
    Dual<S> c;
    c.v = S(float(0)) - a.v;
#pragma unroll
    for (int i = 0; i < 3; ++i) c.d[i] = S(float(0)) - a.d[i];
    return c;
}

template <class S>
SD_D2 inline Dual<S> operator*(const Dual<S>& a, const Dual<S>& b) {
    Dual<S> c;
    c.v = a.v * b.v;
#pragma unroll
    for (int i = 0; i < 3; ++i) c.d[i] = a.d[i] * b.v + a.v * b.d[i];
    return c;
}

template <class S>
SD_D2 inline Dual<S> operator/(const Dual<S>& a, const Dual<S>& b) {
    Dual<S> c;
    c.v = a.v / b.v;
#pragma unroll
    for (int i = 0; i < 3; ++i) c.d[i] = (a.d[i] - c.v * b.d[i]) / b.v;
    return c;
}

template <class S>
SD_D2 inline Dual<S> sqrt_(const Dual<S>& a) {
    Dual<S> c;
    c.v = sqrt_(a.v);
    const S half = S(0.5f) / c.v;
#pragma unroll
    for (int i = 0; i < 3; ++i) c.d[i] = a.d[i] * half;
    return c;
}

template <class S>
SD_D2 inline Dual<S> atan2_(const Dual<S>& y, const Dual<S>& x) {
    Dual<S> c;
    c.v = atan2_(y.v, x.v);
    const S denominator = x.v * x.v + y.v * y.v;
#pragma unroll
    for (int i = 0; i < 3; ++i)
        c.d[i] = (x.v * y.d[i] - y.v * x.d[i]) / denominator;
    return c;
}

// Row r of the 2x3 projection Jacobian (pixel row, camera axis) plus its
// derivatives with respect to the camera-space point.
struct Jacobian {
    float j[2][3]{};
    float dj[2][3][3]{};
    bool valid = false;
};

// Exact d(pixel)/d(t) and d^2(pixel)/d(t)^2 of project_equirect, matching the
// validity tests in splat_drender/camera.h so the backward never chains
// through a ray the forward rejected.
SD_D2 inline Jacobian project_jacobian(float3 t, int width, int height) {
    Jacobian out;
    if (width <= 0 || height <= 0) return out;
    const float l2 = t.x * t.x + t.y * t.y + t.z * t.z;
    const float horiz = sqrtf(t.x * t.x + t.z * t.z);
    if (!(l2 > 1.0e-12f) || !(horiz >= 1.0e-5f)) return out;
    if (!isfinite(t.x) || !isfinite(t.y) || !isfinite(t.z)) return out;

    using DD = Dual<Dual<float>>;
    constexpr float kPi = 3.14159265358979323846f;
    const DD x = make_variable(t.x, 0);
    const DD y = make_variable(t.y, 1);
    const DD z = make_variable(t.z, 2);

    const DD horizontal = sqrt_(x * x + z * z);
    const DD u = (atan2_(x, z) / DD(2.f * kPi) + DD(0.5f)) *
                 DD(static_cast<float>(width));
    const DD v = (atan2_(y, horizontal) / DD(kPi) + DD(0.5f)) *
                 DD(static_cast<float>(height));
    const DD rows[2] = {u, v};
#pragma unroll
    for (int row = 0; row < 2; ++row) {
#pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            out.j[row][axis] = rows[row].v.d[axis];
#pragma unroll
            for (int other = 0; other < 3; ++other)
                out.dj[row][axis][other] = rows[row].d[other].d[axis];
        }
    }
    out.valid = isfinite(out.j[0][0]) && isfinite(out.j[0][1]) &&
                isfinite(out.j[0][2]) && isfinite(out.j[1][0]) &&
                isfinite(out.j[1][1]) && isfinite(out.j[1][2]);
    return out;
}

}  // namespace splat_drender::equirect
