#pragma once
#include "matrix.cuh"
#include "splat_drender/camera.h"

namespace splat_drender::gut {
// Conservative camera-origin halfspaces of the finite alpha-support ellipsoid.
// Count caches the resulting tile rectangle; emit consumes that exact rectangle.
struct SupportBounds {
    float3 c[3], origin;
    float sign, support2;
    CameraIntrinsics k;
    __device__ bool outside(int axis, float pixel, bool lower) const {
        const float slope = (pixel - (axis ? k.cy : k.cx)) / (axis ? k.fy : k.fx);
        const float s = lower ? 1.f : -1.f;
        const float3 n = axis ? make_float3(0, s, -s * slope) : make_float3(s, 0, -s * slope);
        const float3 local = make_float3(sign * mat::dot3(c[0], n), sign * mat::dot3(c[1], n),
                                         sign * mat::dot3(c[2], n));
        float center = -mat::dot3(local, origin);
        center += 1e-3f * (fabsf(local.x * origin.x) + fabsf(local.y * origin.y) +
                           fabsf(local.z * origin.z));
        return center < 0 && center * center > support2 * mat::dot3(local, local) * 1.0001f;
    }
    __device__ int2 axis_bounds(int axis, int extent) const {
        float lo = 0, hi = float(extent - 1), left = lo, right = hi;
        if (outside(axis, lo, true) || outside(axis, hi, false))
            return make_int2(1, 0);
        if (outside(axis, lo, false)) {
            for (int j = 0; j < 12; ++j) {
                float mid = (lo + hi) * .5f;
                if (outside(axis, mid, false))
                    lo = mid;
                else
                    hi = mid;
            }
            left = fmaxf(0, lo - 1);
        }
        lo = 0;
        hi = float(extent - 1);
        if (outside(axis, hi, true)) {
            for (int j = 0; j < 12; ++j) {
                float mid = (lo + hi) * .5f;
                if (outside(axis, mid, true))
                    hi = mid;
                else
                    lo = mid;
            }
            right = fminf(float(extent - 1), hi + 1);
        }
        return make_int2(int(floorf(left / 16)), int(floorf(right / 16)) + 1);
    }
};
__device__ inline float3 support_cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ inline ushort4 pinhole_support_bounds(mat::Mat3 r, float3 scales, float3 delta,
                                                 const CameraView& cam, CameraIntrinsics k,
                                                 float modifier, float opacity, int gx, int gy) {
    ushort4 fallback = make_ushort4(0, gx, 0, gy);
    if (!(k.fx > 0 && k.fy > 0 && modifier > 0) || cam.width < 1 || cam.height < 1 ||
        cam.width >= 65520 || cam.height >= 65520)
        return fallback;
    float3 b[3], a[3];
    SupportBounds s;
    s.k = k;
    for (int i = 0; i < 3; ++i) {
        const float scale = i == 0 ? scales.x : (i == 1 ? scales.y : scales.z);
        const float inv = 1.f / (scale * modifier);
        b[i] = make_float3(r.m[i][0] * inv, r.m[i][1] * inv, r.m[i][2] * inv);
        const auto* w = cam.world_to_camera;
        a[i] = make_float3(b[i].x * w[0] + b[i].y * w[4] + b[i].z * w[8],
                           b[i].x * w[1] + b[i].y * w[5] + b[i].z * w[9],
                           b[i].x * w[2] + b[i].y * w[6] + b[i].z * w[10]);
    }
    float l[3] = {mat::dot3(a[0], a[0]), mat::dot3(a[1], a[1]), mat::dot3(a[2], a[2])};
    if (!(fmaxf(l[0], fmaxf(l[1], l[2])) < 1e8f * fminf(l[0], fminf(l[1], l[2]))))
        return fallback;
    s.c[0] = support_cross(a[1], a[2]);
    s.c[1] = support_cross(a[2], a[0]);
    s.c[2] = support_cross(a[0], a[1]);
    float det = mat::dot3(a[0], s.c[0]);
    if (!isfinite(det) || fabsf(det) < 1e-20f)
        return fallback;
    s.sign = det >= 0 ? 1.f : -1.f;
    s.origin = make_float3(mat::dot3(b[0], delta), mat::dot3(b[1], delta), mat::dot3(b[2], delta));
    s.support2 = fmaxf(0, -2.f * logf((1.f / 255.f) / opacity)) + 1e-3f;
    int2 x = s.axis_bounds(0, cam.width), y = s.axis_bounds(1, cam.height);
    return make_ushort4(x.x, x.y, y.x, y.y);
}
} // namespace splat_drender::gut
