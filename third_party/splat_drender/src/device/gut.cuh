// 3DGUT: unscented projection for binning, exact 3D ray response for blending.
// The binning transform is deliberately outside the derivative (as are tile
// selection and sorting in 3DGS). Derivatives below follow the ray response.
#pragma once
#include "matrix.cuh"
#include "splat_drender/api.h"

namespace splat_drender::gut {

// Forward-mode differentiation of the ten activated geometry parameters.
// This reference CUDA path prioritizes matching forward/backward arithmetic.
struct Dual {
  float v, d[10];
  __device__ Dual(float x = 0.f) : v(x), d{} {}
  __device__ static Dual variable(float x, int k) {
    Dual a(x);
    a.d[k] = 1.f;
    return a;
  }
};
__device__ inline float value(float x) { return x; }
__device__ inline float value(Dual x) { return x.v; }
#define GUT_BINARY(OP, DERIV)                                                  \
  __device__ inline Dual operator OP(Dual a, Dual b) {                         \
    Dual c(a.v OP b.v);                                                        \
    for (int k = 0; k < 10; ++k)                                               \
      c.d[k] = DERIV;                                                          \
    return c;                                                                  \
  }
GUT_BINARY(+, a.d[k] + b.d[k])
GUT_BINARY(-, a.d[k] - b.d[k])
GUT_BINARY(*, a.d[k] * b.v + a.v * b.d[k])
GUT_BINARY(/, (a.d[k] - c.v * b.d[k]) / b.v)
#undef GUT_BINARY
__device__ inline float exponential(float x) { return expf(x); }
__device__ inline Dual exponential(Dual a) {
  Dual b(expf(a.v));
  for (int k = 0; k < 10; ++k)
    b.d[k] = b.v * a.d[k];
  return b;
}

template <class T> struct Response {
  T weight, depth, normal[3];
};

// R maps world directions into Gaussian principal axes, matching matrix.cuh.
template <class T>
__device__ Response<T> response(const T *p, float3 origin, float3 direction,
                                float ray_z, const float *view,
                                float modifier) {
  const T w = p[6], x = p[7], y = p[8], z = p[9];
  T R[3][3] = {{T(1) - T(2) * (y * y + z * z), T(2) * (x * y + w * z),
                T(2) * (x * z - w * y)},
               {T(2) * (x * y - w * z), T(1) - T(2) * (x * x + z * z),
                T(2) * (y * z + w * x)},
               {T(2) * (x * z + w * y), T(2) * (y * z - w * x),
                T(1) - T(2) * (x * x + y * y)}};
  T o[3], d[3], od(0), dd(0);
  for (int a = 0; a < 3; ++a) {
    o[a] = (R[a][0] * (T(origin.x) - p[0]) + R[a][1] * (T(origin.y) - p[1]) +
            R[a][2] * (T(origin.z) - p[2])) /
           (p[3 + a] * T(modifier));
    d[a] = (R[a][0] * T(direction.x) + R[a][1] * T(direction.y) +
            R[a][2] * T(direction.z)) /
           (p[3 + a] * T(modifier));
    od = od + o[a] * d[a];
    dd = dd + d[a] * d[a];
  }
  Response<T> r{};
  const T t = T(0) - od / dd;
  if (!(value(t) > 0.f) || !(value(dd) > 0.f))
    return r;
  T distance(0);
  for (int a = 0; a < 3; ++a) {
    const T e = o[a] + t * d[a];
    distance = distance + e * e;
  }
  r.weight = exponential(T(-.5f) * distance);
  r.depth = t * T(ray_z);
  int axis = 0;
  if (value(p[4]) < value(p[3 + axis]))
    axis = 1;
  if (value(p[5]) < value(p[3 + axis]))
    axis = 2;
  const float facing = value(R[axis][0]) * direction.x +
                       value(R[axis][1]) * direction.y +
                       value(R[axis][2]) * direction.z;
  const T sign(facing > 0.f ? -1.f : 1.f);
  for (int a = 0; a < 3; ++a)
    r.normal[a] =
        sign * (T(view[a]) * R[axis][0] + T(view[4 + a]) * R[axis][1] +
                T(view[8 + a]) * R[axis][2]);
  return r;
}

__device__ inline void parameters(const Gaussians &g, int i, float *p) {
  for (int a = 0; a < 3; ++a) {
    p[a] = g.means[3 * i + a];
    p[3 + a] = g.scales[3 * i + a];
  }
  for (int a = 0; a < 4; ++a)
    p[6 + a] = g.rotations[4 * i + a];
}

__device__ inline void ray(const CameraView &cam, CameraIntrinsics K, int x,
                           int y, float3 &origin, float3 &direction, float &z) {
  float3 d = pixel_unit_ray(float(x), float(y), K);
  z = is_equirect(K.mode) ? 1.f : d.z;
  origin = make_float3(cam.center[0], cam.center[1], cam.center[2]);
  if (is_orthographic(K.mode)) {
    float3 offset = mat::xform_dir_transpose(
        make_float3((x - K.cx) / K.fx, (y - K.cy) / K.fy, 0),
        cam.world_to_camera);
    origin.x += offset.x;
    origin.y += offset.y;
    origin.z += offset.z;
  }
  direction = mat::xform_dir_transpose(d, cam.world_to_camera);
}
} // namespace splat_drender::gut
