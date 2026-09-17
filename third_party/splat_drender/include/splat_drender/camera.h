// Camera models used by the rasterizer: pinhole, OpenCV fisheye (k1..k4),
// and equirectangular. Projection returns the pixel-space mean plus the
// 2x3 Jacobian d(pixel)/d(camera point), which drives EWA splatting.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define SD_HD __host__ __device__
#define SD_D  __device__
#else
#define SD_HD
#define SD_D
#endif

namespace splat_drender {

// Integer values match aetherscan::CameraModel for the models the
// rasterizer understands.
enum class CameraMode : int {
    pinhole = 0,
    opencv_fisheye = 1,
    equirectangular = 3,
};

struct CameraIntrinsics {
    float fx = 1.f, fy = 1.f;
    float cx = 0.f, cy = 0.f;
    CameraMode mode = CameraMode::pinhole;
    float k1 = 0.f, k2 = 0.f, k3 = 0.f, k4 = 0.f;
};

SD_HD inline bool is_equirect(CameraMode m) { return m == CameraMode::equirectangular; }
SD_HD inline bool is_fisheye(CameraMode m) { return m == CameraMode::opencv_fisheye; }
SD_HD inline bool is_pinhole(CameraMode m) { return !is_fisheye(m) && !is_equirect(m); }

// Horizontal wrap-around used by equirectangular images: map dx into
// [-w/2, w/2) so instances on either seam edge hit the correct pixels.
SD_HD inline float wrap_dx(float dx, int width, CameraMode m) {
    if (!is_equirect(m) || width <= 0) return dx;
    const float w = static_cast<float>(width);
    return dx - w * floorf((dx + 0.5f * w) / w);
}

// Row-major 2x3 Jacobian stored as three rows; only the first two rows
// are non-zero for all supported models.
struct ProjectionJacobian {
    float du[3];
    float dv[3];
};

struct Projection {
    float pixel_x = 0.f, pixel_y = 0.f;
    ProjectionJacobian J{};
    bool valid = false;
};

SD_HD inline Projection project_pinhole(float3 t, const CameraIntrinsics& K) {
    Projection p;
    if (!(t.z > 0.2f)) return p;
    const float iz = 1.f / t.z;
    p.pixel_x = t.x * iz * K.fx + K.cx;
    p.pixel_y = t.y * iz * K.fy + K.cy;
    p.J.du[0] = K.fx * iz;  p.J.du[1] = 0.f;  p.J.du[2] = -(K.fx * t.x) * iz * iz;
    p.J.dv[0] = 0.f;        p.J.dv[1] = K.fy * iz;  p.J.dv[2] = -(K.fy * t.y) * iz * iz;
    p.valid = true;
    return p;
}

SD_HD inline Projection project_fisheye(float3 t, const CameraIntrinsics& K) {
    // OpenCV fisheye: theta_d(theta) = theta (1 + k1 th^2 + ... + k4 th^8),
    // pixel = focal * theta_d * (x, y) / r.
    Projection p;
    if (!(t.z > 1.0e-6f)) return p;
    const float r = hypotf(t.x, t.y);
    const float theta = atan2f(r, t.z);
    constexpr float kHalfPi = 1.57079632679f;
    if (!(theta < kHalfPi)) return p;
    const float th2 = theta * theta;
    const float poly = 1.f + th2 * (K.k1 + th2 * (K.k2 + th2 * (K.k3 + th2 * K.k4)));
    const float theta_d = theta * poly;
    if (!(theta_d >= 0.f)) return p;
    const float dtheta_d = 1.f + th2 * (3.f * K.k1 + th2 * (5.f * K.k2 + th2 * (7.f * K.k3 + th2 * 9.f * K.k4)));
    if (!(dtheta_d > 1.0e-6f)) return p;

    const float r2 = t.x * t.x + t.y * t.y;
    const float l2 = r2 + t.z * t.z;
    float du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz;
    if (r2 < 1.0e-6f * t.z * t.z) {
        // Taylor branch keeps the Jacobian finite and smooth on the optical
        // axis instead of switching to an unrelated constant.
        const float iz = 1.f / t.z;
        const float q = r2 * iz * iz;
        const float a = K.k1 - 1.f / 3.f;
        const float b = K.k2 - K.k1 + 1.f / 5.f;
        const float s = iz * (1.f + a * q + b * q * q);
        const float ds = 2.f * iz * iz * iz * (a + 2.f * b * q);
        const float dz = -iz * iz * (1.f + 3.f * a * q + 5.f * b * q * q);
        p.pixel_x = K.cx + K.fx * t.x * s;
        p.pixel_y = K.cy + K.fy * t.y * s;
        du_dx = K.fx * (s + t.x * t.x * ds); du_dy = K.fx * t.x * t.y * ds; du_dz = K.fx * t.x * dz;
        dv_dx = K.fy * t.x * t.y * ds;       dv_dy = K.fy * (s + t.y * t.y * ds); dv_dz = K.fy * t.y * dz;
    } else {
        const float rho = theta_d / r;
        p.pixel_x = K.fx * rho * t.x + K.cx;
        p.pixel_y = K.fy * rho * t.y + K.cy;
        const float dth_dx = t.z * t.x / (l2 * r);
        const float dth_dy = t.z * t.y / (l2 * r);
        const float dth_dz = -r / l2;
        const float dtd_dx = dtheta_d * dth_dx;
        const float dtd_dy = dtheta_d * dth_dy;
        const float dtd_dz = dtheta_d * dth_dz;
        const float ir3 = 1.f / (r * r * r);
        const float drho_dx = dtd_dx / r - theta_d * t.x * ir3;
        const float drho_dy = dtd_dy / r - theta_d * t.y * ir3;
        const float drho_dz = dtd_dz / r;
        du_dx = K.fx * (rho + t.x * drho_dx); du_dy = K.fx * t.x * drho_dy; du_dz = K.fx * t.x * drho_dz;
        dv_dx = K.fy * t.y * drho_dx;         dv_dy = K.fy * (rho + t.y * drho_dy); dv_dz = K.fy * t.y * drho_dz;
    }
    p.J.du[0] = du_dx; p.J.du[1] = du_dy; p.J.du[2] = du_dz;
    p.J.dv[0] = dv_dx; p.J.dv[1] = dv_dy; p.J.dv[2] = dv_dz;
    p.valid = isfinite(p.pixel_x) && isfinite(p.pixel_y);
    return p;
}

SD_HD inline Projection project_equirect(float3 t, int width, int height) {
    Projection p;
    const float l = sqrtf(t.x * t.x + t.y * t.y + t.z * t.z);
    if (!(l > 1.0e-6f) || width <= 0 || height <= 0) return p;
    const float horiz = hypotf(t.x, t.z);
    if (horiz < 1.0e-5f) return p;
    constexpr float kPi = 3.14159265358979323846f;
    const float az = atan2f(t.x, t.z);
    const float el = atan2f(t.y, horiz);
    p.pixel_x = (az / (2.f * kPi) + 0.5f) * static_cast<float>(width);
    p.pixel_y = (el / kPi + 0.5f) * static_cast<float>(height);
    const float l2 = l * l;
    const float h2 = horiz * horiz;
    const float daz_dx = t.z / h2, daz_dy = 0.f, daz_dz = -t.x / h2;
    const float del_dx = -t.y * t.x / (horiz * l2);
    const float del_dy = horiz / l2;
    const float del_dz = -t.y * t.z / (horiz * l2);
    const float su = static_cast<float>(width) / (2.f * kPi);
    const float sv = static_cast<float>(height) / kPi;
    p.J.du[0] = su * daz_dx; p.J.du[1] = su * daz_dy; p.J.du[2] = su * daz_dz;
    p.J.dv[0] = sv * del_dx; p.J.dv[1] = sv * del_dy; p.J.dv[2] = sv * del_dz;
    p.valid = isfinite(p.pixel_x) && isfinite(p.pixel_y);
    return p;
}

SD_HD inline Projection project(float3 t, const CameraIntrinsics& K, int width, int height) {
    if (is_fisheye(K.mode)) return project_fisheye(t, K);
    if (is_equirect(K.mode)) return project_equirect(t, width, height);
    return project_pinhole(t, K);
}

SD_HD inline bool camera_visible(float3 t, CameraMode m) {
    if (is_equirect(m))
        return isfinite(t.x) && isfinite(t.y) && isfinite(t.z) &&
               sqrtf(t.x * t.x + t.y * t.y + t.z * t.z) > 1.0e-6f;
    return t.z > (is_fisheye(m) ? 1.0e-6f : 0.2f);
}

#if defined(__CUDACC__)
// Unit ray through a pixel; .z converts a ray-length depth to camera Z.
// Device-only (uses rnorm3df and the inverse fisheye solve).
SD_D inline float3 pixel_unit_ray(float px, float py, const CameraIntrinsics& K) {
    const float x = (px - K.cx) / K.fx;
    const float y = (py - K.cy) / K.fy;
    if (is_equirect(K.mode)) {
        // The panorama convention (fx = fy = width / 2pi, cx = width / 2)
        // makes those offsets the azimuth and elevation in radians, so the ray
        // is the spherical direction rather than a perspective (x, y, 1).
        const float cos_elevation = cosf(y);
        return make_float3(cos_elevation * sinf(x), sinf(y),
                           cos_elevation * cosf(x));
    }
    if (!is_fisheye(K.mode)) {
        const float inv = rnorm3df(x, y, 1.f);
        return make_float3(x * inv, y * inv, inv);
    }
    const float r = hypotf(x, y);
    float lo = 0.f, hi = 1.57079632679f;
    for (int i = 0; i < 28; ++i) {
        const float t = 0.5f * (lo + hi), q = t * t;
        const float td = t * (1.f + q * (K.k1 + q * (K.k2 + q * (K.k3 + q * K.k4))));
        if (td < r) lo = t; else hi = t;
    }
    const float theta = 0.5f * (lo + hi);
    const float scale = r > 1.0e-8f ? sinf(theta) / r : 1.f;
    return make_float3(x * scale, y * scale, fmaxf(0.f, cosf(theta)));
}

SD_D inline float pixel_ray_z(float px, float py, const CameraIntrinsics& K) {
    // The rasterizer accumulates |t|, the distance along the ray. Pinhole and
    // fisheye convert that to camera-space Z, but a panorama has no global Z
    // axis: the distance along the ray is already its depth.
    if (is_equirect(K.mode)) return 1.f;
    return pixel_unit_ray(px, py, K).z;
}
#endif  // __CUDACC__

}  // namespace splat_drender

#undef SD_HD
#undef SD_D
