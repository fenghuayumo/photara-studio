#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define RASTER_HD __host__ __device__
#define RASTER_D __device__
#include <cuda.h>
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#else
#define RASTER_HD
#define RASTER_D
#endif

#include <glm/glm.hpp>

// Integer values match aetherscan::CameraModel for pinhole, opencv_fisheye,
// and equirectangular. automatic is never sent to the rasterizer.
enum RasterCameraModel {
    RASTER_PINHOLE = 0,
    RASTER_OPENCV_FISHEYE = 1,
    RASTER_EQUIRECTANGULAR = 3
};

struct RasterIntrinsics {
    float focal_x = 1.f;
    float focal_y = 1.f;
    float center_x = 0.f;
    float center_y = 0.f;
    int model = RASTER_PINHOLE;
    float k1 = 0.f;
    float k2 = 0.f;
    float k3 = 0.f;
    float k4 = 0.f;
};

RASTER_HD inline bool raster_is_equirect(const int model) {
    return model == RASTER_EQUIRECTANGULAR;
}

RASTER_HD inline bool raster_is_fisheye(const int model) {
    return model == RASTER_OPENCV_FISHEYE;
}

RASTER_HD inline bool raster_is_pinhole(const int model) {
    return model != RASTER_OPENCV_FISHEYE && model != RASTER_EQUIRECTANGULAR;
}

RASTER_HD inline float wrap_delta_x(float dx, const int width, const int model) {
    if (!raster_is_equirect(model) || width <= 0)
        return dx;
    const float w = static_cast<float>(width);
    return dx - w * floorf((dx + 0.5f * w) / w);
}

struct RasterProjection {
    float2 mean;
    glm::mat3 J;
    bool valid;
};

RASTER_D inline RasterProjection project_pinhole_camera(
    const float3 t, const RasterIntrinsics& K) {
    RasterProjection out{};
    if (!(t.z > 0.2f))
        return out;
    const float inv_z = 1.f / t.z;
    out.mean = {
        t.x * inv_z * K.focal_x + K.center_x,
        t.y * inv_z * K.focal_y + K.center_y};
    out.J = glm::mat3(
        K.focal_x * inv_z, 0.f, -(K.focal_x * t.x) * inv_z * inv_z,
        0.f, K.focal_y * inv_z, -(K.focal_y * t.y) * inv_z * inv_z,
        0.f, 0.f, 0.f);
    out.valid = true;
    return out;
}

RASTER_D inline RasterProjection project_fisheye_camera(
    const float3 t, const RasterIntrinsics& K) {
    RasterProjection out{};
    if (!(t.z > 0.2f))
        return out;
    const float radius = hypotf(t.x, t.y);
    const float theta = atan2f(radius, t.z);
    constexpr float k_half_pi = 1.57079632679f;
    if (!(theta < k_half_pi - 1e-4f))
        return out;
    const float t2 = theta * theta;
    const float poly =
        1.f + t2 * (K.k1 + t2 * (K.k2 + t2 * (K.k3 + t2 * K.k4)));
    const float theta_d = theta * poly;
    if (theta_d < 0.f)
        return out;
    const float dthetad_dtheta =
        1.f + t2 * (3.f * K.k1 +
                    t2 * (5.f * K.k2 + t2 * (7.f * K.k3 + t2 * 9.f * K.k4)));
    const float R2 = t.x * t.x + t.y * t.y + t.z * t.z;
    float du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz;
    if (radius < 1e-6f) {
        const float inv_z = 1.f / t.z;
        const float scale = dthetad_dtheta * inv_z;
        out.mean = {K.center_x, K.center_y};
        du_dx = K.focal_x * scale;
        du_dy = 0.f;
        du_dz = 0.f;
        dv_dx = 0.f;
        dv_dy = K.focal_y * scale;
        dv_dz = 0.f;
    } else {
        const float rho = theta_d / radius;
        out.mean = {
            K.focal_x * rho * t.x + K.center_x,
            K.focal_y * rho * t.y + K.center_y};
        const float dtheta_dx = t.z * t.x / (R2 * radius);
        const float dtheta_dy = t.z * t.y / (R2 * radius);
        const float dtheta_dz = -radius / R2;
        const float dthetad_dx = dthetad_dtheta * dtheta_dx;
        const float dthetad_dy = dthetad_dtheta * dtheta_dy;
        const float dthetad_dz = dthetad_dtheta * dtheta_dz;
        const float inv_r3 = 1.f / (radius * radius * radius);
        const float drho_dx = dthetad_dx / radius - theta_d * t.x * inv_r3;
        const float drho_dy = dthetad_dy / radius - theta_d * t.y * inv_r3;
        const float drho_dz = dthetad_dz / radius;
        du_dx = K.focal_x * (rho + t.x * drho_dx);
        du_dy = K.focal_x * t.x * drho_dy;
        du_dz = K.focal_x * t.x * drho_dz;
        dv_dx = K.focal_y * t.y * drho_dx;
        dv_dy = K.focal_y * (rho + t.y * drho_dy);
        dv_dz = K.focal_y * t.y * drho_dz;
    }
    out.J = glm::mat3(
        du_dx, du_dy, du_dz,
        dv_dx, dv_dy, dv_dz,
        0.f, 0.f, 0.f);
    out.valid = isfinite(out.mean.x) && isfinite(out.mean.y);
    return out;
}

RASTER_D inline RasterProjection project_equirect_camera(
    const float3 t, const int width, const int height) {
    RasterProjection out{};
    const float length = sqrtf(t.x * t.x + t.y * t.y + t.z * t.z);
    if (!(length > 1e-6f) || width <= 0 || height <= 0)
        return out;
    const float horiz = hypotf(t.x, t.z);
    if (horiz < 1e-5f)
        return out;
    constexpr float k_pi = 3.14159265358979323846f;
    const float azimuth = atan2f(t.x, t.z);
    const float elevation = atan2f(t.y, horiz);
    out.mean = {
        (azimuth / (2.f * k_pi) + 0.5f) * static_cast<float>(width),
        (elevation / k_pi + 0.5f) * static_cast<float>(height)};
    const float R2 = length * length;
    const float horiz2 = horiz * horiz;
    const float daz_dx = t.z / horiz2;
    const float daz_dy = 0.f;
    const float daz_dz = -t.x / horiz2;
    const float del_dx = -t.y * t.x / (horiz * R2);
    const float del_dy = horiz / R2;
    const float del_dz = -t.y * t.z / (horiz * R2);
    const float su = static_cast<float>(width) / (2.f * k_pi);
    const float sv = static_cast<float>(height) / k_pi;
    out.J = glm::mat3(
        su * daz_dx, su * daz_dy, su * daz_dz,
        sv * del_dx, sv * del_dy, sv * del_dz,
        0.f, 0.f, 0.f);
    out.valid = isfinite(out.mean.x) && isfinite(out.mean.y);
    return out;
}

RASTER_D inline RasterProjection project_raster_camera(
    const float3 t, const RasterIntrinsics& K, const int width, const int height) {
    if (raster_is_fisheye(K.model))
        return project_fisheye_camera(t, K);
    if (raster_is_equirect(K.model))
        return project_equirect_camera(t, width, height);
    return project_pinhole_camera(t, K);
}

RASTER_D inline bool camera_visible(
    const float3 t, const int model) {
    if (raster_is_equirect(model))
        return isfinite(t.x) && isfinite(t.y) && isfinite(t.z) &&
               sqrtf(t.x * t.x + t.y * t.y + t.z * t.z) > 1e-6f;
    return t.z > 0.2f;
}

#undef RASTER_HD
#undef RASTER_D
