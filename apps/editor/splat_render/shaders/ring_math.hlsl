#ifndef RING_MATH_HLSL
#define RING_MATH_HLSL

// std140 layout of splat_render::Renderer::FrameData. Column-major view matrix.
struct SplatFrame {
    [[vk::offset(0)]] float4 world_to_camera[4];
    [[vk::offset(64)]] float4 eye;
    [[vk::offset(80)]] float4 intrinsics;
    [[vk::offset(96)]] float4 distortion;
    [[vk::offset(112)]] float4 viewport;
    [[vk::offset(128)]] uint4 meta;
};

static const float k_pi = 3.14159265;
static const float k_sh_c0 = 0.28209479177387814;
static const float k_sh_c1 = 0.4886025119029199;
static const float k_sh_c2[5] = {
    1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
    -1.0925484305920792, 0.5462742152960396};
static const float k_sh_c3[7] = {
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
    0.3731763325901154, -0.4570457994644658, 1.445305721320277,
    -0.5900435899266435};
static const uint k_model_pinhole = 0;
static const uint k_model_ortho = 1;
static const uint k_model_fisheye = 2;
static const uint k_model_equirect = 3;

float3 transform_point(SplatFrame f, float3 world) {
    float4 p = float4(world, 1.0);
    float4 c0 = f.world_to_camera[0];
    float4 c1 = f.world_to_camera[1];
    float4 c2 = f.world_to_camera[2];
    float4 c3 = f.world_to_camera[3];
    return float3(
        c0.x * p.x + c1.x * p.y + c2.x * p.z + c3.x,
        c0.y * p.x + c1.y * p.y + c2.y * p.z + c3.y,
        c0.z * p.x + c1.z * p.y + c2.z * p.z + c3.z);
}

float3 transform_vector(SplatFrame f, float3 direction) {
    float4 c0 = f.world_to_camera[0];
    float4 c1 = f.world_to_camera[1];
    float4 c2 = f.world_to_camera[2];
    return float3(
        c0.x * direction.x + c1.x * direction.y + c2.x * direction.z,
        c0.y * direction.x + c1.y * direction.y + c2.y * direction.z,
        c0.z * direction.x + c1.z * direction.y + c2.z * direction.z);
}

float load_coeff(ByteAddressBuffer harmonics, uint index) {
    return asfloat(harmonics.Load(index * 4));
}

float3 sh_color(
    ByteAddressBuffer harmonics, uint id, float3 center, uint degree, uint bases,
    float3 eye) {
    uint stride = bases * 3;
    uint base = id * stride;
    float3 rgb = k_sh_c0 * float3(
        load_coeff(harmonics, base),
        load_coeff(harmonics, base + 1),
        load_coeff(harmonics, base + 2));
    float3 dir = center - eye;
    float len = length(dir);
    if (degree == 0 || len < 1e-8) return max(rgb + 0.5, 0.0);
    dir /= len;
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    float3 c1 = float3(
        load_coeff(harmonics, base + 3), load_coeff(harmonics, base + 4),
        load_coeff(harmonics, base + 5));
    float3 c2 = float3(
        load_coeff(harmonics, base + 6), load_coeff(harmonics, base + 7),
        load_coeff(harmonics, base + 8));
    float3 c3 = float3(
        load_coeff(harmonics, base + 9), load_coeff(harmonics, base + 10),
        load_coeff(harmonics, base + 11));
    rgb += k_sh_c1 * (-c1 * y + c2 * z - c3 * x);
    if (degree > 1 && bases >= 9) {
        float xx = x * x;
        float yy = y * y;
        float zz = z * z;
        float xy = x * y;
        float yz = y * z;
        float xz = x * z;
        float q2[5] = {xy, yz, 2.0 * zz - xx - yy, xz, xx - yy};
        [unroll] for (int i = 0; i < 5; ++i) {
            uint o = base + (uint)(4 + i) * 3;
            rgb += k_sh_c2[i] * q2[i] * float3(
                load_coeff(harmonics, o), load_coeff(harmonics, o + 1),
                load_coeff(harmonics, o + 2));
        }
        if (degree > 2 && bases >= 16) {
            float q3[7] = {
                y * (3.0 * xx - yy), xy * z, y * (4.0 * zz - xx - yy),
                z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
                x * (4.0 * zz - xx - yy), z * (xx - yy), x * (xx - 3.0 * yy)};
            [unroll] for (int j = 0; j < 7; ++j) {
                uint o = base + (uint)(9 + j) * 3;
                rgb += k_sh_c3[j] * q3[j] * float3(
                    load_coeff(harmonics, o), load_coeff(harmonics, o + 1),
                    load_coeff(harmonics, o + 2));
            }
        }
    }
    return max(rgb + 0.5, 0.0);
}

// Same screen covariance as splat_drender's project_splat with kernel 0.
// The ellipse edge is the alpha = 1/255 contour of that covariance, which is
// the footprint the EWA forward can paint.
static const float k_near = 0.2;
static const float k_alpha_floor = 1.0 / 255.0;

uint camera_model(SplatFrame f) { return (uint)(f.eye.w + 0.5); }

uint depth_sort_key(float value) {
    uint bits = asuint(value);
    uint mask = (bits & 0x80000000u) != 0u ? 0xffffffffu : 0x80000000u;
    return bits ^ mask;
}

bool camera_in_front(SplatFrame f, float3 camera) {
    uint model = camera_model(f);
    if (model == k_model_equirect) return length(camera) > 1.0e-6;
    if (model == k_model_fisheye) return camera.z > 1.0e-6;
    return camera.z > k_near;
}

// Pixel matches project_pinhole / project_ortho / project_fisheye / project_equirect.
// du, dv are the camera-space derivatives used to build the 2D covariance.
bool screen_jacobian(SplatFrame f, float3 t, out float2 pixel, out float3 du, out float3 dv) {
    pixel = 0.0;
    du = 0.0;
    dv = 0.0;
    float2 resolution = max(f.viewport.xy, float2(1.0, 1.0));
    float fx = f.intrinsics.x;
    float fy = f.intrinsics.y;
    float cx = f.intrinsics.z;
    float cy = f.intrinsics.w;
    uint model = camera_model(f);
    if (model == k_model_ortho) {
        if (!(t.z > k_near)) return false;
        pixel = float2(t.x * fx + cx, t.y * fy + cy);
        du = float3(fx, 0.0, 0.0);
        dv = float3(0.0, fy, 0.0);
        return true;
    }
    if (model == k_model_pinhole) {
        if (!(t.z > k_near)) return false;
        float iz = 1.0 / t.z;
        pixel = float2(t.x * iz * fx + cx, t.y * iz * fy + cy);
        float limx = 1.3 * resolution.x / (2.0 * fx);
        float limy = 1.3 * resolution.y / (2.0 * fy);
        float x = clamp(t.x * iz, -limx, limx) * t.z;
        float y = clamp(t.y * iz, -limy, limy) * t.z;
        float iz2 = iz * iz;
        du = float3(fx * iz, 0.0, -(fx * x) * iz2);
        dv = float3(0.0, fy * iz, -(fy * y) * iz2);
        return true;
    }
    if (model == k_model_equirect) {
        float len = length(t);
        float horiz = length(t.xz);
        if (!(len > 1.0e-6) || !(horiz > 1.0e-5)) return false;
        float azimuth = atan2(t.x, t.z);
        float elevation = atan2(t.y, horiz);
        pixel.x = (azimuth / (2.0 * k_pi) + 0.5) * resolution.x;
        pixel.y = (elevation / k_pi + 0.5) * resolution.y;
        float len2 = len * len;
        float horiz2 = horiz * horiz;
        float su = resolution.x / (2.0 * k_pi);
        float sv = resolution.y / k_pi;
        du = float3(su * t.z / horiz2, 0.0, su * (-t.x) / horiz2);
        dv = float3(
            sv * (-t.y * t.x) / (horiz * len2),
            sv * horiz / len2,
            sv * (-t.y * t.z) / (horiz * len2));
        return true;
    }
    if (!(t.z > 1.0e-6)) return false;
    float radius = length(t.xy);
    float theta = atan2(radius, t.z);
    if (!(theta < 1.57079632679)) return false;
    float theta2 = theta * theta;
    float4 k = f.distortion;
    float poly = 1.0 + theta2 * (k.x + theta2 * (k.y + theta2 * (k.z + theta2 * k.w)));
    float theta_d = theta * poly;
    if (!(theta_d >= 0.0)) return false;
    float radius2 = dot(t.xy, t.xy);
    float len2 = radius2 + t.z * t.z;
    if (radius2 < 1.0e-6 * t.z * t.z) {
        float iz = 1.0 / t.z;
        float q = radius2 * iz * iz;
        float a = k.x - 1.0 / 3.0;
        float b = k.y - k.x + 1.0 / 5.0;
        float s = iz * (1.0 + a * q + b * q * q);
        pixel = float2(cx + fx * t.x * s, cy + fy * t.y * s);
        float ds = 2.0 * iz * iz * iz * (a + 2.0 * b * q);
        float dz = -iz * iz * (1.0 + 3.0 * a * q + 5.0 * b * q * q);
        du = float3(fx * (s + t.x * t.x * ds), fx * t.x * t.y * ds, fx * t.x * dz);
        dv = float3(fy * t.x * t.y * ds, fy * (s + t.y * t.y * ds), fy * t.y * dz);
        return true;
    }
    float rho = theta_d / radius;
    pixel = float2(fx * rho * t.x + cx, fy * rho * t.y + cy);
    float dtheta =
        1.0 + theta2 * (3.0 * k.x + theta2 * (5.0 * k.y + theta2 * (7.0 * k.z + theta2 * 9.0 * k.w)));
    float ds = (dtheta * t.z / len2 - rho) / radius2;
    du = float3(fx * (rho + t.x * t.x * ds), fx * t.x * t.y * ds, -fx * dtheta * t.x / len2);
    dv = float3(fy * t.x * t.y * ds, fy * (rho + t.y * t.y * ds), -fy * dtheta * t.y / len2);
    return true;
}

bool project_pixel(SplatFrame f, float3 world, out float2 pixel) {
    float3 unused_du;
    float3 unused_dv;
    return screen_jacobian(f, transform_point(f, world), pixel, unused_du, unused_dv);
}

bool ellipse_from_axes(
    float2 p0, float2 p1, float2 p2, float cutoff, out float2 axis1, out float2 axis2) {
    axis1 = 0.0;
    axis2 = 0.0;
    float a = dot(float3(p0.x, p1.x, p2.x), float3(p0.x, p1.x, p2.x));
    float b = p0.x * p0.y + p1.x * p1.y + p2.x * p2.y;
    float c = dot(float3(p0.y, p1.y, p2.y), float3(p0.y, p1.y, p2.y));
    float mid = 0.5 * (a + c);
    float radius = length(float2(0.5 * (a - c), b));
    float lambda1 = max(mid + radius, 0.0);
    float lambda2 = max(mid - radius, 0.0);
    float2 eigen = float2(b, lambda1 - a);
    float eigen_len = length(eigen);
    float2 direction = eigen_len > 1e-6 ? eigen / eigen_len : float2(1.0, 0.0);
    float len1 = cutoff * sqrt(lambda1);
    float len2 = cutoff * sqrt(lambda2);
    if (!(len1 > 1.0e-4) || !(len1 < 1.0e6)) return false;
    axis1 = len1 * direction;
    axis2 = len2 * float2(direction.y, -direction.x);
    return true;
}

bool project_axes(
    SplatFrame f, float3 center, float3 row0, float3 row1, float3 row2, float3 scale,
    float opacity, out float2 pixel, out float2 axis1, out float2 axis2) {
    pixel = 0.0;
    axis1 = 0.0;
    axis2 = 0.0;
    float3 camera = transform_point(f, center);
    float3 du;
    float3 dv;
    if (!screen_jacobian(f, camera, pixel, du, dv)) return false;
    float cutoff = 2.0 * log(opacity / k_alpha_floor);
    if (!(cutoff > 0.0)) return false;
    cutoff = sqrt(cutoff);
    float3 world_axis[3] = {scale.x * row0, scale.y * row1, scale.z * row2};
    float2 projected[3];
    [unroll] for (int i = 0; i < 3; ++i) {
        float3 cam = transform_vector(f, world_axis[i]);
        projected[i] = float2(dot(du, cam), dot(dv, cam));
    }
    return ellipse_from_axes(projected[0], projected[1], projected[2], cutoff, axis1, axis2);
}

#endif
