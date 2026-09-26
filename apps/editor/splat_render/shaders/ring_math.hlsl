#ifndef RING_MATH_HLSL
#define RING_MATH_HLSL

// std140 layout of splat_render::Renderer::FrameData. Column-major view
// matrix, matching the GLSL GutFrame block the 3DGUT shaders already read.
struct GutFrame {
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

float3 transform_point(GutFrame f, float3 world) {
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

float3 transform_vector(GutFrame f, float3 direction) {
    float4 c0 = f.world_to_camera[0];
    float4 c1 = f.world_to_camera[1];
    float4 c2 = f.world_to_camera[2];
    return float3(
        c0.x * direction.x + c1.x * direction.y + c2.x * direction.z,
        c0.y * direction.x + c1.y * direction.y + c2.y * direction.z,
        c0.z * direction.x + c1.z * direction.y + c2.z * direction.z);
}

float wrap_delta(float delta, float period) {
    float x = delta / period;
    x -= floor(x);
    if (x > 0.5) x -= 1.0;
    return x * period;
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

// OpenCV pixel, no screen-margin reject. A ring centred just outside the
// raster still draws when its ellipse overlaps the view.
bool project_pixel(GutFrame f, float3 world, out float2 pixel) {
    float3 camera = transform_point(f, world);
    float2 resolution = max(f.viewport.xy, float2(1.0, 1.0));
    uint model = (uint)(f.eye.w + 0.5);
    float fx = f.intrinsics.x;
    float fy = f.intrinsics.y;
    float cx = f.intrinsics.z;
    float cy = f.intrinsics.w;
    pixel = 0.0;
    if (model == k_model_equirect) {
        float len = length(camera);
        if (!(len > 1e-8)) return false;
        float azimuth = atan2(camera.x, camera.z);
        float elevation = asin(clamp(camera.y / len, -1.0, 1.0));
        pixel.x = (azimuth / (2.0 * k_pi) + 0.5) * resolution.x;
        pixel.y = (elevation / k_pi + 0.5) * resolution.y;
        return true;
    }
    if (model == k_model_fisheye) {
        if (!(camera.z > 1e-8)) return false;
        float radius = length(camera.xy);
        float theta = atan2(radius, camera.z);
        float t2 = theta * theta;
        float4 k = f.distortion;
        float poly = 1.0 + t2 * (k.x + t2 * (k.y + t2 * (k.z + t2 * k.w)));
        float theta_d = theta * poly;
        if (theta_d < 0.0) return false;
        if (radius < 1e-12) {
            pixel = float2(cx, cy);
            return true;
        }
        float scale = theta_d / radius;
        pixel = float2(fx * scale * camera.x + cx, fy * scale * camera.y + cy);
        return true;
    }
    if (model == k_model_ortho) {
        if (!(camera.z > 1e-4)) return false;
        pixel = float2(fx * camera.x + cx, fy * camera.y + cy);
        return true;
    }
    if (!(camera.z > 1e-4)) return false;
    pixel = float2(fx * camera.x / camera.z + cx, fy * camera.y / camera.z + cy);
    return true;
}

// SuperSplat contour. sigma is the inspector value; 2*sqrt(2) puts the quad
// edge where the Gaussian has fallen to exp(-4). 0.3px dilation keeps a
// tiny splat visible, and both axes scale together when the long one hits
// the screen cap.
bool ring_from_axes(
    float2 p0, float2 p1, float2 p2, float contour, float2 resolution,
    out float2 axis1, out float2 axis2) {
    axis1 = 0.0;
    axis2 = 0.0;
    float a = dot(float3(p0.x, p1.x, p2.x), float3(p0.x, p1.x, p2.x)) + 0.3;
    float b = p0.x * p0.y + p1.x * p1.y + p2.x * p2.y;
    float c = dot(float3(p0.y, p1.y, p2.y), float3(p0.y, p1.y, p2.y)) + 0.3;
    float mid = 0.5 * (a + c);
    float radius = length(float2(0.5 * (a - c), b));
    float lambda1 = max(mid + radius, 0.0);
    float lambda2 = max(mid - radius, 0.1);
    float2 eigen = float2(b, lambda1 - a);
    float eigen_len = length(eigen);
    float2 direction = eigen_len > 1e-6 ? eigen / eigen_len : float2(1.0, 0.0);
    float max_radius = min(1024.0, min(resolution.x, resolution.y));
    float len1 = contour * sqrt(lambda1);
    if (!(len1 > 0.05)) return false;
    float fit = min(1.0, max_radius / len1);
    axis1 = len1 * fit * direction;
    axis2 = contour * sqrt(lambda2) * fit * float2(direction.y, -direction.x);
    return true;
}

bool project_axes(
    GutFrame f, float3 center, float3 row0, float3 row1, float3 row2, float3 scale,
    float2 center_px, out float2 axis1, out float2 axis2) {
    axis1 = 0.0;
    axis2 = 0.0;
    float2 resolution = max(f.viewport.xy, float2(1.0, 1.0));
    float contour = f.viewport.z > 0.0 ? f.viewport.z : 2.828427;
    uint model = (uint)(f.eye.w + 0.5);
    float3 world_axis[3] = {scale.x * row0, scale.y * row1, scale.z * row2};
    float2 projected[3];
    if (model == k_model_pinhole || model == k_model_ortho) {
        float3 camera = transform_point(f, center);
        if (!(camera.z > 1e-4)) return false;
        float fx = f.intrinsics.x;
        float fy = f.intrinsics.y;
        [unroll] for (int i = 0; i < 3; ++i) {
            float3 cam = transform_vector(f, world_axis[i]);
            if (model == k_model_ortho) {
                projected[i] = float2(fx * cam.x, fy * cam.y);
            } else {
                float inv_z = 1.0 / camera.z;
                float inv_z2 = inv_z * inv_z;
                projected[i] = float2(
                    fx * (inv_z * cam.x - camera.x * inv_z2 * cam.z),
                    fy * (inv_z * cam.y - camera.y * inv_z2 * cam.z));
            }
        }
    } else {
        [unroll] for (int i = 0; i < 3; ++i) {
            float2 end_px;
            if (!project_pixel(f, center + world_axis[i], end_px)) {
                projected[i] = 0.0;
                continue;
            }
            projected[i] = end_px - center_px;
            if (model == k_model_equirect)
                projected[i].x = wrap_delta(projected[i].x, resolution.x);
        }
    }
    return ring_from_axes(
        projected[0], projected[1], projected[2], contour, resolution, axis1, axis2);
}

#endif
