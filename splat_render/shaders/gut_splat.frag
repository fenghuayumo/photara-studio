#version 450
#extension GL_GOOGLE_include_directive : require
#include "gut_frame.glsl"

layout(set = 0, binding = 0) uniform FrameBlock { GutFrame frame; } frame_block;

layout(location = 0) flat in vec4 v_color;
layout(location = 1) flat in vec3 v_center;
layout(location = 2) flat in vec3 v_scale;
layout(location = 3) flat in vec3 v_row0;
layout(location = 4) flat in vec3 v_row1;
layout(location = 5) flat in vec3 v_row2;
layout(location = 0) out vec4 out_color;

bool camera_ray(
    vec2 pixel, int model, vec2 resolution, out vec3 origin_cam, out vec3 direction_cam) {
    GutFrame frame = frame_block.frame;
    origin_cam = vec3(0.0);
    if (model == k_model_ortho) {
        origin_cam = vec3(
            (pixel.x - frame.intrinsics.z) / frame.intrinsics.x,
            (pixel.y - frame.intrinsics.w) / frame.intrinsics.y, 0.0);
        direction_cam = vec3(0.0, 0.0, 1.0);
        return true;
    }
    if (model == k_model_equirect) {
        float azimuth = 6.2831853 * (pixel.x / resolution.x - 0.5);
        float elevation = 3.14159265 * (pixel.y / resolution.y - 0.5);
        float cos_el = cos(elevation);
        direction_cam = vec3(
            cos_el * sin(azimuth), sin(elevation), cos_el * cos(azimuth));
        return true;
    }
    if (model == k_model_fisheye) {
        float xn = (pixel.x - frame.intrinsics.z) / frame.intrinsics.x;
        float yn = (pixel.y - frame.intrinsics.w) / frame.intrinsics.y;
        float radius = length(vec2(xn, yn));
        if (radius < 1e-6) {
            direction_cam = vec3(0.0, 0.0, 1.0);
            return true;
        }
        vec4 k = frame.distortion;
        float hi = 1.5707963 - 1e-4;
        float hi2 = hi * hi;
        float hi_poly = 1.0 + hi2 * (k.x + hi2 * (k.y + hi2 * (k.z + hi2 * k.w)));
        if (radius >= hi * hi_poly) return false;
        float theta = min(radius, 0.5 * hi);
        for (int i = 0; i < 8; ++i) {
            float t2 = theta * theta;
            float poly = 1.0 + t2 * (k.x + t2 * (k.y + t2 * (k.z + t2 * k.w)));
            float derivative =
                1.0 + t2 * (3.0 * k.x + t2 * (5.0 * k.y + t2 * (7.0 * k.z + t2 * 9.0 * k.w)));
            float error = theta * poly - radius;
            theta = clamp(theta - error / max(derivative, 1e-6), 0.0, hi);
        }
        float scale = sin(theta) / radius;
        direction_cam = vec3(scale * xn, scale * yn, cos(theta));
        return true;
    }
    direction_cam = normalize(vec3(
        (pixel.x - frame.intrinsics.z) / frame.intrinsics.x,
        (pixel.y - frame.intrinsics.w) / frame.intrinsics.y, 1.0));
    return true;
}

void main() {
    if (v_color.a <= k_alpha_cull) discard;
    GutFrame frame = frame_block.frame;
    vec2 resolution = max(frame.viewport.xy, vec2(1.0));
    int model = int(frame.eye.w + 0.5);
    vec3 origin_cam;
    vec3 dir_cam;
    if (!camera_ray(gl_FragCoord.xy, model, resolution, origin_cam, dir_cam)) discard;
    mat3 rotation = mat3(frame.world_to_camera);
    mat3 camera_to_world = transpose(rotation);
    vec3 direction = normalize(camera_to_world * dir_cam);
    vec3 origin = frame.eye.xyz + camera_to_world * origin_cam;
    vec3 rel = origin - v_center;
    vec3 local_o = vec3(dot(v_row0, rel), dot(v_row1, rel), dot(v_row2, rel)) / v_scale;
    vec3 local_d = vec3(
        dot(v_row0, direction), dot(v_row1, direction), dot(v_row2, direction)) / v_scale;
    float len = length(local_d);
    if (len < 1e-8) discard;
    local_d /= len;
    float dist2 = dot(cross(local_d, local_o), cross(local_d, local_o));
    float response = exp(-0.5 * dist2);
    if (response <= k_kernel_min_response) discard;
    float alpha = min(k_alpha_clamp, response * v_color.a);
    if (alpha <= k_alpha_cull) discard;
    out_color = vec4(v_color.rgb, alpha);
}
