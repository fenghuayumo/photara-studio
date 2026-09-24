#version 450
#extension GL_ARB_shader_draw_parameters : require
#extension GL_GOOGLE_include_directive : require
#include "gut_frame.glsl"

layout(set = 0, binding = 0) uniform FrameBlock { GutFrame frame; } frame_block;
layout(set = 0, binding = 5) readonly buffer Quads { vec4 quads[]; };

layout(location = 0) flat out vec4 v_color;
layout(location = 1) flat out vec3 v_center;
layout(location = 2) flat out vec3 v_scale;
layout(location = 3) flat out vec3 v_row0;
layout(location = 4) flat out vec3 v_row1;
layout(location = 5) flat out vec3 v_row2;

const vec2 k_corners[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));

void main() {
    uint base = uint(gl_InstanceIndex) * 7u;
    vec4 color = quads[base];
    vec4 center = quads[base + 1u];
    vec4 box = quads[base + 6u];
    v_color = color;
    v_center = center.xyz;
    v_scale = quads[base + 2u].xyz;
    v_row0 = quads[base + 3u].xyz;
    v_row1 = quads[base + 4u].xyz;
    v_row2 = quads[base + 5u].xyz;
    if (center.w < 0.5) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        return;
    }
    vec2 resolution = max(frame_block.frame.viewport.xy, vec2(1.0));
    vec2 pixel = box.xy + k_corners[gl_VertexIndex] * box.zw;
    vec2 ndc = vec2(
        pixel.x / resolution.x * 2.0 - 1.0,
        pixel.y / resolution.y * 2.0 - 1.0);
    gl_Position = vec4(ndc, 0.5, 1.0);
}
