#version 450

layout(location = 0) in vec3 v_cam_normal;
layout(location = 1) in vec3 v_colour;
layout(location = 2) in vec2 v_uv;

layout(location = 0) out vec4 out_colour;

void main() {
    const vec3 edge = v_colour * 0.18;
    out_colour = vec4(edge, 1.0);
}
