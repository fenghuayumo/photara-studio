#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_colour;
layout(location = 3) in vec2 in_uv;

layout(location = 0) out vec3 v_cam_normal;
layout(location = 1) out vec3 v_colour;
layout(location = 2) out vec2 v_uv;

layout(push_constant) uniform Push {
    layout(row_major) mat4 world_to_clip;
    vec4 cam0;
    vec4 cam1;
    vec4 cam2;
    vec4 clay_flags;
} pc;

void main() {
    gl_Position = pc.world_to_clip * vec4(in_position, 1.0);
    v_cam_normal = vec3(
        dot(pc.cam0.xyz, in_normal),
        dot(pc.cam1.xyz, in_normal),
        dot(pc.cam2.xyz, in_normal));
    v_uv = in_uv;
    v_colour = (pc.clay_flags.w > 0.5 && pc.clay_flags.w < 1.5)
        ? in_colour
        : pc.clay_flags.xyz;
}
