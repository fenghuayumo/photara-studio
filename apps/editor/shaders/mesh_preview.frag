#version 450

layout(location = 0) in vec3 v_cam_normal;
layout(location = 1) in vec3 v_colour;
layout(location = 2) in vec2 v_uv;

layout(location = 0) out vec4 out_colour;

layout(set = 0, binding = 0) uniform sampler2D albedo;

layout(push_constant) uniform Push {
    layout(row_major) mat4 world_to_clip;
    vec4 cam0;
    vec4 cam1;
    vec4 cam2;
    vec4 clay_flags;
} pc;

void main() {
    // Textured mode shows the baked albedo as-is so the mesh preview can be
    // compared against 3DGS/photos without an editor-side shading term. The
    // headlight only shapes untextured clay / vertex-colour previews.
    const bool textured = pc.clay_flags.w > 1.5;
    float light = 1.0;
    if (!textured) {
        vec3 n = v_cam_normal;
        const float len = length(n);
        if (len > 1e-8)
            n /= len;
        if (n.z > 0.0)
            n = -n;
        light = 0.22 + 0.78 * abs(n.z);
    }
    vec3 base = v_colour;
    if (textured)
        base = texture(albedo, v_uv).rgb;
    out_colour = vec4(base * light, 1.0);
}
