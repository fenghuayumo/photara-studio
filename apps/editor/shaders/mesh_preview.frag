#version 450

layout(location = 0) in vec3 v_cam_normal;
layout(location = 1) in vec3 v_colour;

layout(location = 0) out vec4 out_colour;

void main() {
    vec3 n = v_cam_normal;
    const float len = length(n);
    if (len > 1e-8)
        n /= len;
    if (n.z > 0.0)
        n = -n;
    const float light = 0.22 + 0.78 * abs(n.z);
    out_colour = vec4(v_colour * light, 1.0);
}
