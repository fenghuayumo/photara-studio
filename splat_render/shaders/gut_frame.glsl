// Shared frame block for the Vulkan 3DGUT preview.
// std140 layout must match editor::gpu::GutFrame in gut_renderer.cpp.
//
// Projection and the quadratic ray kernel follow NVIDIA vk_gaussian_splatting
// (Apache-2.0) and Wu et al., 3DGUT. Cameras are Photara's OpenCV convention
// (X right, Y down, Z forward). Framebuffer Y grows downward, so a top-left
// pixel maps to NDC y = -1.

struct GutFrame {
    mat4 world_to_camera;
    vec4 eye;          // xyz = camera position, w = camera model
    vec4 intrinsics;   // fx, fy, cx, cy. Orthographic fx/fy are pixels per world unit.
    vec4 distortion;   // OpenCV fisheye k1..k4
    vec4 viewport;     // width, height, ring contour in sigmas (0 = 2*sqrt(2)), unused
    uvec4 meta;        // sh degree, splat count, sh bases, shading (0 gaussian, 1 rings)
};

// Inspector combo order: Perspective, Orthographic, Fisheye, Panorama.
// Not photara::CameraModel (there fisheye is 1 and equirectangular is 3).
const int k_model_pinhole = 0;
const int k_model_ortho = 1;
const int k_model_fisheye = 2;
const int k_model_equirect = 3;

const float k_gut_delta = 1.73205080757;
const float k_gut_weight_i = 1.0 / 6.0;
const float k_gut_weight0_cov = 2.0;
const float k_cov_dilation = 0.3;
const float k_alpha_threshold = 0.01;
const float k_alpha_cull = 1.0 / 255.0;
const float k_alpha_clamp = 0.99;
const float k_kernel_min_response = 0.0113;
const float k_sh_c0 = 0.28209479177387814;
const float k_sh_c1 = 0.4886025119029199;
const float k_sh_c2[5] = float[5](
    1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
    -1.0925484305920792, 0.5462742152960396);
const float k_sh_c3[7] = float[7](
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658,
    0.3731763325901154, -0.4570457994644658, 1.445305721320277,
    -0.5900435899266435);

uint gut_sortable(float value) {
    uint bits = floatBitsToUint(value);
    uint mask = (bits & 0x80000000u) != 0u ? 0xffffffffu : 0x80000000u;
    return bits ^ mask;
}
