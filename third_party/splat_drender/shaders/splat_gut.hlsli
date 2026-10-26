// 3DGUT forward geometry. Keep the UT weights, bin expansion and response
// consistent with CUDA src/gut.cu and src/device/gut.cuh.
Projection gut_project(float3 t, uint mode, int width, int height,
                       float fx, float fy, float cx, float cy,
                       float k1, float k2, float k3, float k4) {
    if (mode == kModeOrtho) return project_ortho(t, fx, fy, cx, cy);
    return project_camera(t, mode, width, height, fx, fy, cx, cy, k1, k2, k3, k4);
}

void gut_preprocess(uint index, float3 mean, float3 t, float3 scale,
                    float4 rotation, float opacity) {
    uint mode = pc.u16;
    int width = int(pc.u4), height = int(pc.u5);
    float fx = asfloat(pc.u8), fy = asfloat(pc.u9);
    float cx = asfloat(pc.u10), cy = asfloat(pc.u11);
    float k1 = asfloat(pc.u12), k2 = asfloat(pc.u13);
    float k3 = asfloat(pc.u14), k4 = asfloat(pc.u15);
    float modifier = asfloat(pc.u18);
    Projection center = gut_project(t, mode, width, height, fx, fy, cx, cy, k1, k2, k3, k4);
    if (!center.valid || !(opacity >= kAlphaFloor) || any(scale <= 0.0f) || !(modifier > 0.0f)) return;
    Mat3 R = quat_rotation(rotation);
    float2 points[7];
    points[0] = float2(center.pixel_x, center.pixel_y);
    float2 mu = 0.25f * points[0];
    bool boundary = false;
    // Seven-point UT, lambda=1: central weight 1/4, others 1/8.
    [unroll] for (int axis = 0; axis < 3; ++axis) {
        [unroll] for (int side = 0; side < 2; ++side) {
            float spread = (side != 0 ? 2.0f : -2.0f) * modifier * scale[axis];
            float3 sigma = mean + spread * float3(R.m[axis][0], R.m[axis][1], R.m[axis][2]);
            float3 st = xform_point(sigma, camera[0], camera[1], camera[2], camera[4], camera[5], camera[6],
                                    camera[8], camera[9], camera[10], camera[12], camera[13], camera[14]);
            Projection p = gut_project(st, mode, width, height, fx, fy, cx, cy, k1, k2, k3, k4);
            boundary = boundary || !p.valid;
            float2 sigma_pixel = float2(center.pixel_x + wrap_dx(p.pixel_x - center.pixel_x, width, mode), p.pixel_y);
            points[1 + axis * 2 + side] = sigma_pixel;
            mu += 0.125f * sigma_pixel;
        }
    }
    float3 cov = 0.0f;
    [unroll] for (int j = 0; j < 7; ++j) {
        float2 d = points[j] - mu;
        cov += (j == 0 ? 0.25f : 0.125f) * float3(d.x*d.x, d.x*d.y, d.y*d.y);
    }
    cov = 4.0f * cov + float3(1.0f, 0.0f, 1.0f);
    if (boundary) {
        mu = float2(width, height) * 0.5f;
        cov = float3(float(width)*width, 0.0f, float(height)*height);
    }
    float det = cov.x*cov.z - cov.y*cov.y;
    if (!(det > 0.0f) || !finite1(det)) return;
    float4 conic = float4(cov.z/det, -cov.y/det, cov.x/det, opacity);
    float radius = ceil(3.0f * sqrt(max(cov.x, cov.z)));
    // Rings use a proxy bin opacity to cover their requested contour.
    if ((pc.u3 & 64u) != 0u) {
        if (opacity <= kAlphaFloor) return;
        float sigma = float((pc.u3 >> 16u) & 4095u) / 256.0f;
        conic.w = kAlphaFloor * exp(0.5f * sigma * sigma);
    }
    uint touched = enumerate_tiles(mu, conic, int(pc.u6), int(pc.u7), index, 0, int(pc.u19), 0, 0);
    if (touched == 0u) return;
    float3 eye = float3(camera[16], camera[17], camera[18]);
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;
    float3 rgb;
    if ((pc.u3 & 1u) != 0u)
        rgb = evaluate_sh(pc.u1, mean, eye, color_src, index*pc.u2*3u, c0, c1, c2);
    else rgb = float3(color_src[3u*index], color_src[3u*index+1u], color_src[3u*index+2u]);
    float selection_depth = mode == kModeEquirect ? length(t) : t.z;
    if (camera[19] > 0.5f && (selection_depth < camera[20] || selection_depth > camera[21])) rgb *= 0.18f;
    Mat3 W = world_rotation_transposed(camera[0], camera[1], camera[2], camera[4], camera[5], camera[6],
                                      camera[8], camera[9], camera[10]);
    Mat3 A = mat_mul(mat_mul(mat_diag(1.0f/(scale.x*modifier), 1.0f/(scale.y*modifier),
                                     1.0f/(scale.z*modifier)), R), W);
    float3 origin = mat_mul_vec(R, eye - mean) / (scale*modifier);
    int axis = 0;
    if (scale.y < scale[axis]) axis = 1;
    if (scale.z < scale[axis]) axis = 2;
    float3 normal = mat_mul_vec(mat_transpose(W), float3(R.m[axis][0], R.m[axis][1], R.m[axis][2]));
    uint base = index*7u;
    gauss_f[base] = float4(mu, length(t), asfloat(uint(radius)));
    gauss_f[base+1u] = conic;
    gauss_f[base+2u] = float4(rgb, opacity);
    // Camera-space ray -> principal axes, plus local ray origin in .w.
    gauss_f[base+3u] = float4(A.m[0][0], A.m[0][1], A.m[0][2], origin.x);
    gauss_f[base+4u] = float4(A.m[1][0], A.m[1][1], A.m[1][2], origin.y);
    gauss_f[base+5u] = float4(A.m[2][0], A.m[2][1], A.m[2][2], origin.z);
    gauss_f[base+6u] = float4(normal, 0.0f);
    gauss_u[index] = touched;
    gauss_u[pc.u0+index] = 1u;
    gauss_u[2u*pc.u0+index] = asuint(length(t));
    gauss_u[3u*pc.u0+index] = uint(radius);
}
