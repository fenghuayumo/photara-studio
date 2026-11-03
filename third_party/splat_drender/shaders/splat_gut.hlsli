// 3DGUT forward geometry. Keep the UT weights, bin expansion and response
// consistent with CUDA src/gut.cu and src/device/gut.cuh.
#include "splat_gut_ray.hlsli"
#include "splat_gut_sh.hlsli"
Projection gut_project(float3 t, uint mode, int width, int height,
                       float fx, float fy, float cx, float cy,
                       float k1, float k2, float k3, float k4) {
    if (mode == kModeOrtho) return project_ortho(t, fx, fy, cx, cy);
    return project_camera(t, mode, width, height, fx, fy, cx, cy, k1, k2, k3, k4);
}

void gut_preprocess(uint index, float3 mean, float3 t, float3 scale,
                    float4 rotation, float opacity, float3 sh_mean, float3 sh_eye) {
    uint mode = pc.u16;
    int width = int(pc.u4), height = int(pc.u5);
    float fx = asfloat(pc.u8), fy = asfloat(pc.u9);
    float cx = asfloat(pc.u10), cy = asfloat(pc.u11);
    float k1 = asfloat(pc.u12), k2 = asfloat(pc.u13);
    float k3 = asfloat(pc.u14), k4 = asfloat(pc.u15);
    float modifier = asfloat(pc.u18);
    Projection center = gut_project(t, mode, width, height, fx, fy, cx, cy, k1, k2, k3, k4);
    bool pole = mode == kModeEquirect && !center.valid && camera_visible(t, mode);
    if (pole) {
        center.pixel_x = float(width)*0.5f;
        center.pixel_y = t.y > 0.0f ? float(height) : 0.0f;
        center.valid = true;
    }
    if (!center.valid || !(opacity >= kAlphaFloor) || any(scale <= 0.0f) || !(modifier > 0.0f)) return;
    GutRotation shared_rotation=gut_rotation(rotation.x,rotation.y,rotation.z,rotation.w);
    Mat3 R;
    [unroll] for(uint rr=0u;rr<3u;++rr)
        [unroll] for(uint cc=0u;cc<3u;++cc) R.m[rr][cc]=shared_rotation.m[rr][cc];
    float2 points[7];
    points[0] = float2(center.pixel_x, center.pixel_y);
    float2 mu = 0.25f * points[0];
    bool boundary = pole;
    if (mode == kModeEquirect) {
        float3 eye = float3(camera[16],camera[17],camera[18]);
        float3 up = float3(camera[1],camera[5],camera[9]);
        float3 o = mat_mul_vec(R,eye-mean)/(scale*modifier);
        float3 d = mat_mul_vec(R,up)/(scale*modifier);
        float dd = dot(d,d), peak = -dot(o,d)/dd;
        float3 q = o+peak*d;
        float contour = float((pc.u3>>16u)&4095u)/256.0f;
        float support2 = (pc.u3 & 64u)!=0u ? contour*contour : -2.0f*log(kAlphaFloor/opacity);
        boundary = boundary || (dd > 0.0f && dot(q,q) <= support2*1.001f);
    }
    // Seven-point UT, lambda=1: central weight 1/4, others 1/8.
    [unroll] for (int axis = 0; axis < 3; ++axis) {
        [unroll] for (int side = 0; side < 2; ++side) {
            float spread = (side != 0 ? 2.0f : -2.0f) * modifier * scale[axis];
            float3 sigma = mean + spread * float3(R.m[axis][0], R.m[axis][1], R.m[axis][2]);
            float3 st = gut_camera_point(sigma);
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
    if (boundary && mode == kModeEquirect) conic.xyz = float3(-1.0f,0.0f,-1.0f);
    float radius = ceil(3.0f * sqrt(max(cov.x, cov.z)));
    // Rings use a proxy bin opacity to cover their requested contour.
    if ((pc.u3 & 64u) != 0u) {
        if (opacity <= kAlphaFloor) return;
        float sigma = float((pc.u3 >> 16u) & 4095u) / 256.0f;
        conic.w = kAlphaFloor * exp(0.5f * sigma * sigma);
    }
    float3 eye = float3(camera[16], camera[17], camera[18]);
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;
    float3 rgb;
    if ((pc.u3 & 1u) != 0u) {
        rgb = evaluate_gut_sh_unclamped(pc.u1, sh_mean, sh_eye, color_src, index*pc.u2*3u, c0, c1, c2);
        if (!reconstruction_enabled()) rgb = max(rgb,0.f);
    }
    else rgb = float3(color_src[3u*index], color_src[3u*index+1u], color_src[3u*index+2u]);
    rgb = reconstruction_color(index,rgb);
    float selection_depth = mode == kModeEquirect ? length(t) : t.z;
    if (camera[19] > 0.5f && (selection_depth < camera[20] || selection_depth > camera[21])) rgb *= 0.18f;
    Mat3 W = world_rotation_transposed(camera[0], camera[1], camera[2], camera[4], camera[5], camera[6],
                                      camera[8], camera[9], camera[10]);
    Mat3 B = mat_zero();
    [unroll] for (uint row=0u;row<3u;++row) {
        float inv = gut_rcp(gut_mul(scale[row],modifier));
        [unroll] for (uint col=0u;col<3u;++col) B.m[row][col] = gut_mul(R.m[row][col],inv);
    }
    Mat3 A = mat_mul(B,W); // Camera-space support bounds only.
    float3 delta=eye-mean;
    float3 origin=float3(gut_dot(float3(B.m[0][0],B.m[0][1],B.m[0][2]),delta),
                         gut_dot(float3(B.m[1][0],B.m[1][1],B.m[1][2]),delta),
                         gut_dot(float3(B.m[2][0],B.m[2][1],B.m[2][2]),delta));
    float support2=(pc.u3 & 64u)!=0u ? pow(float((pc.u3>>16u)&4095u)/256.0f,2.0f)
                                             : -2.0f*log(kAlphaFloor/opacity);
    gut_init_support(float4(A.m[0][0],A.m[0][1],A.m[0][2],origin.x),
                     float4(A.m[1][0],A.m[1][1],A.m[1][2],origin.y),
                     float4(A.m[2][0],A.m[2][1],A.m[2][2],origin.z),opacity,support2,
                     float4(fx,fy,cx,cy),uint2(width,height),(pc.u3 & 512u)!=0u && width<65520 && height<65520);
    uint touched = enumerate_tiles(mu, conic, int(pc.u6), int(pc.u7), index, 0, int(pc.u19), 0, 0);
    if (touched == 0u) return;
    int axis = 0;
    if (scale.y < scale[axis]) axis = 1;
    if (scale.z < scale[axis]) axis = 2;
    float3 normal = mat_mul_vec(mat_transpose(W), float3(R.m[axis][0], R.m[axis][1], R.m[axis][2]));
    uint base = index*((pc.u3 & 512u)!=0u ? 11u : 7u);
    gauss_f[base] = float4(mu, gut_norm(t), asfloat(uint(radius)));
    gauss_f[base+1u] = conic;
    gauss_f[base+2u] = float4(rgb, opacity);
    // World-space ray -> principal axes, matching CUDA's cached gut_rows.
    gauss_f[base+3u] = float4(B.m[0][0], B.m[0][1], B.m[0][2], origin.x);
    gauss_f[base+4u] = float4(B.m[1][0], B.m[1][1], B.m[1][2], origin.y);
    gauss_f[base+5u] = float4(B.m[2][0], B.m[2][1], B.m[2][2], origin.z);
    gauss_f[base+6u] = float4(normal, 0.0f);
    // Panorama ADC uses a sphere tangent plane; its longitude chart is
    // singular at the sides/poles. Inference retains its surface normal.
    if ((pc.u3 & 8u) != 0u) {
        if (mode == kModeEquirect) {
            float factor = 6.28318530718f*gut_norm(t);
            gauss_f[base+6u] = float4((mean-eye)*(6.28318530718f/factor), factor);
        }
        else if (mode == kModeFisheye)
            gauss_f[base+6u] = float4(t,0.0f);
        else gauss_f[base+6u] = float4(1.0f/(abs(center.du0)+1.0e-6f),
                                      1.0f/(abs(center.dv1)+1.0e-6f), 0.0f, 0.0f);
    }
    // Cache the support test itself, not only the source transform. Count and
    // emit must make bit-identical discrete decisions across shader variants.
    if ((pc.u3 & 512u)!=0u) {
    gauss_f[base+7u]=float4(gut_c0,gut_origin.x);
    gauss_f[base+8u]=float4(gut_c1,gut_origin.y);
    gauss_f[base+9u]=float4(gut_c2,gut_origin.z);
    uint bx=uint(gut_tile_bounds.x)|(uint(gut_tile_bounds.z)<<12u);
    uint by=uint(gut_tile_bounds.y)|(uint(gut_tile_bounds.w)<<12u);
    gauss_f[base+10u]=float4(float(bx),float(by),gut_support2,gut_support_enabled ? gut_sign : 0.0f);
    }
    gauss_u[index] = touched;
    gauss_u[pc.u0+index] = 1u;
    gauss_u[2u*pc.u0+index] = asuint(gut_norm(t));
    gauss_u[3u*pc.u0+index] = uint(radius);
}
