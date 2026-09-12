// Differential probe: reference computeCov2DCUDA vs splat_drender
// splat_backward on a single Gaussian with isolated gradient inputs.
#include "device/geometry.cuh"
#include "rasterizer_impl.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }

__global__ void run_splat_backward(
    float3 mean, float3 scale, float4 rot, float opacity, const float* view,
    float fx, float fy, int W, int H, float kernel_size,
    float4 d_conic, float4 d_ray_plane, float3 d_normal, float2 d_mean2d,
    float3* out_mean, float3* out_scale, float4* out_rot, float* out_opa,
    float* out_cov) {
    splat_drender::geo::SplatBackward io;
    io.d_conic = d_conic;
    io.d_ray_plane = d_ray_plane;
    io.d_normal = d_normal;
    io.d_mean2d = d_mean2d;
    io.mean = mean;
    io.view = view;
    io.K.fx = fx; io.K.fy = fy; io.K.cx = 31.2f; io.K.cy = 23.1f;
    io.K.mode = splat_drender::CameraMode::pinhole;
    io.width = W; io.height = H;
    io.kernel_size = kernel_size;
    io.scale_modifier = 1.f;
    io.scale = &scale;
    io.rotation = &rot;
    io.opacity = opacity;
    splat_drender::geo::splat_backward(io);
    *out_mean = io.grad_mean;
    *out_scale = io.grad_scale;
    *out_rot = io.grad_rotation;
    *out_opa = io.grad_opacity;
    for (int i = 0; i < 6; ++i) out_cov[i] = io.grad_cov6[i];
}

// Forward declaration of the reference kernel (defined in render_backward.cu).
__global__ void computeCov2DCUDA(
    int P, const float3* means, const int* radii, const float* cov3Ds,
    const glm::vec3* scales, const float4* rotations, const float* opacities,
    const float mod, RasterIntrinsics K, int image_width, int image_height,
    float tan_fovx, float tan_fovy, float kernel_size,
    const float* view_matrix, const float4* dL_dconics,
    const float4* dL_dray_planes, const glm::vec3* dL_dnormals,
    glm::vec3* dL_dmeans, float* dL_dcov, glm::vec3* dL_dscales,
    glm::vec4* dL_drots, float* dL_dopacity);int main() {
    const int W = 64, H = 48;
    const float fx = 45.f, fy = 43.f, kernel_size = 0.1f;

    float3 mean = make_float3(0.3f, -0.2f, 2.5f);
    float3 scale = make_float3(0.08f, 0.05f, 0.03f);
    float4 rot = make_float4(0.8f, 0.2f, -0.4f, 0.4f);
    const float rn = 1.f / sqrtf(0.8f*0.8f + 0.2f*0.2f + 0.4f*0.4f + 0.4f*0.4f);
    rot.x *= rn; rot.y *= rn; rot.z *= rn; rot.w *= rn;
    const float opacity = 0.6f;

    // Column-major view matrix: slight rotation, no translation.
    float view_h[16] = {};
    const float ca = cosf(0.4f), sa = sinf(0.4f);
    view_h[0] = ca;  view_h[1] = 0.f; view_h[2] = -sa;
    view_h[4] = sa * sinf(0.3f); view_h[5] = cosf(0.3f); view_h[6] = ca * sinf(0.3f);
    view_h[8] = sa * cosf(0.3f); view_h[9] = -sinf(0.3f); view_h[10] = ca * cosf(0.3f);
    view_h[15] = 1.f;

    float *view, *means, *scales, *rots, *opas, *radii;
    float4 *d_conics, *d_ray_planes, *rot_out, *new_rot_out;
    float3 *d_normals, *mean_out, *scale_out, *new_mean_out, *new_scale_out;
    float *opa_out, *new_opa_out, *cov_out, *new_cov_out;
    check(cudaMalloc(&view, 64));  check(cudaMemcpy(view, view_h, 64, cudaMemcpyHostToDevice));
    check(cudaMalloc(&means, 12)); check(cudaMemcpy(means, &mean, 12, cudaMemcpyHostToDevice));
    check(cudaMalloc(&scales, 12)); check(cudaMemcpy(scales, &scale, 12, cudaMemcpyHostToDevice));
    check(cudaMalloc(&rots, 16));  check(cudaMemcpy(rots, &rot, 16, cudaMemcpyHostToDevice));
    check(cudaMalloc(&opas, 4));   check(cudaMemcpy(opas, &opacity, 4, cudaMemcpyHostToDevice));
    check(cudaMalloc(&radii, 4));  check(cudaMemset(radii, 0, 4));
    check(cudaMalloc(&d_conics, 16));
    check(cudaMalloc(&d_ray_planes, 16));
    check(cudaMalloc(&d_normals, 12));
    check(cudaMalloc(&mean_out, 12)); check(cudaMalloc(&scale_out, 12));
    check(cudaMalloc(&rot_out, 16));  check(cudaMalloc(&opa_out, 4)); check(cudaMalloc(&cov_out, 24));
    check(cudaMalloc(&new_mean_out, 12)); check(cudaMalloc(&new_scale_out, 12));
    check(cudaMalloc(&new_rot_out, 16));  check(cudaMalloc(&new_opa_out, 4)); check(cudaMalloc(&new_cov_out, 24));

    struct Case { const char* name; float4 conic; float4 plane; float3 nrm; float2 m2d; };
    const Case cases[] = {
        {"rsigma    ", {0,0,0,0}, {0,0,0,1}, {0,0,0}, {0,0}},
        {"tc        ", {0,0,0,0}, {0,0,1,0}, {0,0,0}, {0,0}},
        {"plane_x   ", {0,0,0,0}, {1,0,0,0}, {0,0,0}, {0,0}},
        {"plane_y   ", {0,0,0,0}, {0,1,0,0}, {0,0,0}, {0,0}},
        {"normal_x  ", {0,0,0,0}, {0,0,0,0}, {1,0,0}, {0,0}},
        {"normal_yz ", {0,0,0,0}, {0,0,0,0}, {0,0.4f,-0.7f}, {0,0}},
        {"all       ", {0.3f,-0.2f,0.5f,0.7f}, {0.2f,-0.3f,0.5f,0.8f}, {0.4f,-0.2f,0.6f}, {0.05f,-0.06f}},
    };

    const float tan_fovx = float(W) / (2.f * fx);
    const float tan_fovy = float(H) / (2.f * fy);
    RasterIntrinsics K{fx, fy, 31.2f, 23.1f, RASTER_PINHOLE, 0, 0, 0, 0};

    for (const Case& c : cases) {
        check(cudaMemcpy(d_conics, &c.conic, 16, cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_ray_planes, &c.plane, 16, cudaMemcpyHostToDevice));
        check(cudaMemcpy(d_normals, &c.nrm, 12, cudaMemcpyHostToDevice));
        check(cudaMemset(mean_out, 0, 12)); check(cudaMemset(scale_out, 0, 12));
        check(cudaMemset(rot_out, 0, 16)); check(cudaMemset(opa_out, 0, 4)); check(cudaMemset(cov_out, 0, 24));
        check(cudaMemset(new_mean_out, 0, 12)); check(cudaMemset(new_scale_out, 0, 12));
        check(cudaMemset(new_rot_out, 0, 16)); check(cudaMemset(new_opa_out, 0, 4)); check(cudaMemset(new_cov_out, 0, 24));
        check(cudaMemset(radii, 1, 4));

        computeCov2DCUDA<<<1, 1>>>(
            1, (const float3*)means, (const int*)radii, nullptr,
            (const glm::vec3*)scales, (const float4*)rots, opas, 1.f, K, W, H,
            tan_fovx, tan_fovy, kernel_size, view, d_conics, d_ray_planes,
            (const glm::vec3*)d_normals, (glm::vec3*)mean_out, cov_out, (glm::vec3*)scale_out,
            (glm::vec4*)rot_out, opa_out);
        run_splat_backward<<<1, 1>>>(
            mean, scale, rot, opacity, view, fx, fy, W, H, kernel_size,
            c.conic, c.plane, c.nrm, c.m2d,
            new_mean_out, new_scale_out, new_rot_out, new_opa_out, new_cov_out);
        check(cudaDeviceSynchronize());

        float3 rm, nm2, rs, ns2; float4 rr, nr2;
        float ro, no; float rc[6], nc[6];
        check(cudaMemcpy(&rm, mean_out, 12, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&nm2, new_mean_out, 12, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&rs, scale_out, 12, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&ns2, new_scale_out, 12, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&rr, rot_out, 16, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&nr2, new_rot_out, 16, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&ro, opa_out, 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&no, new_opa_out, 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(rc, cov_out, 24, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(nc, new_cov_out, 24, cudaMemcpyDeviceToHost));

        printf("== %s ==\n", c.name);
        printf("mean  ref (%12.6f %12.6f %12.6f)  new (%12.6f %12.6f %12.6f)\n",
               rm.x, rm.y, rm.z, nm2.x, nm2.y, nm2.z);
        printf("scale ref (%12.6f %12.6f %12.6f)  new (%12.6f %12.6f %12.6f)\n",
               rs.x, rs.y, rs.z, ns2.x, ns2.y, ns2.z);
        printf("rot   ref (%12.6f %12.6f %12.6f %12.6f)\n", rr.x, rr.y, rr.z, rr.w);
        printf("      new (%12.6f %12.6f %12.6f %12.6f)\n", nr2.x, nr2.y, nr2.z, nr2.w);
        printf("opa   ref %12.6f  new %12.6f\n", ro, no);
        if (c.conic.x != 0.f) {
            printf("cov   ref"); for (int i = 0; i < 6; ++i) printf(" %10.5f", rc[i]); printf("\n");
            printf("cov   new"); for (int i = 0; i < 6; ++i) printf(" %10.5f", nc[i]); printf("\n");
        }
    }
    return 0;
}
