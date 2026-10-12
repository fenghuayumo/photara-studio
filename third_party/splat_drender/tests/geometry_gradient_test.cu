#include "device/geometry.cuh"
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace splat_drender;

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void evaluate(const float* p, const float* g, float* out) {
    const float view[16] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    float3 mean = make_float3(p[0], p[1], p[2]);
    float3 scale = make_float3(p[3], p[4], p[5]);
    float4 rotation = make_float4(p[6], p[7], p[8], p[9]);
    CameraIntrinsics K;
    K.fx=45; K.fy=43; K.cx=31.2f; K.cy=23.1f; K.mode=CameraMode::pinhole;
    geo::Splat splat;
    geo::project_splat(mean, view, K, 64, 48, .1f, 1.f,
                      &scale, &rotation, nullptr, splat);
    out[0] = splat.normal.x*g[0]+splat.normal.y*g[1]+splat.normal.z*g[2]+
             splat.ray_plane.x*g[3]+splat.ray_plane.y*g[4]+
             splat.ray_plane.z*g[5]+splat.ray_plane.w*g[6];
    geo::SplatBackward io;
    io.mean=mean; io.view=view; io.K=K; io.width=64; io.height=48;
    io.kernel_size=.1f; io.scale=&scale; io.rotation=&rotation; io.opacity=.6f;
    io.d_normal=make_float3(g[0],g[1],g[2]);
    io.d_ray_plane=make_float4(g[3],g[4],g[5],g[6]);
    geo::splat_backward(io);
    out[1]=io.grad_mean.x; out[2]=io.grad_mean.y; out[3]=io.grad_mean.z;
    out[4]=io.grad_scale.x; out[5]=io.grad_scale.y; out[6]=io.grad_scale.z;
    // The caller normalizes scalar-first quaternions before projection.
    float dot = p[6]*io.grad_rotation.x+p[7]*io.grad_rotation.y+
                p[8]*io.grad_rotation.z+p[9]*io.grad_rotation.w;
    out[7]=io.grad_rotation.x-p[6]*dot; out[8]=io.grad_rotation.y-p[7]*dot;
    out[9]=io.grad_rotation.z-p[8]*dot; out[10]=io.grad_rotation.w-p[9]*dot;
}

void normalize(std::array<float,10>& p) {
    float norm=0;
    for (int i=6;i<10;++i) norm+=p[i]*p[i];
    for (int i=6;i<10;++i) p[i]/=std::sqrt(norm);
}

// Compare the RGB specialization against the generic derivative on clipped
// means, anisotropic scales, rotated cameras and varying kernel footprints.
// Passing intrinsics at runtime keeps the reference's camera dispatch live.
__global__ void compare_rgb(CameraIntrinsics K, float* out) {
    const int i = threadIdx.x;
    const float view[16] = {.8f,.6f,0,0,-.6f,.8f,0,0,0,0,1,0,.1f,-.2f,.3f,1};
    const float3 scale = make_float3(.001f + .003f*(i%7), .02f, .07f);
    const float4 rotation = make_float4(.8f,.2f,-.4f,.4f);
    geo::SplatBackward reference;
    reference.mean = make_float3(i%3 == 0 ? 8.f : .03f*(i-64), -.01f*i, 1.f+.02f*i);
    reference.view = view; reference.K = K;
    reference.width = 64; reference.height = 48;
    reference.kernel_size = i%2 ? .1f : 0.f;
    reference.scale_modifier = .7f;
    reference.scale = &scale; reference.rotation = &rotation;
    reference.opacity = .6f;
    reference.d_conic = make_float4(.2f, -.03f, .1f, .7f);
    reference.d_mean2d = make_float2(.3f, -.2f);
    auto specialized = reference;
    geo::splat_backward(reference);
    geo::splat_backward<true>(specialized);
    const auto store = [&](const geo::SplatBackward& io, int offset) {
        float* p = out + i*22 + offset;
        p[0]=io.grad_mean.x; p[1]=io.grad_mean.y; p[2]=io.grad_mean.z;
        p[3]=io.grad_scale.x; p[4]=io.grad_scale.y; p[5]=io.grad_scale.z;
        p[6]=io.grad_rotation.x; p[7]=io.grad_rotation.y;
        p[8]=io.grad_rotation.z; p[9]=io.grad_rotation.w;
        p[10]=io.grad_opacity;
    };
    store(reference, 0); store(specialized, 11);
}

void check_rgb_specialization() {
    CameraIntrinsics K;
    K.fx=45; K.fy=43; K.cx=31.2f; K.cy=23.1f;
    std::array<float,128*22> values;
    float* device=nullptr;
    check(cudaMalloc(&device, sizeof(values)));
    compare_rgb<<<1,128>>>(K, device);
    check(cudaGetLastError());
    check(cudaMemcpy(values.data(), device, sizeof(values), cudaMemcpyDeviceToHost));
    check(cudaFree(device));
    float max_scaled_error=0.f;
    bool differs=false;
    for (int i=0;i<128;++i) for (int c=0;c<11;++c) {
        const float a=values[i*22+c], b=values[i*22+11+c];
        max_scaled_error=std::fmax(max_scaled_error, std::abs(a-b)/(1.f+std::abs(a)));
        // Camera specialization changes compiler contraction/register spilling.
        // Thin rotated splats have cancellation in the covariance chain, so
        // compare with a mixed absolute/relative FP32 tolerance.
        if (!std::isfinite(a) || !std::isfinite(b) ||
            std::abs(a-b)>1e-4f+1e-4f*std::abs(a)) {
            std::cerr<<"RGB case="<<i<<" channel="<<c<<" generic="<<a<<" specialized="<<b<<'\n';
            differs=true;
        }
    }
    std::cout<<"Pinhole RGB specialization max |error|/(1+|reference|)="<<max_scaled_error<<" (128 cases)\n";
    if (differs) throw std::runtime_error("Pinhole RGB specialization changes geometry gradients");
}
}

int main() {
    try {
        check_rgb_specialization();
        std::array<float,10> base{.3f,-.2f,2.5f,.08f,.05f,.03f,.8f,.2f,-.4f,.4f};
        normalize(base);
        const std::array<std::array<float,7>,5> losses{{
            {1,0,0,0,0,0,0}, {0,.4f,-.7f,0,0,0,0},
            {0,0,0,.3f,-.2f,0,0}, {0,0,0,0,0,.4f,0},
            {0,0,0,0,0,0,.001f}}};
        float *dp=nullptr,*dg=nullptr,*dout=nullptr;
        check(cudaMalloc(&dp,40)); check(cudaMalloc(&dg,28)); check(cudaMalloc(&dout,44));
        const auto run=[&](const std::array<float,10>& p,const std::array<float,7>& g) {
            check(cudaMemcpy(dp,p.data(),40,cudaMemcpyHostToDevice));
            check(cudaMemcpy(dg,g.data(),28,cudaMemcpyHostToDevice));
            evaluate<<<1,1>>>(dp,dg,dout);
            check(cudaGetLastError());
            std::array<float,11> result;
            check(cudaMemcpy(result.data(),dout,44,cudaMemcpyDeviceToHost));
            return result;
        };
        for (std::size_t channel=0;channel<losses.size();++channel) {
            const auto analytic=run(base,losses[channel]);
            for (int i=0;i<10;++i) {
                const float step=(i>=3 && i<=5)?1e-4f:5e-4f;
                auto plus=base,minus=base;
                plus[i]+=step; minus[i]-=step;
                if (i>=6) { normalize(plus); normalize(minus); }
                const float numerical=(run(plus,losses[channel])[0]-run(minus,losses[channel])[0])/(2*step);
                const float tolerance=3e-4f+5e-4f*std::abs(numerical);
                if (!std::isfinite(analytic[i+1]) || std::abs(numerical-analytic[i+1])>tolerance) {
                    std::cerr<<"channel="<<channel<<" parameter="<<i
                             <<" analytic="<<analytic[i+1]<<" numerical="<<numerical<<'\n';
                    throw std::runtime_error("Projection geometry gradient differs from finite difference");
                }
            }
        }
        check(cudaFree(dp)); check(cudaFree(dg)); check(cudaFree(dout));
        std::cout<<"Normal/ray-plane gradients match finite differences\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
