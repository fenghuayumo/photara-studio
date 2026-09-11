// Diagnostic A/B harness. Link with splat_drender and gggs_reference.
#include "splat_drender/api.h"
#include "rasterizer.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <stdexcept>
#include <fstream>

void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Arena {
    std::vector<void*> allocations;
    ~Arena() { for(auto p:allocations) cudaFree(p); }
    template<class T=float> T* get(size_t n) {
        T* p; check(cudaMallocManaged(&p,std::max(n,size_t(1))*sizeof(T)));
        check(cudaMemset(p,0,std::max(n,size_t(1))*sizeof(T)));
        check(cudaDeviceSynchronize()); allocations.push_back(p); return p;
    }
};
struct Pool {
    char* p=nullptr; size_t capacity=0;
    ~Pool(){cudaFree(p);}
    char* operator()(size_t n) {
        if(n>capacity) {cudaFree(p);check(cudaMalloc(&p,n));capacity=n;}
        return p;
    }
};
float* ref_m2d=nullptr;
void compare(const char* name,const float* a,const float* b,int n) {
    double error=0,energy=0; float maximum=0;int worst=0;
    for(int i=0;i<n;++i) {double d=a[i]-b[i];error+=d*d;energy+=double(a[i])*a[i];
        if(std::abs(d)>maximum){maximum=float(std::abs(d));worst=i;}}
    printf("%s relative_l2=%g max=%g index=%d ref=%g new=%g\n",name,sqrt(error/std::max(energy,1e-30)),maximum,worst,a[worst],b[worst]);
}
int main(int argc,char** argv) {
    const bool captured=argc>2;
    const bool geometry_arg=argc>1 && std::string(argv[1])=="geometry";
    const int N=captured?22229:512,W=captured?640:64,H=captured?360:48,P=W*H;
    const bool geometry=geometry_arg;
    Arena a;
    auto means=a.get(3*N),scales=a.get(3*N),rot=a.get(4*N),opa=a.get(N),sh=a.get(48*N);
    auto view=a.get(16),center=a.get(3),bg=a.get(3);
    view[0]=view[10]=cosf(.6f);view[2]=-sinf(.6f);view[8]=sinf(.6f);
    view[5]=view[15]=1;bg[0]=.07f;bg[1]=.11f;bg[2]=.03f;
    std::mt19937 random(42);std::uniform_real_distribution<float> u(-1,1);
    for(int i=0;i<N;++i) {
        means[3*i]=u(random);means[3*i+1]=.7f*u(random);means[3*i+2]=2+.3f*u(random);
        for(int j=0;j<3;++j) scales[3*i+j]=.02f+.10f*std::abs(u(random));
        float norm=0;for(int j=0;j<4;++j){rot[4*i+j]=u(random);norm+=rot[4*i+j]*rot[4*i+j];}
        for(int j=0;j<4;++j)rot[4*i+j]/=sqrtf(norm);
        opa[i]=.15f+.7f*std::abs(u(random));
        for(int j=0;j<48;++j) sh[48*i+j]=.1f*u(random);
    }
    auto gc=a.get(3*P),ga=a.get(P),gd=a.get(P),gn=a.get(3*P);
    for(int i=0;i<P;++i) {
        ga[i]=u(random);if(geometry)gd[i]=.1f*u(random);
        for(int j=0;j<3;++j){gc[j*P+i]=u(random);if(geometry)gn[j*P+i]=u(random);}
    }
    float intrinsics[7]={45,43,31.2f,23.1f,.07f,.11f,.03f};
    if(captured) {
        auto read=[&](const char* suffix,float* ptr,int n){std::ifstream file(std::string(argv[2])+suffix,std::ios::binary);
            file.read(reinterpret_cast<char*>(ptr),n*sizeof(float));if(!file)throw std::runtime_error(suffix);};
        read(".inputmeans",means,3*N);read(".inputscales",scales,3*N);read(".inputrot",rot,4*N);
        read(".inputopa",opa,N);read(".inputsh",sh,48*N);read(".gc",gc,3*P);read(".ga",ga,P);
        float constants[19];read(".inputcamera",constants,19);std::copy(constants,constants+16,view);std::copy(constants+16,constants+19,center);
        read(".hostcamera",intrinsics,7);std::copy(intrinsics+4,intrinsics+7,bg);
    }
    splat_drender::RenderOutputs out[2];splat_drender::ModelGradients grad[2];
    for(int k=0;k<2;++k) {
        out[k]={a.get(3*P),a.get(P),a.get(P),a.get(3*P),a.get(N),a.get<int>(N)};
        grad[k]={a.get(3*N),a.get(48*N),a.get(3*N),a.get(N),a.get(3*N),a.get(4*N),a.get(6*N),a.get(N)};
    }
    Pool pool[10];auto cb=[&](int i){return [&,i](size_t n){return pool[i](n);};};
    int instances=CudaRasterizer::Rasterizer::forward(cb(0),cb(1),cb(2),cb(3),N,3,16,0,0,bg,W,H,
        means,nullptr,opa,scales,rot,nullptr,sh,nullptr,nullptr,nullptr,1,view,center,intrinsics[0],intrinsics[1],intrinsics[2],intrinsics[3],0,false,
        out[0].color,out[0].median_depth,out[0].alpha,out[0].normal,out[0].visibility,out[0].radii,geometry);
    auto zero_scratch=[&](size_t n){auto p=pool[4](n);check(cudaMemset(p,0,n));return p;};
    CudaRasterizer::Rasterizer::backward(zero_scratch,N,3,16,0,0,instances,bg,W,H,
        means,nullptr,opa,scales,rot,nullptr,sh,nullptr,nullptr,nullptr,1,view,center,intrinsics[0],intrinsics[1],intrinsics[2],intrinsics[3],0,
        out[0].radii,out[0].alpha,out[0].normal,out[0].median_depth,pool[0].p,pool[1].p,pool[2].p,pool[3].p,
        gc,gd,ga,gn,grad[0].means,ref_m2d=a.get(3*N),grad[0].colors,grad[0].opacities,grad[0].scales,grad[0].rotations,
        grad[0].covariances,grad[0].sh,nullptr,nullptr,nullptr,grad[0].refine_weight,geometry);
    check(cudaDeviceSynchronize());
    splat_drender::WorkspacePools pools{cb(5),cb(6),cb(7),cb(8),cb(9),{}};
    splat_drender::Gaussians g{N,means,sh,nullptr,opa,scales,rot,nullptr,3,16};
    splat_drender::CameraView camera;camera.width=W;camera.height=H;camera.fx=45;camera.fy=43;
    camera.cx=31.2f;camera.cy=23.1f;camera.world_to_camera=view;camera.center=center;
    camera.fx=intrinsics[0];camera.fy=intrinsics[1];camera.cx=intrinsics[2];camera.cy=intrinsics[3];
    splat_drender::RenderSettings settings;settings.need_depth=geometry;
    std::copy(bg,bg+3,settings.background);
    auto f=splat_drender::Rasterizer::forward(pools,g,camera,settings,out[1]);
    splat_drender::Rasterizer::backward(pools,g,camera,settings,f,
        {out[1].alpha,out[1].median_depth,out[1].normal,out[1].radii},{gc,ga,gd,gn},grad[1]);
    check(cudaDeviceSynchronize());
    printf("instances ref=%d new=%d geometry=%d\n",instances,f.instance_count,geometry);
    if(geometry){
        const size_t a4=(size_t(N)*16+127)/128*128, a3=(size_t(N)*12+127)/128*128;
        std::vector<float> refg(4*N*4+4), newg(4*N*4+4);
        check(cudaMemcpy(refg.data(),pool[4].p,(a4+a3+size_t(N)*16),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(newg.data(),pool[6].p,(2*a4+a3+size_t(N)*12),cudaMemcpyDeviceToHost));
        std::vector<float> rm(3*N,0); check(cudaMemcpy(rm.data(),ref_m2d,3*N*sizeof(float),cudaMemcpyDeviceToHost));
        const float* rc=refg.data(); const float* rn=refg.data()+a4/4; const float* rco=refg.data()+(a4+a3)/4;
        const float* nc=newg.data(); const float* nr=newg.data()+a4/4; const float* nn=newg.data()+2*a4/4;
        const float* nm=newg.data()+(2*a4+a3)/4;
        compare("mid_conic_x",rco,nc,3*N);
        compare("mid_conic_w",rco+3,nc+3,N);
        compare("mid_ray_xy",rc,nr,2*N);
        compare("mid_ray_tc",rc+2,nr+2,N);
        compare("mid_ray_rsigma",rc+3,nr+3,N);
        compare("mid_normal",rn,nn,3*N);
        compare("mid_mean2d",rm.data(),nm,2*N);
    }    compare("color",out[0].color,out[1].color,3*P);compare("alpha",out[0].alpha,out[1].alpha,P);
    compare("depth",out[0].median_depth,out[1].median_depth,P);compare("normal",out[0].normal,out[1].normal,3*P);
    compare("mean",grad[0].means,grad[1].means,3*N);compare("scale",grad[0].scales,grad[1].scales,3*N);
    compare("rotation",grad[0].rotations,grad[1].rotations,4*N);compare("opacity",grad[0].opacities,grad[1].opacities,N);
    compare("sh",grad[0].sh,grad[1].sh,48*N);compare("refine",grad[0].refine_weight,grad[1].refine_weight,N);
}
