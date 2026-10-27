#pragma once
#include "splat/rasterizer.hpp"
#include "../src/splat/photometric_loss.hpp"
#include "vulkan/backend.hpp"
#include <algorithm>
#include <numeric>
#include <map>
#include <string>

namespace gut_validation {
using namespace photara::splat;
using tinytensor::Device;
using tinytensor::Tensor;
inline std::map<std::string,std::pair<double,double>> parity_errors;

inline void compare(const char* name, const std::vector<float>& a,
                    const std::vector<float>& b, double relative=0.003,
                    double absolute=2e-5) {
    require(a.size()==b.size(), "GUT parity shape mismatch");
    double error=0,reference=0,maximum=0;
    for (std::size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "GUT parity nonfinite value");
        const double d=double(a[i])-b[i];
        error+=d*d;reference+=double(a[i])*a[i];maximum=std::max(maximum,std::abs(d));
    }
    double ratio=std::sqrt(error/std::max(reference,1e-30));
    auto& worst=parity_errors[name];
    worst.first=std::max(worst.first,ratio);worst.second=std::max(worst.second,maximum);
    if (ratio>relative && maximum>absolute) {
        std::cerr<<name<<" relative_l2="<<ratio<<" max="<<maximum<<'\n';
        throw std::runtime_error("Vulkan GUT differs from CUDA");
    }
}

inline GaussianModel model(Device device, bool filter, std::size_t count=40) {
    constexpr std::size_t bases=16;
    std::vector<float> means(count*3),scales(count*3),rotations(count*4),opacity(count),sh(count*bases*3),filters(count);
    for(std::size_t i=0;i<count;++i) {
        means[3*i]=.13f*std::sin(float(i));means[3*i+1]=.1f*std::cos(float(i));means[3*i+2]=2.f+.025f*i;
        for(int axis=0;axis<3;++axis) scales[3*i+axis]=std::log(.19f+.065f*axis+.001f*i);
        rotations[4*i]=.9f;rotations[4*i+1]=.1f;rotations[4*i+2]=-.16f;rotations[4*i+3]=.07f;
        // Low opacity exercises more than one 32-instance bucket; the
        // filtered fixture also exercises early transmittance termination.
        opacity[i]=filter ? 1.2f+.03f*i : -3.1f+.01f*i;
        filters[i]=filter ? .04f : 0.f;
        for(std::size_t k=0;k<bases*3;++k) sh[i*bases*3+k]=k<3 ? .1f+.1f*(k%3) : .03f*std::sin(float(i+k));
    }
    GaussianModel result;result.render_method=RenderMethod::gaussian_3dgut;result.sh_degree=3;
    result.means=Tensor::from_vector(means,{count,3U},device);
    result.log_scales=Tensor::from_vector(scales,{count,3U},device);
    result.quaternions=Tensor::from_vector(rotations,{count,4U},device);
    result.opacity_logits=Tensor::from_vector(opacity,{count,1U},device);
    result.sh=Tensor::from_vector(sh,{count,bases,3U},device);
    result.filter_3d=Tensor::from_vector(filters,{count,1U},device);
    return result;
}

inline void finite_difference(GaussianModel model, const Camera& camera, RasterizeOptions options) {
    // Two overlapping primitives and interior rays keep the finite difference
    // away from discrete alpha-floor, early-termination and tile boundaries.
    model=gut_validation::model(Device::Vulkan,true,2);
    options.rasterization.alpha_clip=.99f;
    Rasterizer raster;
    const std::size_t pixels=camera.width*camera.height;
    std::vector<float> color(3*pixels),alpha(pixels);
    for(unsigned y=camera.height/2-2;y<=camera.height/2+2;++y)
      for(unsigned x=camera.width/2-2;x<=camera.width/2+2;++x) {
        const std::size_t i=y*camera.width+x;
        for(unsigned channel=0;channel<3;++channel)
            color[channel*pixels+i]=std::sin(float(channel*pixels+i)*.013f)/25.f;
        alpha[i]=.2f*std::cos(float(i)*.019f)/25.f;
      }
    auto dc=Tensor::from_vector(color,{3U,camera.height,camera.width},Device::Vulkan);
    auto da=Tensor::from_vector(alpha,{camera.height,camera.width},Device::Vulkan);
    std::cout<<"GUT finite difference forward"<<std::endl;
    auto frame=raster.forward(model,camera,options);
    auto grad=raster.backward(model,frame,dc,da,{},{});
    std::cout<<"GUT finite difference backward"<<std::endl;
    Tensor* parameters[]{&model.means,&model.log_scales,&model.quaternions,&model.opacity_logits,&model.sh};
    const Tensor* derivatives[]{&grad.means,&grad.log_scales,&grad.quaternions,&grad.opacity_logits,&grad.sh};
    options.record_backward_state=false;
    const auto objective=[&]() {
        auto f=raster.forward(model,camera,options);
        auto rgb=f.color.to_vector(),a=f.alpha.to_vector();
        double result=0;
        for(std::size_t i=0;i<rgb.size();++i) result+=double(rgb[i])*color[i];
        for(std::size_t i=0;i<a.size();++i) result+=double(a[i])*alpha[i];
        return result;
    };
    for(unsigned family=0;family<5;++family) {
        Tensor original=*parameters[family];auto values=original.to_vector();
        auto analytic=derivatives[family]->to_vector();
        constexpr float epsilon=.002f;
        double predicted=0;
        for(std::size_t i=0;i<values.size();++i) predicted+=analytic[i]*std::sin(float(i+1)*.7f);
        auto plus=values,minus=values;
        for(std::size_t i=0;i<values.size();++i) {
            const float delta=epsilon*std::sin(float(i+1)*.7f);
            plus[i]+=delta;minus[i]-=delta;
        }
        *parameters[family]=Tensor::from_vector(plus,original.shape(),Device::Vulkan);
        double upper=objective();
        *parameters[family]=Tensor::from_vector(minus,original.shape(),Device::Vulkan);
        double lower=objective();
        *parameters[family]=original;
        const double numeric=(upper-lower)/(2*epsilon);
        std::cout<<"GUT finite difference family="<<family<<" analytic="<<predicted<<" numeric="<<numeric<<'\n';
        require(std::abs(predicted-numeric)<3e-5+.03*std::abs(numeric),"GUT Vulkan finite difference mismatch");
    }
}

inline void run() {
    require(tinytensor::vulkan::available(), "GUT Vulkan validation needs a Vulkan device");
    for(unsigned mode : {0U,1U,3U,4U}) for(bool filter : {false,true}) {
        std::cout<<"GUT parity camera="<<mode<<" filter="<<filter<<std::endl;
        Camera camera;camera.width=35;camera.height=29;camera.fx=24;camera.fy=25;camera.cx=17;camera.cy=14;
        camera.model=mode==4 ? CameraModel::pinhole : static_cast<CameraModel>(mode);camera.orthographic=mode==4;
        camera.k1=.025f;camera.k2=-.002f;
        const float angle=.12f,c=std::cos(angle),s=std::sin(angle);
        camera.world_to_camera={c,0,-s,0, 0,1,0,0, s,0,c,0, -.08f*c,0,.08f*s,1};
        camera.position={.08f,0,0};
        auto cuda=model(Device::CUDA,filter),vk=model(Device::Vulkan,filter);
        RasterizeOptions options;options.require_depth=false;options.active_sh_degree=3;
        options.background={.13f,.07f,.21f};options.scale_modifier=1.13f;
        options.rasterization.alpha_clip=filter ? .55f : .99f;
        options.rasterization.pixel_center_offset=.5f;
        Rasterizer cr,vr;
        auto cf=cr.forward(cuda,camera,options),vf=vr.forward(vk,camera,options);
        compare("RGB",cf.color.to_vector(),vf.color.to_vector(),.001,2e-5);
        compare("alpha",cf.alpha.to_vector(),vf.alpha.to_vector(),.001,2e-5);
        const std::size_t pixels=camera.width*camera.height;
        std::vector<float> target(3*pixels),mask(pixels),alpha(pixels);
        for(std::size_t i=0;i<target.size();++i) target[i]=.25f+.1f*std::sin(float(i)*.017f);
        for(std::size_t i=0;i<pixels;++i) {mask[i]=i%7 ? 1.f : 0.f;alpha[i]=.001f*std::cos(float(i));}
        auto ct=Tensor::from_vector(target,{3U,camera.height,camera.width},Device::CUDA);
        auto cm=Tensor::from_vector(mask,{camera.height,camera.width},Device::CUDA);
        auto vt=ct.to(Device::Vulkan),vm=cm.to(Device::Vulkan);
        for(bool masked : {false,true}) {
            Tensor dc=Tensor::zeros(cf.color.shape(),Device::CUDA),dv;
            auto scalar=Tensor::zeros({1U},Device::CUDA);
            detail::photometric_loss(cf.color,ct,cm,masked,dc,scalar.ptr<float>(),camera.width,camera.height,{.2f,.8f});
            const float vl=vr.photometric_loss(vf,vt,vm,masked,.2f,.8f,true,&dv);
            compare("loss",scalar.to_vector(),{vl},.001,2e-5);
            compare("image gradient",dc.to_vector(),dv.to_vector(),.003,2e-5);
            auto ca=Tensor::from_vector(alpha,{camera.height,camera.width},Device::CUDA);
            auto va=ca.to(Device::Vulkan);
            auto cg=cr.backward(cuda,cf,dc,ca,{},{}),vg=vr.backward(vk,vf,dv,va,{},{});
            compare("means",cg.means.to_vector(),vg.means.to_vector());
            compare("scales",cg.log_scales.to_vector(),vg.log_scales.to_vector());
            compare("rotations",cg.quaternions.to_vector(),vg.quaternions.to_vector());
            compare("opacity",cg.opacity_logits.to_vector(),vg.opacity_logits.to_vector());
            compare("SH",cg.sh.to_vector(),vg.sh.to_vector());
            compare("ADC",cg.refine_weight.to_vector(),vg.refine_weight.to_vector(),.006,2e-5);
        }
        if(mode==0 && filter) finite_difference(vk,camera,options);
        if(mode==0 && !filter) {
            bool rejected=false;
            try { (void)vr.sample_depth(vk,Tensor::zeros({1U,3U},Device::Vulkan),camera,options); }
            catch(const std::invalid_argument&) { rejected=true; }
            require(rejected,"GUT Vulkan point queries must not use the 3DGS primitive");
        }
    }
    for(const auto& [name,error] : parity_errors)
        std::cout<<"GUT parity "<<name<<" worst_relative_l2="<<error.first<<" worst_absolute="<<error.second<<'\n';
    std::cout<<"GUT Vulkan/CUDA forward, masked L1/SSIM, alpha and model-gradient parity passed\n";
}
}
