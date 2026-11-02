#pragma once
#include <array>

// Independent double-precision inverse-ray finite differences. A single loss
// pixel lets the renderer's per-pixel ADC norm be recovered from its mean
// gradient, without reimplementing the ray-response backward.
inline void fisheye_adc_regression() {
    using namespace gut_validation;
    Camera camera;
    camera.model=CameraModel::opencv_fisheye;
    camera.width=128;camera.height=96;
    camera.fx=30;camera.fy=28;camera.cx=64;camera.cy=48;
    camera.world_to_camera={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    for (auto device:{Device::CUDA,Device::Vulkan})
    for (bool transformed:{false,true})
    for (bool distorted:{false,true})
    for (const auto& position:std::vector<std::vector<float>>{
             {0,0,2},{.7F,.4F,1.2F},{2,.3F,.002F},{-1,.8F,.02F}}) {
        auto seed=model(device,false,1);
        camera.world_to_camera=transformed
            ? std::array<float,16>{0,0,-1,0,0,1,0,0,1,0,0,0,.2F,-.1F,.3F,1}
            : std::array<float,16>{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        std::vector<float> world(3);
        for(int j=0;j<3;++j) {
            camera.position[j]=0;
            for(int i=0;i<3;++i) {
                world[j]+=camera.world_to_camera[4*j+i]*(position[i]-camera.world_to_camera[12+i]);
                camera.position[j]-=camera.world_to_camera[4*j+i]*camera.world_to_camera[12+i];
            }
        }
        camera.k1=distorted?.02F:0.F;camera.k2=distorted?-.001F:0.F;
        camera.k3=distorted?-.0002F:0.F;camera.k4=distorted?.00003F:0.F;
        seed.means=Tensor::from_vector(world,{1U,3U},device);
        seed.log_scales=Tensor::from_vector(std::vector<float>{-.9F,-1.4F,-1.8F},{1U,3U},device);
        seed.opacity_logits=Tensor::from_vector(std::vector<float>{0.F},{1U,1U},device);
        const double radius=std::hypot(std::hypot(position[0],position[1]),position[2]);
        const double transverse=std::hypot(position[0],position[1]);
        const double theta=std::atan2(transverse,position[2]);
        const double q=theta*theta;
        const double theta_d=theta*(1+q*(camera.k1+q*(camera.k2+q*(camera.k3+q*camera.k4))));
        const double u=transverse ? theta_d*position[0]/transverse : 0;
        const double v=transverse ? theta_d*position[1]/transverse : 0;
        // Derivative of the constant-distance inverse equidistant camera map.
        auto point=[&](double px,double py) {
            const double x=(px-camera.cx)/camera.fx,y=(py-camera.cy)/camera.fy;
            const double a=std::hypot(x,y);
            double angle=a;
            for(int i=0;i<12;++i) {
                const double q=angle*angle;
                angle-=(angle*(1+q*(camera.k1+q*(camera.k2+q*(camera.k3+q*camera.k4))))-a)/
                    (1+q*(3*camera.k1+q*(5*camera.k2+q*(7*camera.k3+9*q*camera.k4))));
            }
            const double s=a>1e-12 ? std::sin(angle)/a : 1.;
            return std::array<double,3>{radius*x*s,radius*y*s,radius*std::cos(angle)};
        };
        const double px=camera.cx+camera.fx*u,py=camera.cy+camera.fy*v,h=1e-3;
        const auto up=point(px+h,py),um=point(px-h,py);
        const auto vp=point(px,py+h),vm=point(px,py-h);
        Rasterizer raster;RasterizeOptions options;options.require_depth=false;options.active_sh_degree=0;
        auto frame=raster.forward(seed,camera,options);
        const unsigned x=unsigned(std::round(px)+(position[0]<0 ? 2 : -2)),y=unsigned(std::round(py))+1;
        const std::size_t pixel=y*camera.width+x,pixels=camera.width*camera.height;
        const float alpha=frame.alpha.to_vector().at(pixel);
        require(alpha>1e-3F,"Fisheye ADC fixture has no contributing ray");
        std::vector<float> loss(3*pixels,0.F);loss[pixel]=1.F;
        auto gradient=raster.backward(seed,frame,
            Tensor::from_vector(loss,{3U,camera.height,camera.width},device),
            Tensor::zeros({camera.height,camera.width},device),{},{});
        const auto gm=gradient.means.to_vector();
        double gu=0,gv=0;
        for(int k=0;k<3;++k) {
            double camera_gradient=0;
            for(int j=0;j<3;++j)camera_gradient+=camera.world_to_camera[4*j+k]*gm[j];
            gu+=camera_gradient*(up[k]-um[k])/(2*h);
            gv+=camera_gradient*(vp[k]-vm[k])/(2*h);
        }
        const double expected=std::hypot(camera.width*gu,camera.height*gv)/alpha;
        const double actual=gradient.refine_weight.to_vector()[0];
        std::cout<<"fisheye ADC z="<<position[2]<<" expected="<<expected<<" actual="<<actual<<std::endl;
        require(std::isfinite(actual)&&std::abs(actual-expected)<.003*expected+1e-4,
                "Fisheye ADC must use the full inverse projection tangent, including camera Z");
    }
}
