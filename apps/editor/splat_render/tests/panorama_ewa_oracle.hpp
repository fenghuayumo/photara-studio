#pragma once

// Independent double-precision EWA oracle: project each Gaussian once and
// evaluate every pixel. No tile walker or shader projection helper is used.
SplatForwardOutput panorama_ewa_reference(const Scene& scene,const SplatCamera& c,const SplatSettings& s) {
    const unsigned pixels=c.width*c.height;
    SplatForwardOutput out;out.color.assign(3*pixels,0);out.alpha.assign(pixels,0);
    struct Projected {double x,y,a,b,d,opacity;Vec rgb;};
    std::vector<Projected> projected;
    Mat world_to_camera{};
    for(int row=0;row<3;++row)for(int col=0;col<3;++col)
        world_to_camera[row][col]=c.world_to_camera[4*col+row];
    for(std::size_t i=0;i<scene.opacity.size();++i) {
        Vec delta{scene.means[3*i]-c.center[0],scene.means[3*i+1]-c.center[1],scene.means[3*i+2]-c.center[2]};
        const auto t=mul(world_to_camera,delta);
        const double h=std::hypot(t[0],t[2]),l2=dot(t,t),h2=h*h;
        require(h>1e-5 && l2>1e-12,"Invalid EWA oracle fixture");
        const double su=c.width/(2*3.141592653589793),sv=c.height/3.141592653589793;
        Vec du{su*t[2]/h2,0,-su*t[0]/h2};
        Vec dv{-sv*t[1]*t[0]/(h*l2),sv*h/l2,-sv*t[1]*t[2]/(h*l2)};
        Vec wu{},wv{};
        for(int j=0;j<3;++j)for(int k=0;k<3;++k) {
            wu[j]+=du[k]*world_to_camera[k][j];wv[j]+=dv[k]*world_to_camera[k][j];
        }
        const Mat R=rotation(scene.rotations.data()+4*i);
        const auto u=mul(R,wu),v=mul(R,wv);
        double xx=s.kernel_size,xy=0,yy=s.kernel_size;
        for(int axis=0;axis<3;++axis) {
            const double variance=std::pow(double(scene.scales[3*i+axis])*s.scale_modifier,2);
            xx+=variance*u[axis]*u[axis];xy+=variance*u[axis]*v[axis];yy+=variance*v[axis]*v[axis];
        }
        const double lambda = .5*(xx+yy) + std::hypot(.5*(xx-yy),xy);
        const double limit = std::min(1024.0,double(std::min(c.width,c.height)));
        const double scale = std::min(1.0,limit*limit/(8*lambda));
        xx*=scale; xy*=scale; yy*=scale;
        const double det=xx*yy-xy*xy;
        projected.push_back({(std::atan2(t[0],t[2])/(2*3.141592653589793)+.5)*c.width,
            (std::atan2(t[1],h)/3.141592653589793+.5)*c.height,
            yy/det,-xy/det,xx/det,scene.opacity[i],
            {scene.colors[3*i],scene.colors[3*i+1],scene.colors[3*i+2]}});
    }
    // Fixtures use one Gaussian; compositing a periodic copy would violate
    // this oracle even if its individual 2D response is otherwise correct.
    require(projected.size()==1,"Single-Gaussian EWA oracle expected");
    for(unsigned y=0;y<c.height;++y)for(unsigned x=0;x<c.width;++x) {
        const auto& g=projected[0];double dx=g.x-x,dy=g.y-y;
        dx-=c.width*std::floor((dx+c.width*.5)/c.width);
        const double q=g.a*dx*dx+2*g.b*dx*dy+g.d*dy*dy;
        double alpha=std::min(double(s.alpha_clip),g.opacity*std::exp(-.5*q));
        if(alpha<1./255.)alpha=0;
        const unsigned pixel=y*c.width+x;out.alpha[pixel]=float(alpha);
        for(unsigned channel=0;channel<3;++channel)
            out.color[channel*pixels+pixel]=float(alpha*g.rgb[channel]+(1-alpha)*s.background[channel]);
    }
    return out;
}

void panorama_ewa_tests(bool validation) {
    photara::vk::DeviceRequest request;
    request.validation=validation;request.synchronization_validation=validation;
    auto device=photara::vk::Device::create(request);
    ContextOptions options;
    options.external_device.instance=device.instance();
    options.external_device.physical_device=device.physical();
    options.external_device.device=device.handle();
    options.external_device.queue=device.queue();
    options.external_device.queue_family=device.queue_family();
    Context context(options);
    for(unsigned width:{7U,35U,96U})for(float opacity:{.006F,.9F})
    for(Vec mean:{Vec{0,2,.02},Vec{.02,-2,.03},Vec{.01,.1,-2},Vec{.1,.1,2}})
    for(bool anisotropic:{false,true}) {
        Scene scene;scene.means={float(mean[0]),float(mean[1]),float(mean[2])};
        scene.scales=anisotropic ? std::vector<float>{.2F,.6F,.1F} : std::vector<float>{.2F,.2F,.2F};
        scene.rotations.resize(4);if(!anisotropic)scene.rotations={1,0,0,0};
        scene.opacity={opacity};scene.colors={.8F,.3F,.2F};
        auto camera=scene.camera(3);camera.width=width;camera.height=63;
        camera.fx=width/(2.F*3.14159265359F);camera.fy=63.F/3.14159265359F;
        camera.cx=width*.5F;camera.cy=31.5F;
        SplatSettings settings;settings.primitive=SplatPrimitive::gaussian_3dgs;
        settings.kernel_size=.3F;settings.compensate_2d=false;
        settings.background[0]=.13F;settings.background[1]=.07F;settings.background[2]=.11F;
        const auto expected=panorama_ewa_reference(scene,camera,settings);
        SplatRasterizer raster(context);
        raster.upload_model(scene.gaussians());
        for(bool geometry:{false,true}) {
            settings.need_depth=geometry;
            const auto actual=raster.render(camera,settings);
            for (std::size_t p=0;p<actual.alpha.size();++p) if (std::abs(actual.alpha[p]-expected.alpha[p])>2e-4F) {
                std::cerr<<"EWA fixture width="<<width<<" opacity="<<opacity<<" mean="<<mean[0]<<","<<mean[1]<<","<<mean[2]<<" anisotropic="<<anisotropic<<" geometry="<<geometry<<" pixel="<<p%width<<","<<p/width<<'\n';
                break;
            }
            close(actual.alpha,expected.alpha,2e-4,"panorama EWA alpha");
            close(actual.color,expected.color,2e-4,"panorama EWA RGB");
            require(actual.instance_count<=((width+15)/16)*4U,"Duplicate periodic EWA tile");
        }
        settings.need_depth=false;
        std::vector<std::uint8_t> rgba;raster.render_rgba(camera,settings,rgba);
        for(unsigned p=0;p<camera.width*camera.height;++p)for(unsigned channel=0;channel<3;++channel)
            require(std::abs(int(rgba[4*p+channel])-int(std::clamp(expected.color[channel*camera.width*camera.height+p],0.F,1.F)*255.F+.5F))<=1,
                    "Panorama packed EWA RGB differs from oracle");
    }
    require(device.validation_error_count()==0,"Vulkan validation error in panorama EWA test");
    std::cout<<"Panorama EWA single-contributor oracle passed (wide footprints, seam, partial tiles, RGBA)\n";
}
