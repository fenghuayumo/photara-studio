// Offline, per-Gaussian architectural reconstruction demo. Source projects are read only.
#include "sparse_view.hpp"
#include "camera_timeline.hpp"
#include "splat_render/renderer.hpp"
#include "project/document.hpp"
#include "splat/trainer.hpp"
#include "io/image.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

using Key = photara::project::CameraKeyframe;
using V3 = std::array<float, 3>;

editor::OrbitCamera orbit_for(const Key& key) {
    editor::OrbitCamera orbit;
    orbit.target = {key.target[0], key.target[1], key.target[2]};
    orbit.yaw = key.yaw; orbit.pitch = key.pitch; orbit.roll = key.roll;
    orbit.distance = key.distance; orbit.fov_degrees = key.fov_degrees;
    orbit.ortho_height = key.ortho_height; orbit.fisheye_k1 = key.fisheye_k1;
    orbit.projection = static_cast<editor::EditorProjection>(key.projection);
    return orbit;
}
splat_render::Camera camera_for(const Key& key, unsigned width, unsigned height) {
    const auto source = editor::make_preview_camera(orbit_for(key), width, height);
    splat_render::Camera camera;
    camera.world_to_camera = source.world_to_camera; camera.position = source.position;
    camera.width = width; camera.height = height;
    camera.fx = source.fx; camera.fy = source.fy; camera.cx = source.cx; camera.cy = source.cy;
    camera.k1 = source.k1; camera.k2 = source.k2; camera.k3 = source.k3; camera.k4 = source.k4;
    camera.model = key.projection;
    if (key.projection == 1) {
        camera.fx = camera.fy = float(height) / std::max(.0001F, key.ortho_height);
        camera.cx = width * .5F; camera.cy = height * .5F;
    }
    return camera;
}
float smooth(float a, float b, float x) {
    const float u = std::clamp((x-a)/(b-a), 0.F, 1.F);
    return u*u*(3.F-2.F*u);
}
std::uint32_t hash_index(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; return x ^ (x >> 16);
}

// Small quarter-resolution bloom keeps tiny luminous splats visible at video resolution.
void bloom(photara::io::RgbImage& image, float strength) {
    if (strength <= 0.F) return;
    const unsigned w = (image.width+3)/4, h = (image.height+3)/4;
    std::vector<float> a(w*h*3, 0.F), b(a.size());
    for (unsigned y=0; y<image.height; ++y) for (unsigned x=0; x<image.width; ++x) {
        const auto src=(std::size_t(y)*image.width+x)*3, dst=(std::size_t(y/4)*w+x/4)*3;
        const float cyan = std::max(0.F, (float(image.pixels[src+1])+image.pixels[src+2])*.5F-image.pixels[src]-35.F)/220.F;
        for (int c=0;c<3;++c) a[dst+c] += cyan*image.pixels[src+c]/16.F;
    }
    for (int pass=0;pass<4;++pass) {
        for (unsigned y=0;y<h;++y) for (unsigned x=0;x<w;++x) for (int c=0;c<3;++c) {
            float value=0.F;
            for (int k=-3;k<=3;++k) {
                const unsigned xx=pass%2 ? x : std::clamp(int(x)+k,0,int(w)-1);
                const unsigned yy=pass%2 ? std::clamp(int(y)+k,0,int(h)-1) : y;
                value += a[(std::size_t(yy)*w+xx)*3+c];
            }
            b[(std::size_t(y)*w+x)*3+c]=value/7.F;
        }
        a.swap(b);
    }
    for (unsigned y=0;y<image.height;++y) for (unsigned x=0;x<image.width;++x) for(int c=0;c<3;++c) {
        const auto i=(std::size_t(y)*image.width+x)*3+c;
        image.pixels[i]=std::uint8_t(std::min(255.F,image.pixels[i]+strength*a[(std::size_t(y/4)*w+x/4)*3+c]));
    }
}

int main(int argc, char** argv) try {
    if (argc < 3) throw std::runtime_error("Usage: photara_camera_demo project.ascan output-directory [inspect|preview|render] [width height fps]");
    std::cout << std::unitbuf;
    const auto project = std::filesystem::u8path(argv[1]);
    const auto output = std::filesystem::u8path(argv[2]);
    const std::string mode = argc>3 ? argv[3] : "preview";
    if (mode!="inspect" && mode!="preview" && mode!="baseline" && mode!="render") throw std::runtime_error("Unknown mode");
    const unsigned width=argc>4 ? std::stoul(argv[4]) : 1280, height=argc>5 ? std::stoul(argv[5]) : 720;
    const unsigned fps=argc>6 ? std::stoul(argv[6]) : 30;
    if (width<2 || height<2 || width>8192 || height>8192 || fps<1 || fps>120) throw std::runtime_error("Invalid video dimensions or fps");
    const auto archive = photara::project::Archive::open(project);
    const auto settings = photara::project::read_settings(archive);
    const auto& keys = settings.camera_keyframes;
    if (keys.size()<2) throw std::runtime_error("Project needs at least two camera views");
    for (std::size_t i=0;i<keys.size();++i) {
        if (!std::isfinite(keys[i].time) || (i && keys[i].time<=keys[i-1].time)) throw std::runtime_error("Invalid camera keyframe times");
        if (keys[i].projection!=0) throw std::runtime_error("This architectural demo currently requires perspective keyframes");
    }
    const float duration=keys.back().time-keys.front().time;
    if (duration<2.F || duration>600.F) throw std::runtime_error("Demo duration must be 2 to 600 seconds");
    std::cout << "Project: " << project << "\nViews: " << keys.size() << ", duration: " << duration << "s\n";
    std::cout << "Gaussian sources: " << settings.gaussian_models.size() << "\n";
    // Avoid silently rendering incorrect placements. Multi-source projects require the Studio merger.
    if (settings.gaussian_models.size()>1) throw std::runtime_error("Multi-source project needs a merged Gaussian model");
    if (!settings.gaussian_models.empty()) {
        const auto& g=settings.gaussian_models.front();
        for (int c=0;c<3;++c) if (g.offset[c]!=0 || g.rotation_degrees[c]!=0) throw std::runtime_error("Nonidentity source placement is not supported by this offline tool");
        if (g.scale!=1 || !g.visible) throw std::runtime_error("Nonidentity or hidden source placement is not supported by this offline tool");
    }
    for (float t : {0.F,duration*.25F,duration*.5F,duration*.75F,duration}) {
        const auto key=editor::camera_timeline::evaluate(keys,t+keys.front().time);
        const auto camera=camera_for(key,width,height);
        std::cout << "t=" << t << " eye=" << camera.position[0] << ',' << camera.position[1] << ',' << camera.position[2]
            << " target=" << key.target[0] << ',' << key.target[1] << ',' << key.target[2] << "\n";
    }
    if (mode=="inspect") return 0;
    auto model=photara::splat::decode_gaussians(archive.chunk(photara::project::ChunkType::gaussians),tinytensor::Device::CPU);
    std::cout << "Gaussians: " << model.size() << ", render method: " << int(model.render_method) << ", filter: " << int(model.filter_mode) << "\n";
    const std::size_t count=model.size(), bases=model.sh.shape()[1], sh_stride=bases*3;
    if (!count) throw std::runtime_error("Project has no Gaussians");
    const float* means=model.means.ptr<float>();
    std::vector<float> base_scales(model.log_scales.ptr<float>(),model.log_scales.ptr<float>()+count*3);
    std::vector<float> base_opacity(model.opacity_logits.ptr<float>(),model.opacity_logits.ptr<float>()+count);
    std::vector<float> base_sh(model.sh.ptr<float>(),model.sh.ptr<float>()+count*sh_stride);
    float* scales=model.log_scales.ptr<float>(); float* opacity=model.opacity_logits.ptr<float>(); float* sh=model.sh.ptr<float>();
    // Robust scene scale: exclude distant floaters rather than using raw AABB extremes.
    std::vector<float> coordinates;
    float extents[3]{};
    for(int c=0;c<3;++c) {
        coordinates.clear();
        for(std::size_t i=0;i<count;i+=31) if(std::isfinite(means[i*3+c])) coordinates.push_back(means[i*3+c]);
        std::sort(coordinates.begin(),coordinates.end());
        if(coordinates.empty()) throw std::runtime_error("Nonfinite Gaussian positions");
        extents[c]=coordinates[coordinates.size()*98/100]-coordinates[coordinates.size()*2/100];
    }
    const float scene_scale=std::max({extents[0],extents[1],extents[2],.001F});
    std::cout << "Robust scene extent: " << extents[0] << ',' << extents[1] << ',' << extents[2] << "\n";
    struct Sample { V3 position; float time; };
    std::vector<Sample> samples;
    for(int j=0;j<=156;++j) {
        const float t=duration*j/156.F;
        const auto camera=camera_for(editor::camera_timeline::evaluate(keys,keys.front().time+t),width,height);
        samples.push_back({camera.position,t});
    }
    std::vector<float> arrival(count);
    #pragma omp parallel for
    for(std::int64_t index=0;index<std::int64_t(count);++index) {
        const auto i=std::size_t(index);
        float first=duration*2.F;
        for(const auto& sample:samples) {
            float distance=0.F;
            for(int c=0;c<3;++c) {const float d=means[i*3+c]-sample.position[c];distance+=d*d;}
            // Earliest expanding wave from the travelled path. This remains stable on revisits.
            first=std::min(first,sample.time*.72F+std::sqrt(distance)/(scene_scale*.085F));
        }
        arrival[i]=first;
    }
    coordinates=arrival; std::sort(coordinates.begin(),coordinates.end());
    const float low=coordinates[count/100], high=coordinates[count*99/100];
    const float finish=duration*.80F;
    for(std::size_t i=0;i<count;++i) {
        const float jitter=(float(hash_index(unsigned(i))%10000)/10000.F-.5F)*.18F;
        arrival[i]=.85F+std::clamp((arrival[i]-low)/std::max(.01F,high-low),0.F,1.F)*(finish-2.F)+jitter;
    }
    coordinates.clear(); coordinates.shrink_to_fit();
    std::filesystem::create_directories(output);
    std::ofstream manifest(output/"render_info.txt");
    manifest << "Source: " << project << "\nViews: " << keys.size() << "\nDuration: " << duration << "\nSize: " << width << 'x' << height << "\nFPS: " << fps
        << "\nEffect: per-Gaussian route wave, sparse cyan seeds, spherical-to-anisotropic growth, material restoration, bloom.\n";
    photara::vk::DeviceRequest request; request.want.graphics=true;
    auto device=photara::vk::Device::create(request);
    splat_render::Renderer renderer; renderer.attach(device);
    splat_render::GaussianCloud cloud;
    cloud.count=unsigned(count); cloud.sh_degree=model.sh_degree; cloud.sh_bases=unsigned(bases);
    cloud.means=means; cloud.log_scales=scales; cloud.opacity_logits=opacity;
    cloud.quaternions=model.quaternions.ptr<float>(); cloud.sh=sh;
    cloud.render_method=static_cast<splat_render::RenderMethod>(model.render_method);
    cloud.filter_mode=static_cast<splat_render::FilterMode>(model.filter_mode);
    cloud.filter_3d=model.filter_3d.is_valid() && model.filter_3d.numel()==count ? model.filter_3d.ptr<float>() : nullptr;
    std::vector<float> times;
    if(mode=="preview" || mode=="baseline") times={0.F,1.5F,3.F,5.F,8.F,12.F,16.F,20.F,duration};
    else for(int i=0,n=int(std::ceil(duration*fps))+1;i<n;++i) times.push_back(std::min(duration,float(i)/fps));
    const auto started=std::chrono::steady_clock::now();
    for(std::size_t frame=0;frame<times.size();++frame) {
        const float time=times[frame];
        #pragma omp parallel for
        for(std::int64_t index=0;index<std::int64_t(count);++index) {
            const auto i=std::size_t(index);
            const float age=mode=="baseline" ? 10000.F : time-arrival[i];
            const float grow=smooth(-.05F,1.20F,age), material=smooth(.05F,1.25F,age);
            const auto hash=hash_index(unsigned(i));
            const float seed=float(hash%100000)/100000.F;
            const bool dormant_seed=seed<.008F;
            const float illuminate=smooth(-.75F,-.12F,age);
            const float base_alpha=1.F/(1.F+std::exp(-base_opacity[i]));
            float alpha;
            if(age>=1.65F) alpha=base_alpha;
            else if(age<-.75F) alpha=dormant_seed ? .4F*smooth(0.F,1.F,time) : .00001F;
            else alpha=std::max(base_alpha*grow,(seed<illuminate*.22F ? .72F : .00001F)*(1.F-grow));
            alpha=std::clamp(alpha,.000001F,.999999F);
            opacity[i]=age>=1.65F ? base_opacity[i] : std::log(alpha/(1.F-alpha));
            const float radius=scene_scale*(age<-.75F ? .00026F : .00036F);
            const float log_radius=std::log(radius);
            const float glow=(1.F-material)*illuminate;
            const float cold[3]{.025F,.48F,1.05F};
            for(int c=0;c<3;++c) {
                scales[i*3+c]=age>=1.65F ? base_scales[i*3+c] : log_radius*(1.F-grow)+base_scales[i*3+c]*grow;
                const float raw=.5F+.28209479177387814F*base_sh[i*sh_stride+c];
                const float value=cold[c]*(1.F-material)+raw*material+glow*(c==0 ? .01F : .16F);
                sh[i*sh_stride+c]=age>=1.65F ? base_sh[i*sh_stride+c] : (value-.5F)/.28209479177387814F;
            }
            for(std::size_t c=3;c<sh_stride;++c) sh[i*sh_stride+c]=base_sh[i*sh_stride+c]*material;
        }
        // Geometry changes can multiply the tile count suddenly. Re-seed the
        // rasterizer's indirect-sort capacity instead of exporting an overflow frame.
        renderer.clear_model();
        if(!renderer.upload(cloud,"route-reconstruction")) throw std::runtime_error(renderer.failure());
        const auto camera=camera_for(editor::camera_timeline::evaluate(keys,keys.front().time+time),width,height);
        splat_render::FrameTarget target; splat_render::DisplayOptions display; display.near_clip=.05F;
        if(!renderer.draw(camera,target,splat_render::Shading::gaussian,display)) throw std::runtime_error(renderer.failure());
        photara::io::RgbImage image;
        if(!renderer.download_rgb(image.pixels,image.width,image.height)) throw std::runtime_error(renderer.failure());
        if(mode!="baseline") bloom(image,1.15F*(1.F-smooth(finish,finish+2.F,time)));
        char filename[64]; std::snprintf(filename,sizeof(filename),mode!="render" ? "preview_%02zu.png" : "frame_%05zu.png",frame);
        photara::io::save_rgb_png(image,output/filename);
        const float seconds=std::chrono::duration<float>(std::chrono::steady_clock::now()-started).count();
        std::cout << "Frame " << frame+1 << '/' << times.size() << " t=" << time << " elapsed=" << seconds << "s\n";
    }
    return 0;
} catch(const std::exception& error) {std::cerr << "Camera demo: " << error.what() << '\n'; return 1;}
