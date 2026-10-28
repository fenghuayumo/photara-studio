// Offline, per-Gaussian architectural reconstruction demo. Source projects are read only.
#include "sparse_view.hpp"
#include "camera_timeline.hpp"
#include "camera_reconstruction.hpp"
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
int main(int argc, char** argv) try {
    if (argc < 3) throw std::runtime_error("Usage: photara_camera_demo project.ascan output-directory [inspect|preview|render|baseline|benchmark|benchmark-preview] [width height fps]");
    std::cout << std::unitbuf;
    const auto project = std::filesystem::u8path(argv[1]);
    const auto output = std::filesystem::u8path(argv[2]);
    const std::string mode = argc>3 ? argv[3] : "preview";
    if (mode!="inspect" && mode!="preview" && mode!="baseline" && mode!="render" && mode!="benchmark" && mode!="benchmark-preview") throw std::runtime_error("Unknown mode");
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
    auto effect=settings.camera_effect.enabled && settings.camera_effect.reconstruction()
        ? settings.camera_effect : photara::project::camera_scan_demo(keys.front().time,keys.back().time);
    editor::camera_effects::Reconstruction reconstruction(model);
    photara::vk::DeviceRequest request; request.want.graphics=true;
    auto device=photara::vk::Device::create(request);
    reconstruction.attach(device);
    splat_render::Renderer preview_renderer;
    if (mode=="benchmark-preview") {
        preview_renderer.attach(device);
        splat_render::GaussianCloud cloud;
        cloud.count=unsigned(model.size()); cloud.sh_degree=model.sh_degree;
        cloud.sh_bases=unsigned(model.sh.shape()[1]);
        cloud.means=model.means.ptr<float>(); cloud.log_scales=model.log_scales.ptr<float>();
        cloud.quaternions=model.quaternions.ptr<float>(); cloud.opacity_logits=model.opacity_logits.ptr<float>();
        cloud.sh=model.sh.ptr<float>(); cloud.render_method=static_cast<splat_render::RenderMethod>(model.render_method);
        cloud.filter_mode=static_cast<splat_render::FilterMode>(model.filter_mode);
        cloud.filter_3d=model.filter_3d.is_valid() ? model.filter_3d.ptr<float>() : nullptr;
        if (!preview_renderer.upload(cloud,"demo-preview")) throw std::runtime_error(preview_renderer.failure());
        reconstruction.prepare(keys,{},effect,float(width)/height);
        preview_renderer.set_reconstruction_schedule(reconstruction.arrival_times());
    }
    std::filesystem::create_directories(output);
    std::ofstream manifest(output/"render_info.txt");
    manifest << "Source: " << project << "\nViews: " << keys.size() << "\nDuration: " << duration << "\nSize: " << width << 'x' << height << "\nFPS: " << fps
        << "\nEffect: shared Studio camera-path reconstruction renderer.\n";
    std::vector<float> times;
    if(mode=="preview" || mode=="baseline") times={0.F,1.5F,3.F,5.F,8.F,12.F,16.F,20.F,duration};
    else if (mode=="benchmark" || mode=="benchmark-preview") for (unsigned i=0;i<120;++i) times.push_back(std::min(duration,8.F+float(i)/fps));
    else for(int i=0,n=int(std::ceil(duration*fps))+1;i<n;++i) times.push_back(std::min(duration,float(i)/fps));
    const auto started=std::chrono::steady_clock::now();
    for(std::size_t frame=0;frame<times.size();++frame) {
        const float time=times[frame];
        const auto camera=camera_for(editor::camera_timeline::evaluate(keys,keys.front().time+time),width,height);
        splat_render::DisplayOptions display; display.near_clip=.05F;
        if (mode=="benchmark-preview") {
            const auto frame_started=std::chrono::steady_clock::now();
            splat_render::FrameTarget target;
            if (!preview_renderer.draw(camera,target,splat_render::Shading::gaussian,
                reconstruction.display_options(effect,keys.front().time+time,display))) throw std::runtime_error(preview_renderer.failure());
            std::cout << "Frame " << frame+1 << '/' << times.size() << " draw_ms="
                << std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-frame_started).count() << '\n';
            continue;
        }
        auto image=reconstruction.render(camera,keys,{},effect,
            mode=="baseline" ? effect.end_time : keys.front().time+time,display);
        char filename[64]; std::snprintf(filename,sizeof(filename),mode!="render" ? "preview_%02zu.png" : "frame_%05zu.png",frame);
        if (mode!="benchmark") photara::io::save_rgb_png(image,output/filename);
        const float seconds=std::chrono::duration<float>(std::chrono::steady_clock::now()-started).count();
        std::cout << "Frame " << frame+1 << '/' << times.size() << " t=" << time << " elapsed=" << seconds << "s\n";
        if (mode=="benchmark") {
            const auto timing=reconstruction.frame_milliseconds();
            std::cout << "draw_ms=" << timing[0] << " readback_ms=" << timing[1] << " bloom_ms=" << timing[2] << '\n';
        }
    }
    return 0;
} catch(const std::exception& error) {std::cerr << "Camera demo: " << error.what() << '\n'; return 1;}
