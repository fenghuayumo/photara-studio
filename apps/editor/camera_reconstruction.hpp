#pragma once

#include "sparse_view.hpp"
#include "camera_timeline.hpp"
#include "project/document.hpp"
#include "splat_render/renderer.hpp"
#include "io/image.hpp"
#include <limits>
#include <numeric>
#include <chrono>

namespace editor::camera_effects {

inline float reconstruction_smooth(float a, float b, float x) {
    const float u = std::clamp((x-a)/(b-a), 0.F, 1.F);
    return u*u*(3.F-2.F*u);
}
inline std::uint32_t reconstruction_hash(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; return x ^ (x >> 16);
}

// Small quarter-resolution bloom keeps tiny luminous splats visible at video resolution.
inline void reconstruction_bloom(photara::io::RgbImage& image, float strength) {
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

// Owns CPU attributes and a separate renderer: temporary geometry must never
// replace the editable model or contaminate the normal viewport renderer.
class Reconstruction {
public:
    explicit Reconstruction(const photara::splat::GaussianModel& source) {
        base_ = source;
        base_.means = source.means.cpu(); base_.log_scales = source.log_scales.cpu();
        base_.quaternions = source.quaternions.cpu(); base_.opacity_logits = source.opacity_logits.cpu();
        base_.sh = source.sh.cpu();
        if (source.filter_3d.is_valid()) base_.filter_3d = source.filter_3d.cpu();
        if (!base_.size() || !base_.sh.is_valid()) throw std::runtime_error("Scan reveal needs a Gaussian model");
        // Robust extent excludes extreme reconstruction floaters.
        std::vector<float> values;
        for (int c=0;c<3;++c) {
            values.clear();
            for (std::size_t i=0;i<base_.size();i+=std::max<std::size_t>(1,base_.size()/100000)) {
                const float value=base_.means.ptr<float>()[i*3+c];
                if (std::isfinite(value)) values.push_back(value);
            }
            if (values.empty()) throw std::runtime_error("Scan reveal has no finite geometry");
            std::sort(values.begin(),values.end());
            scale_=std::max(scale_,values[values.size()*98/100]-values[values.size()*2/100]);
        }
    }
    void attach(const photara::vk::Device& runtime) { renderer_.attach(runtime); }
    const std::vector<float>& arrival_times() const { return arrival_; }
    std::uint64_t model_upload_revision() const { return renderer_.model_upload_revision(); }
    std::array<double,3> frame_milliseconds() const { return frame_milliseconds_; }

    // Shared animation uniforms for direct GPU preview and offline RGB export.
    splat_render::DisplayOptions display_options(const photara::project::CameraEffectTrack& effect,
        float time, splat_render::DisplayOptions display = {}) const {
        if (!effect.contains(time)) { display.reconstruction={}; return display; }
        const float phase_scale=std::min(1.F,(effect.end_time-effect.start_time)*.85F/(effect.scan_hold+effect.scan_growth));
        display.reconstruction={true,time,effect.start_time,effect.end_time,
            effect.scan_hold*phase_scale,effect.scan_growth*phase_scale,
            scale_*.00036F*effect.point_size*.5F,effect.scan_density,effect.scan_glow};
        return display;
    }

    void prepare(const std::vector<photara::project::CameraKeyframe>& keys,
        std::span<const splat_render::ModelTransform> transforms,
        const photara::project::CameraEffectTrack& effect, float aspect) {
        effect.validate();
        if (keys.size()<2 || keys.back().time<=keys.front().time || !std::isfinite(aspect) || aspect<=0)
            throw std::invalid_argument("Scan reveal needs a valid camera path");
        std::vector<float> signature{float(effect.kind),effect.clip_start,aspect,effect.start_time,effect.end_time,effect.scan_hold,
            effect.scan_growth,effect.scan_range};
        for (const auto& k:keys) {
            signature.insert(signature.end(),{k.time,k.target[0],k.target[1],k.target[2],k.yaw,k.pitch,k.roll,
                k.distance,k.fov_degrees,k.ortho_height,k.fisheye_k1,float(k.projection),float(k.interpolation)});
        }
        for (const auto& t:transforms) {
            signature.push_back(float(t.end_index)); signature.push_back(t.scale);
            signature.insert(signature.end(),t.translation.begin(),t.translation.end());
            signature.insert(signature.end(),t.rotation.begin(),t.rotation.end());
        }
        if (signature==signature_) return;
        struct Sample { std::array<float,3> eye, forward, right, up; float time, tx, ty, ortho; int projection; };
        std::vector<Sample> samples;
        for (int j=0;j<=156;++j) {
            const float t=keys.front().time+(keys.back().time-keys.front().time)*float(j)/156.F;
            const auto k=camera_timeline::evaluate(keys,t);
            OrbitCamera orbit;
            orbit.target={k.target[0],k.target[1],k.target[2]}; orbit.yaw=k.yaw; orbit.pitch=k.pitch;
            orbit.roll=k.roll; orbit.distance=k.distance; orbit.fov_degrees=k.fov_degrees;
            orbit.ortho_height=k.ortho_height; orbit.fisheye_k1=k.fisheye_k1;
            orbit.projection=static_cast<EditorProjection>(k.projection);
            const auto camera=make_preview_camera(orbit,1000,unsigned(std::max(1.F,1000.F/aspect)));
            const auto& m=camera.world_to_camera;
            samples.push_back({camera.position,{m[2],m[6],m[10]}, {m[0],m[4],m[8]}, {m[1],m[5],m[9]},
                t,float(camera.width)/(2.F*camera.fx)*1.12F,float(camera.height)/(2.F*camera.fy)*1.12F,
                k.ortho_height,k.projection});
        }
        float world_scale=scale_;
        for (const auto& t:transforms) world_scale=std::max(world_scale,scale_*t.scale);
        arrival_.resize(base_.size());
        const float span=effect.end_time-effect.start_time;
        const float phase=std::min(effect.scan_hold+effect.scan_growth,span*.85F);
        // The wave originates at the shot's orbit target, independent of which
        // parts of the scene happen to be visible to the animated camera.
        const auto origin=camera_timeline::evaluate(keys,effect.start_time);
        float wave_radius=std::max(1e-6F,world_scale*effect.scan_range);
        for (std::size_t i=0;i<base_.size();++i) {
            std::array<float,3> point{base_.means.ptr<float>()[i*3],base_.means.ptr<float>()[i*3+1],base_.means.ptr<float>()[i*3+2]};
            if (!transforms.empty()) {
                const auto t=std::upper_bound(transforms.begin(),transforms.end(),i,
                    [](std::size_t index,const auto& placement) {return index<placement.end_index;});
                if (t==transforms.end()) throw std::invalid_argument("Scan placements do not cover the model");
                point=splat_render::transform_point(*t,point);
            }
            if (effect.kind == 2) {
                float distance2=0.F;
                for (int c=0;c<3;++c) distance2+=(point[c]-origin.target[c])*(point[c]-origin.target[c]);
                arrival_[i]=std::sqrt(distance2);
                wave_radius=std::max(wave_radius,arrival_[i]);
                continue;
            }
            float first=effect.end_time-phase;
            for (const auto& s:samples) {
                std::array<float,3> delta{};
                for (int c=0;c<3;++c) delta[c]=point[c]-s.eye[c];
                const auto dot=[&](const auto& v){return delta[0]*v[0]+delta[1]*v[1]+delta[2]*v[2];};
                const float z=dot(s.forward), x=dot(s.right), y=dot(s.up);
                bool visible=z>world_scale*.001F && z<world_scale*effect.scan_range;
                if (s.projection==0) visible=visible && std::abs(x)<z*s.tx && std::abs(y)<z*s.ty;
                else if (s.projection==1) visible=visible && std::abs(x)<s.ortho*aspect*.56F && std::abs(y)<s.ortho*.56F;
                else if (s.projection==3) visible=std::sqrt(x*x+y*y+z*z)<world_scale*effect.scan_range;
                // A conservative front hemisphere for fisheye reveal scheduling.
                if (visible) {
                    // Preserve the recorded camera timing. Compressing every
                    // arrival into the clip would reveal later rooms before
                    // the camera reaches them (the original short-lived effect).
                    first=s.time-keys.front().time+effect.start_time-.7F-.4F;
                    break;
                }
            }
            const float jitter=(float(reconstruction_hash(unsigned(i))%10000)/10000.F-.5F)*.18F;
            arrival_[i]=std::clamp(first+jitter,effect.start_time,effect.end_time-phase);
        }
        if (effect.kind==2) for (auto& distance:arrival_)
            distance=effect.start_time+(distance/wave_radius)*(span-phase);
        renderer_.set_reconstruction_schedule(arrival_);
        signature_=std::move(signature);
        last_frame_.clear();
    }

    photara::io::RgbImage render(const splat_render::Camera& camera,
        const std::vector<photara::project::CameraKeyframe>& keys,
        std::span<const splat_render::ModelTransform> transforms,
        const photara::project::CameraEffectTrack& effect,float time,
        const splat_render::DisplayOptions& display, bool cpu_reference = false, float schedule_aspect = 0.F) {
        prepare(keys,transforms,effect,schedule_aspect > 0.F ? schedule_aspect : float(camera.width)/camera.height);
        std::vector<float> frame{time,float(camera.width),float(camera.height),camera.fx,camera.fy,camera.cx,camera.cy,
            camera.k1,camera.k2,camera.k3,camera.k4,float(camera.model),display.near_clip,
            effect.clip_start,effect.point_size,effect.scan_density,effect.scan_glow,float(cpu_reference)};
        frame.insert(frame.end(),camera.world_to_camera.begin(),camera.world_to_camera.end());
        frame.insert(frame.end(),camera.position.begin(),camera.position.end());
        if (frame==last_frame_) return image_;
        const auto count=base_.size(), stride=base_.sh.shape()[1]*3;
        const float span=effect.end_time-effect.start_time;
        const float phase_scale=std::min(1.F,span*.85F/(effect.scan_hold+effect.scan_growth));
        const float hold=effect.scan_hold*phase_scale, growth=effect.scan_growth*phase_scale;
        if (!cpu_reference) {
            if (!gpu_model_loaded_) {
                splat_render::GaussianCloud cloud;
                cloud.count=unsigned(count); cloud.sh_degree=base_.sh_degree; cloud.sh_bases=unsigned(stride/3);
                cloud.means=base_.means.ptr<float>(); cloud.log_scales=base_.log_scales.ptr<float>();
                cloud.opacity_logits=base_.opacity_logits.ptr<float>(); cloud.quaternions=base_.quaternions.ptr<float>();
                cloud.sh=base_.sh.ptr<float>();
                cloud.render_method=static_cast<splat_render::RenderMethod>(base_.render_method);
                cloud.filter_mode=static_cast<splat_render::FilterMode>(base_.filter_mode);
                cloud.filter_3d=base_.filter_3d.is_valid() && base_.filter_3d.numel()==count ? base_.filter_3d.ptr<float>() : nullptr;
                if (!renderer_.upload(cloud,"camera-reconstruction")) throw std::runtime_error(renderer_.failure());
                renderer_.set_reconstruction_schedule(arrival_);
                gpu_model_loaded_=true;
            }
            if (!renderer_.set_model_transforms(transforms)) throw std::runtime_error(renderer_.failure());
            const auto settings=display_options(effect,time,display);
            splat_render::FrameTarget target;
            const auto draw_started=std::chrono::steady_clock::now();
            if (!renderer_.draw(camera,target,splat_render::Shading::gaussian,settings)) throw std::runtime_error(renderer_.failure());
            const auto readback_started=std::chrono::steady_clock::now();
            if (!renderer_.download_rgb(image_.pixels,image_.width,image_.height)) throw std::runtime_error(renderer_.failure());
            const auto bloom_started=std::chrono::steady_clock::now();
            if (effect.contains(time)) reconstruction_bloom(image_,effect.scan_glow/.7F*1.15F*(1.F-reconstruction_smooth(effect.end_time-1.F,effect.end_time,time)));
            const auto finished=std::chrono::steady_clock::now();
            frame_milliseconds_={std::chrono::duration<double,std::milli>(readback_started-draw_started).count(),
                std::chrono::duration<double,std::milli>(bloom_started-readback_started).count(),
                std::chrono::duration<double,std::milli>(finished-bloom_started).count()};
            last_frame_=std::move(frame);
            return image_;
        }
        scales_.resize(count*3); opacity_.resize(count); sh_.resize(base_.sh.numel());
        for (std::size_t i=0;i<count;++i) {
            const float age=time-arrival_[i];
            const bool complete=!effect.contains(time) || time>=effect.end_time || age>=hold+growth;
            if (complete) {
                std::copy_n(base_.log_scales.ptr<float>()+i*3,3,scales_.data()+i*3);
                opacity_[i]=base_.opacity_logits.ptr<float>()[i];
                std::copy_n(base_.sh.ptr<float>()+i*stride,stride,sh_.data()+i*stride);
                continue;
            }
            const float grow=reconstruction_smooth(hold,hold+growth*.88F,age);
            const float material=reconstruction_smooth(hold+growth*.04F,hold+growth,age);
            const float seed=float(reconstruction_hash(unsigned(i))%100000)/100000.F;
            const float illuminate=reconstruction_smooth(-.75F,-.12F,age);
            const float base_alpha=1.F/(1.F+std::exp(-base_.opacity_logits.ptr<float>()[i]));
            float alpha=age<-.75F ? (seed<.008F ? .4F*reconstruction_smooth(effect.start_time-.7F,effect.start_time+.3F,time) : .00001F)
                : std::max(base_alpha*grow,(seed<illuminate*effect.scan_density ? .72F : .00001F)*(1.F-grow));
            alpha=std::clamp(alpha,.000001F,.999999F);
            opacity_[i]=std::log(alpha/(1.F-alpha));
            const float radius=scale_*(age<-.75F ? .00026F : .00036F)*effect.point_size*.5F;
            const float log_radius=std::log(radius);
            const float cold[]{.025F,.48F,1.05F};
            for(int c=0;c<3;++c) {
                scales_[i*3+c]=log_radius*(1.F-grow)+base_.log_scales.ptr<float>()[i*3+c]*grow;
                const float raw=.5F+.28209479177387814F*base_.sh.ptr<float>()[i*stride+c];
                const float glow=(1.F-material)*illuminate*effect.scan_glow/.7F;
                sh_[i*stride+c]=(cold[c]*(1.F-material)+raw*material+glow*(c==0 ? .01F : .16F)-.5F)/.28209479177387814F;
            }
            for(std::size_t c=3;c<stride;++c) sh_[i*stride+c]=base_.sh.ptr<float>()[i*stride+c]*material;
        }
        splat_render::GaussianCloud cloud;
        cloud.count=unsigned(count); cloud.sh_degree=base_.sh_degree; cloud.sh_bases=unsigned(stride/3);
        cloud.means=base_.means.ptr<float>(); cloud.log_scales=scales_.data(); cloud.opacity_logits=opacity_.data();
        cloud.quaternions=base_.quaternions.ptr<float>(); cloud.sh=sh_.data();
        cloud.render_method=static_cast<splat_render::RenderMethod>(base_.render_method);
        cloud.filter_mode=static_cast<splat_render::FilterMode>(base_.filter_mode);
        cloud.filter_3d=base_.filter_3d.is_valid() && base_.filter_3d.numel()==count ? base_.filter_3d.ptr<float>() : nullptr;
        gpu_model_loaded_=false;
        renderer_.clear_model(); // CPU reference only; interactive/export animation stays on the GPU.
        if (!renderer_.upload(cloud,"camera-reconstruction") || !renderer_.set_model_transforms(transforms))
            throw std::runtime_error(renderer_.failure());
        splat_render::FrameTarget target;
        if (!renderer_.draw(camera,target,splat_render::Shading::gaussian,display) ||
            !renderer_.download_rgb(image_.pixels,image_.width,image_.height)) throw std::runtime_error(renderer_.failure());
        if (effect.contains(time)) reconstruction_bloom(image_,effect.scan_glow/.7F*1.15F*(1.F-reconstruction_smooth(effect.end_time-1.F,effect.end_time,time)));
        last_frame_=std::move(frame);
        return image_;
    }
private:
    photara::splat::GaussianModel base_;
    splat_render::Renderer renderer_;
    bool gpu_model_loaded_{};
    std::array<double,3> frame_milliseconds_{};
    float scale_{.001F};
    std::vector<float> scales_,opacity_,sh_,arrival_,signature_,last_frame_;
    photara::io::RgbImage image_;
};
} // namespace editor::camera_effects
