#pragma once

#include "sparse_view.hpp"
#include "splat_render/renderer.hpp"
#include "project/camera_effect.hpp"
#include "io/image.hpp"
#include "core/camera_projection.hpp"
#include <limits>

namespace editor::camera_effects {

// Use the resident model and another frame slot at the viewport's existing
// extent. Resizing here would invalidate images already queued in ImGui.
template <typename Key>
inline bool capture_keyframe_thumbnail(splat_render::Renderer& renderer,const Key& key,
    unsigned width,unsigned height,float near_clip,photara::io::RgbImage& image) {
    if (!renderer.has_frame_size(width,height)) return false;
    OrbitCamera orbit;
    orbit.target={key.target[0],key.target[1],key.target[2]};
    orbit.yaw=key.yaw; orbit.pitch=key.pitch; orbit.roll=key.roll; orbit.distance=key.distance;
    orbit.fov_degrees=key.fov_degrees; orbit.ortho_height=key.ortho_height;
    orbit.fisheye_k1=key.fisheye_k1; orbit.projection=static_cast<EditorProjection>(key.projection);
    const auto source=make_preview_camera(orbit,width,height);
    splat_render::Camera camera;
    camera.world_to_camera=source.world_to_camera; camera.position=source.position;
    camera.width=width; camera.height=height;
    camera.fx=source.fx; camera.fy=source.fy; camera.cx=source.cx; camera.cy=source.cy;
    camera.k1=source.k1; camera.k2=source.k2; camera.k3=source.k3; camera.k4=source.k4;
    camera.model=key.projection;
    if (orbit.projection==EditorProjection::orthographic) {
        camera.fx=camera.fy=float(height)/std::max(.0001F,key.ortho_height);
        camera.cx=width*.5F; camera.cy=height*.5F;
    }
    splat_render::DisplayOptions display; display.near_clip=near_clip;
    splat_render::FrameTarget frame;
    return renderer.draw(camera,frame,splat_render::Shading::gaussian,display) &&
        renderer.download_rgb(image.pixels,image.width,image.height,256);
}

inline SparseScene model_points(const photara::splat::GaussianModel& model,
    std::span<const splat_render::ModelTransform> transforms = {}, std::size_t max_points = 600000) {
    SparseScene scene;
    // Keep the same centre locations and DC colours as the rendered model.
    // CPU tensors share storage here; GPU tensors are downloaded once.
    const auto means = model.means.cpu(), sh = model.sh.cpu();
    const float* positions = means.ptr<float>();
    const float* colours = sh.ptr<float>();
    const std::size_t bases = sh.shape()[1];
    max_points = std::max<std::size_t>(1, max_points);
    const std::size_t stride = std::max<std::size_t>(1, model.size()/max_points + (model.size()%max_points != 0));
    scene.gaussian_sample_stride = stride;
    scene.points.reserve((model.size() + stride - 1) / stride);
    scene.colours.reserve(scene.points.capacity());
    for (std::size_t index = 0; index < model.size(); index += stride) {
        scene.points.push_back({positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]});
        const auto channel = [&](std::size_t c) {
            return static_cast<int>(std::lround(std::clamp(.5F + .28209479177387814F * colours[index * bases * 3 + c], 0.F, 1.F) * 255.F));
        };
        scene.colours.push_back(IM_COL32(channel(0), channel(1), channel(2), 255));
    }
    scene.compute_bounds();
    scene.set_model_transforms(transforms);
    return scene;
}

inline photara::io::RgbImage blend(photara::io::RgbImage a, const photara::io::RgbImage& b, float u) {
    if (a.width != b.width || a.height != b.height || a.pixels.size() != b.pixels.size())
        throw std::invalid_argument("Camera effect frame dimensions differ");
    u = std::clamp(u, 0.F, 1.F);
    for (std::size_t i = 0; i < a.pixels.size(); ++i)
        a.pixels[i] = static_cast<std::uint8_t>(std::lround(a.pixels[i] * (1.F - u) + b.pixels[i] * u));
    return a;
}

// The export path uses the same weights, point source and render passes as
// the viewport; only the final blend is performed on downloaded RGB frames.
inline photara::io::RgbImage render(
    splat_render::Renderer& renderer, splat_render::Camera camera,
    const OrbitCamera& orbit, const SparseScene& points, ViewOptions options,
    const photara::project::CameraEffectTrack& effect, float time,
    const splat_render::DisplayOptions& display) {
    const auto weights = photara::project::camera_effect_weights(effect, time);
    options.point_size = effect.point_size; options.ring_scale = effect.ring_scale;
    options.show_cloud = true; options.colour_by_depth = false;
    camera.ring_sigma = effect.ring_scale;
    const auto gaussian = [&](splat_render::Shading shading, bool capture_depth = false) {
        splat_render::FrameTarget target;
        photara::io::RgbImage image;
        auto settings = display; settings.capture_depth = capture_depth;
        if (!renderer.draw(camera, target, shading, settings) ||
            !renderer.download_rgb(image.pixels, image.width, image.height))
            throw std::runtime_error(renderer.failure().empty() ? "Cannot render camera effect frame" : renderer.failure());
        return image;
    };
    if (effect.enabled && effect.reconstruction())
        throw std::logic_error("Scan reveal requires the camera-path reconstruction renderer");
    if (weights.points > 0.F) {
        auto image = render_point_cloud_image(points, orbit, options, camera.width, camera.height);
        if (weights.rings > 0.F)
            return blend(std::move(image), gaussian(splat_render::Shading::rings), weights.rings);
        return image;
    }
    if (weights.rings > 0.F) {
        auto image = gaussian(splat_render::Shading::rings);
        if (weights.splat > 0.F)
            return blend(std::move(image), gaussian(splat_render::Shading::gaussian), weights.splat);
        return image;
    }
    return gaussian(splat_render::Shading::gaussian);
}

} // namespace editor::camera_effects
