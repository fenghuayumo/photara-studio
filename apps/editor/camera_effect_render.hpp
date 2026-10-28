#pragma once

#include "sparse_view.hpp"
#include "splat_render/renderer.hpp"
#include "project/camera_effect.hpp"
#include "io/image.hpp"
#include "core/camera_projection.hpp"
#include <limits>

namespace editor::camera_effects {

inline SparseScene model_points(const photara::splat::GaussianModel& model,
    std::span<const splat_render::ModelTransform> transforms = {}) {
    SparseScene scene;
    // Keep the same centre locations and DC colours as the rendered model.
    // CPU tensors share storage here; GPU tensors are downloaded once.
    const auto means = model.means.cpu(), sh = model.sh.cpu();
    const float* positions = means.ptr<float>();
    const float* colours = sh.ptr<float>();
    const std::size_t bases = sh.shape()[1];
    const std::size_t stride = std::max<std::size_t>(1, (model.size() + 599999) / 600000);
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

inline float scan_progress(const photara::project::CameraEffectTrack& effect, float time) {
    float u = std::clamp((time - effect.start_time) / (effect.end_time - effect.start_time), 0.F, 1.F);
    return effect.easing ? u * u * (3.F - 2.F * u) : u;
}

struct ScanPlane {
    std::array<float, 3> normal{};
    float position{}, band{};
};

inline ScanPlane scan_plane(const SparseScene& points,
    const photara::project::CameraEffectTrack& effect, float progress) {
    ScanPlane plane;
    if (effect.scan_axis < 3) plane.normal[effect.scan_axis] = 1.F;
    else std::copy_n(effect.scan_direction, 3, plane.normal.begin());
    const float length = std::sqrt(plane.normal[0]*plane.normal[0] + plane.normal[1]*plane.normal[1] + plane.normal[2]*plane.normal[2]);
    for (auto& n : plane.normal) n *= (effect.scan_reverse ? -1.F : 1.F) / length;
    float low = std::numeric_limits<float>::max(), high = std::numeric_limits<float>::lowest();
    for (std::size_t i = 0; i < points.points.size(); ++i) {
        const auto p = points.world_point(i);
        const float value = p.x*plane.normal[0] + p.y*plane.normal[1] + p.z*plane.normal[2];
        if (!std::isfinite(value)) continue;
        low = std::min(low, value); high = std::max(high, value);
    }
    if (high < low) throw std::runtime_error("No finite scene bounds for scan reveal");
    const float extent = std::max(.001F, high - low);
    plane.band = extent * effect.scan_softness;
    plane.position = low - plane.band * 2.F + progress * (extent + plane.band * 4.F);
    return plane;
}

// Median depth is camera Z, except for panoramas where it is ray distance.
inline std::array<float, 3> scan_world_point(
    const splat_render::Camera& camera, unsigned x, unsigned y, float depth) {
    float cx, cy, cz;
    if (camera.model == splat_render::k_camera_orthographic) {
        cx = (float(x) - camera.cx) / camera.fx; cy = (float(y) - camera.cy) / camera.fy; cz = depth;
    } else if (camera.model == splat_render::k_camera_equirectangular || camera.model == splat_render::k_camera_fisheye) {
        const auto ray = camera.model == splat_render::k_camera_equirectangular
            ? photara::unproject_equirectangular_camera(x, y, camera.width, camera.height)
            : photara::unproject_fisheye_camera(x, y, camera.fx, camera.fy, camera.cx, camera.cy,
                camera.k1, camera.k2, camera.k3, camera.k4);
        if (!ray.valid || (camera.model != splat_render::k_camera_equirectangular && ray.z <= 1e-6))
            return {NAN, NAN, NAN};
        const double scale = camera.model == splat_render::k_camera_equirectangular ? depth : depth / ray.z;
        cx = float(ray.x * scale); cy = float(ray.y * scale); cz = float(ray.z * scale);
    } else {
        cx = (float(x) - camera.cx) / camera.fx * depth;
        cy = (float(y) - camera.cy) / camera.fy * depth; cz = depth;
    }
    const auto& m = camera.world_to_camera;
    return {camera.position[0] + m[0]*cx + m[1]*cy + m[2]*cz,
            camera.position[1] + m[4]*cx + m[5]*cy + m[6]*cz,
            camera.position[2] + m[8]*cx + m[9]*cy + m[10]*cz};
}

inline photara::io::RgbImage scan_composite(
    photara::io::RgbImage full, const photara::io::RgbImage& dots,
    const splat_render::Renderer::DepthFrame& geometry, const splat_render::Camera& camera,
    const ScanPlane& plane, float progress, float glow) {
    const std::size_t pixels = std::size_t(full.width) * full.height;
    if (full.width != dots.width || full.height != dots.height ||
        full.width != camera.width || full.height != camera.height ||
        geometry.width != full.width || geometry.height != full.height ||
        geometry.depth.size() != pixels || geometry.alpha.size() != pixels ||
        full.pixels.size() != pixels * 3 || dots.pixels.size() != pixels * 3)
        throw std::runtime_error("Scan reveal needs matching RGB and depth frames");
    for (std::size_t i = 0; i < pixels; ++i) {
        float reveal = progress, highlight = 0.F;
        if (std::isfinite(geometry.depth[i]) && geometry.depth[i] > 0.F && geometry.alpha[i] > .01F) {
            const auto p = scan_world_point(camera, unsigned(i % full.width), unsigned(i / full.width), geometry.depth[i]);
            const float distance = p[0]*plane.normal[0] + p[1]*plane.normal[1] + p[2]*plane.normal[2] - plane.position;
            if (std::isfinite(distance)) {
                reveal = std::clamp(.5F - distance / (2.F * plane.band), 0.F, 1.F);
                reveal = reveal * reveal * (3.F - 2.F * reveal);
                const float edge = distance / (plane.band * .4F);
                highlight = std::exp(-.5F * edge * edge) * glow * std::clamp(geometry.alpha[i], 0.F, 1.F);
            }
        }
        const float cyan[]{38.F, 217.F, 255.F};
        for (std::size_t c = 0; c < 3; ++c) {
            const float value = dots.pixels[i*3+c] * (1.F-reveal) + full.pixels[i*3+c] * reveal;
            full.pixels[i*3+c] = static_cast<std::uint8_t>(std::lround(std::clamp(
                value * (1.F-highlight) + cyan[c] * highlight, 0.F, 255.F)));
        }
    }
    return full;
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
    if (effect.enabled && effect.kind == 1) {
        const float progress = scan_progress(effect, time);
        if (progress >= 1.F) return gaussian(splat_render::Shading::gaussian);
        auto dots = render_point_cloud_image(points, orbit, options, camera.width, camera.height);
        if (progress <= 0.F) return dots;
        auto full = gaussian(splat_render::Shading::gaussian, true);
        return scan_composite(std::move(full), dots, renderer.depth_frame(), camera,
            scan_plane(points, effect, progress), progress, effect.scan_glow);
    }
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
