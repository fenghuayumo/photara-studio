#include "splat/trainer.hpp"

#include "core/logging.hpp"
#include "io/image.hpp"
#include "mvs/densify.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace aetherscan::splat {
namespace {

mvs::MvsView make_geometry_view(const mvs::MvsView& source) {
    mvs::MvsView result;
    result.id = source.id;
    result.sfm_image_id = source.sfm_image_id;
    result.path = source.path;
    result.pose = source.pose;
    result.fx = source.fx;
    result.fy = source.fy;
    result.cx = source.cx;
    result.cy = source.cy;
    result.width = source.width;
    result.height = source.height;
    result.src_fx = source.src_fx;
    result.src_fy = source.src_fy;
    result.src_cx = source.src_cx;
    result.src_cy = source.src_cy;
    result.k1 = source.k1;
    result.k2 = source.k2;
    result.p1 = source.p1;
    result.p2 = source.p2;
    result.src_width = source.src_width;
    result.src_height = source.src_height;
    result.neighbors = source.neighbors;
    return result;
}

void save_geometry_diagnostics(
    const std::vector<float>& depth, const std::vector<float>& normal,
    const std::vector<float>& alpha, const std::vector<float>& mask,
    const std::uint32_t width, const std::uint32_t height,
    const float alpha_threshold, const std::size_t view_index,
    const std::filesystem::path& directory) {
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    std::vector<float> valid_depth;
    valid_depth.reserve(pixels / 2);
    for (std::size_t pixel = 0; pixel < pixels; ++pixel)
        if (mask[pixel] > 0.5F && alpha[pixel] >= alpha_threshold &&
            depth[pixel] > 0.F && std::isfinite(depth[pixel]))
            valid_depth.push_back(depth[pixel]);
    if (valid_depth.empty()) return;
    const auto percentile = [&](const float fraction) {
        const std::size_t index = std::min(
            valid_depth.size() - 1,
            static_cast<std::size_t>(fraction * valid_depth.size()));
        std::nth_element(
            valid_depth.begin(),
            valid_depth.begin() + static_cast<std::ptrdiff_t>(index),
            valid_depth.end());
        return valid_depth[index];
    };
    const float near_depth = percentile(0.02F);
    const float far_depth = percentile(0.98F);
    const float inverse_range =
        1.F / std::max(far_depth - near_depth, 1e-6F);

    io::RgbImage depth_image{width, height, std::vector<std::uint8_t>(3 * pixels)};
    io::RgbImage normal_image{width, height, std::vector<std::uint8_t>(3 * pixels)};
    io::RgbImage alpha_image{width, height, std::vector<std::uint8_t>(3 * pixels)};
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        const bool valid = mask[pixel] > 0.5F &&
            alpha[pixel] >= alpha_threshold && depth[pixel] > 0.F &&
            std::isfinite(depth[pixel]);
        if (valid) {
            const float t = std::clamp(
                (depth[pixel] - near_depth) * inverse_range, 0.F, 1.F);
            // Compact blue-cyan-yellow-red map; near is warm, far is cool.
            const float q = 1.F - t;
            const std::array<float, 3> color{
                std::clamp(1.5F - std::abs(4.F * q - 3.F), 0.F, 1.F),
                std::clamp(1.5F - std::abs(4.F * q - 2.F), 0.F, 1.F),
                std::clamp(1.5F - std::abs(4.F * q - 1.F), 0.F, 1.F)};
            mvs::Vec3f n{
                normal[pixel], normal[pixels + pixel],
                normal[2 * pixels + pixel]};
            if (n.allFinite() && n.squaredNorm() > 1e-12F) n.normalize();
            for (int channel = 0; channel < 3; ++channel) {
                depth_image.pixels[3 * pixel + channel] =
                    static_cast<std::uint8_t>(std::lround(color[channel] * 255.F));
                normal_image.pixels[3 * pixel + channel] =
                    static_cast<std::uint8_t>(std::lround(std::clamp(
                        n(channel) * 0.5F + 0.5F, 0.F, 1.F) * 255.F));
            }
        }
        const auto alpha_byte = static_cast<std::uint8_t>(std::lround(
            std::clamp(alpha[pixel], 0.F, 1.F) * 255.F));
        for (int channel = 0; channel < 3; ++channel)
            alpha_image.pixels[3 * pixel + channel] = alpha_byte;
    }
    std::filesystem::create_directories(directory);
    const std::string suffix = "_view_" + std::to_string(view_index) + ".png";
    io::save_rgb_png(depth_image, directory / ("gggs_depth" + suffix));
    io::save_rgb_png(normal_image, directory / ("gggs_normal" + suffix));
    io::save_rgb_png(alpha_image, directory / ("gggs_alpha" + suffix));
    core::Logger::instance().info(
        "gggs geometry diagnostics: view=", view_index,
        " depth_p02=", near_depth, " depth_p98=", far_depth,
        " directory=", directory);
}

}  // namespace

GggsMeshResult extract_gggs_mesh(
    const GaussianModel& model, const mvs::MvsScene& scene,
    const TrainingOptions& training_options,
    const GggsMeshOptions& mesh_options) {
    if (model.size() == 0)
        throw std::invalid_argument("GGGS mesh extraction requires Gaussians");
    if (scene.views.size() < 2)
        throw std::invalid_argument(
            "GGGS mesh extraction requires at least two registered views");
    if (!(mesh_options.alpha_threshold > 0.F &&
          mesh_options.alpha_threshold < 1.F))
        throw std::invalid_argument(
            "GGGS mesh alpha threshold must be in (0, 1)");

    core::StageScope render_stage("gggs.mesh_render_geometry");
    mvs::MvsScene geometry_scene;
    geometry_scene.views.reserve(scene.views.size());
    for (const auto& view : scene.views)
        geometry_scene.views.push_back(make_geometry_view(view));
    geometry_scene.sparse_points = scene.sparse_points;
    geometry_scene.roi = scene.roi;
    geometry_scene.roi_automatic = scene.roi_automatic;
    geometry_scene.has_ground_plane = scene.has_ground_plane;
    geometry_scene.ground_normal = scene.ground_normal;
    geometry_scene.ground_offset = scene.ground_offset;
    geometry_scene.thread_count = scene.thread_count;

    mvs::DensifyOptions fusion_options = mesh_options.fusion;
    fusion_options.build_mesh = true;
    fusion_options.ncc_keep_threshold = std::max(
        1.F - mesh_options.alpha_threshold, 1e-3F);
    // GGGS depths are already a coherent learned surface. Keep the geometric
    // checks, but do not require the stricter PatchMatch photometric preset.
    fusion_options.depth_diff_threshold = std::max(
        fusion_options.depth_diff_threshold, 0.015F);
    fusion_options.normal_diff_threshold_deg = std::max(
        fusion_options.normal_diff_threshold_deg, 30.F);

    bool need_neighbors = false;
    for (const auto& view : geometry_scene.views)
        need_neighbors = need_neighbors || view.neighbors.empty();
    if (need_neighbors) mvs::select_neighbors(geometry_scene, fusion_options);

    TrainingOptions render_training_options = training_options;
    // Mesh fusion uses the MVS working calibration. This keeps CPU fusion
    // bounded while preserving the exact undistorted camera convention.
    render_training_options.use_source_resolution = false;
    RasterizeOptions raster_options;
    raster_options.active_sh_degree = model.sh_degree;
    raster_options.kernel_size = training_options.kernel_size;
    raster_options.scale_modifier = training_options.scale_modifier;
    raster_options.require_depth = true;
    Rasterizer rasterizer;

    std::size_t valid_depth_pixels = 0;
    std::size_t compared_depth_normal_pixels = 0;
    std::size_t rejected_depth_normal_pixels = 0;
    for (std::size_t view_index = 0;
         view_index < geometry_scene.views.size(); ++view_index) {
        auto& geometry_view = geometry_scene.views[view_index];
        const TrainingView target = make_training_view(
            scene.views[view_index], render_training_options);
        const RenderResult rendered = rasterizer.forward(
            model, target.camera, raster_options);
        const std::vector<float> depth = rendered.median_depth.to_vector();
        const std::vector<float> normal = rendered.normal.to_vector();
        const std::vector<float> alpha = rendered.alpha.to_vector();
        const std::vector<float> mask = target.mask.to_vector();
        const std::size_t pixels = static_cast<std::size_t>(
            geometry_view.width) * geometry_view.height;
        if (depth.size() != pixels || alpha.size() != pixels ||
            mask.size() != pixels || normal.size() != 3 * pixels)
            throw std::runtime_error(
                "GGGS mesh render returned an unexpected tensor shape");

        if (!mesh_options.diagnostics_dir.empty() &&
            (view_index == 0 || view_index == geometry_scene.views.size() / 2 ||
             view_index + 1 == geometry_scene.views.size()))
            save_geometry_diagnostics(
                depth, normal, alpha, mask, geometry_view.width,
                geometry_view.height, mesh_options.alpha_threshold,
                view_index, mesh_options.diagnostics_dir);

        auto& depth_map = geometry_view.depth_map;
        depth_map.view_id = geometry_view.id;
        depth_map.resize(geometry_view.width, geometry_view.height);
        geometry_view.foreground_mask.assign(pixels, 0);
        float minimum_depth = std::numeric_limits<float>::infinity();
        float maximum_depth = 0.F;
        for (std::uint32_t y = 0; y < geometry_view.height; ++y) {
            for (std::uint32_t x = 0; x < geometry_view.width; ++x) {
                const std::size_t pixel =
                    static_cast<std::size_t>(y) * geometry_view.width + x;
                const float d = depth[pixel];
                if (mask[pixel] <= 0.5F ||
                    alpha[pixel] < mesh_options.alpha_threshold ||
                    !std::isfinite(d) || d <= 0.F ||
                    (mesh_options.max_depth > 0.F &&
                     d > mesh_options.max_depth))
                    continue;
                mvs::Vec3f n{
                    normal[pixel], normal[pixels + pixel],
                    normal[2 * pixels + pixel]};
                if (!n.allFinite() || n.squaredNorm() < 0.25F) continue;
                n.normalize();
                if (mesh_options.min_depth_normal_cosine >= -1.F && x > 0 &&
                    x + 1 < geometry_view.width && y > 0 &&
                    y + 1 < geometry_view.height) {
                    const auto point_at = [&](const std::uint32_t px,
                                              const std::uint32_t py) {
                        const float neighbor_depth =
                            depth[static_cast<std::size_t>(py) *
                                      geometry_view.width + px];
                        return mvs::Vec3f{
                            (static_cast<float>(px) - geometry_view.cx) /
                                geometry_view.fx * neighbor_depth,
                            (static_cast<float>(py) - geometry_view.cy) /
                                geometry_view.fy * neighbor_depth,
                            neighbor_depth};
                    };
                    const float top = depth[pixel - geometry_view.width];
                    const float bottom = depth[pixel + geometry_view.width];
                    const float left = depth[pixel - 1];
                    const float right = depth[pixel + 1];
                    if (std::isfinite(top) && std::isfinite(bottom) &&
                        std::isfinite(left) && std::isfinite(right) &&
                        top > 0.F && bottom > 0.F && left > 0.F &&
                        right > 0.F) {
                        mvs::Vec3f depth_normal =
                            (point_at(x, y + 1) - point_at(x, y - 1))
                                .cross(point_at(x + 1, y) -
                                       point_at(x - 1, y));
                        if (depth_normal.allFinite() &&
                            depth_normal.squaredNorm() > 1e-12F) {
                            depth_normal.normalize();
                            ++compared_depth_normal_pixels;
                            if (n.dot(depth_normal) <
                                mesh_options.min_depth_normal_cosine) {
                                ++rejected_depth_normal_pixels;
                                continue;
                            }
                        }
                    }
                }
                const mvs::Vec3f camera_ray{
                    (static_cast<float>(x) - geometry_view.cx) /
                        geometry_view.fx,
                    (static_cast<float>(y) - geometry_view.cy) /
                        geometry_view.fy,
                    1.F};
                // Fusion expects camera-facing normals.
                if (n.dot(camera_ray) > 0.F) n = -n;
                depth_map.depth[pixel] = d;
                depth_map.normal[pixel] = n;
                depth_map.confidence[pixel] = std::clamp(
                    1.F - alpha[pixel], 0.F,
                    fusion_options.ncc_keep_threshold);
                geometry_view.foreground_mask[pixel] = 1;
                minimum_depth = std::min(minimum_depth, d);
                maximum_depth = std::max(maximum_depth, d);
                ++valid_depth_pixels;
            }
        }
        depth_map.depth_min = std::isfinite(minimum_depth)
            ? minimum_depth
            : 0.F;
        depth_map.depth_max = maximum_depth;
    }
    render_stage.finish();

    core::Logger::instance().info(
        "gggs mesh depth: views=", geometry_scene.views.size(),
        " valid_pixels=", valid_depth_pixels,
        " alpha_threshold=", mesh_options.alpha_threshold,
        " depth_normal_compared=", compared_depth_normal_pixels,
        " depth_normal_rejected=", rejected_depth_normal_pixels,
        " min_depth_normal_cosine=", mesh_options.min_depth_normal_cosine);
    mvs::fuse_depth_maps(geometry_scene, fusion_options);
    if (geometry_scene.dense_cloud.points.empty())
        throw std::runtime_error(
            "GGGS mesh depth fusion produced no surface points");
    mvs::reconstruct_mesh(geometry_scene, fusion_options);
    if (geometry_scene.mesh.vertices.empty() ||
        geometry_scene.mesh.faces.empty())
        throw std::runtime_error(
            "GGGS mesh extraction produced no triangle surface");

    GggsMeshResult result;
    result.surface_cloud = std::move(geometry_scene.dense_cloud);
    result.mesh = std::move(geometry_scene.mesh);
    result.valid_depth_pixels = valid_depth_pixels;
    return result;
}

}  // namespace aetherscan::splat
