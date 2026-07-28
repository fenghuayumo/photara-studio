#include "texture/mask.hpp"

#include "asdiff_render/asdiff_render.hpp"
#include "core/logging.hpp"
#include "io/image.hpp"
#include "texture/projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace aetherscan::texture {
namespace {

std::pair<float, float> depth_range(
    const mvs::MvsView& view, const mvs::Mesh& mesh) {
    const Eigen::Matrix3f rotation = view.pose.R.cast<float>();
    const Eigen::Vector3f center = view.pose.C.cast<float>();
    float near_depth = std::numeric_limits<float>::infinity();
    float far_depth = 0.F;
    for (const Eigen::Vector3f& world : mesh.vertices) {
        const float depth = (rotation * (world - center)).z();
        if (!std::isfinite(depth) || depth <= 1e-5F) continue;
        near_depth = std::min(near_depth, depth);
        far_depth = std::max(far_depth, depth);
    }
    if (!std::isfinite(near_depth) || !(far_depth > near_depth))
        throw std::runtime_error(
            "MVS mesh is outside a mask-rendering camera frustum");
    return {
        std::max(near_depth * 0.5F, 1e-4F),
        std::max(far_depth * 1.5F, near_depth + 1e-3F)};
}

mvs::MvsView source_render_view(
    const mvs::MvsView& source, const std::uint32_t supersample) {
    mvs::MvsView result = source;
    const std::uint32_t width =
        source.src_width != 0 ? source.src_width : source.width;
    const std::uint32_t height =
        source.src_height != 0 ? source.src_height : source.height;
    const float fx = source.src_width != 0 ? source.src_fx : source.fx;
    const float fy = source.src_height != 0 ? source.src_fy : source.fy;
    const float cx = source.src_width != 0 ? source.src_cx : source.cx;
    const float cy = source.src_height != 0 ? source.src_cy : source.cy;
    result.width = width * supersample;
    result.height = height * supersample;
    result.fx = fx * supersample;
    result.fy = fy * supersample;
    result.cx = (cx + 0.5F) * supersample - 0.5F;
    result.cy = (cy + 0.5F) * supersample - 0.5F;
    return result;
}

float median_mesh_edge_length(const mvs::Mesh& mesh) {
    std::vector<float> lengths;
    lengths.reserve(mesh.faces.size() * 3U);
    for (const Eigen::Vector3i& face : mesh.faces) {
        if (face.minCoeff() < 0 ||
            face.maxCoeff() >= static_cast<int>(mesh.vertices.size()))
            continue;
        for (int slot = 0; slot < 3; ++slot) {
            const float length =
                (mesh.vertices[static_cast<std::size_t>(face[slot])] -
                 mesh.vertices[static_cast<std::size_t>(
                     face[(slot + 1) % 3])])
                    .norm();
            if (std::isfinite(length) && length > 0.F)
                lengths.push_back(length);
        }
    }
    if (lengths.empty()) return 0.F;
    const auto middle =
        lengths.begin() + static_cast<std::ptrdiff_t>(lengths.size() / 2U);
    std::nth_element(lengths.begin(), middle, lengths.end());
    return *middle;
}

std::uint64_t remove_small_mask_components(
    io::RgbImage& mask, const bool foreground,
    const std::uint32_t maximum_pixels) {
    if (maximum_pixels == 0 || mask.width == 0 || mask.height == 0)
        return 0;
    const std::size_t pixels =
        static_cast<std::size_t>(mask.width) * mask.height;
    std::vector<std::uint8_t> visited(pixels, 0);
    std::vector<std::uint32_t> component;
    component.reserve(maximum_pixels + 1U);
    std::queue<std::uint32_t> pending;
    std::uint64_t changed = 0;
    constexpr int neighbors[8][2] = {
        {-1, -1}, {0, -1}, {1, -1}, {-1, 0},
        {1, 0}, {-1, 1}, {0, 1}, {1, 1}};
    constexpr int background_neighbors[4] = {1, 3, 4, 6};
    for (std::uint32_t seed = 0; seed < pixels; ++seed) {
        if (visited[seed]) continue;
        const bool selected = mask.pixels[3U * seed] != 0;
        if (selected != foreground) {
            visited[seed] = 1;
            continue;
        }
        component.clear();
        bool touches_boundary = false;
        visited[seed] = 1;
        pending.push(seed);
        while (!pending.empty()) {
            const std::uint32_t pixel = pending.front();
            pending.pop();
            component.push_back(pixel);
            const int x = static_cast<int>(pixel % mask.width);
            const int y = static_cast<int>(pixel / mask.width);
            touches_boundary = touches_boundary || x == 0 || y == 0 ||
                x + 1 == static_cast<int>(mask.width) ||
                y + 1 == static_cast<int>(mask.height);
            const int neighbor_count = foreground ? 8 : 4;
            for (int neighbor = 0; neighbor < neighbor_count; ++neighbor) {
                // Background uses the orthogonal entries of the same table.
                const int table_index = foreground
                    ? neighbor
                    : background_neighbors[neighbor];
                const int nx = x + neighbors[table_index][0];
                const int ny = y + neighbors[table_index][1];
                if (nx < 0 || ny < 0 ||
                    nx >= static_cast<int>(mask.width) ||
                    ny >= static_cast<int>(mask.height))
                    continue;
                const std::uint32_t next =
                    static_cast<std::uint32_t>(ny) * mask.width +
                    static_cast<std::uint32_t>(nx);
                if (visited[next]) continue;
                const bool next_selected =
                    mask.pixels[3U * next] != 0;
                if (next_selected != foreground) continue;
                visited[next] = 1;
                pending.push(next);
            }
        }
        if (component.size() > maximum_pixels ||
            (!foreground && touches_boundary))
            continue;
        const std::uint8_t value = foreground ? 0 : 255;
        for (const std::uint32_t pixel : component) {
            mask.pixels[3U * pixel] = value;
            mask.pixels[3U * pixel + 1U] = value;
            mask.pixels[3U * pixel + 2U] = value;
        }
        changed += component.size();
    }
    return changed;
}

}  // namespace

MeshMaskSummary render_mesh_foreground_masks(
    const mvs::MvsScene& scene,
    const std::filesystem::path& output_directory,
    const MeshMaskOptions& options) {
    if (scene.mesh.vertices.empty() || scene.mesh.faces.empty())
        throw std::invalid_argument(
            "Cannot render foreground masks from an empty MVS mesh");
    if (scene.views.empty())
        throw std::invalid_argument(
            "Cannot render foreground masks without cameras");
    if (output_directory.empty())
        throw std::invalid_argument(
            "Foreground-mask output directory is empty");
    const std::uint32_t supersample =
        std::clamp(options.supersample, 1U, 4U);
    const float median_edge = median_mesh_edge_length(scene.mesh);
    const float maximum_edge =
        options.maximum_edge_factor > 0.F && median_edge > 0.F
        ? median_edge * std::max(options.maximum_edge_factor, 1.F)
        : 0.F;

    std::vector<std::uint32_t> indices;
    indices.reserve(scene.mesh.faces.size() * 3U);
    std::vector<Eigen::Vector3f> rendered_face_normals;
    rendered_face_normals.reserve(scene.mesh.faces.size());
    MeshMaskSummary summary;
    for (const Eigen::Vector3i& face : scene.mesh.faces) {
        if (face.minCoeff() < 0 ||
            face.maxCoeff() >=
                static_cast<int>(scene.mesh.vertices.size()))
            continue;
        bool bridge = false;
        if (maximum_edge > 0.F) {
            for (int slot = 0; slot < 3; ++slot) {
                const float length =
                    (scene.mesh.vertices[
                         static_cast<std::size_t>(face[slot])] -
                     scene.mesh.vertices[static_cast<std::size_t>(
                         face[(slot + 1) % 3])])
                        .norm();
                if (!std::isfinite(length) || length > maximum_edge) {
                    bridge = true;
                    break;
                }
            }
        }
        if (bridge) {
            ++summary.rejected_bridge_faces;
            continue;
        }
        indices.push_back(static_cast<std::uint32_t>(face.x()));
        indices.push_back(static_cast<std::uint32_t>(face.y()));
        indices.push_back(static_cast<std::uint32_t>(face.z()));
        Eigen::Vector3f normal =
            (scene.mesh.vertices[static_cast<std::size_t>(face.y())] -
             scene.mesh.vertices[static_cast<std::size_t>(face.x())])
                .cross(
                    scene.mesh.vertices[static_cast<std::size_t>(face.z())] -
                    scene.mesh.vertices[static_cast<std::size_t>(face.x())]);
        const float norm = normal.norm();
        if (std::isfinite(norm) && norm > 1e-12F)
            normal /= norm;
        else
            normal = Eigen::Vector3f{0.F, 0.F, -1.F};
        rendered_face_normals.push_back(normal);
        ++summary.rendered_faces;
    }
    if (indices.empty())
        throw std::invalid_argument(
            "MVS mesh contains no valid triangles for mask rendering");

    std::filesystem::create_directories(output_directory);
    if (!options.preview_directory.empty())
        std::filesystem::create_directories(options.preview_directory);
    core::StageScope stage("mask.asdiff_render_mvs_mesh");
    asdiff_render::Context context(
        {options.vulkan_device_index, false});
    asdiff_render::Rasterizer rasterizer(context);
    std::vector<float> clip_positions(scene.mesh.vertices.size() * 4U);

    for (const mvs::MvsView& view : scene.views) {
        const std::uint32_t target_width =
            view.src_width != 0 ? view.src_width : view.width;
        const std::uint32_t target_height =
            view.src_height != 0 ? view.src_height : view.height;
        if (target_width == 0 || target_height == 0)
            throw std::runtime_error(
                "Mask-rendering camera has empty source dimensions");
        const mvs::MvsView render_view =
            source_render_view(view, supersample);
        const auto [near_z, far_z] = depth_range(render_view, scene.mesh);
        const auto matrix =
            world_to_clip_row_major(render_view, near_z, far_z);
        for (std::size_t index = 0;
             index < scene.mesh.vertices.size(); ++index) {
            const Eigen::Vector3f& point = scene.mesh.vertices[index];
            for (int row = 0; row < 4; ++row)
                clip_positions[4U * index + static_cast<std::size_t>(row)] =
                    matrix[4U * static_cast<std::size_t>(row)] * point.x() +
                    matrix[4U * static_cast<std::size_t>(row) + 1U] *
                        point.y() +
                    matrix[4U * static_cast<std::size_t>(row) + 2U] *
                        point.z() +
                    matrix[4U * static_cast<std::size_t>(row) + 3U];
        }

        asdiff_render::RasterizeOptions raster_options;
        raster_options.width = render_view.width;
        raster_options.height = render_view.height;
        raster_options.cull_mode = asdiff_render::CullMode::none;
        raster_options.output_barycentric_derivatives = false;
        const asdiff_render::RasterizeOutput rendered =
            rasterizer.forward(clip_positions, indices, raster_options);
        const std::size_t expected =
            static_cast<std::size_t>(render_view.width) *
            render_view.height * 4U;
        if (rendered.raster.size() != expected)
            throw std::runtime_error(
                "asdiff_render returned an unexpected mask raster size");

        const std::size_t pixels =
            static_cast<std::size_t>(target_width) * target_height;
        io::RgbImage mask{
            target_width, target_height,
            std::vector<std::uint8_t>(pixels * 3U)};
        io::RgbImage preview;
        if (!options.preview_directory.empty()) {
            preview.width = target_width;
            preview.height = target_height;
            preview.pixels.assign(pixels * 3U, 0);
        }
        const Eigen::Matrix3f world_to_camera =
            view.pose.R.cast<float>();
        const unsigned sample_count = supersample * supersample;
        for (std::uint32_t y = 0; y < target_height; ++y) {
            for (std::uint32_t x = 0; x < target_width; ++x) {
                unsigned covered = 0;
                Eigen::Vector3f preview_sum = Eigen::Vector3f::Zero();
                for (std::uint32_t sy = 0; sy < supersample; ++sy) {
                    for (std::uint32_t sx = 0; sx < supersample; ++sx) {
                        const std::size_t high_pixel =
                            static_cast<std::size_t>(
                                y * supersample + sy) *
                                render_view.width +
                            x * supersample + sx;
                        const float triangle_value =
                            rendered.raster[4U * high_pixel + 3U];
                        if (!(triangle_value > 0.F)) continue;
                        ++covered;
                        if (preview.pixels.empty()) continue;
                        const auto triangle = static_cast<std::size_t>(
                            std::llround(triangle_value) - 1LL);
                        if (triangle >= rendered_face_normals.size()) continue;
                        Eigen::Vector3f normal =
                            world_to_camera * rendered_face_normals[triangle];
                        if (normal.z() > 0.F) normal = -normal;
                        const float light =
                            0.25F + 0.75F * std::abs(normal.z());
                        const Eigen::Vector3f tint{
                            0.35F + 0.35F * (normal.x() + 1.F),
                            0.30F + 0.35F * (normal.y() + 1.F),
                            0.55F};
                        preview_sum +=
                            (tint * light).cwiseMax(0.F).cwiseMin(1.F);
                    }
                }
                const auto value = static_cast<std::uint8_t>(
                    (covered * 255U + sample_count / 2U) / sample_count);
                const std::size_t pixel =
                    static_cast<std::size_t>(y) * target_width + x;
                mask.pixels[3U * pixel] = value;
                mask.pixels[3U * pixel + 1U] = value;
                mask.pixels[3U * pixel + 2U] = value;
                if (covered != 0 && !preview.pixels.empty()) {
                    const Eigen::Vector3f color =
                        preview_sum / static_cast<float>(covered);
                    for (int channel = 0; channel < 3; ++channel)
                        preview.pixels[3U * pixel +
                                       static_cast<std::size_t>(channel)] =
                            static_cast<std::uint8_t>(std::lround(
                                std::clamp(color[channel], 0.F, 1.F) *
                                255.F));
                }
            }
        }
        summary.removed_island_pixels += remove_small_mask_components(
            mask, true, options.minimum_island_pixels);
        summary.filled_hole_pixels += remove_small_mask_components(
            mask, false, options.maximum_hole_pixels);
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::uint8_t value = mask.pixels[3U * pixel];
            summary.foreground_pixels += value >= 128;
            summary.soft_edge_pixels += value != 0 && value != 255;
        }
        io::save_rgb_png(
            mask,
            output_directory /
                (view.path.stem().string() + ".png"));
        if (!preview.pixels.empty()) {
            io::save_rgb_png(
                preview,
                options.preview_directory /
                    (view.path.stem().string() + ".png"));
            ++summary.preview_count;
        }
        ++summary.image_count;
        summary.pixel_count += pixels;
    }

    core::Logger::instance().info(
        "MVS mesh masks: renderer=asdiff_render views=",
        summary.image_count,
        " resolution=source supersample=", supersample,
        " foreground_fraction=",
        summary.pixel_count == 0
            ? 0.0
            : static_cast<double>(summary.foreground_pixels) /
                  static_cast<double>(summary.pixel_count),
        " soft_edge_pixels=", summary.soft_edge_pixels,
        " median_edge=", median_edge,
        " maximum_edge=", maximum_edge,
        " rendered_faces=", summary.rendered_faces,
        " rejected_bridge_faces=", summary.rejected_bridge_faces,
        " removed_island_pixels=", summary.removed_island_pixels,
        " filled_hole_pixels=", summary.filled_hole_pixels,
        " previews=", summary.preview_count,
        options.preview_directory.empty()
            ? std::string{}
            : " preview_directory=" +
                  options.preview_directory.string(),
        " directory=", output_directory);
    stage.finish();
    return summary;
}

MeshMaskSummary export_view_foreground_masks(
    const mvs::MvsScene& scene,
    const std::filesystem::path& output_directory) {
    std::filesystem::create_directories(output_directory);
    MeshMaskSummary summary;
    for (const mvs::MvsView& view : scene.views) {
        const std::size_t working_pixels =
            static_cast<std::size_t>(view.width) * view.height;
        if (view.foreground_mask.size() != working_pixels) continue;
        const std::uint32_t width =
            view.src_width != 0 ? view.src_width : view.width;
        const std::uint32_t height =
            view.src_height != 0 ? view.src_height : view.height;
        const float fx = view.src_width != 0 ? view.src_fx : view.fx;
        const float fy = view.src_height != 0 ? view.src_fy : view.fy;
        const float cx = view.src_width != 0 ? view.src_cx : view.cx;
        const float cy = view.src_height != 0 ? view.src_cy : view.cy;
        const std::size_t pixels =
            static_cast<std::size_t>(width) * height;
        io::RgbImage mask{
            width, height, std::vector<std::uint8_t>(pixels * 3U)};
        const auto sample = [&](const int x, const int y) {
            if (x < 0 || y < 0 || x >= static_cast<int>(view.width) ||
                y >= static_cast<int>(view.height))
                return 0.F;
            return static_cast<float>(
                       view.foreground_mask[
                           static_cast<std::size_t>(y) * view.width + x]) /
                255.F;
        };
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                const float xn = (static_cast<float>(x) - cx) / fx;
                const float yn = (static_cast<float>(y) - cy) / fy;
                const float u = view.fx * xn + view.cx;
                const float v = view.fy * yn + view.cy;
                const int x0 = static_cast<int>(std::floor(u));
                const int y0 = static_cast<int>(std::floor(v));
                const float tx = u - static_cast<float>(x0);
                const float ty = v - static_cast<float>(y0);
                const float coverage = std::clamp(
                    (sample(x0, y0) * (1.F - tx) +
                     sample(x0 + 1, y0) * tx) *
                            (1.F - ty) +
                        (sample(x0, y0 + 1) * (1.F - tx) +
                         sample(x0 + 1, y0 + 1) * tx) *
                            ty,
                    0.F, 1.F);
                const auto value = static_cast<std::uint8_t>(
                    std::lround(coverage * 255.F));
                const std::size_t pixel =
                    static_cast<std::size_t>(y) * width + x;
                mask.pixels[3U * pixel] = value;
                mask.pixels[3U * pixel + 1U] = value;
                mask.pixels[3U * pixel + 2U] = value;
                summary.foreground_pixels += value >= 128;
                summary.soft_edge_pixels += value != 0 && value != 255;
            }
        }
        io::save_rgb_png(
            mask, output_directory / (view.path.stem().string() + ".png"));
        ++summary.image_count;
        summary.pixel_count += pixels;
    }
    core::Logger::instance().info(
        "view foreground masks: source=depth_roi views=",
        summary.image_count,
        " resolution=source foreground_fraction=",
        summary.pixel_count == 0
            ? 0.0
            : static_cast<double>(summary.foreground_pixels) /
                  static_cast<double>(summary.pixel_count),
        " soft_edge_pixels=", summary.soft_edge_pixels,
        " directory=", output_directory);
    return summary;
}

}  // namespace aetherscan::texture
