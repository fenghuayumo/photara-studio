#include "texture/bake.hpp"

#include "core/logging.hpp"
#include "io/image.hpp"
#include "parallel/thread_pool.hpp"
#include "texture/delight.hpp"
#include "texture/export.hpp"
#include "texture/projection.hpp"

#include "aether_drender/aether_drender.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace aetherscan::texture {
namespace {

[[nodiscard]] float srgb_to_linear(const float c) {
    return c <= 0.04045F ? c / 12.92F
                         : std::pow((c + 0.055F) / 1.055F, 2.4F);
}

[[nodiscard]] sfm::Vec2 project_distorted(
    const mvs::MvsView& view, const double xn, const double yn) {
    const auto pixel = project_camera_plane(view.source_model, xn, yn,
        view.k1, view.k2, view.p1, view.p2);
    return {view.src_fx * pixel.x + view.src_cx, view.src_fy * pixel.y + view.src_cy};
}

[[nodiscard]] std::filesystem::path find_texture_mask(
    const std::filesystem::path& directory,
    const std::filesystem::path& image_path) {
    if (directory.empty() || !std::filesystem::is_directory(directory))
        return {};
    const std::filesystem::path exact = directory / image_path.filename();
    if (std::filesystem::is_regular_file(exact)) return exact;
    static constexpr std::array<const char*, 6> extensions{
        ".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"};
    for (const char* extension : extensions) {
        const std::filesystem::path candidate =
            directory / (image_path.stem().string() + extension);
        if (std::filesystem::is_regular_file(candidate)) return candidate;
    }
    return {};
}

[[nodiscard]] std::pair<float, float> depth_range_for_view(
    const mvs::MvsView& view, const mvs::Mesh& mesh) {
    float near_z = std::numeric_limits<float>::max();
    float far_z = 0.F;
    for (const auto& v : mesh.vertices) {
        const Eigen::Vector3f cam =
            view.pose.transform_world_to_camera(v.cast<double>()).cast<float>();
        if (cam.z() > 1e-6F) {
            near_z = std::min(near_z, cam.z());
            far_z = std::max(far_z, cam.z());
        }
    }
    if (!(near_z < far_z) || !std::isfinite(near_z) || !std::isfinite(far_z))
        throw std::runtime_error(
            "Mesh is entirely behind a camera; cannot bake textures");
    near_z = std::max(1e-4F, near_z * 0.5F);
    far_z = far_z * 1.5F;
    return {near_z, far_z};
}

std::vector<TextureViewImage> load_texture_views(
    const mvs::MvsScene& scene, const TextureOptions& options) {
    core::StageScope stage("texture.load_views");
    std::vector<TextureViewImage> views(scene.views.size());
    const unsigned threads =
        parallel::resolve_thread_count(scene.thread_count);
    core::ProgressReporter progress("texture.load_views", scene.views.size());

    parallel::parallel_for(
        scene.views.size(), threads, [&](const std::size_t i) {
            const mvs::MvsView& view = scene.views[i];
            const io::RgbImage source = io::load_rgb(view.path);
            const bool has_effective_mask =
                has_effective_foreground_mask(view);
            io::GrayImage source_mask;
            if (!has_effective_mask) {
                const std::filesystem::path mask_path =
                    find_texture_mask(options.mask_dir, view.path);
                if (!mask_path.empty()) source_mask = io::load_gray(mask_path);
            }
            const bool has_distortion =
                view.source_model == CameraModel::opencv_fisheye || view.k1 != 0.F || view.k2 != 0.F || view.p1 != 0.F ||
                view.p2 != 0.F;

            auto sample_channel =
                [&](const float x, const float y, const int c) -> float {
                if (x < 0.F || y < 0.F ||
                    x > static_cast<float>(source.width - 1) ||
                    y > static_cast<float>(source.height - 1))
                    return 0.F;
                const int x0 = static_cast<int>(x);
                const int y0 = static_cast<int>(y);
                const int x1 =
                    std::min(x0 + 1, static_cast<int>(source.width) - 1);
                const int y1 =
                    std::min(y0 + 1, static_cast<int>(source.height) - 1);
                const float tx = x - static_cast<float>(x0);
                const float ty = y - static_cast<float>(y0);
                const auto at = [&](const int px, const int py) {
                    return static_cast<float>(
                        source.pixels
                            [(static_cast<std::size_t>(py) * source.width +
                              static_cast<std::size_t>(px)) *
                                 3U +
                             static_cast<std::size_t>(c)]);
                };
                return (at(x0, y0) * (1.F - tx) + at(x1, y0) * tx) *
                           (1.F - ty) +
                       (at(x0, y1) * (1.F - tx) + at(x1, y1) * tx) * ty;
            };

            TextureViewImage& out = views[i];
            out.width = view.width;
            out.height = view.height;
            out.rgb.resize(
                static_cast<std::size_t>(view.width) * view.height * 3U);
            if (has_effective_mask || !source_mask.pixels.empty())
                out.mask.resize(
                    static_cast<std::size_t>(view.width) * view.height);

            for (std::uint32_t y = 0; y < view.height; ++y) {
                for (std::uint32_t x = 0; x < view.width; ++x) {
                    float sx = 0.F;
                    float sy = 0.F;
                    if (!has_distortion) {
                        const float scale_x =
                            static_cast<float>(view.width) /
                            static_cast<float>(source.width);
                        const float scale_y =
                            static_cast<float>(view.height) /
                            static_cast<float>(source.height);
                        sx = (static_cast<float>(x) + 0.5F) / scale_x - 0.5F;
                        sy = (static_cast<float>(y) + 0.5F) / scale_y - 0.5F;
                    } else {
                        const double xn =
                            (static_cast<double>(x) - view.cx) /
                            static_cast<double>(view.fx);
                        const double yn =
                            (static_cast<double>(y) - view.cy) /
                            static_cast<double>(view.fy);
                        const sfm::Vec2 distorted =
                            project_distorted(view, xn, yn);
                        sx = static_cast<float>(distorted.x());
                        sy = static_cast<float>(distorted.y());
                    }
                    const std::size_t dst =
                        (static_cast<std::size_t>(y) * view.width + x) * 3U;
                    for (int c = 0; c < 3; ++c) {
                        out.rgb[dst + static_cast<std::size_t>(c)] =
                            srgb_to_linear(
                                sample_channel(sx, sy, c) / 255.F);
                    }
                    if (!out.mask.empty()) {
                        const std::size_t output_index =
                            static_cast<std::size_t>(y) * view.width + x;
                        if (has_effective_mask) {
                            out.mask[output_index] =
                                effective_foreground_coverage(
                                    view, output_index);
                        } else {
                            const float scale_x =
                                static_cast<float>(source_mask.width) /
                                static_cast<float>(source.width);
                            const float scale_y =
                                static_cast<float>(source_mask.height) /
                                static_cast<float>(source.height);
                            const float mx = (sx + 0.5F) * scale_x - 0.5F;
                            const float my = (sy + 0.5F) * scale_y - 0.5F;
                            const int mask_x = std::clamp(
                                static_cast<int>(std::lround(mx)), 0,
                                static_cast<int>(source_mask.width) - 1);
                            const int mask_y = std::clamp(
                                static_cast<int>(std::lround(my)), 0,
                                static_cast<int>(source_mask.height) - 1);
                            out.mask[output_index] =
                                source_mask.pixels[
                                    static_cast<std::size_t>(mask_y) *
                                        source_mask.width +
                                    static_cast<std::size_t>(mask_x)] >= 128
                                ? 1.F
                                : 0.F;
                        }
                    }
                }
            }
            progress.advance();
        });
    stage.finish();
    return views;
}

void compute_vertex_normals(
    const std::vector<float>& positions,
    const std::vector<std::uint32_t>& indices, std::vector<float>& normals) {
    const std::size_t vertex_count = positions.size() / 3U;
    normals.assign(positions.size(), 0.F);
    for (std::size_t f = 0; f + 2 < indices.size(); f += 3) {
        const std::uint32_t i0 = indices[f];
        const std::uint32_t i1 = indices[f + 1];
        const std::uint32_t i2 = indices[f + 2];
        const Eigen::Vector3f p0(
            positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2]);
        const Eigen::Vector3f p1(
            positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2]);
        const Eigen::Vector3f p2(
            positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2]);
        const Eigen::Vector3f n = (p1 - p0).cross(p2 - p0);
        for (const std::uint32_t idx : {i0, i1, i2}) {
            normals[idx * 3] += n.x();
            normals[idx * 3 + 1] += n.y();
            normals[idx * 3 + 2] += n.z();
        }
    }
    for (std::size_t i = 0; i < vertex_count; ++i) {
        Eigen::Vector3f n(
            normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
        const float len = n.norm();
        if (len > 1e-20F) n /= len;
        normals[i * 3] = n.x();
        normals[i * 3 + 1] = n.y();
        normals[i * 3 + 2] = n.z();
    }
}

struct SeamEdgeUse {
    std::uint64_t key{};
    std::uint32_t chart{};
    mvs::Vec2f low_uv{mvs::Vec2f::Zero()};
    mvs::Vec2f high_uv{mvs::Vec2f::Zero()};
};

// UVAtlas duplicates vertices along chart boundaries. Match the two copies of
// each original manifold edge and sample corresponding UVs for the native
// aether_drender seam-polish pass.
std::vector<float> build_dense_seam_pairs(
    const aether_drender::UvAtlasOutput& mesh,
    const std::uint32_t samples_per_edge) {
    if (samples_per_edge == 0 || mesh.indices.size() % 3U != 0 ||
        mesh.uv.size() % 2U != 0 ||
        mesh.vertex_remap.size() != mesh.uv.size() / 2U ||
        mesh.face_chart_ids.size() != mesh.indices.size() / 3U)
        return {};

    std::vector<SeamEdgeUse> uses;
    uses.reserve(mesh.indices.size());
    for (std::size_t face = 0; face < mesh.indices.size() / 3U; ++face) {
        for (int edge = 0; edge < 3; ++edge) {
            const std::uint32_t a =
                mesh.indices[face * 3U + static_cast<std::size_t>(edge)];
            const std::uint32_t b = mesh.indices[
                face * 3U + static_cast<std::size_t>((edge + 1) % 3)];
            if (a >= mesh.vertex_remap.size() || b >= mesh.vertex_remap.size())
                continue;
            const std::uint32_t original_a = mesh.vertex_remap[a];
            const std::uint32_t original_b = mesh.vertex_remap[b];
            if (original_a == original_b) continue;
            const bool swap = original_a > original_b;
            const std::uint32_t low = std::min(original_a, original_b);
            const std::uint32_t high = std::max(original_a, original_b);
            const auto uv = [&](const std::uint32_t vertex) {
                return mvs::Vec2f{
                    mesh.uv[static_cast<std::size_t>(vertex) * 2U],
                    mesh.uv[static_cast<std::size_t>(vertex) * 2U + 1U]};
            };
            uses.push_back({
                (static_cast<std::uint64_t>(low) << 32U) | high,
                mesh.face_chart_ids[face], swap ? uv(b) : uv(a),
                swap ? uv(a) : uv(b)});
        }
    }
    std::sort(uses.begin(), uses.end(), [](const auto& a, const auto& b) {
        return std::tie(a.key, a.chart) < std::tie(b.key, b.chart);
    });

    std::vector<float> pairs;
    for (std::size_t begin = 0; begin < uses.size();) {
        std::size_t end = begin + 1;
        while (end < uses.size() && uses[end].key == uses[begin].key) ++end;
        if (end - begin == 2U && uses[begin].chart != uses[begin + 1U].chart) {
            for (std::uint32_t sample = 0; sample < samples_per_edge; ++sample) {
                const float t =
                    (static_cast<float>(sample) + 0.5F) /
                    static_cast<float>(samples_per_edge);
                const mvs::Vec2f first =
                    uses[begin].low_uv * (1.F - t) + uses[begin].high_uv * t;
                const mvs::Vec2f second = uses[begin + 1U].low_uv * (1.F - t) +
                    uses[begin + 1U].high_uv * t;
                pairs.insert(
                    pairs.end(), {first.x(), first.y(), second.x(), second.y()});
            }
        }
        begin = end;
    }
    return pairs;
}

}  // namespace

TexturedMesh bake_mesh_texture(
    const mvs::MvsScene& scene, const TextureOptions& options) {
    if (scene.mesh.faces.empty() || scene.mesh.vertices.empty())
        throw std::runtime_error("Cannot bake texture: mesh is empty");
    if (scene.views.size() < 2)
        throw std::runtime_error("Cannot bake texture: need at least 2 views");
    if (!aether_drender::has_uv_atlas_backend())
        throw std::runtime_error(
            "UVAtlas backend unavailable; rebuild with AETHER_ENABLE_UVATLAS");

    auto views = load_texture_views(scene, options);
    if (options.delight) {
        delight_texture_views(views, options);
    }

    std::vector<float> positions(scene.mesh.vertices.size() * 3U);
    for (std::size_t i = 0; i < scene.mesh.vertices.size(); ++i) {
        positions[i * 3] = scene.mesh.vertices[i].x();
        positions[i * 3 + 1] = scene.mesh.vertices[i].y();
        positions[i * 3 + 2] = scene.mesh.vertices[i].z();
    }
    std::vector<std::uint32_t> indices(scene.mesh.faces.size() * 3U);
    for (std::size_t i = 0; i < scene.mesh.faces.size(); ++i) {
        indices[i * 3] = static_cast<std::uint32_t>(scene.mesh.faces[i][0]);
        indices[i * 3 + 1] = static_cast<std::uint32_t>(scene.mesh.faces[i][1]);
        indices[i * 3 + 2] = static_cast<std::uint32_t>(scene.mesh.faces[i][2]);
    }

    core::StageScope unwrap_stage("texture.uv_unwrap");
    aether_drender::UvAtlasOptions uv_opts;
    uv_opts.width = options.atlas_resolution;
    uv_opts.height = options.atlas_resolution;
    uv_opts.gutter = options.uv_gutter;
    uv_opts.max_stretch = options.uv_max_stretch;
    uv_opts.parallel_partitions = options.uv_parallel_partitions;
    core::Logger::instance().info(
        "texture UV config: faces=", scene.mesh.faces.size(),
        " requested_partitions=", uv_opts.parallel_partitions,
        " atlas=", uv_opts.width, "x", uv_opts.height,
        " gutter=", uv_opts.gutter,
        " max_stretch=", uv_opts.max_stretch);
    const aether_drender::UvAtlasOutput unwrapped =
        aether_drender::unwrap_uv(positions, indices, uv_opts);
    core::Logger::instance().info(
        "texture UV result: partitions=", unwrapped.partition_count,
        " charts=", unwrapped.chart_count,
        " vertices=", unwrapped.positions.size() / 3U,
        " max_stretch=", unwrapped.max_stretch);
    unwrap_stage.finish();

    std::vector<float> normals;
    if (scene.mesh.normals.size() == scene.mesh.vertices.size()) {
        normals.resize(unwrapped.positions.size());
        for (std::size_t i = 0; i < unwrapped.vertex_remap.size(); ++i) {
            const std::uint32_t src = unwrapped.vertex_remap[i];
            normals[i * 3] = scene.mesh.normals[src].x();
            normals[i * 3 + 1] = scene.mesh.normals[src].y();
            normals[i * 3 + 2] = scene.mesh.normals[src].z();
        }
    } else {
        compute_vertex_normals(
            unwrapped.positions, unwrapped.indices, normals);
    }

    std::vector<aether_drender::ProjectionView> projections;
    projections.reserve(scene.views.size());
    for (std::size_t i = 0; i < scene.views.size(); ++i) {
        const mvs::MvsView& view = scene.views[i];
        const auto [near_z, far_z] = depth_range_for_view(view, scene.mesh);
        aether_drender::ProjectionView proj;
        proj.width = views[i].width;
        proj.height = views[i].height;
        proj.channel_count = 3;
        proj.image = views[i].rgb;
        proj.visibility_mask = views[i].mask;
        proj.world_to_clip = world_to_clip_row_major(view, near_z, far_z);
        const Eigen::Vector3f cam = view.pose.C.cast<float>();
        proj.camera_position = {cam.x(), cam.y(), cam.z()};
        projections.push_back(std::move(proj));
    }

    core::StageScope bake_stage("texture.project");
    aether_drender::Context context({options.vulkan_device_index, false});
    aether_drender::TextureBaker baker(context);
    aether_drender::TextureBakeOptions bake_opts;
    bake_opts.width = options.atlas_resolution;
    bake_opts.height = options.atlas_resolution;
    bake_opts.blend_mode =
        options.blend_mode == BlendMode::best_view
            ? aether_drender::ProjectionBlendMode::best_view
            : aether_drender::ProjectionBlendMode::weighted_average;
    bake_opts.visibility_mode =
        options.visibility_mode == VisibilityMode::shadow_map
            ? aether_drender::VisibilityMode::shadow_map
            : aether_drender::VisibilityMode::hybrid_ray_query;
    bake_opts.pcf_radius = options.pcf_radius;
    bake_opts.allow_visibility_fallback = options.allow_visibility_fallback;

    const aether_drender::TextureBakeOutput baked = baker.bake(
        unwrapped.positions, normals, unwrapped.uv, unwrapped.indices,
        projections, bake_opts);
    bake_stage.finish();

    std::vector<float> final_rgb(
        static_cast<std::size_t>(baked.width) * baked.height * 3U);
    const std::uint32_t baked_channels =
        baked.color.size() /
                (static_cast<std::size_t>(baked.width) * baked.height) >=
            4U
        ? 4U
        : 3U;
    for (std::size_t pixel = 0;
         pixel < static_cast<std::size_t>(baked.width) * baked.height; ++pixel)
        for (std::size_t channel = 0; channel < 3U; ++channel)
            final_rgb[pixel * 3U + channel] =
                baked.color[pixel * baked_channels + channel];

    aether_drender::TextureRefineOutput refined;
    std::vector<float> seam_pairs;
    if (options.optimize) {
        if (options.optimize_steps == 0 || options.optimize_batch_size == 0)
            throw std::invalid_argument(
                "Texture optimization steps and batch size must be positive");
        seam_pairs = build_dense_seam_pairs(
            unwrapped, options.seam_samples_per_edge);
        core::StageScope refine_stage("texture.optimize");
        aether_drender::TextureRefineOptions refine_opts;
        refine_opts.width = baked.width;
        refine_opts.height = baked.height;
        refine_opts.steps = options.optimize_steps;
        refine_opts.batch_size = options.optimize_batch_size;
        refine_opts.learning_rate = options.optimize_learning_rate;
        refine_opts.minimum_learning_rate = options.optimize_min_learning_rate;
        refine_opts.seam_polish_steps = options.seam_polish_steps;
        refine_opts.seam_learning_rate = options.seam_learning_rate;
        aether_drender::TextureRefiner refiner(context);
        refined = refiner.refine(
            final_rgb, unwrapped.positions, unwrapped.uv, unwrapped.indices,
            projections, seam_pairs, refine_opts);
        final_rgb = refined.color;
        refine_stage.finish();
        core::Logger::instance().info(
            "texture optimize: steps=", refined.loss_history.size(),
            " seam_pairs=", seam_pairs.size() / 4U,
            " seam_steps=", refined.seam_loss_history.size(),
            " precompute_s=", refined.precompute_seconds,
            " optimize_s=", refined.optimization_seconds,
            refined.loss_history.empty() ? "" : " loss_first=",
            refined.loss_history.empty() ? 0.F : refined.loss_history.front(),
            refined.loss_history.empty() ? "" : " loss_last=",
            refined.loss_history.empty() ? 0.F : refined.loss_history.back());
    }

    TexturedMesh result;
    result.positions.resize(unwrapped.positions.size() / 3U);
    result.normals.resize(normals.size() / 3U);
    result.uvs.resize(unwrapped.uv.size() / 2U);
    for (std::size_t i = 0; i < result.positions.size(); ++i) {
        result.positions[i] = {
            unwrapped.positions[i * 3], unwrapped.positions[i * 3 + 1],
            unwrapped.positions[i * 3 + 2]};
        result.normals[i] = {
            normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]};
        result.uvs[i] = {unwrapped.uv[i * 2], unwrapped.uv[i * 2 + 1]};
    }
    result.indices = unwrapped.indices;
    result.vertex_remap = unwrapped.vertex_remap;
    result.face_chart_ids = unwrapped.face_chart_ids;
    result.chart_count = unwrapped.chart_count;
    result.max_stretch = unwrapped.max_stretch;
    result.atlas_width = baked.width;
    result.atlas_height = baked.height;
    result.atlas_confidence = baked.confidence;
    result.atlas_valid = baked.valid_mask;
    result.used_ray_query = baked.used_ray_query;
    result.delighted = options.delight;
    result.optimized = options.optimize;
    result.atlas_rgb = std::move(final_rgb);
    result.optimization_loss = std::move(refined.loss_history);
    result.seam_loss = std::move(refined.seam_loss_history);
    result.optimization_precompute_seconds = refined.precompute_seconds;
    result.optimization_seconds = refined.optimization_seconds;

    std::size_t valid = 0;
    for (const auto v : result.atlas_valid)
        if (v > 0.5F) ++valid;
    core::Logger::instance().info(
        "texture bake: atlas=", result.atlas_width, "x", result.atlas_height,
        " charts=", result.chart_count,
        " coverage=",
        result.atlas_valid.empty()
            ? 0.0
            : static_cast<double>(valid) /
                  static_cast<double>(result.atlas_valid.size()),
        " ray_query=", result.used_ray_query,
        " delight=", result.delighted,
        " optimized=", result.optimized);
    return result;
}

void bake_and_export(
    const mvs::MvsScene& scene, const std::filesystem::path& output_stem,
    const TextureOptions& options) {
    const TexturedMesh mesh = bake_mesh_texture(scene, options);
    save_textured_obj(mesh, output_stem);
}

}  // namespace aetherscan::texture
