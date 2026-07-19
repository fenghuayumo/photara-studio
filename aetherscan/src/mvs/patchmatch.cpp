#include "mvs/densify.hpp"
#include "mvs/internal.hpp"

#include "core/logging.hpp"
#include "parallel/thread_pool.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <limits>
#include <random>
#include <vector>

namespace aetherscan::mvs {
namespace {

constexpr int k_half_window = 4;
constexpr int k_step = 2;
constexpr int k_texels = 25;

struct ScaledView {
    std::uint32_t width{};
    std::uint32_t height{};
    float fx{}, fy{}, cx{}, cy{};
    std::vector<float> gray;
    const std::vector<float>* gray_external{};
    std::vector<std::uint8_t> mask;
    const std::vector<std::uint8_t>* mask_external{};
    DepthMap depth;

    [[nodiscard]] const std::vector<float>& gray_pixels() const {
        return gray_external != nullptr ? *gray_external : gray;
    }

    [[nodiscard]] const std::vector<std::uint8_t>& mask_pixels() const {
        return mask_external != nullptr ? *mask_external : mask;
    }

    [[nodiscard]] bool foreground(const int x, const int y) const {
        const auto& pixels = mask_pixels();
        return pixels.empty() ||
               pixels[static_cast<std::size_t>(y) * width +
                      static_cast<std::size_t>(x)] != 0;
    }

    [[nodiscard]] Mat3f K() const {
        Mat3f k = Mat3f::Identity();
        k(0, 0) = fx;
        k(1, 1) = fy;
        k(0, 2) = cx;
        k(1, 2) = cy;
        return k;
    }

    [[nodiscard]] Vec3f unproject(const float u, const float v, const float d) const {
        return {(u - cx) / fx * d, (v - cy) / fy * d, d};
    }

    [[nodiscard]] bool project(const Vec3f& cam, float& u, float& v) const {
        if (cam.z() <= 1e-6F) return false;
        const float inv_z = 1.F / cam.z();
        u = fx * cam.x() * inv_z + cx;
        v = fy * cam.y() * inv_z + cy;
        return std::isfinite(u) && std::isfinite(v);
    }
};

struct SourceContext {
    const ScaledView* view{};
    const sfm::Pose3D* pose{};
    const DepthMap* source_depth{};
    // H = K_src (R + t*n^T/d) K_ref^-1. The constant rotation
    // and translation factors are cached once per source/level.
    Mat3f rotation_h{Mat3f::Identity()};
    Vec3f translation_k{Vec3f::Zero()};
    Mat3f ref_k_inverse{Mat3f::Identity()};
};

using ImagePyramids = std::vector<std::vector<ScaledView>>;

struct PatchRef {
    float texels[k_texels]{};
    float norm_sq{0.F};
    Vec3f x0{Vec3f::Zero()};
};

[[nodiscard]] ScaledView make_scaled(
    const MvsView& view, const detail::ViewImage& image, const unsigned scale_div) {
    ScaledView out;
    out.width = std::max(1U, view.width / scale_div);
    out.height = std::max(1U, view.height / scale_div);
    const float sx = static_cast<float>(out.width) / static_cast<float>(view.width);
    const float sy = static_cast<float>(out.height) / static_cast<float>(view.height);
    out.fx = view.fx * sx;
    out.fy = view.fy * sy;
    out.cx = view.cx * sx;
    out.cy = view.cy * sy;
    if (scale_div == 1 && out.width == view.width && out.height == view.height) {
        out.gray_external = &image.gray;
        if (!image.mask.empty()) out.mask_external = &image.mask;
        return out;
    }
    out.gray.resize(static_cast<std::size_t>(out.width) * out.height);
    for (std::uint32_t y = 0; y < out.height; ++y) {
        for (std::uint32_t x = 0; x < out.width; ++x) {
            const float u =
                (static_cast<float>(x) + 0.5F) / sx - 0.5F;
            const float v =
                (static_cast<float>(y) + 0.5F) / sy - 0.5F;
            float sample = 0.F;
            (void)detail::sample_gray(
                image.gray, view.width, view.height, u, v, sample);
            out.gray[static_cast<std::size_t>(y) * out.width + x] = sample;
        }
    }
    if (!image.mask.empty()) {
        out.mask.resize(static_cast<std::size_t>(out.width) * out.height);
        for (std::uint32_t y = 0; y < out.height; ++y) {
            for (std::uint32_t x = 0; x < out.width; ++x) {
                const int source_x = std::clamp(
                    static_cast<int>(std::lround(
                        (static_cast<float>(x) + 0.5F) / sx - 0.5F)),
                    0, static_cast<int>(view.width) - 1);
                const int source_y = std::clamp(
                    static_cast<int>(std::lround(
                        (static_cast<float>(y) + 0.5F) / sy - 0.5F)),
                    0, static_cast<int>(view.height) - 1);
                out.mask[static_cast<std::size_t>(y) * out.width + x] =
                    image.mask
                        [static_cast<std::size_t>(source_y) * view.width +
                         static_cast<std::size_t>(source_x)];
            }
        }
    }
    return out;
}

[[nodiscard]] ScaledView make_working_view(const ScaledView& cached) {
    ScaledView out;
    out.width = cached.width;
    out.height = cached.height;
    out.fx = cached.fx;
    out.fy = cached.fy;
    out.cx = cached.cx;
    out.cy = cached.cy;
    out.gray_external = &cached.gray_pixels();
    const auto& mask = cached.mask_pixels();
    if (!mask.empty()) out.mask_external = &mask;
    return out;
}

ImagePyramids build_image_pyramids(
    const MvsScene& scene, const std::vector<detail::ViewImage>& images,
    const unsigned levels, const unsigned threads) {
    core::StageScope stage("mvs.build_pyramids");
    ImagePyramids pyramids(scene.views.size());
    parallel::parallel_for(
        scene.views.size(), threads, [&](const std::size_t view_id) {
            auto& pyramid = pyramids[view_id];
            pyramid.reserve(levels);
            for (unsigned level = 0; level < levels; ++level) {
                const unsigned scale_div = 1U << (levels - 1 - level);
                pyramid.push_back(
                    make_scaled(scene.views[view_id], images[view_id], scale_div));
            }
        });
    stage.finish();
    return pyramids;
}

template <class Function>
void run_view_tile_batches(
    const std::size_t count, const unsigned threads,
    const unsigned requested_concurrent_views, Function&& function) {
    if (count == 0) return;
    const unsigned maximum_groups = std::max(
        1U, std::min({requested_concurrent_views, threads,
                      static_cast<unsigned>(count)}));
    for (std::size_t begin = 0; begin < count; begin += maximum_groups) {
        const unsigned active = static_cast<unsigned>(
            std::min<std::size_t>(maximum_groups, count - begin));
        const unsigned threads_per_view = std::max(1U, threads / active);
        std::vector<std::future<void>> futures;
        futures.reserve(active);
        for (unsigned group = 0; group < active; ++group) {
            const std::size_t view = begin + group;
            futures.emplace_back(std::async(
                std::launch::async, [&, view, threads_per_view] {
                    function(view, threads_per_view);
                }));
        }
        parallel::wait_all(futures);
    }
}

[[nodiscard]] bool fill_patch(
    const ScaledView& view, const int x, const int y, PatchRef& patch) {
    if (x < k_half_window || y < k_half_window ||
        x + k_half_window >= static_cast<int>(view.width) ||
        y + k_half_window >= static_cast<int>(view.height))
        return false;
    if (!view.foreground(x, y)) return false;
    float sum = 0.F;
    int n = 0;
    int foreground_texels = 0;
    const auto& pixels = view.gray_pixels();
    for (int dy = -k_half_window; dy <= k_half_window; dy += k_step) {
        for (int dx = -k_half_window; dx <= k_half_window; dx += k_step) {
            const float v = pixels
                [static_cast<std::size_t>(y + dy) * view.width +
                 static_cast<std::size_t>(x + dx)];
            patch.texels[n++] = v;
            sum += v;
            foreground_texels += view.foreground(x + dx, y + dy) ? 1 : 0;
        }
    }
    if (foreground_texels < k_texels * 4 / 5) return false;
    const float mean = sum / static_cast<float>(k_texels);
    patch.norm_sq = 0.F;
    for (int i = 0; i < k_texels; ++i) {
        patch.texels[i] -= mean;
        patch.norm_sq += patch.texels[i] * patch.texels[i];
    }
    // Near-constant patches have arbitrary NCC optima and create large sheets
    // of random depth. Keep a deliberately low floor so weakly textured
    // surfaces still pass while flat/no-data regions do not.
    if (patch.norm_sq < 2.5e-4F) return false;
    patch.x0 = {(static_cast<float>(x) - view.cx) / view.fx,
                (static_cast<float>(y) - view.cy) / view.fy, 1.F};
    return true;
}

[[nodiscard]] Mat3f plane_H(
    const SourceContext& source, const float depth, const Vec3f& normal,
    const Vec3f& x0) {
    const float denom = normal.dot(x0) * depth;
    Mat3f homography = source.rotation_h;
    if (std::abs(denom) > 1e-8F) {
        homography += source.translation_k *
                      (normal.transpose() * source.ref_k_inverse) / denom;
    }
    return homography;
}

[[nodiscard]] float score_ncc(
    const PatchRef& patch, const int cx, const int cy,
    const SourceContext& source, const float depth, const Vec3f& normal,
    const float robust) {
    if (depth <= 0.F || normal.dot(patch.x0) >= 0.F) return robust;
    const Mat3f H = plane_H(source, depth, normal, patch.x0);
    const ScaledView& src = *source.view;
    const auto& source_pixels = src.gray_pixels();
    float sum = 0.F;
    float sum_sq = 0.F;
    float num = 0.F;
    int n = 0;
    for (int dy = -k_half_window; dy <= k_half_window; dy += k_step) {
        for (int dx = -k_half_window; dx <= k_half_window; dx += k_step) {
            const Eigen::Vector3f p =
                H * Eigen::Vector3f{
                        static_cast<float>(cx + dx), static_cast<float>(cy + dy),
                        1.F};
            if (std::abs(p.z()) < 1e-8F) return robust;
            float sample = 0.F;
            const float source_x = p.x() / p.z();
            const float source_y = p.y() / p.z();
            const int mask_x = static_cast<int>(std::lround(source_x));
            const int mask_y = static_cast<int>(std::lround(source_y));
            if (mask_x < 0 || mask_y < 0 ||
                mask_x >= static_cast<int>(src.width) ||
                mask_y >= static_cast<int>(src.height) ||
                !src.foreground(mask_x, mask_y))
                return robust;
            if (!detail::sample_gray(
                    source_pixels, src.width, src.height, source_x, source_y,
                    sample))
                return robust;
            sum += sample;
            sum_sq += sample * sample;
            num += patch.texels[n++] * sample;
        }
    }
    const float mean = sum / static_cast<float>(k_texels);
    const float norm_sq1 = sum_sq - static_cast<float>(k_texels) * mean * mean;
    const float nrm = patch.norm_sq * norm_sq1;
    if (nrm <= 1e-16F) return robust;
    const float ncc = std::clamp(num / std::sqrt(nrm), -1.F, 1.F);
    return std::min(2.F, 1.F - ncc);
}

[[nodiscard]] float geometric_penalty(
    const ScaledView& ref, const sfm::Pose3D& ref_pose,
    const SourceContext& source,
    const sfm::Pose3D& src_pose, const int x, const int y, const float depth,
    const Vec3f& /*normal*/) {
    const ScaledView& src = *source.view;
    if (source.source_depth == nullptr) return 4.F;
    const DepthMap& source_depth = *source.source_depth;
    if (source_depth.depth.empty() || depth <= 0.F) return 4.F;
    const Vec3f cam0 = ref.unproject(static_cast<float>(x), static_cast<float>(y), depth);
    const Vec3f world =
        ref_pose.transform_camera_to_world(cam0.cast<double>()).cast<float>();
    const Vec3f cam1 =
        src_pose.transform_world_to_camera(world.cast<double>()).cast<float>();
    float u = 0.F;
    float v = 0.F;
    if (!src.project(cam1, u, v) || cam1.z() <= 1e-6F) return 4.F;
    if (u < 1.F || v < 1.F || u >= static_cast<float>(src.width - 1) ||
        v >= static_cast<float>(src.height - 1))
        return 4.F;
    // A projected correspondence is sub-pixel. Searching the surrounding
    // 3x3 neighborhood avoids turning normal projection quantization into a
    // false geometric inconsistency, while the depth gate prevents crossing
    // a real discontinuity.
    int sx = -1;
    int sy = -1;
    float d1 = 0.F;
    float best = std::numeric_limits<float>::infinity();
    const int center_x = static_cast<int>(std::lround(u));
    const int center_y = static_cast<int>(std::lround(v));
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            const int px = center_x + ox;
            const int py = center_y + oy;
            if (px < 0 || py < 0 || px >= static_cast<int>(src.width) ||
                py >= static_cast<int>(src.height))
                continue;
            const float candidate = source_depth.depth[source_depth.index(px, py)];
            if (candidate <= 0.F) continue;
            const float relative =
                std::abs(cam1.z() - candidate) / std::max(cam1.z(), candidate);
            if (relative > 0.03F) continue;
            const float pixel = std::hypot(
                static_cast<float>(px) - u, static_cast<float>(py) - v);
            const float score = relative + pixel * 1e-3F;
            if (score < best) {
                best = score;
                sx = px;
                sy = py;
                d1 = candidate;
            }
        }
    }
    if (sx < 0) return 4.F;
    const Vec3f cam_back = src.unproject(static_cast<float>(sx), static_cast<float>(sy), d1);
    const Vec3f world_back =
        src_pose.transform_camera_to_world(cam_back.cast<double>()).cast<float>();
    const Vec3f cam_ref =
        ref_pose.transform_world_to_camera(world_back.cast<double>()).cast<float>();
    float bu = 0.F;
    float bv = 0.F;
    if (!ref.project(cam_ref, bu, bv)) return 4.F;
    const float dist = std::hypot(bu - static_cast<float>(x), bv - static_cast<float>(y));
    return std::min(4.F, std::sqrt(dist * (dist + 2.F)));
}

[[nodiscard]] float score_views(
    const PatchRef& patch, const int x, const int y, const ScaledView& ref,
    const sfm::Pose3D& ref_pose, const std::vector<SourceContext>& sources,
    const float depth,
    const Vec3f& normal, const float robust, const bool use_geo,
    const float geo_weight, const unsigned min_patch_views) {
    if (sources.empty()) return robust;
    // This function is on the hottest PatchMatch path. A fixed buffer avoids
    // one heap allocation for every pixel/hypothesis evaluation.
    std::array<float, 64> scores{};
    std::size_t count = 0;
    const std::size_t count_views = std::min(sources.size(), scores.size());
    for (std::size_t i = 0; i < count_views; ++i) {
        const float photo = score_ncc(
            patch, x, y, sources[i], depth, normal, robust);
        if (!(photo < robust)) continue;
        float score = photo;
        if (use_geo && geo_weight > 0.F) {
            const float geo = geometric_penalty(
                ref, ref_pose, sources[i], *sources[i].pose, x, y, depth, normal);
            // A missing/inconsistent neighbor depth is not evidence for this
            // hypothesis; do not let a good photometric match hide it.
            if (!(geo < 4.F)) continue;
            score += geo_weight * geo;
        }
        scores[count++] = score;
    }
    const std::size_t required = std::min<std::size_t>(
        std::max(1U, min_patch_views), count_views);
    if (count < required) return robust;
    if (required < count) {
        std::nth_element(
            scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(required),
            scores.begin() + static_cast<std::ptrdiff_t>(count));
    }
    float sum = 0.F;
    for (std::size_t i = 0; i < required; ++i) sum += scores[i];
    return sum / static_cast<float>(required);
}

[[nodiscard]] Vec3f random_normal(std::mt19937& rng, const Vec3f& x0) {
    std::uniform_real_distribution<float> uni(-1.F, 1.F);
    Vec3f n;
    do {
        n = Vec3f{uni(rng), uni(rng), uni(rng)}.normalized();
        if (n.dot(x0) > 0.F) n = -n;
    } while (n.dot(x0) >= -0.1F);
    return n;
}

[[nodiscard]] Vec3f perturb_normal(
    std::mt19937& rng, const Vec3f& normal, const Vec3f& x0, const float angle_rad) {
    std::normal_distribution<float> gauss(0.F, angle_rad);
    Eigen::AngleAxisf ax(gauss(rng), Vec3f::UnitX());
    Eigen::AngleAxisf ay(gauss(rng), Vec3f::UnitY());
    Vec3f n = (ay * ax * normal).normalized();
    if (n.dot(x0) > 0.F) n = -n;
    return n;
}

void upsample_depth(const DepthMap& coarse, DepthMap& fine) {
    fine.depth_min = coarse.depth_min;
    fine.depth_max = coarse.depth_max;
    for (std::uint32_t y = 0; y < fine.height; ++y) {
        for (std::uint32_t x = 0; x < fine.width; ++x) {
            const float u =
                (static_cast<float>(x) + 0.5F) * static_cast<float>(coarse.width) /
                    static_cast<float>(fine.width) -
                0.5F;
            const float v =
                (static_cast<float>(y) + 0.5F) * static_cast<float>(coarse.height) /
                    static_cast<float>(fine.height) -
                0.5F;
            const int cx = std::clamp(
                static_cast<int>(std::lround(u)), 0,
                static_cast<int>(coarse.width) - 1);
            const int cy = std::clamp(
                static_cast<int>(std::lround(v)), 0,
                static_cast<int>(coarse.height) - 1);
            const std::size_t ci = coarse.index(cx, cy);
            const std::size_t fi = fine.index(static_cast<int>(x), static_cast<int>(y));
            fine.depth[fi] = coarse.depth[ci];
            fine.normal[fi] = coarse.normal[ci];
            fine.confidence[fi] = coarse.confidence[ci];
        }
    }
}

void run_patchmatch_level(
    ScaledView& ref, const sfm::Pose3D& ref_pose,
    const std::vector<const ScaledView*>& neighbors,
    const std::vector<sfm::Pose3D>& neighbor_poses,
    const std::vector<const DepthMap*>& neighbor_depths,
    const float d_min, const float d_max, const DensifyOptions& options,
    const unsigned random_seed, const unsigned threads, const bool use_geo,
    const bool initialize_invalid) {
    DepthMap& dm = ref.depth;
    std::vector<SourceContext> sources;
    sources.reserve(neighbors.size());
    const Mat3f ref_k_inverse = ref.K().inverse();
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        const Mat3f rotation =
            (neighbor_poses[i].R * ref_pose.R.transpose()).cast<float>();
        const Vec3f translation =
            (neighbor_poses[i].R * (ref_pose.C - neighbor_poses[i].C)).cast<float>();
        const Mat3f source_k = neighbors[i]->K();
        sources.push_back(SourceContext{
            neighbors[i], &neighbor_poses[i],
            i < neighbor_depths.size() ? neighbor_depths[i] : nullptr,
            source_k * rotation * ref_k_inverse, source_k * translation,
            ref_k_inverse});
    }
    constexpr float robust = 2.F;
    const float inv_min = 1.F / d_max;
    const float inv_max = 1.F / d_min;

    const unsigned tile_rows = std::max(1U, options.patchmatch_tile_rows);
    const std::size_t tile_count =
        (static_cast<std::size_t>(ref.height) + tile_rows - 1U) / tile_rows;
    parallel::parallel_for(
        tile_count, threads, [&](const std::size_t tile) {
            std::mt19937 random(
                random_seed ^ static_cast<unsigned>(tile * 0x9E3779B9u));
            std::uniform_real_distribution<float> uniform(0.F, 1.F);
            const std::uint32_t begin = static_cast<std::uint32_t>(tile) * tile_rows;
            const std::uint32_t end =
                std::min(ref.height, begin + tile_rows);
            for (std::uint32_t y = begin; y < end; ++y) {
                for (std::uint32_t x = 0; x < ref.width; ++x) {
                    const std::size_t idx =
                        dm.index(static_cast<int>(x), static_cast<int>(y));
                    PatchRef patch;
                    if (!fill_patch(
                            ref, static_cast<int>(x), static_cast<int>(y), patch)) {
                        dm.depth[idx] = 0.F;
                        continue;
                    }
                    if (dm.depth[idx] <= 0.F) {
                        if (!initialize_invalid) {
                            dm.confidence[idx] = robust;
                            continue;
                        }
                        const float t = uniform(random);
                        // Uniform inverse depth approximately samples disparity.
                        dm.depth[idx] =
                            1.F / (inv_min + t * (inv_max - inv_min));
                        dm.normal[idx] = random_normal(random, patch.x0);
                    }
                    dm.confidence[idx] = score_views(
                        patch, static_cast<int>(x), static_cast<int>(y), ref,
                        ref_pose, sources, dm.depth[idx], dm.normal[idx], robust,
                        use_geo, options.geometric_weight,
                        options.min_patch_views);
                }
            }
        });

    // Geometric rounds are coordinated globally in estimate_depth_maps so
    // every sweep sees the previous round's updated neighbor maps.
    const unsigned iters = use_geo ? 1U : options.estimation_iters;
    for (unsigned iter = 0; iter < iters; ++iter) {
        // Red/black propagation makes every pixel in one phase independent.
        // This permits view-internal tile parallelism while preserving local
        // plane propagation between the two phases.
        for (unsigned parity = 0; parity < 2; ++parity) {
            parallel::parallel_for(
                tile_count, threads, [&](const std::size_t tile) {
                    std::mt19937 random(
                        random_seed ^ (iter + 1U) * 0x85EBCA6Bu ^
                        (parity + 1U) * 0xC2B2AE35u ^
                        static_cast<unsigned>(tile * 0x27D4EB2Du));
                    const int begin = static_cast<int>(tile * tile_rows);
                    const int end = std::min(
                        static_cast<int>(ref.height),
                        begin + static_cast<int>(tile_rows));
                    const int direction = (iter & 1U) == 0U ? -1 : 1;
                    const int offsets[2][2] = {
                        {direction, 0}, {0, direction}};
                    for (int y = begin; y < end; ++y) {
                        for (int x = 0; x < static_cast<int>(ref.width); ++x) {
                            if ((static_cast<unsigned>(x + y) & 1U) != parity)
                                continue;
                            const std::size_t idx = dm.index(x, y);
                            if (dm.depth[idx] <= 0.F) continue;
                            PatchRef patch;
                            if (!fill_patch(ref, x, y, patch)) continue;

                            float best_depth = dm.depth[idx];
                            Vec3f best_normal = dm.normal[idx];
                            float best_conf = dm.confidence[idx];
                            for (const auto& offset : offsets) {
                                const int px = x + offset[0];
                                const int py = y + offset[1];
                                if (px < 0 || py < 0 ||
                                    px >= static_cast<int>(ref.width) ||
                                    py >= static_cast<int>(ref.height))
                                    continue;
                                const std::size_t nidx = dm.index(px, py);
                                if (dm.depth[nidx] <= 0.F) continue;
                                const Vec3f& normal = dm.normal[nidx];
                                const Vec3f neighbor_ray{
                                    (static_cast<float>(px) - ref.cx) / ref.fx,
                                    (static_cast<float>(py) - ref.cy) / ref.fy,
                                    1.F};
                                const float plane_distance =
                                    normal.dot(neighbor_ray) * dm.depth[nidx];
                                const float denominator = normal.dot(patch.x0);
                                if (std::abs(denominator) < 1e-8F) continue;
                                const float candidate_depth =
                                    plane_distance / denominator;
                                if (candidate_depth < d_min ||
                                    candidate_depth > d_max)
                                    continue;
                                const float confidence = score_views(
                                    patch, x, y, ref, ref_pose, sources,
                                    candidate_depth, normal, robust, use_geo,
                                    options.geometric_weight,
                                    options.min_patch_views);
                                if (confidence < best_conf) {
                                    best_conf = confidence;
                                    best_depth = candidate_depth;
                                    best_normal = normal;
                                }
                            }

                            float depth_range =
                                best_depth * (use_geo ? 0.05F : 0.5F);
                            float angle_range = use_geo ? 0.2F : 1.0F;
                            for (unsigned trial = 0;
                                 trial < options.random_iters; ++trial) {
                                std::uniform_real_distribution<float> depth_offset(
                                    -depth_range, depth_range);
                                const float candidate_depth = std::clamp(
                                    best_depth + depth_offset(random), d_min, d_max);
                                const Vec3f candidate_normal = perturb_normal(
                                    random, best_normal, patch.x0, angle_range);
                                const float confidence = score_views(
                                    patch, x, y, ref, ref_pose, sources,
                                    candidate_depth, candidate_normal, robust,
                                    use_geo, options.geometric_weight,
                                    options.min_patch_views);
                                if (confidence < best_conf) {
                                    best_conf = confidence;
                                    best_depth = candidate_depth;
                                    best_normal = candidate_normal;
                                }
                                depth_range *= 0.5F;
                                angle_range *= 0.5F;
                            }
                            dm.depth[idx] = best_depth;
                            dm.normal[idx] = best_normal;
                            dm.confidence[idx] = best_conf;
                        }
                    }
                });
        }
    }
}

void init_from_sparse(
    ScaledView& view, const MvsView& full, const MvsScene& scene, const float d_min,
    const float d_max) {
    view.depth.view_id = full.id;
    view.depth.resize(view.width, view.height);
    view.depth.depth_min = d_min;
    view.depth.depth_max = d_max;
    const float sx = static_cast<float>(view.width) / static_cast<float>(full.width);
    const float sy = static_cast<float>(view.height) / static_cast<float>(full.height);
    for (const auto& point : scene.sparse_points) {
        bool observes = false;
        for (const Index id : point.view_ids) {
            if (id == full.id) {
                observes = true;
                break;
            }
        }
        if (!observes) continue;
        const Vec3f cam =
            full.pose.transform_world_to_camera(point.position.cast<double>())
                .cast<float>();
        if (cam.z() <= 1e-6F) continue;
        float u = 0.F;
        float v = 0.F;
        if (!full.project(cam, u, v)) continue;
        const int x = static_cast<int>(std::lround(u * sx));
        const int y = static_cast<int>(std::lround(v * sy));
        if (x < 0 || y < 0 || x >= static_cast<int>(view.width) ||
            y >= static_cast<int>(view.height))
            continue;
        if (!view.foreground(x, y)) continue;
        const std::size_t idx = view.depth.index(x, y);
        view.depth.depth[idx] = cam.z();
        view.depth.normal[idx] = -cam.normalized();
        view.depth.confidence[idx] = 0.5F;
    }
}

void estimate_one_view_photometric(
    MvsScene& scene, const ImagePyramids& pyramids, const Index view_id,
    const DensifyOptions& options, const unsigned threads,
    const unsigned random_seed) {
    MvsView& view = scene.views[view_id];
    if (view.neighbors.empty()) return;

    std::vector<Index> neighbor_ids;
    neighbor_ids.reserve(view.neighbors.size());
    for (const auto& n : view.neighbors) neighbor_ids.push_back(n.view_id);

    float d_min = 1e9F;
    float d_max = 0.F;
    for (const auto& point : scene.sparse_points) {
        bool observes = false;
        for (const Index id : point.view_ids) {
            if (id == view_id) {
                observes = true;
                break;
            }
        }
        if (!observes) continue;
        const double z =
            view.pose.transform_world_to_camera(point.position.cast<double>()).z();
        if (z > 1e-6) {
            d_min = std::min(d_min, static_cast<float>(z));
            d_max = std::max(d_max, static_cast<float>(z));
        }
    }
    if (!(d_min < d_max)) {
        d_min = 0.1F;
        d_max = 100.F;
    } else {
        d_min *= 0.8F;
        d_max *= 1.25F;
    }

    const unsigned levels = static_cast<unsigned>(pyramids[view_id].size());
    ScaledView current;
    for (unsigned level = 0; level < levels; ++level) {
        ScaledView scaled = make_working_view(pyramids[view_id][level]);
        std::vector<const ScaledView*> neighbors;
        std::vector<sfm::Pose3D> neighbor_poses;
        std::vector<const DepthMap*> neighbor_depths;
        neighbors.reserve(neighbor_ids.size());
        neighbor_poses.reserve(neighbor_ids.size());
        neighbor_depths.reserve(neighbor_ids.size());
        for (const Index nid : neighbor_ids) {
            neighbors.push_back(&pyramids[nid][level]);
            neighbor_poses.push_back(scene.views[nid].pose);
            neighbor_depths.push_back(nullptr);
        }

        if (level == 0) {
            init_from_sparse(scaled, view, scene, d_min, d_max);
        } else {
            scaled.depth.resize(scaled.width, scaled.height);
            upsample_depth(current.depth, scaled.depth);
        }

        run_patchmatch_level(
            scaled, view.pose, neighbors, neighbor_poses, neighbor_depths, d_min,
            d_max, options, random_seed ^ level * 0x9E3779B9u, threads, false,
            true);
        current = std::move(scaled);
    }

    view.depth_map = std::move(current.depth);
    view.depth_map.view_id = view.id;

    for (std::size_t i = 0; i < view.depth_map.depth.size(); ++i) {
        if (view.depth_map.confidence[i] > options.ncc_keep_threshold) {
            view.depth_map.depth[i] = 0.F;
            view.depth_map.normal[i] = Vec3f::Zero();
        }
    }
}

void refine_one_view_geometric(
    MvsScene& scene, const ImagePyramids& pyramids,
    const std::vector<DepthMap>& depth_snapshot, const Index view_id,
    const DensifyOptions& options, const unsigned threads,
    const unsigned random_seed) {
    MvsView& view = scene.views[view_id];
    if (view.neighbors.empty() || view.depth_map.depth.empty()) return;

    std::vector<Index> neighbor_ids;
    for (const auto& n : view.neighbors) neighbor_ids.push_back(n.view_id);

    ScaledView ref = make_working_view(pyramids[view_id].back());
    ref.depth = depth_snapshot[view_id];
    std::vector<const ScaledView*> neighbors;
    std::vector<sfm::Pose3D> neighbor_poses;
    std::vector<const DepthMap*> neighbor_depths;
    for (const Index nid : neighbor_ids) {
        neighbors.push_back(&pyramids[nid].back());
        neighbor_poses.push_back(scene.views[nid].pose);
        neighbor_depths.push_back(&depth_snapshot[nid]);
    }

    run_patchmatch_level(
        ref, view.pose, neighbors, neighbor_poses, neighbor_depths,
        view.depth_map.depth_min, view.depth_map.depth_max, options, random_seed,
        threads, true, false);

    view.depth_map = std::move(ref.depth);
    for (std::size_t i = 0; i < view.depth_map.depth.size(); ++i) {
        if (view.depth_map.confidence[i] > options.ncc_keep_threshold) {
            view.depth_map.depth[i] = 0.F;
            view.depth_map.normal[i] = Vec3f::Zero();
        }
    }
}

void filter_one_depth_map(
    MvsScene& scene, const std::vector<DepthMap>& depth_snapshot,
    const Index view_id, const DensifyOptions& options) {
    MvsView& ref = scene.views[view_id];
    const DepthMap& input = depth_snapshot[view_id];
    if (input.depth.empty() || ref.neighbors.empty()) return;

    DepthMap output = input;
    const unsigned required = std::min<unsigned>(
        options.min_views_filter, static_cast<unsigned>(ref.neighbors.size()));
    const float cos_normal = std::cos(
        options.normal_diff_threshold_deg * 3.14159265358979323846F / 180.F);
    const float relative_threshold = options.depth_diff_threshold * 1.2F;
    std::size_t kept = 0;

    for (std::uint32_t y = 0; y < ref.height; ++y) {
        for (std::uint32_t x = 0; x < ref.width; ++x) {
            const std::size_t index = input.index(
                static_cast<int>(x), static_cast<int>(y));
            const float depth0 = input.depth[index];
            const Vec3f normal0 = input.normal[index];
            if (depth0 <= 0.F || !normal0.allFinite() ||
                normal0.squaredNorm() < 0.5F) {
                output.depth[index] = 0.F;
                output.normal[index] = Vec3f::Zero();
                continue;
            }

            const Vec3f camera0 = ref.unproject(
                static_cast<float>(x), static_cast<float>(y), depth0);
            const Vec3f world0 =
                ref.pose.transform_camera_to_world(camera0.cast<double>()).cast<float>();
            const Vec3f world_normal0 =
                (ref.pose.R.transpose().cast<float>() * normal0).normalized();
            const Vec3f viewing_ray0 =
                (world0 - ref.pose.C.cast<float>()).normalized();
            const float reference_incidence =
                std::clamp(-world_normal0.dot(viewing_ray0), 0.F, 1.F);
            if (reference_incidence <= 0.F) {
                output.depth[index] = 0.F;
                output.normal[index] = Vec3f::Zero();
                output.confidence[index] = 2.F;
                continue;
            }
            const float reference_incidence_weight =
                options.grazing_weight_floor +
                (1.F - options.grazing_weight_floor) * reference_incidence;
            const float reference_weight = reference_incidence_weight * std::max(
                0.05F, 1.F - input.confidence[index] /
                                   std::max(options.ncc_keep_threshold, 1e-3F));
            float depth_sum = depth0 * reference_weight;
            Vec3f world_normal_sum = world_normal0 * reference_weight;
            float weight_sum = reference_weight;
            unsigned agreeing = 0;

            for (const NeighborScore& neighbor : ref.neighbors) {
                if (neighbor.view_id >= scene.views.size()) continue;
                const MvsView& src = scene.views[neighbor.view_id];
                const DepthMap& source_depth = depth_snapshot[neighbor.view_id];
                if (source_depth.depth.empty()) continue;

                const Vec3f predicted =
                    src.pose.transform_world_to_camera(world0.cast<double>()).cast<float>();
                float u = 0.F;
                float v = 0.F;
                if (!src.project(predicted, u, v)) continue;
                const int center_x = static_cast<int>(std::lround(u));
                const int center_y = static_cast<int>(std::lround(v));

                int best_x = -1;
                int best_y = -1;
                float best_depth = 0.F;
                float best_score = std::numeric_limits<float>::infinity();
                for (int oy = -1; oy <= 1; ++oy) {
                    for (int ox = -1; ox <= 1; ++ox) {
                        const int sx = center_x + ox;
                        const int sy = center_y + oy;
                        if (sx < 0 || sy < 0 ||
                            sx >= static_cast<int>(src.width) ||
                            sy >= static_cast<int>(src.height))
                            continue;
                        const std::size_t source_index = source_depth.index(sx, sy);
                        const float candidate = source_depth.depth[source_index];
                        if (candidate <= 0.F) continue;
                        const float relative = std::abs(predicted.z() - candidate) /
                                               std::max(predicted.z(), candidate);
                        if (relative > relative_threshold) continue;
                        const float pixel = std::hypot(
                            static_cast<float>(sx) - u,
                            static_cast<float>(sy) - v);
                        const float score = relative + pixel * 1e-3F;
                        if (score < best_score) {
                            best_score = score;
                            best_x = sx;
                            best_y = sy;
                            best_depth = candidate;
                        }
                    }
                }
                if (best_x < 0) continue;

                const std::size_t source_index =
                    source_depth.index(best_x, best_y);
                const Vec3f source_normal = source_depth.normal[source_index];
                if (!source_normal.allFinite() ||
                    source_normal.squaredNorm() < 0.5F)
                    continue;
                const Vec3f world_normal =
                    (src.pose.R.transpose().cast<float>() * source_normal).normalized();
                if (world_normal0.dot(world_normal) < cos_normal) continue;

                const Vec3f source_camera = src.unproject(
                    static_cast<float>(best_x), static_cast<float>(best_y), best_depth);
                const Vec3f source_world =
                    src.pose.transform_camera_to_world(source_camera.cast<double>())
                        .cast<float>();
                const Vec3f source_viewing_ray =
                    (source_world - src.pose.C.cast<float>()).normalized();
                const float source_incidence = std::clamp(
                    -world_normal.dot(source_viewing_ray), 0.F, 1.F);
                if (source_incidence <= 0.F) continue;
                const Vec3f back_camera =
                    ref.pose.transform_world_to_camera(source_world.cast<double>())
                        .cast<float>();
                float back_u = 0.F;
                float back_v = 0.F;
                if (!ref.project(back_camera, back_u, back_v) ||
                    std::hypot(
                        back_u - static_cast<float>(x),
                        back_v - static_cast<float>(y)) >
                        options.reprojection_error_px)
                    continue;

                const float source_incidence_weight =
                    options.grazing_weight_floor +
                    (1.F - options.grazing_weight_floor) * source_incidence;
                const float weight = source_incidence_weight * std::max(
                    0.05F, 1.F - source_depth.confidence[source_index] /
                                       std::max(options.ncc_keep_threshold, 1e-3F));
                depth_sum += back_camera.z() * weight;
                world_normal_sum += world_normal * weight;
                weight_sum += weight;
                ++agreeing;
            }

            if (agreeing < required || weight_sum <= 0.F) {
                output.depth[index] = 0.F;
                output.normal[index] = Vec3f::Zero();
                output.confidence[index] = 2.F;
                continue;
            }
            if (options.adjust_filtered_depth) {
                const float adjusted_depth = depth_sum / weight_sum;
                if (adjusted_depth >= input.depth_min &&
                    adjusted_depth <= input.depth_max)
                    output.depth[index] = adjusted_depth;
                if (world_normal_sum.squaredNorm() > 1e-10F) {
                    const Vec3f adjusted_world_normal = world_normal_sum.normalized();
                    output.normal[index] =
                        (ref.pose.R.cast<float>() * adjusted_world_normal).normalized();
                }
            }
            ++kept;
        }
    }
    ref.depth_map = std::move(output);
    (void)kept;
}

}  // namespace

void estimate_depth_maps(MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.estimate_depth");
    const auto images = detail::load_view_images(scene, options);
    const unsigned threads = parallel::resolve_thread_count(scene.thread_count);
    const unsigned levels = options.sub_resolution_levels + 1;
    const ImagePyramids pyramids =
        build_image_pyramids(scene, images, levels, threads);

    {
        core::ProgressReporter progress("mvs.estimate_depth", scene.views.size());
        run_view_tile_batches(
            scene.views.size(), threads, options.patchmatch_concurrent_views,
            [&](const std::size_t i, const unsigned view_threads) {
                estimate_one_view_photometric(
                    scene, pyramids, static_cast<Index>(i), options,
                    view_threads,
                    static_cast<unsigned>(0xA37E5CA) ^
                        static_cast<unsigned>(i) * 0x9E3779B9u);
                progress.advance();
            });
    }

    if (options.geometric_consistency && options.geometric_iters > 0) {
        core::StageScope geo_stage("mvs.geometric_consistency");
        for (unsigned round = 0; round < options.geometric_iters; ++round) {
            // Snapshot every global round. Writers never race with neighbor
            // readers, and the next round observes all updates.
            std::vector<DepthMap> depth_snapshot(scene.views.size());
            for (std::size_t i = 0; i < scene.views.size(); ++i)
                depth_snapshot[i] = scene.views[i].depth_map;
            core::ProgressReporter progress(
                "mvs.geometric_consistency", scene.views.size());
            run_view_tile_batches(
                scene.views.size(), threads,
                options.patchmatch_concurrent_views,
                [&](const std::size_t i, const unsigned view_threads) {
                    refine_one_view_geometric(
                        scene, pyramids, depth_snapshot,
                        static_cast<Index>(i), options, view_threads,
                        static_cast<unsigned>(
                            0xC0FFEE + round * 0x10001U) ^
                            static_cast<unsigned>(i) * 0x9E3779B9u);
                    progress.advance();
                });
        }
        geo_stage.finish();
    }

    if (options.filter_depth_maps && options.min_views_filter > 0) {
        core::StageScope filter_stage("mvs.filter_depth");
        std::vector<DepthMap> depth_snapshot(scene.views.size());
        for (std::size_t i = 0; i < scene.views.size(); ++i)
            depth_snapshot[i] = scene.views[i].depth_map;
        core::ProgressReporter progress("mvs.filter_depth", scene.views.size());
        parallel::parallel_for(
            scene.views.size(), threads, [&](const std::size_t i) {
                filter_one_depth_map(
                    scene, depth_snapshot, static_cast<Index>(i), options);
                progress.advance();
            });
        filter_stage.finish();
    }

    std::size_t valid_pixels = 0;
    for (const auto& view : scene.views)
        for (const float d : view.depth_map.depth)
            if (d > 0.F) ++valid_pixels;
    core::Logger::instance().info("mvs depth: valid_pixels=", valid_pixels);
    stage.finish();
}

}  // namespace aetherscan::mvs
