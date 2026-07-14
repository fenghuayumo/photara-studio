#include "mvs/densify.hpp"
#include "mvs/internal.hpp"

#include "core/logging.hpp"
#include "parallel/thread_pool.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
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
    const DepthMap* depth_external{};

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

    [[nodiscard]] const DepthMap& source_depth() const {
        return depth_external != nullptr ? *depth_external : depth;
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
    // H = K_src (R + t*n^T/d) K_ref^-1. The constant rotation
    // and translation factors are cached once per source/level.
    Mat3f rotation_h{Mat3f::Identity()};
    Vec3f translation_k{Vec3f::Zero()};
    Mat3f ref_k_inverse{Mat3f::Identity()};
};

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
    const ScaledView& ref, const sfm::Pose3D& ref_pose, const ScaledView& src,
    const sfm::Pose3D& src_pose, const int x, const int y, const float depth,
    const Vec3f& /*normal*/) {
    const DepthMap& source_depth = src.source_depth();
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
    float d1 = 0.F;
    // Nearest integer sample with depth similarity gate.
    const int sx = static_cast<int>(std::lround(u));
    const int sy = static_cast<int>(std::lround(v));
    d1 = source_depth.depth[source_depth.index(sx, sy)];
    if (d1 <= 0.F) return 4.F;
    if (std::abs(cam1.z() - d1) > 0.03F * std::max(cam1.z(), d1)) return 4.F;
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
                ref, ref_pose, *sources[i].view, *sources[i].pose, x, y, depth,
                normal);
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
    const std::vector<ScaledView>& neighbors,
    const std::vector<sfm::Pose3D>& neighbor_poses,
    const float d_min, const float d_max, const DensifyOptions& options,
    std::mt19937& rng, const bool use_geo,
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
        const Mat3f source_k = neighbors[i].K();
        sources.push_back(SourceContext{
            &neighbors[i], &neighbor_poses[i], source_k * rotation * ref_k_inverse,
            source_k * translation, ref_k_inverse});
    }
    constexpr float robust = 2.F;
    const float inv_min = 1.F / d_max;
    const float inv_max = 1.F / d_min;
    std::uniform_real_distribution<float> uni01(0.F, 1.F);
    auto random_depth = [&]() {
        const float t = uni01(rng);
        // Uniform inverse-depth allocates hypotheses approximately uniformly
        // in image disparity instead of wasting most trials far from camera.
        return 1.F / (inv_min + t * (inv_max - inv_min));
    };

    for (std::uint32_t y = 0; y < ref.height; ++y) {
        for (std::uint32_t x = 0; x < ref.width; ++x) {
            const std::size_t idx = dm.index(static_cast<int>(x), static_cast<int>(y));
            PatchRef patch;
            if (!fill_patch(ref, static_cast<int>(x), static_cast<int>(y), patch)) {
                dm.depth[idx] = 0.F;
                continue;
            }
            if (dm.depth[idx] <= 0.F) {
                if (!initialize_invalid) {
                    dm.confidence[idx] = robust;
                    continue;
                }
                dm.depth[idx] = random_depth();
                dm.normal[idx] = random_normal(rng, patch.x0);
            }
            dm.confidence[idx] = score_views(
                patch, static_cast<int>(x), static_cast<int>(y), ref, ref_pose,
                sources, dm.depth[idx], dm.normal[idx], robust, use_geo,
                options.geometric_weight,
                options.min_patch_views);
        }
    }

    const unsigned iters = use_geo ? std::max(1U, options.geometric_iters)
                                   : options.estimation_iters;
    for (unsigned iter = 0; iter < iters; ++iter) {
        const bool forward = (iter % 2) == 0;
        const int y0 = forward ? 0 : static_cast<int>(ref.height) - 1;
        const int y1 = forward ? static_cast<int>(ref.height) : -1;
        const int ys = forward ? 1 : -1;
        const int x0 = forward ? 0 : static_cast<int>(ref.width) - 1;
        const int x1 = forward ? static_cast<int>(ref.width) : -1;
        const int xs = forward ? 1 : -1;

        for (int y = y0; y != y1; y += ys) {
            for (int x = x0; x != x1; x += xs) {
                const std::size_t idx = dm.index(x, y);
                if (dm.depth[idx] <= 0.F) continue;
                PatchRef patch;
                if (!fill_patch(ref, x, y, patch)) continue;

                float best_depth = dm.depth[idx];
                Vec3f best_normal = dm.normal[idx];
                float best_conf = dm.confidence[idx];

                const int nxs[2] = {x - xs, x};
                const int nys[2] = {y, y - ys};
                for (int k = 0; k < 2; ++k) {
                    const int px = nxs[k];
                    const int py = nys[k];
                    if (px < 0 || py < 0 || px >= static_cast<int>(ref.width) ||
                        py >= static_cast<int>(ref.height))
                        continue;
                    const std::size_t nidx = dm.index(px, py);
                    if (dm.depth[nidx] <= 0.F) continue;
                    const Vec3f& n = dm.normal[nidx];
                    const Vec3f x0n{
                        (static_cast<float>(px) - ref.cx) / ref.fx,
                        (static_cast<float>(py) - ref.cy) / ref.fy, 1.F};
                    const float plane_d = n.dot(x0n) * dm.depth[nidx];
                    const float denom = n.dot(patch.x0);
                    if (std::abs(denom) < 1e-8F) continue;
                    const float depth_p = plane_d / denom;
                    if (depth_p < d_min || depth_p > d_max) continue;
                    const float conf = score_views(
                        patch, x, y, ref, ref_pose, sources, depth_p, n, robust,
                        use_geo,
                        options.geometric_weight, options.min_patch_views);
                    if (conf < best_conf) {
                        best_conf = conf;
                        best_depth = depth_p;
                        best_normal = n;
                    }
                }

                float depth_range = best_depth * (use_geo ? 0.05F : 0.5F);
                float angle_range = use_geo ? 0.2F : 1.0F;
                for (unsigned r = 0; r < options.random_iters; ++r) {
                    std::uniform_real_distribution<float> depth_off(
                        -depth_range, depth_range);
                    float nd = std::clamp(best_depth + depth_off(rng), d_min, d_max);
                    Vec3f nn =
                        perturb_normal(rng, best_normal, patch.x0, angle_range);
                    const float conf = score_views(
                        patch, x, y, ref, ref_pose, sources, nd, nn, robust,
                        use_geo,
                        options.geometric_weight, options.min_patch_views);
                    if (conf < best_conf) {
                        best_conf = conf;
                        best_depth = nd;
                        best_normal = nn;
                    }
                    depth_range *= 0.5F;
                    angle_range *= 0.5F;
                }

                dm.depth[idx] = best_depth;
                dm.normal[idx] = best_normal;
                dm.confidence[idx] = best_conf;
            }
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
    MvsScene& scene, const std::vector<detail::ViewImage>& images,
    const Index view_id, const DensifyOptions& options, std::mt19937& rng) {
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

    const unsigned levels = options.sub_resolution_levels + 1;
    ScaledView current;
    for (unsigned level = 0; level < levels; ++level) {
        const unsigned scale_div = 1U << (levels - 1 - level);
        ScaledView scaled = make_scaled(view, images[view_id], scale_div);
        std::vector<ScaledView> neighbors;
        std::vector<sfm::Pose3D> neighbor_poses;
        neighbors.reserve(neighbor_ids.size());
        neighbor_poses.reserve(neighbor_ids.size());
        for (const Index nid : neighbor_ids) {
            neighbors.push_back(make_scaled(scene.views[nid], images[nid], scale_div));
            neighbor_poses.push_back(scene.views[nid].pose);
        }

        if (level == 0) {
            init_from_sparse(scaled, view, scene, d_min, d_max);
        } else {
            scaled.depth.resize(scaled.width, scaled.height);
            upsample_depth(current.depth, scaled.depth);
        }

        run_patchmatch_level(
            scaled, view.pose, neighbors, neighbor_poses, d_min, d_max, options,
            rng, false, true);
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
    MvsScene& scene, const std::vector<detail::ViewImage>& images,
    const std::vector<DepthMap>& depth_snapshot, const Index view_id,
    const DensifyOptions& options, std::mt19937& rng) {
    MvsView& view = scene.views[view_id];
    if (view.neighbors.empty() || view.depth_map.depth.empty()) return;

    std::vector<Index> neighbor_ids;
    for (const auto& n : view.neighbors) neighbor_ids.push_back(n.view_id);

    ScaledView ref = make_scaled(view, images[view_id], 1);
    ref.depth = depth_snapshot[view_id];
    std::vector<ScaledView> neighbors;
    std::vector<sfm::Pose3D> neighbor_poses;
    for (const Index nid : neighbor_ids) {
        ScaledView nb = make_scaled(scene.views[nid], images[nid], 1);
        nb.depth_external = &depth_snapshot[nid];
        neighbors.push_back(std::move(nb));
        neighbor_poses.push_back(scene.views[nid].pose);
    }

    run_patchmatch_level(
        ref, view.pose, neighbors, neighbor_poses, view.depth_map.depth_min,
        view.depth_map.depth_max, options, rng, true, false);

    view.depth_map = std::move(ref.depth);
    for (std::size_t i = 0; i < view.depth_map.depth.size(); ++i) {
        if (view.depth_map.confidence[i] > options.ncc_keep_threshold) {
            view.depth_map.depth[i] = 0.F;
            view.depth_map.normal[i] = Vec3f::Zero();
        }
    }
}

}  // namespace

void estimate_depth_maps(MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.estimate_depth");
    const auto images = detail::load_view_images(scene, options);
    const unsigned threads = parallel::resolve_thread_count(scene.thread_count);

    {
        core::ProgressReporter progress("mvs.estimate_depth", scene.views.size());
        parallel::parallel_for(scene.views.size(), threads, [&](const std::size_t i) {
            std::mt19937 rng(
                static_cast<unsigned>(0xA37E5CA) ^
                static_cast<unsigned>(i) * 0x9E3779B9u);
            estimate_one_view_photometric(
                scene, images, static_cast<Index>(i), options, rng);
            progress.advance();
        });
    }

    if (options.geometric_consistency && options.geometric_iters > 0) {
        core::StageScope geo_stage("mvs.geometric_consistency");
        // Snapshot depths so parallel refine only reads a frozen neighbor state.
        std::vector<DepthMap> depth_snapshot(scene.views.size());
        for (std::size_t i = 0; i < scene.views.size(); ++i)
            depth_snapshot[i] = scene.views[i].depth_map;
        core::ProgressReporter progress("mvs.geometric_consistency", scene.views.size());
        parallel::parallel_for(scene.views.size(), threads, [&](const std::size_t i) {
            std::mt19937 rng(
                static_cast<unsigned>(0xC0FFEE) ^
                static_cast<unsigned>(i) * 0x9E3779B9u);
            refine_one_view_geometric(
                scene, images, depth_snapshot, static_cast<Index>(i), options, rng);
            progress.advance();
        });
        geo_stage.finish();
    }

    std::size_t valid_pixels = 0;
    for (const auto& view : scene.views)
        for (const float d : view.depth_map.depth)
            if (d > 0.F) ++valid_pixels;
    core::Logger::instance().info("mvs depth: valid_pixels=", valid_pixels);
    stage.finish();
}

}  // namespace aetherscan::mvs
