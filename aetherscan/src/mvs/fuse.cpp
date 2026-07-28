#include "mvs/densify.hpp"

#include "core/logging.hpp"
#include "io/image.hpp"
#include "parallel/thread_pool.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace aetherscan::mvs {
namespace {

constexpr std::size_t k_max_fusion_views = 64;

[[nodiscard]] bool depth_similar(
    const float d0, const float d1, const float rel_thresh) {
    return d0 > 0.F && d1 > 0.F &&
           std::abs(d0 - d1) <= rel_thresh * std::max(d0, d1);
}

[[nodiscard]] Vec3f sample_color(
    const io::RgbImage& image, const float u, const float v) {
    if (image.pixels.empty() || u < 0.F || v < 0.F ||
        u > static_cast<float>(image.width - 1) ||
        v > static_cast<float>(image.height - 1))
        return Vec3f{0.5F, 0.5F, 0.5F};
    const int x0 = static_cast<int>(u);
    const int y0 = static_cast<int>(v);
    const int x1 = std::min(x0 + 1, static_cast<int>(image.width) - 1);
    const int y1 = std::min(y0 + 1, static_cast<int>(image.height) - 1);
    const float tx = u - static_cast<float>(x0);
    const float ty = v - static_cast<float>(y0);
    const auto at = [&](const int x, const int y) {
        const std::size_t i =
            (static_cast<std::size_t>(y) * image.width +
             static_cast<std::size_t>(x)) *
            3;
        return Vec3f{
            image.pixels[i] / 255.F, image.pixels[i + 1] / 255.F,
            image.pixels[i + 2] / 255.F};
    };
    return ((1.F - tx) * at(x0, y0) + tx * at(x1, y0)) * (1.F - ty) +
           ((1.F - tx) * at(x0, y1) + tx * at(x1, y1)) * ty;
}

[[nodiscard]] Vec3f sample_view_color(
    const io::RgbImage& image, const MvsView& view, const float u,
    const float v) {
    const float xn = (u - view.cx) / view.fx;
    const float yn = (v - view.cy) / view.fy;
    const float r2 = xn * xn + yn * yn;
    const float radial = 1.F + view.k1 * r2 + view.k2 * r2 * r2;
    const float xd =
        xn * radial + 2.F * view.p1 * xn * yn +
        view.p2 * (r2 + 2.F * xn * xn);
    const float yd =
        yn * radial + view.p1 * (r2 + 2.F * yn * yn) +
        2.F * view.p2 * xn * yn;
    return sample_color(
        image, view.src_fx * xd + view.src_cx,
        view.src_fy * yd + view.src_cy);
}

void remove_depth_speckles(
    DepthMap& dm, const unsigned min_size, const float rel_threshold) {
    if (dm.depth.empty() || min_size <= 1) return;
    std::vector<std::uint8_t> visited(dm.size(), 0);
    std::vector<std::size_t> queue;
    queue.reserve(std::min<std::size_t>(dm.size(), 4096));

    for (std::uint32_t y = 0; y < dm.height; ++y) {
        for (std::uint32_t x = 0; x < dm.width; ++x) {
            const std::size_t seed = dm.index(static_cast<int>(x), static_cast<int>(y));
            if (visited[seed] || dm.depth[seed] <= 0.F) continue;
            queue.clear();
            queue.push_back(seed);
            visited[seed] = 1;
            for (std::size_t head = 0; head < queue.size(); ++head) {
                const std::size_t index = queue[head];
                const int px = static_cast<int>(index % dm.width);
                const int py = static_cast<int>(index / dm.width);
                constexpr int dx[4] = {-1, 1, 0, 0};
                constexpr int dy[4] = {0, 0, -1, 1};
                for (int k = 0; k < 4; ++k) {
                    const int nx = px + dx[k];
                    const int ny = py + dy[k];
                    if (nx < 0 || ny < 0 || nx >= static_cast<int>(dm.width) ||
                        ny >= static_cast<int>(dm.height))
                        continue;
                    const std::size_t ni = dm.index(nx, ny);
                    if (visited[ni] ||
                        !depth_similar(
                            dm.depth[index], dm.depth[ni], rel_threshold))
                        continue;
                    visited[ni] = 1;
                    queue.push_back(ni);
                }
            }
            if (queue.size() >= min_size) continue;
            for (const std::size_t index : queue) {
                dm.depth[index] = 0.F;
                dm.normal[index] = Vec3f::Zero();
                dm.confidence[index] = 2.F;
            }
        }
    }
}

[[nodiscard]] float estimate_pixel_footprint(const MvsScene& scene) {
    std::vector<float> footprints;
    footprints.reserve(scene.sparse_points.size() * 2);
    for (const auto& point : scene.sparse_points) {
        for (const Index view_id : point.view_ids) {
            if (view_id >= scene.views.size()) continue;
            const auto& view = scene.views[view_id];
            const float z = static_cast<float>(
                view.pose.transform_world_to_camera(point.position.cast<double>()).z());
            if (z > 0.F && std::isfinite(z))
                footprints.push_back(z / std::max(view.fx, 1.F));
        }
    }
    if (footprints.empty()) return 1e-3F;
    const std::size_t middle = footprints.size() / 2;
    std::nth_element(
        footprints.begin(), footprints.begin() + static_cast<std::ptrdiff_t>(middle),
        footprints.end());
    return std::max(footprints[middle], 1e-7F);
}

struct GridKey {
    std::int64_t x{}, y{}, z{};
    std::int8_t normal_bin{};
    bool operator==(const GridKey&) const = default;
};

struct GridHash {
    std::size_t operator()(const GridKey& key) const noexcept {
        auto mix = [](std::uint64_t x) {
            x ^= x >> 30;
            x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27;
            x *= 0x94d049bb133111ebULL;
            return x ^ (x >> 31);
        };
        return static_cast<std::size_t>(
            mix(static_cast<std::uint64_t>(key.x)) ^
            (mix(static_cast<std::uint64_t>(key.y)) << 1) ^
            (mix(static_cast<std::uint64_t>(key.z)) << 2) ^
            static_cast<std::uint8_t>(key.normal_bin));
    }
};

[[nodiscard]] std::int8_t normal_bin(const Vec3f& normal) {
    int axis = 0;
    normal.cwiseAbs().maxCoeff(&axis);
    return static_cast<std::int8_t>(axis * 2 + (normal[axis] >= 0.F ? 1 : 0));
}

struct Accumulator {
    Vec3f position{Vec3f::Zero()};
    Vec3f normal{Vec3f::Zero()};
    Vec3f color{Vec3f::Zero()};
    float weight{0.F};
    float color_weight{0.F};
    std::array<Index, 32> views{};
    std::array<float, 32> view_weights{};
    std::uint8_t view_count{0};
};

void add_view(
    Accumulator& accumulator, const Index view, const float weight) {
    if (!(weight > 0.F) || !std::isfinite(weight)) return;
    for (std::uint8_t i = 0; i < accumulator.view_count; ++i) {
        if (accumulator.views[i] != view) continue;
        accumulator.view_weights[i] += weight;
        return;
    }
    if (accumulator.view_count < accumulator.views.size()) {
        const std::uint8_t index = accumulator.view_count++;
        accumulator.views[index] = view;
        accumulator.view_weights[index] = weight;
    }
}

void merge_accumulator(Accumulator& target, const Accumulator& source) {
    target.position += source.position;
    target.normal += source.normal;
    target.color += source.color;
    target.weight += source.weight;
    target.color_weight += source.color_weight;
    for (std::uint8_t i = 0; i < source.view_count; ++i)
        add_view(target, source.views[i], source.view_weights[i]);
}

}  // namespace

void fuse_depth_maps(MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.fuse");
    scene.dense_cloud.points.clear();
    const unsigned threads = parallel::resolve_thread_count(scene.thread_count);

    parallel::parallel_for(
        scene.views.size(), threads, [&](const std::size_t i) {
            remove_depth_speckles(
                scene.views[i].depth_map, options.speckle_size,
                options.depth_diff_threshold * 0.7F);
        });

    std::vector<io::RgbImage> colors(scene.views.size());
    parallel::parallel_for(scene.views.size(), threads, [&](const std::size_t i) {
        try {
            colors[i] = io::load_rgb(scene.views[i].path);
        } catch (...) {
            colors[i] = {};
        }
    });

    const float cos_normal = std::cos(
        options.normal_diff_threshold_deg * 3.14159265358979323846F / 180.F);
    const float voxel = estimate_pixel_footprint(scene) * 0.7F;

    std::vector<std::pair<Index, std::uint32_t>> rows;
    for (std::size_t i = 0; i < scene.views.size(); ++i) {
        if (scene.views[i].depth_map.depth.empty()) continue;
        for (std::uint32_t y = 0; y < scene.views[i].height; ++y)
            rows.emplace_back(static_cast<Index>(i), y);
    }

    using Grid = std::unordered_map<GridKey, Accumulator, GridHash>;
    const unsigned worker_count = std::max(
        1U, std::min<unsigned>(threads, static_cast<unsigned>(rows.size())));
    // Keep every worker's writes private, but partition them by the final key
    // hash.  The former single Grid per worker made pixel fusion parallel and
    // then serialized tens of millions of entries into one global map.  With
    // the same partitioning on every worker, each shard can be reduced
    // independently without locks.
    const unsigned shard_count = worker_count;
    std::vector<std::vector<Grid>> worker_grids(worker_count);
    for (auto& shards : worker_grids) shards.resize(shard_count);
    const std::size_t reserve_per_worker =
        rows.empty() ? 0 : std::min<std::size_t>(
            1'000'000, rows.size() * 256 / worker_count);
    const std::size_t reserve_per_shard =
        (reserve_per_worker + shard_count - 1) / shard_count;
    for (auto& shards : worker_grids)
        for (auto& grid : shards) grid.reserve(reserve_per_shard);

    parallel::parallel_for(
        rows.size(), worker_count,
        [&](const std::size_t row_index, const unsigned worker_id) {
            const auto [ref_id, y] = rows[row_index];
            const MvsView& ref = scene.views[ref_id];
            const DepthMap& rdm = ref.depth_map;
            auto& grids = worker_grids[worker_id];

            for (std::uint32_t x = 0; x < ref.width; ++x) {
                const std::size_t index =
                    rdm.index(static_cast<int>(x), static_cast<int>(y));
                if ((ref.foreground_mask.size() == rdm.size() &&
                     ref.foreground_mask[index] == 0) ||
                    rdm.depth[index] <= 0.F ||
                    !(rdm.confidence[index] <= options.ncc_keep_threshold))
                    continue;
                const Vec3f normal0 = rdm.normal[index];
                if (!normal0.allFinite() || normal0.squaredNorm() < 0.5F) continue;

                std::array<Vec3f, k_max_fusion_views> positions{};
                std::array<Vec3f, k_max_fusion_views> normals{};
                std::array<Vec3f, k_max_fusion_views> color_samples{};
                std::array<float, k_max_fusion_views> weights{};
                std::array<Index, k_max_fusion_views> view_ids{};
                std::array<std::uint8_t, k_max_fusion_views> has_color{};
                std::size_t count = 1;

                const Vec3f cam0 = ref.unproject(
                    static_cast<float>(x), static_cast<float>(y), rdm.depth[index]);
                const Vec3f world0 =
                    ref.pose.transform_camera_to_world(cam0.cast<double>()).cast<float>();
                if (scene.roi.valid && !scene.roi.contains(world0)) continue;
                const Vec3f world_n0 =
                    (ref.pose.R.transpose().cast<float>() * normal0).normalized();
                const Vec3f viewing_ray0 =
                    (world0 - ref.pose.C.cast<float>()).normalized();
                const float reference_incidence =
                    std::clamp(-world_n0.dot(viewing_ray0), 0.F, 1.F);
                if (reference_incidence <= 0.F) continue;
                const float reference_incidence_weight =
                    options.grazing_weight_floor +
                    (1.F - options.grazing_weight_floor) * reference_incidence;
                const float w0 = reference_incidence_weight * std::max(
                    0.02F,
                    1.F - rdm.confidence[index] /
                              std::max(options.ncc_keep_threshold, 1e-3F));
                positions[0] = world0;
                normals[0] = world_n0;
                weights[0] = w0;
                view_ids[0] = ref.id;
                if (!colors[ref_id].pixels.empty()) {
                    color_samples[0] = sample_view_color(
                        colors[ref_id], ref, static_cast<float>(x),
                        static_cast<float>(y));
                    has_color[0] = 1;
                }

                for (const auto& neighbor : ref.neighbors) {
                    if (count == k_max_fusion_views) break;
                    if (neighbor.view_id >= scene.views.size()) continue;
                    const MvsView& src = scene.views[neighbor.view_id];
                    const DepthMap& sdm = src.depth_map;
                    if (sdm.depth.empty()) continue;
                    const Vec3f cam_src =
                        src.pose.transform_world_to_camera(world0.cast<double>()).cast<float>();
                    float u = 0.F;
                    float v = 0.F;
                    if (!src.project(cam_src, u, v)) continue;
                    // Search the projected neighborhood instead of rounding to
                    // one pixel. This is important at oblique angles, where a
                    // one-pixel choice can jump to a different depth layer.
                    int sx = -1;
                    int sy = -1;
                    std::size_t source_index = 0;
                    float best_score = std::numeric_limits<float>::infinity();
                    const int center_x = static_cast<int>(std::lround(u));
                    const int center_y = static_cast<int>(std::lround(v));
                    const int radius = std::max(
                        1, static_cast<int>(
                               std::ceil(options.reprojection_error_px)));
                    for (int oy = -radius; oy <= radius; ++oy) {
                        for (int ox = -radius; ox <= radius; ++ox) {
                            const int px = center_x + ox;
                            const int py = center_y + oy;
                            if (px < 0 || py < 0 ||
                                px >= static_cast<int>(src.width) ||
                                py >= static_cast<int>(src.height))
                                continue;
                            const std::size_t candidate_index = sdm.index(px, py);
                            if (src.foreground_mask.size() == sdm.size() &&
                                src.foreground_mask[candidate_index] == 0)
                                continue;
                            const float candidate_depth = sdm.depth[candidate_index];
                            if (!depth_similar(
                                    cam_src.z(), candidate_depth,
                                    options.depth_diff_threshold) ||
                                !(sdm.confidence[candidate_index] <=
                                  options.ncc_keep_threshold))
                                continue;
                            const float relative =
                                std::abs(cam_src.z() - candidate_depth) /
                                std::max(cam_src.z(), candidate_depth);
                            const float pixel = std::hypot(
                                static_cast<float>(px) - u,
                                static_cast<float>(py) - v);
                            const float score = relative + pixel * 1e-3F;
                            if (score < best_score) {
                                best_score = score;
                                sx = px;
                                sy = py;
                                source_index = candidate_index;
                            }
                        }
                    }
                    if (sx < 0) continue;

                    const Vec3f source_normal = sdm.normal[source_index];
                    if (!source_normal.allFinite() ||
                        source_normal.squaredNorm() < 0.5F)
                        continue;
                    const Vec3f world_normal =
                        (src.pose.R.transpose().cast<float>() * source_normal)
                            .normalized();
                    if (world_n0.dot(world_normal) < cos_normal) continue;

                    const Vec3f source_cam_point = src.unproject(
                        static_cast<float>(sx), static_cast<float>(sy),
                        sdm.depth[source_index]);
                    const Vec3f world_point =
                        src.pose
                            .transform_camera_to_world(source_cam_point.cast<double>())
                            .cast<float>();
                    const Vec3f source_viewing_ray =
                        (world_point - src.pose.C.cast<float>()).normalized();
                    const float source_incidence = std::clamp(
                        -world_normal.dot(source_viewing_ray), 0.F, 1.F);
                    if (source_incidence <= 0.F) continue;
                    const Vec3f cam_back =
                        ref.pose.transform_world_to_camera(world_point.cast<double>())
                            .cast<float>();
                    float back_u = 0.F;
                    float back_v = 0.F;
                    if (!ref.project(cam_back, back_u, back_v) ||
                        std::hypot(
                            back_u - static_cast<float>(x),
                            back_v - static_cast<float>(y)) >
                            options.reprojection_error_px)
                        continue;

                    positions[count] = world_point;
                    normals[count] = world_normal;
                    const float source_incidence_weight =
                        options.grazing_weight_floor +
                        (1.F - options.grazing_weight_floor) * source_incidence;
                    weights[count] = source_incidence_weight * std::max(
                        0.02F,
                        1.F - sdm.confidence[source_index] /
                                  std::max(options.ncc_keep_threshold, 1e-3F));
                    view_ids[count] = src.id;
                    if (!colors[neighbor.view_id].pixels.empty()) {
                        color_samples[count] = sample_view_color(
                            colors[neighbor.view_id], src, u, v);
                        has_color[count] = 1;
                    }
                    ++count;
                }

                if (count < options.min_views_fuse) continue;

                // Robustly choose the consensus layer along the reference
                // viewing ray. Pairwise-valid samples can still straddle a
                // thin surface; a weighted median followed by a tight inlier
                // average avoids turning those samples into a thick shell.
                std::array<std::pair<float, std::size_t>, k_max_fusion_views>
                    ordered{};
                for (std::size_t i = 0; i < count; ++i) {
                    const Vec3f ref_camera =
                        ref.pose
                            .transform_world_to_camera(positions[i].cast<double>())
                            .cast<float>();
                    ordered[i] = {ref_camera.z(), i};
                }
                std::sort(ordered.begin(), ordered.begin() + count);
                float total_weight = 0.F;
                for (std::size_t i = 0; i < count; ++i)
                    total_weight += weights[i];
                float cumulative = 0.F;
                float median_depth = ordered[0].first;
                for (std::size_t i = 0; i < count; ++i) {
                    cumulative += weights[ordered[i].second];
                    if (cumulative >= total_weight * 0.5F) {
                        median_depth = ordered[i].first;
                        break;
                    }
                }
                Vec3f position = Vec3f::Zero();
                Vec3f normal = Vec3f::Zero();
                float weight = 0.F;
                std::size_t inlier_count = 0;
                for (std::size_t i = 0; i < count; ++i) {
                    if (!depth_similar(
                            ordered[i].first, median_depth,
                            options.depth_diff_threshold * 0.75F))
                        continue;
                    const std::size_t sample = ordered[i].second;
                    position += positions[sample] * weights[sample];
                    normal += normals[sample] * weights[sample];
                    weight += weights[sample];
                    ++inlier_count;
                }
                if (inlier_count < options.min_views_fuse) continue;
                position /= std::max(weight, 1e-6F);
                if (!position.allFinite() || normal.squaredNorm() < 1e-10F) continue;
                if (scene.roi.valid && !scene.roi.contains(position)) continue;
                normal.normalize();

                const GridKey key{
                    static_cast<std::int64_t>(std::floor(position.x() / voxel)),
                    static_cast<std::int64_t>(std::floor(position.y() / voxel)),
                    static_cast<std::int64_t>(std::floor(position.z() / voxel)),
                    normal_bin(normal)};
                Grid& grid =
                    grids[GridHash{}(key) % static_cast<std::size_t>(shard_count)];
                Accumulator& accumulator = grid[key];
                accumulator.position += position * weight;
                accumulator.normal += normal * weight;
                accumulator.weight += weight;
                for (std::size_t i = 0; i < count; ++i) {
                    if (!depth_similar(
                            ordered[i].first, median_depth,
                            options.depth_diff_threshold * 0.75F))
                        continue;
                    const std::size_t sample = ordered[i].second;
                    add_view(
                        accumulator, view_ids[sample], weights[sample]);
                    if (has_color[sample]) {
                        accumulator.color +=
                            color_samples[sample] * weights[sample];
                        accumulator.color_weight += weights[sample];
                    }
                }
            }
        });

    std::size_t grid_entries = 0;
    for (const auto& shards : worker_grids)
        for (const auto& grid : shards) grid_entries += grid.size();

    std::vector<Grid> fused_shards(shard_count);
    parallel::parallel_for(
        shard_count, worker_count, [&](const std::size_t shard) {
            std::size_t shard_entries = 0;
            for (const auto& grids : worker_grids)
                shard_entries += grids[shard].size();
            Grid& fused = fused_shards[shard];
            fused.reserve(shard_entries);
            for (const auto& grids : worker_grids)
                for (const auto& [key, value] : grids[shard])
                    merge_accumulator(fused[key], value);
        });

    std::vector<std::vector<DensePoint>> shard_points(shard_count);
    parallel::parallel_for(
        shard_count, worker_count, [&](const std::size_t shard) {
            Grid& fused = fused_shards[shard];
            auto& points = shard_points[shard];
            points.reserve(fused.size());
            for (auto& [key, accumulator] : fused) {
                (void)key;
                if (accumulator.weight <= 0.F ||
                    accumulator.view_count < options.min_views_fuse ||
                    accumulator.normal.squaredNorm() < 1e-10F)
                    continue;
                DensePoint point;
                point.position = accumulator.position / accumulator.weight;
                point.normal = accumulator.normal.normalized();
                point.color = accumulator.color_weight > 0.F
                                  ? accumulator.color /
                                        accumulator.color_weight
                                  : Vec3f{0.5F, 0.5F, 0.5F};
                point.weight = accumulator.weight;
                point.views.assign(
                    accumulator.views.begin(),
                    accumulator.views.begin() + accumulator.view_count);
                point.view_weights.assign(
                    accumulator.view_weights.begin(),
                    accumulator.view_weights.begin() +
                        accumulator.view_count);
                points.push_back(std::move(point));
            }
        });

    std::size_t point_count = 0;
    for (const auto& points : shard_points) point_count += points.size();
    scene.dense_cloud.points.reserve(point_count);
    for (auto& points : shard_points)
        std::move(
            points.begin(), points.end(),
            std::back_inserter(scene.dense_cloud.points));

    core::Logger::instance().info(
        "mvs fuse: points=", scene.dense_cloud.points.size(),
        " voxel=", voxel, " workers=", worker_count,
        " reduce_shards=", shard_count,
        " worker_grid_entries=", grid_entries);
    stage.finish();
}

}  // namespace aetherscan::mvs
