#include "mvs/internal.hpp"

#include "core/logging.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aetherscan::mvs::detail {
namespace {

struct VoxelKey {
    std::int64_t x{};
    std::int64_t y{};
    std::int64_t z{};

    [[nodiscard]] bool operator==(const VoxelKey&) const noexcept = default;
};

[[nodiscard]] bool key_less(const VoxelKey& a, const VoxelKey& b) noexcept {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}

struct VoxelHash {
    [[nodiscard]] std::size_t operator()(const VoxelKey& key) const noexcept {
        const auto mix = [](std::uint64_t value) {
            value ^= value >> 30U;
            value *= 0xbf58476d1ce4e5b9ULL;
            value ^= value >> 27U;
            value *= 0x94d049bb133111ebULL;
            return value ^ (value >> 31U);
        };
        const auto x = mix(static_cast<std::uint64_t>(key.x));
        const auto y = mix(static_cast<std::uint64_t>(key.y));
        const auto z = mix(static_cast<std::uint64_t>(key.z));
        return static_cast<std::size_t>(x ^ (y << 1U) ^ (z << 7U));
    }
};

struct EdgeKey {
    VoxelKey a;
    VoxelKey b;

    [[nodiscard]] bool operator==(const EdgeKey&) const noexcept = default;
};

struct EdgeHash {
    [[nodiscard]] std::size_t operator()(const EdgeKey& edge) const noexcept {
        const VoxelHash hash;
        const std::size_t a = hash(edge.a);
        const std::size_t b = hash(edge.b);
        return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6U) + (a >> 2U));
    }
};

struct Accumulator {
    double signed_distance{};
    double weight{};
};

using Field = std::unordered_map<VoxelKey, Accumulator, VoxelHash>;

constexpr std::array<std::array<int, 3>, 8> k_corners{{
    {{0, 0, 0}}, {{1, 0, 0}}, {{1, 1, 0}}, {{0, 1, 0}},
    {{0, 0, 1}}, {{1, 0, 1}}, {{1, 1, 1}}, {{0, 1, 1}},
}};

// Six tetrahedra sharing the 0-6 cube diagonal. The subdivision is identical
// on shared cube faces, so interpolated edge vertices remain crack-free.
constexpr std::array<std::array<int, 4>, 6> k_tetrahedra{{
    {{0, 5, 1, 6}}, {{0, 1, 2, 6}}, {{0, 2, 3, 6}},
    {{0, 3, 7, 6}}, {{0, 7, 4, 6}}, {{0, 4, 5, 6}},
}};

constexpr std::array<std::array<int, 2>, 6> k_tetra_edges{{
    {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}},
}};

[[nodiscard]] VoxelKey offset_key(
    const VoxelKey& key, const int x, const int y, const int z) noexcept {
    return {key.x + x, key.y + y, key.z + z};
}

[[nodiscard]] Vec3f position_of(
    const VoxelKey& key, const float voxel_size) noexcept {
    return Vec3f{
        static_cast<float>(key.x) * voxel_size,
        static_cast<float>(key.y) * voxel_size,
        static_cast<float>(key.z) * voxel_size};
}

[[nodiscard]] VoxelKey key_of(
    const Vec3f& position, const float inverse_voxel) noexcept {
    return {
        static_cast<std::int64_t>(
            std::llround(static_cast<double>(position.x() * inverse_voxel))),
        static_cast<std::int64_t>(
            std::llround(static_cast<double>(position.y() * inverse_voxel))),
        static_cast<std::int64_t>(
            std::llround(static_cast<double>(position.z() * inverse_voxel)))};
}

[[nodiscard]] float estimate_voxel_size(const MvsScene& scene) {
    std::vector<float> footprints;
    footprints.reserve(200'000);
    for (const MvsView& view : scene.views) {
        const DepthMap& map = view.depth_map;
        if (map.depth.size() != map.size() || !(view.fx > 0.F) ||
            !(view.fy > 0.F))
            continue;
        const unsigned stride = 16;
        for (unsigned y = 0; y < map.height; y += stride) {
            for (unsigned x = 0; x < map.width; x += stride) {
                const float depth = map.depth[map.index(x, y)];
                if (!(depth > 0.F) || !std::isfinite(depth)) continue;
                const float footprint = depth / std::min(view.fx, view.fy);
                if (footprint > 0.F && std::isfinite(footprint))
                    footprints.push_back(footprint);
            }
        }
    }
    if (footprints.empty()) return 0.F;
    const auto middle = footprints.begin() +
        static_cast<std::ptrdiff_t>(footprints.size() / 2);
    std::nth_element(footprints.begin(), middle, footprints.end());
    return *middle;
}

void integrate_view(
    const MvsView& view, const DensifyOptions& options,
    const OrientedBoundingBox& roi, const float voxel_size, Field& field,
    std::size_t& valid_pixels) {
    const DepthMap& map = view.depth_map;
    if (map.depth.size() != map.size()) return;

    const float inverse_voxel = 1.F / voxel_size;
    const float band_voxels = std::max(options.mesh_tsdf_truncation_voxels, 1.F);
    const float truncation = voxel_size * band_voxels;
    const int band_steps = std::max(1, static_cast<int>(std::ceil(band_voxels)));
    const unsigned pixel_step = std::max(options.mesh_tsdf_pixel_step, 1U);
    const Mat3f camera_to_world = view.pose.R.transpose().cast<float>();
    const Vec3f camera_center = view.pose.C.cast<float>();

    for (unsigned y = 0; y < map.height; y += pixel_step) {
        for (unsigned x = 0; x < map.width; x += pixel_step) {
            const std::size_t index = map.index(x, y);
            const float surface_depth = map.depth[index];
            if (!(surface_depth > 0.F) || !std::isfinite(surface_depth))
                continue;
            ++valid_pixels;

            const Vec3f ray = view.unproject(
                static_cast<float>(x) + 0.5F,
                static_cast<float>(y) + 0.5F, 1.F);
            const float ray_length = ray.norm();
            if (!(ray_length > 0.F) || !std::isfinite(ray_length)) continue;

            float weight = 1.F;
            if (map.confidence.size() == map.size()) {
                const float confidence = map.confidence[index];
                if (std::isfinite(confidence))
                    weight /= 1.F + std::max(confidence, 0.F);
            }
            if (map.normal.size() == map.size() &&
                map.normal[index].squaredNorm() > 1e-12F) {
                const float incidence = std::abs(
                    map.normal[index].normalized().dot(ray.normalized()));
                weight *= std::max(incidence, options.grazing_weight_floor);
            }

            bool have_last = false;
            VoxelKey last{};
            const float z_step = voxel_size / ray_length;
            for (int step = -band_steps; step <= band_steps; ++step) {
                const float sample_depth =
                    surface_depth + static_cast<float>(step) * z_step;
                if (!(sample_depth > 0.F)) continue;
                const Vec3f camera_point = ray * sample_depth;
                const Vec3f world_point =
                    camera_to_world * camera_point + camera_center;
                const VoxelKey key = key_of(world_point, inverse_voxel);
                if (have_last && key == last) continue;
                have_last = true;
                last = key;

                const Vec3f center_world = position_of(key, voxel_size);
                if (roi.valid && !roi.contains(center_world))
                    continue;
                const Vec3f center_camera =
                    view.pose
                        .transform_world_to_camera(center_world.cast<double>())
                        .cast<float>();
                const float signed_distance = std::clamp(
                    (surface_depth - center_camera.z()) * ray_length /
                        truncation,
                    -1.F, 1.F);
                Accumulator& value = field[key];
                value.signed_distance +=
                    static_cast<double>(signed_distance * weight);
                value.weight += static_cast<double>(weight);
            }
        }
    }
}

[[nodiscard]] EdgeKey edge_key(VoxelKey a, VoxelKey b) noexcept {
    if (key_less(b, a)) std::swap(a, b);
    return {a, b};
}

}  // namespace

bool reconstruct_mesh_tsdf(MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.mesh.tsdf");
    const float inferred_voxel = estimate_voxel_size(scene);
    const float voxel_size = options.mesh_tsdf_voxel_size > 0.F
        ? options.mesh_tsdf_voxel_size
        : inferred_voxel;
    if (!(voxel_size > 0.F) || !std::isfinite(voxel_size)) {
        core::Logger::instance().warning(
            "mvs mesh TSDF: unable to infer a valid voxel size");
        stage.finish();
        return false;
    }

    Field field;
    field.reserve(2'000'000);
    std::size_t valid_pixels = 0;
    for (const MvsView& view : scene.views)
        integrate_view(
            view, options, scene.roi, voxel_size, field, valid_pixels);

    core::Logger::instance().info(
        "mvs mesh TSDF integrate: voxel=", voxel_size,
        " truncation_voxels=", options.mesh_tsdf_truncation_voxels,
        " pixel_step=", options.mesh_tsdf_pixel_step,
        " depth_samples=", valid_pixels, " active_voxels=", field.size());
    if (field.empty()) {
        stage.finish();
        return false;
    }

    Mesh mesh;
    std::unordered_map<EdgeKey, int, EdgeHash> edge_vertices;
    edge_vertices.reserve(field.size() / 2);
    std::unordered_set<VoxelKey, VoxelHash> processed_cells;
    processed_cells.reserve(field.size());

    const auto value_at = [&](const VoxelKey& key, float& value) {
        const auto found = field.find(key);
        if (found == field.end() ||
            found->second.weight < options.mesh_tsdf_min_weight)
            return false;
        value = static_cast<float>(
            found->second.signed_distance / found->second.weight);
        return std::isfinite(value);
    };

    const auto vertex_on_edge = [&] (
        const VoxelKey& a, const VoxelKey& b, const float va,
        const float vb) {
        const EdgeKey edge = edge_key(a, b);
        const auto existing = edge_vertices.find(edge);
        if (existing != edge_vertices.end()) return existing->second;
        const float denominator = va - vb;
        const float t = std::abs(denominator) > 1e-12F
            ? std::clamp(va / denominator, 0.F, 1.F)
            : 0.5F;
        const Vec3f position =
            position_of(a, voxel_size) * (1.F - t) +
            position_of(b, voxel_size) * t;
        const int index = static_cast<int>(mesh.vertices.size());
        mesh.vertices.push_back(position);
        edge_vertices.emplace(edge, index);
        return index;
    };

    struct Crossing {
        int vertex{-1};
        Vec3f position{Vec3f::Zero()};
    };
    const float minimum_area_squared =
        voxel_size * voxel_size * voxel_size * voxel_size * 1e-8F;
    const auto append_triangle = [&] (
        int a, int b, int c, const Vec3f& outside_direction) {
        if (a == b || b == c || c == a) return;
        const Vec3f& p0 = mesh.vertices[static_cast<std::size_t>(a)];
        const Vec3f& p1 = mesh.vertices[static_cast<std::size_t>(b)];
        const Vec3f& p2 = mesh.vertices[static_cast<std::size_t>(c)];
        const Vec3f normal = (p1 - p0).cross(p2 - p0);
        if (!normal.allFinite() ||
            normal.squaredNorm() <= minimum_area_squared)
            return;
        if (normal.dot(outside_direction) < 0.F) std::swap(b, c);
        mesh.faces.emplace_back(a, b, c);
    };

    for (const auto& [negative_key, accumulated] : field) {
        if (!(accumulated.weight >= options.mesh_tsdf_min_weight) ||
            accumulated.signed_distance / accumulated.weight >= 0.0)
            continue;
        for (int dz = -1; dz <= 0; ++dz) {
            for (int dy = -1; dy <= 0; ++dy) {
                for (int dx = -1; dx <= 0; ++dx) {
                    const VoxelKey cell =
                        offset_key(negative_key, dx, dy, dz);
                    if (!processed_cells.emplace(cell).second) continue;

                    std::array<VoxelKey, 8> keys{};
                    std::array<float, 8> values{};
                    std::array<bool, 8> available{};
                    for (std::size_t corner = 0; corner < k_corners.size();
                         ++corner) {
                        keys[corner] = offset_key(
                            cell, k_corners[corner][0], k_corners[corner][1],
                            k_corners[corner][2]);
                        available[corner] =
                            value_at(keys[corner], values[corner]);
                    }

                    for (const auto& tetrahedron : k_tetrahedra) {
                        unsigned positive = 0;
                        unsigned negative = 0;
                        for (const int corner : tetrahedron) {
                            if (!available[static_cast<std::size_t>(corner)])
                                continue;
                            if (values[static_cast<std::size_t>(corner)] >= 0.F)
                                ++positive;
                            else
                                ++negative;
                        }
                        // Missing samples are tolerated only when this tetra
                        // already contains observed support on both sides of
                        // zero. This seals small inter-view gaps without
                        // inventing a second surface at the back of the TSDF
                        // truncation band.
                        if (positive == 0 || negative == 0)
                            continue;
                        for (const int corner : tetrahedron)
                            if (!available[static_cast<std::size_t>(corner)])
                                values[static_cast<std::size_t>(corner)] = 1.F;

                        Vec3f outside = Vec3f::Zero();
                        Vec3f inside = Vec3f::Zero();
                        unsigned outside_count = 0;
                        unsigned inside_count = 0;
                        for (const int corner : tetrahedron) {
                            const Vec3f position = position_of(
                                keys[static_cast<std::size_t>(corner)],
                                voxel_size);
                            if (values[static_cast<std::size_t>(corner)] >= 0.F) {
                                outside += position;
                                ++outside_count;
                            } else {
                                inside += position;
                                ++inside_count;
                            }
                        }
                        Vec3f direction =
                            outside / static_cast<float>(outside_count) -
                            inside / static_cast<float>(inside_count);
                        if (direction.squaredNorm() <= 1e-12F) continue;
                        direction.normalize();

                        std::vector<Crossing> crossings;
                        crossings.reserve(4);
                        for (const auto& edge : k_tetra_edges) {
                            const int ca = tetrahedron[edge[0]];
                            const int cb = tetrahedron[edge[1]];
                            const float va = values[static_cast<std::size_t>(ca)];
                            const float vb = values[static_cast<std::size_t>(cb)];
                            if ((va >= 0.F) == (vb >= 0.F)) continue;
                            const int vertex = vertex_on_edge(
                                keys[static_cast<std::size_t>(ca)],
                                keys[static_cast<std::size_t>(cb)], va, vb);
                            if (std::any_of(
                                    crossings.begin(), crossings.end(),
                                    [vertex](const Crossing& crossing) {
                                        return crossing.vertex == vertex;
                                    }))
                                continue;
                            crossings.push_back({
                                vertex,
                                mesh.vertices[static_cast<std::size_t>(vertex)]});
                        }
                        if (crossings.size() < 3) continue;

                        Vec3f centroid = Vec3f::Zero();
                        for (const Crossing& crossing : crossings)
                            centroid += crossing.position;
                        centroid /= static_cast<float>(crossings.size());
                        Vec3f basis = crossings.front().position - centroid;
                        basis -= direction * basis.dot(direction);
                        if (basis.squaredNorm() <= 1e-12F)
                            basis = direction.unitOrthogonal();
                        else
                            basis.normalize();
                        const Vec3f tangent = direction.cross(basis).normalized();
                        std::sort(
                            crossings.begin(), crossings.end(),
                            [&](const Crossing& a, const Crossing& b) {
                                const Vec3f ra = a.position - centroid;
                                const Vec3f rb = b.position - centroid;
                                return std::atan2(
                                           ra.dot(tangent), ra.dot(basis)) <
                                    std::atan2(rb.dot(tangent), rb.dot(basis));
                            });
                        for (std::size_t triangle = 1;
                             triangle + 1 < crossings.size(); ++triangle) {
                            append_triangle(
                                crossings[0].vertex,
                                crossings[triangle].vertex,
                                crossings[triangle + 1].vertex, direction);
                        }
                    }
                }
            }
        }
    }

    core::Logger::instance().info(
        "mvs mesh TSDF extract: cells=", processed_cells.size(),
        " vertices=", mesh.vertices.size(), " faces=", mesh.faces.size());
    if (mesh.faces.empty()) {
        stage.finish();
        return false;
    }
    scene.mesh = std::move(mesh);
    stage.finish();
    return true;
}

}  // namespace aetherscan::mvs::detail
