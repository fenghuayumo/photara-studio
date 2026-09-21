#include "mvs/internal.hpp"

#include "core/logging.hpp"
#include "io/image.hpp"
#include "marching_cubes_const.hpp"
#include "parallel/thread_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace photara::mvs::detail {
namespace {

// Match Open3D ScalableTSDFVolume, which is the backend used by pygsplat's
// gs2mesh.py: 16^3 sparse volume units allocated around sampled depth points.
constexpr int k_block_resolution = 16;
constexpr int k_block_voxel_count =
    k_block_resolution * k_block_resolution * k_block_resolution;
constexpr std::size_t k_positive_block_count = 8;
constexpr std::size_t k_cell_mask_words =
    (k_block_voxel_count + 63) / 64;
constexpr std::size_t k_edges_per_block =
    static_cast<std::size_t>(k_block_voxel_count) * 3;
constexpr std::size_t k_edge_mask_words =
    (k_edges_per_block + 63) / 64;
constexpr std::size_t k_invalid_block =
    std::numeric_limits<std::size_t>::max();
static_assert(
    k_edges_per_block <=
    static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()));

struct GridKey {
    std::int64_t x{};
    std::int64_t y{};
    std::int64_t z{};

    [[nodiscard]] bool operator==(const GridKey&) const noexcept = default;
};

[[nodiscard]] bool key_less(const GridKey& a, const GridKey& b) noexcept {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}

struct GridHash {
    [[nodiscard]] std::size_t operator()(const GridKey& key) const noexcept {
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

struct TsdfVoxel {
    float value{};
    float weight{};
};

struct VolumeBlock {
    std::array<TsdfVoxel, k_block_voxel_count> voxels{};
};

using Volume =
    std::unordered_map<GridKey, std::unique_ptr<VolumeBlock>, GridHash>;

struct TouchedBlock {
    GridKey key;
    VolumeBlock* block{};
};

[[nodiscard]] int voxel_index(const int x, const int y, const int z) noexcept {
    return (x * k_block_resolution + y) * k_block_resolution + z;
}

[[nodiscard]] GridKey floor_key(
    const Vec3f& position, const float inverse_length) noexcept {
    return {
        static_cast<std::int64_t>(std::floor(
            static_cast<double>(position.x() * inverse_length))),
        static_cast<std::int64_t>(std::floor(
            static_cast<double>(position.y() * inverse_length))),
        static_cast<std::int64_t>(std::floor(
            static_cast<double>(position.z() * inverse_length)))};
}

[[nodiscard]] GridKey global_key(
    const GridKey& block, const int x, const int y, const int z) noexcept {
    return {
        block.x * k_block_resolution + x,
        block.y * k_block_resolution + y,
        block.z * k_block_resolution + z};
}

[[nodiscard]] std::int64_t floor_div(
    const std::int64_t value, const std::int64_t divisor) noexcept {
    std::int64_t quotient = value / divisor;
    const std::int64_t remainder = value % divisor;
    if (remainder < 0) --quotient;
    return quotient;
}

[[nodiscard]] Vec3f position_of(
    const GridKey& key, const float voxel_size) noexcept {
    // UniformTSDFVolume stores samples at voxel centers, not grid corners.
    return Vec3f{
        (static_cast<float>(key.x) + 0.5F) * voxel_size,
        (static_cast<float>(key.y) + 0.5F) * voxel_size,
        (static_cast<float>(key.z) + 0.5F) * voxel_size};
}

[[nodiscard]] float estimate_voxel_size(const MvsScene& scene) {
    std::vector<float> footprints;
    footprints.reserve(200'000);
    for (const MvsView& view : scene.views) {
        const DepthMap& map = view.depth_map;
        if (map.depth.size() != map.size() || !(view.fx > 0.F) ||
            !(view.fy > 0.F))
            continue;
        constexpr unsigned stride = 16;
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

[[nodiscard]] VolumeBlock* open_block(
    Volume& volume, const GridKey& key) {
    auto [entry, inserted] = volume.try_emplace(key);
    if (inserted) entry->second = std::make_unique<VolumeBlock>();
    return entry->second.get();
}

[[nodiscard]] std::vector<TouchedBlock> allocate_view_blocks(
    const MvsView& view, const OrientedBoundingBox& subject_bounds,
    const float block_length, const float truncation,
    const unsigned depth_sampling_stride, Volume& volume,
    std::size_t& valid_depth_samples) {
    std::unordered_set<GridKey, GridHash> touched;
    const DepthMap& map = view.depth_map;
    if (map.depth.size() != map.size()) return {};

    const float inverse_block_length = 1.F / block_length;
    const Vec3f truncation_vector = Vec3f::Constant(truncation);
    for (unsigned y = 0; y < map.height; y += depth_sampling_stride) {
        for (unsigned x = 0; x < map.width; x += depth_sampling_stride) {
            const float depth = map.depth[map.index(x, y)];
            if (!(depth > 0.F) || !std::isfinite(depth)) continue;
            ++valid_depth_samples;
            const Vec3f camera_point = view.unproject(
                static_cast<float>(x), static_cast<float>(y), depth);
            const Vec3f world_point = view.pose
                .transform_camera_to_world(camera_point.cast<double>())
                .cast<float>();
            if (!world_point.allFinite() ||
                (subject_bounds.valid &&
                 !subject_bounds.contains(world_point, truncation)))
                continue;

            const GridKey minimum = floor_key(
                world_point - truncation_vector, inverse_block_length);
            const GridKey maximum = floor_key(
                world_point + truncation_vector, inverse_block_length);
            for (std::int64_t bx = minimum.x; bx <= maximum.x; ++bx)
                for (std::int64_t by = minimum.y; by <= maximum.y; ++by)
                    for (std::int64_t bz = minimum.z; bz <= maximum.z; ++bz)
                        touched.emplace(GridKey{bx, by, bz});
        }
    }

    std::vector<TouchedBlock> blocks;
    blocks.reserve(touched.size());
    for (const GridKey& key : touched)
        blocks.push_back({key, open_block(volume, key)});
    return blocks;
}

[[nodiscard]] std::uint64_t integrate_view(
    const MvsView& view, const OrientedBoundingBox& subject_bounds,
    const float voxel_size, const float truncation,
    const std::vector<TouchedBlock>& blocks) {
    const DepthMap& map = view.depth_map;
    std::uint64_t integrated_voxels = 0;

#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 8) reduction(+ : integrated_voxels)
#endif
    for (std::int64_t block_index = 0;
         block_index < static_cast<std::int64_t>(blocks.size());
         ++block_index) {
        const TouchedBlock& touched =
            blocks[static_cast<std::size_t>(block_index)];
        VolumeBlock& block = *touched.block;
        for (int x = 0; x < k_block_resolution; ++x) {
            for (int y = 0; y < k_block_resolution; ++y) {
                for (int z = 0; z < k_block_resolution; ++z) {
                    const GridKey key = global_key(touched.key, x, y, z);
                    const Vec3f world = position_of(key, voxel_size);
                    if (subject_bounds.valid &&
                        !subject_bounds.contains(world))
                        continue;
                    const Vec3f camera = view.pose
                        .transform_world_to_camera(world.cast<double>())
                        .cast<float>();
                    if (!(camera.z() > 0.F) || !camera.allFinite()) continue;

                    // Open3D rounds projected coordinates by adding 0.5 and
                    // truncating to an integer pixel.
                    const float inverse_z = 1.F / camera.z();
                    const float uf = view.fx * camera.x() * inverse_z +
                                     view.cx + 0.5F;
                    const float vf = view.fy * camera.y() * inverse_z +
                                     view.cy + 0.5F;
                    if (!(uf >= 0.F && vf >= 0.F) ||
                        uf >= static_cast<float>(map.width) ||
                        vf >= static_cast<float>(map.height))
                        continue;
                    const int u = static_cast<int>(uf);
                    const int v = static_cast<int>(vf);
                    const float source_depth = map.depth[map.index(u, v)];
                    if (!(source_depth > 0.F) ||
                        !std::isfinite(source_depth))
                        continue;
                    // RGBDImage.create_from_color_and_depth receives a uint16
                    // millimetre image in gs2mesh.py.
                    const float depth = std::floor(source_depth * 1000.F) /
                        1000.F;
                    if (!(depth > 0.F)) continue;

                    const float px =
                        (static_cast<float>(u) - view.cx) / view.fx;
                    const float py =
                        (static_cast<float>(v) - view.cy) / view.fy;
                    const float ray_length =
                        std::sqrt(1.F + px * px + py * py);
                    const float sdf = (depth - camera.z()) * ray_length;
                    if (!(sdf > -truncation)) continue;

                    const float tsdf = std::min(1.F, sdf / truncation);
                    TsdfVoxel& voxel = block.voxels[static_cast<std::size_t>(
                        voxel_index(x, y, z))];
                    voxel.value =
                        (voxel.value * voxel.weight + tsdf) /
                        (voxel.weight + 1.F);
                    voxel.weight += 1.F;
                    ++integrated_voxels;
                }
            }
        }
    }
    return integrated_voxels;
}

[[nodiscard]] const TsdfVoxel* find_voxel(
    const Volume& volume, const GridKey& key) noexcept {
    const GridKey block_key{
        floor_div(key.x, k_block_resolution),
        floor_div(key.y, k_block_resolution),
        floor_div(key.z, k_block_resolution)};
    const auto found = volume.find(block_key);
    if (found == volume.end()) return nullptr;
    const int x = static_cast<int>(
        key.x - block_key.x * k_block_resolution);
    const int y = static_cast<int>(
        key.y - block_key.y * k_block_resolution);
    const int z = static_cast<int>(
        key.z - block_key.z * k_block_resolution);
    return &found->second->voxels[static_cast<std::size_t>(
        voxel_index(x, y, z))];
}

struct SupportClosingStats {
    std::uint64_t zero_weight_voxels{};
    std::uint64_t bilateral_candidates{};
    std::uint64_t filled_voxels{};
};

struct PendingSupportVoxel {
    TsdfVoxel* voxel{};
    float value{};
};

void export_tsdf_frames(
    const MvsScene& scene, const float voxel_size,
    const float truncation, const std::filesystem::path& directory) {
    if (directory.empty()) return;
    std::filesystem::create_directories(directory);
    std::ofstream manifest(directory / "frames.csv");
    if (!manifest)
        throw std::runtime_error(
            "failed to create TSDF frame manifest: " +
            (directory / "frames.csv").string());
    manifest << std::setprecision(17)
             << "file,width,height,fx,fy,cx,cy,"
                "r00,r01,r02,r10,r11,r12,r20,r21,r22,t0,t1,t2\n";

    std::uint64_t valid_pixels = 0;
    std::uint64_t clipped_pixels = 0;
    std::uint64_t bytes_written = 0;
    for (std::size_t view_index = 0;
         view_index < scene.views.size(); ++view_index) {
        const MvsView& view = scene.views[view_index];
        const DepthMap& map = view.depth_map;
        if (map.depth.size() != map.size())
            throw std::runtime_error(
                "TSDF frame export received an incomplete depth map");

        std::vector<std::uint16_t> depth_mm(map.size(), 0);
        for (std::size_t pixel = 0; pixel < map.size(); ++pixel) {
            const float depth = map.depth[pixel];
            if (!(depth > 0.F) || !std::isfinite(depth)) continue;
            const double millimetres =
                std::floor(static_cast<double>(depth) * 1000.0);
            if (!(millimetres >= 1.0) ||
                millimetres >
                    static_cast<double>(
                        std::numeric_limits<std::uint16_t>::max())) {
                ++clipped_pixels;
                continue;
            }
            depth_mm[pixel] =
                static_cast<std::uint16_t>(millimetres);
            ++valid_pixels;
        }

        std::ostringstream filename;
        filename << "depth_" << std::setfill('0') << std::setw(4)
                 << view_index << ".u16";
        const std::filesystem::path output = directory / filename.str();
        std::ofstream depth_file(output, std::ios::binary);
        if (!depth_file)
            throw std::runtime_error(
                "failed to create TSDF depth frame: " + output.string());
        const std::streamsize byte_count = static_cast<std::streamsize>(
            depth_mm.size() * sizeof(std::uint16_t));
        depth_file.write(
            reinterpret_cast<const char*>(depth_mm.data()), byte_count);
        if (!depth_file)
            throw std::runtime_error(
                "failed to write TSDF depth frame: " + output.string());
        bytes_written += static_cast<std::uint64_t>(byte_count);

        const auto& R = view.pose.R;
        const auto t = view.pose.translation();
        manifest << filename.str() << ',' << map.width << ','
                 << map.height << ',' << view.fx << ',' << view.fy
                 << ',' << view.cx << ',' << view.cy;
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                manifest << ',' << R(row, column);
        manifest << ',' << t.x() << ',' << t.y() << ',' << t.z()
                 << '\n';
    }
    if (!manifest)
        throw std::runtime_error(
            "failed to write TSDF frame manifest: " +
            (directory / "frames.csv").string());

    std::ofstream metadata(directory / "metadata.txt");
    if (!metadata)
        throw std::runtime_error(
            "failed to create TSDF frame metadata");
    metadata << std::setprecision(17)
             << "voxel_size=" << voxel_size << '\n'
             << "sdf_trunc=" << truncation << '\n'
             << "depth_scale=1000\n"
             << "depth_trunc=65.535\n"
             << "frame_count=" << scene.views.size() << '\n';
    core::Logger::instance().info(
        "mvs mesh TSDF frame export: frames=", scene.views.size(),
        " valid_pixels=", valid_pixels,
        " clipped_pixels=", clipped_pixels,
        " bytes=", bytes_written, " directory=", directory);
}

[[nodiscard]] SupportClosingStats close_internal_tsdf_support(
    Volume& volume, const OrientedBoundingBox& bounds,
    const float voxel_size, const float minimum_weight,
    const unsigned required_axes) {
    SupportClosingStats stats;
    if (required_axes == 0 || required_axes > 3) return stats;

    constexpr std::array<GridKey, 3> axis_offsets{{
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    std::vector<TouchedBlock> blocks;
    blocks.reserve(volume.size());
    for (auto& [key, block] : volume)
        blocks.push_back({key, block.get()});
    std::vector<std::vector<PendingSupportVoxel>> block_pending(
        blocks.size());
    std::vector<std::uint64_t> block_zero_counts(blocks.size(), 0);
    std::vector<std::uint64_t> block_candidate_counts(blocks.size(), 0);
    const float valid_weight = std::max(
        minimum_weight, std::numeric_limits<float>::epsilon());

    // Scan first and apply later: newly synthesized support must never seed
    // another fill in this pass.
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t block_index = 0;
         block_index < static_cast<std::int64_t>(blocks.size());
         ++block_index) {
        const TouchedBlock& touched =
            blocks[static_cast<std::size_t>(block_index)];
        std::vector<PendingSupportVoxel>& pending =
            block_pending[static_cast<std::size_t>(block_index)];
        std::uint64_t& zero_count =
            block_zero_counts[static_cast<std::size_t>(block_index)];
        std::uint64_t& candidate_count =
            block_candidate_counts[static_cast<std::size_t>(block_index)];
        for (int x = 0; x < k_block_resolution; ++x) {
            for (int y = 0; y < k_block_resolution; ++y) {
                for (int z = 0; z < k_block_resolution; ++z) {
                    TsdfVoxel& center =
                        touched.block->voxels[static_cast<std::size_t>(
                            voxel_index(x, y, z))];
                    if (center.weight != 0.F) continue;
                    ++zero_count;

                    const GridKey key =
                        global_key(touched.key, x, y, z);
                    if (bounds.valid &&
                        !bounds.contains(position_of(key, voxel_size)))
                        continue;

                    unsigned bilateral_axes = 0;
                    double value_sum = 0.0;
                    double axis_weight_sum = 0.0;
                    for (const GridKey& offset : axis_offsets) {
                        const GridKey negative{
                            key.x - offset.x, key.y - offset.y,
                            key.z - offset.z};
                        const GridKey positive{
                            key.x + offset.x, key.y + offset.y,
                            key.z + offset.z};
                        const TsdfVoxel* a = find_voxel(volume, negative);
                        const TsdfVoxel* b = find_voxel(volume, positive);
                        if (a == nullptr || b == nullptr ||
                            a->weight < valid_weight ||
                            b->weight < valid_weight)
                            continue;

                        ++bilateral_axes;
                        // Use the weaker side as the axis confidence so a
                        // heavily observed side cannot drag the fill across a
                        // weakly supported surface.
                        const double axis_weight =
                            std::min(a->weight, b->weight);
                        const double pair_value =
                            (static_cast<double>(a->value) * a->weight +
                             static_cast<double>(b->value) * b->weight) /
                            (static_cast<double>(a->weight) + b->weight);
                        value_sum += axis_weight * pair_value;
                        axis_weight_sum += axis_weight;
                    }
                    if (bilateral_axes < required_axes ||
                        !(axis_weight_sum > 0.0))
                        continue;

                    ++candidate_count;
                    const float value = static_cast<float>(
                        value_sum / axis_weight_sum);
                    if (!std::isfinite(value)) continue;
                    pending.push_back({
                        &center, std::clamp(value, -1.F, 1.F)});
                }
            }
        }
    }

    for (std::size_t block_index = 0;
         block_index < blocks.size(); ++block_index) {
        stats.zero_weight_voxels += block_zero_counts[block_index];
        stats.bilateral_candidates +=
            block_candidate_counts[block_index];
        for (const PendingSupportVoxel& fill :
             block_pending[block_index]) {
            fill.voxel->value = fill.value;
            fill.voxel->weight = 1.F;
            ++stats.filled_voxels;
        }
    }
    return stats;
}

[[nodiscard]] std::array<std::uint8_t, 3> support_color(
    const float value, const float brightness = 1.F) noexcept {
    const float t = std::clamp(value, 0.F, 1.F);
    const float scale = std::clamp(brightness, 0.F, 1.F);
    const float red = t <= 0.5F ? 1.F : 2.F * (1.F - t);
    const float green = t <= 0.5F ? 2.F * t : 1.F;
    return {
        static_cast<std::uint8_t>(std::lround(255.F * scale * red)),
        static_cast<std::uint8_t>(std::lround(255.F * scale * green)),
        0U};
}

[[nodiscard]] std::vector<std::size_t> diagnostic_view_indices(
    const std::size_t view_count) {
    if (view_count == 0) return {};
    std::vector<std::size_t> indices{0, view_count / 2, view_count - 1};
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

void save_tsdf_diagnostics(
    const MvsScene& scene, const Volume& volume, const float voxel_size,
    const float relative_depth_threshold,
    const std::filesystem::path& directory) {
    if (directory.empty() || scene.views.empty()) return;
    std::filesystem::create_directories(directory);
    const float consistency_threshold =
        std::max(relative_depth_threshold, 1e-4F);
    const float full_weight =
        static_cast<float>(std::clamp<std::size_t>(
            scene.views.size(), std::size_t{2}, std::size_t{8}));

    for (const std::size_t view_index :
         diagnostic_view_indices(scene.views.size())) {
        const MvsView& reference = scene.views[view_index];
        const DepthMap& reference_map = reference.depth_map;
        if (reference_map.depth.size() != reference_map.size() ||
            reference_map.width == 0 || reference_map.height == 0)
            continue;

        const std::size_t pixels = reference_map.size();
        io::RgbImage consistency_image{
            reference_map.width, reference_map.height,
            std::vector<std::uint8_t>(3 * pixels, 0U)};
        io::RgbImage weight_image{
            reference_map.width, reference_map.height,
            std::vector<std::uint8_t>(3 * pixels, 0U)};
        std::uint64_t valid_reference_pixels = 0;
        std::uint64_t compared_depths = 0;
        std::uint64_t consistent_depths = 0;
        std::uint64_t pixels_without_comparison = 0;
        std::uint64_t zero_weight_pixels = 0;
        double minimum_weight_sum = 0.0;

        for (std::uint32_t y = 0; y < reference_map.height; ++y) {
            for (std::uint32_t x = 0; x < reference_map.width; ++x) {
                const std::size_t pixel = reference_map.index(x, y);
                const float depth = reference_map.depth[pixel];
                if (!(depth > 0.F) || !std::isfinite(depth)) continue;
                ++valid_reference_pixels;
                const Vec3f camera_point = reference.unproject(
                    static_cast<float>(x), static_cast<float>(y), depth);
                const Vec3f world = reference.pose
                    .transform_camera_to_world(camera_point.cast<double>())
                    .cast<float>();
                if (!world.allFinite()) continue;

                unsigned compared = 0;
                unsigned consistent = 0;
                for (const NeighborScore& score : reference.neighbors) {
                    const std::size_t neighbor_index =
                        static_cast<std::size_t>(score.view_id);
                    if (neighbor_index >= scene.views.size() ||
                        neighbor_index == view_index)
                        continue;
                    const MvsView& neighbor = scene.views[neighbor_index];
                    const DepthMap& neighbor_map = neighbor.depth_map;
                    if (neighbor_map.depth.size() != neighbor_map.size())
                        continue;
                    const Vec3f projected = neighbor.pose
                        .transform_world_to_camera(world.cast<double>())
                        .cast<float>();
                    if (!(projected.z() > 0.F) || !projected.allFinite())
                        continue;
                    const float inverse_z = 1.F / projected.z();
                    const float uf = neighbor.fx * projected.x() * inverse_z +
                        neighbor.cx + 0.5F;
                    const float vf = neighbor.fy * projected.y() * inverse_z +
                        neighbor.cy + 0.5F;
                    if (!(uf >= 0.F && vf >= 0.F) ||
                        uf >= static_cast<float>(neighbor_map.width) ||
                        vf >= static_cast<float>(neighbor_map.height))
                        continue;
                    const int u = static_cast<int>(uf);
                    const int v = static_cast<int>(vf);
                    const float neighbor_depth =
                        neighbor_map.depth[neighbor_map.index(u, v)];
                    if (!(neighbor_depth > 0.F) ||
                        !std::isfinite(neighbor_depth))
                        continue;
                    const float tolerance = consistency_threshold *
                        std::max(neighbor_depth, projected.z());
                    const float delta = neighbor_depth - projected.z();
                    // A closer neighbor surface occludes this reference point;
                    // it is not evidence that the reference depth is wrong.
                    if (delta < -tolerance) continue;
                    ++compared;
                    if (std::abs(delta) <= tolerance) ++consistent;
                }

                const std::size_t color_offset = 3 * pixel;
                if (compared == 0) {
                    consistency_image.pixels[color_offset + 2] = 128U;
                    ++pixels_without_comparison;
                } else {
                    const float ratio =
                        static_cast<float>(consistent) /
                        static_cast<float>(compared);
                    const float brightness = std::min(
                        1.F, static_cast<float>(compared) / 4.F);
                    const auto color = support_color(ratio, brightness);
                    for (std::size_t channel = 0; channel < 3; ++channel)
                        consistency_image.pixels[color_offset + channel] =
                            color[channel];
                    compared_depths += compared;
                    consistent_depths += consistent;
                }

                const Vec3f lattice =
                    world / voxel_size - Vec3f::Constant(0.5F);
                const GridKey cell{
                    static_cast<std::int64_t>(std::floor(lattice.x())),
                    static_cast<std::int64_t>(std::floor(lattice.y())),
                    static_cast<std::int64_t>(std::floor(lattice.z()))};
                float minimum_weight =
                    std::numeric_limits<float>::infinity();
                for (int corner = 0; corner < 8; ++corner) {
                    const GridKey key{
                        cell.x + shift[corner].x(),
                        cell.y + shift[corner].y(),
                        cell.z + shift[corner].z()};
                    const TsdfVoxel* voxel = find_voxel(volume, key);
                    minimum_weight = std::min(
                        minimum_weight, voxel == nullptr ? 0.F : voxel->weight);
                }
                if (!std::isfinite(minimum_weight)) minimum_weight = 0.F;
                minimum_weight_sum += minimum_weight;
                if (!(minimum_weight > 0.F)) ++zero_weight_pixels;
                const auto weight_color =
                    support_color(minimum_weight / full_weight);
                for (std::size_t channel = 0; channel < 3; ++channel)
                    weight_image.pixels[color_offset + channel] =
                        weight_color[channel];
            }
        }

        const std::string suffix =
            "_view_" + std::to_string(view_index) + ".png";
        io::save_rgb_png(
            consistency_image,
            directory / ("tsdf_depth_consistency" + suffix));
        io::save_rgb_png(
            weight_image, directory / ("tsdf_weight" + suffix));
        core::Logger::instance().info(
            "TSDF diagnostics: view=", view_index,
            " valid_reference_pixels=", valid_reference_pixels,
            " compared_depths=", compared_depths,
            " consistent_depths=", consistent_depths,
            " consistency_ratio=",
            compared_depths > 0
                ? static_cast<double>(consistent_depths) /
                    static_cast<double>(compared_depths)
                : 0.0,
            " pixels_without_comparison=", pixels_without_comparison,
            " zero_min_corner_weight_pixels=", zero_weight_pixels,
            " average_min_corner_weight=",
            valid_reference_pixels > 0
                ? minimum_weight_sum /
                    static_cast<double>(valid_reference_pixels)
                : 0.0,
            " directory=", directory);
    }
}

struct MarchingBlock {
    GridKey key;
    const VolumeBlock* volume_block{};
    std::array<std::size_t, k_positive_block_count> positive_neighbors{};
    std::uint32_t supported_cells{};
    std::uint32_t active_cells{};
    std::uint32_t face_count{};
    std::size_t vertex_offset{};
    std::size_t face_offset{};
};

struct MarchingCubesStats {
    std::uint64_t scanned_cells{};
    std::uint64_t supported_cells{};
    std::uint64_t active_cells{};
    std::size_t scratch_bytes{};
    unsigned thread_count{};
};

struct LocalEdgeMasks {
    std::array<
        std::uint64_t,
        k_positive_block_count * k_edge_mask_words> words{};
};

struct OwnedEdge {
    std::size_t block{k_invalid_block};
    std::size_t local_edge{};
};

// A cell owned by a block only reaches that block or one of its seven
// positive neighbors. Encoding the three positive-boundary bits turns all
// corner/edge reads into direct array access instead of sparse-volume hashes.
[[nodiscard]] std::size_t positive_neighbor_slot(
    const int x, const int y, const int z) noexcept {
    return (x >= k_block_resolution ? 4U : 0U) |
        (y >= k_block_resolution ? 2U : 0U) |
        (z >= k_block_resolution ? 1U : 0U);
}

[[nodiscard]] const TsdfVoxel* marching_voxel(
    const std::vector<MarchingBlock>& blocks, const MarchingBlock& block,
    const int x, const int y, const int z) noexcept {
    const std::size_t owner = block.positive_neighbors[
        positive_neighbor_slot(x, y, z)];
    if (owner == k_invalid_block) return nullptr;
    return &blocks[owner].volume_block->voxels[static_cast<std::size_t>(
        voxel_index(
            x % k_block_resolution,
            y % k_block_resolution,
            z % k_block_resolution))];
}

[[nodiscard]] OwnedEdge marching_owned_edge(
    const MarchingBlock& block, const int x, const int y, const int z,
    const int edge) noexcept {
    // edge_shift always selects the lower grid coordinate on the edge axis.
    // The block containing that coordinate is therefore the unique global
    // owner, including edges shared by cells in adjacent sparse blocks.
    const int edge_x = x + edge_shift[edge].x();
    const int edge_y = y + edge_shift[edge].y();
    const int edge_z = z + edge_shift[edge].z();
    const std::size_t owner = block.positive_neighbors[
        positive_neighbor_slot(edge_x, edge_y, edge_z)];
    return {
        owner,
        static_cast<std::size_t>(
            voxel_index(
                edge_x % k_block_resolution,
                edge_y % k_block_resolution,
                edge_z % k_block_resolution)) *
                3U +
            static_cast<std::size_t>(edge_shift[edge].w())};
}

[[nodiscard]] std::vector<MarchingBlock> build_marching_blocks(
    const Volume& volume) {
    std::vector<MarchingBlock> blocks;
    blocks.reserve(volume.size());
    for (const auto& [key, block] : volume) {
        MarchingBlock marching_block;
        marching_block.key = key;
        marching_block.volume_block = block.get();
        marching_block.positive_neighbors.fill(k_invalid_block);
        blocks.push_back(marching_block);
    }
    std::sort(
        blocks.begin(), blocks.end(),
        [](const MarchingBlock& a, const MarchingBlock& b) {
            return key_less(a.key, b.key);
        });

    // Sorted blocks give stable output ranges. This temporary hash is only
    // used once to resolve the eight direct positive-neighbor indices.
    std::unordered_map<GridKey, std::size_t, GridHash> block_indices;
    block_indices.reserve(blocks.size());
    for (std::size_t index = 0; index < blocks.size(); ++index)
        block_indices.emplace(blocks[index].key, index);
    for (MarchingBlock& block : blocks) {
        for (int dx = 0; dx <= 1; ++dx) {
            for (int dy = 0; dy <= 1; ++dy) {
                for (int dz = 0; dz <= 1; ++dz) {
                    const GridKey neighbor{
                        block.key.x + dx,
                        block.key.y + dy,
                        block.key.z + dz};
                    const auto found = block_indices.find(neighbor);
                    if (found != block_indices.end()) {
                        block.positive_neighbors[
                            positive_neighbor_slot(
                                dx * k_block_resolution,
                                dy * k_block_resolution,
                                dz * k_block_resolution)] = found->second;
                    }
                }
            }
        }
    }
    return blocks;
}

[[nodiscard]] std::uint32_t cube_face_count(const int cube_index) noexcept {
    std::uint32_t count = 0;
    for (int triangle = 0;
         tri_table[cube_index][triangle] != -1;
         triangle += 3)
        ++count;
    return count;
}

[[nodiscard]] int marching_cube_index(
    const std::vector<MarchingBlock>& blocks, const MarchingBlock& block,
    const int x, const int y, const int z,
    const float minimum_weight) noexcept {
    int cube_index = 0;
    for (int corner = 0; corner < 8; ++corner) {
        const TsdfVoxel* voxel = marching_voxel(
            blocks, block,
            x + shift[corner].x(),
            y + shift[corner].y(),
            z + shift[corner].z());
        if (voxel == nullptr || voxel->weight < minimum_weight) return -1;
        if (voxel->value < 0.F) cube_index |= 1 << corner;
    }
    return cube_index;
}

[[nodiscard]] std::size_t edge_vertex_index(
    const std::vector<MarchingBlock>& blocks,
    const std::vector<std::uint64_t>& edge_masks,
    const std::vector<std::uint16_t>& edge_prefix,
    const MarchingBlock& block, const int x, const int y, const int z,
    const int edge) {
    const OwnedEdge owned = marching_owned_edge(
        block, x, y, z, edge);
    if (owned.block == k_invalid_block)
        throw std::runtime_error("Marching Cubes edge owner is missing");
    const std::size_t word = owned.local_edge / 64U;
    const unsigned bit = static_cast<unsigned>(owned.local_edge % 64U);
    const std::uint64_t mask =
        edge_masks[owned.block * k_edge_mask_words + word];
    if ((mask & (std::uint64_t{1} << bit)) == 0)
        throw std::runtime_error(
            "Marching Cubes referenced an unallocated edge");
    const std::uint64_t before =
        bit == 0U ? 0U : mask & ((std::uint64_t{1} << bit) - 1U);
    const std::size_t rank =
        edge_prefix[
            owned.block * (k_edge_mask_words + 1U) + word] +
        static_cast<std::size_t>(std::popcount(before));
    return blocks[owned.block].vertex_offset + rank;
}

[[nodiscard]] MarchingCubesStats extract_marching_cubes(
    const Volume& volume, const float voxel_size,
    const float minimum_weight, const unsigned requested_thread_count,
    Mesh& mesh) {
    core::StageScope stage("mvs.mesh.tsdf.marching_cubes");
    std::vector<MarchingBlock> blocks = build_marching_blocks(volume);
    const unsigned thread_count = std::max(
        1U, parallel::resolve_thread_count(requested_thread_count));

    std::vector<std::uint64_t> active_cell_masks(
        blocks.size() * k_cell_mask_words);
    std::vector<std::uint64_t> edge_masks(
        blocks.size() * k_edge_mask_words);
    std::vector<LocalEdgeMasks> worker_edge_masks(thread_count);

    // Pass 1: classify cells and count faces. Edge ownership is accumulated
    // into per-worker masks and merged one word at a time, avoiding a
    // contended global edge hash and per-edge atomics.
    parallel::parallel_for(
        blocks.size(), thread_count,
        [&](const std::size_t block_index, const unsigned worker_id) {
            MarchingBlock& block = blocks[block_index];
            LocalEdgeMasks& local = worker_edge_masks[worker_id];
            local.words.fill(0);
            std::uint32_t supported_cells = 0;
            std::uint32_t active_cells = 0;
            std::uint32_t face_count = 0;
            for (int x = 0; x < k_block_resolution; ++x) {
                for (int y = 0; y < k_block_resolution; ++y) {
                    for (int z = 0; z < k_block_resolution; ++z) {
                        const int cube_index = marching_cube_index(
                            blocks, block, x, y, z, minimum_weight);
                        if (cube_index < 0) continue;
                        ++supported_cells;
                        if (cube_index == 0 || cube_index == 255) continue;
                        ++active_cells;
                        const std::size_t cell = static_cast<std::size_t>(
                            voxel_index(x, y, z));
                        active_cell_masks[
                            block_index * k_cell_mask_words + cell / 64U] |=
                            std::uint64_t{1} << (cell % 64U);
                        face_count += cube_face_count(cube_index);

                        const int used_edges = edge_table[cube_index];
                        for (int edge = 0; edge < 12; ++edge) {
                            if ((used_edges & (1 << edge)) == 0) continue;
                            const OwnedEdge owned = marching_owned_edge(
                                block, x, y, z, edge);
                            if (owned.block == k_invalid_block)
                                throw std::runtime_error(
                                    "Marching Cubes active cell has no edge "
                                    "owner");
                            const std::size_t owner_slot =
                                positive_neighbor_slot(
                                    x + edge_shift[edge].x(),
                                    y + edge_shift[edge].y(),
                                    z + edge_shift[edge].z());
                            local.words[
                                owner_slot * k_edge_mask_words +
                                owned.local_edge / 64U] |=
                                std::uint64_t{1} <<
                                (owned.local_edge % 64U);
                        }
                    }
                }
            }
            block.supported_cells = supported_cells;
            block.active_cells = active_cells;
            block.face_count = face_count;

            for (std::size_t slot = 0;
                 slot < k_positive_block_count; ++slot) {
                const std::size_t owner = block.positive_neighbors[slot];
                if (owner == k_invalid_block) continue;
                for (std::size_t word = 0;
                     word < k_edge_mask_words; ++word) {
                    const std::uint64_t bits =
                        local.words[slot * k_edge_mask_words + word];
                    if (bits == 0) continue;
                    std::atomic_ref<std::uint64_t>(
                        edge_masks[owner * k_edge_mask_words + word])
                        .fetch_or(bits, std::memory_order_relaxed);
                }
            }
        });

    std::vector<std::uint16_t> edge_prefix(
        blocks.size() * (k_edge_mask_words + 1U));
    std::size_t total_vertices = 0;
    std::size_t total_faces = 0;
    MarchingCubesStats stats;
    stats.thread_count = static_cast<unsigned>(std::min<std::size_t>(
        thread_count, std::max<std::size_t>(1U, blocks.size())));
    stats.scanned_cells =
        static_cast<std::uint64_t>(blocks.size()) *
        static_cast<std::uint64_t>(k_block_voxel_count);
    for (std::size_t block_index = 0;
         block_index < blocks.size(); ++block_index) {
        MarchingBlock& block = blocks[block_index];
        block.vertex_offset = total_vertices;
        block.face_offset = total_faces;
        std::uint16_t prefix = 0;
        for (std::size_t word = 0; word < k_edge_mask_words; ++word) {
            edge_prefix[
                block_index * (k_edge_mask_words + 1U) + word] = prefix;
            prefix = static_cast<std::uint16_t>(
                prefix +
                std::popcount(
                    edge_masks[block_index * k_edge_mask_words + word]));
        }
        edge_prefix[
            block_index * (k_edge_mask_words + 1U) +
            k_edge_mask_words] = prefix;
        total_vertices += prefix;
        total_faces += block.face_count;
        stats.supported_cells += block.supported_cells;
        stats.active_cells += block.active_cells;
    }
    if (total_vertices >
        static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error(
            "Marching Cubes vertex count exceeds 32-bit mesh indices");

    mesh = {};
    mesh.vertices.resize(total_vertices);
    mesh.faces.resize(total_faces);

    // Pass 2: each block writes its prefix-assigned slices. Faces only revisit
    // active cells; vertex indices come from the canonical owner bit rank.
    parallel::parallel_for(
        blocks.size(), thread_count,
        [&](const std::size_t block_index) {
            const MarchingBlock& block = blocks[block_index];
            const std::size_t prefix_base =
                block_index * (k_edge_mask_words + 1U);
            for (std::size_t word = 0; word < k_edge_mask_words; ++word) {
                std::uint64_t bits =
                    edge_masks[block_index * k_edge_mask_words + word];
                std::size_t output =
                    block.vertex_offset + edge_prefix[prefix_base + word];
                while (bits != 0) {
                    const unsigned bit =
                        static_cast<unsigned>(std::countr_zero(bits));
                    const std::size_t local_edge = word * 64U + bit;
                    const std::size_t local_voxel = local_edge / 3U;
                    const int axis = static_cast<int>(local_edge % 3U);
                    const int x = static_cast<int>(
                        local_voxel /
                        (k_block_resolution * k_block_resolution));
                    const int y = static_cast<int>(
                        (local_voxel / k_block_resolution) %
                        k_block_resolution);
                    const int z = static_cast<int>(
                        local_voxel % k_block_resolution);
                    const TsdfVoxel& a =
                        block.volume_block->voxels[local_voxel];
                    const TsdfVoxel* b = marching_voxel(
                        blocks, block,
                        x + (axis == 0 ? 1 : 0),
                        y + (axis == 1 ? 1 : 0),
                        z + (axis == 2 ? 1 : 0));
                    if (b == nullptr)
                        throw std::runtime_error(
                            "Marching Cubes edge endpoint is missing");
                    const float aa = std::abs(a.value);
                    const float ab = std::abs(b->value);
                    const float t =
                        aa + ab > 1e-12F ? aa / (aa + ab) : 0.5F;
                    const GridKey start = global_key(block.key, x, y, z);
                    GridKey end = start;
                    if (axis == 0) ++end.x;
                    else if (axis == 1) ++end.y;
                    else ++end.z;
                    mesh.vertices[output++] =
                        position_of(start, voxel_size) * (1.F - t) +
                        position_of(end, voxel_size) * t;
                    bits &= bits - 1U;
                }
            }

            std::size_t face_output = block.face_offset;
            for (std::size_t word = 0; word < k_cell_mask_words; ++word) {
                std::uint64_t bits =
                    active_cell_masks[
                        block_index * k_cell_mask_words + word];
                while (bits != 0) {
                    const unsigned bit =
                        static_cast<unsigned>(std::countr_zero(bits));
                    const std::size_t cell = word * 64U + bit;
                    const int x = static_cast<int>(
                        cell /
                        (k_block_resolution * k_block_resolution));
                    const int y = static_cast<int>(
                        (cell / k_block_resolution) %
                        k_block_resolution);
                    const int z = static_cast<int>(
                        cell % k_block_resolution);
                    const int cube_index = marching_cube_index(
                        blocks, block, x, y, z, minimum_weight);
                    if (cube_index <= 0 || cube_index >= 255)
                        throw std::runtime_error(
                            "Marching Cubes active-cell mask is inconsistent");
                    for (int triangle = 0;
                         tri_table[cube_index][triangle] != -1;
                         triangle += 3) {
                        const std::size_t a = edge_vertex_index(
                            blocks, edge_masks, edge_prefix, block,
                            x, y, z,
                            tri_table[cube_index][triangle]);
                        const std::size_t b = edge_vertex_index(
                            blocks, edge_masks, edge_prefix, block,
                            x, y, z,
                            tri_table[cube_index][triangle + 2]);
                        const std::size_t c = edge_vertex_index(
                            blocks, edge_masks, edge_prefix, block,
                            x, y, z,
                            tri_table[cube_index][triangle + 1]);
                        mesh.faces[face_output++] = Eigen::Vector3i{
                            static_cast<int>(a),
                            static_cast<int>(b),
                            static_cast<int>(c)};
                    }
                    bits &= bits - 1U;
                }
            }
            if (face_output != block.face_offset + block.face_count)
                throw std::runtime_error(
                    "Marching Cubes face prefix is inconsistent");
        });

    stats.scratch_bytes =
        active_cell_masks.size() * sizeof(active_cell_masks.front()) +
        edge_masks.size() * sizeof(edge_masks.front()) +
        edge_prefix.size() * sizeof(edge_prefix.front()) +
        worker_edge_masks.size() * sizeof(worker_edge_masks.front());
    stage.finish();
    return stats;
}

}  // namespace

bool reconstruct_mesh_tsdf(MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.mesh.tsdf");
    const OrientedBoundingBox& bounds = scene.subject_bounds;
    const float inferred_voxel = estimate_voxel_size(scene);
    const float voxel_size = options.mesh_tsdf_voxel_size > 0.F
        ? options.mesh_tsdf_voxel_size
        : inferred_voxel * options.mesh_tsdf_voxel_scale;
    if (!(voxel_size > 0.F) || !std::isfinite(voxel_size)) {
        core::Logger::instance().warning(
            "mvs mesh TSDF: unable to infer a valid voxel size");
        stage.finish();
        return false;
    }

    const float truncation = voxel_size *
        std::max(options.mesh_tsdf_truncation_voxels, 1.F);
    const float block_length = voxel_size * k_block_resolution;
    Volume volume;
    volume.reserve(4096);
    std::size_t valid_depth_samples = 0;
    std::uint64_t integrated_voxels = 0;
    std::size_t maximum_view_blocks = 0;
    const unsigned depth_sampling_stride =
        std::max(options.mesh_tsdf_pixel_step, 1U);
    for (const MvsView& view : scene.views) {
        const std::vector<TouchedBlock> blocks = allocate_view_blocks(
            view, bounds, block_length, truncation, depth_sampling_stride, volume,
            valid_depth_samples);
        maximum_view_blocks = std::max(maximum_view_blocks, blocks.size());
        integrated_voxels += integrate_view(
            view, bounds, voxel_size, truncation, blocks);
    }

    core::Logger::instance().info(
        "mvs mesh TSDF integrate (gs2mesh/Open3D compatible): voxel=",
        voxel_size, " inferred_voxel=", inferred_voxel,
        " voxel_scale=", options.mesh_tsdf_voxel_scale,
        " truncation=", truncation,
        " depth_sampling_stride=", depth_sampling_stride,
        " bounds_enabled=", bounds.valid,
        " depth_samples=", valid_depth_samples,
        " volume_blocks=", volume.size(),
        " max_view_blocks=", maximum_view_blocks,
        " voxel_updates=", integrated_voxels);
    if (volume.empty()) {
        stage.finish();
        return false;
    }
    export_tsdf_frames(
        scene, voxel_size, truncation,
        options.mesh_tsdf_frame_export_dir);
    const SupportClosingStats closing = close_internal_tsdf_support(
        volume, bounds, voxel_size, options.mesh_tsdf_min_weight,
        options.mesh_tsdf_support_closing_axes);
    core::Logger::instance().info(
        "mvs mesh TSDF support closing: required_axes=",
        options.mesh_tsdf_support_closing_axes,
        " zero_weight_voxels=", closing.zero_weight_voxels,
        " bilateral_candidates=", closing.bilateral_candidates,
        " filled_voxels=", closing.filled_voxels,
        " synthetic_weight=1 single_pass=true");
    save_tsdf_diagnostics(
        scene, volume, voxel_size, options.depth_diff_threshold,
        options.mesh_tsdf_diagnostics_dir);

    Mesh mesh;
    const MarchingCubesStats marching = extract_marching_cubes(
        volume, voxel_size, options.mesh_tsdf_min_weight,
        options.thread_count, mesh);

    core::Logger::instance().info(
        "mvs mesh TSDF Marching Cubes: scanned_cells=",
        marching.scanned_cells,
        " supported_cells=", marching.supported_cells,
        " active_cells=", marching.active_cells,
        " vertices=", mesh.vertices.size(), " faces=", mesh.faces.size(),
        " threads=", marching.thread_count,
        " scratch_mib=",
        static_cast<double>(marching.scratch_bytes) / (1024.0 * 1024.0),
        " deterministic_grid_edges=true");
    if (mesh.faces.empty()) {
        stage.finish();
        return false;
    }
    scene.mesh = std::move(mesh);
    stage.finish();
    return true;
}

}  // namespace photara::mvs::detail
