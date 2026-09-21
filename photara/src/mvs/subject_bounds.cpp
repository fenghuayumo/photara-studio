#include "mvs/internal.hpp"

#include "core/logging.hpp"
#include "parallel/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace photara::mvs::detail {
namespace {

struct Cell {
    int x{}, y{}, z{};
    bool operator==(const Cell&) const = default;
};

struct CellHash {
    std::size_t operator()(const Cell& cell) const noexcept {
        std::size_t hash =
            static_cast<std::uint32_t>(cell.x) * 73856093U;
        hash ^= static_cast<std::uint32_t>(cell.y) * 19349663U;
        hash ^= static_cast<std::uint32_t>(cell.z) * 83492791U;
        return hash;
    }
};

}  // namespace

bool estimate_subject_bounds(
    const std::vector<Vec3f>& points,
    OrientedBoundingBox& result,
    const unsigned thread_count, const float padding_scale) {
    core::StageScope stage("sfm.subject_bounds");
    result = {};
    if (points.empty()) {
        stage.finish();
        return false;
    }

    Vec3f raw_min = points.front();
    Vec3f raw_max = points.front();
    for (const Vec3f& point : points) {
        raw_min = raw_min.cwiseMin(point);
        raw_max = raw_max.cwiseMax(point);
    }
    const float diagonal = (raw_max - raw_min).norm();
    if (!(diagonal > 1e-8F) || !std::isfinite(diagonal)) {
        stage.finish();
        return false;
    }

    // Match gs2mesh.py's radius-outlier removal before computing its padded
    // sparse-reconstruction AABB. This bound is intentionally conservative:
    // it limits allocation and extraction, but never deletes SfM/GGGS points.
    const float radius = std::max(0.1F, 0.05F * diagonal);
    const float inverse_radius = 1.F / radius;
    const auto cell_of = [&](const Vec3f& point) {
        return Cell{
            static_cast<int>(std::floor(point.x() * inverse_radius)),
            static_cast<int>(std::floor(point.y() * inverse_radius)),
            static_cast<int>(std::floor(point.z() * inverse_radius))};
    };
    std::unordered_map<Cell, std::vector<std::size_t>, CellHash> grid;
    grid.reserve(points.size() / 8U + 1U);
    for (std::size_t index = 0; index < points.size(); ++index)
        grid[cell_of(points[index])].push_back(index);

    const std::size_t required_neighbors =
        std::min<std::size_t>(10U, points.size());
    const float radius_squared = radius * radius;
    std::vector<std::uint8_t> retained(points.size(), 0);
    const auto& lookup = grid;
    parallel::parallel_for(
        points.size(), parallel::resolve_thread_count(thread_count),
        [&](const std::size_t index) {
            const Cell center = cell_of(points[index]);
            std::size_t neighbors = 0;
            for (int dz = -1; dz <= 1 && neighbors < required_neighbors; ++dz)
                for (int dy = -1;
                     dy <= 1 && neighbors < required_neighbors; ++dy)
                    for (int dx = -1;
                         dx <= 1 && neighbors < required_neighbors; ++dx) {
                        const auto found = lookup.find(
                            Cell{
                                center.x + dx, center.y + dy,
                                center.z + dz});
                        if (found == lookup.end()) continue;
                        for (const std::size_t candidate : found->second) {
                            if ((points[candidate] - points[index])
                                        .squaredNorm() <= radius_squared &&
                                ++neighbors >= required_neighbors)
                                break;
                        }
                    }
            retained[index] =
                neighbors >= required_neighbors ? std::uint8_t{1}
                                                : std::uint8_t{0};
        });

    std::size_t retained_count = static_cast<std::size_t>(std::count(
        retained.begin(), retained.end(), std::uint8_t{1}));
    const bool use_filtered = retained_count >= 32U;
    if (!use_filtered) retained_count = points.size();
    Vec3f minimum =
        Vec3f::Constant(std::numeric_limits<float>::infinity());
    Vec3f maximum =
        Vec3f::Constant(-std::numeric_limits<float>::infinity());
    for (std::size_t index = 0; index < points.size(); ++index) {
        if (use_filtered && retained[index] == 0) continue;
        minimum = minimum.cwiseMin(points[index]);
        maximum = maximum.cwiseMax(points[index]);
    }

    const Vec3f center = 0.5F * (minimum + maximum);
    Vec3f half_extent = 0.5F * (maximum - minimum);
    const float minimum_padding = std::max(1e-3F, 0.01F * diagonal);
    const float applied_padding = std::max(padding_scale, 1.F);
    half_extent =
        (half_extent * applied_padding)
            .cwiseMax(Vec3f::Constant(minimum_padding));

    result.valid = center.allFinite() && half_extent.allFinite() &&
        (half_extent.array() > 0.F).all();
    result.center = center;
    result.axes = Mat3f::Identity();
    result.half_extent = half_extent;
    core::Logger::instance().info(
        "SfM SubjectBounds: retained=", retained_count, '/', points.size(),
        " radius=", radius, " padding_scale=", applied_padding,
        " min=", (center - half_extent).transpose(),
        " max=", (center + half_extent).transpose());
    stage.finish();
    return result.valid;
}

bool estimate_subject_bounds(
    const std::vector<SparsePoint>& sparse_points,
    OrientedBoundingBox& result,
    const unsigned thread_count, const float padding_scale) {
    std::vector<Vec3f> points;
    points.reserve(sparse_points.size());
    for (const SparsePoint& point : sparse_points)
        if (point.position.allFinite() && point.view_ids.size() >= 2)
            points.push_back(point.position);
    return estimate_subject_bounds(
        points, result, thread_count, padding_scale);
}

}  // namespace photara::mvs::detail
