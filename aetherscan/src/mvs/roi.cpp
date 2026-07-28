#include "mvs/internal.hpp"

#include "core/logging.hpp"
#include "parallel/thread_pool.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>

namespace aetherscan::mvs::detail {
namespace {

struct Cell {
    int x{}, y{}, z{};
    bool operator==(const Cell&) const = default;
};

struct CellHash {
    std::size_t operator()(const Cell& c) const noexcept {
        std::size_t h = static_cast<std::uint32_t>(c.x) * 73856093U;
        h ^= static_cast<std::uint32_t>(c.y) * 19349663U;
        h ^= static_cast<std::uint32_t>(c.z) * 83492791U;
        return h;
    }
};

constexpr std::size_t k_maximum_component_samples = 500'000;

[[nodiscard]] std::pair<Vec3f, Vec3f> bounds(
    const std::vector<DensePoint>& points) {
    std::array<std::vector<float>, 3> coordinates;
    const std::size_t stride = std::max<std::size_t>(1, points.size() / 100000);
    for (std::size_t i = 0; i < points.size(); i += stride) {
        for (int axis = 0; axis < 3; ++axis)
            coordinates[static_cast<std::size_t>(axis)].push_back(
                points[i].position[axis]);
    }
    Vec3f lo, hi;
    for (int axis = 0; axis < 3; ++axis) {
        auto& values = coordinates[static_cast<std::size_t>(axis)];
        const std::size_t lower = values.size() / 100;
        const std::size_t upper = values.size() - 1 - lower;
        std::nth_element(values.begin(), values.begin() + lower, values.end());
        lo[axis] = values[lower];
        std::nth_element(values.begin(), values.begin() + upper, values.end());
        hi[axis] = values[upper];
    }
    return {lo, hi};
}

[[nodiscard]] Vec3f viewing_target(const MvsScene& scene) {
    Eigen::Matrix3f A = Eigen::Matrix3f::Zero();
    Vec3f b = Vec3f::Zero();
    for (const auto& view : scene.views) {
        const Vec3f c = view.pose.C.cast<float>();
        Vec3f d = view.pose.R.transpose().cast<float>() * Vec3f{0.F, 0.F, 1.F};
        if (d.squaredNorm() < 1e-8F) continue;
        d.normalize();
        const Mat3f q = Mat3f::Identity() - d * d.transpose();
        A += q;
        b += q * c;
    }
    if (std::abs(A.determinant()) > 1e-8F) return A.ldlt().solve(b);
    if (!scene.dense_cloud.points.empty()) {
        Vec3f sum = Vec3f::Zero();
        for (const auto& p : scene.dense_cloud.points) sum += p.position;
        return sum / static_cast<float>(scene.dense_cloud.points.size());
    }
    return Vec3f::Zero();
}

[[nodiscard]] bool estimate_ground_plane(
    const MvsScene& scene, const float threshold, const unsigned iterations,
    Vec3f& normal, float& offset) {
    const auto& points = scene.dense_cloud.points;
    if (points.size() < 100) return false;
    const std::size_t max_samples = 100000;
    const std::size_t stride = std::max<std::size_t>(1, points.size() / max_samples);
    std::vector<Vec3f> sample;
    sample.reserve(std::min(points.size(), max_samples));
    for (std::size_t i = 0; i < points.size(); i += stride)
        sample.push_back(points[i].position);
    if (sample.size() < 100) return false;

    const Vec3f target = viewing_target(scene);
    Vec3f up = Vec3f::Zero();
    Vec3f first_up = Vec3f::Zero();
    for (const auto& view : scene.views) {
        Vec3f candidate = view.pose.R.transpose().cast<float>() *
                          Vec3f{0.F, -1.F, 0.F};
        if (candidate.squaredNorm() < 1e-8F) continue;
        candidate.normalize();
        if (first_up.squaredNorm() == 0.F) first_up = candidate;
        if (candidate.dot(first_up) < 0.F) candidate = -candidate;
        up += candidate;
    }
    const bool reliable_up = up.norm() > 0.35F * scene.views.size();
    if (reliable_up) up.normalize();
    std::mt19937 rng(0xA37E51U);
    std::uniform_int_distribution<std::size_t> pick(0, sample.size() - 1);
    std::size_t best_support = 0;
    Vec3f best_n = Vec3f::Zero();
    float best_d = 0.F;
    for (unsigned iter = 0; iter < iterations; ++iter) {
        const Vec3f a = sample[pick(rng)];
        const Vec3f b = sample[pick(rng)];
        const Vec3f c = sample[pick(rng)];
        Vec3f n = (b - a).cross(c - a);
        if (n.squaredNorm() < threshold * threshold) continue;
        n.normalize();
        float d = n.dot(a);
        // Reject walls/backdrops when camera roll provides a reliable world-up
        // estimate. A 0.45 cosine still permits sloped tabletops.
        if (reliable_up && std::abs(n.dot(up)) < 0.45F) continue;

        unsigned positive_cameras = 0;
        unsigned negative_cameras = 0;
        for (const auto& view : scene.views) {
            const float side = n.dot(view.pose.C.cast<float>()) - d;
            if (side >= 0.F) ++positive_cameras;
            else ++negative_cameras;
        }
        const unsigned camera_majority = std::max(positive_cameras, negative_cameras);
        if (!scene.views.empty() &&
            camera_majority * 5U < static_cast<unsigned>(scene.views.size()) * 4U)
            continue;
        if (positive_cameras < negative_cameras) {
            n = -n;
            d = -d;
        }
        // The tabletop/ground must be behind the camera-facing subject target.
        if (n.dot(target) - d < -2.F * threshold) continue;
        std::size_t support = 0;
        for (const Vec3f& p : sample)
            if (std::abs(n.dot(p) - d) <= threshold) ++support;
        if (support > best_support) {
            best_support = support;
            best_n = n;
            best_d = d;
        }
    }
    if (best_support < std::max<std::size_t>(50, sample.size() / 50)) return false;

    Vec3f centroid = Vec3f::Zero();
    std::size_t count = 0;
    for (const Vec3f& p : sample) {
        if (std::abs(best_n.dot(p) - best_d) <= threshold) {
            centroid += p;
            ++count;
        }
    }
    centroid /= static_cast<float>(std::max<std::size_t>(count, 1));
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const Vec3f& p : sample) {
        if (std::abs(best_n.dot(p) - best_d) <= threshold) {
            const Vec3f q = p - centroid;
            covariance += q * q.transpose();
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(covariance);
    if (eig.info() == Eigen::Success) best_n = eig.eigenvectors().col(0).normalized();
    best_d = best_n.dot(centroid);
    if (best_n.dot(target) < best_d) {
        best_n = -best_n;
        best_d = -best_d;
    }
    normal = best_n;
    offset = best_d;
    return true;
}

[[nodiscard]] std::vector<std::size_t> subject_component(
    const MvsScene& scene, const Vec3f& target, const float voxel,
    const bool has_plane, const Vec3f& plane_n, const float plane_d,
    const float plane_threshold, std::size_t& occupied_cells,
    std::size_t& minimum_cell_support) {
    const auto& points = scene.dense_cloud.points;
    struct CellAccumulator {
        std::size_t points{};
        Vec3f sum{Vec3f::Zero()};
        std::size_t representative{};
        float representative_distance{
            std::numeric_limits<float>::infinity()};
    };
    std::unordered_map<Cell, CellAccumulator, CellHash> cells;
    cells.reserve(
        std::min(points.size(), k_maximum_component_samples) / 2 + 1);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Vec3f& p = points[i].position;
        if (has_plane && plane_n.dot(p) - plane_d <= 1.5F * plane_threshold) continue;
        const Cell key{
            static_cast<int>(std::floor(p.x() / voxel)),
            static_cast<int>(std::floor(p.y() / voxel)),
            static_cast<int>(std::floor(p.z() / voxel))};
        auto& cell = cells[key];
        ++cell.points;
        cell.sum += p;
        const Vec3f center{
            (static_cast<float>(key.x) + 0.5F) * voxel,
            (static_cast<float>(key.y) + 0.5F) * voxel,
            (static_cast<float>(key.z) + 0.5F) * voxel};
        const float distance = (p - center).squaredNorm();
        if (distance < cell.representative_distance) {
            cell.representative_distance = distance;
            cell.representative = i;
        }
    }
    if (cells.empty()) return {};

    // Fusion output order varies with parallel reduction. A fixed-stride
    // point sample consequently changed the component graph between otherwise
    // identical runs and sometimes connected the subject to its support.
    // Accumulate every point into an order-independent occupancy grid instead.
    //
    // On large dense clouds, real surface voxels contain tens to thousands of
    // samples while accidental MVS bridges are far below the median. Removing
    // cells below one quarter of the median breaks those unstable bridges before
    // component selection. Small/synthetic clouds keep every occupied cell.
    minimum_cell_support = 1;
    if (points.size() > k_maximum_component_samples) {
        std::vector<std::size_t> counts;
        counts.reserve(cells.size());
        for (const auto& entry : cells)
            counts.push_back(entry.second.points);
        const auto middle = counts.begin() +
            static_cast<std::ptrdiff_t>(counts.size() / 2);
        std::nth_element(counts.begin(), middle, counts.end());
        const std::size_t total_samples = std::accumulate(
            counts.begin(), counts.end(), std::size_t{0});
        const std::size_t mean_support =
            total_samples / std::max<std::size_t>(counts.size(), 1);
        minimum_cell_support = std::max({
            std::size_t{2}, *middle / 4, mean_support / 16});
        for (auto it = cells.begin(); it != cells.end();) {
            if (it->second.points < minimum_cell_support)
                it = cells.erase(it);
            else
                ++it;
        }
    }
    occupied_cells = cells.size();
    if (cells.empty()) return {};

    std::unordered_map<Cell, unsigned, CellHash> labels;
    labels.reserve(cells.size());
    struct Component { std::vector<Cell> cells; std::size_t points{}; Vec3f sum{Vec3f::Zero()}; };
    std::vector<Component> components;
    for (const auto& entry : cells) {
        const Cell& seed = entry.first;
        if (labels.contains(seed)) continue;
        const unsigned label = static_cast<unsigned>(components.size());
        components.emplace_back();
        std::queue<Cell> queue;
        queue.push(seed);
        labels.emplace(seed, label);
        while (!queue.empty()) {
            const Cell cell = queue.front();
            queue.pop();
            auto& component = components.back();
            component.cells.push_back(cell);
            const auto& accumulator = cells.at(cell);
            component.points += accumulator.points;
            component.sum += accumulator.sum;
            for (int dz = -1; dz <= 1; ++dz)
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        const Cell next{cell.x + dx, cell.y + dy, cell.z + dz};
                        if (cells.contains(next) && !labels.contains(next)) {
                            labels.emplace(next, label);
                            queue.push(next);
                        }
                    }
        }
    }
    std::size_t best = 0;
    float best_score = -1.F;
    for (std::size_t i = 0; i < components.size(); ++i) {
        const auto& c = components[i];
        float minimum_distance = std::numeric_limits<float>::max();
        for (const Cell& cell : c.cells) {
            const Vec3f center{
                (static_cast<float>(cell.x) + 0.5F) * voxel,
                (static_cast<float>(cell.y) + 0.5F) * voxel,
                (static_cast<float>(cell.z) + 0.5F) * voxel};
            minimum_distance = std::min(minimum_distance, (center - target).norm());
        }
        const float distance_voxels = minimum_distance / voxel;
        const float score = std::log1p(static_cast<float>(c.points)) /
                            (1.F + distance_voxels);
        if (score > best_score) { best_score = score; best = i; }
    }
    std::vector<std::size_t> selected;
    selected.reserve(components[best].cells.size());
    for (const Cell& cell : components[best].cells)
        selected.push_back(cells.at(cell).representative);
    return selected;
}

[[nodiscard]] OrientedBoundingBox fit_obb(
    const std::vector<DensePoint>& points,
    const std::vector<std::size_t>& indices, const float margin_fraction,
    const Vec3f* ground_normal) {
    OrientedBoundingBox roi;
    if (indices.size() < 16) return roi;
    Vec3f mean = Vec3f::Zero();
    for (const auto i : indices) mean += points[i].position;
    mean /= static_cast<float>(indices.size());
    Mat3f covariance = Mat3f::Zero();
    for (const auto i : indices) {
        const Vec3f q = points[i].position - mean;
        covariance += q * q.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Mat3f> eig(covariance);
    if (eig.info() != Eigen::Success) return roi;
    Mat3f axes;
    axes.col(0) = eig.eigenvectors().col(2).normalized();
    axes.col(1) = eig.eigenvectors().col(1).normalized();
    axes.col(2) = axes.col(0).cross(axes.col(1)).normalized();
    if (ground_normal != nullptr && ground_normal->squaredNorm() > 0.5F) {
        const Vec3f z = ground_normal->normalized();
        Vec3f x = eig.eigenvectors().col(2) -
                  z * z.dot(eig.eigenvectors().col(2));
        if (x.squaredNorm() < 1e-6F)
            x = eig.eigenvectors().col(1) -
                z * z.dot(eig.eigenvectors().col(1));
        if (x.squaredNorm() > 1e-6F) {
            x.normalize();
            const Vec3f y = z.cross(x).normalized();
            axes.col(0) = x;
            axes.col(1) = y;
            axes.col(2) = z;
        }
    }

    std::array<std::vector<float>, 3> coordinates;
    for (auto& values : coordinates) values.reserve(indices.size());
    for (const auto i : indices) {
        const Vec3f q = axes.transpose() * (points[i].position - mean);
        for (int axis = 0; axis < 3; ++axis)
            coordinates[static_cast<std::size_t>(axis)].push_back(q[axis]);
    }
    Vec3f lo, hi;
    for (int axis = 0; axis < 3; ++axis) {
        auto& values = coordinates[static_cast<std::size_t>(axis)];
        const std::size_t trim = values.size() / 200;  // robust 0.5% tails
        const std::size_t upper = values.size() - 1 - trim;
        std::nth_element(values.begin(), values.begin() + trim, values.end());
        lo[axis] = values[trim];
        std::nth_element(values.begin(), values.begin() + upper, values.end());
        hi[axis] = values[upper];
    }
    const Vec3f local_center = 0.5F * (lo + hi);
    roi.valid = true;
    roi.axes = axes;
    roi.center = mean + axes * local_center;
    roi.half_extent = (0.5F * (hi - lo)).cwiseMax(Vec3f::Constant(1e-6F));
    roi.half_extent.array() *= 1.F + std::max(0.F, margin_fraction);
    return roi;
}

void rasterize_triangle(
    const MvsView& view, const Vec3f& a, const Vec3f& b, const Vec3f& c,
    std::vector<float>& zbuffer, std::vector<std::uint8_t>& mask) {
    const Mat3f R = view.pose.R.cast<float>();
    const Vec3f C = view.pose.C.cast<float>();
    const Vec3f pa = R * (a - C), pb = R * (b - C), pc = R * (c - C);
    if (pa.z() <= 1e-5F || pb.z() <= 1e-5F || pc.z() <= 1e-5F) return;
    float ax{}, ay{}, bx{}, by{}, cx{}, cy{};
    if (!view.project(pa, ax, ay) || !view.project(pb, bx, by) ||
        !view.project(pc, cx, cy)) return;
    const float area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (std::abs(area) < 1e-6F) return;
    const int min_x = std::max(0, static_cast<int>(std::floor(std::min({ax, bx, cx}))));
    const int max_x = std::min(static_cast<int>(view.width) - 1,
        static_cast<int>(std::ceil(std::max({ax, bx, cx}))));
    const int min_y = std::max(0, static_cast<int>(std::floor(std::min({ay, by, cy}))));
    const int max_y = std::min(static_cast<int>(view.height) - 1,
        static_cast<int>(std::ceil(std::max({ay, by, cy}))));
    for (int y = min_y; y <= max_y; ++y) for (int x = min_x; x <= max_x; ++x) {
        const float px = static_cast<float>(x) + 0.5F;
        const float py = static_cast<float>(y) + 0.5F;
        const float wa = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area;
        const float wb = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area;
        const float wc = 1.F - wa - wb;
        if (wa < -1e-4F || wb < -1e-4F || wc < -1e-4F) continue;
        const float z = wa * pa.z() + wb * pb.z() + wc * pc.z();
        const std::size_t index = static_cast<std::size_t>(y) * view.width + x;
        if (z < zbuffer[index]) { zbuffer[index] = z; mask[index] = 255; }
    }
}

void dilate(
    std::vector<std::uint8_t>& mask, const std::uint32_t width,
    const std::uint32_t height, const unsigned radius) {
    if (radius == 0 || mask.empty()) return;
    std::vector<std::uint8_t> horizontal(mask.size(), 0), output(mask.size(), 0);
    for (std::uint32_t y = 0; y < height; ++y) {
        int active = 0;
        for (int x = -static_cast<int>(radius); x < static_cast<int>(width + radius); ++x) {
            const int add = x + static_cast<int>(radius);
            const int remove = x - static_cast<int>(radius) - 1;
            if (add >= 0 && add < static_cast<int>(width) && mask[static_cast<std::size_t>(y) * width + add]) ++active;
            if (remove >= 0 && remove < static_cast<int>(width) && mask[static_cast<std::size_t>(y) * width + remove]) --active;
            if (x >= 0 && x < static_cast<int>(width) && active > 0) horizontal[static_cast<std::size_t>(y) * width + x] = 255;
        }
    }
    for (std::uint32_t x = 0; x < width; ++x) {
        int active = 0;
        for (int y = -static_cast<int>(radius); y < static_cast<int>(height + radius); ++y) {
            const int add = y + static_cast<int>(radius);
            const int remove = y - static_cast<int>(radius) - 1;
            if (add >= 0 && add < static_cast<int>(height) && horizontal[static_cast<std::size_t>(add) * width + x]) ++active;
            if (remove >= 0 && remove < static_cast<int>(height) && horizontal[static_cast<std::size_t>(remove) * width + x]) --active;
            if (y >= 0 && y < static_cast<int>(height) && active > 0) output[static_cast<std::size_t>(y) * width + x] = 255;
        }
    }
    mask.swap(output);
}

void erode(
    std::vector<std::uint8_t>& mask, const std::uint32_t width,
    const std::uint32_t height, const unsigned radius) {
    if (radius == 0 || mask.empty()) return;
    std::vector<std::uint8_t> horizontal(mask.size(), 0), output(mask.size(), 0);
    const int diameter = 2 * static_cast<int>(radius) + 1;
    for (std::uint32_t y = 0; y < height; ++y) {
        int active = 0;
        for (int x = -static_cast<int>(radius);
             x < static_cast<int>(width + radius); ++x) {
            const int add = x + static_cast<int>(radius);
            const int remove = x - static_cast<int>(radius) - 1;
            if (add >= 0 && add < static_cast<int>(width) &&
                mask[static_cast<std::size_t>(y) * width + add])
                ++active;
            if (remove >= 0 && remove < static_cast<int>(width) &&
                mask[static_cast<std::size_t>(y) * width + remove])
                --active;
            if (x >= 0 && x < static_cast<int>(width) && active == diameter)
                horizontal[static_cast<std::size_t>(y) * width + x] = 255;
        }
    }
    for (std::uint32_t x = 0; x < width; ++x) {
        int active = 0;
        for (int y = -static_cast<int>(radius);
             y < static_cast<int>(height + radius); ++y) {
            const int add = y + static_cast<int>(radius);
            const int remove = y - static_cast<int>(radius) - 1;
            if (add >= 0 && add < static_cast<int>(height) &&
                horizontal[static_cast<std::size_t>(add) * width + x])
                ++active;
            if (remove >= 0 && remove < static_cast<int>(height) &&
                horizontal[static_cast<std::size_t>(remove) * width + x])
                --active;
            if (y >= 0 && y < static_cast<int>(height) && active == diameter)
                output[static_cast<std::size_t>(y) * width + x] = 255;
        }
    }
    mask.swap(output);
}

void close_broken_silhouette(
    std::vector<std::uint8_t>& mask, const std::uint32_t width,
    const std::uint32_t height, const unsigned radius) {
    if (radius == 0) return;
    dilate(mask, width, height, radius);
    erode(mask, width, height, radius);
}

std::size_t fill_enclosed_holes(
    std::vector<std::uint8_t>& mask, const std::uint32_t width,
    const std::uint32_t height) {
    if (width == 0 || height == 0 || mask.empty()) return 0;

    // Mark all background reachable from the image border. Any zero pixel
    // left afterwards is an enclosed hole in the projected silhouette.
    // Reuse value 1 as the temporary exterior marker so this needs only a
    // compact traversal queue in addition to the mask itself.
    std::vector<std::uint32_t> queue;
    queue.reserve(
        2U * static_cast<std::size_t>(width + height));
    const auto enqueue = [&](const std::uint32_t x, const std::uint32_t y) {
        const std::size_t index = static_cast<std::size_t>(y) * width + x;
        if (mask[index] == 0) {
            mask[index] = 1;
            queue.push_back(static_cast<std::uint32_t>(index));
        }
    };
    for (std::uint32_t x = 0; x < width; ++x) {
        enqueue(x, 0);
        if (height > 1) enqueue(x, height - 1);
    }
    for (std::uint32_t y = 1; y + 1 < height; ++y) {
        enqueue(0, y);
        if (width > 1) enqueue(width - 1, y);
    }

    for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
        const std::uint32_t index = queue[cursor];
        const std::uint32_t x = index % width;
        const std::uint32_t y = index / width;
        if (x > 0) enqueue(x - 1, y);
        if (x + 1 < width) enqueue(x + 1, y);
        if (y > 0) enqueue(x, y - 1);
        if (y + 1 < height) enqueue(x, y + 1);
    }

    std::size_t filled = 0;
    for (auto& pixel : mask) {
        if (pixel == 0) {
            pixel = 255;
            ++filled;
        } else if (pixel == 1) {
            pixel = 0;
        }
    }
    return filled;
}

}  // namespace

bool load_manual_roi(
    const std::filesystem::path& path, OrientedBoundingBox& roi) {
    std::ifstream input(path);
    std::array<float, 15> v{};
    for (float& value : v)
        if (!(input >> value)) return false;
    roi.center = {v[0], v[1], v[2]};
    // Nine row-major values; orthonormalize to make hand-edited files safe.
    Mat3f raw;
    raw << v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11];
    Eigen::JacobiSVD<Mat3f> svd(raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
    roi.axes = svd.matrixU() * svd.matrixV().transpose();
    if (roi.axes.determinant() < 0.F) roi.axes.col(2) = -roi.axes.col(2);
    roi.half_extent = Vec3f{v[12], v[13], v[14]}.cwiseAbs();
    roi.valid = roi.center.allFinite() && roi.axes.allFinite() &&
                roi.half_extent.allFinite() && (roi.half_extent.array() > 0.F).all();
    return roi.valid;
}

bool estimate_automatic_roi(MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.auto_roi");
    if (scene.dense_cloud.points.size() < 100) return false;
    const std::size_t source_points = scene.dense_cloud.points.size();
    const auto [lo, hi] = bounds(scene.dense_cloud.points);
    const float diagonal = (hi - lo).norm();
    if (!(diagonal > 1e-8F)) return false;
    const float plane_threshold = std::max(
        diagonal * options.auto_roi_plane_threshold_fraction, 1e-6F);
    Vec3f plane_n = Vec3f::Zero();
    float plane_d = 0.F;
    const bool has_plane = estimate_ground_plane(
        scene, plane_threshold, options.auto_roi_ransac_iters, plane_n, plane_d);
    const float voxel = std::max(
        diagonal * options.auto_roi_component_voxel_fraction,
        plane_threshold * 1.5F);
    std::size_t occupied_component_cells = 0;
    std::size_t minimum_cell_support = 1;
    const auto selected = subject_component(
        scene, viewing_target(scene), voxel, has_plane, plane_n, plane_d,
        plane_threshold, occupied_component_cells, minimum_cell_support);
    OrientedBoundingBox roi = fit_obb(
        scene.dense_cloud.points, selected, options.roi_margin_fraction,
        has_plane ? &plane_n : nullptr);
    if (!roi.valid) return false;

    const std::size_t subject_count = static_cast<std::size_t>(std::count_if(
        scene.dense_cloud.points.begin(), scene.dense_cloud.points.end(),
        [&](const DensePoint& point) {
            return roi.contains(point.position);
        }));
    std::vector<DensePoint> subject;
    subject.reserve(subject_count);
    for (auto& point : scene.dense_cloud.points)
        if (roi.contains(point.position))
            subject.push_back(std::move(point));
    scene.dense_cloud.points.swap(subject);
    scene.roi = roi;
    scene.roi_automatic = true;
    scene.has_ground_plane = has_plane;
    scene.ground_normal = plane_n;
    scene.ground_offset = plane_d;
    core::Logger::instance().info(
        "auto ROI: ground=", has_plane ? "yes" : "no",
        " source_points=", source_points,
        " component_cells=", selected.size(),
        " occupied_cells=", occupied_component_cells,
        " minimum_cell_support=", minimum_cell_support,
        " subject_points=", scene.dense_cloud.points.size(),
        " half_extent=", roi.half_extent.transpose());
    return true;
}

void build_projected_foreground_masks(
    MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.project_roi_masks");
    Mesh box;
    if (scene.roi.valid) {
        box.vertices.resize(8);
        for (int i = 0; i < 8; ++i) {
            const Vec3f local{
                (i & 1) ? scene.roi.half_extent.x() : -scene.roi.half_extent.x(),
                (i & 2) ? scene.roi.half_extent.y() : -scene.roi.half_extent.y(),
                (i & 4) ? scene.roi.half_extent.z() : -scene.roi.half_extent.z()};
            box.vertices[i] = scene.roi.center + scene.roi.axes * local;
        }
        constexpr int f[][3] = {
            {0,2,3},{0,3,1},{4,5,7},{4,7,6},{0,1,5},{0,5,4},
            {2,6,7},{2,7,3},{0,4,6},{0,6,2},{1,3,7},{1,7,5}};
        for (const auto& face : f) box.faces.emplace_back(face[0], face[1], face[2]);
    }
    const Mesh* source = scene.mesh.faces.empty() ? &box : &scene.mesh;
    const unsigned threads =
        parallel::resolve_thread_count(scene.thread_count);
    std::uint32_t minimum_view_dimension =
        std::numeric_limits<std::uint32_t>::max();
    for (const auto& view : scene.views)
        minimum_view_dimension = std::min(
            minimum_view_dimension, std::min(view.width, view.height));
    if (minimum_view_dimension == std::numeric_limits<std::uint32_t>::max())
        minimum_view_dimension = 0;
    const unsigned adaptive_close_radius = std::clamp(
        std::max(
            3U * options.auto_roi_mask_dilate_px,
            minimum_view_dimension / 40U),
        1U, 32U);
    const unsigned close_radius = options.auto_roi_mask_close_px != 0
        ? options.auto_roi_mask_close_px
        : adaptive_close_radius;
    std::vector<std::size_t> filled_hole_pixels(scene.views.size(), 0);
    parallel::parallel_for(
        scene.views.size(), threads, [&](const std::size_t view_index) {
        auto& view = scene.views[view_index];
        const std::size_t size = static_cast<std::size_t>(view.width) * view.height;
        const bool has_input_mask = view.foreground_mask.size() == size;
        std::vector<std::uint8_t> projected(size, 0);
        std::vector<float> zbuffer(size, std::numeric_limits<float>::max());
        for (const auto& face : source->faces) {
            const int a = face.x(), b = face.y(), c = face.z();
            if (a < 0 || b < 0 || c < 0 ||
                static_cast<std::size_t>(std::max({a,b,c})) >= source->vertices.size()) continue;
            rasterize_triangle(view, source->vertices[a], source->vertices[b],
                               source->vertices[c], zbuffer, projected);
        }
        if (source != &box) {
            close_broken_silhouette(
                projected, view.width, view.height, close_radius);
            filled_hole_pixels[view_index] =
                fill_enclosed_holes(projected, view.width, view.height);
        }
        dilate(projected, view.width, view.height, options.auto_roi_mask_dilate_px);
        if (has_input_mask) {
            // A coarse geometric proxy is intentionally conservative. Using
            // its silhouette as a hard intersection makes every coarse hole
            // irreversible in the final PatchMatch pass. An existing input
            // mask is already the stronger 2D foreground cue, so constrain it
            // only by the projected OBB envelope. Keep the coarse silhouette
            // path for datasets without masks.
            if (!box.faces.empty() && source != &box) {
                std::vector<std::uint8_t> envelope(size, 0);
                std::fill(
                    zbuffer.begin(), zbuffer.end(),
                    std::numeric_limits<float>::max());
                for (const auto& face : box.faces)
                    rasterize_triangle(
                        view, box.vertices[static_cast<std::size_t>(face.x())],
                        box.vertices[static_cast<std::size_t>(face.y())],
                        box.vertices[static_cast<std::size_t>(face.z())],
                        zbuffer, envelope);
                dilate(
                    envelope, view.width, view.height,
                    options.auto_roi_mask_dilate_px);
                projected.swap(envelope);
            }
            for (std::size_t i = 0; i < size; ++i)
                projected[i] = (projected[i] && view.foreground_mask[i]) ? 255 : 0;
        }
        view.foreground_mask.swap(projected);
    });
    const std::size_t views_with_holes = static_cast<std::size_t>(std::count_if(
        filled_hole_pixels.begin(), filled_hole_pixels.end(),
        [](const std::size_t pixels) { return pixels != 0; }));
    const std::size_t total_filled = std::accumulate(
        filled_hole_pixels.begin(), filled_hole_pixels.end(), std::size_t{0});
    const std::size_t maximum_filled = filled_hole_pixels.empty()
        ? 0
        : *std::max_element(
              filled_hole_pixels.begin(), filled_hole_pixels.end());
    core::Logger::instance().info(
        "projected foreground masks: views=", scene.views.size(),
        " coarse_mesh=", source != &box ? 1 : 0,
        " close_radius=", source != &box ? close_radius : 0,
        " hole_filled_views=", views_with_holes,
        " filled_hole_pixels=", total_filled,
        " maximum_view_hole_pixels=", maximum_filled);
}

}  // namespace aetherscan::mvs::detail
