#include "splat/trainer.hpp"

#include "core/logging.hpp"
#include "cuda_ops.hpp"
#include "pam_cuda.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#if defined(PHOTARA_HAS_CGAL)
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#endif

namespace photara::splat {
namespace {

class KnnTree {
public:
    explicit KnnTree(std::vector<mvs::Vec3f> points)
        : points_(std::move(points)), order_(points_.size()) {
        std::iota(order_.begin(), order_.end(), std::size_t{0});
        nodes_.reserve(points.size());
        root_ = build(0, order_.size());
    }

    [[nodiscard]] std::vector<std::size_t> nearest(
        const mvs::Vec3f& query, const std::size_t requested) const {
        using Entry = std::pair<float, std::size_t>;
        std::priority_queue<Entry> best;
        search(root_, query, std::max<std::size_t>(requested, 1), best);
        std::vector<std::size_t> result(best.size());
        for (std::size_t i = best.size(); i != 0; --i) {
            result[i - 1] = best.top().second;
            best.pop();
        }
        return result;
    }

    [[nodiscard]] tinytensor::Tensor upload_nodes() const {
        std::vector<int> packed;
        packed.reserve(nodes_.size() * 4);
        for (const auto& node : nodes_) {
            packed.push_back(static_cast<int>(node.point));
            packed.push_back(node.left);
            packed.push_back(node.right);
            packed.push_back(node.axis);
        }
        return tinytensor::Tensor::from_vector(
            packed, {nodes_.size(), std::size_t{4}}, tinytensor::Device::CUDA);
    }

    [[nodiscard]] int root() const { return root_; }

    [[nodiscard]] tinytensor::Tensor upload_bounds() const {
        std::vector<float> packed;
        packed.reserve(nodes_.size() * 6);
        for (const auto& node : nodes_) {
            packed.insert(packed.end(), node.minimum.data(), node.minimum.data() + 3);
            packed.insert(packed.end(), node.maximum.data(), node.maximum.data() + 3);
        }
        return tinytensor::Tensor::from_vector(
            packed, {nodes_.size(), std::size_t{6}}, tinytensor::Device::CUDA);
    }

private:
    struct Node {
        std::size_t point{};
        int left{-1};
        int right{-1};
        std::uint8_t axis{};
        mvs::Vec3f minimum, maximum;
    };

    int build(const std::size_t begin, const std::size_t end) {
        if (begin >= end) return -1;
        mvs::Vec3f minimum = points_[order_[begin]];
        mvs::Vec3f maximum = minimum;
        for (std::size_t i = begin + 1; i < end; ++i) {
            minimum = minimum.cwiseMin(points_[order_[i]]);
            maximum = maximum.cwiseMax(points_[order_[i]]);
        }
        Eigen::Index axis{};
        (maximum - minimum).maxCoeff(&axis);
        const std::size_t middle = begin + (end - begin) / 2;
        std::nth_element(
            order_.begin() + static_cast<std::ptrdiff_t>(begin),
            order_.begin() + static_cast<std::ptrdiff_t>(middle),
            order_.begin() + static_cast<std::ptrdiff_t>(end),
            [&](const std::size_t left, const std::size_t right) {
                return points_[left](axis) < points_[right](axis);
            });
        const int node = static_cast<int>(nodes_.size());
        nodes_.push_back({
            order_[middle], -1, -1, static_cast<std::uint8_t>(axis), minimum, maximum});
        nodes_[static_cast<std::size_t>(node)].left = build(begin, middle);
        nodes_[static_cast<std::size_t>(node)].right =
            build(middle + 1, end);
        return node;
    }

    void search(
        const int node_index, const mvs::Vec3f& query,
        const std::size_t requested,
        std::priority_queue<std::pair<float, std::size_t>>& best) const {
        if (node_index < 0) return;
        const Node& node = nodes_[static_cast<std::size_t>(node_index)];
        const mvs::Vec3f& point = points_[node.point];
        const float distance2 = (query - point).squaredNorm();
        if (best.size() < requested)
            best.emplace(distance2, node.point);
        else if (distance2 < best.top().first) {
            best.pop();
            best.emplace(distance2, node.point);
        }
        const float delta = query(node.axis) - point(node.axis);
        const int near = delta < 0.F ? node.left : node.right;
        const int far = delta < 0.F ? node.right : node.left;
        search(near, query, requested, best);
        if (best.size() < requested || delta * delta <= best.top().first)
            search(far, query, requested, best);
    }

    std::vector<mvs::Vec3f> points_;
    std::vector<std::size_t> order_;
    std::vector<Node> nodes_;
    int root_{-1};
};

struct HostGaussianField {
    std::vector<mvs::Vec3f> means;
    std::vector<mvs::Vec3f> normals;
    std::vector<mvs::Vec3f> scales;
    std::vector<mvs::Vec3f> inverse_variances;
    std::vector<Eigen::Matrix3f> rotations;
    std::vector<float> opacities;
    KnnTree tree;

    explicit HostGaussianField(const GaussianModel& model)
        : means(build_means(model)), normals(model.size()),
          scales(model.size()), inverse_variances(model.size()), rotations(model.size()),
          opacities(model.size()), tree(means) {
        const auto features = model.normal_features.to_vector();
        const detail::ActivatedParameters activated =
            detail::activate_parameters(model);
        const auto scales = activated.scales.to_vector();
        const auto quaternions = activated.quaternions.to_vector();
        opacities = activated.opacities.to_vector();
        for (std::size_t index = 0; index < model.size(); ++index) {
            mvs::Vec3f direction{
                features[4 * index], features[4 * index + 1],
                features[4 * index + 2]};
            if (direction.allFinite() && direction.squaredNorm() > 1e-20F)
                direction.normalize();
            else
                direction = mvs::Vec3f::UnitZ();
            normals[index] =
                direction * std::tanh(features[4 * index + 3]);
            this->scales[index] = mvs::Vec3f{
                scales[3 * index], scales[3 * index + 1],
                scales[3 * index + 2]};
            inverse_variances[index] = mvs::Vec3f{
                1.F / std::max(
                    scales[3 * index] * scales[3 * index], 1e-20F),
                1.F / std::max(
                    scales[3 * index + 1] * scales[3 * index + 1],
                    1e-20F),
                1.F / std::max(
                    scales[3 * index + 2] * scales[3 * index + 2],
                    1e-20F)};
            rotations[index] = Eigen::Quaternionf(
                quaternions[4 * index], quaternions[4 * index + 1],
                quaternions[4 * index + 2],
                quaternions[4 * index + 3]).toRotationMatrix();
        }
    }

    [[nodiscard]] mvs::Vec3f gradient(
        const mvs::Vec3f& point, const std::size_t neighbors) const {
        mvs::Vec3f result = mvs::Vec3f::Zero();
        for (const std::size_t index : tree.nearest(point, neighbors)) {
            const mvs::Vec3f delta = point - means[index];
            if (normals[index].dot(delta) < 0.F) continue;
            const mvs::Vec3f local = rotations[index].transpose() * delta;
            const float exponent = -0.5F *
                (local.array().square() *
                 inverse_variances[index].array()).sum();
            const float gaussian = std::clamp(
                opacities[index] * std::exp(exponent), 0.F, 1.F - 1e-7F);
            const mvs::Vec3f sigma_inverse_delta = rotations[index] *
                (inverse_variances[index].array() * local.array()).matrix();
            result += gaussian / (1.F - gaussian + 1e-8F) *
                      sigma_inverse_delta;
        }
        return result;
    }

    [[nodiscard]] detail::PamGpuField upload() const {
        std::vector<float> packed;
        packed.reserve(means.size() * 19);
        for (std::size_t i = 0; i < means.size(); ++i) {
            for (const auto* vector : {&means[i], &normals[i], &inverse_variances[i]})
                packed.insert(packed.end(), vector->data(), vector->data() + 3);
            packed.insert(packed.end(), rotations[i].data(), rotations[i].data() + 9);
            packed.push_back(opacities[i]);
        }
        return {tree.upload_nodes(), tinytensor::Tensor::from_vector(
            packed, {means.size(), std::size_t{19}}, tinytensor::Device::CUDA),
            tree.root(), tree.upload_bounds()};
    }

private:
    static std::vector<mvs::Vec3f> build_means(
        const GaussianModel& model) {
        const auto values = model.means.to_vector();
        std::vector<mvs::Vec3f> result(model.size());
        for (std::size_t index = 0; index < result.size(); ++index)
            result[index] = mvs::Vec3f{
                values[3 * index], values[3 * index + 1],
                values[3 * index + 2]};
        return result;
    }
};

tinytensor::Tensor upload_points(std::span<const mvs::Vec3f> points) {
    std::vector<float> packed;
    packed.reserve(points.size() * 3);
    for (const auto& point : points)
        packed.insert(packed.end(), point.data(), point.data() + 3);
    return tinytensor::Tensor::from_vector(
        packed, {points.size(), std::size_t{3}}, tinytensor::Device::CUDA);
}

tinytensor::Tensor evaluate_global_occupancy_device(
    const GaussianModel& model, const mvs::MvsScene& scene,
    const tinytensor::Tensor& points,
    const TrainingOptions& training_options,
    const std::size_t chunk_size, const float mask_background_threshold,
    OccupancyCache* cache = nullptr) {
    const std::size_t point_count = points.shape()[0];
    auto result = tinytensor::Tensor::empty({point_count}, tinytensor::Device::CUDA);
    Rasterizer rasterizer;
    RasterizeOptions raster_options;
    raster_options.kernel_size = training_options.kernel_size;
    raster_options.scale_modifier = training_options.scale_modifier;
    const std::size_t capacity = std::max<std::size_t>(chunk_size, 1);
    for (std::size_t begin = 0; begin < point_count; begin += capacity) {
        const std::size_t count = std::min(capacity, point_count - begin);
        const auto query = points.slice(0, begin, begin + count);
        std::vector<float> host_query;
        auto occupancy = tinytensor::Tensor::ones(
            {count}, tinytensor::Device::CUDA);
        auto observed = tinytensor::Tensor::zeros(
            {count}, tinytensor::Device::CUDA,
            tinytensor::DataType::Bool);
        for (std::size_t view_index = 0; view_index < scene.views.size(); ++view_index) {
            const mvs::MvsView& view = scene.views[view_index];
            const OccupancyResult view_result = cache
                ? rasterizer.evaluate_occupancy(*cache, view_index, query)
                : rasterizer.evaluate_occupancy(
                    model, query, camera_from_mvs_view(view), raster_options);
            tinytensor::Tensor possibly_foreground = view_result.inside;
            tinytensor::Tensor definitely_background =
                tinytensor::Tensor::zeros_like(view_result.inside);
            const std::size_t mask_pixels =
                static_cast<std::size_t>(view.width) * view.height;
            if (view.foreground_mask.size() == mask_pixels) {
                // Retain the established double-precision mask projection.
                // Unmasked refinement never downloads points between steps.
                if (host_query.empty()) host_query = query.to_vector();
                const float mask_denominator =
                    *std::max_element(
                        view.foreground_mask.begin(),
                        view.foreground_mask.end()) <= 1
                    ? 1.F
                    : 255.F;
                std::vector<float> sampled_mask(count, 0.F);
                for (std::size_t i = 0; i < count; ++i) {
                    const Eigen::Vector3d camera =
                        view.pose.transform_world_to_camera(
                            Eigen::Map<const mvs::Vec3f>(host_query.data() + 3 * i).cast<double>());
                    if (!(camera.z() > 0.0)) continue;
                    const float x = view.fx *
                        static_cast<float>(camera.x() / camera.z()) + view.cx;
                    const float y = view.fy *
                        static_cast<float>(camera.y() / camera.z()) + view.cy;
                    if (x < 0.F || y < 0.F ||
                        x > static_cast<float>(view.width - 1) ||
                        y > static_cast<float>(view.height - 1))
                        continue;
                    const auto x0 = static_cast<std::uint32_t>(std::floor(x));
                    const auto y0 = static_cast<std::uint32_t>(std::floor(y));
                    const auto x1 = std::min(x0 + 1, view.width - 1);
                    const auto y1 = std::min(y0 + 1, view.height - 1);
                    const float tx = x - static_cast<float>(x0);
                    const float ty = y - static_cast<float>(y0);
                    const auto alpha = [&](const std::uint32_t px,
                                           const std::uint32_t py) {
                        return static_cast<float>(view.foreground_mask[
                            static_cast<std::size_t>(py) * view.width + px]) /
                            mask_denominator;
                    };
                    sampled_mask[i] =
                        (1.F - ty) * ((1.F - tx) * alpha(x0, y0) +
                                      tx * alpha(x1, y0)) +
                        ty * ((1.F - tx) * alpha(x0, y1) +
                              tx * alpha(x1, y1));
                }
                auto mask = tinytensor::Tensor::from_vector(
                    sampled_mask, {count}, tinytensor::Device::CUDA);
                possibly_foreground = view_result.inside.logical_and(
                    mask.ge(mask_background_threshold));
                definitely_background = view_result.inside.logical_and(
                    mask.lt(mask_background_threshold));
            }
            occupancy = tinytensor::Tensor::where(
                possibly_foreground,
                occupancy.minimum(view_result.occupancy),
                tinytensor::Tensor::where(
                    definitely_background,
                    tinytensor::Tensor::zeros_like(occupancy), occupancy));
            observed = observed.logical_or(view_result.inside);
        }
        occupancy = tinytensor::Tensor::where(
            observed, occupancy, tinytensor::Tensor::zeros_like(occupancy));
        result.slice(0, begin, begin + count).copy_(occupancy);
    }
    return result;
}

std::vector<float> evaluate_global_occupancy(
    const GaussianModel& model, const mvs::MvsScene& scene,
    const std::vector<mvs::Vec3f>& points,
    const TrainingOptions& training_options,
    const std::size_t chunk_size, const float mask_background_threshold,
    OccupancyCache* cache = nullptr) {
    // CPU callers can have tens of millions of tetrahedral samples. Stream
    // the same query chunks instead of uploading a second full-size copy and
    // staging it through a power-of-two pinned host allocation.
    std::vector<float> result(points.size());
    const std::size_t capacity = std::max<std::size_t>(chunk_size, 1);
    const std::span<const mvs::Vec3f> all_points(points);
    for (std::size_t begin = 0; begin < points.size(); begin += capacity) {
        const std::size_t count = std::min(capacity, points.size() - begin);
        const auto values = evaluate_global_occupancy_device(
            model, scene, upload_points(all_points.subspan(begin, count)),
            training_options, chunk_size, mask_background_threshold, cache).to_vector();
        std::copy(values.begin(), values.end(), result.begin() + begin);
    }
    return result;
}

bool in_camera_frustum(
    const mvs::Vec3f& point, const mvs::MvsView& view) {
    const Eigen::Vector3d camera = view.pose.transform_world_to_camera(
        point.cast<double>());
    if (!(camera.z() > 0.0)) return false;
    const float x = view.fx * static_cast<float>(camera.x() / camera.z()) +
                    view.cx;
    const float y = view.fy * static_cast<float>(camera.y() / camera.z()) +
                    view.cy;
    return x >= 0.F && y >= 0.F && x < static_cast<float>(view.width) &&
           y < static_cast<float>(view.height);
}

std::vector<mvs::Vec3f> sample_seed_mesh(
    const mvs::Mesh& mesh, const mvs::MvsScene& scene,
    const std::size_t count, std::mt19937& random) {
    std::vector<double> weights(mesh.faces.size(), 0.0);
    const auto face_count = static_cast<std::ptrdiff_t>(mesh.faces.size());
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(static) if(face_count >= 1024)
#endif
    for (std::ptrdiff_t face_index = 0; face_index < face_count; ++face_index) {
        const Eigen::Vector3i face = mesh.faces[face_index];
        const mvs::Vec3f a = mesh.vertices[static_cast<std::size_t>(face.x())];
        const mvs::Vec3f b = mesh.vertices[static_cast<std::size_t>(face.y())];
        const mvs::Vec3f c = mesh.vertices[static_cast<std::size_t>(face.z())];
        const float area = 0.5F * (b - a).cross(c - a).norm();
        const mvs::Vec3f center = (a + b + c) / 3.F;
        float minimum_distance2 = std::numeric_limits<float>::infinity();
        for (const mvs::MvsView& view : scene.views)
            if (in_camera_frustum(center, view))
                minimum_distance2 = std::min(
                    minimum_distance2,
                    (center - view.pose.C.cast<float>()).squaredNorm());
        if (std::isfinite(minimum_distance2) && area > 0.F)
            weights[face_index] =
                static_cast<double>(area) /
                std::max(static_cast<double>(minimum_distance2), 1e-12);
    }
    if (std::accumulate(weights.begin(), weights.end(), 0.0) <= 0.0)
        throw std::runtime_error(
            "PAM seed mesh has no camera-visible surface area");
    std::discrete_distribution<std::size_t> select_face(
        weights.begin(), weights.end());
    std::uniform_real_distribution<float> uniform(0.F, 1.F);
    std::vector<mvs::Vec3f> points;
    points.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const Eigen::Vector3i face = mesh.faces[select_face(random)];
        const mvs::Vec3f& a =
            mesh.vertices[static_cast<std::size_t>(face.x())];
        const mvs::Vec3f& b =
            mesh.vertices[static_cast<std::size_t>(face.y())];
        const mvs::Vec3f& c =
            mesh.vertices[static_cast<std::size_t>(face.z())];
        // Same triangular parameterization used by GaussianWrapping.
        const float u = uniform(random);
        const float v = uniform(random) * (1.F - u);
        points.push_back(u * a + v * b + (1.F - u - v) * c);
    }
    return points;
}

std::vector<std::size_t> sample_gaussian_indices(
    const HostGaussianField& field, const std::size_t requested,
    std::mt19937& random) {
    struct WeightedIndex {
        double key{};
        std::size_t index{};
    };
    std::vector<WeightedIndex> weighted;
    weighted.reserve(field.means.size());
    std::uniform_real_distribution<double> uniform(
        std::numeric_limits<double>::min(), 1.0);
    for (std::size_t index = 0; index < field.means.size(); ++index) {
        const float normal2 = field.normals[index].squaredNorm();
        const float opacity = field.opacities[index];
        const float minimum_scale = field.scales[index].minCoeff();
        const float maximum_scale = field.scales[index].maxCoeff();
        if (!(normal2 > 1e-8F) || !(opacity > 0.01F) ||
            !(minimum_scale > 0.F) || !std::isfinite(maximum_scale))
            continue;
        // Weighted sampling without replacement.  Anisotropic Gaussians are
        // characteristic of wires/edges, while opacity prevents transparent
        // noise from consuming the detail budget.
        const double anisotropy = std::clamp(
            static_cast<double>(maximum_scale / minimum_scale), 1.0, 100.0);
        const double weight =
            std::max(1e-8, static_cast<double>(opacity) * std::sqrt(anisotropy));
        weighted.push_back({std::log(uniform(random)) / weight, index});
    }
    const std::size_t count = std::min(requested, weighted.size());
    if (count == 0) return {};
    if (count < weighted.size()) {
        std::nth_element(
            weighted.begin(), weighted.begin() +
                static_cast<std::ptrdiff_t>(count), weighted.end(),
            [](const WeightedIndex& left, const WeightedIndex& right) {
                return left.key > right.key;
            });
        weighted.resize(count);
    }

    std::vector<std::size_t> selected;
    selected.reserve(count);
    for (const WeightedIndex& item : weighted)
        selected.push_back(item.index);
    return selected;
}

[[nodiscard]] float gaussian_normal_sigma(
    const HostGaussianField& field, const std::size_t index,
    const mvs::Vec3f& normal) {
    const mvs::Vec3f local_normal =
        field.rotations[index].transpose() * normal;
    const float sigma2 =
        (local_normal.array().square() *
         field.scales[index].array().square()).sum();
    return sigma2 > 0.F && std::isfinite(sigma2)
        ? std::sqrt(sigma2)
        : 0.F;
}

std::vector<mvs::Vec3f> sample_gaussian_detail_seeds(
    const HostGaussianField& field, const std::size_t requested,
    std::mt19937& random) {
    const std::vector<std::size_t> selected = sample_gaussian_indices(
        field, requested, random);
    std::vector<mvs::Vec3f> seeds;
    seeds.reserve(selected.size());
    for (const std::size_t index : selected) {
        const mvs::Vec3f normal = field.normals[index].normalized();
        const float sigma = gaussian_normal_sigma(field, index, normal);
        if (!(sigma > 0.F)) continue;
        // Start on the learned outward side of the Gaussian.  Half a standard
        // deviation is close enough for the occupancy/vector-field projection
        // but avoids the zero gradient exactly at the Gaussian mean.
        seeds.push_back(field.means[index] + 0.5F * sigma * normal);
    }
    return seeds;
}

void compute_vertex_normals(mvs::Mesh& mesh) {
    mesh.normals.assign(mesh.vertices.size(), mvs::Vec3f::Zero());
    for (const Eigen::Vector3i& face : mesh.faces) {
        const mvs::Vec3f cross =
            (mesh.vertices[static_cast<std::size_t>(face.y())] -
             mesh.vertices[static_cast<std::size_t>(face.x())])
                .cross(
                    mesh.vertices[static_cast<std::size_t>(face.z())] -
                    mesh.vertices[static_cast<std::size_t>(face.x())]);
        for (int corner = 0; corner < 3; ++corner)
            mesh.normals[static_cast<std::size_t>(face(corner))] += cross;
    }
    for (mvs::Vec3f& normal : mesh.normals)
        if (normal.squaredNorm() > 1e-20F) normal.normalize();
}

#if defined(PHOTARA_HAS_CGAL)
struct PivotEdge {
    std::size_t a{};
    std::size_t b{};

    [[nodiscard]] bool operator==(const PivotEdge&) const noexcept = default;
};

struct PivotEdgeHash {
    [[nodiscard]] std::size_t operator()(const PivotEdge& edge) const noexcept {
        const std::size_t seed = std::hash<std::size_t>{}(edge.a);
        return seed ^ (std::hash<std::size_t>{}(edge.b) +
            0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
    }
};

mvs::Mesh build_gaussian_pivot_seed_mesh(
    const GaussianModel& model, const mvs::MvsScene& scene,
    const TrainingOptions& training_options, const HostGaussianField& field,
    const PamMeshOptions& options, std::mt19937& random, OccupancyCache* cache) {
    const std::size_t gaussian_budget = std::max<std::size_t>(
        options.pivot_max_points / 2, 2);
    const std::vector<std::size_t> selected = sample_gaussian_indices(
        field, gaussian_budget, random);
    std::vector<mvs::Vec3f> pivots;
    std::vector<float> pivot_scales;
    pivots.reserve(2 * selected.size());
    pivot_scales.reserve(2 * selected.size());
    for (const std::size_t index : selected) {
        const mvs::Vec3f normal = field.normals[index].normalized();
        const float sigma = gaussian_normal_sigma(field, index, normal);
        if (!(sigma > 0.F)) continue;
        const mvs::Vec3f offset = field.means[index] +
            options.pivot_std_factor * sigma * normal;
        const float edge_scale = 3.F * field.scales[index].maxCoeff();
        pivots.push_back(offset);
        pivot_scales.push_back(edge_scale);
        pivots.push_back(field.means[index]);
        pivot_scales.push_back(edge_scale);
    }
    if (pivots.size() < 4)
        throw std::runtime_error(
            "Gaussian pivot generation produced fewer than four vertices");
    core::Logger::instance().info(
        "splat pivot tetra_triangulation: selected_gaussians=", selected.size(),
        " pivots=", pivots.size(),
        " std_factor=", options.pivot_std_factor);

    const std::vector<float> occupancy = evaluate_global_occupancy(
        model, scene, pivots, training_options,
        options.occupancy_chunk_size, options.mask_background_threshold, cache);

    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using VertexBase =
        CGAL::Triangulation_vertex_base_with_info_3<std::size_t, Kernel>;
    using CellBase = CGAL::Triangulation_cell_base_with_info_3<
        unsigned char, Kernel>;
    using DataStructure =
        CGAL::Triangulation_data_structure_3<VertexBase, CellBase>;
    using Delaunay =
        CGAL::Delaunay_triangulation_3<Kernel, DataStructure>;
    using Point = Kernel::Point_3;
    std::vector<std::pair<Point, std::size_t>> insertion;
    insertion.reserve(pivots.size());
    for (std::size_t index = 0; index < pivots.size(); ++index)
        insertion.emplace_back(
            Point(pivots[index].x(), pivots[index].y(), pivots[index].z()),
            index);
    Delaunay triangulation(insertion.begin(), insertion.end());
    if (!triangulation.is_valid() || triangulation.dimension() != 3)
        throw std::runtime_error(
            "Gaussian pivot tetra_triangulation failed");

    constexpr std::array<std::array<int, 7>, 16> triangle_edges{{
        {{-1,-1,-1,-1,-1,-1,-1}}, {{0,3,2,-1,-1,-1,-1}},
        {{0,1,4,-1,-1,-1,-1}}, {{1,4,2,2,4,3,-1}},
        {{1,2,5,-1,-1,-1,-1}}, {{0,3,5,0,5,1,-1}},
        {{0,2,5,0,5,4,-1}}, {{5,4,3,-1,-1,-1,-1}},
        {{3,4,5,-1,-1,-1,-1}}, {{4,5,0,5,2,0,-1}},
        {{1,5,0,5,3,0,-1}}, {{5,2,1,-1,-1,-1,-1}},
        {{3,4,2,2,4,1,-1}}, {{4,1,0,-1,-1,-1,-1}},
        {{2,3,0,-1,-1,-1,-1}}, {{-1,-1,-1,-1,-1,-1,-1}}
    }};
    constexpr std::array<std::array<int, 2>, 6> edge_corners{{
        {{0,1}}, {{1,2}}, {{2,0}}, {{0,3}}, {{1,3}}, {{2,3}}
    }};
    mvs::Mesh mesh;
    std::unordered_map<PivotEdge, int, PivotEdgeHash> edge_vertices;
    edge_vertices.reserve(triangulation.number_of_finite_cells());
    std::size_t rejected_large_edges = 0;
    for (auto cell = triangulation.finite_cells_begin();
         cell != triangulation.finite_cells_end(); ++cell) {
        std::array<std::size_t, 4> source{};
        unsigned configuration = 0;
        for (int corner = 0; corner < 4; ++corner) {
            source[static_cast<std::size_t>(corner)] =
                cell->vertex(corner)->info();
            if (occupancy[source[static_cast<std::size_t>(corner)]] >
                options.occupancy_iso_value)
                configuration |= 1U << static_cast<unsigned>(corner);
        }
        const auto& edges = triangle_edges[configuration];
        for (int triangle = 0; triangle < 6 && edges[triangle] >= 0;
             triangle += 3) {
            Eigen::Vector3i face;
            bool valid = true;
            for (int vertex = 0; vertex < 3; ++vertex) {
                const auto corners = edge_corners[
                    static_cast<std::size_t>(edges[triangle + vertex])];
                std::size_t a = source[static_cast<std::size_t>(corners[0])];
                std::size_t b = source[static_cast<std::size_t>(corners[1])];
                if (a > b) std::swap(a, b);
                const PivotEdge key{a, b};
                auto found = edge_vertices.find(key);
                if (found == edge_vertices.end()) {
                    if ((pivots[a] - pivots[b]).norm() >
                        pivot_scales[a] + pivot_scales[b]) {
                        ++rejected_large_edges;
                        valid = false;
                        break;
                    }
                    const float denominator = occupancy[b] - occupancy[a];
                    const float t = std::abs(denominator) > 1e-8F
                        ? std::clamp(
                            (options.occupancy_iso_value - occupancy[a]) /
                                denominator,
                            0.F, 1.F)
                        : 0.5F;
                    const int index = static_cast<int>(mesh.vertices.size());
                    mesh.vertices.push_back(
                        (1.F - t) * pivots[a] + t * pivots[b]);
                    found = edge_vertices.emplace(key, index).first;
                }
                face(vertex) = found->second;
            }
            if (valid && face.x() != face.y() && face.y() != face.z() &&
                face.z() != face.x())
                mesh.faces.push_back(face);
        }
    }
    if (mesh.faces.empty())
        throw std::runtime_error(
            "Gaussian pivot marching tetrahedra produced no surface");
    compute_vertex_normals(mesh);
    core::Logger::instance().info(
        "splat pivot mesh: tetrahedra=",
        triangulation.number_of_finite_cells(),
        " vertices=", mesh.vertices.size(),
        " faces=", mesh.faces.size(),
        " rejected_large_edges=", rejected_large_edges);
    return mesh;
}
#endif

}  // namespace

detail::PamGpuField detail::make_pam_gpu_field(const GaussianModel& model) {
    return HostGaussianField(model).upload();
}

std::vector<float> detail::pam_field_gradients_reference(
    const GaussianModel& model, const std::vector<float>& points, unsigned neighbors) {
    if (points.size() % 3 != 0 || neighbors == 0)
        throw std::invalid_argument("Invalid PAM reference query");
    HostGaussianField field(model);
    std::vector<float> result;
    result.reserve(points.size());
    for (std::size_t i = 0; i < points.size(); i += 3) {
        const auto gradient = field.gradient(mvs::Vec3f(points[i], points[i+1], points[i+2]), neighbors);
        result.insert(result.end(), gradient.data(), gradient.data() + 3);
    }
    return result;
}

PamMeshResult extract_pam_mesh(
    const GaussianModel& model, const mvs::MvsScene& scene,
    const mvs::Mesh& seed_mesh,
    const TrainingOptions& training_options,
    const PamMeshOptions& options) {
#if !defined(PHOTARA_HAS_CGAL)
    (void)model;
    (void)scene;
    (void)seed_mesh;
    (void)training_options;
    (void)options;
    throw std::runtime_error(
        "PAM requires a CGAL-enabled Photara build");
#else
    if (model.size() == 0 || !model.normal_features.is_valid())
        throw std::invalid_argument(
            "PAM requires Gaussians with normal-field features");
    if (model.normal_features.shape() !=
        tinytensor::TensorShape{model.size(), std::size_t{4}})
        throw std::invalid_argument(
            "PAM normal-field features must have shape [N, 4]");
    if (scene.views.empty())
        throw std::invalid_argument("PAM requires registered cameras");
    if ((seed_mesh.vertices.empty()) != (seed_mesh.faces.empty()))
        throw std::invalid_argument(
            "PAM seed mesh must be complete or entirely omitted");
    if (options.max_points < 4 || options.pivot_max_points < 4 ||
        options.vector_field_neighbors == 0 ||
        options.points_per_tetrahedron == 0 ||
        !std::isfinite(options.pivot_std_factor) ||
        !(options.pivot_std_factor > 0.F) ||
        !std::isfinite(options.gaussian_seed_fraction) ||
        options.gaussian_seed_fraction < 0.F ||
        options.gaussian_seed_fraction > 1.F ||
        !(options.occupancy_iso_value > 0.F &&
          options.occupancy_iso_value < 1.F) ||
        options.vacancy_threshold < 0.F)
        throw std::invalid_argument("Invalid PAM extraction options");

    core::StageScope stage("splat.pam");
    HostGaussianField field(model);
    const bool use_gpu = options.gpu_refinement &&
        options.vector_field_neighbors <= detail::pam_gpu_max_neighbors;
    const detail::PamGpuField gpu_field = use_gpu ? field.upload() : detail::PamGpuField{};
    core::Logger::instance().info("splat PAM field backend=", use_gpu ? "cuda" : "cpu");
    std::vector<Camera> occupancy_cameras;
    occupancy_cameras.reserve(scene.views.size());
    for (const auto& view : scene.views)
        occupancy_cameras.push_back(camera_from_mvs_view(view));
    RasterizeOptions occupancy_options;
    occupancy_options.kernel_size = training_options.kernel_size;
    occupancy_options.scale_modifier = training_options.scale_modifier;
    auto occupancy_cache = Rasterizer().prepare_occupancy_cache(
        model, occupancy_cameras, occupancy_options);
    std::mt19937 random(options.seed);
    mvs::Mesh generated_seed_mesh;
    const mvs::Mesh* active_seed_mesh = &seed_mesh;
    if (seed_mesh.vertices.empty()) {
        generated_seed_mesh = build_gaussian_pivot_seed_mesh(
            model, scene, training_options, field, options, random, &occupancy_cache);
        active_seed_mesh = &generated_seed_mesh;
    }
    std::vector<mvs::Vec3f> candidates;
    candidates.reserve(options.max_points);
    const auto refine_and_append =
        [&](std::vector<mvs::Vec3f> sampled) {
            if (sampled.empty()) return;
            core::StageScope refinement_stage("splat.pam.refinement");
            if (use_gpu) {
                auto points = upload_points(sampled);
                for (unsigned step = 0; step < options.refinement_steps; ++step) {
                    core::Logger::instance().info(
                        "splat PAM refinement: step=", step + 1, '/', options.refinement_steps,
                        " samples=", sampled.size(), " backend=cuda");
                    const auto occupancy = evaluate_global_occupancy_device(
                        model, scene, points, training_options, options.occupancy_chunk_size,
                        options.mask_background_threshold, &occupancy_cache);
                    detail::pam_refine_points(gpu_field, points, occupancy,
                        options.vector_field_neighbors, options.occupancy_iso_value,
                        options.minimum_gradient_norm_squared, options.refinement_step);
                }
                const auto occupancy = evaluate_global_occupancy_device(
                    model, scene, points, training_options, options.occupancy_chunk_size,
                    options.mask_background_threshold, &occupancy_cache).to_vector();
                const auto host_points = points.to_vector();
                for (std::size_t i = 0; i < sampled.size() && candidates.size() < options.max_points; ++i)
                    if (std::abs(occupancy[i] - options.occupancy_iso_value) <= options.vacancy_threshold)
                        candidates.emplace_back(host_points[3*i], host_points[3*i+1], host_points[3*i+2]);
                return;
            }
            const auto sample_count = static_cast<std::ptrdiff_t>(sampled.size());
            for (unsigned step = 0; step < options.refinement_steps; ++step) {
                core::Logger::instance().info(
                    "splat PAM refinement: step=", step + 1, '/',
                    options.refinement_steps, " samples=", sampled.size());
                const std::vector<float> occupancy = evaluate_global_occupancy(
                    model, scene, sampled, training_options,
                    options.occupancy_chunk_size,
                    options.mask_background_threshold, &occupancy_cache);
#if defined(PHOTARA_HAS_OPENMP)
#pragma omp parallel for schedule(dynamic, 256) if(sample_count >= 1024)
#endif
                for (std::ptrdiff_t index = 0; index < sample_count; ++index) {
                    const mvs::Vec3f gradient = field.gradient(
                        sampled[index], options.vector_field_neighbors);
                    const float norm_squared = gradient.squaredNorm();
                    if (!(norm_squared >
                          options.minimum_gradient_norm_squared))
                        continue;
                    const float alpha = std::clamp(
                        (occupancy[index] - options.occupancy_iso_value) /
                            norm_squared,
                        -1.F, 1.F);
                    sampled[index] +=
                        options.refinement_step * alpha * gradient;
                }
            }
            const std::vector<float> occupancy = evaluate_global_occupancy(
                model, scene, sampled, training_options,
                options.occupancy_chunk_size,
                options.mask_background_threshold, &occupancy_cache);
            for (std::size_t index = 0;
                 index < sampled.size() &&
                 candidates.size() < options.max_points;
                 ++index)
                if (std::abs(
                        occupancy[index] - options.occupancy_iso_value) <=
                        options.vacancy_threshold)
                    candidates.push_back(sampled[index]);
        };

    const std::size_t gaussian_seed_count =
        (std::min)(
            model.size(),
            static_cast<std::size_t>(std::llround(
                static_cast<double>(options.max_points) *
                options.gaussian_seed_fraction)));
    refine_and_append(sample_gaussian_detail_seeds(
        field, gaussian_seed_count, random));
    const std::size_t gaussian_candidates = candidates.size();
    const unsigned rounds = std::max(options.max_resample_rounds, 1U);
    for (unsigned round = 0;
         round < rounds && candidates.size() < options.max_points; ++round) {
        core::Logger::instance().info(
            "splat PAM sampling: round=", round + 1, '/', rounds,
            " candidate_cap=", options.max_points,
            " accepted=", candidates.size());
        const std::size_t missing = options.max_points - candidates.size();
        const std::size_t sample_count = missing *
            std::max(options.oversampling_factor, 1U);
        refine_and_append(sample_seed_mesh(
            *active_seed_mesh, scene, sample_count, random));
    }
    core::Logger::instance().info(
        "splat PAM seeds: gaussian_requested=", gaussian_seed_count,
        " gaussian_accepted=", gaussian_candidates,
        " mesh_accepted=", candidates.size() - gaussian_candidates);
    if (candidates.size() < 4)
        throw std::runtime_error(
            "PAM occupancy refinement produced fewer than four candidates");

    using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
    using VertexBase =
        CGAL::Triangulation_vertex_base_with_info_3<std::size_t, Kernel>;
    using CellBase =
        CGAL::Triangulation_cell_base_with_info_3<unsigned char, Kernel>;
    using DataStructure =
        CGAL::Triangulation_data_structure_3<VertexBase, CellBase>;
    using Delaunay =
        CGAL::Delaunay_triangulation_3<Kernel, DataStructure>;
    using Point = Kernel::Point_3;
    std::vector<std::pair<Point, std::size_t>> insertion;
    insertion.reserve(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index)
        insertion.emplace_back(
            Point(
                candidates[index].x(), candidates[index].y(),
                candidates[index].z()),
            index);
    Delaunay triangulation;
    {
        core::StageScope triangulation_stage("splat.pam.delaunay");
        triangulation.insert(insertion.begin(), insertion.end());
    }
    if (!triangulation.is_valid() || triangulation.dimension() != 3)
        throw std::runtime_error("PAM Delaunay triangulation failed");

    std::vector<Delaunay::Cell_handle> cells;
    cells.reserve(triangulation.number_of_finite_cells());
    std::uniform_real_distribution<float> uniform(0.F, 1.F);
    const std::size_t cell_chunk = std::max<std::size_t>(
        1, options.occupancy_chunk_size /
               options.points_per_tetrahedron);
    std::size_t occupied_count = 0;
    {
        core::StageScope occupancy_stage("splat.pam.cell_occupancy");
        for (auto begin = triangulation.finite_cells_begin();
             begin != triangulation.finite_cells_end();) {
            cells.clear();
            std::vector<mvs::Vec3f> queries;
            queries.reserve(cell_chunk * options.points_per_tetrahedron);
            for (; begin != triangulation.finite_cells_end() &&
                   cells.size() < cell_chunk;
                 ++begin) {
                Delaunay::Cell_handle cell = begin;
                cells.push_back(cell);
                std::array<mvs::Vec3f, 4> vertices;
                for (int corner = 0; corner < 4; ++corner)
                    vertices[static_cast<std::size_t>(corner)] =
                        candidates[cell->vertex(corner)->info()];
                for (unsigned sample = 0;
                     sample < options.points_per_tetrahedron; ++sample) {
                    if (options.points_per_tetrahedron == 1) {
                        queries.push_back(
                            0.25F * (vertices[0] + vertices[1] + vertices[2] +
                                     vertices[3]));
                    } else {
                        std::array<float, 4> weights{
                            uniform(random), uniform(random), uniform(random),
                            uniform(random)};
                        const float sum = std::accumulate(
                            weights.begin(), weights.end(), 0.F);
                        mvs::Vec3f query = mvs::Vec3f::Zero();
                        for (int corner = 0; corner < 4; ++corner)
                            query += weights[static_cast<std::size_t>(corner)] /
                                     sum * vertices[static_cast<std::size_t>(corner)];
                        queries.push_back(query);
                    }
                }
            }
            const std::vector<float> occupancy = evaluate_global_occupancy(
                model, scene, queries, training_options,
                options.occupancy_chunk_size,
                options.mask_background_threshold, &occupancy_cache);
            for (std::size_t cell_index = 0; cell_index < cells.size();
                 ++cell_index) {
                float mean = 0.F;
                for (unsigned sample = 0;
                     sample < options.points_per_tetrahedron; ++sample)
                    mean += occupancy[
                        cell_index * options.points_per_tetrahedron + sample];
                mean /= static_cast<float>(options.points_per_tetrahedron);
                cells[cell_index]->info() =
                    mean > options.occupancy_iso_value ? 1U : 0U;
                occupied_count += cells[cell_index]->info() != 0U;
            }
        }
    }

    PamMeshResult result;
    result.seed_mesh = std::move(generated_seed_mesh);
    result.tetrahedron_count = triangulation.number_of_finite_cells();
    result.occupied_tetrahedron_count = occupied_count;
    std::vector<int> remap(candidates.size(), -1);
    auto map_vertex = [&](const std::size_t source) {
        int& mapped = remap[source];
        if (mapped < 0) {
            mapped = static_cast<int>(result.mesh.vertices.size());
            result.mesh.vertices.push_back(candidates[source]);
        }
        return mapped;
    };
    for (auto cell = triangulation.finite_cells_begin();
         cell != triangulation.finite_cells_end(); ++cell) {
        if (cell->info() == 0U) continue;
        for (int opposite = 0; opposite < 4; ++opposite) {
            const auto neighbor = cell->neighbor(opposite);
            if (!triangulation.is_infinite(neighbor) &&
                neighbor->info() != 0U)
                continue;
            std::array<int, 3> corners{};
            int cursor = 0;
            for (int corner = 0; corner < 4; ++corner)
                if (corner != opposite) corners[cursor++] = corner;
            const mvs::Vec3f interior =
                candidates[cell->vertex(opposite)->info()];
            const mvs::Vec3f a =
                candidates[cell->vertex(corners[0])->info()];
            const mvs::Vec3f b =
                candidates[cell->vertex(corners[1])->info()];
            const mvs::Vec3f c =
                candidates[cell->vertex(corners[2])->info()];
            if ((b - a).cross(c - a).dot(interior - (a + b + c) / 3.F) >
                0.F)
                std::swap(corners[1], corners[2]);
            result.mesh.faces.emplace_back(
                map_vertex(cell->vertex(corners[0])->info()),
                map_vertex(cell->vertex(corners[1])->info()),
                map_vertex(cell->vertex(corners[2])->info()));
        }
    }
    if (result.mesh.faces.empty())
        throw std::runtime_error("PAM produced no occupied boundary surface");
    // Independently classified Delaunay cells can alternate occupied/empty
    // around an edge. Their raw boundary is closed, but several surface sheets
    // can then share an edge. CGAL preserves every triangle while duplicating
    // only the singular vertices required to turn the soup into an orientable,
    // combinatorially manifold surface. This matters for bicycle spokes and
    // other thin components that an area/component filter could otherwise lose.
    std::vector<Point> soup_points;
    soup_points.reserve(result.mesh.vertices.size());
    for (const mvs::Vec3f& vertex : result.mesh.vertices)
        soup_points.emplace_back(vertex.x(), vertex.y(), vertex.z());
    std::vector<std::vector<std::size_t>> soup_faces;
    soup_faces.reserve(result.mesh.faces.size());
    for (const Eigen::Vector3i& face : result.mesh.faces)
        soup_faces.push_back({
            static_cast<std::size_t>(face.x()),
            static_cast<std::size_t>(face.y()),
            static_cast<std::size_t>(face.z())});
    const std::size_t vertices_before_orientation = soup_points.size();
    CGAL::Polygon_mesh_processing::orient_polygon_soup(
        soup_points, soup_faces);
    result.mesh.vertices.clear();
    result.mesh.vertices.reserve(soup_points.size());
    for (const Point& point : soup_points)
        result.mesh.vertices.emplace_back(
            static_cast<float>(CGAL::to_double(point.x())),
            static_cast<float>(CGAL::to_double(point.y())),
            static_cast<float>(CGAL::to_double(point.z())));
    result.mesh.faces.clear();
    result.mesh.faces.reserve(soup_faces.size());
    for (const std::vector<std::size_t>& face : soup_faces)
        result.mesh.faces.emplace_back(
            static_cast<int>(face[0]), static_cast<int>(face[1]),
            static_cast<int>(face[2]));
    const std::size_t topology_duplicated_vertices =
        soup_points.size() - vertices_before_orientation;
    compute_vertex_normals(result.mesh);
    const std::vector<float> candidate_gradients = use_gpu
        ? detail::pam_field_gradients(gpu_field, upload_points(candidates),
            options.vector_field_neighbors).to_vector()
        : std::vector<float>{};
    result.candidate_cloud.points.reserve(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto& point = candidates[i];
        mvs::DensePoint dense;
        dense.position = point;
        const mvs::Vec3f gradient = use_gpu
            ? mvs::Vec3f(candidate_gradients[3*i], candidate_gradients[3*i+1], candidate_gradients[3*i+2])
            : field.gradient(point, options.vector_field_neighbors);
        if (gradient.squaredNorm() > 1e-20F)
            dense.normal = gradient.normalized();
        dense.weight = 1.F;
        result.candidate_cloud.points.push_back(std::move(dense));
    }
    core::Logger::instance().info(
        "splat PAM: candidates=", candidates.size(),
        " tetrahedra=", result.tetrahedron_count,
        " occupied_tetrahedra=", result.occupied_tetrahedron_count,
        " vertices=", result.mesh.vertices.size(),
        " faces=", result.mesh.faces.size(),
        " topology_duplicated_vertices=", topology_duplicated_vertices);
    const auto cache_stats = Rasterizer().occupancy_cache_stats(occupancy_cache);
    core::Logger::instance().info(
        "splat PAM occupancy cache: prepared_views=", cache_stats.prepared_views,
        " queries=", cache_stats.queries,
        " device_bytes=", cache_stats.device_bytes,
        " host_bytes=", cache_stats.host_bytes);
    core::Logger::instance().info(
        "splat PAM occupancy preparations=", cache_stats.preparations);
    stage.finish();
    return result;
#endif
}

}  // namespace photara::splat
