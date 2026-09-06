#include "mvs/internal.hpp"

#include "core/logging.hpp"
#include "mvs/maxflow.hpp"
#include "parallel/thread_pool.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(AETHERSCAN_HAS_CGAL)
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Spatial_sort_traits_adapter_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/spatial_sort.h>
#endif

namespace aetherscan::mvs::detail {

#if defined(AETHERSCAN_HAS_CGAL)
namespace {

struct GlobalVertex {
    Vec3f position{Vec3f::Zero()};
    Vec3f normal{Vec3f::Zero()};
    Vec3f color{Vec3f::Zero()};
    std::vector<Index> views;
    std::vector<float> view_weights;
};

// Keep all CGAL predicates and graph-cut distances in a canonical local
// coordinate frame. Dense reconstructions frequently use georeferenced or
// otherwise offset world coordinates; subtracting the cloud centre and
// normalizing its span avoids losing the small Delaunay edges in float-valued
// energy calculations. The transform is a positive uniform similarity, so it
// cannot change the Delaunay topology.
struct WorkingTransform {
    Vec3f origin{Vec3f::Zero()};
    float scale{1.F};

    [[nodiscard]] Vec3f to_working(const Vec3f& point) const {
        return (point - origin) * scale;
    }
    [[nodiscard]] Vec3f to_world(const Vec3f& point) const {
        return point / scale + origin;
    }
};

WorkingTransform make_working_transform(
    const std::vector<GlobalVertex>& vertices) {
    Vec3f minimum = Vec3f::Constant(std::numeric_limits<float>::max());
    Vec3f maximum = Vec3f::Constant(std::numeric_limits<float>::lowest());
    for (const GlobalVertex& vertex : vertices) {
        minimum = minimum.cwiseMin(vertex.position);
        maximum = maximum.cwiseMax(vertex.position);
    }
    WorkingTransform transform;
    if (vertices.empty()) return transform;
    transform.origin = minimum + (maximum - minimum) * 0.5F;
    const float span = (maximum - minimum).maxCoeff();
    if (std::isfinite(span) && span > 1e-20F)
        transform.scale = 2.F / span;
    return transform;
}

std::vector<GlobalVertex> collect_global_vertices(
    const MvsScene& scene, const DensifyOptions& options) {
    std::vector<GlobalVertex> vertices;
    vertices.reserve(scene.dense_cloud.points.size());
    for (const DensePoint& point : scene.dense_cloud.points) {
        if (!point.position.allFinite() || point.views.empty())
            continue;
        GlobalVertex vertex;
        vertex.position = point.position;
        vertex.normal = point.normal;
        vertex.color = point.color;
        const float fallback_weight = std::max(
            point.weight /
                static_cast<float>(std::max<std::size_t>(1, point.views.size())),
            1e-3F);
        const bool has_view_weights =
            point.view_weights.size() == point.views.size();
        for (std::size_t index = 0; index < point.views.size(); ++index) {
            const Index view = point.views[index];
            if (view >= scene.views.size()) continue;
            const float raw_weight = has_view_weights
                ? point.view_weights[index]
                : fallback_weight;
            const float view_weight =
                std::isfinite(raw_weight) && raw_weight > 0.F
                ? std::max(raw_weight, 1e-3F)
                : fallback_weight;
            const auto existing =
                std::find(vertex.views.begin(), vertex.views.end(), view);
            if (existing == vertex.views.end()) {
                vertex.views.push_back(view);
                vertex.view_weights.push_back(view_weight);
            } else {
                const std::size_t target_index = static_cast<std::size_t>(
                    existing - vertex.views.begin());
                vertex.view_weights[target_index] += view_weight;
            }
        }
        if (vertex.views.empty()) continue;
        vertices.push_back(std::move(vertex));
    }
    if (options.mesh_max_points > 0 &&
        vertices.size() > options.mesh_max_points) {
        std::mt19937 random(7);
        std::shuffle(vertices.begin(), vertices.end(), random);
        vertices.resize(static_cast<std::size_t>(options.mesh_max_points));
    }
    core::Logger::instance().info(
        "mvs global mesh candidates=", vertices.size(),
        " source_points=", scene.dense_cloud.points.size());
    return vertices;
}

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;

struct VertexInfo {
    std::size_t index{std::numeric_limits<std::size_t>::max()};
};

struct CellInfo {
    int node{-1};
    float source{0.F};
    float sink{0.F};
    std::array<float, 4> facet{};
};

using VertexBase = CGAL::Triangulation_vertex_base_with_info_3<VertexInfo, Kernel>;
using CellBase = CGAL::Triangulation_cell_base_with_info_3<CellInfo, Kernel>;
using DataStructure =
    CGAL::Triangulation_data_structure_3<VertexBase, CellBase>;
using Delaunay = CGAL::Delaunay_triangulation_3<Kernel, DataStructure>;
using CellHandle = Delaunay::Cell_handle;
using VertexHandle = Delaunay::Vertex_handle;

Point to_point(const Vec3f& value) {
    return {static_cast<double>(value.x()), static_cast<double>(value.y()),
            static_cast<double>(value.z())};
}

Vec3f to_vec(const Point& point) {
    return {static_cast<float>(point.x()), static_cast<float>(point.y()),
            static_cast<float>(point.z())};
}

bool projects_near(
    const MvsScene& scene, const GlobalVertex& candidate,
    const Vec3f& existing, const float maximum_distance_pixels,
    const float relative_depth_threshold) {
    bool compared = false;
    const float maximum_distance_squared =
        maximum_distance_pixels * maximum_distance_pixels;
    for (const Index view_id : candidate.views) {
        if (view_id >= scene.views.size()) continue;
        const MvsView& view = scene.views[view_id];
        const Vec3f a = view.pose
                            .transform_world_to_camera(
                                candidate.position.cast<double>())
                            .cast<float>();
        const Vec3f b = view.pose
                            .transform_world_to_camera(existing.cast<double>())
                            .cast<float>();
        if (!(a.z() > 1e-8F) || !(b.z() > 1e-8F)) return false;
        compared = true;
        const float depth_scale = std::max({a.z(), b.z(), 1e-6F});
        if (std::abs(a.z() - b.z()) >
            std::max(relative_depth_threshold, 1e-5F) * depth_scale)
            return false;
        const float dx = view.fx * (a.x() / a.z() - b.x() / b.z());
        const float dy = view.fy * (a.y() / a.z() - b.y() / b.z());
        if (dx * dx + dy * dy > maximum_distance_squared) return false;
    }
    return compared;
}

void merge_observations(GlobalVertex& target, const GlobalVertex& source) {
    for (std::size_t source_index = 0; source_index < source.views.size();
         ++source_index) {
        const Index view = source.views[source_index];
        const auto existing =
            std::find(target.views.begin(), target.views.end(), view);
        const float weight = source_index < source.view_weights.size()
            ? source.view_weights[source_index]
            : 1e-3F;
        if (existing == target.views.end()) {
            target.views.push_back(view);
            target.view_weights.push_back(weight);
        } else {
            const std::size_t target_index =
                static_cast<std::size_t>(existing - target.views.begin());
            target.view_weights[target_index] += weight;
        }
    }
}

void build_projection_filtered_delaunay(
    std::vector<GlobalVertex> candidates, const MvsScene& scene,
    const DensifyOptions& options, const WorkingTransform& transform,
    Delaunay& triangulation,
    std::vector<GlobalVertex>& vertices) {
    vertices.clear();
    vertices.reserve(candidates.size());
    std::vector<Point> points;
    points.reserve(candidates.size());
    std::vector<std::ptrdiff_t> order(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        points.push_back(to_point(transform.to_working(candidates[index].position)));
        order[index] = static_cast<std::ptrdiff_t>(index);
    }
    using SearchTraits =
        CGAL::Spatial_sort_traits_adapter_3<Kernel, Point*>;
    CGAL::spatial_sort(
        order.begin(), order.end(),
        SearchTraits(points.data(), triangulation.geom_traits()));

    VertexHandle hint;
    for (const std::ptrdiff_t candidate_index : order) {
        GlobalVertex& candidate =
            candidates[static_cast<std::size_t>(candidate_index)];
        const Point& point = points[static_cast<std::size_t>(candidate_index)];
        if (options.mesh_dist_insert_px > 0.F &&
            triangulation.number_of_vertices() > 0) {
            const VertexHandle nearest = hint == VertexHandle()
                ? triangulation.nearest_vertex(point)
                : triangulation.nearest_vertex(point, hint->cell());
            const std::size_t nearest_index = nearest->info().index;
            if (nearest_index < vertices.size() &&
                projects_near(
                    scene, candidate, vertices[nearest_index].position,
                    options.mesh_dist_insert_px,
                    options.mesh_depth_diff_threshold)) {
                merge_observations(vertices[nearest_index], candidate);
                hint = nearest;
                continue;
            }
        }

        const std::size_t index = vertices.size();
        const VertexHandle inserted = hint == VertexHandle()
            ? triangulation.insert(point)
            : triangulation.insert(point, hint->cell());
        const std::size_t existing_index = inserted->info().index;
        if (existing_index < vertices.size()) {
            merge_observations(vertices[existing_index], candidate);
            hint = inserted;
            continue;
        }
        vertices.push_back(std::move(candidate));
        inserted->info().index = index;
        hint = inserted;
    }
    core::Logger::instance().info(
        "mvs global mesh projection_filtered_vertices=", vertices.size(),
        " dist_insert_px=", options.mesh_dist_insert_px);
}

// OpenMVS numbers cells in the spatial insertion order before allocating the
// max-flow graph. CGAL's cell-container order is fragmented by insertion and
// produces avoidable cache misses during every visibility walk and cut pass.
int assign_spatial_cell_nodes(Delaunay& triangulation) {
    std::size_t cell_count = 0;
    for (auto cell = triangulation.all_cells_begin();
         cell != triangulation.all_cells_end(); ++cell)
        ++cell_count;
    if (cell_count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Delaunay graph exceeds the supported node count");

    const std::size_t vertex_slots =
        static_cast<std::size_t>(triangulation.number_of_vertices()) + 1U;
    std::vector<std::size_t> keys;
    keys.reserve(cell_count);
    std::vector<std::size_t> offsets(vertex_slots + 1U, 0U);
    for (auto cell = triangulation.all_cells_begin();
         cell != triangulation.all_cells_end(); ++cell) {
        std::size_t key = 0;
        for (int vertex = 0; vertex < 4; ++vertex) {
            if (triangulation.is_infinite(cell->vertex(vertex))) continue;
            const std::size_t insertion = cell->vertex(vertex)->info().index;
            if (insertion != std::numeric_limits<std::size_t>::max())
                key = std::max(key, std::min(insertion + 1U, vertex_slots));
        }
        keys.push_back(key);
        ++offsets[key + 1U];
    }
    for (std::size_t i = 1; i < offsets.size(); ++i)
        offsets[i] += offsets[i - 1U];
    std::size_t cell_index = 0;
    for (auto cell = triangulation.all_cells_begin();
         cell != triangulation.all_cells_end(); ++cell) {
        cell->info() = {};
        cell->info().node = static_cast<int>(offsets[keys[cell_index++]]++);
    }
    return static_cast<int>(cell_count);
}

Vec3f cell_center(const Delaunay& triangulation, const CellHandle& cell) {
    Vec3f center = Vec3f::Zero();
    unsigned count = 0;
    for (int i = 0; i < 4; ++i) {
        if (triangulation.is_infinite(cell->vertex(i))) continue;
        center += to_vec(cell->vertex(i)->point());
        ++count;
    }
    if (count == 0) return Vec3f::Zero();
    center /= static_cast<float>(count);
    return center;
}

float plane_sphere_angle(
    const Delaunay& triangulation, const CellHandle& cell, const int opposite) {
    if (triangulation.is_infinite(cell)) return 1.F;
    std::array<Vec3f, 3> triangle{};
    const std::array<int, 3> oriented =
        oriented_tetrahedron_facet_vertices(opposite);
    for (std::size_t i = 0; i < oriented.size(); ++i)
        triangle[i] = to_vec(cell->vertex(oriented[i])->point());
    const Vec3f normal =
        (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]);
    if (normal.squaredNorm() <= 1e-20F) return 0.5F;
    const Point center = triangulation.geom_traits()
                             .construct_circumcenter_3_object()(
                                 cell->vertex(0)->point(),
                                 cell->vertex(1)->point(),
                                 cell->vertex(2)->point(),
                                 cell->vertex(3)->point());
    const Vec3f cotangent = to_vec(center) - triangle[0];
    if (cotangent.squaredNorm() <= 1e-20F) return 0.5F;
    return std::clamp(
        normal.dot(cotangent) /
            std::sqrt(normal.squaredNorm() * cotangent.squaredNorm()),
        -1.F, 1.F);
}

int neighbor_slot(const CellHandle& from, const CellHandle& to) {
    for (int i = 0; i < 4; ++i)
        if (from->neighbor(i) == to) return i;
    return -1;
}

float ray_facet_distance(
    const Delaunay& triangulation, const CellHandle& cell,
    const int opposite_vertex, const Vec3f& point,
    const Vec3f& unit_direction, const Vec3f& fallback_center) {
    if (opposite_vertex < 0 || opposite_vertex >= 4) return 0.F;
    std::array<Vec3f, 3> triangle;
    int index = 0;
    for (int vertex = 0; vertex < 4; ++vertex) {
        if (vertex == opposite_vertex) continue;
        if (triangulation.is_infinite(cell->vertex(vertex)))
            return (fallback_center - point).norm();
        triangle[static_cast<std::size_t>(index++)] =
            to_vec(cell->vertex(vertex)->point());
    }
    const Vec3f normal =
        (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]);
    const float denominator = std::abs(normal.dot(unit_direction));
    if (!(denominator > 1e-12F) || !std::isfinite(denominator))
        return (fallback_center - point).norm();
    const float distance =
        std::abs(normal.dot(point - triangle[0])) / denominator;
    return std::isfinite(distance)
        ? distance
        : (fallback_center - point).norm();
}

float estimate_sigma(const Delaunay& triangulation, const float multiplier) {
    std::vector<float> lengths;
    lengths.reserve(static_cast<std::size_t>(triangulation.number_of_finite_edges()));
    for (auto edge = triangulation.finite_edges_begin();
         edge != triangulation.finite_edges_end(); ++edge) {
        const CellHandle cell = edge->first;
        const Vec3f a = to_vec(cell->vertex(edge->second)->point());
        const Vec3f b = to_vec(cell->vertex(edge->third)->point());
        lengths.push_back((a - b).squaredNorm());
    }
    if (lengths.empty()) return 1e-3F;
    const std::size_t middle = lengths.size() / 2;
    std::nth_element(
        lengths.begin(), lengths.begin() + static_cast<std::ptrdiff_t>(middle),
        lengths.end());
    return std::max(
        std::sqrt(std::max(lengths[middle], 1e-16F)) * multiplier, 1e-6F);
}

std::vector<float> estimate_vertex_sigmas(
    const Delaunay& triangulation, const std::size_t vertex_count,
    const float global_sigma, const float multiplier) {
    std::vector<std::size_t> offsets(vertex_count + 1U, 0U);
    for (auto edge = triangulation.finite_edges_begin();
         edge != triangulation.finite_edges_end(); ++edge) {
        const CellHandle cell = edge->first;
        const std::size_t a = cell->vertex(edge->second)->info().index;
        const std::size_t b = cell->vertex(edge->third)->info().index;
        if (a < vertex_count) ++offsets[a + 1U];
        if (b < vertex_count) ++offsets[b + 1U];
    }
    for (std::size_t i = 1; i < offsets.size(); ++i)
        offsets[i] += offsets[i - 1U];
    std::vector<std::size_t> cursor = offsets;
    std::vector<float> lengths(offsets.back());
    for (auto edge = triangulation.finite_edges_begin();
         edge != triangulation.finite_edges_end(); ++edge) {
        const CellHandle cell = edge->first;
        const auto first = cell->vertex(edge->second);
        const auto second = cell->vertex(edge->third);
        const std::size_t a = first->info().index;
        const std::size_t b = second->info().index;
        const float length = (to_vec(first->point()) - to_vec(second->point())).norm();
        if (a < vertex_count) lengths[cursor[a]++] = length;
        if (b < vertex_count) lengths[cursor[b]++] = length;
    }

    const float minimum = global_sigma * 0.25F;
    const float maximum = global_sigma * 4.F;
    std::vector<float> sigma(vertex_count, global_sigma);
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        const std::size_t begin = offsets[vertex];
        const std::size_t end = offsets[vertex + 1U];
        if (begin == end) continue;
        const std::size_t middle = begin + (end - begin) / 2U;
        std::nth_element(
            lengths.begin() + static_cast<std::ptrdiff_t>(begin),
            lengths.begin() + static_cast<std::ptrdiff_t>(middle),
            lengths.begin() + static_cast<std::ptrdiff_t>(end));
        sigma[vertex] = std::clamp(
            lengths[middle] * multiplier, minimum, maximum);
    }
    return sigma;
}

struct CellAccum {
    float sink{0.F};
    std::array<float, 4> facet{};
};

float free_space_support(
    const CellHandle& cell, const std::vector<CellAccum>& weights) {
    if (cell == CellHandle()) return 0.F;
    float support = 0.F;
    for (int facet = 0; facet < 4; ++facet) {
        const CellHandle neighbor = cell->neighbor(facet);
        const int incoming = neighbor_slot(neighbor, cell);
        if (incoming < 0) continue;
        const int node = neighbor->info().node;
        if (node < 0 || static_cast<std::size_t>(node) >= weights.size())
            continue;
        support += weights[static_cast<std::size_t>(node)]
                       .facet[static_cast<std::size_t>(incoming)];
    }
    return support;
}

struct VisibilityAccum {
    std::unordered_map<std::uint64_t, float> weights;
    std::size_t rays{0};
};

struct WeakSurfaceStats {
    std::size_t rays{0};
    std::size_t measured{0};
    std::size_t candidates{0};
    std::size_t reinforced{0};
    std::size_t zero_sink{0};
    float maximum_multiplier{0.F};
};

constexpr std::uint64_t k_cell_weight_slots = 5;
constexpr std::uint64_t k_sink_weight_slot = 4;
constexpr std::size_t k_local_weight_flush_size = 65'536;

bool extract_surface(
    Delaunay& triangulation, const std::vector<GlobalVertex>& vertices,
    const MvsScene& scene, const DensifyOptions& options,
    const WorkingTransform& transform, Mesh& mesh) {
    const int node_count = assign_spatial_cell_nodes(triangulation);
    if (node_count == 0) return false;
    std::vector<Vec3f> cell_centers(static_cast<std::size_t>(node_count));
    for (auto cell = triangulation.all_cells_begin();
         cell != triangulation.all_cells_end(); ++cell)
        cell_centers[static_cast<std::size_t>(cell->info().node)] =
            cell_center(triangulation, cell);

    const float sigma = estimate_sigma(triangulation, options.mesh_k_sigma);
    const std::vector<float> vertex_sigmas = options.mesh_adaptive_sigma
        ? estimate_vertex_sigmas(
              triangulation, vertices.size(), sigma, options.mesh_k_sigma)
        : std::vector<float>{};
    core::Logger::instance().info(
        "mvs global mesh cells=", node_count, " sigma=", sigma);

    // The unbounded Delaunay cells represent known outside/free space.
    for (auto cell = triangulation.all_cells_begin();
         cell != triangulation.all_cells_end(); ++cell)
        if (triangulation.is_infinite(cell))
            cell->info().source = options.mesh_k_inf;

    // Camera locations are shared by many rays. Locate them once, both to
    // constrain known free space and to provide traversal hints.
    std::vector<Point> camera_points(
        scene.views.size(), Point(0.0, 0.0, 0.0));
    std::vector<CellHandle> camera_cells(scene.views.size());
    for (std::size_t id = 0; id < scene.views.size(); ++id) {
        camera_points[id] = to_point(
            transform.to_working(scene.views[id].pose.C.cast<float>()));
        camera_cells[id] = triangulation.locate(camera_points[id]);
        if (camera_cells[id] != CellHandle())
            camera_cells[id]->info().source = std::max(
                camera_cells[id]->info().source, options.mesh_k_inf);
    }

    std::vector<VertexHandle> surface_vertices;
    surface_vertices.reserve(
        static_cast<std::size_t>(triangulation.number_of_vertices()));
    for (auto vertex = triangulation.finite_vertices_begin();
         vertex != triangulation.finite_vertices_end(); ++vertex)
        surface_vertices.push_back(vertex);

    const unsigned thread_count = std::max(
        1U, parallel::resolve_thread_count(options.thread_count));
    std::vector<VisibilityAccum> worker_accums(thread_count);
    for (VisibilityAccum& accum : worker_accums)
        accum.weights.reserve(k_local_weight_flush_size);
    std::vector<CellAccum> cell_weights(static_cast<std::size_t>(node_count));
    std::mutex merge_mutex;
    const auto flush_weights = [&](VisibilityAccum& local) {
        if (local.weights.empty()) return;
        std::lock_guard lock(merge_mutex);
        for (const auto& [key, weight] : local.weights) {
            const std::size_t node =
                static_cast<std::size_t>(key / k_cell_weight_slots);
            const std::uint64_t slot = key % k_cell_weight_slots;
            if (slot == k_sink_weight_slot)
                cell_weights[node].sink += weight;
            else
                cell_weights[node].facet[static_cast<std::size_t>(slot)] +=
                    weight;
        }
        local.weights.clear();
    };
    const auto add_weight = [](VisibilityAccum& local, const int node,
                               const std::uint64_t slot, const float weight) {
        const std::uint64_t key =
            static_cast<std::uint64_t>(node) * k_cell_weight_slots + slot;
        local.weights[key] += weight;
    };

    parallel::parallel_for(
        surface_vertices.size(), thread_count,
        [&](const std::size_t surface_index, const unsigned worker_id) {
            const VertexHandle vertex = surface_vertices[surface_index];
            const std::size_t vertex_index = vertex->info().index;
            if (vertex_index >= vertices.size()) return;
            const GlobalVertex& sample = vertices[vertex_index];
            const Vec3f point = sample.position;
            const float sample_sigma = vertex_sigmas.empty()
                ? sigma
                : vertex_sigmas[vertex_index];
            const float sample_inverse_two_sigma_squared =
                0.5F / (sample_sigma * sample_sigma);
            VisibilityAccum& local = worker_accums[worker_id];
            for (std::size_t view_index = 0;
                 view_index < sample.views.size(); ++view_index) {
                const Index id = sample.views[view_index];
                if (id >= scene.views.size()) continue;
                const float alpha = view_index < sample.view_weights.size()
                    ? sample.view_weights[view_index]
                    : 1e-3F;
                const Vec3f camera = transform.to_working(
                    scene.views[id].pose.C.cast<float>());
                const Vec3f ray = point - camera;
                const float ray_length = ray.norm();
                if (!(ray_length > 1e-6F)) continue;
                const Vec3f direction = ray / ray_length;
                const Point surface_point = vertex->point();
                const CellHandle camera_cell = camera_cells[id];

                CellHandle previous;
                for (const CellHandle& cell :
                     triangulation.segment_traverser_cell_handles(
                         camera_points[id], surface_point, camera_cell)) {
                    if (previous != CellHandle()) {
                        const int facet = neighbor_slot(previous, cell);
                        if (facet >= 0) {
                            const int node = previous->info().node;
                            const float distance = ray_facet_distance(
                                triangulation, previous, facet, point,
                                direction,
                                cell_centers[static_cast<std::size_t>(node)]);
                            add_weight(
                                local, node, static_cast<std::uint64_t>(facet),
                                alpha *
                                    (1.F - std::exp(
                                               -distance * distance *
                                               sample_inverse_two_sigma_squared)));
                        }
                    }
                    previous = cell;
                }

                const Vec3f behind =
                    point + direction * sample_sigma *
                                std::max(options.mesh_k_behind, 1.F);
                const Point behind_point = to_point(behind);
                const CellHandle sink_cell =
                    triangulation.locate(behind_point, vertex->cell());
                if (sink_cell != CellHandle())
                    add_weight(
                        local, sink_cell->info().node, k_sink_weight_slot,
                        alpha);

                // Traversing behind->point weights the mirrored directed
                // facets, preserving the outside->inside orientation.
                previous = CellHandle();
                for (const CellHandle& cell :
                     triangulation.segment_traverser_cell_handles(
                         behind_point, surface_point, sink_cell)) {
                    if (previous != CellHandle()) {
                        const int facet = neighbor_slot(cell, previous);
                        if (facet >= 0) {
                            const int node = cell->info().node;
                            const float distance = ray_facet_distance(
                                triangulation, cell, facet, point, direction,
                                cell_centers[static_cast<std::size_t>(node)]);
                            add_weight(
                                local, node, static_cast<std::uint64_t>(facet),
                                alpha *
                                    (1.F - std::exp(
                                               -distance * distance *
                                               sample_inverse_two_sigma_squared)));
                        }
                    }
                    previous = cell;
                }
                ++local.rays;
                if (local.weights.size() >= k_local_weight_flush_size)
                    flush_weights(local);
            }
        });

    std::size_t rays = 0;
    for (VisibilityAccum& local : worker_accums) {
        flush_weights(local);
        rays += local.rays;
    }
    for (auto cell = triangulation.all_cells_begin();
         cell != triangulation.all_cells_end(); ++cell) {
        const CellAccum& weights =
            cell_weights[static_cast<std::size_t>(cell->info().node)];
        cell->info().sink = weights.sink;
        cell->info().facet = weights.facet;
    }
    core::Logger::instance().info(
        "mvs global mesh visibility_rays=", rays,
        " threads=", thread_count);

    if (options.mesh_use_free_space_support) {
        core::StageScope weak_stage("mvs.mesh_weak_surface");
        const auto measure_weak_surface =
            [&](const VertexHandle& vertex, const Vec3f& point,
                const Index id, const float sample_sigma,
                float& beta, float& gamma,
                CellHandle& endpoint_cell) {
                const float front_distance = sample_sigma *
                    std::max(options.mesh_k_free_space_front, 0.F);
                const float back_distance = sample_sigma *
                    std::max(options.mesh_k_free_space_back, 0.F);
                const float near_distance =
                    std::max(sample_sigma * 1e-4F, 1e-8F);
                const Vec3f camera = transform.to_working(
                    scene.views[id].pose.C.cast<float>());
                const Vec3f ray = point - camera;
                const float ray_length = ray.norm();
                if (!(ray_length > 1e-6F) || !(front_distance > 0.F) ||
                    !(back_distance > 0.F))
                    return false;
                const Vec3f direction = ray / ray_length;

                const Point front_near =
                    to_point(point - direction * near_distance);
                const Point front_end =
                    to_point(point - direction * front_distance);
                const CellHandle front_cell =
                    triangulation.locate(front_near, vertex->cell());
                if (front_cell == CellHandle() ||
                    triangulation.is_infinite(front_cell))
                    return false;
                beta = 0.F;
                std::size_t beta_samples = 0;
                CellHandle previous;
                for (const CellHandle& cell :
                     triangulation.segment_traverser_cell_handles(
                         front_near, front_end, front_cell)) {
                    if (previous != CellHandle() &&
                        !triangulation.is_infinite(previous)) {
                        beta = std::max(
                            beta,
                            free_space_support(previous, cell_weights));
                        ++beta_samples;
                    }
                    previous = cell;
                }
                if (beta_samples == 0) return false;

                const Point back_near =
                    to_point(point + direction * near_distance);
                const Point back_end =
                    to_point(point + direction * back_distance);
                const CellHandle back_cell =
                    triangulation.locate(back_near, vertex->cell());
                if (back_cell == CellHandle() ||
                    triangulation.is_infinite(back_cell))
                    return false;
                float gamma_min = std::numeric_limits<float>::infinity();
                float gamma_max = 0.F;
                std::size_t gamma_samples = 0;
                previous = CellHandle();
                for (const CellHandle& cell :
                     triangulation.segment_traverser_cell_handles(
                         back_near, back_end, back_cell)) {
                    if (previous != CellHandle() &&
                        !triangulation.is_infinite(previous)) {
                        const float support =
                            free_space_support(previous, cell_weights);
                        gamma_min = std::min(gamma_min, support);
                        gamma_max = std::max(gamma_max, support);
                        ++gamma_samples;
                    }
                    previous = cell;
                }
                endpoint_cell = previous;
                if (gamma_samples == 0 || endpoint_cell == CellHandle() ||
                    triangulation.is_infinite(endpoint_cell))
                    return false;
                gamma = (gamma_min + gamma_max) * 0.5F;
                return std::isfinite(beta) && std::isfinite(gamma);
            };

        const auto quantile = [](std::vector<float> values, const float q) {
            if (values.empty()) return 0.F;
            const std::size_t index = std::min(
                values.size() - 1,
                static_cast<std::size_t>(
                    q * static_cast<float>(values.size() - 1)));
            std::nth_element(
                values.begin(),
                values.begin() + static_cast<std::ptrdiff_t>(index),
                values.end());
            return values[index];
        };

        // OpenMVS's fixed absolute beta/gamma thresholds assume its
        // Conf2Weight scale. Estimate the conversion from a deterministic ray
        // sample so fused AetherScan weights retain the same geometric test.
        std::vector<std::vector<std::array<float, 2>>> worker_samples(
            thread_count);
        for (auto& samples : worker_samples) samples.reserve(512);
        if (options.mesh_k_free_space_calibration_quantile > 0.F) {
            parallel::parallel_for(
                surface_vertices.size(), thread_count,
                [&](const std::size_t surface_index,
                    const unsigned worker_id) {
                    const VertexHandle vertex = surface_vertices[surface_index];
                    const std::size_t vertex_index = vertex->info().index;
                    if (vertex_index >= vertices.size()) return;
                    const GlobalVertex& sample = vertices[vertex_index];
                    for (const Index id : sample.views) {
                        if (id >= scene.views.size()) continue;
                        const std::uint64_t sample_hash =
                            static_cast<std::uint64_t>(surface_index) *
                                0x9E3779B97F4A7C15ULL ^
                            static_cast<std::uint64_t>(id) *
                                0xBF58476D1CE4E5B9ULL;
                        if ((sample_hash & 1023ULL) != 0ULL) continue;
                        float beta = 0.F;
                        float gamma = 0.F;
                        CellHandle endpoint_cell;
                        const float sample_sigma = vertex_sigmas.empty()
                            ? sigma
                            : vertex_sigmas[vertex_index];
                        if (measure_weak_surface(
                                vertex, sample.position, id, sample_sigma,
                                beta, gamma,
                                endpoint_cell))
                            worker_samples[worker_id].push_back({beta, gamma});
                    }
                });
        }

        std::vector<float> calibration_differences;
        std::vector<float> sampled_beta;
        std::vector<float> sampled_gamma;
        std::vector<float> sampled_ratio;
        std::size_t sample_count = 0;
        for (const auto& samples : worker_samples)
            sample_count += samples.size();
        calibration_differences.reserve(sample_count);
        sampled_beta.reserve(sample_count);
        sampled_gamma.reserve(sample_count);
        sampled_ratio.reserve(sample_count);
        for (const auto& samples : worker_samples) {
            for (const auto& sample : samples) {
                const float beta = sample[0];
                const float gamma = sample[1];
                const float ratio = beta > 0.F ? gamma / beta : 1.F;
                sampled_beta.push_back(beta);
                sampled_gamma.push_back(gamma);
                sampled_ratio.push_back(ratio);
                if (ratio < options.mesh_k_free_space_rel && beta > gamma)
                    calibration_differences.push_back(beta - gamma);
            }
        }
        const float calibration_difference = quantile(
            calibration_differences,
            std::clamp(
                options.mesh_k_free_space_calibration_quantile, 0.F,
                0.999F));
        const float support_scale = weak_surface_support_scale(
            calibration_difference, options);

        std::vector<WeakSurfaceStats> worker_stats(thread_count);

        parallel::parallel_for(
            surface_vertices.size(), thread_count,
            [&](const std::size_t surface_index, const unsigned worker_id) {
                const VertexHandle vertex = surface_vertices[surface_index];
                const std::size_t vertex_index = vertex->info().index;
                if (vertex_index >= vertices.size()) return;
                const GlobalVertex& sample = vertices[vertex_index];
                const Vec3f point = sample.position;
                const float sample_sigma = vertex_sigmas.empty()
                    ? sigma
                    : vertex_sigmas[vertex_index];
                WeakSurfaceStats& stats = worker_stats[worker_id];

                for (const Index id : sample.views) {
                    if (id >= scene.views.size()) continue;
                    ++stats.rays;
                    float beta = 0.F;
                    float gamma = 0.F;
                    CellHandle endpoint_cell;
                    if (!measure_weak_surface(
                            vertex, point, id, sample_sigma, beta, gamma,
                            endpoint_cell))
                        continue;
                    ++stats.measured;

                    const float multiplier = weak_surface_sink_multiplier(
                        beta / support_scale, gamma / support_scale, options);
                    if (!(multiplier > 0.F)) continue;
                    ++stats.candidates;
                    stats.maximum_multiplier =
                        std::max(stats.maximum_multiplier, multiplier);

                    const int endpoint_node = endpoint_cell->info().node;
                    if (endpoint_node < 0 ||
                        static_cast<std::size_t>(endpoint_node) >=
                            cell_weights.size())
                        continue;
                    float& sink_value =
                        cell_weights[static_cast<std::size_t>(endpoint_node)]
                            .sink;
                    std::atomic_ref<float> sink(sink_value);
                    float current = sink.load(std::memory_order_relaxed);
                    if (!(current > 0.F)) {
                        ++stats.zero_sink;
                        continue;
                    }
                    float desired = 0.F;
                    do {
                        desired = std::min(
                            current * multiplier, options.mesh_k_inf);
                    } while (!sink.compare_exchange_weak(
                        current, desired, std::memory_order_relaxed,
                        std::memory_order_relaxed));
                    ++stats.reinforced;
                }
            });

        WeakSurfaceStats total;
        for (const WeakSurfaceStats& stats : worker_stats) {
            total.rays += stats.rays;
            total.measured += stats.measured;
            total.candidates += stats.candidates;
            total.reinforced += stats.reinforced;
            total.zero_sink += stats.zero_sink;
            total.maximum_multiplier =
                std::max(total.maximum_multiplier, stats.maximum_multiplier);
        }
        for (auto cell = triangulation.all_cells_begin();
             cell != triangulation.all_cells_end(); ++cell)
            cell->info().sink =
                cell_weights[static_cast<std::size_t>(cell->info().node)].sink;
        core::Logger::instance().info(
            "mvs weak surface: rays=", total.rays,
            " measured=", total.measured,
            " candidates=", total.candidates,
            " reinforced=", total.reinforced,
            " zero_sink=", total.zero_sink,
            " max_multiplier=", total.maximum_multiplier,
            " support_scale=", support_scale,
            " calibration_difference=", calibration_difference);
        core::Logger::instance().info(
            "mvs weak surface sample: count=", sample_count,
            " beta_p50=", quantile(sampled_beta, 0.5F),
            " beta_p95=", quantile(sampled_beta, 0.95F),
            " gamma_p50=", quantile(sampled_gamma, 0.5F),
            " qualified_difference_p50=",
            quantile(calibration_differences, 0.5F),
            " qualified_difference_p95=",
            quantile(calibration_differences, 0.95F),
            " ratio_p50=", quantile(sampled_ratio, 0.5F),
            " ratio_p95=", quantile(sampled_ratio, 0.95F));
        weak_stage.finish();
    }

    maxflow::Graph graph(static_cast<std::size_t>(node_count));
    for (auto cell = triangulation.all_cells_begin();
         cell != triangulation.all_cells_end(); ++cell) {
        const int node = cell->info().node;
        graph.add_tweights(
            static_cast<std::size_t>(node), cell->info().source,
            std::min(cell->info().sink, options.mesh_k_inf));
        for (int facet = 0; facet < 4; ++facet) {
            const CellHandle adjacent = cell->neighbor(facet);
            if (adjacent->info().node < node)
                continue;
            const int back = neighbor_slot(adjacent, cell);
            const float quality =
                (1.F - std::min(
                           plane_sphere_angle(triangulation, cell, facet),
                           back >= 0
                               ? plane_sphere_angle(triangulation, adjacent, back)
                               : 1.F)) *
                options.mesh_k_qual;
            const float forward =
                cell->info().facet[static_cast<std::size_t>(facet)] +
                quality;
            const float reverse =
                (back >= 0
                     ? adjacent->info().facet[static_cast<std::size_t>(back)]
                     : 0.F) +
                quality;
            graph.add_edge(
                static_cast<std::size_t>(node),
                static_cast<std::size_t>(adjacent->info().node), forward, reverse);
        }
    }
    const float flow = graph.maxflow();
    core::Logger::instance().info("mvs global mesh maxflow=", flow);

    mesh = {};
    std::unordered_map<std::size_t, int> vertex_map;
    vertex_map.reserve(vertices.size());
    auto map_vertex = [&](const Delaunay::Vertex_handle& handle) {
        const std::size_t index = handle->info().index;
        auto [it, inserted] =
            vertex_map.emplace(index, static_cast<int>(vertex_map.size()));
        if (inserted) {
            mesh.vertices.push_back(transform.to_world(to_vec(handle->point())));
            mesh.colors.push_back(
                index < vertices.size() ? vertices[index].color
                                        : Vec3f{0.7F, 0.7F, 0.7F});
        }
        return it->second;
    };

    const auto facet_longest_edge_squared =
        [&](const CellHandle& cell, const int opposite) {
            std::array<Vec3f, 3> triangle{};
            int cursor = 0;
            for (int vertex = 0; vertex < 4; ++vertex)
                if (vertex != opposite)
                    triangle[static_cast<std::size_t>(cursor++)] =
                        to_vec(cell->vertex(vertex)->point());
            return std::max({
                (triangle[0] - triangle[1]).squaredNorm(),
                (triangle[1] - triangle[2]).squaredNorm(),
                (triangle[2] - triangle[0]).squaredNorm()});
        };
    float gate_edge_squared = std::numeric_limits<float>::max();
    float median_cut_edge = 0.F;
    if (options.mesh_max_edge_scale > 0.F) {
        std::vector<float> cut_edges;
        cut_edges.reserve(vertices.size() * 2U);
        for (auto facet = triangulation.finite_facets_begin();
             facet != triangulation.finite_facets_end(); ++facet) {
            const CellHandle first = facet->first;
            const int opposite = facet->second;
            const CellHandle second = first->neighbor(opposite);
            if (graph.is_source_side(
                    static_cast<std::size_t>(first->info().node)) ==
                graph.is_source_side(
                    static_cast<std::size_t>(second->info().node)))
                continue;
            cut_edges.push_back(facet_longest_edge_squared(first, opposite));
        }
        if (!cut_edges.empty()) {
            const std::size_t middle = cut_edges.size() / 2U;
            std::nth_element(
                cut_edges.begin(),
                cut_edges.begin() + static_cast<std::ptrdiff_t>(middle),
                cut_edges.end());
            median_cut_edge = std::sqrt(cut_edges[middle]);
            gate_edge_squared = cut_edges[middle] *
                options.mesh_max_edge_scale * options.mesh_max_edge_scale;
        }
    }

    std::size_t unsupported_facets_removed = 0;
    float longest_surface_edge = 0.F;
    for (auto facet = triangulation.finite_facets_begin();
         facet != triangulation.finite_facets_end(); ++facet) {
        const CellHandle first = facet->first;
        const int opposite = facet->second;
        const CellHandle second = first->neighbor(opposite);
        const bool first_source = graph.is_source_side(
            static_cast<std::size_t>(first->info().node));
        const bool second_source = graph.is_source_side(
            static_cast<std::size_t>(second->info().node));
        if (first_source == second_source) continue;
        const float longest_edge_squared =
            facet_longest_edge_squared(first, opposite);
        longest_surface_edge = std::max(
            longest_surface_edge, std::sqrt(longest_edge_squared));
        if (longest_edge_squared > gate_edge_squared) {
            ++unsupported_facets_removed;
            continue;
        }
        std::array<int, 3> ids{};
        const auto oriented = oriented_tetrahedron_facet_vertices(opposite);
        for (std::size_t i = 0; i < 3; ++i)
            ids[i] = map_vertex(first->vertex(oriented[i]));
        Eigen::Vector3i face{ids[0], ids[1], ids[2]};
        // CGAL facet parity gives the orientation even at the convex hull,
        // where an infinite cell has no geometric centroid.
        if (!first_source) std::swap(face[0], face[2]);
        mesh.faces.push_back(face);
    }
    core::Logger::instance().info(
        "mvs global mesh surface_faces=", mesh.faces.size(),
        " unsupported_facets_removed=", unsupported_facets_removed,
        " median_cut_edge=", median_cut_edge,
        " edge_limit=", median_cut_edge * options.mesh_max_edge_scale,
        " longest_edge=", longest_surface_edge);
    return !mesh.faces.empty();
}

}  // namespace
#endif

bool reconstruct_mesh_global_cgal(
    MvsScene& scene, const DensifyOptions& options) {
#if !defined(AETHERSCAN_HAS_CGAL)
    (void)scene;
    (void)options;
    core::Logger::instance().warning(
        "mvs global mesh unavailable: rebuild with CGAL");
    return false;
#else
    core::StageScope stage("mvs.mesh_global_cgal");
    std::vector<GlobalVertex> candidates =
        collect_global_vertices(scene, options);
    if (candidates.size() < 4) return false;
    const WorkingTransform transform = make_working_transform(candidates);
    Delaunay triangulation;
    std::vector<GlobalVertex> vertices;
    build_projection_filtered_delaunay(
        std::move(candidates), scene, options, transform, triangulation,
        vertices);
    core::Logger::instance().info(
        "mvs global mesh delaunay vertices=", triangulation.number_of_vertices(),
        " finite_cells=", triangulation.number_of_finite_cells());
    if (triangulation.dimension() != 3 ||
        triangulation.number_of_finite_cells() == 0)
        return false;

    // From this boundary onward all ray lengths and uncertainty estimates use
    // the same canonical frame as the tetrahedralization.
    for (GlobalVertex& vertex : vertices)
        vertex.position = transform.to_working(vertex.position);
    core::Logger::instance().info(
        "mvs global mesh canonical_origin=", transform.origin.transpose(),
        " canonical_scale=", transform.scale);

    const bool success =
        extract_surface(
            triangulation, vertices, scene, options, transform, scene.mesh);
    stage.finish();
    return success;
#endif
}

}  // namespace aetherscan::mvs::detail
