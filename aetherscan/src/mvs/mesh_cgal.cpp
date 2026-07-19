#include "mvs/internal.hpp"

#include "core/logging.hpp"
#include "mvs/maxflow.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(AETHERSCAN_HAS_CGAL)
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#endif

namespace aetherscan::mvs::detail {

#if defined(AETHERSCAN_HAS_CGAL)
namespace {

struct GlobalVertex {
    Vec3f position{Vec3f::Zero()};
    Vec3f normal{Vec3f::Zero()};
    Vec3f color{Vec3f::Zero()};
    float weight{0.F};
    std::vector<Index> views;
};

struct GridKey {
    std::int64_t x{};
    std::int64_t y{};
    std::int64_t z{};
    bool operator==(const GridKey&) const = default;
};

struct GridHash {
    std::size_t operator()(const GridKey& key) const noexcept {
        return (static_cast<std::size_t>(key.x) * 73856093u) ^
               (static_cast<std::size_t>(key.y) * 19349663u) ^
               (static_cast<std::size_t>(key.z) * 83492791u);
    }
};

float median_pixel_footprint(const MvsScene& scene) {
    std::vector<float> values;
    values.reserve(scene.dense_cloud.points.size() / 64 + 1);
    const std::size_t stride =
        std::max<std::size_t>(1, scene.dense_cloud.points.size() / 100000);
    for (std::size_t i = 0; i < scene.dense_cloud.points.size(); i += stride) {
        const DensePoint& point = scene.dense_cloud.points[i];
        for (const Index view_id : point.views) {
            if (view_id >= scene.views.size()) continue;
            const MvsView& view = scene.views[view_id];
            const Vec3f camera = view.pose
                                     .transform_world_to_camera(
                                         point.position.cast<double>())
                                     .cast<float>();
            if (camera.z() > 0.F)
                values.push_back(
                    camera.z() / std::max(std::min(view.fx, view.fy), 1.F));
            break;
        }
    }
    if (values.empty()) return 1e-3F;
    const std::size_t middle = values.size() / 2;
    std::nth_element(
        values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle),
        values.end());
    return std::max(values[middle], 1e-8F);
}

std::vector<GlobalVertex> downsample_global_vertices(
    const MvsScene& scene, const DensifyOptions& options) {
    if (scene.dense_cloud.points.empty()) return {};
    Vec3f minimum = scene.dense_cloud.points.front().position;
    Vec3f maximum = minimum;
    for (const DensePoint& point : scene.dense_cloud.points) {
        minimum = minimum.cwiseMin(point.position);
        maximum = maximum.cwiseMax(point.position);
    }
    const float diagonal = std::max((maximum - minimum).norm(), 1e-6F);
    const float footprint = median_pixel_footprint(scene);
    const float voxel = std::max(
        diagonal * 2e-4F,
        footprint * std::max(options.mesh_dist_insert_px, 0.5F));

    std::unordered_map<GridKey, GlobalVertex, GridHash> grid;
    grid.reserve(std::min<std::size_t>(
        scene.dense_cloud.points.size(), 1'000'000));
    for (const DensePoint& point : scene.dense_cloud.points) {
        if (!point.position.allFinite() || point.views.empty()) continue;
        const GridKey key{
            static_cast<std::int64_t>(std::floor(point.position.x() / voxel)),
            static_cast<std::int64_t>(std::floor(point.position.y() / voxel)),
            static_cast<std::int64_t>(std::floor(point.position.z() / voxel))};
        auto [it, inserted] = grid.try_emplace(key);
        GlobalVertex& vertex = it->second;
        if (inserted || point.weight > vertex.weight) {
            vertex.position = point.position;
            vertex.normal = point.normal;
            vertex.color = point.color;
            vertex.weight = point.weight;
        }
        for (const Index view : point.views) {
            if (std::find(vertex.views.begin(), vertex.views.end(), view) ==
                vertex.views.end())
                vertex.views.push_back(view);
        }
    }

    std::vector<GlobalVertex> vertices;
    vertices.reserve(grid.size());
    for (auto& [key, vertex] : grid) {
        (void)key;
        vertices.push_back(std::move(vertex));
    }
    if (options.mesh_max_points > 0 &&
        vertices.size() > options.mesh_max_points) {
        std::mt19937 random(7);
        std::shuffle(vertices.begin(), vertices.end(), random);
        vertices.resize(static_cast<std::size_t>(options.mesh_max_points));
    }
    core::Logger::instance().info(
        "mvs global mesh samples=", vertices.size(), " voxel=", voxel,
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

Point to_point(const Vec3f& value) {
    return {static_cast<double>(value.x()), static_cast<double>(value.y()),
            static_cast<double>(value.z())};
}

Vec3f to_vec(const Point& point) {
    return {static_cast<float>(point.x()), static_cast<float>(point.y()),
            static_cast<float>(point.z())};
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
    int cursor = 0;
    for (int i = 0; i < 4; ++i)
        if (i != opposite)
            triangle[static_cast<std::size_t>(cursor++)] =
                to_vec(cell->vertex(i)->point());
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

bool extract_surface(
    Delaunay& triangulation, const std::vector<GlobalVertex>& vertices,
    const MvsScene& scene, const DensifyOptions& options, Mesh& mesh) {
    int node_count = 0;
    for (auto cell = triangulation.all_cells_begin();
         cell != triangulation.all_cells_end(); ++cell)
        cell->info().node = node_count++;
    if (node_count == 0) return false;

    const float sigma = estimate_sigma(triangulation, options.mesh_k_sigma);
    const float inverse_two_sigma_squared = 0.5F / (sigma * sigma);
    core::Logger::instance().info(
        "mvs global mesh cells=", node_count, " sigma=", sigma);

    // The unbounded Delaunay cells represent known outside/free space.
    for (auto cell = triangulation.all_cells_begin();
         cell != triangulation.all_cells_end(); ++cell)
        if (triangulation.is_infinite(cell))
            cell->info().source = options.mesh_k_inf;

    std::size_t rays = 0;
    for (auto vertex = triangulation.finite_vertices_begin();
         vertex != triangulation.finite_vertices_end(); ++vertex) {
        const std::size_t vertex_index = vertex->info().index;
        if (vertex_index >= vertices.size()) continue;
        const GlobalVertex& sample = vertices[vertex_index];
        const Vec3f point = sample.position;
        const float alpha = std::max(sample.weight, 1e-3F);
        const std::size_t view_count = std::min<std::size_t>(sample.views.size(), 6);
        for (std::size_t view_index = 0; view_index < view_count; ++view_index) {
            const Index id = sample.views[view_index];
            if (id >= scene.views.size()) continue;
            const Vec3f camera = scene.views[id].pose.C.cast<float>();
            const Vec3f ray = point - camera;
            const float ray_length = ray.norm();
            if (!(ray_length > 1e-6F)) continue;
            const Vec3f direction = ray / ray_length;
            const Point camera_point = to_point(camera);
            const Point surface_point = vertex->point();
            CellHandle hint = triangulation.locate(camera_point);
            if (hint != CellHandle())
                hint->info().source = std::max(
                    hint->info().source, options.mesh_k_inf);

            CellHandle previous;
            for (const CellHandle& cell : triangulation.segment_traverser_cell_handles(
                     camera_point, surface_point, hint)) {
                if (previous != CellHandle()) {
                    const int facet = neighbor_slot(previous, cell);
                    if (facet >= 0) {
                        const float distance =
                            (cell_center(triangulation, previous) - point).norm();
                        previous->info().facet[static_cast<std::size_t>(facet)] +=
                            alpha *
                            (1.F - std::exp(
                                       -distance * distance *
                                       inverse_two_sigma_squared));
                    }
                }
                previous = cell;
            }

            const Vec3f behind = point + direction * sigma *
                                 std::max(options.mesh_k_behind, 1.F);
            CellHandle sink_cell = triangulation.locate(to_point(behind), vertex->cell());
            if (sink_cell != CellHandle())
                sink_cell->info().sink += alpha;

            // Weight the same outside->inside direction on facets behind the
            // sample. Traversing behind->point means the directed graph edge
            // is the mirror (current cell back to the previous cell).
            previous = CellHandle();
            for (const CellHandle& cell : triangulation.segment_traverser_cell_handles(
                     to_point(behind), surface_point, sink_cell)) {
                if (previous != CellHandle()) {
                    const int facet = neighbor_slot(cell, previous);
                    if (facet >= 0) {
                        const float distance =
                            (cell_center(triangulation, cell) - point).norm();
                        cell->info().facet[static_cast<std::size_t>(facet)] +=
                            alpha *
                            (1.F - std::exp(
                                       -distance * distance *
                                       inverse_two_sigma_squared));
                    }
                }
                previous = cell;
            }
            ++rays;
        }
    }
    core::Logger::instance().info("mvs global mesh visibility_rays=", rays);

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
            mesh.vertices.push_back(to_vec(handle->point()));
            mesh.colors.push_back(
                index < vertices.size() ? vertices[index].color
                                        : Vec3f{0.7F, 0.7F, 0.7F});
        }
        return it->second;
    };

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
        std::array<int, 3> ids{};
        int cursor = 0;
        for (int i = 0; i < 4; ++i)
            if (i != opposite) ids[static_cast<std::size_t>(cursor++)] =
                map_vertex(first->vertex(i));
        Eigen::Vector3i face{ids[0], ids[1], ids[2]};
        const float median_edge = sigma /
                                  std::max(options.mesh_k_sigma, 1e-6F);
        const float maximum_edge = median_edge *
                                   std::max(options.mesh_max_edge_voxels, 1.F);
        const Vec3f edge01 =
            mesh.vertices[static_cast<std::size_t>(face[1])] -
            mesh.vertices[static_cast<std::size_t>(face[0])];
        const Vec3f edge12 =
            mesh.vertices[static_cast<std::size_t>(face[2])] -
            mesh.vertices[static_cast<std::size_t>(face[1])];
        const Vec3f edge20 =
            mesh.vertices[static_cast<std::size_t>(face[0])] -
            mesh.vertices[static_cast<std::size_t>(face[2])];
        if (std::max({edge01.norm(), edge12.norm(), edge20.norm()}) >
            maximum_edge)
            continue;
        const Vec3f normal =
            (mesh.vertices[static_cast<std::size_t>(face[1])] -
             mesh.vertices[static_cast<std::size_t>(face[0])])
                .cross(mesh.vertices[static_cast<std::size_t>(face[2])] -
                       mesh.vertices[static_cast<std::size_t>(face[0])]);
        const Vec3f first_center = cell_center(triangulation, first);
        const Vec3f second_center = triangulation.is_infinite(second)
            ? first_center + normal
            : cell_center(triangulation, second);
        const Vec3f outside_to_inside = first_source
            ? second_center - first_center
            : first_center - second_center;
        if (normal.dot(outside_to_inside) > 0.F) std::swap(face[1], face[2]);
        mesh.faces.push_back(face);
    }
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
    std::vector<GlobalVertex> vertices =
        downsample_global_vertices(scene, options);
    if (vertices.size() < 4) return false;

    Delaunay triangulation;
    std::vector<std::pair<Point, VertexInfo>> points;
    points.reserve(vertices.size());
    for (std::size_t i = 0; i < vertices.size(); ++i)
        points.emplace_back(to_point(vertices[i].position), VertexInfo{i});
    triangulation.insert(points.begin(), points.end());
    core::Logger::instance().info(
        "mvs global mesh delaunay vertices=", triangulation.number_of_vertices(),
        " finite_cells=", triangulation.number_of_finite_cells());
    if (triangulation.dimension() != 3 ||
        triangulation.number_of_finite_cells() == 0)
        return false;

    const bool success =
        extract_surface(triangulation, vertices, scene, options, scene.mesh);
    stage.finish();
    return success;
#endif
}

}  // namespace aetherscan::mvs::detail
