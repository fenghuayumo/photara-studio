#include "mvs/internal.hpp"

#include "core/logging.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aetherscan::mvs::detail {
namespace {

struct EdgeKey {
    int a{};
    int b{};
    bool operator==(const EdgeKey&) const = default;
};

struct EdgeHash {
    std::size_t operator()(const EdgeKey& edge) const noexcept {
        return (static_cast<std::size_t>(edge.a) * 73856093u) ^
               (static_cast<std::size_t>(edge.b) * 19349663u);
    }
};

struct FaceKey {
    int a{};
    int b{};
    int c{};
    bool operator==(const FaceKey&) const = default;
};

struct FaceHash {
    std::size_t operator()(const FaceKey& face) const noexcept {
        return (static_cast<std::size_t>(face.a) * 73856093u) ^
               (static_cast<std::size_t>(face.b) * 19349663u) ^
               (static_cast<std::size_t>(face.c) * 83492791u);
    }
};

[[nodiscard]] EdgeKey edge_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return {a, b};
}

struct EdgeOwner {
    int first_face{-1};
    int first_slot{-1};
    bool first_forward{false};
    std::uint8_t count{0};
};

struct Connectivity {
    std::vector<std::array<int, 3>> neighbor;
    std::vector<std::array<std::uint8_t, 3>> same_direction;
    std::unordered_map<EdgeKey, EdgeOwner, EdgeHash> edges;
};

Connectivity build_connectivity(const std::vector<Eigen::Vector3i>& faces) {
    Connectivity out;
    out.neighbor.assign(faces.size(), std::array<int, 3>{-1, -1, -1});
    out.same_direction.assign(
        faces.size(), std::array<std::uint8_t, 3>{0, 0, 0});
    out.edges.reserve(faces.size() * 2);
    for (std::size_t fi = 0; fi < faces.size(); ++fi) {
        const Eigen::Vector3i& face = faces[fi];
        for (int slot = 0; slot < 3; ++slot) {
            const int a = face[slot];
            const int b = face[(slot + 1) % 3];
            const EdgeKey key = edge_key(a, b);
            const bool forward = a == key.a;
            auto [it, inserted] = out.edges.emplace(
                key, EdgeOwner{static_cast<int>(fi), slot, forward, 1});
            if (inserted) continue;
            EdgeOwner& owner = it->second;
            if (owner.count == 1) {
                out.neighbor[fi][static_cast<std::size_t>(slot)] =
                    owner.first_face;
                out.neighbor[static_cast<std::size_t>(owner.first_face)]
                            [static_cast<std::size_t>(owner.first_slot)] =
                    static_cast<int>(fi);
                const std::uint8_t same = owner.first_forward == forward ? 1U : 0U;
                out.same_direction[fi][static_cast<std::size_t>(slot)] = same;
                out.same_direction[static_cast<std::size_t>(owner.first_face)]
                                  [static_cast<std::size_t>(owner.first_slot)] = same;
            }
            if (owner.count < std::numeric_limits<std::uint8_t>::max())
                ++owner.count;
        }
    }
    return out;
}

void orient_components(std::vector<Eigen::Vector3i>& faces) {
    const Connectivity connectivity = build_connectivity(faces);
    std::vector<std::int8_t> flip(faces.size(), -1);
    std::queue<int> queue;
    for (std::size_t seed = 0; seed < faces.size(); ++seed) {
        if (flip[seed] >= 0) continue;
        flip[seed] = 0;
        queue.push(static_cast<int>(seed));
        while (!queue.empty()) {
            const int face = queue.front();
            queue.pop();
            for (int slot = 0; slot < 3; ++slot) {
                const int adjacent =
                    connectivity.neighbor[static_cast<std::size_t>(face)]
                                         [static_cast<std::size_t>(slot)];
                if (adjacent < 0) continue;
                const std::int8_t expected = static_cast<std::int8_t>(
                    flip[static_cast<std::size_t>(face)] ^
                    connectivity.same_direction[static_cast<std::size_t>(face)]
                                                   [static_cast<std::size_t>(slot)]);
                if (flip[static_cast<std::size_t>(adjacent)] < 0) {
                    flip[static_cast<std::size_t>(adjacent)] = expected;
                    queue.push(adjacent);
                }
            }
        }
    }
    for (std::size_t i = 0; i < faces.size(); ++i)
        if (flip[i] == 1) std::swap(faces[i][1], faces[i][2]);
}

void remove_small_components(
    std::vector<Eigen::Vector3i>& faces, const unsigned minimum_faces) {
    if (faces.empty() || minimum_faces <= 1) return;
    const Connectivity connectivity = build_connectivity(faces);
    std::vector<int> component(faces.size(), -1);
    std::vector<unsigned> sizes;
    std::queue<int> queue;
    for (std::size_t seed = 0; seed < faces.size(); ++seed) {
        if (component[seed] >= 0) continue;
        const int id = static_cast<int>(sizes.size());
        unsigned size = 0;
        component[seed] = id;
        queue.push(static_cast<int>(seed));
        while (!queue.empty()) {
            const int face = queue.front();
            queue.pop();
            ++size;
            for (const int adjacent :
                 connectivity.neighbor[static_cast<std::size_t>(face)]) {
                if (adjacent < 0 ||
                    component[static_cast<std::size_t>(adjacent)] >= 0)
                    continue;
                component[static_cast<std::size_t>(adjacent)] = id;
                queue.push(adjacent);
            }
        }
        sizes.push_back(size);
    }
    std::vector<Eigen::Vector3i> kept;
    kept.reserve(faces.size());
    for (std::size_t i = 0; i < faces.size(); ++i)
        if (sizes[static_cast<std::size_t>(component[i])] >= minimum_faces)
            kept.push_back(faces[i]);
    faces = std::move(kept);
}

unsigned close_small_holes(Mesh& mesh, const unsigned maximum_edges) {
    if (maximum_edges < 3 || mesh.faces.empty()) return 0;
    const Connectivity connectivity = build_connectivity(mesh.faces);
    std::vector<std::array<int, 2>> boundary_neighbors(
        mesh.vertices.size(), std::array<int, 2>{-1, -1});
    std::vector<std::uint8_t> degree(mesh.vertices.size(), 0);
    std::unordered_map<EdgeKey, bool, EdgeHash> oriented_forward;
    oriented_forward.reserve(connectivity.edges.size());

    for (const auto& [edge, owner] : connectivity.edges) {
        if (owner.count != 1) continue;
        if (degree[static_cast<std::size_t>(edge.a)] >= 2 ||
            degree[static_cast<std::size_t>(edge.b)] >= 2)
            continue;
        boundary_neighbors[static_cast<std::size_t>(edge.a)]
                          [degree[static_cast<std::size_t>(edge.a)]++] = edge.b;
        boundary_neighbors[static_cast<std::size_t>(edge.b)]
                          [degree[static_cast<std::size_t>(edge.b)]++] = edge.a;
        oriented_forward.emplace(edge, owner.first_forward);
    }

    std::unordered_set<EdgeKey, EdgeHash> visited;
    visited.reserve(oriented_forward.size());
    unsigned closed = 0;
    for (const auto& [seed_edge, seed_forward] : oriented_forward) {
        if (visited.contains(seed_edge)) continue;
        std::vector<int> loop{seed_edge.a, seed_edge.b};
        visited.insert(seed_edge);
        int previous = seed_edge.a;
        int current = seed_edge.b;
        bool valid = true;
        while (current != loop.front()) {
            if (degree[static_cast<std::size_t>(current)] != 2) {
                valid = false;
                break;
            }
            const auto& adjacent =
                boundary_neighbors[static_cast<std::size_t>(current)];
            const int next = adjacent[0] == previous ? adjacent[1] : adjacent[0];
            const EdgeKey edge = edge_key(current, next);
            if (visited.contains(edge)) {
                if (next != loop.front()) valid = false;
                current = next;
                break;
            }
            visited.insert(edge);
            previous = current;
            current = next;
            if (current != loop.front()) loop.push_back(current);
            if (loop.size() > maximum_edges) {
                valid = false;
                break;
            }
        }
        if (!valid || current != loop.front() || loop.size() < 3 ||
            loop.size() > maximum_edges)
            continue;

        Vec3f center = Vec3f::Zero();
        Vec3f color = Vec3f::Zero();
        for (const int vertex : loop) {
            center += mesh.vertices[static_cast<std::size_t>(vertex)];
            if (mesh.colors.size() == mesh.vertices.size())
                color += mesh.colors[static_cast<std::size_t>(vertex)];
        }
        center /= static_cast<float>(loop.size());
        color /= static_cast<float>(loop.size());
        const int center_id = static_cast<int>(mesh.vertices.size());
        mesh.vertices.push_back(center);
        if (!mesh.colors.empty()) mesh.colors.push_back(color);

        // Existing boundary faces use seed_edge.a->seed_edge.b when
        // seed_forward is true. New triangles must traverse shared edges in
        // the opposite direction.
        const bool traversal_matches_existing = seed_forward;
        for (std::size_t i = 0; i < loop.size(); ++i) {
            const int a = loop[i];
            const int b = loop[(i + 1) % loop.size()];
            if (traversal_matches_existing)
                mesh.faces.emplace_back(b, a, center_id);
            else
                mesh.faces.emplace_back(a, b, center_id);
        }
        ++closed;
    }
    return closed;
}

void compact_and_compute_normals(Mesh& mesh) {
    std::vector<int> remap(mesh.vertices.size(), -1);
    for (const Eigen::Vector3i& face : mesh.faces)
        for (int k = 0; k < 3; ++k)
            remap[static_cast<std::size_t>(face[k])] = 0;
    std::vector<Vec3f> vertices;
    std::vector<Vec3f> colors;
    vertices.reserve(mesh.vertices.size());
    if (!mesh.colors.empty()) colors.reserve(mesh.colors.size());
    for (std::size_t i = 0; i < remap.size(); ++i) {
        if (remap[i] < 0) continue;
        remap[i] = static_cast<int>(vertices.size());
        vertices.push_back(mesh.vertices[i]);
        if (mesh.colors.size() == mesh.vertices.size())
            colors.push_back(mesh.colors[i]);
    }
    for (Eigen::Vector3i& face : mesh.faces)
        for (int k = 0; k < 3; ++k)
            face[k] = remap[static_cast<std::size_t>(face[k])];
    mesh.vertices = std::move(vertices);
    mesh.colors = std::move(colors);
    mesh.normals.assign(mesh.vertices.size(), Vec3f::Zero());
    for (const Eigen::Vector3i& face : mesh.faces) {
        const Vec3f normal =
            (mesh.vertices[static_cast<std::size_t>(face[1])] -
             mesh.vertices[static_cast<std::size_t>(face[0])])
                .cross(mesh.vertices[static_cast<std::size_t>(face[2])] -
                       mesh.vertices[static_cast<std::size_t>(face[0])]);
        for (int k = 0; k < 3; ++k)
            mesh.normals[static_cast<std::size_t>(face[k])] += normal;
    }
    for (Vec3f& normal : mesh.normals)
        if (normal.squaredNorm() > 1e-12F) normal.normalize();
}

void smooth_mesh(Mesh& mesh, const DensifyOptions& options) {
    if (options.mesh_smooth_iters == 0 || mesh.faces.empty()) return;
    const Connectivity connectivity = build_connectivity(mesh.faces);
    std::vector<std::vector<int>> adjacent(mesh.vertices.size());
    std::vector<std::uint8_t> boundary(mesh.vertices.size(), 0);
    for (const auto& [edge, owner] : connectivity.edges) {
        adjacent[static_cast<std::size_t>(edge.a)].push_back(edge.b);
        adjacent[static_cast<std::size_t>(edge.b)].push_back(edge.a);
        if (owner.count == 1) {
            boundary[static_cast<std::size_t>(edge.a)] = 1;
            boundary[static_cast<std::size_t>(edge.b)] = 1;
        }
    }
    const float lambda = std::clamp(options.mesh_smooth_lambda, 0.F, 0.5F);
    std::vector<Vec3f> next(mesh.vertices.size());
    for (unsigned iteration = 0; iteration < options.mesh_smooth_iters; ++iteration) {
        next = mesh.vertices;
        for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
            if (boundary[i] || adjacent[i].empty()) continue;
            Vec3f mean = Vec3f::Zero();
            for (const int neighbor : adjacent[i])
                mean += mesh.vertices[static_cast<std::size_t>(neighbor)];
            mean /= static_cast<float>(adjacent[i].size());
            next[i] = mesh.vertices[i] * (1.F - lambda) + mean * lambda;
        }
        mesh.vertices.swap(next);
    }
}

}  // namespace

void clean_mesh(Mesh& mesh, const DensifyOptions& options) {
    if (!options.mesh_clean || mesh.faces.empty()) {
        compact_and_compute_normals(mesh);
        return;
    }
    core::StageScope stage("mvs.mesh_clean");
    const std::size_t input_faces = mesh.faces.size();

    std::unordered_set<FaceKey, FaceHash> unique;
    unique.reserve(mesh.faces.size());
    std::unordered_map<EdgeKey, std::uint8_t, EdgeHash> edge_counts;
    edge_counts.reserve(mesh.faces.size() * 2);
    std::vector<Eigen::Vector3i> accepted;
    accepted.reserve(mesh.faces.size());
    for (const Eigen::Vector3i& face : mesh.faces) {
        if (face.minCoeff() < 0 ||
            face.maxCoeff() >= static_cast<int>(mesh.vertices.size()) ||
            face[0] == face[1] || face[1] == face[2] || face[2] == face[0])
            continue;
        const Vec3f cross =
            (mesh.vertices[static_cast<std::size_t>(face[1])] -
             mesh.vertices[static_cast<std::size_t>(face[0])])
                .cross(mesh.vertices[static_cast<std::size_t>(face[2])] -
                       mesh.vertices[static_cast<std::size_t>(face[0])]);
        if (!cross.allFinite() || cross.squaredNorm() <= 1e-20F) continue;
        std::array<int, 3> sorted{face[0], face[1], face[2]};
        std::sort(sorted.begin(), sorted.end());
        if (!unique.emplace(FaceKey{sorted[0], sorted[1], sorted[2]}).second)
            continue;
        const std::array<EdgeKey, 3> edges{
            edge_key(face[0], face[1]), edge_key(face[1], face[2]),
            edge_key(face[2], face[0])};
        bool manifold = true;
        for (const EdgeKey& edge : edges) {
            const auto found = edge_counts.find(edge);
            if (found != edge_counts.end() && found->second >= 2) {
                manifold = false;
                break;
            }
        }
        if (!manifold) continue;
        accepted.push_back(face);
        for (const EdgeKey& edge : edges) ++edge_counts[edge];
    }
    mesh.faces = std::move(accepted);
    orient_components(mesh.faces);
    remove_small_components(mesh.faces, options.mesh_min_component_faces);
    orient_components(mesh.faces);
    const unsigned holes = close_small_holes(mesh, options.mesh_close_hole_edges);
    smooth_mesh(mesh, options);
    compact_and_compute_normals(mesh);

    core::Logger::instance().info(
        "mvs mesh clean: faces=", input_faces, " -> ", mesh.faces.size(),
        " vertices=", mesh.vertices.size(), " holes_closed=", holes);
    stage.finish();
}

}  // namespace aetherscan::mvs::detail
