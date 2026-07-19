#include "mvs/densify.hpp"
#include "mvs/internal.hpp"
#include "mvs/maxflow.hpp"

#include "core/logging.hpp"
#include "parallel/thread_pool.hpp"

#include <Eigen/LU>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aetherscan::mvs {
namespace {

constexpr float k_eps = 1e-9F;

struct Vert {
    Vec3f p{};
    Vec3f n{};
    Vec3f c{};
    std::vector<Index> views;
    float weight{1.F};
};

struct Tet {
    int v[4]{-1, -1, -1, -1};
    int neigh[4]{-1, -1, -1, -1};  // opposite vertex i
    float f[4]{0, 0, 0, 0};
    float s{0};
    float t{0};
    bool alive{true};
};

[[nodiscard]] float orient3d(
    const Vec3f& a, const Vec3f& b, const Vec3f& c, const Vec3f& d) {
    // scalar triple product (b-a, c-a, d-a)
    const Vec3f ad = d - a;
    return (b - a).cross(c - a).dot(ad);
}

[[nodiscard]] bool insphere(
    const Vec3f& a, const Vec3f& b, const Vec3f& c, const Vec3f& d,
    const Vec3f& p) {
    // Point p inside circumsphere of tet abcd (positive orientation).
    const Vec3f ba = b - a;
    const Vec3f ca = c - a;
    const Vec3f da = d - a;
    const float ba2 = ba.squaredNorm();
    const float ca2 = ca.squaredNorm();
    const float da2 = da.squaredNorm();
    // Rows of A are edge vectors; solve A * center_local = 0.5 * ||edge||^2.
    Eigen::Matrix3f A;
    A.row(0) = ba.transpose();
    A.row(1) = ca.transpose();
    A.row(2) = da.transpose();
    const Eigen::Vector3f rhs(0.5F * ba2, 0.5F * ca2, 0.5F * da2);
    const Eigen::Vector3f center_local = A.fullPivLu().solve(rhs);
    if (!center_local.allFinite()) return false;
    const Vec3f center = a + center_local;
    const float r2 = (center - a).squaredNorm();
    return (p - center).squaredNorm() + 1e-8F < r2;
}

[[nodiscard]] bool point_in_tet(
    const Vec3f& a, const Vec3f& b, const Vec3f& c, const Vec3f& d,
    const Vec3f& p) {
    // Same-side test against each face's opposite vertex.
    const float o0v = orient3d(b, c, d, a);
    const float o0p = orient3d(b, c, d, p);
    const float o1v = orient3d(a, d, c, b);
    const float o1p = orient3d(a, d, c, p);
    const float o2v = orient3d(a, b, d, c);
    const float o2p = orient3d(a, b, d, p);
    const float o3v = orient3d(a, c, b, d);
    const float o3p = orient3d(a, c, b, p);
    constexpr float eps = 1e-5F;
    return o0v * o0p >= -eps && o1v * o1p >= -eps && o2v * o2p >= -eps &&
           o3v * o3p >= -eps;
}

[[nodiscard]] bool touches_super(const Tet& tet, const int super_count) {
    for (int i = 0; i < 4; ++i)
        if (tet.v[i] >= 0 && tet.v[i] < super_count) return true;
    return false;
}

// Opposite face vertices for tet vertex slot i (CCW when viewed from i).
constexpr int k_face_verts[4][3] = {
    {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};

void reconstruct_mesh_projective(MvsScene& scene, const DensifyOptions& options);

std::vector<Vert> downsample_cloud(
    const DenseCloud& cloud, const MvsScene& scene, const DensifyOptions& options) {
    std::vector<const DensePoint*> pts;
    pts.reserve(cloud.points.size());
    for (const auto& p : cloud.points) {
        if (p.views.empty()) continue;
        pts.push_back(&p);
    }
    if (options.mesh_max_points > 0 &&
        pts.size() > options.mesh_max_points) {
        std::mt19937 rng(7);
        std::shuffle(pts.begin(), pts.end(), rng);
        pts.resize(static_cast<std::size_t>(options.mesh_max_points));
    }
    if (pts.empty()) return {};

    Vec3f bmin = pts[0]->position;
    Vec3f bmax = bmin;
    for (const auto* p : pts) {
        bmin = bmin.cwiseMin(p->position);
        bmax = bmax.cwiseMax(p->position);
    }
    const float diag = std::max((bmax - bmin).norm(), 1e-3F);
    // Voxel size from typical pixel footprint at median depth.
    float voxel = diag * 0.002F;
    if (options.mesh_dist_insert_px > 0.F && !scene.views.empty()) {
        float median_fx = 0.F;
        for (const auto& v : scene.views) median_fx += v.fx;
        median_fx /= static_cast<float>(scene.views.size());
        voxel = std::max(voxel, (diag / median_fx) * options.mesh_dist_insert_px);
    }

    struct Key {
        int x, y, z;
        bool operator==(const Key& o) const {
            return x == o.x && y == o.y && z == o.z;
        }
    };
    struct Hash {
        std::size_t operator()(const Key& k) const noexcept {
            return (static_cast<std::size_t>(k.x) * 73856093u) ^
                   (static_cast<std::size_t>(k.y) * 19349663u) ^
                   (static_cast<std::size_t>(k.z) * 83492791u);
        }
    };
    std::unordered_map<Key, Vert, Hash> grid;
    for (const auto* p : pts) {
        const Key key{
            static_cast<int>(std::floor((p->position.x() - bmin.x()) / voxel)),
            static_cast<int>(std::floor((p->position.y() - bmin.y()) / voxel)),
            static_cast<int>(std::floor((p->position.z() - bmin.z()) / voxel))};
        auto it = grid.find(key);
        if (it == grid.end()) {
            Vert v;
            v.p = p->position;
            v.n = p->normal;
            v.c = p->color;
            v.views = p->views;
            v.weight = p->weight;
            grid.emplace(key, std::move(v));
        } else if (p->weight > it->second.weight) {
            it->second.p = p->position;
            it->second.n = p->normal;
            it->second.c = p->color;
            it->second.views = p->views;
            it->second.weight = p->weight;
        } else {
            // Merge visibility.
            for (const Index id : p->views) {
                if (std::find(
                        it->second.views.begin(), it->second.views.end(), id) ==
                    it->second.views.end())
                    it->second.views.push_back(id);
            }
        }
    }
    std::vector<Vert> out;
    out.reserve(grid.size());
    for (auto& kv : grid) out.push_back(std::move(kv.second));
    return out;
}

// Incremental Bowyer-Watson 3D Delaunay.
class Delaunay3 {
public:
    std::vector<Vec3f> points;
    std::vector<Tet> tets;

    void build(const std::vector<Vert>& verts) {
        points.clear();
        tets.clear();
        if (verts.size() < 4) return;

        Vec3f bmin = verts[0].p;
        Vec3f bmax = bmin;
        for (const auto& v : verts) {
            bmin = bmin.cwiseMin(v.p);
            bmax = bmax.cwiseMax(v.p);
        }
        const Vec3f center = 0.5F * (bmin + bmax);
        const float radius = (bmax - bmin).norm() * 2.F + 1.F;

        // Super tetrahedron vertices (indices 0..3).
        points.push_back(center + Vec3f{0, 0, radius});
        points.push_back(center + Vec3f{0, radius * 0.9428F, -radius * 0.333F});
        points.push_back(
            center + Vec3f{radius * 0.8165F, -radius * 0.4714F, -radius * 0.333F});
        points.push_back(
            center + Vec3f{-radius * 0.8165F, -radius * 0.4714F, -radius * 0.333F});

        const int base = 4;
        points.reserve(base + verts.size());
        for (const auto& v : verts) points.push_back(v.p);

        Tet super;
        super.v[0] = 0;
        super.v[1] = 1;
        super.v[2] = 2;
        super.v[3] = 3;
        // Ensure positive orientation.
        if (orient3d(points[0], points[1], points[2], points[3]) < 0.F)
            std::swap(super.v[2], super.v[3]);
        tets.push_back(super);

        for (std::size_t pi = 0; pi < verts.size(); ++pi) {
            const int pid = base + static_cast<int>(pi);
            insert_point(pid);
        }
        // Keep super-touching tets alive: they represent the outside region and
        // receive source capacity during graph-cut.
        super_count_ = base;
    }

    [[nodiscard]] int super_count() const { return super_count_; }

private:
    int super_count_{0};

    void insert_point(const int pid) {
        std::vector<int> bad;
        for (int ti = 0; ti < static_cast<int>(tets.size()); ++ti) {
            const Tet& tet = tets[static_cast<std::size_t>(ti)];
            if (!tet.alive) continue;
            if (insphere(
                    points[tet.v[0]], points[tet.v[1]], points[tet.v[2]],
                    points[tet.v[3]], points[pid]))
                bad.push_back(ti);
        }
        if (bad.empty()) return;

        struct FaceKey {
            int a, b, c;
            bool operator==(const FaceKey& o) const {
                return a == o.a && b == o.b && c == o.c;
            }
        };
        struct FaceHash {
            std::size_t operator()(const FaceKey& f) const noexcept {
                return (static_cast<std::size_t>(f.a) * 73856093u) ^
                       (static_cast<std::size_t>(f.b) * 19349663u) ^
                       (static_cast<std::size_t>(f.c) * 83492791u);
            }
        };
        auto canon = [](int a, int b, int c) {
            int v[3] = {a, b, c};
            std::sort(v, v + 3);
            return FaceKey{v[0], v[1], v[2]};
        };

        // face -> (oriented verts, exterior neighbor tet, exterior face index)
        struct Boundary {
            std::array<int, 3> verts{};
            int exterior_tet{-1};
            int exterior_face{-1};
            int count{0};
        };
        std::unordered_map<FaceKey, Boundary, FaceHash> faces;
        std::unordered_set<int> bad_set(bad.begin(), bad.end());

        for (const int ti : bad) {
            Tet& tet = tets[static_cast<std::size_t>(ti)];
            for (int fi = 0; fi < 4; ++fi) {
                const int a = tet.v[k_face_verts[fi][0]];
                const int b = tet.v[k_face_verts[fi][1]];
                const int c = tet.v[k_face_verts[fi][2]];
                const FaceKey key = canon(a, b, c);
                Boundary& slot = faces[key];
                slot.verts = {a, b, c};
                slot.count += 1;
                const int nb = tet.neigh[fi];
                if (nb >= 0 && bad_set.count(nb) == 0) {
                    slot.exterior_tet = nb;
                    // Find face index on neighbor.
                    for (int k = 0; k < 4; ++k) {
                        if (tets[static_cast<std::size_t>(nb)].neigh[k] == ti) {
                            slot.exterior_face = k;
                            break;
                        }
                    }
                }
            }
            tet.alive = false;
        }

        std::vector<int> created;
        for (auto& kv : faces) {
            Boundary& b = kv.second;
            if (b.count != 1) continue;
            Tet nt;
            nt.v[0] = pid;
            nt.v[1] = b.verts[0];
            nt.v[2] = b.verts[1];
            nt.v[3] = b.verts[2];
            if (orient3d(
                    points[nt.v[0]], points[nt.v[1]], points[nt.v[2]],
                    points[nt.v[3]]) < 0.F)
                std::swap(nt.v[2], nt.v[3]);
            // Face opposite pid (index 0) attaches to exterior.
            if (b.exterior_tet >= 0) {
                nt.neigh[0] = b.exterior_tet;
                if (b.exterior_face >= 0)
                    tets[static_cast<std::size_t>(b.exterior_tet)]
                        .neigh[b.exterior_face] = static_cast<int>(tets.size());
            }
            created.push_back(static_cast<int>(tets.size()));
            tets.push_back(nt);
        }

        // Link new tets to each other on faces that include pid.
        std::unordered_map<FaceKey, std::pair<int, int>, FaceHash> new_faces;
        for (const int ti : created) {
            Tet& tet = tets[static_cast<std::size_t>(ti)];
            for (int fi = 1; fi < 4; ++fi) {  // faces adjacent to pid
                const FaceKey key = canon(
                    tet.v[k_face_verts[fi][0]], tet.v[k_face_verts[fi][1]],
                    tet.v[k_face_verts[fi][2]]);
                auto it = new_faces.find(key);
                if (it == new_faces.end()) {
                    new_faces.emplace(key, std::make_pair(ti, fi));
                } else {
                    const int tj = it->second.first;
                    const int fj = it->second.second;
                    tet.neigh[fi] = tj;
                    tets[static_cast<std::size_t>(tj)].neigh[fj] = ti;
                    new_faces.erase(it);
                }
            }
        }
    }

    // Kept for rare full repair; not used on hot path.
    void rebuild_neighbors() {}
};

[[nodiscard]] bool segment_triangle_intersect(
    const Vec3f& p0, const Vec3f& p1, const Vec3f& a, const Vec3f& b,
    const Vec3f& c, float& t_hit) {
    const Vec3f dir = p1 - p0;
    const Vec3f ab = b - a;
    const Vec3f ac = c - a;
    const Vec3f n = ab.cross(ac);
    const float denom = n.dot(dir);
    if (std::abs(denom) < 1e-12F) return false;
    const float t = n.dot(a - p0) / denom;
    if (t < -1e-4F || t > 1.F + 1e-4F) return false;
    const Vec3f q = p0 + t * dir;
    const Vec3f c0 = (b - a).cross(q - a);
    const Vec3f c1 = (c - b).cross(q - b);
    const Vec3f c2 = (a - c).cross(q - c);
    if (c0.dot(n) < -1e-6F || c1.dot(n) < -1e-6F || c2.dot(n) < -1e-6F)
        return false;
    t_hit = std::clamp(t, 0.F, 1.F);
    return true;
}

void weight_visibility(
    Delaunay3& mesh, const std::vector<Vert>& verts, const MvsScene& scene,
    const DensifyOptions& options) {
    std::vector<float> edge2;
    edge2.reserve(mesh.tets.size() * 6);
    for (const auto& tet : mesh.tets) {
        if (!tet.alive) continue;
        for (int i = 0; i < 4; ++i)
            for (int j = i + 1; j < 4; ++j)
                edge2.push_back(
                    (mesh.points[tet.v[i]] - mesh.points[tet.v[j]]).squaredNorm());
    }
    if (edge2.empty()) return;
    const std::size_t mid = edge2.size() / 2;
    std::nth_element(
        edge2.begin(), edge2.begin() + static_cast<std::ptrdiff_t>(mid), edge2.end());
    const float sigma =
        options.mesh_k_sigma * std::sqrt(std::max(edge2[mid], 1e-12F));
    const float inv2s2 = 1.F / (2.F * sigma * sigma);
    const int base = mesh.super_count();

    auto find_tet = [&](const Vec3f& p) -> int {
        for (int ti = 0; ti < static_cast<int>(mesh.tets.size()); ++ti) {
            const Tet& tet = mesh.tets[static_cast<std::size_t>(ti)];
            if (!tet.alive) continue;
            if (point_in_tet(
                    mesh.points[tet.v[0]], mesh.points[tet.v[1]],
                    mesh.points[tet.v[2]], mesh.points[tet.v[3]], p))
                return ti;
        }
        return -1;
    };

    // Ray-walk C -> P through tet adjacency; weight only the outgoing facet of
    // each crossed cell (camera/free-space side).
    auto walk_ray = [&](int start, const Vec3f& C, const Vec3f& P, float alpha) {
        if (start < 0) return;
        int ti = start;
        for (int step = 0; step < 4096; ++step) {
            Tet& tet = mesh.tets[static_cast<std::size_t>(ti)];
            if (!tet.alive) return;
            if (point_in_tet(
                    mesh.points[tet.v[0]], mesh.points[tet.v[1]],
                    mesh.points[tet.v[2]], mesh.points[tet.v[3]], P)) {
                tet.t += alpha;
                return;
            }

            int best_fi = -1;
            float best_t = -1.F;
            for (int fi = 0; fi < 4; ++fi) {
                const Vec3f& a = mesh.points[tet.v[k_face_verts[fi][0]]];
                const Vec3f& b = mesh.points[tet.v[k_face_verts[fi][1]]];
                const Vec3f& c = mesh.points[tet.v[k_face_verts[fi][2]]];
                float th = 0.F;
                if (!segment_triangle_intersect(C, P, a, b, c, th)) continue;
                if (th > best_t) {
                    best_t = th;
                    best_fi = fi;
                }
            }
            if (best_fi < 0) {
                tet.t += alpha * 0.25F;
                return;
            }

            const float dist = (1.F - best_t) * (P - C).norm();
            const float w = alpha * (1.F - std::exp(-dist * dist * inv2s2));
            tet.f[best_fi] += w;

            const int next = tet.neigh[best_fi];
            if (next < 0) return;
            ti = next;
        }
    };

    // Outside capacity for any tet that touches the super tetrahedron.
    for (auto& tet : mesh.tets) {
        if (!tet.alive) continue;
        if (touches_super(tet, base))
            tet.s = std::max(tet.s, options.mesh_k_inf);
    }

    for (std::size_t vi = 0; vi < verts.size(); ++vi) {
        const Vert& vert = verts[vi];
        const int pid = base + static_cast<int>(vi);
        const Vec3f& P = mesh.points[pid];
        const float alpha = std::max(vert.weight, 1e-3F);

        for (const Index view_id : vert.views) {
            if (view_id >= scene.views.size()) continue;
            const MvsView& view = scene.views[view_id];
            const Vec3f C = view.pose.C.cast<float>();
            const int cam_tet = find_tet(C);
            if (cam_tet >= 0)
                mesh.tets[static_cast<std::size_t>(cam_tet)].s =
                    std::max(
                        mesh.tets[static_cast<std::size_t>(cam_tet)].s,
                        options.mesh_k_inf);

            walk_ray(cam_tet >= 0 ? cam_tet : find_tet(P), C, P, alpha);

            // Beyond-surface sink support.
            const Vec3f dir = (P - C).normalized();
            const Vec3f E = P + dir * sigma;
            const int end_tet = find_tet(E);
            if (end_tet >= 0)
                mesh.tets[static_cast<std::size_t>(end_tet)].t += alpha;
        }
    }
}

void extract_cut_mesh(
    const Delaunay3& delaunay, const std::vector<Vert>& verts,
    const maxflow::Graph& graph, const std::vector<int>& tet_node,
    Mesh& out) {
    out = {};
    const int base = delaunay.super_count();
    std::unordered_map<int, int> vmap;
    auto get_v = [&](int pid) -> int {
        if (pid < base) return -1;  // never emit super vertices
        auto it = vmap.find(pid);
        if (it != vmap.end()) return it->second;
        const int id = static_cast<int>(out.vertices.size());
        out.vertices.push_back(delaunay.points[pid]);
        const int vi = pid - base;
        if (vi >= 0 && vi < static_cast<int>(verts.size())) {
            out.normals.push_back(verts[static_cast<std::size_t>(vi)].n);
            out.colors.push_back(verts[static_cast<std::size_t>(vi)].c);
        } else {
            out.normals.push_back(Vec3f::UnitZ());
            out.colors.push_back(Vec3f{0.7F, 0.7F, 0.7F});
        }
        vmap.emplace(pid, id);
        return id;
    };

    for (int ti = 0; ti < static_cast<int>(delaunay.tets.size()); ++ti) {
        const Tet& tet = delaunay.tets[static_cast<std::size_t>(ti)];
        if (!tet.alive) continue;
        const int ni = tet_node[static_cast<std::size_t>(ti)];
        if (ni < 0) continue;
        const bool src_i = graph.is_source_side(static_cast<std::size_t>(ni));
        for (int fi = 0; fi < 4; ++fi) {
            const int tj = tet.neigh[fi];
            if (tj < 0 || tj < ti) continue;
            const Tet& other = delaunay.tets[static_cast<std::size_t>(tj)];
            if (!other.alive) continue;
            const int nj = tet_node[static_cast<std::size_t>(tj)];
            if (nj < 0) continue;
            const bool src_j = graph.is_source_side(static_cast<std::size_t>(nj));
            if (src_i == src_j) continue;

            // Skip faces that involve super vertices.
            bool has_super = false;
            for (int k = 0; k < 3; ++k) {
                if (tet.v[k_face_verts[fi][k]] < base) {
                    has_super = true;
                    break;
                }
            }
            if (has_super) continue;

            int a = get_v(tet.v[k_face_verts[fi][0]]);
            int b = get_v(tet.v[k_face_verts[fi][1]]);
            int c = get_v(tet.v[k_face_verts[fi][2]]);
            if (a < 0 || b < 0 || c < 0) continue;
            // Face of current tet points outward from that tet. We want
            // inside(sink)->outside(source). If current is source/outside,
            // flip so the normal points outward from sink.
            if (src_i) std::swap(b, c);
            out.faces.emplace_back(a, b, c);
        }
    }

    // Recompute normals from oriented faces.
    out.normals.assign(out.vertices.size(), Vec3f::Zero());
    for (const auto& f : out.faces) {
        const Vec3f n = (out.vertices[f[1]] - out.vertices[f[0]])
                            .cross(out.vertices[f[2]] - out.vertices[f[0]]);
        out.normals[f[0]] += n;
        out.normals[f[1]] += n;
        out.normals[f[2]] += n;
    }
    for (auto& n : out.normals)
        if (n.squaredNorm() > 1e-12F) n.normalize();
}

void reconstruct_mesh_delaunay(MvsScene& scene, const DensifyOptions& options) {
    core::Logger::instance().info("mvs mesh: delaunay_cut");
    auto verts = downsample_cloud(scene.dense_cloud, scene, options);
    core::Logger::instance().info("mvs mesh: delaunay_verts=", verts.size());
    if (verts.size() < 4) {
        core::Logger::instance().warning(
            "mvs mesh: too few points for Delaunay, falling back to projective");
        reconstruct_mesh_projective(scene, options);
        return;
    }

    // The in-tree Bowyer-Watson implementation is intentionally only a
    // small-cloud backend. Randomly reducing a production cloud to 8000
    // points destroyed geometry; preserve full detail with the scalable
    // projective backend instead.
    constexpr std::size_t k_soft_cap = 8000;
    if (verts.size() > k_soft_cap) {
        core::Logger::instance().warning(
            "mvs mesh: delaunay input ", verts.size(),
            " exceeds scalable limit ", k_soft_cap,
            "; using full-resolution projective meshing");
        reconstruct_mesh_projective(scene, options);
        return;
    }

    Delaunay3 delaunay;
    {
        core::StageScope stage("mvs.delaunay");
        delaunay.build(verts);
        stage.finish();
    }

    std::size_t live = 0;
    for (const auto& t : delaunay.tets)
        if (t.alive) ++live;
    core::Logger::instance().info("mvs mesh: tets=", live);
    if (live == 0) {
        reconstruct_mesh_projective(scene, options);
        return;
    }

    {
        core::StageScope stage("mvs.visibility_weights");
        weight_visibility(delaunay, verts, scene, options);
        stage.finish();
    }

    // Graph cut.
    std::vector<int> tet_node(delaunay.tets.size(), -1);
    int node_count = 0;
    for (int ti = 0; ti < static_cast<int>(delaunay.tets.size()); ++ti) {
        if (!delaunay.tets[static_cast<std::size_t>(ti)].alive) continue;
        tet_node[static_cast<std::size_t>(ti)] = node_count++;
    }
    maxflow::Graph graph(static_cast<std::size_t>(node_count));
    for (int ti = 0; ti < static_cast<int>(delaunay.tets.size()); ++ti) {
        const Tet& tet = delaunay.tets[static_cast<std::size_t>(ti)];
        if (!tet.alive) continue;
        const int ni = tet_node[static_cast<std::size_t>(ti)];
        graph.add_tweights(
            static_cast<std::size_t>(ni), tet.s, std::min(tet.t, options.mesh_k_inf));
        for (int fi = 0; fi < 4; ++fi) {
            const int tj = tet.neigh[fi];
            if (tj < 0 || tj < ti) continue;
            const Tet& other = delaunay.tets[static_cast<std::size_t>(tj)];
            if (!other.alive) continue;
            const int nj = tet_node[static_cast<std::size_t>(tj)];
            int back = -1;
            for (int k = 0; k < 4; ++k)
                if (other.neigh[k] == ti) {
                    back = k;
                    break;
                }
            const float cap_ij = tet.f[fi] + options.mesh_k_qual;
            const float cap_ji =
                (back >= 0 ? other.f[back] : 0.F) + options.mesh_k_qual;
            graph.add_edge(
                static_cast<std::size_t>(ni), static_cast<std::size_t>(nj), cap_ij,
                cap_ji);
        }
    }

    float flow = 0.F;
    {
        core::StageScope stage("mvs.graph_cut");
        flow = graph.maxflow();
        stage.finish();
    }
    core::Logger::instance().info("mvs mesh: maxflow=", flow);

    extract_cut_mesh(delaunay, verts, graph, tet_node, scene.mesh);
    core::Logger::instance().info(
        "mvs mesh: vertices=", scene.mesh.vertices.size(),
        " faces=", scene.mesh.faces.size());
}

// Full-resolution projective meshing. Each depth map contributes locally
// coherent triangles; vertices are then welded in world space and duplicate
// faces / small disconnected islands are removed.
void reconstruct_mesh_projective(MvsScene& scene, const DensifyOptions& options) {
    scene.mesh = {};
    if (scene.views.empty()) return;

    struct Key {
        std::int64_t x, y, z;
        bool operator==(const Key&) const = default;
    };
    struct Hash {
        std::size_t operator()(const Key& k) const noexcept {
            auto mix = [](std::uint64_t x) {
                x ^= x >> 30;
                x *= 0xbf58476d1ce4e5b9ULL;
                x ^= x >> 27;
                x *= 0x94d049bb133111ebULL;
                return x ^ (x >> 31);
            };
            return static_cast<std::size_t>(
                mix(static_cast<std::uint64_t>(k.x)) ^
                (mix(static_cast<std::uint64_t>(k.y)) << 1) ^
                (mix(static_cast<std::uint64_t>(k.z)) << 2));
        }
    };

    std::vector<float> footprints;
    footprints.reserve(std::min<std::size_t>(scene.dense_cloud.points.size(), 200000));
    for (std::size_t i = 0;
         i < scene.dense_cloud.points.size() && footprints.size() < 200000; ++i) {
        const DensePoint& point = scene.dense_cloud.points[i];
        if (point.views.empty()) continue;
        const Index view_id = point.views.front();
        if (view_id >= scene.views.size()) continue;
        const MvsView& view = scene.views[view_id];
        const float depth = static_cast<float>(
            view.pose.transform_world_to_camera(point.position.cast<double>()).z());
        if (depth > 0.F && std::isfinite(depth))
            footprints.push_back(depth / std::max(view.fx, 1.F));
    }
    if (footprints.empty()) return;
    const std::size_t footprint_mid = footprints.size() / 2;
    std::nth_element(
        footprints.begin(),
        footprints.begin() + static_cast<std::ptrdiff_t>(footprint_mid),
        footprints.end());
    const float pixel_footprint = std::max(footprints[footprint_mid], 1e-7F);
    const float quant = std::max(
        pixel_footprint * options.mesh_weld_pixel_fraction, 1e-8F);
    const unsigned step = std::max(1U, options.mesh_pixel_step);

    struct LocalMesh {
        std::vector<Vec3f> vertices;
        std::vector<Vec3f> normals;
        std::vector<Eigen::Vector3i> faces;
    };
    std::vector<LocalMesh> locals(scene.views.size());
    const unsigned threads = parallel::resolve_thread_count(scene.thread_count);
    parallel::parallel_for(scene.views.size(), threads, [&](const std::size_t vi) {
        const MvsView& view = scene.views[vi];
        const DepthMap& dm = view.depth_map;
        if (dm.depth.empty() || view.width <= step || view.height <= step) return;
        LocalMesh& local = locals[vi];
        std::vector<int> vertex_ids(dm.size(), -1);
        const Mat3f world_rotation = view.pose.R.transpose().cast<float>();

        auto vertex = [&](const int x, const int y) -> int {
            const std::size_t index = dm.index(x, y);
            int& id = vertex_ids[index];
            if (id >= 0) return id;
            const Vec3f camera = view.unproject(
                static_cast<float>(x), static_cast<float>(y), dm.depth[index]);
            id = static_cast<int>(local.vertices.size());
            local.vertices.push_back(
                view.pose.transform_camera_to_world(camera.cast<double>()).cast<float>());
            Vec3f normal = world_rotation * dm.normal[index];
            if (normal.squaredNorm() > 1e-10F) normal.normalize();
            local.normals.push_back(normal);
            return id;
        };

        auto emit = [&](const int ax, const int ay, const int bx, const int by,
                        const int cx, const int cy) {
            const std::size_t ai = dm.index(ax, ay);
            const std::size_t bi = dm.index(bx, by);
            const std::size_t ci = dm.index(cx, cy);
            const float da = dm.depth[ai];
            const float db = dm.depth[bi];
            const float dc = dm.depth[ci];
            if (da <= 0.F || db <= 0.F || dc <= 0.F) return;
            if (!(dm.confidence[ai] <= options.ncc_keep_threshold) ||
                !(dm.confidence[bi] <= options.ncc_keep_threshold) ||
                !(dm.confidence[ci] <= options.ncc_keep_threshold))
                return;
            const float min_depth = std::min({da, db, dc});
            const float max_depth = std::max({da, db, dc});
            if (max_depth - min_depth >
                options.mesh_depth_diff_threshold * max_depth)
                return;

            const int a = vertex(ax, ay);
            const int b = vertex(bx, by);
            const int c = vertex(cx, cy);
            const Vec3f& pa = local.vertices[static_cast<std::size_t>(a)];
            const Vec3f& pb = local.vertices[static_cast<std::size_t>(b)];
            const Vec3f& pc = local.vertices[static_cast<std::size_t>(c)];
            const float local_footprint =
                ((da + db + dc) / 3.F) /
                std::max(std::min(view.fx, view.fy), 1.F);
            const float max_edge =
                local_footprint * options.mesh_max_edge_voxels *
                static_cast<float>(step);
            if ((pa - pb).norm() > max_edge || (pb - pc).norm() > max_edge ||
                (pc - pa).norm() > max_edge)
                return;
            Vec3f face_normal = (pb - pa).cross(pc - pa);
            if (!face_normal.allFinite() || face_normal.squaredNorm() < 1e-16F)
                return;
            const Vec3f expected =
                local.normals[static_cast<std::size_t>(a)] +
                local.normals[static_cast<std::size_t>(b)] +
                local.normals[static_cast<std::size_t>(c)];
            if (face_normal.dot(expected) < 0.F)
                local.faces.emplace_back(a, c, b);
            else
                local.faces.emplace_back(a, b, c);
        };

        for (std::uint32_t y = 0; y + step < view.height; y += step) {
            for (std::uint32_t x = 0; x + step < view.width; x += step) {
                const int x0 = static_cast<int>(x);
                const int y0 = static_cast<int>(y);
                const int x1 = static_cast<int>(x + step);
                const int y1 = static_cast<int>(y + step);
                const float d00 = dm.depth[dm.index(x0, y0)];
                const float d10 = dm.depth[dm.index(x1, y0)];
                const float d01 = dm.depth[dm.index(x0, y1)];
                const float d11 = dm.depth[dm.index(x1, y1)];
                if ((d00 > 0.F && d11 > 0.F) &&
                    (!(d10 > 0.F && d01 > 0.F) ||
                     std::abs(d00 - d11) <= std::abs(d10 - d01))) {
                    emit(x0, y0, x1, y0, x1, y1);
                    emit(x0, y0, x1, y1, x0, y1);
                } else {
                    emit(x0, y0, x1, y0, x0, y1);
                    emit(x1, y0, x1, y1, x0, y1);
                }
            }
        }
    });

    auto key_of = [&](const Vec3f& world) {
        const Key key{
            static_cast<std::int64_t>(std::llround(world.x() / quant)),
            static_cast<std::int64_t>(std::llround(world.y() / quant)),
            static_cast<std::int64_t>(std::llround(world.z() / quant))};
        return key;
    };

    struct VertexAccum {
        Vec3f position{Vec3f::Zero()};
        Vec3f normal{Vec3f::Zero()};
        Vec3f color{Vec3f::Zero()};
        float weight{0.F};
    };
    std::unordered_map<Key, Vec3f, Hash> cloud_colors;
    cloud_colors.reserve(scene.dense_cloud.points.size());
    for (const DensePoint& point : scene.dense_cloud.points)
        cloud_colors.emplace(key_of(point.position), point.color);

    Mesh& mesh = scene.mesh;
    std::unordered_map<Key, int, Hash> weld;
    std::vector<VertexAccum> vertex_accum;
    std::vector<std::vector<int>> remaps(locals.size());
    for (std::size_t vi = 0; vi < locals.size(); ++vi) {
        const LocalMesh& local = locals[vi];
        auto& remap = remaps[vi];
        remap.resize(local.vertices.size());
        for (std::size_t i = 0; i < local.vertices.size(); ++i) {
            const Key key = key_of(local.vertices[i]);
            auto [it, inserted] = weld.emplace(key, static_cast<int>(weld.size()));
            const int id = it->second;
            if (inserted) vertex_accum.emplace_back();
            remap[i] = id;
            VertexAccum& accumulator = vertex_accum[static_cast<std::size_t>(id)];
            accumulator.position += local.vertices[i];
            accumulator.normal += local.normals[i];
            const auto color = cloud_colors.find(key);
            accumulator.color +=
                color != cloud_colors.end() ? color->second : Vec3f{0.7F, 0.7F, 0.7F};
            accumulator.weight += 1.F;
        }
    }
    mesh.vertices.resize(vertex_accum.size());
    mesh.colors.resize(vertex_accum.size());
    for (std::size_t i = 0; i < vertex_accum.size(); ++i) {
        const float inverse = 1.F / std::max(vertex_accum[i].weight, 1.F);
        mesh.vertices[i] = vertex_accum[i].position * inverse;
        mesh.colors[i] = vertex_accum[i].color * inverse;
    }

    struct FaceKey {
        int a{}, b{}, c{};
        bool operator==(const FaceKey&) const = default;
    };
    struct FaceHash {
        std::size_t operator()(const FaceKey& face) const noexcept {
            return (static_cast<std::size_t>(face.a) * 73856093u) ^
                   (static_cast<std::size_t>(face.b) * 19349663u) ^
                   (static_cast<std::size_t>(face.c) * 83492791u);
        }
    };
    std::unordered_set<FaceKey, FaceHash> unique_faces;
    for (std::size_t vi = 0; vi < locals.size(); ++vi) {
        for (const Eigen::Vector3i& local_face : locals[vi].faces) {
            Eigen::Vector3i face{
                remaps[vi][static_cast<std::size_t>(local_face[0])],
                remaps[vi][static_cast<std::size_t>(local_face[1])],
                remaps[vi][static_cast<std::size_t>(local_face[2])]};
            if (face[0] == face[1] || face[1] == face[2] || face[2] == face[0])
                continue;
            std::array<int, 3> sorted{face[0], face[1], face[2]};
            std::sort(sorted.begin(), sorted.end());
            if (!unique_faces.emplace(
                    FaceKey{sorted[0], sorted[1], sorted[2]})
                     .second)
                continue;
            mesh.faces.push_back(face);
        }
    }

    // Edge-connected component cleanup. Connecting through edges rather than
    // a single welded vertex avoids retaining point-touching speckle islands.
    if (!mesh.faces.empty()) {
        std::vector<int> parent(mesh.faces.size());
        std::vector<unsigned> component_size(mesh.faces.size(), 1);
        for (std::size_t i = 0; i < parent.size(); ++i) parent[i] = static_cast<int>(i);
        auto find = [&](int value) {
            int root = value;
            while (parent[static_cast<std::size_t>(root)] != root)
                root = parent[static_cast<std::size_t>(root)];
            while (parent[static_cast<std::size_t>(value)] != value) {
                const int next = parent[static_cast<std::size_t>(value)];
                parent[static_cast<std::size_t>(value)] = root;
                value = next;
            }
            return root;
        };
        auto unite = [&](int a, int b) {
            a = find(a);
            b = find(b);
            if (a == b) return;
            if (component_size[static_cast<std::size_t>(a)] <
                component_size[static_cast<std::size_t>(b)])
                std::swap(a, b);
            parent[static_cast<std::size_t>(b)] = a;
            component_size[static_cast<std::size_t>(a)] +=
                component_size[static_cast<std::size_t>(b)];
        };
        struct EdgeKey {
            int a{}, b{};
            bool operator==(const EdgeKey&) const = default;
        };
        struct EdgeHash {
            std::size_t operator()(const EdgeKey& edge) const noexcept {
                return (static_cast<std::size_t>(edge.a) * 73856093u) ^
                       (static_cast<std::size_t>(edge.b) * 19349663u);
            }
        };
        struct EdgeState {
            int first_face{-1};
            std::uint8_t count{0};
        };
        std::unordered_map<EdgeKey, EdgeState, EdgeHash> edges;
        edges.reserve(mesh.faces.size() * 2);
        std::vector<std::uint8_t> accepted(mesh.faces.size(), 0);
        for (std::size_t fi = 0; fi < mesh.faces.size(); ++fi) {
            const auto& face = mesh.faces[fi];
            std::array<EdgeKey, 3> face_edges{};
            bool manifold = true;
            for (int edge = 0; edge < 3; ++edge) {
                int a = face[edge];
                int b = face[(edge + 1) % 3];
                if (a > b) std::swap(a, b);
                face_edges[static_cast<std::size_t>(edge)] = EdgeKey{a, b};
                const auto it = edges.find(EdgeKey{a, b});
                if (it != edges.end() && it->second.count >= 2) {
                    manifold = false;
                    break;
                }
            }
            if (!manifold) continue;
            accepted[fi] = 1;
            for (const EdgeKey& edge : face_edges) {
                auto [it, inserted] = edges.emplace(
                    edge, EdgeState{static_cast<int>(fi), 1});
                if (!inserted) {
                    unite(static_cast<int>(fi), it->second.first_face);
                    ++it->second.count;
                }
            }
        }
        std::vector<unsigned> final_size(mesh.faces.size(), 0);
        for (std::size_t i = 0; i < mesh.faces.size(); ++i)
            if (accepted[i])
                ++final_size[static_cast<std::size_t>(find(static_cast<int>(i)))];
        std::vector<Eigen::Vector3i> kept_faces;
        kept_faces.reserve(mesh.faces.size());
        const unsigned min_component =
            std::max(1U, options.mesh_min_component_faces);
        for (std::size_t i = 0; i < mesh.faces.size(); ++i) {
            if (accepted[i] &&
                final_size[static_cast<std::size_t>(find(static_cast<int>(i)))] >=
                    min_component)
                kept_faces.push_back(mesh.faces[i]);
        }
        mesh.faces = std::move(kept_faces);
    }

    // Split bow-tie vertices: edge-manifold meshes can still have multiple
    // disconnected face fans touching at one vertex, which breaks UV unwrap.
    // Each disconnected one-ring fan gets its own copy of the vertex.
    {
        const std::size_t original_vertex_count = mesh.vertices.size();
        std::vector<std::vector<int>> incident(original_vertex_count);
        for (std::size_t fi = 0; fi < mesh.faces.size(); ++fi)
            for (int k = 0; k < 3; ++k)
                incident[static_cast<std::size_t>(mesh.faces[fi][k])].push_back(
                    static_cast<int>(fi));

        for (std::size_t vertex = 0; vertex < original_vertex_count; ++vertex) {
            const std::vector<int>& faces = incident[vertex];
            if (faces.size() <= 1) continue;
            std::vector<int> parent(faces.size());
            for (std::size_t i = 0; i < parent.size(); ++i)
                parent[i] = static_cast<int>(i);
            auto find_local = [&](int value) {
                int root = value;
                while (parent[static_cast<std::size_t>(root)] != root)
                    root = parent[static_cast<std::size_t>(root)];
                while (parent[static_cast<std::size_t>(value)] != value) {
                    const int next = parent[static_cast<std::size_t>(value)];
                    parent[static_cast<std::size_t>(value)] = root;
                    value = next;
                }
                return root;
            };
            auto unite_local = [&](int a, int b) {
                a = find_local(a);
                b = find_local(b);
                if (a != b) parent[static_cast<std::size_t>(b)] = a;
            };
            for (std::size_t i = 0; i < faces.size(); ++i) {
                const Eigen::Vector3i& a =
                    mesh.faces[static_cast<std::size_t>(faces[i])];
                for (std::size_t j = i + 1; j < faces.size(); ++j) {
                    const Eigen::Vector3i& b =
                        mesh.faces[static_cast<std::size_t>(faces[j])];
                    bool share_edge = false;
                    for (int ai = 0; ai < 3 && !share_edge; ++ai) {
                        if (a[ai] == static_cast<int>(vertex)) continue;
                        for (int bi = 0; bi < 3; ++bi) {
                            if (a[ai] == b[bi]) {
                                share_edge = true;
                                break;
                            }
                        }
                    }
                    if (share_edge)
                        unite_local(static_cast<int>(i), static_cast<int>(j));
                }
            }

            const int first_root = find_local(0);
            std::unordered_map<int, int> duplicated;
            for (std::size_t i = 0; i < faces.size(); ++i) {
                const int root = find_local(static_cast<int>(i));
                if (root == first_root) continue;
                auto [it, inserted] = duplicated.emplace(
                    root, static_cast<int>(mesh.vertices.size()));
                if (inserted) {
                    mesh.vertices.push_back(mesh.vertices[vertex]);
                    mesh.colors.push_back(mesh.colors[vertex]);
                }
                Eigen::Vector3i& face =
                    mesh.faces[static_cast<std::size_t>(faces[i])];
                for (int k = 0; k < 3; ++k)
                    if (face[k] == static_cast<int>(vertex)) {
                        face[k] = it->second;
                        break;
                    }
            }
        }
    }

    // Compact unused vertices and recompute area-weighted normals.
    std::vector<int> compact(mesh.vertices.size(), -1);
    for (const auto& face : mesh.faces)
        for (int k = 0; k < 3; ++k) compact[static_cast<std::size_t>(face[k])] = 0;
    std::vector<Vec3f> vertices;
    std::vector<Vec3f> colors;
    vertices.reserve(mesh.vertices.size());
    colors.reserve(mesh.colors.size());
    for (std::size_t i = 0; i < compact.size(); ++i) {
        if (compact[i] < 0) continue;
        compact[i] = static_cast<int>(vertices.size());
        vertices.push_back(mesh.vertices[i]);
        colors.push_back(mesh.colors[i]);
    }
    for (auto& face : mesh.faces)
        for (int k = 0; k < 3; ++k)
            face[k] = compact[static_cast<std::size_t>(face[k])];
    mesh.vertices = std::move(vertices);
    mesh.colors = std::move(colors);
    mesh.normals.assign(mesh.vertices.size(), Vec3f::Zero());
    for (const auto& face : mesh.faces) {
        const Vec3f normal =
            (mesh.vertices[static_cast<std::size_t>(face[1])] -
             mesh.vertices[static_cast<std::size_t>(face[0])])
                .cross(
                    mesh.vertices[static_cast<std::size_t>(face[2])] -
                    mesh.vertices[static_cast<std::size_t>(face[0])]);
        for (int k = 0; k < 3; ++k)
            mesh.normals[static_cast<std::size_t>(face[k])] += normal;
    }
    for (Vec3f& normal : mesh.normals)
        if (normal.squaredNorm() > 1e-12F) normal.normalize();

    core::Logger::instance().info(
        "mvs mesh projective: footprint=", pixel_footprint,
        " weld=", quant, " vertices=", mesh.vertices.size(),
        " faces=", mesh.faces.size());
}

}  // namespace

void reconstruct_mesh(MvsScene& scene, const DensifyOptions& options) {
    core::StageScope stage("mvs.mesh");
    scene.mesh = {};
    if (options.mesh_method == MeshMethod::none) {
        stage.finish();
        return;
    }
    if (options.mesh_method == MeshMethod::delaunay_cut) {
        try {
            if (!detail::reconstruct_mesh_global_cgal(scene, options)) {
                core::Logger::instance().warning(
                    "mvs global delaunay unavailable/empty; fallback projective");
                reconstruct_mesh_projective(scene, options);
            }
        } catch (const std::exception& ex) {
            core::Logger::instance().warning(
                "mvs delaunay_cut failed: ", ex.what(), "; fallback projective");
            reconstruct_mesh_projective(scene, options);
        }
    } else {
        reconstruct_mesh_projective(scene, options);
    }
    detail::clean_mesh(
        scene.mesh, options, scene.roi.valid ? &scene.roi : nullptr);
    stage.finish();
}

}  // namespace aetherscan::mvs
