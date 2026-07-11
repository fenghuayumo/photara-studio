#include "sfm/global_rotation.hpp"

#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

namespace aetherscan::sfm {
namespace {

constexpr double k_pi = 3.14159265358979323846;

struct Edge {
    Index a{};
    Index b{};
    Mat3 relative{Mat3::Identity()};
    double weight{};
    Index pair_index{};
};

class DisjointSet {
public:
    explicit DisjointSet(const std::size_t size) : parent_(size), rank_(size, 0) {
        std::iota(parent_.begin(), parent_.end(), Index{0});
    }

    Index find(Index x) {
        if (parent_[x] != x) parent_[x] = find(parent_[x]);
        return parent_[x];
    }

    bool unite(Index a, Index b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
        return true;
    }

private:
    std::vector<Index> parent_;
    std::vector<unsigned> rank_;
};

Vec3 rotation_log(const Mat3& rotation) {
    Eigen::AngleAxisd angle_axis(rotation);
    if (!std::isfinite(angle_axis.angle()) || angle_axis.angle() < 1e-12)
        return Vec3::Zero();
    return angle_axis.axis() * angle_axis.angle();
}

Mat3 rotation_exp(const Vec3& tangent) {
    const double angle = tangent.norm();
    if (angle < 1e-12) return Mat3::Identity();
    return Eigen::AngleAxisd(angle, tangent / angle).toRotationMatrix();
}

std::vector<Edge> collect_edges(const Scene& scene, const bool weighted) {
    std::vector<Edge> edges;
    edges.reserve(scene.pairs.size());
    for (Index pair_index = 0; pair_index < scene.pairs.size(); ++pair_index) {
        const ImagePair& pair = scene.pairs[pair_index];
        if (!pair.active || !pair.relative_pose || pair.matches.empty()) continue;
        const double weight =
            weighted ? pair.composite_weight() : static_cast<double>(pair.num_inliers());
        if (weight <= 0.0) continue;
        edges.push_back({pair.id1, pair.id2, pair.relative_pose->R, weight, pair_index});
    }
    return edges;
}

bool initialize_mst(
    const std::size_t image_count,
    const std::vector<Edge>& edges,
    std::vector<Mat3>& rotations,
    std::vector<char>& valid,
    Index& root) {
    if (edges.empty()) return false;

    DisjointSet components(image_count);
    for (const Edge& edge : edges) components.unite(edge.a, edge.b);
    std::unordered_map<Index, unsigned> component_sizes;
    for (Index i = 0; i < image_count; ++i) ++component_sizes[components.find(i)];
    Index largest = components.find(edges.front().a);
    for (const auto& [component, size] : component_sizes) {
        if (size > component_sizes[largest]) largest = component;
    }
    if (component_sizes[largest] < 2) return false;

    std::vector<Index> order(edges.size());
    std::iota(order.begin(), order.end(), Index{0});
    std::sort(order.begin(), order.end(), [&](const Index lhs, const Index rhs) {
        return edges[lhs].weight > edges[rhs].weight;
    });

    DisjointSet tree_sets(image_count);
    std::vector<std::vector<std::pair<Index, Index>>> tree(image_count);
    for (Index edge_index : order) {
        const Edge& edge = edges[edge_index];
        if (components.find(edge.a) != largest || components.find(edge.b) != largest)
            continue;
        if (!tree_sets.unite(edge.a, edge.b)) continue;
        tree[edge.a].push_back({edge.b, edge_index});
        tree[edge.b].push_back({edge.a, edge_index});
    }

    root = k_invalid;
    for (Index i = 0; i < image_count; ++i) {
        if (components.find(i) != largest) continue;
        if (root == k_invalid || tree[i].size() > tree[root].size()) root = i;
    }
    if (root == k_invalid) return false;

    rotations.assign(image_count, Mat3::Identity());
    valid.assign(image_count, 0);
    valid[root] = 1;
    std::queue<Index> queue;
    queue.push(root);
    while (!queue.empty()) {
        const Index current = queue.front();
        queue.pop();
        for (const auto& [child, edge_index] : tree[current]) {
            if (valid[child]) continue;
            const Edge& edge = edges[edge_index];
            if (edge.a == current)
                rotations[child] = edge.relative * rotations[current];
            else
                rotations[child] = edge.relative.transpose() * rotations[current];
            valid[child] = 1;
            queue.push(child);
        }
    }
    return true;
}

double pair_rotation_error(const Edge& edge, const std::vector<Mat3>& rotations) {
    return rotation_log(
        rotations[edge.b].transpose() * edge.relative * rotations[edge.a]).norm();
}

}  // namespace

GlobalRotationSummary estimate_global_rotations(
    Scene& scene,
    const GlobalRotationOptions& options) {
    GlobalRotationSummary summary;
    std::vector<Edge> edges = collect_edges(scene, options.use_pair_weights);
    if (edges.empty()) return summary;

    std::vector<Mat3> rotations;
    std::vector<char> valid;
    Index fixed = k_invalid;
    if (!initialize_mst(scene.images.size(), edges, rotations, valid, fixed)) return summary;

    std::vector<Index> free_images;
    std::vector<Index> image_to_free(scene.images.size(), k_invalid);
    for (Index image_id = 0; image_id < valid.size(); ++image_id) {
        if (!valid[image_id] || image_id == fixed) continue;
        image_to_free[image_id] = static_cast<Index>(free_images.size());
        free_images.push_back(image_id);
    }
    if (free_images.empty()) return summary;

    double max_base_weight = 0.0;
    for (const Edge& edge : edges) max_base_weight = std::max(max_base_weight, edge.weight);
    max_base_weight = std::max(max_base_weight, 1e-12);
    const double sigma = options.irls_sigma_deg * k_pi / 180.0;
    const unsigned total_iterations =
        options.max_l1_iterations + options.max_irls_iterations;

    for (unsigned iteration = 0; iteration < total_iterations; ++iteration) {
        std::vector<Eigen::Triplet<double>> normal_triplets;
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(3 * free_images.size());
        std::vector<Eigen::Matrix3d> diagonal(free_images.size(), Mat3::Zero());
        struct OffDiagonal {
            Index i{};
            Index j{};
            double weight{};
        };
        std::vector<OffDiagonal> off_diagonal;
        off_diagonal.reserve(edges.size());

        for (const Edge& edge : edges) {
            if (!valid[edge.a] || !valid[edge.b]) continue;
            const Vec3 residual = -rotation_log(
                rotations[edge.b].transpose() * edge.relative * rotations[edge.a]);
            const double residual_norm = residual.norm();
            double robust_weight = 1.0;
            if (iteration < options.max_l1_iterations) {
                robust_weight = 1.0 / std::max(residual_norm, 1e-3);
            } else {
                const double sigma_sq = sigma * sigma;
                const double denominator = sigma_sq + residual.squaredNorm();
                robust_weight = sigma_sq / (denominator * denominator);
            }
            const double base_weight = options.use_pair_weights
                ? std::sqrt(edge.weight / max_base_weight)
                : 1.0;
            const double weight = std::max(robust_weight * base_weight, 1e-12);

            const Index ia = image_to_free[edge.a];
            const Index ib = image_to_free[edge.b];
            if (ia != k_invalid) {
                diagonal[ia].diagonal().array() += weight;
                rhs.segment<3>(3 * ia) -= weight * residual;
            }
            if (ib != k_invalid) {
                diagonal[ib].diagonal().array() += weight;
                rhs.segment<3>(3 * ib) += weight * residual;
            }
            if (ia != k_invalid && ib != k_invalid)
                off_diagonal.push_back({ia, ib, weight});
        }

        for (Index i = 0; i < diagonal.size(); ++i) {
            for (int axis = 0; axis < 3; ++axis)
                normal_triplets.emplace_back(3 * i + axis, 3 * i + axis, diagonal[i](axis, axis));
        }
        for (const OffDiagonal& off : off_diagonal) {
            for (int axis = 0; axis < 3; ++axis) {
                normal_triplets.emplace_back(3 * off.i + axis, 3 * off.j + axis, -off.weight);
                normal_triplets.emplace_back(3 * off.j + axis, 3 * off.i + axis, -off.weight);
            }
        }

        Eigen::SparseMatrix<double> normal(3 * free_images.size(), 3 * free_images.size());
        normal.setFromTriplets(normal_triplets.begin(), normal_triplets.end());
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(normal);
        if (solver.info() != Eigen::Success) return summary;
        const Eigen::VectorXd step = solver.solve(rhs);
        if (solver.info() != Eigen::Success || !step.allFinite()) return summary;

        double average_step = 0.0;
        for (Index i = 0; i < free_images.size(); ++i) {
            const Vec3 update = step.segment<3>(3 * i);
            rotations[free_images[i]] *= rotation_exp(-update);
            average_step += update.norm();
        }
        average_step /= static_cast<double>(free_images.size());
        summary.iterations = iteration + 1;
        if (average_step < options.step_convergence_threshold) break;
    }

    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        scene.images[image_id].registered = valid[image_id] != 0;
        if (valid[image_id]) {
            scene.images[image_id].pose.R = rotations[image_id];
            scene.images[image_id].pose.C.setZero();
            ++summary.estimated_images;
        }
    }

    const double max_error = options.max_relative_rotation_error_deg * k_pi / 180.0;
    if (max_error > 0.0) {
        for (const Edge& edge : edges) {
            if (!valid[edge.a] || !valid[edge.b]) continue;
            if (pair_rotation_error(edge, rotations) <= max_error) continue;
            scene.pairs[edge.pair_index].active = false;
            ++summary.filtered_pairs;
        }
    }

    summary.success = summary.estimated_images >= 2;
    summary.used_pairs = static_cast<unsigned>(edges.size());
    summary.fixed_image = fixed;
    return summary;
}

}  // namespace aetherscan::sfm
