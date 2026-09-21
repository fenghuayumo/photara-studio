#include "sfm/global_rotation.hpp"

#include "core/logging.hpp"

#include <Eigen/IterativeLinearSolvers>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

namespace photara::sfm {
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
    explicit DisjointSet(std::size_t size) : parent_(size), rank_(size, 0) {
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
    const Eigen::AngleAxisd angle_axis(rotation);
    if (!std::isfinite(angle_axis.angle()) || angle_axis.angle() < 1e-12)
        return Vec3::Zero();
    return angle_axis.axis() * angle_axis.angle();
}

Mat3 rotation_exp(const Vec3& tangent) {
    const double angle = tangent.norm();
    if (angle < 1e-12) return Mat3::Identity();
    return Eigen::AngleAxisd(angle, tangent / angle).toRotationMatrix();
}

std::vector<Edge> collect_edges(
    const Scene& scene, bool use_pair_weights, bool reject_planar_pairs) {
    std::vector<Edge> edges;
    std::vector<Edge> planar_fallbacks;
    edges.reserve(scene.pairs.size());
    planar_fallbacks.reserve(scene.pairs.size() / 4);
    for (Index pair_index = 0; pair_index < scene.pairs.size(); ++pair_index) {
        const ImagePair& pair = scene.pairs[pair_index];
        if (!pair.active || !pair.relative_pose.has_value()) continue;
        if (pair.id1 >= scene.images.size() ||
            pair.id2 >= scene.images.size() ||
            pair.id1 == pair.id2) {
            core::Logger::instance().warning(
                "global rotation: ignoring invalid pair index=", pair_index,
                " endpoints=", pair.id1, ',', pair.id2,
                " images=", scene.images.size());
            continue;
        }
        // HasValidWeight() is checked before openMVS chooses either weight.
        if (pair.composite_weight() <= 0.F) continue;
        const double weight = use_pair_weights
            ? static_cast<double>(pair.composite_weight())
            : static_cast<double>(pair.num_inliers());
        if (weight <= 0.0) continue;
        Edge edge{
            pair.id1, pair.id2, pair.relative_pose->R, weight, pair_index};
        if (reject_planar_pairs && pair.degenerate_planar)
            planar_fallbacks.push_back(std::move(edge));
        else
            edges.push_back(std::move(edge));
    }
    if (reject_planar_pairs && !planar_fallbacks.empty()) {
        DisjointSet components(scene.images.size());
        for (const Edge& edge : edges) components.unite(edge.a, edge.b);
        std::stable_sort(
            planar_fallbacks.begin(), planar_fallbacks.end(),
            [](const Edge& left, const Edge& right) {
                return left.weight > right.weight;
            });
        for (Edge& edge : planar_fallbacks)
            if (components.unite(edge.a, edge.b))
                edges.push_back(std::move(edge));
    }
    return edges;
}

bool initialize_from_mst(
    std::size_t image_count,
    const std::vector<Edge>& edges,
    std::vector<Mat3>& rotations,
    std::vector<char>& valid,
    Index& fixed) {
    DisjointSet components(image_count);
    for (const Edge& edge : edges) components.unite(edge.a, edge.b);

    std::unordered_map<Index, unsigned> component_sizes;
    for (Index i = 0; i < image_count; ++i) ++component_sizes[components.find(i)];
    Index largest = components.find(0);
    for (Index i = 1; i < image_count; ++i) {
        const Index component = components.find(i);
        if (component_sizes[component] > component_sizes[largest])
            largest = component;
    }
    if (component_sizes[largest] < 2) return false;

    std::vector<Index> order(edges.size());
    std::iota(order.begin(), order.end(), Index{0});
    std::stable_sort(order.begin(), order.end(), [&](Index lhs, Index rhs) {
        return edges[lhs].weight > edges[rhs].weight;
    });

    DisjointSet forest(image_count);
    std::vector<std::vector<std::pair<Index, Index>>> adjacency(image_count);
    for (Index edge_index : order) {
        const Edge& edge = edges[edge_index];
        if (!forest.unite(edge.a, edge.b)) continue;
        if (components.find(edge.a) != largest) continue;
        adjacency[edge.a].push_back({edge.b, edge_index});
        adjacency[edge.b].push_back({edge.a, edge_index});
    }

    fixed = k_invalid;
    for (Index i = 0; i < image_count; ++i) {
        if (components.find(i) != largest) continue;
        if (fixed == k_invalid || adjacency[i].size() > adjacency[fixed].size())
            fixed = i;
    }
    if (fixed == k_invalid) return false;

    rotations.assign(image_count, Mat3::Identity());
    valid.assign(image_count, 0);
    valid[fixed] = 1;
    std::queue<Index> queue;
    queue.push(fixed);
    while (!queue.empty()) {
        const Index current = queue.front();
        queue.pop();
        for (const auto [child, edge_index] : adjacency[current]) {
            if (valid[child]) continue;
            const Edge& edge = edges[edge_index];
            if (edge.a == current)
                rotations[child] = edge.relative * rotations[current];
            else
                rotations[child] =
                    edge.relative.transpose() * rotations[current];
            valid[child] = 1;
            queue.push(child);
        }
    }
    return true;
}

Eigen::VectorXd shrinkage(const Eigen::VectorXd& value, double kappa) {
    const Eigen::VectorXd plus = value.array() + kappa;
    const Eigen::VectorXd minus = value.array() - kappa;
    return plus.cwiseMin(0.0) + minus.cwiseMax(0.0);
}

class LadSolver {
public:
    explicit LadSolver(const Eigen::SparseMatrix<double>& matrix)
        : matrix_(matrix),
          normal_matrix_(matrix_.transpose() * matrix_) {
        normal_matrix_.makeCompressed();
        solver_.setMaxIterations(std::max(
            100,
            static_cast<int>(std::min<Eigen::Index>(
                2000, normal_matrix_.cols() * 4))));
        solver_.setTolerance(1e-10);
        solver_.compute(normal_matrix_);
    }

    bool valid() const { return solver_.info() == Eigen::Success; }

    bool solve(const Eigen::VectorXd& rhs, Eigen::VectorXd& solution) {
        Eigen::VectorXd z = Eigen::VectorXd::Zero(matrix_.rows());
        Eigen::VectorXd old_z(matrix_.rows());
        Eigen::VectorXd u = Eigen::VectorXd::Zero(matrix_.rows());
        Eigen::VectorXd ax(matrix_.rows());
        Eigen::VectorXd relaxed_ax(matrix_.rows());
        const double rhs_norm = rhs.norm();
        const double primal_base = std::sqrt(static_cast<double>(matrix_.rows())) * 1e-4;
        const double dual_base = std::sqrt(static_cast<double>(matrix_.cols())) * 1e-4;

        // openMVS intentionally uses ten inner ADMM iterations here.
        for (int iteration = 0; iteration < 10; ++iteration) {
            const Eigen::VectorXd normal_rhs =
                matrix_.transpose() * (rhs + z - u);
            solution = solver_.solve(normal_rhs);
            if (solver_.info() != Eigen::Success) return false;
            ax.noalias() = matrix_ * solution;
            relaxed_ax = ax;  // rho = alpha = 1
            std::swap(z, old_z);
            z = shrinkage(relaxed_ax - rhs + u, 1.0);
            u.noalias() += relaxed_ax - z - rhs;

            const double primal = (ax - z - rhs).norm();
            const double dual =
                (-matrix_.transpose() * (z - old_z)).norm();
            const double eps_primal = primal_base + 1e-2 *
                std::max(rhs_norm, std::max(ax.norm(), z.norm()));
            const double eps_dual =
                dual_base + 1e-2 * (matrix_.transpose() * u).norm();
            if (primal < eps_primal && dual < eps_dual) break;
        }
        return true;
    }

private:
    const Eigen::SparseMatrix<double>& matrix_;
    Eigen::SparseMatrix<double> normal_matrix_;
    Eigen::ConjugateGradient<
        Eigen::SparseMatrix<double>,
        Eigen::Lower | Eigen::Upper,
        Eigen::DiagonalPreconditioner<double>>
        solver_;
};

void compute_residuals(
    const std::vector<Edge>& edges,
    const std::vector<Mat3>& rotations,
    const std::vector<char>& valid,
    Eigen::VectorXd& residuals) {
    Index row = 0;
    for (const Edge& edge : edges) {
        if (!valid[edge.a] || !valid[edge.b]) continue;
        residuals.segment<3>(row) = -rotation_log(
            rotations[edge.b].transpose() * edge.relative * rotations[edge.a]);
        row += 3;
    }
}

double apply_step(
    const Eigen::VectorXd& step,
    const std::vector<Index>& free_images,
    std::vector<Mat3>& rotations) {
    double total = 0.0;
    for (Index i = 0; i < free_images.size(); ++i) {
        const Vec3 update = step.segment<3>(3 * i);
        rotations[free_images[i]] *= rotation_exp(-update);
        total += update.norm();
    }
    return total / static_cast<double>(free_images.size());
}

}  // namespace

GlobalRotationSummary estimate_global_rotations(
    Scene& scene,
    const GlobalRotationOptions& options) {
    GlobalRotationSummary summary;
    const std::vector<Edge> edges = collect_edges(
        scene, options.use_pair_weights, options.reject_planar_pairs);
    core::Logger::instance().debug(
        "global rotation: collected edges=", edges.size(),
        " pairs=", scene.pairs.size(),
        " images=", scene.images.size());
    if (edges.empty() || scene.images.empty()) return summary;

    std::vector<Mat3> rotations;
    std::vector<char> valid;
    Index fixed = k_invalid;
    if (!initialize_from_mst(
            scene.images.size(), edges, rotations, valid, fixed))
        return summary;
    core::Logger::instance().debug(
        "global rotation: initialized MST fixed=", fixed);

    std::vector<Index> free_images;
    std::vector<Index> image_to_free(scene.images.size(), k_invalid);
    for (Index image_id = 0; image_id < valid.size(); ++image_id) {
        if (!valid[image_id] || image_id == fixed) continue;
        image_to_free[image_id] = static_cast<Index>(free_images.size());
        free_images.push_back(image_id);
    }
    if (free_images.empty()) return summary;

    std::vector<Eigen::Triplet<double>> triplets;
    std::vector<double> row_weights;
    unsigned valid_edge_count = 0;
    triplets.reserve(edges.size() * 6);
    row_weights.reserve(edges.size() * 3);
    for (const Edge& edge : edges) {
        if (!valid[edge.a] || !valid[edge.b]) continue;
        const Index row = valid_edge_count * 3;
        const Index free_a = image_to_free[edge.a];
        const Index free_b = image_to_free[edge.b];
        for (int axis = 0; axis < 3; ++axis) {
            if (free_a != k_invalid)
                triplets.emplace_back(row + axis, free_a * 3 + axis, -1.0);
            if (free_b != k_invalid)
                triplets.emplace_back(row + axis, free_b * 3 + axis, 1.0);
            row_weights.push_back(options.use_pair_weights ? edge.weight : 1.0);
        }
        ++valid_edge_count;
    }
    if (valid_edge_count == 0) return summary;

    Eigen::SparseMatrix<double> tangent_matrix(
        valid_edge_count * 3, free_images.size() * 3);
    tangent_matrix.setFromTriplets(triplets.begin(), triplets.end());
    core::Logger::instance().debug(
        "global rotation: tangent rows=", tangent_matrix.rows(),
        " cols=", tangent_matrix.cols(),
        " nonzeros=", tangent_matrix.nonZeros());
    Eigen::ArrayXd base_weights =
        Eigen::Map<const Eigen::ArrayXd>(row_weights.data(), row_weights.size());
    // OpenMVS relies on a direct sparse factorization for the LAD pass.  The
    // iterative fallback below is much more sensitive to the raw magnitude of
    // composite weights, so normalize by the median while preserving every
    // relative weight.
    {
        std::vector<double> finite_weights;
        finite_weights.reserve(row_weights.size());
        for (const double weight : row_weights) {
            if (weight > 0.0 && std::isfinite(weight))
                finite_weights.push_back(weight);
        }
        if (!finite_weights.empty()) {
            std::nth_element(
                finite_weights.begin(),
                finite_weights.begin() + finite_weights.size() / 2,
                finite_weights.end());
            const double weight_scale =
                finite_weights[finite_weights.size() / 2];
            if (std::isfinite(weight_scale) && weight_scale > 0.0)
                base_weights /= weight_scale;
        }
    }
    Eigen::VectorXd residuals(tangent_matrix.rows());
    Eigen::VectorXd step(tangent_matrix.cols());
    compute_residuals(edges, rotations, valid, residuals);
    core::Logger::instance().debug(
        "global rotation: initial residual_norm=", residuals.norm());

    if (options.max_l1_iterations > 0) {
        const Eigen::SparseMatrix<double> weighted_matrix =
            base_weights.matrix().asDiagonal() * tangent_matrix;
        LadSolver lad(weighted_matrix);
        if (!lad.valid()) {
            core::Logger::instance().warning(
                "global rotation: LAD factorization failed; continuing with IRLS");
        } else {
            core::Logger::instance().debug(
                "global rotation: LAD initialized iterations=",
                options.max_l1_iterations);
            double current_norm = 0.0;
            for (unsigned iteration = 0; iteration < options.max_l1_iterations;
                 ++iteration) {
                step.setZero();
                if (!lad.solve(
                        base_weights.matrix().asDiagonal() * residuals,
                        step) ||
                    !step.allFinite())
                    break;
                const double previous_norm = current_norm;
                current_norm = step.norm();
                const double average_step =
                    apply_step(step, free_images, rotations);
                compute_residuals(edges, rotations, valid, residuals);
                ++summary.iterations;
                if (average_step < options.step_convergence_threshold ||
                    std::abs(previous_norm - current_norm) < 1e-10)
                    break;
            }
            core::Logger::instance().debug(
                "global rotation: LAD finished residual_norm=",
                residuals.norm());
        }
    }

    if (options.max_irls_iterations > 0) {
        Eigen::ConjugateGradient<
            Eigen::SparseMatrix<double>,
            Eigen::Lower | Eigen::Upper,
            Eigen::DiagonalPreconditioner<double>>
            solver;
        solver.setMaxIterations(std::max(
            100,
            static_cast<int>(std::min<Eigen::Index>(
                2000, tangent_matrix.cols() * 4))));
        solver.setTolerance(1e-10);
        const double sigma = options.irls_sigma_deg * k_pi / 180.0;
        Eigen::ArrayXd robust_weights(tangent_matrix.rows());
        for (unsigned iteration = 0; iteration < options.max_irls_iterations;
             ++iteration) {
            for (Index row = 0; row < valid_edge_count * 3; row += 3) {
                const double residual_sq =
                    residuals.segment<3>(row).squaredNorm();
                double weight = 0.0;
                if (options.weight_type ==
                    GlobalRotationOptions::WeightType::geman_mcclure) {
                    const double sigma_sq = sigma * sigma;
                    const double denominator = sigma_sq + residual_sq;
                    weight = sigma_sq / (denominator * denominator);
                } else {
                    weight =
                        1.0 / std::max(std::sqrt(residual_sq), sigma);
                }
                robust_weights.segment<3>(row) =
                    weight * base_weights.segment<3>(row);
            }
            const Eigen::SparseMatrix<double> at_weight =
                tangent_matrix.transpose() *
                robust_weights.matrix().asDiagonal();
            Eigen::SparseMatrix<double> normal_matrix =
                at_weight * tangent_matrix;
            normal_matrix.makeCompressed();
            solver.compute(normal_matrix);
            if (solver.info() != Eigen::Success) return summary;
            step = solver.solve(at_weight * residuals);
            if (solver.info() != Eigen::Success || !step.allFinite())
                return summary;
            const double average_step = apply_step(step, free_images, rotations);
            compute_residuals(edges, rotations, valid, residuals);
            ++summary.iterations;
            if (average_step < options.step_convergence_threshold) break;
        }
        core::Logger::instance().debug(
            "global rotation: IRLS finished residual_norm=", residuals.norm());
    }

    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        scene.images[image_id].registered = valid[image_id] != 0;
        if (!valid[image_id]) continue;
        scene.images[image_id].pose.R = rotations[image_id];
        scene.images[image_id].pose.C.setZero();
        ++summary.estimated_images;
    }

    const double max_error =
        options.max_relative_rotation_error_deg * k_pi / 180.0;
    if (max_error > 0.0) {
        for (const Edge& edge : edges) {
            if (!valid[edge.a] || !valid[edge.b]) continue;
            const double error = rotation_log(
                rotations[edge.b].transpose() * edge.relative *
                rotations[edge.a]).norm();
            if (error <= max_error) continue;
            scene.pairs[edge.pair_index].active = false;
            ++summary.filtered_pairs;
        }
    }

    summary.success = true;
    summary.used_pairs = valid_edge_count;
    summary.fixed_image = fixed;
    return summary;
}

}  // namespace photara::sfm
