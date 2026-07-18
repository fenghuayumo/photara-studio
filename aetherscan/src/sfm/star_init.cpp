#include "sfm/star_init.hpp"

#include "core/logging.hpp"
#include "sfm/bundle.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aetherscan::sfm {
namespace {

bool triangulate_midpoint(
    const Mat3& R,
    const Vec3& C,
    const Vec3& b1,
    const Vec3& b2,
    Vec3& X) {
    const Vec3 d1 = b1.normalized();
    const Vec3 d2 = (R.transpose() * b2).normalized();
    const Mat3 A =
        Mat3::Identity() - d1 * d1.transpose() + Mat3::Identity() - d2 * d2.transpose();
    const Vec3 rhs = (Mat3::Identity() - d2 * d2.transpose()) * C;
    Eigen::LDLT<Mat3> ldlt(A);
    if (ldlt.info() != Eigen::Success) return false;
    X = ldlt.solve(rhs);
    return X.allFinite();
}

struct ScaleConstraint {
    Index first{k_invalid};
    Index second{k_invalid};
    double log_ratio{};
    double weight{1.0};
};

std::vector<double> solve_pair_scales(
    const std::size_t pair_count,
    const Index anchor_pair,
    const std::vector<ScaleConstraint>& constraints) {
    std::vector<double> scales(pair_count, 1.0);
    if (anchor_pair == k_invalid || anchor_pair >= pair_count ||
        constraints.empty()) {
        return scales;
    }

    std::unordered_map<Index, std::vector<std::size_t>> adjacency;
    adjacency.reserve(constraints.size() * 2);
    for (std::size_t i = 0; i < constraints.size(); ++i) {
        adjacency[constraints[i].first].push_back(i);
        adjacency[constraints[i].second].push_back(i);
    }

    std::unordered_set<Index> connected;
    connected.reserve(adjacency.size());
    std::queue<Index> pending;
    connected.insert(anchor_pair);
    pending.push(anchor_pair);
    while (!pending.empty()) {
        const Index pair = pending.front();
        pending.pop();
        const auto adjacency_it = adjacency.find(pair);
        if (adjacency_it == adjacency.end()) continue;
        for (const std::size_t constraint_id : adjacency_it->second) {
            const ScaleConstraint& constraint = constraints[constraint_id];
            const Index neighbor =
                constraint.first == pair ? constraint.second : constraint.first;
            if (connected.insert(neighbor).second) pending.push(neighbor);
        }
    }
    if (connected.size() <= 1) return scales;

    std::vector<Index> variables;
    variables.reserve(connected.size() - 1);
    for (const Index pair : connected)
        if (pair != anchor_pair) variables.push_back(pair);
    std::sort(variables.begin(), variables.end());
    std::unordered_map<Index, std::size_t> variable_index;
    variable_index.reserve(variables.size());
    for (std::size_t i = 0; i < variables.size(); ++i)
        variable_index.emplace(variables[i], i);

    Eigen::VectorXd solution = Eigen::VectorXd::Zero(variables.size());
    std::vector<double> robust_weights(constraints.size(), 1.0);
    for (unsigned iteration = 0; iteration < 6; ++iteration) {
        Eigen::MatrixXd normal =
            Eigen::MatrixXd::Zero(variables.size(), variables.size());
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(variables.size());
        for (std::size_t constraint_id = 0;
             constraint_id < constraints.size(); ++constraint_id) {
            const ScaleConstraint& constraint = constraints[constraint_id];
            if (!connected.contains(constraint.first) ||
                !connected.contains(constraint.second)) {
                continue;
            }
            const double weight = std::max(
                1e-6, constraint.weight * robust_weights[constraint_id]);
            const auto first = variable_index.find(constraint.first);
            const auto second = variable_index.find(constraint.second);
            if (first != variable_index.end()) {
                normal(first->second, first->second) += weight;
                rhs(first->second) += weight * constraint.log_ratio;
            }
            if (second != variable_index.end()) {
                normal(second->second, second->second) += weight;
                rhs(second->second) -= weight * constraint.log_ratio;
            }
            if (first != variable_index.end() && second != variable_index.end()) {
                normal(first->second, second->second) -= weight;
                normal(second->second, first->second) -= weight;
            }
        }
        normal.diagonal().array() += 1e-10;
        Eigen::LDLT<Eigen::MatrixXd> ldlt(normal);
        if (ldlt.info() != Eigen::Success) return scales;
        const Eigen::VectorXd updated = ldlt.solve(rhs);
        if (ldlt.info() != Eigen::Success || !updated.allFinite()) return scales;
        solution = updated;

        std::vector<double> absolute_residuals;
        absolute_residuals.reserve(constraints.size());
        std::vector<double> residuals(constraints.size(), 0.0);
        for (std::size_t constraint_id = 0;
             constraint_id < constraints.size(); ++constraint_id) {
            const ScaleConstraint& constraint = constraints[constraint_id];
            if (!connected.contains(constraint.first) ||
                !connected.contains(constraint.second)) {
                continue;
            }
            const auto value = [&](const Index pair) {
                if (pair == anchor_pair) return 0.0;
                const auto it = variable_index.find(pair);
                return it == variable_index.end() ? 0.0 : solution(it->second);
            };
            const double residual =
                value(constraint.first) - value(constraint.second) -
                constraint.log_ratio;
            residuals[constraint_id] = residual;
            absolute_residuals.push_back(std::abs(residual));
        }
        if (absolute_residuals.empty()) break;
        const std::size_t middle = absolute_residuals.size() / 2;
        std::nth_element(
            absolute_residuals.begin(),
            absolute_residuals.begin() + middle,
            absolute_residuals.end());
        const double sigma = std::max(
            1e-4, 1.4826 * absolute_residuals[middle]);
        const double huber = 2.5 * sigma;
        double max_change = 0.0;
        for (std::size_t constraint_id = 0;
             constraint_id < constraints.size(); ++constraint_id) {
            const double magnitude = std::abs(residuals[constraint_id]);
            const double updated_weight =
                magnitude <= huber || magnitude <= 1e-12
                    ? 1.0
                    : huber / magnitude;
            max_change = std::max(
                max_change,
                std::abs(updated_weight - robust_weights[constraint_id]));
            robust_weights[constraint_id] = updated_weight;
        }
        if (max_change < 1e-3) break;
    }

    for (std::size_t i = 0; i < variables.size(); ++i) {
        const double scale = std::exp(solution(i));
        if (std::isfinite(scale) && scale > 1e-6 && scale < 1e6)
            scales[variables[i]] = scale;
    }
    return scales;
}

}  // namespace

Index select_reference_view(const Scene& scene) {
    std::vector<unsigned> degree(scene.images.size(), 0);
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.usable_for_init()) continue;
        degree[pair.id1] += pair.num_inliers();
        degree[pair.id2] += pair.num_inliers();
    }
    Index best = k_invalid;
    unsigned max_degree = 0;
    for (Index i = 0; i < degree.size(); ++i) {
        if (degree[i] > max_degree) {
            max_degree = degree[i];
            best = i;
        }
    }
    return best;
}

bool star_initialize(Scene& scene, const StarInitConfig& config) {
    core::StageScope stage("sfm.star_initialize");
    for (auto& image : scene.images) {
        image.registered = false;
        image.pose = Pose3D::identity();
    }
    for (auto& track : scene.tracks) track.num_inliers = 0;

    const Index ref = select_reference_view(scene);
    if (ref == k_invalid) {
        core::Logger::instance().error("star_init: no reference view");
        return false;
    }

    struct Neighbor {
        Index image_id{};
        Index pair_index{};
        unsigned inliers{};
        float weight{};
    };
    std::vector<Neighbor> neighbors;
    for (Index pi = 0; pi < scene.pairs.size(); ++pi) {
        const ImagePair& pair = scene.pairs[pi];
        // Exclude H-dominant / low-parallax pairs from star seeds (still in view graph).
        if (!pair.usable_for_init()) continue;
        Index other = k_invalid;
        if (pair.id1 == ref) other = pair.id2;
        else if (pair.id2 == ref) other = pair.id1;
        else continue;
        neighbors.push_back(
            {other, pi, pair.num_inliers(), pair.composite_weight()});
    }
    std::sort(neighbors.begin(), neighbors.end(), [](const Neighbor& a, const Neighbor& b) {
        return a.weight > b.weight || (a.weight == b.weight && a.inliers > b.inliers);
    });
    neighbors.erase(
        std::remove_if(
            neighbors.begin(), neighbors.end(),
            [&](const Neighbor& neighbor) {
                return neighbor.inliers < config.min_tracks_per_view;
            }),
        neighbors.end());
    if (neighbors.size() > config.max_views)
        neighbors.resize(config.max_views);
    if (neighbors.size() + 1 < config.min_views) {
        core::Logger::instance().error(
            "star_init: insufficient neighbors (", neighbors.size(), ')');
        return false;
    }

    scene.images[ref].registered = true;
    scene.images[ref].pose = Pose3D::identity();

    // Seed absolute poses from relative poses (unit translation); then scale.
    std::vector<double> pair_scales(scene.pairs.size(), 1.0);
    std::unordered_set<Index> star_views{ref};
    for (const Neighbor& n : neighbors) star_views.insert(n.image_id);

    // Scale averaging: for each star pair, collect depths; constrain scales of
    // pairs that share a (view, feature) via median log-ratio (openMVS-inspired).
    struct FeatDepth {
        Index pair_index{};
        double depth{};
    };
    std::unordered_map<Index, std::unordered_map<Index, std::vector<FeatDepth>>> view_feat_depths;

    for (Index pi = 0; pi < scene.pairs.size(); ++pi) {
        const ImagePair& pair = scene.pairs[pi];
        if (!pair.relative_pose.has_value()) continue;
        if (!star_views.count(pair.id1) || !star_views.count(pair.id2)) continue;
        const Image& img1 = scene.images[pair.id1];
        const Image& img2 = scene.images[pair.id2];
        const PinholeCamera& cam1 = scene.camera_of(img1);
        const PinholeCamera& cam2 = scene.camera_of(img2);
        const Pose3D& rel = *pair.relative_pose;
        const double cos_reproj = std::cos(
            0.5 * (cam1.pixel_error_to_angular(config.max_reproj_error) +
                   cam2.pixel_error_to_angular(config.max_reproj_error)));
        const double max_cos_angle = std::cos(0.5 * 3.14159265358979323846 / 180.0);

        for (const FeatureMatch& match : pair.matches) {
            const auto& kp1 = img1.features.keypoints[match.query];
            const auto& kp2 = img2.features.keypoints[match.train];
            const Vec3 b1 = cam1.unproject_normalized({kp1.x, kp1.y});
            const Vec3 b2 = cam2.unproject_normalized({kp2.x, kp2.y});
            Vec3 X;
            if (!triangulate_midpoint(rel.R, rel.C, b1, b2, X)) continue;
            const double n1 = X.norm();
            if (n1 < 1e-12) continue;
            if (b1.dot(X / n1) < cos_reproj) continue;
            const Vec3 X2 = rel.transform_world_to_camera(X);
            const double n2 = X2.norm();
            if (n2 < 1e-12) continue;
            if (b2.dot(X2 / n2) < cos_reproj) continue;
            const double cos_a = X.normalized().dot((X - rel.C).normalized());
            if (cos_a > max_cos_angle) continue;
            view_feat_depths[pair.id1][match.query].push_back({pi, n1});
            view_feat_depths[pair.id2][match.train].push_back({pi, (X - rel.C).norm()});
        }
    }

    // Collect pairwise scale ratios
    std::unordered_map<Index, std::unordered_map<Index, std::vector<double>>> ratios;
    for (const auto& [view_id, feats] : view_feat_depths) {
        (void)view_id;
        for (const auto& [feat_id, depths] : feats) {
            (void)feat_id;
            if (depths.size() < 2) continue;
            for (std::size_t i = 0; i + 1 < depths.size(); ++i) {
                for (std::size_t j = i + 1; j < depths.size(); ++j) {
                    Index p1 = depths[i].pair_index;
                    Index p2 = depths[j].pair_index;
                    double d1 = depths[i].depth;
                    double d2 = depths[j].depth;
                    if (p1 > p2) {
                        std::swap(p1, p2);
                        std::swap(d1, d2);
                    }
                    if (d1 < 1e-12) continue;
                    ratios[p1][p2].push_back(d2 / d1);
                }
            }
        }
    }

    // Build a global log-scale system. A single propagation pass is order
    // dependent and leaves valid pairs at the implicit scale 1 when their
    // constraint is visited before the anchor-connected part of the graph.
    std::vector<ScaleConstraint> scale_constraints;
    Index anchor_pair = neighbors.empty() ? k_invalid : neighbors.front().pair_index;
    for (const auto& [p1, map] : ratios) {
        for (const auto& [p2, vals] : map) {
            if (vals.size() < 8) continue;
            std::vector<double> sorted = vals;
            std::sort(sorted.begin(), sorted.end());
            const double median = sorted[sorted.size() / 2];
            if (!std::isfinite(median) || median <= 1e-12) continue;
            scale_constraints.push_back({
                p1, p2, std::log(median),
                std::sqrt(static_cast<double>(vals.size()))});
        }
    }
    pair_scales = solve_pair_scales(
        scene.pairs.size(), anchor_pair, scale_constraints);
    core::Logger::instance().debug(
        "star scale averaging: constraints=", scale_constraints.size(),
        " anchor_pair=", anchor_pair);

    // relative_pose is the pose of id2 in id1's frame (id1 at identity).
    for (const Neighbor& n : neighbors) {
        ImagePair& pair = scene.pairs[n.pair_index];
        Pose3D rel = *pair.relative_pose;
        rel.C *= pair_scales[n.pair_index];
        pair.relative_pose = rel;
        if (pair.id1 == ref) {
            scene.images[pair.id2].pose = rel;
            scene.images[pair.id2].registered = true;
        } else {
            scene.images[pair.id1].pose = rel.inverse();
            scene.images[pair.id1].registered = true;
        }
    }

    unsigned n_tracks = triangulate_tracks(
        scene, false, config.max_reproj_error, config.min_angle_deg);
    if (n_tracks < config.min_initial_tracks) {
        core::Logger::instance().error(
            "star_init: too few tracks after triangulation (", n_tracks, ')');
        return false;
    }

    // Stage 1: poses/structure only — avoid absorbing init scale into focal.
    BundleOptions ba;
    ba.optimizer.maximum_iterations = 10;
    ba.optimizer.huber_delta = 2.0;
    ba.optimizer.optimize_focal = false;
    ba.optimizer.optimize_distortion = false;
    if (!run_bundle_adjustment(scene, ba).success) {
        core::Logger::instance().error("star_init: initial BA failed");
        return false;
    }

    triangulate_tracks(
        scene, true, std::max(config.max_reproj_error - 1.F, 1.F),
        std::min(config.min_angle_deg + 0.5F, 3.F));
    filter_tracks(
        scene, std::max(config.max_reproj_error - 1.F, 1.F),
        std::min(config.min_angle_deg + 0.5F, 3.F));

    n_tracks = 0;
    for (const Track& t : scene.tracks) {
        if (t.is_triangulated()) ++n_tracks;
    }
    if (n_tracks < config.min_initial_tracks) {
        core::Logger::instance().error(
            "star_init: too few tracks after filter (", n_tracks, ')');
        return false;
    }

    // Stage 2: open focal with prior/bounds; keep distortion fixed.
    ba.optimizer.maximum_iterations = 20;
    ba.optimizer.optimize_focal = true;
    ba.optimizer.optimize_aspect_ratio = true;
    ba.optimizer.optimize_distortion = false;
    run_bundle_adjustment(scene, ba);

    // Stage 3: short distortion polish once geometry is stable.
    ba.optimizer.maximum_iterations = 10;
    ba.optimizer.optimize_focal = true;
    ba.optimizer.optimize_aspect_ratio = true;
    ba.optimizer.optimize_distortion = true;
    run_bundle_adjustment(scene, ba);

    triangulate_tracks(
        scene, true, std::max(config.max_reproj_error - 2.F, 1.F),
        std::min(config.min_angle_deg + 1.F, 3.F));
    filter_tracks(
        scene, std::max(config.max_reproj_error - 2.F, 1.F),
        std::min(config.min_angle_deg + 1.F, 3.F));

    core::Logger::instance().info(
        "star_init: ref=", ref, " views=", scene.registered_count(),
        " tracks=", n_tracks);
    return scene.registered_count() >= config.min_views;
}

}  // namespace aetherscan::sfm
