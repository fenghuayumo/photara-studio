#include "sfm/global_positioning.hpp"

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <thread>
#include <vector>

namespace aetherscan::sfm {
namespace {

struct PositionObservation {
    Index image_id{};
    Index track_id{};
    Vec3 direction{Vec3::Zero()};
};

// Matches openMVS BATAPairwiseDirectionCostFunctor:
//   residual = direction - scale * (point - center)
struct PairwiseDirectionCost {
    explicit PairwiseDirectionCost(const Vec3& observed_direction)
        : observed_direction(observed_direction) {}

    template <typename T>
    bool operator()(
        const T* position1,
        const T* position2,
        const T* scale,
        T* residuals) const {
        using Vector3 = Eigen::Matrix<T, 3, 1>;
        Eigen::Map<Vector3> residual(residuals);
        residual = observed_direction.cast<T>() -
            scale[0] *
                (Eigen::Map<const Vector3>(position2) -
                 Eigen::Map<const Vector3>(position1));
        return true;
    }

    Vec3 observed_direction;
};

Vec3 observed_world_ray(const Scene& scene, const Observation& observation) {
    const Image& image = scene.images[observation.image_id];
    const PinholeCamera& camera = scene.camera_of(image);
    const auto& keypoint = image.features.keypoints[observation.feature_id];
    return (image.pose.R.transpose() *
            camera.unproject_normalized({keypoint.x, keypoint.y}))
        .normalized();
}

std::vector<PositionObservation> collect_largest_position_component(
    Scene& scene,
    const unsigned min_views_per_track,
    std::vector<char>& valid_tracks) {
    std::vector<PositionObservation> observations;
    std::vector<std::vector<Index>> camera_adjacency(scene.images.size());
    valid_tracks.assign(scene.tracks.size(), 0);

    for (Index track_id = 0; track_id < scene.tracks.size(); ++track_id) {
        const Track& track = scene.tracks[track_id];
        std::vector<Index> valid_images;
        for (const Observation& observation : track.observations) {
            if (observation.image_id >= scene.images.size()) continue;
            const Image& image = scene.images[observation.image_id];
            if (!image.registered ||
                observation.feature_id >= image.features.keypoints.size())
                continue;
            valid_images.push_back(observation.image_id);
        }
        if (valid_images.size() < min_views_per_track) continue;
        valid_tracks[track_id] = 1;
        for (std::size_t i = 1; i < valid_images.size(); ++i) {
            camera_adjacency[valid_images.front()].push_back(valid_images[i]);
            camera_adjacency[valid_images[i]].push_back(valid_images.front());
        }
        for (const Observation& observation : track.observations) {
            if (observation.image_id >= scene.images.size()) continue;
            const Image& image = scene.images[observation.image_id];
            if (!image.registered ||
                observation.feature_id >= image.features.keypoints.size())
                continue;
            observations.push_back({
                observation.image_id,
                track_id,
                observed_world_ray(scene, observation)});
        }
    }
    if (observations.empty()) return {};

    std::vector<int> component(scene.images.size(), -1);
    std::vector<unsigned> component_size;
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        if (!scene.images[image_id].registered ||
            camera_adjacency[image_id].empty() ||
            component[image_id] >= 0)
            continue;
        const int component_id = static_cast<int>(component_size.size());
        component_size.push_back(0);
        std::queue<Index> queue;
        queue.push(image_id);
        component[image_id] = component_id;
        while (!queue.empty()) {
            const Index current = queue.front();
            queue.pop();
            ++component_size[component_id];
            for (Index neighbor : camera_adjacency[current]) {
                if (component[neighbor] >= 0) continue;
                component[neighbor] = component_id;
                queue.push(neighbor);
            }
        }
    }
    if (component_size.empty()) return {};
    const int largest_component = static_cast<int>(std::distance(
        component_size.begin(),
        std::max_element(component_size.begin(), component_size.end())));

    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        if (component[image_id] != largest_component)
            scene.images[image_id].registered = false;
    }

    // After dropping cameras outside the largest component, require each track
    // still to have enough views. Thin leftovers leave free ray parameters and
    // make the Hessian indefinite for CHOLMOD.
    std::vector<unsigned> track_views(scene.tracks.size(), 0);
    observations.erase(
        std::remove_if(
            observations.begin(),
            observations.end(),
            [&](const PositionObservation& observation) {
                return component[observation.image_id] != largest_component;
            }),
        observations.end());
    for (const PositionObservation& observation : observations)
        ++track_views[observation.track_id];
    observations.erase(
        std::remove_if(
            observations.begin(),
            observations.end(),
            [&](const PositionObservation& observation) {
                return track_views[observation.track_id] < min_views_per_track;
            }),
        observations.end());
    std::fill(valid_tracks.begin(), valid_tracks.end(), 0);
    for (const PositionObservation& observation : observations)
        valid_tracks[observation.track_id] = 1;

    std::vector<char> camera_used(scene.images.size(), 0);
    for (const PositionObservation& observation : observations)
        camera_used[observation.image_id] = 1;
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        if (!camera_used[image_id]) scene.images[image_id].registered = false;
    }
    return observations;
}

unsigned add_camera_to_camera_constraints(
    ceres::Problem& problem,
    ceres::LossFunction* loss,
    Scene& scene,
    std::vector<double>& scales) {
    unsigned added = 0;
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || !pair.relative_pose.has_value()) continue;
        if (pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size())
            continue;
        Image& image1 = scene.images[pair.id1];
        Image& image2 = scene.images[pair.id2];
        if (!image1.registered || !image2.registered) continue;

        // openMVS: world-frame relative translation direction
        //   t_world = -(R2^T * t_rel), t_rel = -R_rel * C_rel
        const Vec3 t_rel = pair.relative_pose->translation();
        const Vec3 direction = -(image2.pose.R.transpose() * t_rel);
        const double norm = direction.norm();
        if (norm < 1e-12) continue;

        scales.push_back(1.0);
        double& scale = scales.back();
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<PairwiseDirectionCost, 3, 3, 3, 1>(
                new PairwiseDirectionCost(direction / norm));
        problem.AddResidualBlock(
            cost, loss, image1.pose.C.data(), image2.pose.C.data(), &scale);
        problem.SetParameterLowerBound(&scale, 0, 1e-5);
        ++added;
    }
    return added;
}

bool solve_with_options(
    ceres::Problem& problem,
    ceres::Solver::Options solver_options,
    ceres::Solver::Summary& summary) {
    ceres::Solve(solver_options, &problem, &summary);
    return summary.IsSolutionUsable();
}

}  // namespace

GlobalPositioningSummary solve_global_positions(
    Scene& scene,
    const GlobalPositioningOptions& options) {
    GlobalPositioningSummary summary;
    std::vector<char> valid_tracks;
    const std::vector<PositionObservation> observations =
        collect_largest_position_component(
            scene, options.min_views_per_track, valid_tracks);
    summary.observations = static_cast<unsigned>(observations.size());
    if (observations.empty()) return summary;

    std::mt19937 random(123);
    std::uniform_real_distribution<double> distribution(-100.0, 100.0);
    unsigned registered_images = 0;
    for (Image& image : scene.images) {
        if (!image.registered) continue;
        image.pose.C = Vec3(
            distribution(random), distribution(random), distribution(random));
        ++registered_images;
    }
    for (Index track_id = 0; track_id < scene.tracks.size(); ++track_id) {
        if (!valid_tracks[track_id]) continue;
        scene.tracks[track_id].position = Vec3(
            distribution(random), distribution(random), distribution(random));
    }
    if (registered_images < 2) return summary;

    // Pre-size scales for point-camera + camera-camera residuals.
    std::size_t pair_capacity = 0;
    for (const ImagePair& pair : scene.pairs) {
        if (pair.active && pair.relative_pose.has_value()) ++pair_capacity;
    }
    std::vector<double> scales;
    scales.reserve(observations.size() + pair_capacity);

    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    std::unique_ptr<ceres::LossFunction> loss(
        new ceres::HuberLoss(options.huber_threshold));

    // Camera-camera constraints first (openMVS optional path). On video
    // sequences they stabilize the otherwise weakly constrained BATA system.
    const unsigned camera_pairs =
        add_camera_to_camera_constraints(problem, loss.get(), scene, scales);
    const std::size_t point_scale_offset = scales.size();
    scales.resize(point_scale_offset + observations.size(), 1.0);

    for (std::size_t i = 0; i < observations.size(); ++i) {
        const PositionObservation& observation = observations[i];
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<PairwiseDirectionCost, 3, 3, 3, 1>(
                new PairwiseDirectionCost(observation.direction));
        problem.AddResidualBlock(
            cost,
            loss.get(),
            scene.images[observation.image_id].pose.C.data(),
            scene.tracks[observation.track_id].position.data(),
            &scales[point_scale_offset + i]);
        problem.SetParameterLowerBound(&scales[point_scale_offset + i], 0, 1e-5);
    }

    // openMVS ConfigureProblem ordering: scales, points, cameras.
    auto* ordering = new ceres::ParameterBlockOrdering;
    for (double& scale : scales) {
        if (problem.HasParameterBlock(&scale))
            ordering->AddElementToGroup(&scale, 0);
    }
    for (Index track_id = 0; track_id < scene.tracks.size(); ++track_id) {
        if (!valid_tracks[track_id]) continue;
        if (problem.HasParameterBlock(scene.tracks[track_id].position.data()))
            ordering->AddElementToGroup(
                scene.tracks[track_id].position.data(), 1);
    }
    for (Image& image : scene.images) {
        if (!image.registered) continue;
        if (problem.HasParameterBlock(image.pose.C.data()))
            ordering->AddElementToGroup(image.pose.C.data(), 2);
    }

    // Fix scene scale + one camera center (translation gauge).
    for (double& scale : scales) {
        if (problem.HasParameterBlock(&scale)) {
            problem.SetParameterBlockConstant(&scale);
            break;
        }
    }
    for (Image& image : scene.images) {
        if (!image.registered) continue;
        if (problem.HasParameterBlock(image.pose.C.data())) {
            problem.SetParameterBlockConstant(image.pose.C.data());
            break;
        }
    }

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations =
        static_cast<int>(options.max_num_iterations);
    solver_options.num_threads = static_cast<int>(
        scene.thread_count == 0 ? std::thread::hardware_concurrency()
                                : scene.thread_count);
    solver_options.function_tolerance = options.function_tolerance;
    solver_options.visibility_clustering_type = ceres::CANONICAL_VIEWS;
    solver_options.linear_solver_ordering.reset(ordering);

    ceres::Solver::Summary ceres_summary;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
    if (!solve_with_options(problem, solver_options, ceres_summary)) {
        // Direct Schur/CHOLMOD often fails on large weakly-constrained video
        // graphs; iterative Schur tolerates a mild nullspace better.
        std::cerr << "global positioning: sparse schur failed ("
                  << ceres_summary.message << "), retrying iterative schur\n";
        solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
        solver_options.use_explicit_schur_complement = true;
        if (!solve_with_options(problem, solver_options, ceres_summary)) {
            std::cerr << "global positioning: " << ceres_summary.BriefReport()
                      << " camera_pairs=" << camera_pairs << '\n';
            return summary;
        }
    }

    double residual_sum = 0.0;
    for (std::size_t i = 0; i < observations.size(); ++i) {
        const PositionObservation& observation = observations[i];
        const double scale = scales[point_scale_offset + i];
        const Vec3 residual = observation.direction -
            scale *
                (scene.tracks[observation.track_id].position -
                 scene.images[observation.image_id].pose.C);
        residual_sum += residual.norm();
    }

    for (const Image& image : scene.images) {
        if (image.registered) ++summary.positioned_images;
    }
    for (char valid : valid_tracks) {
        if (valid) ++summary.positioned_tracks;
    }
    summary.success =
        summary.positioned_images >= 2 && summary.positioned_tracks > 0;
    summary.iterations = static_cast<unsigned>(ceres_summary.iterations.size());
    summary.final_residual =
        residual_sum / static_cast<double>(observations.size());
    std::cout << "global positioning: images=" << summary.positioned_images
              << " tracks=" << summary.positioned_tracks
              << " camera_pairs=" << camera_pairs
              << " residual=" << summary.final_residual
              << " iters=" << summary.iterations << '\n';
    return summary;
}

}  // namespace aetherscan::sfm
