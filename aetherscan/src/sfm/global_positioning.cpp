#include "aetherscan/sfm/global_positioning.hpp"

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
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

struct PairwiseDirectionCost {
    explicit PairwiseDirectionCost(const Vec3& observed_direction)
        : observed_direction(observed_direction) {}

    template <typename T>
    bool operator()(
        const T* center,
        const T* point,
        const T* scale,
        T* residuals) const {
        using Vector3 = Eigen::Matrix<T, 3, 1>;
        Eigen::Map<Vector3> residual(residuals);
        residual = observed_direction.cast<T>() -
            scale[0] *
                (Eigen::Map<const Vector3>(point) -
                 Eigen::Map<const Vector3>(center));
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
    observations.erase(
        std::remove_if(
            observations.begin(),
            observations.end(),
            [&](const PositionObservation& observation) {
                return component[observation.image_id] != largest_component;
            }),
        observations.end());
    std::fill(valid_tracks.begin(), valid_tracks.end(), 0);
    for (const PositionObservation& observation : observations)
        valid_tracks[observation.track_id] = 1;
    return observations;
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
    Index fixed_image = k_invalid;
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        Image& image = scene.images[image_id];
        if (!image.registered) continue;
        image.pose.C = Vec3(
            distribution(random), distribution(random), distribution(random));
        if (fixed_image == k_invalid) fixed_image = image_id;
    }
    for (Index track_id = 0; track_id < scene.tracks.size(); ++track_id) {
        if (!valid_tracks[track_id]) continue;
        scene.tracks[track_id].position = Vec3(
            distribution(random), distribution(random), distribution(random));
    }
    if (fixed_image == k_invalid) return summary;

    std::vector<double> scales(observations.size(), 1.0);
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    std::unique_ptr<ceres::LossFunction> loss(
        new ceres::HuberLoss(options.huber_threshold));
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
            &scales[i]);
        problem.SetParameterLowerBound(&scales[i], 0, 1e-5);
    }

    // Remove the translation and scale gauges explicitly. OpenMVS fixes the
    // first scale; fixing one center additionally avoids a rank-deficient solve.
    problem.SetParameterBlockConstant(scene.images[fixed_image].pose.C.data());
    problem.SetParameterBlockConstant(&scales.front());

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations =
        static_cast<int>(options.max_irls_iterations * 25);
    solver_options.num_threads = static_cast<int>(
        scene.thread_count == 0 ? std::thread::hardware_concurrency()
                                : scene.thread_count);
    solver_options.function_tolerance = options.linear_tolerance;
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
    solver_options.visibility_clustering_type = ceres::CANONICAL_VIEWS;

    ceres::Solver::Summary ceres_summary;
    ceres::Solve(solver_options, &problem, &ceres_summary);
    if (!ceres_summary.IsSolutionUsable()) return summary;

    double residual_sum = 0.0;
    for (std::size_t i = 0; i < observations.size(); ++i) {
        const PositionObservation& observation = observations[i];
        const Vec3 residual = observation.direction -
            scales[i] *
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
    return summary;
}

}  // namespace aetherscan::sfm
