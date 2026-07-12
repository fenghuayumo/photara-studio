#include "sfm/global_positioning.hpp"

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <thread>
#include <vector>

namespace aetherscan::sfm {
namespace {

struct PairwiseDirectionCost {
    explicit PairwiseDirectionCost(const Vec3& direction)
        : direction(direction) {}

    template <typename T>
    bool operator()(
        const T* position1,
        const T* position2,
        const T* scale,
        T* residuals) const {
        using Vector3 = Eigen::Matrix<T, 3, 1>;
        Eigen::Map<Vector3> residual_map(residuals);
        residual_map =
            direction.cast<T>() -
            scale[0] *
                (Eigen::Map<const Vector3>(position2) -
                 Eigen::Map<const Vector3>(position1));
        return true;
    }

    Vec3 direction;
};

Vec3 random_point(
    std::mt19937& generator,
    std::uniform_real_distribution<double>& distribution) {
    return {
        distribution(generator),
        distribution(generator),
        distribution(generator)};
}

bool uses_points(GlobalPositioningConstraint constraint) {
    return constraint != GlobalPositioningConstraint::only_cameras;
}

bool uses_cameras(GlobalPositioningConstraint constraint) {
    return constraint != GlobalPositioningConstraint::only_points;
}

}  // namespace

GlobalPositioningSummary solve_global_positions(
    Scene& scene,
    const GlobalPositioningOptions& options) {
    GlobalPositioningSummary result;
    if (scene.images.empty()) return result;
    if (scene.pairs.empty() && uses_cameras(options.constraint)) return result;
    if (scene.tracks.empty() && uses_points(options.constraint)) return result;

    std::size_t point_to_camera_capacity = 0;
    for (const Track& track : scene.tracks) {
        if (track.observations.size() < options.min_views_per_track) continue;
        for (const Observation& observation : track.observations) {
            if (observation.image_id < scene.images.size() &&
                scene.images[observation.image_id].registered)
                ++point_to_camera_capacity;
        }
    }
    if (uses_points(options.constraint) && point_to_camera_capacity == 0)
        return result;

    std::vector<Vec3> original_centers(scene.images.size());
    for (std::size_t i = 0; i < scene.images.size(); ++i)
        original_centers[i] = scene.images[i].pose.C;
    std::vector<Vec3> original_points(scene.tracks.size());
    for (std::size_t i = 0; i < scene.tracks.size(); ++i)
        original_points[i] = scene.tracks[i].position;
    const auto restore_input = [&] {
        for (std::size_t i = 0; i < scene.images.size(); ++i)
            scene.images[i].pose.C = original_centers[i];
        for (std::size_t i = 0; i < scene.tracks.size(); ++i)
            scene.tracks[i].position = original_points[i];
    };

    std::mt19937 generator(options.random_seed);
    std::uniform_real_distribution<double> distribution(-100.0, 100.0);
    unsigned valid_images = 0;
    for (Image& image : scene.images) {
        if (!image.registered) continue;
        if (options.optimize_positions && options.generate_random_positions)
            image.pose.C = random_point(generator, distribution);
        ++valid_images;
    }

    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    auto huber_loss = std::make_shared<ceres::HuberLoss>(
        options.huber_threshold);

    std::vector<double> scales;
    scales.reserve(scene.pairs.size() + point_to_camera_capacity);

    unsigned valid_pairs = 0;
    if (uses_cameras(options.constraint)) {
        for (const ImagePair& pair : scene.pairs) {
            if (!pair.active || !pair.relative_pose.has_value() ||
                pair.composite_weight() <= 0.F)
                continue;
            if (pair.id1 >= scene.images.size() ||
                pair.id2 >= scene.images.size())
                continue;
            Image& image1 = scene.images[pair.id1];
            Image& image2 = scene.images[pair.id2];
            if (!image1.registered || !image2.registered) continue;

            scales.push_back(1.0);
            double& scale = scales.back();
            // openMVS converts t_rel to world coordinates as -R2^T t_rel.
            const Vec3 direction = -(
                image2.pose.R.transpose() *
                pair.relative_pose->translation());
            auto* cost =
                new ceres::AutoDiffCostFunction<
                    PairwiseDirectionCost, 3, 3, 3, 1>(
                    new PairwiseDirectionCost(direction));
            problem.AddResidualBlock(
                cost,
                huber_loss.get(),
                image1.pose.C.data(),
                image2.pose.C.data(),
                &scale);
            problem.SetParameterLowerBound(&scale, 0, 1e-5);
            ++valid_pairs;
        }
    }

    const std::size_t camera_residual_count = problem.NumResidualBlocks();
    double point_weight = 1.0;
    if (camera_residual_count > 0 &&
        options.constraint ==
            GlobalPositioningConstraint::points_and_cameras_balanced) {
        point_weight =
            options.constraint_reweight_scale *
            static_cast<double>(camera_residual_count) /
            static_cast<double>(point_to_camera_capacity);
    }
    auto untrusted_loss = std::make_shared<ceres::ScaledLoss>(
        huber_loss.get(),
        0.5 * point_weight,
        ceres::DO_NOT_TAKE_OWNERSHIP);
    std::shared_ptr<ceres::LossFunction> trusted_loss;
    if (options.constraint ==
        GlobalPositioningConstraint::points_and_cameras_balanced) {
        trusted_loss = std::make_shared<ceres::ScaledLoss>(
            huber_loss.get(),
            point_weight,
            ceres::DO_NOT_TAKE_OWNERSHIP);
    } else {
        trusted_loss = huber_loss;
    }

    unsigned valid_tracks = 0;
    if (uses_points(options.constraint)) {
        for (Track& track : scene.tracks) {
            if (track.observations.size() < options.min_views_per_track)
                continue;
            if (options.optimize_points && options.generate_random_points)
                track.position = random_point(generator, distribution);

            bool has_valid_observation = false;
            for (const Observation& observation : track.observations) {
                if (observation.image_id >= scene.images.size()) continue;
                Image& image = scene.images[observation.image_id];
                if (!image.registered ||
                    observation.feature_id >= image.features.keypoints.size() ||
                    image.camera_id >= scene.cameras.size())
                    continue;

                const auto& keypoint =
                    image.features.keypoints[observation.feature_id];
                const Vec3 direction =
                    (image.pose.R.transpose() *
                     scene.camera_of(image).unproject_normalized(
                         {keypoint.x, keypoint.y}))
                        .normalized();

                scales.push_back(1.0);
                double& scale = scales.back();
                if (!options.generate_scales && track.is_triangulated()) {
                    const Vec3 displacement = track.position - image.pose.C;
                    scale = std::max(
                        1e-5,
                        direction.dot(displacement) /
                            displacement.squaredNorm());
                }

                ceres::LossFunction* loss =
                    scene.camera_of(image).trust_intrinsics
                    ? trusted_loss.get()
                    : untrusted_loss.get();
                auto* cost =
                    new ceres::AutoDiffCostFunction<
                        PairwiseDirectionCost, 3, 3, 3, 1>(
                        new PairwiseDirectionCost(direction));
                problem.AddResidualBlock(
                    cost,
                    loss,
                    image.pose.C.data(),
                    track.position.data(),
                    &scale);
                problem.SetParameterLowerBound(&scale, 0, 1e-5);
                has_valid_observation = true;
                ++result.observations;
            }
            if (has_valid_observation) ++valid_tracks;
        }
    }

    auto* ordering = new ceres::ParameterBlockOrdering;
    for (double& scale : scales)
        if (problem.HasParameterBlock(&scale))
            ordering->AddElementToGroup(&scale, 0);
    int camera_group = 1;
    if (!scene.tracks.empty()) {
        for (Track& track : scene.tracks)
            if (problem.HasParameterBlock(track.position.data()))
                ordering->AddElementToGroup(track.position.data(), 1);
        ++camera_group;
    }
    for (Image& image : scene.images) {
        if (!image.registered) continue;
        if (problem.HasParameterBlock(image.pose.C.data()))
            ordering->AddElementToGroup(image.pose.C.data(), camera_group);
    }

    if (!options.optimize_positions) {
        for (Image& image : scene.images) {
            if (image.registered &&
                problem.HasParameterBlock(image.pose.C.data()))
                problem.SetParameterBlockConstant(image.pose.C.data());
        }
    } else {
        // All residuals depend only on position differences, so the system has
        // a three-dimensional translation gauge. Anchor one registered camera
        // center to keep sparse Cholesky positive definite.
        for (Image& image : scene.images) {
            if (!image.registered ||
                !problem.HasParameterBlock(image.pose.C.data()))
                continue;
            problem.SetParameterBlockConstant(image.pose.C.data());
            break;
        }
    }
    if (!options.optimize_points) {
        for (Track& track : scene.tracks)
            if (problem.HasParameterBlock(track.position.data()))
                problem.SetParameterBlockConstant(track.position.data());
    }
    if (!options.optimize_scales) {
        for (double& scale : scales)
            if (problem.HasParameterBlock(&scale))
                problem.SetParameterBlockConstant(&scale);
    } else {
        // This is openMVS's only gauge constraint.
        for (double& scale : scales) {
            if (!problem.HasParameterBlock(&scale)) continue;
            problem.SetParameterBlockConstant(&scale);
            break;
        }
    }

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations =
        static_cast<int>(options.max_num_iterations);
    solver_options.num_threads = static_cast<int>(
        scene.thread_count == 0
            ? std::thread::hardware_concurrency()
            : scene.thread_count);
    solver_options.function_tolerance = options.function_tolerance;
    solver_options.linear_solver_ordering.reset(ordering);
    solver_options.visibility_clustering_type = ceres::CANONICAL_VIEWS;
    if (!scene.tracks.empty()) {
        solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
        solver_options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
    } else {
        solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        solver_options.preconditioner_type = ceres::JACOBI;
    }

    ceres::Solver::Summary solver_summary;
    ceres::Solve(solver_options, &problem, &solver_summary);
    result.iterations =
        static_cast<unsigned>(solver_summary.iterations.size());
    if (!solver_summary.IsSolutionUsable()) {
        restore_input();
        return result;
    }

    double residual_sum = 0.0;
    unsigned residual_count = 0;
    std::vector<ceres::ResidualBlockId> residual_blocks;
    problem.GetResidualBlocks(&residual_blocks);
    for (const auto residual_id : residual_blocks) {
        double cost = 0.0;
        std::vector<double> residuals;
        ceres::Problem::EvaluateOptions evaluate_options;
        evaluate_options.residual_blocks = {residual_id};
        evaluate_options.apply_loss_function = false;
        if (problem.Evaluate(
                evaluate_options, &cost, &residuals, nullptr, nullptr) &&
            residuals.size() == 3) {
            residual_sum += std::sqrt(
                residuals[0] * residuals[0] +
                residuals[1] * residuals[1] +
                residuals[2] * residuals[2]);
            ++residual_count;
        }
    }

    result.success = true;
    result.positioned_images = valid_images;
    result.positioned_tracks = valid_tracks;
    result.final_residual =
        residual_count == 0
        ? 0.0
        : residual_sum / static_cast<double>(residual_count);
    static_cast<void>(valid_pairs);
    return result;
}

}  // namespace aetherscan::sfm
