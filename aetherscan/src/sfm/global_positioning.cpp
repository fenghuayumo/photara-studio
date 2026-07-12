#include "sfm/global_positioning.hpp"
#include "core/logging.hpp"

#include <ceres/ceres.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

namespace aetherscan::sfm {
namespace {

using Matrix3dRowMajor = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

struct PairwiseDirectionCostAnalytic : ceres::SizedCostFunction<3, 3, 3, 1> {
    explicit PairwiseDirectionCostAnalytic(const Vec3& direction)
        : direction(direction) {}

    bool Evaluate(
        double const* const* parameters,
        double* residuals,
        double** jacobians) const override {
        Eigen::Map<const Vec3> position1(parameters[0]);
        Eigen::Map<const Vec3> position2(parameters[1]);
        const double scale = parameters[2][0];

        const Vec3 diff = position2 - position1;
        Eigen::Map<Vec3> residual_map(residuals);
        residual_map = direction - scale * diff;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Matrix3dRowMajor> jacobian1(jacobians[0]);
                jacobian1.setIdentity();
                jacobian1 *= scale;
            }
            if (jacobians[1]) {
                Eigen::Map<Matrix3dRowMajor> jacobian2(jacobians[1]);
                jacobian2.setIdentity();
                jacobian2 *= -scale;
            }
            if (jacobians[2]) {
                Eigen::Map<Vec3> jacobian_scale(jacobians[2]);
                jacobian_scale = -diff;
            }
        }
        return true;
    }

    Vec3 direction;
};

// Scale-free bearing residual. Eliminating one scale variable per observation
// keeps large only-points refinements small and avoids the ill-conditioned
// multi-level Schur system used by the BATA initialization.
struct BearingDirectionCostAnalytic : ceres::SizedCostFunction<3, 3, 3> {
    explicit BearingDirectionCostAnalytic(const Vec3& direction)
        : direction(direction) {}

    bool Evaluate(
        double const* const* parameters,
        double* residuals,
        double** jacobians) const override {
        Eigen::Map<const Vec3> camera(parameters[0]);
        Eigen::Map<const Vec3> point(parameters[1]);
        const Vec3 displacement = point - camera;
        const double distance = displacement.norm();
        if (!(distance > 1e-10) || !std::isfinite(distance)) return false;

        const Vec3 unit = displacement / distance;
        Eigen::Map<Vec3> residual_map(residuals);
        residual_map = direction - unit;
        if (!jacobians) return true;

        const Mat3 projection =
            (Mat3::Identity() - unit * unit.transpose()) / distance;
        if (jacobians[0]) {
            Eigen::Map<Matrix3dRowMajor> camera_jacobian(jacobians[0]);
            camera_jacobian = projection;
        }
        if (jacobians[1]) {
            Eigen::Map<Matrix3dRowMajor> point_jacobian(jacobians[1]);
            point_jacobian = -projection;
        }
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

enum class LinearSolverStrategy {
    sparse_schur_cluster,
    iterative_schur,
    sparse_normal_cholesky,
};

struct PositioningAttempt {
    GlobalPositioningConstraint constraint;
    LinearSolverStrategy solver;
    bool warm_start_positions;
    bool fix_cameras;
    bool ray_initialize_points;
    const char* label;
};

struct BuiltPositioningProblem {
    ceres::Problem problem;
    std::shared_ptr<ceres::HuberLoss> huber_loss;
    std::shared_ptr<ceres::LossFunction> untrusted_loss;
    std::shared_ptr<ceres::LossFunction> trusted_loss;
    std::vector<double> scales;
    unsigned valid_images{0};
    unsigned valid_pairs{0};
    unsigned valid_tracks{0};
    unsigned observations{0};
    unsigned candidate_tracks{0};
    bool has_point_blocks{false};
};

struct WorldRay {
    Vec3 origin{Vec3::Zero()};
    Vec3 direction{Vec3::UnitZ()};
    bool valid{false};
};

unsigned count_registered_observations(
    const Scene& scene, const Track& track) {
    unsigned count = 0;
    for (const Observation& observation : track.observations) {
        if (observation.image_id < scene.images.size() &&
            scene.images[observation.image_id].registered)
            ++count;
    }
    return count;
}

WorldRay make_world_ray(
    const Scene& scene, const Image& image, const Observation& observation) {
    WorldRay ray;
    if (!image.registered ||
        observation.feature_id >= image.features.keypoints.size() ||
        image.camera_id >= scene.cameras.size())
        return ray;
    const auto& keypoint = image.features.keypoints[observation.feature_id];
    const Vec3 direction =
        (image.pose.R.transpose() *
         scene.camera_of(image).unproject_normalized(
             {keypoint.x, keypoint.y}))
            .normalized();
    if (!direction.allFinite() || direction.squaredNorm() < 1e-18) return ray;
    ray.origin = image.pose.C;
    ray.direction = direction;
    ray.valid = true;
    return ray;
}

// Closest-point midpoint of two skew lines. Returns false if nearly parallel.
bool ray_midpoint(
    const WorldRay& first, const WorldRay& second, Vec3& midpoint) {
    const Vec3 w0 = first.origin - second.origin;
    const double a = first.direction.dot(first.direction);
    const double b = first.direction.dot(second.direction);
    const double c = second.direction.dot(second.direction);
    const double d = first.direction.dot(w0);
    const double e = second.direction.dot(w0);
    const double denom = a * c - b * b;
    if (std::abs(denom) < 1e-10) return false;
    const double t = (b * e - c * d) / denom;
    const double s = (a * e - b * d) / denom;
    if (!(t > 1e-4) || !(s > 1e-4)) return false;
    const Vec3 p = first.origin + t * first.direction;
    const Vec3 q = second.origin + s * second.direction;
    midpoint = 0.5 * (p + q);
    return midpoint.allFinite();
}

double median_camera_baseline(const Scene& scene) {
    std::vector<double> baselines;
    baselines.reserve(scene.pairs.size());
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || pair.id1 >= scene.images.size() ||
            pair.id2 >= scene.images.size())
            continue;
        const Image& image1 = scene.images[pair.id1];
        const Image& image2 = scene.images[pair.id2];
        if (!image1.registered || !image2.registered) continue;
        const double distance = (image1.pose.C - image2.pose.C).norm();
        if (distance > 1e-6 && std::isfinite(distance))
            baselines.push_back(distance);
    }
    if (baselines.empty()) return 1.0;
    const auto mid = baselines.begin() + baselines.size() / 2;
    std::nth_element(baselines.begin(), mid, baselines.end());
    return std::max(1e-3, *mid);
}

// Multi-view ray midpoints when cameras are already positioned; falls back to
// first-ray * median baseline when triangulation is degenerate.
unsigned initialize_points_from_rays(
    Scene& scene, const std::vector<Index>& selected_tracks) {
    const double fallback_depth = 2.0 * median_camera_baseline(scene);
    unsigned initialized = 0;
    for (Index track_id : selected_tracks) {
        Track& track = scene.tracks[track_id];
        std::vector<WorldRay> rays;
        rays.reserve(track.observations.size());
        for (const Observation& observation : track.observations) {
            if (observation.image_id >= scene.images.size()) continue;
            WorldRay ray = make_world_ray(
                scene, scene.images[observation.image_id], observation);
            if (ray.valid) rays.push_back(ray);
        }
        if (rays.empty()) continue;

        Vec3 accumulator = Vec3::Zero();
        unsigned samples = 0;
        const std::size_t limit = std::min<std::size_t>(rays.size(), 8);
        for (std::size_t i = 0; i + 1 < limit; ++i) {
            for (std::size_t j = i + 1; j < limit; ++j) {
                Vec3 midpoint;
                if (!ray_midpoint(rays[i], rays[j], midpoint)) continue;
                accumulator += midpoint;
                ++samples;
            }
        }
        if (samples > 0) {
            track.position = accumulator / static_cast<double>(samples);
        } else {
            track.position =
                rays.front().origin + fallback_depth * rays.front().direction;
        }
        if (track.position.allFinite()) ++initialized;
    }
    return initialized;
}

std::vector<Index> select_tracks_for_positioning(
    const Scene& scene, const GlobalPositioningOptions& options) {
    struct RankedTrack {
        Index track_id{};
        unsigned registered_views{};
        double max_parallax{};
    };
    std::vector<RankedTrack> ranked;
    ranked.reserve(scene.tracks.size());
    for (Index track_id = 0; track_id < scene.tracks.size(); ++track_id) {
        const Track& track = scene.tracks[track_id];
        if (track.observations.size() < options.min_views_per_track) continue;
        const unsigned registered =
            count_registered_observations(scene, track);
        if (registered < options.min_views_per_track) continue;

        std::vector<Vec3> directions;
        directions.reserve(std::min<std::size_t>(
            track.observations.size(), 8));
        for (const Observation& observation : track.observations) {
            if (directions.size() == 8) break;
            if (observation.image_id >= scene.images.size()) continue;
            const WorldRay ray = make_world_ray(
                scene, scene.images[observation.image_id], observation);
            if (ray.valid) directions.push_back(ray.direction);
        }
        double max_parallax = 0.0;
        for (std::size_t i = 0; i + 1 < directions.size(); ++i) {
            for (std::size_t j = i + 1; j < directions.size(); ++j) {
                max_parallax = std::max(
                    max_parallax,
                    std::acos(std::clamp(
                        directions[i].dot(directions[j]), -1.0, 1.0)));
            }
        }
        ranked.push_back({track_id, registered, max_parallax});
    }
    std::stable_sort(
        ranked.begin(), ranked.end(),
        [](const RankedTrack& left, const RankedTrack& right) {
            // Parallax carries more positional information than track length.
            if (left.max_parallax != right.max_parallax)
                return left.max_parallax > right.max_parallax;
            if (left.registered_views != right.registered_views)
                return left.registered_views > right.registered_views;
            return left.track_id < right.track_id;
        });

    const std::size_t registered_images = scene.registered_count();
    const std::size_t adaptive_target = std::max<std::size_t>(
        options.min_tracks_for_positioning,
        registered_images * options.tracks_per_registered_image);
    const std::size_t hard_limit =
        options.max_tracks_for_positioning == 0
            ? ranked.size()
            : options.max_tracks_for_positioning;
    const std::size_t target =
        std::min({ranked.size(), adaptive_target, hard_limit});

    const unsigned grid = std::max(1U, options.coverage_grid_size);
    std::vector<std::uint8_t> covered_cells(
        scene.images.size() * static_cast<std::size_t>(grid) * grid, 0);
    std::vector<unsigned> image_counts(scene.images.size(), 0);
    std::vector<std::uint8_t> chosen(scene.tracks.size(), 0);
    std::vector<Index> selected;
    selected.reserve(target);

    const auto for_each_coverage = [&](
        const Index track_id, const auto& function) {
        const Track& track = scene.tracks[track_id];
        for (const Observation& observation : track.observations) {
            if (observation.image_id >= scene.images.size()) continue;
            const Image& image = scene.images[observation.image_id];
            if (!image.registered ||
                observation.feature_id >= image.features.keypoints.size())
                continue;
            const auto& keypoint =
                image.features.keypoints[observation.feature_id];
            const double width = std::max(1U, image.features.image_width);
            const double height = std::max(1U, image.features.image_height);
            const unsigned x = std::min(
                grid - 1,
                static_cast<unsigned>(
                    std::max(0.0, static_cast<double>(keypoint.x)) /
                    width * grid));
            const unsigned y = std::min(
                grid - 1,
                static_cast<unsigned>(
                    std::max(0.0, static_cast<double>(keypoint.y)) /
                    height * grid));
            const std::size_t cell =
                (static_cast<std::size_t>(observation.image_id) * grid + y) *
                    grid +
                x;
            function(observation.image_id, cell);
        }
    };
    const auto select = [&](const Index track_id) {
        if (chosen[track_id] || selected.size() >= target) return;
        chosen[track_id] = 1;
        selected.push_back(track_id);
        for_each_coverage(
            track_id,
            [&](const Index image_id, const std::size_t cell) {
                covered_cells[cell] = 1;
                ++image_counts[image_id];
            });
    };

    // Pass 1: cover as many image-grid cells as possible.
    for (const RankedTrack& entry : ranked) {
        if (selected.size() >= target) break;
        bool adds_cell = false;
        for_each_coverage(
            entry.track_id,
            [&](const Index, const std::size_t cell) {
                adds_cell = adds_cell || covered_cells[cell] == 0;
            });
        if (adds_cell) select(entry.track_id);
    }

    // Pass 2: guarantee a useful per-camera quota.
    const unsigned per_image_quota =
        std::max(options.min_views_per_track,
                 options.tracks_per_registered_image);
    for (const RankedTrack& entry : ranked) {
        if (selected.size() >= target) break;
        bool helps_undercovered_image = false;
        for_each_coverage(
            entry.track_id,
            [&](const Index image_id, const std::size_t) {
                helps_undercovered_image =
                    helps_undercovered_image ||
                    image_counts[image_id] < per_image_quota;
            });
        if (helps_undercovered_image) select(entry.track_id);
    }

    // Pass 3: fill remaining budget by geometric information score.
    for (const RankedTrack& entry : ranked) {
        if (selected.size() >= target) break;
        select(entry.track_id);
    }
    return selected;
}

void configure_solver(
    ceres::Solver::Options& solver_options,
    const LinearSolverStrategy strategy,
    const bool has_point_blocks,
    const unsigned thread_count,
    const GlobalPositioningOptions& options) {
    solver_options.num_threads = static_cast<int>(thread_count);
    solver_options.visibility_clustering_type = ceres::CANONICAL_VIEWS;
    if (options.max_solver_time_sec > 0.0)
        solver_options.max_solver_time_in_seconds = options.max_solver_time_sec;

    switch (strategy) {
        case LinearSolverStrategy::sparse_schur_cluster:
            if (has_point_blocks) {
                solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
                solver_options.preconditioner_type =
                    ceres::CLUSTER_TRIDIAGONAL;
            } else {
                solver_options.linear_solver_type =
                    ceres::SPARSE_NORMAL_CHOLESKY;
                solver_options.preconditioner_type = ceres::JACOBI;
            }
            break;
        case LinearSolverStrategy::iterative_schur:
            solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
            solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
            solver_options.max_num_iterations = std::min(
                solver_options.max_num_iterations, 40);
            solver_options.max_solver_time_in_seconds =
                options.max_solver_time_sec > 0.0
                    ? std::min(options.max_solver_time_sec, 30.0)
                    : 30.0;
            break;
        case LinearSolverStrategy::sparse_normal_cholesky:
            solver_options.linear_solver_type =
                ceres::SPARSE_NORMAL_CHOLESKY;
            solver_options.preconditioner_type = ceres::JACOBI;
            break;
    }
}

BuiltPositioningProblem build_positioning_problem(
    Scene& scene,
    const GlobalPositioningOptions& options,
    const PositioningAttempt& attempt,
    const bool randomize_positions,
    const bool randomize_points,
    const std::vector<Index>& selected_tracks,
    std::mt19937& generator,
    std::uniform_real_distribution<double>& distribution) {
    BuiltPositioningProblem built;
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    built.problem = ceres::Problem(problem_options);
    built.huber_loss =
        std::make_shared<ceres::HuberLoss>(options.huber_threshold);
    built.candidate_tracks = static_cast<unsigned>(selected_tracks.size());

    std::size_t point_to_camera_capacity = 0;
    if (uses_points(attempt.constraint)) {
        for (Index track_id : selected_tracks) {
            const Track& track = scene.tracks[track_id];
            point_to_camera_capacity +=
                count_registered_observations(scene, track);
        }
    }

    built.scales.reserve(scene.pairs.size() + point_to_camera_capacity);

    for (Image& image : scene.images) {
        if (!image.registered) continue;
        if (options.optimize_positions && randomize_positions)
            image.pose.C = random_point(generator, distribution);
        ++built.valid_images;
    }

    if (uses_cameras(attempt.constraint)) {
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

            built.scales.push_back(1.0);
            double& scale = built.scales.back();
            const Vec3 direction = -(
                image2.pose.R.transpose() *
                pair.relative_pose->translation());
            built.problem.AddResidualBlock(
                new PairwiseDirectionCostAnalytic(direction),
                built.huber_loss.get(),
                image1.pose.C.data(),
                image2.pose.C.data(),
                &scale);
            built.problem.SetParameterLowerBound(&scale, 0, 1e-5);
            ++built.valid_pairs;
        }
    }

    const std::size_t camera_residual_count =
        built.problem.NumResidualBlocks();
    double point_weight = 1.0;
    if (camera_residual_count > 0 &&
        attempt.constraint ==
            GlobalPositioningConstraint::points_and_cameras_balanced &&
        point_to_camera_capacity > 0) {
        point_weight =
            options.constraint_reweight_scale *
            static_cast<double>(camera_residual_count) /
            static_cast<double>(point_to_camera_capacity);
    }
    built.untrusted_loss = std::make_shared<ceres::ScaledLoss>(
        built.huber_loss.get(),
        0.5 * point_weight,
        ceres::DO_NOT_TAKE_OWNERSHIP);
    if (attempt.constraint ==
        GlobalPositioningConstraint::points_and_cameras_balanced) {
        built.trusted_loss = std::make_shared<ceres::ScaledLoss>(
            built.huber_loss.get(),
            point_weight,
            ceres::DO_NOT_TAKE_OWNERSHIP);
    } else {
        built.trusted_loss = built.huber_loss;
    }

    const bool seed_scales_from_depth =
        attempt.ray_initialize_points && !randomize_points;

    if (uses_points(attempt.constraint)) {
        for (Index track_id : selected_tracks) {
            Track& track = scene.tracks[track_id];
            if (options.optimize_points && randomize_points)
                track.position = random_point(generator, distribution);

            bool has_valid_observation = false;
            for (const Observation& observation : track.observations) {
                if (observation.image_id >= scene.images.size()) continue;
                Image& image = scene.images[observation.image_id];
                const WorldRay ray = make_world_ray(scene, image, observation);
                if (!ray.valid) continue;

                built.scales.push_back(1.0);
                double& scale = built.scales.back();
                if (seed_scales_from_depth ||
                    (!options.generate_scales && track.is_triangulated())) {
                    const Vec3 displacement = track.position - image.pose.C;
                    const double depth = ray.direction.dot(displacement);
                    if (depth > 1e-4 && displacement.squaredNorm() > 1e-12) {
                        // residual = ray - scale*(X-C); opt scale ≈ 1/depth.
                        scale = std::max(1e-5, 1.0 / depth);
                    }
                }

                ceres::LossFunction* loss =
                    scene.camera_of(image).trust_intrinsics
                    ? built.trusted_loss.get()
                    : built.untrusted_loss.get();
                built.problem.AddResidualBlock(
                    new PairwiseDirectionCostAnalytic(ray.direction),
                    loss,
                    image.pose.C.data(),
                    track.position.data(),
                    &scale);
                built.problem.SetParameterLowerBound(&scale, 0, 1e-5);
                has_valid_observation = true;
                ++built.observations;
            }
            if (has_valid_observation) {
                ++built.valid_tracks;
                built.has_point_blocks = true;
            }
        }
    }

    const bool optimize_positions =
        options.optimize_positions && !attempt.fix_cameras;
    if (!optimize_positions) {
        for (Image& image : scene.images) {
            if (image.registered &&
                built.problem.HasParameterBlock(image.pose.C.data()))
                built.problem.SetParameterBlockConstant(image.pose.C.data());
        }
    }
    if (!options.optimize_points) {
        for (Index track_id : selected_tracks) {
            Track& track = scene.tracks[track_id];
            if (built.problem.HasParameterBlock(track.position.data()))
                built.problem.SetParameterBlockConstant(track.position.data());
        }
    }
    if (!options.optimize_scales) {
        for (double& scale : built.scales)
            if (built.problem.HasParameterBlock(&scale))
                built.problem.SetParameterBlockConstant(&scale);
    } else {
        // openMVS removes the global scale gauge by fixing one scale variable.
        for (double& scale : built.scales) {
            if (!built.problem.HasParameterBlock(&scale)) continue;
            built.problem.SetParameterBlockConstant(&scale);
            break;
        }
    }

    return built;
}

double mean_residual_norm(ceres::Problem& problem) {
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
    return residual_count == 0
        ? 0.0
        : residual_sum / static_cast<double>(residual_count);
}

GlobalPositioningSummary refine_only_points_bearings(
    Scene& scene,
    const GlobalPositioningOptions& options,
    const std::vector<Index>& selected_tracks) {
    GlobalPositioningSummary result;
    if (selected_tracks.empty()) return result;

    const unsigned initialized =
        initialize_points_from_rays(scene, selected_tracks);
    if (initialized == 0) return result;

    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    auto loss = std::make_unique<ceres::HuberLoss>(
        std::min(options.huber_threshold, 0.03));
    unsigned observations = 0;
    for (Index track_id : selected_tracks) {
        Track& track = scene.tracks[track_id];
        if (!track.position.allFinite()) continue;
        for (const Observation& observation : track.observations) {
            if (observation.image_id >= scene.images.size()) continue;
            Image& image = scene.images[observation.image_id];
            const WorldRay ray = make_world_ray(scene, image, observation);
            if (!ray.valid) continue;
            problem.AddResidualBlock(
                new BearingDirectionCostAnalytic(ray.direction),
                loss.get(),
                image.pose.C.data(),
                track.position.data());
            ++observations;
        }
    }
    if (observations == 0) return result;

    // Remove translation and global-scale gauges. Rotations already define the
    // world axes, so two fixed camera centers give a stable baseline.
    unsigned fixed_cameras = 0;
    for (Image& image : scene.images) {
        if (!image.registered ||
            !problem.HasParameterBlock(image.pose.C.data()))
            continue;
        problem.SetParameterBlockConstant(image.pose.C.data());
        if (++fixed_cameras == 2) break;
    }

    auto* ordering = new ceres::ParameterBlockOrdering;
    for (Index track_id : selected_tracks) {
        Track& track = scene.tracks[track_id];
        if (problem.HasParameterBlock(track.position.data()))
            ordering->AddElementToGroup(track.position.data(), 0);
    }
    for (Image& image : scene.images) {
        if (image.registered &&
            problem.HasParameterBlock(image.pose.C.data()))
            ordering->AddElementToGroup(image.pose.C.data(), 1);
    }

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations =
        static_cast<int>(std::min(options.max_num_iterations, 50U));
    solver_options.max_solver_time_in_seconds =
        options.max_solver_time_sec > 0.0
            ? options.max_solver_time_sec
            : 45.0;
    solver_options.function_tolerance = options.function_tolerance;
    solver_options.num_threads = static_cast<int>(
        scene.thread_count == 0
            ? std::max(1u, std::thread::hardware_concurrency())
            : scene.thread_count);
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
    solver_options.linear_solver_ordering.reset(ordering);

    core::Logger::instance().info(
        "global positioning attempt: only_points/bearing_schur",
        " tracks=", selected_tracks.size(),
        " observations=", observations,
        " initialized=", initialized);
    ceres::Solver::Summary solver_summary;
    ceres::Solve(solver_options, &problem, &solver_summary);
    if (!solver_summary.IsSolutionUsable()) {
        core::Logger::instance().warning(
            "global positioning attempt failed: only_points/bearing_schur",
            " message=", solver_summary.message);
        return result;
    }

    for (Index track_id : selected_tracks) {
        Track& track = scene.tracks[track_id];
        if (!track.position.allFinite() || track.observations.size() < 2)
            continue;
        track.num_inliers = static_cast<std::uint8_t>(
            std::min<std::size_t>(track.observations.size(), 255));
    }
    result.success = true;
    result.positioned_images = scene.registered_count();
    result.positioned_tracks = initialized;
    result.observations = observations;
    result.iterations =
        static_cast<unsigned>(solver_summary.iterations.size());
    result.final_residual = mean_residual_norm(problem);
    core::Logger::instance().info(
        "global positioning succeeded: only_points/bearing_schur",
        " images=", result.positioned_images,
        " tracks=", result.positioned_tracks,
        " observations=", result.observations,
        " iterations=", result.iterations,
        " residual=", result.final_residual);
    return result;
}

bool attempt_positioning(
    Scene& scene,
    const GlobalPositioningOptions& options,
    const PositioningAttempt& attempt,
    const std::vector<Index>& selected_tracks,
    std::mt19937& generator,
    std::uniform_real_distribution<double>& distribution,
    GlobalPositioningSummary& result) {
    if (uses_cameras(attempt.constraint) && scene.pairs.empty()) return false;
    if (uses_points(attempt.constraint) && selected_tracks.empty()) return false;

    if (attempt.ray_initialize_points && uses_points(attempt.constraint)) {
        const unsigned initialized =
            initialize_points_from_rays(scene, selected_tracks);
        core::Logger::instance().info(
            "global positioning: ray-initialized ", initialized, '/',
            selected_tracks.size(), " tracks");
    }

    BuiltPositioningProblem built = build_positioning_problem(
        scene,
        options,
        attempt,
        options.generate_random_positions && !attempt.warm_start_positions,
        options.generate_random_points && !attempt.ray_initialize_points,
        selected_tracks,
        generator,
        distribution);
    if (built.problem.NumResidualBlocks() == 0) return false;

    core::Logger::instance().info(
        "global positioning attempt: ", attempt.label,
        " residuals=", built.problem.NumResidualBlocks(),
        " pairs=", built.valid_pairs,
        " tracks=", built.valid_tracks, '/', built.candidate_tracks,
        " observations=", built.observations,
        " fix_cameras=", attempt.fix_cameras ? 1 : 0);

    ceres::Solver::Options solver_options;
    solver_options.max_num_iterations =
        static_cast<int>(options.max_num_iterations);
    solver_options.function_tolerance = options.function_tolerance;
    const unsigned thread_count =
        scene.thread_count == 0
            ? std::max(1u, std::thread::hardware_concurrency())
            : scene.thread_count;
    configure_solver(
        solver_options,
        attempt.solver,
        built.has_point_blocks,
        thread_count,
        options);

    auto* ordering = new ceres::ParameterBlockOrdering;
    for (double& scale : built.scales)
        if (built.problem.HasParameterBlock(&scale))
            ordering->AddElementToGroup(&scale, 0);
    int camera_group = 1;
    if (built.has_point_blocks) {
        for (Index track_id : selected_tracks) {
            Track& track = scene.tracks[track_id];
            if (built.problem.HasParameterBlock(track.position.data()))
                ordering->AddElementToGroup(track.position.data(), 1);
        }
        ++camera_group;
    }
    for (Image& image : scene.images) {
        if (!image.registered) continue;
        if (built.problem.HasParameterBlock(image.pose.C.data()))
            ordering->AddElementToGroup(image.pose.C.data(), camera_group);
    }
    solver_options.linear_solver_ordering.reset(ordering);

    ceres::Solver::Summary solver_summary;
    ceres::Solve(solver_options, &built.problem, &solver_summary);
    result.iterations =
        static_cast<unsigned>(solver_summary.iterations.size());
    if (!solver_summary.IsSolutionUsable()) {
        core::Logger::instance().warning(
            "global positioning attempt failed: ", attempt.label,
            " message=", solver_summary.message);
        return false;
    }

    result.success = true;
    result.positioned_images = built.valid_images;
    result.positioned_tracks = built.valid_tracks;
    result.observations = built.observations;
    result.final_residual = mean_residual_norm(built.problem);
    // Mark positioned tracks as triangulated so later densification can skip them
    // and fill the remaining tracks before BA.
    if (uses_points(attempt.constraint)) {
        for (Index track_id : selected_tracks) {
            Track& track = scene.tracks[track_id];
            if (!track.position.allFinite() || track.observations.size() < 2)
                continue;
            track.num_inliers = static_cast<std::uint8_t>(
                std::min<std::size_t>(track.observations.size(), 255));
        }
    }
    core::Logger::instance().info(
        "global positioning succeeded: ", attempt.label,
        " images=", result.positioned_images,
        " tracks=", result.positioned_tracks,
        " observations=", result.observations,
        " iterations=", result.iterations,
        " residual=", result.final_residual);
    return true;
}

std::vector<PositioningAttempt> fallback_attempts(
    const GlobalPositioningOptions& options,
    const std::size_t selected_track_count) {
    std::vector<PositioningAttempt> attempts;
    const auto push = [&](
        GlobalPositioningConstraint constraint,
        LinearSolverStrategy solver,
        bool warm_start_positions,
        bool fix_cameras,
        bool ray_initialize_points,
        const char* label) {
        attempts.push_back(
            {constraint, solver, warm_start_positions, fix_cameras,
             ray_initialize_points, label});
    };

    // Large scenes initialize cameras first. The caller then runs a scale-free
    // bearing-only point/camera refinement on the capped track set.
    constexpr std::size_t k_large_scene_tracks = 2000;
    const bool large_scene = selected_track_count >= k_large_scene_tracks;
    if (large_scene &&
        options.constraint != GlobalPositioningConstraint::only_cameras) {
        push(
            GlobalPositioningConstraint::only_cameras,
            LinearSolverStrategy::sparse_normal_cholesky,
            false, false, false,
            "only_cameras/sparse_normal_cholesky");
        return attempts;
    }

    push(
        options.constraint,
        LinearSolverStrategy::sparse_schur_cluster,
        false, false, false,
        "primary/sparse_schur");
    if (options.constraint != GlobalPositioningConstraint::only_cameras) {
        push(
            options.constraint,
            LinearSolverStrategy::iterative_schur,
            false, false, false,
            "primary/iterative_schur");
        push(
            options.constraint,
            LinearSolverStrategy::sparse_normal_cholesky,
            false, false, false,
            "primary/sparse_normal_cholesky");
    }
    return attempts;
}

}  // namespace

GlobalPositioningSummary solve_global_positions(
    Scene& scene,
    const GlobalPositioningOptions& options) {
    core::StageScope stage("sfm.global_positioning");
    GlobalPositioningSummary result;
    if (scene.images.empty()) return result;
    if (scene.pairs.empty() && uses_cameras(options.constraint)) return result;
    if (scene.tracks.empty() && uses_points(options.constraint)) return result;

    const std::vector<Index> selected_tracks =
        select_tracks_for_positioning(scene, options);
    core::Logger::instance().info(
        "global positioning setup: images=", scene.registered_count(),
        " pairs=", scene.pairs.size(),
        " tracks_selected=", selected_tracks.size(),
        " min_views=", options.min_views_per_track,
        " min_tracks=", options.min_tracks_for_positioning,
        " tracks_per_image=", options.tracks_per_registered_image,
        " max_tracks=", options.max_tracks_for_positioning,
        " coverage_grid=", options.coverage_grid_size,
        " max_solver_time_s=", options.max_solver_time_sec);

    std::vector<Vec3> original_centers(scene.images.size());
    for (std::size_t i = 0; i < scene.images.size(); ++i)
        original_centers[i] = scene.images[i].pose.C;
    std::vector<Vec3> original_points(scene.tracks.size());
    for (std::size_t i = 0; i < scene.tracks.size(); ++i)
        original_points[i] = scene.tracks[i].position;

    const auto restore_all = [&] {
        for (std::size_t i = 0; i < scene.images.size(); ++i)
            scene.images[i].pose.C = original_centers[i];
        for (std::size_t i = 0; i < scene.tracks.size(); ++i)
            scene.tracks[i].position = original_points[i];
    };
    const auto restore_points_only = [&] {
        for (std::size_t i = 0; i < scene.tracks.size(); ++i)
            scene.tracks[i].position = original_points[i];
    };

    std::mt19937 generator(options.random_seed);
    std::uniform_real_distribution<double> distribution(-100.0, 100.0);

    std::vector<Vec3> warm_centers;
    bool have_warm_centers = false;

    for (const PositioningAttempt& attempt :
         fallback_attempts(options, selected_tracks.size())) {
        if (attempt.warm_start_positions) {
            if (!have_warm_centers) continue;
            restore_points_only();
            for (std::size_t i = 0; i < scene.images.size(); ++i)
                scene.images[i].pose.C = warm_centers[i];
        } else {
            restore_all();
        }

        GlobalPositioningSummary attempt_summary;
        if (!attempt_positioning(
                scene, options, attempt, selected_tracks, generator,
                distribution, attempt_summary))
            continue;

        if (attempt.constraint == GlobalPositioningConstraint::only_cameras) {
            warm_centers.resize(scene.images.size());
            for (std::size_t i = 0; i < scene.images.size(); ++i)
                warm_centers[i] = scene.images[i].pose.C;
            have_warm_centers = true;
            core::Logger::instance().info(
                "global positioning: camera-only warm start ready, "
                "continuing with point constraints");
            if (options.constraint != GlobalPositioningConstraint::only_cameras &&
                options.ray_initialize_points) {
                GlobalPositioningSummary bearing_summary =
                    refine_only_points_bearings(
                        scene, options, selected_tracks);
                if (bearing_summary.success) return bearing_summary;
            }
            continue;
        }

        return attempt_summary;
    }

    if (have_warm_centers) {
        for (std::size_t i = 0; i < scene.images.size(); ++i)
            scene.images[i].pose.C = warm_centers[i];
        for (std::size_t i = 0; i < scene.tracks.size(); ++i)
            scene.tracks[i].position = original_points[i];
        result.success = true;
        result.positioned_images = scene.registered_count();
        result.positioned_tracks = 0;
        result.observations = 0;
        result.iterations = 0;
        result.final_residual = 0.0;
        core::Logger::instance().warning(
            "global positioning: falling back to camera-only solution; "
            "structure will be recovered by triangulation/BA");
        return result;
    }

    restore_all();
    core::Logger::instance().error(
        "global positioning: all fallback attempts failed");
    return result;
}

}  // namespace aetherscan::sfm
