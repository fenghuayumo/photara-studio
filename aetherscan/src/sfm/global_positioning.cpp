#include "sfm/global_positioning.hpp"
#include "core/logging.hpp"

#include <ceres/ceres.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
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

struct BaselineLengthCostAnalytic : ceres::SizedCostFunction<1, 3, 3> {
    explicit BaselineLengthCostAnalytic(double baseline) : baseline(baseline) {}

    bool Evaluate(
        double const* const* parameters,
        double* residuals,
        double** jacobians) const override {
        const Eigen::Map<const Vec3> first(parameters[0]);
        const Eigen::Map<const Vec3> second(parameters[1]);
        const Vec3 difference = second - first;
        const double length = difference.norm();
        if (!(length > 1e-10) || !std::isfinite(length)) return false;
        residuals[0] = length - baseline;
        if (jacobians) {
            const Eigen::RowVector3d direction = difference.transpose() / length;
            if (jacobians[0]) {
                Eigen::Map<Eigen::RowVector3d> first_jacobian(jacobians[0]);
                first_jacobian = -direction;
            }
            if (jacobians[1]) {
                Eigen::Map<Eigen::RowVector3d> second_jacobian(jacobians[1]);
                second_jacobian = direction;
            }
        }
        return true;
    }

    double baseline;
};

class AdaptiveHuberLoss final : public ceres::LossFunction {
public:
    AdaptiveHuberLoss(double threshold, double base_weight)
        : huber_(threshold), base_weight_(base_weight) {}

    void Evaluate(double squared_norm, double out[3]) const override {
        huber_.Evaluate(squared_norm, out);
        const double scale = base_weight_ * robust_weight_;
        out[0] *= scale;
        out[1] *= scale;
        out[2] *= scale;
    }

    void set_robust_weight(double weight) {
        robust_weight_ = std::clamp(weight, 0.0, 1.0);
    }
    [[nodiscard]] double robust_weight() const { return robust_weight_; }

private:
    ceres::HuberLoss huber_;
    double base_weight_{1.0};
    double robust_weight_{1.0};
};

enum class PositioningResidualType : std::uint8_t {
    camera_camera,
    camera_point,
};

struct PositioningResidual {
    ceres::ResidualBlockId id{nullptr};
    AdaptiveHuberLoss* loss{nullptr};
    PositioningResidualType type{PositioningResidualType::camera_point};
    unsigned outlier_streak{0};
    bool quarantined{false};
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
    std::vector<std::unique_ptr<AdaptiveHuberLoss>> losses;
    std::vector<PositioningResidual> residuals;
    std::vector<double> scales;
    unsigned valid_images{0};
    unsigned valid_pairs{0};
    unsigned valid_tracks{0};
    unsigned observations{0};
    unsigned candidate_tracks{0};
    bool has_point_blocks{false};
    std::size_t scale_anchor{std::numeric_limits<std::size_t>::max()};
    double scale_anchor_score{-1.0};
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

Index select_anchor_camera(
    const Scene& scene,
    const ceres::Problem& problem,
    const std::vector<Index>& selected_tracks) {
    std::vector<double> scores(scene.images.size(), 0.0);
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || pair.id1 >= scene.images.size() ||
            pair.id2 >= scene.images.size())
            continue;
        const double weight = std::max(0.F, pair.composite_weight());
        scores[pair.id1] += weight;
        scores[pair.id2] += weight;
    }
    for (Index track_id : selected_tracks) {
        if (track_id >= scene.tracks.size()) continue;
        const Track& track = scene.tracks[track_id];
        const double support = static_cast<double>(
            std::max<std::size_t>(1, track.observations.size()));
        for (const Observation& observation : track.observations)
            if (observation.image_id < scores.size())
                scores[observation.image_id] += support;
    }

    Index best = k_invalid;
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        const Image& image = scene.images[image_id];
        if (!image.registered ||
            !problem.HasParameterBlock(image.pose.C.data()))
            continue;
        if (best == k_invalid || scores[image_id] > scores[best]) best = image_id;
    }
    return best;
}

void translate_positioning_gauge(
    Scene& scene,
    const std::vector<Index>& selected_tracks,
    const Vec3& offset) {
    for (Image& image : scene.images)
        if (image.registered) image.pose.C -= offset;
    for (Index track_id : selected_tracks)
        if (track_id < scene.tracks.size() &&
            scene.tracks[track_id].position.allFinite())
            scene.tracks[track_id].position -= offset;
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
    built.losses.reserve(scene.pairs.size() + point_to_camera_capacity);
    built.residuals.reserve(scene.pairs.size() + point_to_camera_capacity);

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
            auto loss = std::make_unique<AdaptiveHuberLoss>(
                options.huber_threshold, 1.0);
            AdaptiveHuberLoss* loss_ptr = loss.get();
            const ceres::ResidualBlockId residual_id =
                built.problem.AddResidualBlock(
                new PairwiseDirectionCostAnalytic(direction),
                loss_ptr,
                image1.pose.C.data(),
                image2.pose.C.data(),
                &scale);
            built.losses.push_back(std::move(loss));
            built.residuals.push_back({
                residual_id, loss_ptr,
                PositioningResidualType::camera_camera});
            built.problem.SetParameterLowerBound(&scale, 0, 1e-5);
            const double anchor_score = pair.composite_weight();
            if (anchor_score > built.scale_anchor_score) {
                built.scale_anchor_score = anchor_score;
                built.scale_anchor = built.scales.size() - 1;
            }
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

                const double calibration_weight =
                    scene.camera_of(image).trust_intrinsics ? 1.0 : 0.5;
                auto loss = std::make_unique<AdaptiveHuberLoss>(
                    options.huber_threshold,
                    point_weight * calibration_weight);
                AdaptiveHuberLoss* loss_ptr = loss.get();
                const ceres::ResidualBlockId residual_id =
                    built.problem.AddResidualBlock(
                    new PairwiseDirectionCostAnalytic(ray.direction),
                    loss_ptr,
                    image.pose.C.data(),
                    track.position.data(),
                    &scale);
                built.losses.push_back(std::move(loss));
                built.residuals.push_back({
                    residual_id, loss_ptr,
                    PositioningResidualType::camera_point});
                built.problem.SetParameterLowerBound(&scale, 0, 1e-5);
                const double anchor_score =
                    static_cast<double>(count_registered_observations(scene, track)) *
                    (scene.camera_of(image).trust_intrinsics ? 1.0 : 0.5);
                if (anchor_score > built.scale_anchor_score) {
                    built.scale_anchor_score = anchor_score;
                    built.scale_anchor = built.scales.size() - 1;
                }
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
    const bool has_similarity_gauge =
        !built.has_point_blocks || options.optimize_points;
    if (!optimize_positions) {
        for (Image& image : scene.images) {
            if (image.registered &&
                built.problem.HasParameterBlock(image.pose.C.data()))
                built.problem.SetParameterBlockConstant(image.pose.C.data());
        }
    } else if (has_similarity_gauge) {
        const Index anchor =
            select_anchor_camera(scene, built.problem, selected_tracks);
        if (anchor != k_invalid) {
            const Vec3 offset = scene.images[anchor].pose.C;
            translate_positioning_gauge(scene, selected_tracks, offset);
            built.problem.SetParameterBlockConstant(
                scene.images[anchor].pose.C.data());
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
    } else if (has_similarity_gauge &&
               built.scale_anchor < built.scales.size() &&
               built.problem.HasParameterBlock(
                   &built.scales[built.scale_anchor])) {
        // Fix the most strongly supported inverse-depth scale instead of the
        // first observation encountered, which can be a weak graph boundary.
        built.problem.SetParameterBlockConstant(
            &built.scales[built.scale_anchor]);
    }

    return built;
}

double mean_residual_norm(ceres::Problem& problem) {
    std::vector<ceres::ResidualBlockId> residual_blocks;
    problem.GetResidualBlocks(&residual_blocks);
    if (residual_blocks.empty()) return 0.0;
    ceres::Problem::EvaluateOptions evaluate_options;
    evaluate_options.residual_blocks = residual_blocks;
    evaluate_options.apply_loss_function = false;
    double cost = 0.0;
    std::vector<double> residuals;
    if (!problem.Evaluate(
            evaluate_options, &cost, &residuals, nullptr, nullptr))
        return 0.0;

    double residual_sum = 0.0;
    std::size_t offset = 0;
    unsigned residual_count = 0;
    for (const ceres::ResidualBlockId residual_id : residual_blocks) {
        const ceres::CostFunction* function =
            problem.GetCostFunctionForResidualBlock(residual_id);
        if (!function) continue;
        const int dimension = function->num_residuals();
        if (dimension <= 0 ||
            offset + static_cast<std::size_t>(dimension) > residuals.size())
            return 0.0;
        double squared_norm = 0.0;
        for (int i = 0; i < dimension; ++i)
            squared_norm += residuals[offset + static_cast<std::size_t>(i)] *
                            residuals[offset + static_cast<std::size_t>(i)];
        residual_sum += std::sqrt(squared_norm);
        offset += static_cast<std::size_t>(dimension);
        ++residual_count;
    }
    return residual_count == 0
        ? 0.0
        : residual_sum / static_cast<double>(residual_count);
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    const std::size_t index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(fraction * (values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

bool evaluate_positioning_residuals(
    ceres::Problem& problem,
    const std::vector<PositioningResidual>& records,
    std::vector<double>& norms) {
    norms.clear();
    if (records.empty()) return false;
    ceres::Problem::EvaluateOptions options;
    options.apply_loss_function = false;
    options.residual_blocks.reserve(records.size());
    for (const PositioningResidual& record : records)
        options.residual_blocks.push_back(record.id);
    double cost = 0.0;
    std::vector<double> residuals;
    if (!problem.Evaluate(options, &cost, &residuals, nullptr, nullptr) ||
        residuals.size() != records.size() * 3)
        return false;
    norms.resize(records.size());
    for (std::size_t i = 0; i < records.size(); ++i) {
        const std::size_t offset = i * 3;
        norms[i] = std::sqrt(
            residuals[offset] * residuals[offset] +
            residuals[offset + 1] * residuals[offset + 1] +
            residuals[offset + 2] * residuals[offset + 2]);
    }
    return true;
}

struct RobustDistribution {
    double median{0.0};
    double p90{0.0};
    double sigma{0.0};
    double cutoff{0.0};
};

RobustDistribution robust_distribution(
    const std::vector<double>& residuals,
    const std::vector<std::size_t>& indices,
    const GlobalPositioningOptions& options) {
    RobustDistribution result;
    if (indices.empty()) return result;
    std::vector<double> values;
    values.reserve(indices.size());
    for (const std::size_t index : indices) values.push_back(residuals[index]);
    result.median = percentile(values, 0.5);
    result.p90 = percentile(values, 0.9);
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (const double value : values)
        deviations.push_back(std::abs(value - result.median));
    result.sigma = std::max(1e-6, 1.4826 * percentile(deviations, 0.5));
    result.cutoff = std::max(
        options.huber_threshold,
        options.irls_tuning_constant * result.sigma);
    return result;
}

struct IrlsUpdate {
    bool valid{false};
    double maximum_weight_change{0.0};
    double median{0.0};
    double p90{0.0};
    unsigned downweighted{0};
    unsigned quarantined{0};
};

IrlsUpdate update_irls_weights(
    ceres::Problem& problem,
    std::vector<PositioningResidual>& records,
    const GlobalPositioningOptions& options) {
    IrlsUpdate update;
    std::vector<double> norms;
    if (!evaluate_positioning_residuals(problem, records, norms)) return update;
    update.valid = true;
    update.median = percentile(norms, 0.5);
    update.p90 = percentile(norms, 0.9);

    std::vector<std::size_t> camera_camera;
    std::vector<std::size_t> camera_point;
    camera_camera.reserve(records.size());
    camera_point.reserve(records.size());
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (records[i].type == PositioningResidualType::camera_camera)
            camera_camera.push_back(i);
        else
            camera_point.push_back(i);
    }
    const RobustDistribution camera_distribution =
        robust_distribution(norms, camera_camera, options);
    const RobustDistribution point_distribution =
        robust_distribution(norms, camera_point, options);
    const double minimum_weight =
        std::clamp(options.irls_min_weight, 1e-8, 1.0);

    for (std::size_t i = 0; i < records.size(); ++i) {
        PositioningResidual& record = records[i];
        const RobustDistribution& distribution =
            record.type == PositioningResidualType::camera_camera
            ? camera_distribution
            : point_distribution;
        const double cutoff = std::max(1e-8, distribution.cutoff);
        if (norms[i] > cutoff) {
            ++record.outlier_streak;
        } else if (record.outlier_streak > 0) {
            --record.outlier_streak;
        }
        if (options.irls_quarantine_after > 0 &&
            record.outlier_streak >= options.irls_quarantine_after)
            record.quarantined = true;

        const double ratio = norms[i] / cutoff;
        const double denominator = 1.0 + ratio * ratio;
        double weight = 1.0 / (denominator * denominator);
        if (record.quarantined) weight = minimum_weight;
        weight = std::clamp(weight, minimum_weight, 1.0);
        update.maximum_weight_change = std::max(
            update.maximum_weight_change,
            std::abs(weight - record.loss->robust_weight()));
        record.loss->set_robust_weight(weight);
        if (weight < 0.95) ++update.downweighted;
        if (record.quarantined) ++update.quarantined;
    }
    return update;
}

struct BaselineAnchor {
    Index first{k_invalid};
    Index second{k_invalid};
    double length{0.0};
};

BaselineAnchor select_baseline_anchor(
    const Scene& scene, const ceres::Problem& problem, Index center_anchor) {
    struct Candidate {
        Index first{};
        Index second{};
        double length{};
        double reliability{};
    };
    std::vector<Candidate> candidates;
    std::vector<double> lengths;
    candidates.reserve(scene.pairs.size());
    lengths.reserve(scene.pairs.size());
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || pair.id1 >= scene.images.size() ||
            pair.id2 >= scene.images.size())
            continue;
        const Image& first = scene.images[pair.id1];
        const Image& second = scene.images[pair.id2];
        if (!first.registered || !second.registered ||
            !problem.HasParameterBlock(first.pose.C.data()) ||
            !problem.HasParameterBlock(second.pose.C.data()))
            continue;
        const double length = (second.pose.C - first.pose.C).norm();
        if (!(length > 1e-6) || !std::isfinite(length)) continue;
        candidates.push_back({
            pair.id1, pair.id2, length,
            std::max(1e-6F, pair.composite_weight())});
        lengths.push_back(length);
    }
    if (!candidates.empty()) {
        auto middle = lengths.begin() + lengths.size() / 2;
        std::nth_element(lengths.begin(), middle, lengths.end());
        const double median = std::max(1e-6, *middle);
        const Candidate* best = &candidates.front();
        double best_score = -1.0;
        for (const Candidate& candidate : candidates) {
            const double baseline_factor =
                std::clamp(candidate.length / median, 0.25, 4.0);
            const double score = candidate.reliability * baseline_factor;
            if (score > best_score) {
                best_score = score;
                best = &candidate;
            }
        }
        return {best->first, best->second, best->length};
    }

    // A point-only scene can legitimately have no pair constraints. In that
    // case anchor scale using the camera farthest from the translation anchor.
    if (center_anchor == k_invalid) return {};
    BaselineAnchor result;
    result.first = center_anchor;
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        if (image_id == center_anchor || !scene.images[image_id].registered ||
            !problem.HasParameterBlock(scene.images[image_id].pose.C.data()))
            continue;
        const double length =
            (scene.images[image_id].pose.C -
             scene.images[center_anchor].pose.C).norm();
        if (std::isfinite(length) && length > result.length) {
            result.second = image_id;
            result.length = length;
        }
    }
    return result;
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
    std::vector<std::unique_ptr<AdaptiveHuberLoss>> losses;
    std::vector<PositioningResidual> residual_records;
    ceres::Problem problem(problem_options);
    unsigned observations = 0;
    for (Index track_id : selected_tracks) {
        Track& track = scene.tracks[track_id];
        if (!track.position.allFinite()) continue;
        for (const Observation& observation : track.observations) {
            if (observation.image_id >= scene.images.size()) continue;
            Image& image = scene.images[observation.image_id];
            const WorldRay ray = make_world_ray(scene, image, observation);
            if (!ray.valid) continue;
            auto loss = std::make_unique<AdaptiveHuberLoss>(
                std::min(options.huber_threshold, 0.03), 1.0);
            AdaptiveHuberLoss* loss_ptr = loss.get();
            const ceres::ResidualBlockId residual_id = problem.AddResidualBlock(
                new BearingDirectionCostAnalytic(ray.direction),
                loss_ptr,
                image.pose.C.data(),
                track.position.data());
            losses.push_back(std::move(loss));
            residual_records.push_back({
                residual_id, loss_ptr,
                PositioningResidualType::camera_point});
            ++observations;
        }
    }
    if (observations == 0) return result;

    // Remove exactly the four unobservable DOFs: one fixed center removes
    // global translation; one scalar baseline residual removes global scale.
    // Fixing two complete centers would also freeze two observable directions.
    const Index center_anchor =
        select_anchor_camera(scene, problem, selected_tracks);
    const BaselineAnchor baseline =
        select_baseline_anchor(scene, problem, center_anchor);
    if (center_anchor == k_invalid || baseline.first == k_invalid ||
        baseline.second == k_invalid || !(baseline.length > 1e-6))
        return result;
    const Vec3 offset = scene.images[center_anchor].pose.C;
    translate_positioning_gauge(scene, selected_tracks, offset);
    problem.SetParameterBlockConstant(
        scene.images[center_anchor].pose.C.data());
    problem.AddResidualBlock(
        new BaselineLengthCostAnalytic(baseline.length), nullptr,
        scene.images[baseline.first].pose.C.data(),
        scene.images[baseline.second].pose.C.data());

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
        " initialized=", initialized,
        " center_anchor=", center_anchor,
        " baseline_pair=", baseline.first, '-', baseline.second,
        " baseline=", baseline.length);
    const auto solve_started = std::chrono::steady_clock::now();
    double total_solve_seconds = 0.0;
    bool have_usable_solution = false;
    for (unsigned pass = 0; pass <= options.max_irls_iterations; ++pass) {
        double remaining_seconds = options.max_solver_time_sec;
        if (options.max_solver_time_sec > 0.0) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - solve_started).count();
            remaining_seconds = options.max_solver_time_sec - elapsed;
            if (remaining_seconds <= 0.05) break;
            const unsigned remaining_passes =
                options.max_irls_iterations - pass + 1;
            solver_options.max_solver_time_in_seconds = std::max(
                0.05,
                pass == 0
                    ? 0.6 * remaining_seconds
                    : remaining_seconds /
                        static_cast<double>(remaining_passes));
        }
        solver_options.max_num_iterations = static_cast<int>(
            pass == 0
            ? std::min(options.max_num_iterations, 30U)
            : std::max(1U, std::min(
                options.irls_inner_iterations, options.max_num_iterations)));
        ceres::Solver::Summary pass_summary;
        ceres::Solve(solver_options, &problem, &pass_summary);
        total_solve_seconds += pass_summary.total_time_in_seconds;
        result.iterations +=
            static_cast<unsigned>(pass_summary.iterations.size());
        if (!pass_summary.IsSolutionUsable()) {
            if (!have_usable_solution) {
                core::Logger::instance().warning(
                    "global positioning attempt failed: only_points/bearing_schur",
                    " message=", pass_summary.message);
                return result;
            }
            break;
        }
        have_usable_solution = true;
        if (options.max_irls_iterations == 0) break;
        const IrlsUpdate update = update_irls_weights(
            problem, residual_records, options);
        if (!update.valid) break;
        result.irls_iterations = pass;
        result.downweighted_constraints = update.downweighted;
        result.quarantined_constraints = update.quarantined;
        result.median_residual = update.median;
        result.p90_residual = update.p90;
        core::Logger::instance().info(
            "global positioning bearing IRLS: pass=", pass,
            " median=", update.median,
            " p90=", update.p90,
            " downweighted=", update.downweighted,
            " quarantined=", update.quarantined,
            " max_weight_change=", update.maximum_weight_change);
        if (pass == options.max_irls_iterations ||
            update.maximum_weight_change < options.irls_weight_convergence)
            break;
        result.irls_iterations = pass + 1;
    }
    if (!have_usable_solution) return result;

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
    result.final_residual = mean_residual_norm(problem);
    core::Logger::instance().info(
        "global positioning succeeded: only_points/bearing_schur",
        " images=", result.positioned_images,
        " tracks=", result.positioned_tracks,
        " observations=", result.observations,
        " iterations=", result.iterations,
        " irls=", result.irls_iterations,
        " downweighted=", result.downweighted_constraints,
        " quarantined=", result.quarantined_constraints,
        " solve_s=", total_solve_seconds,
        " median=", result.median_residual,
        " p90=", result.p90_residual,
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

    const auto solve_started = std::chrono::steady_clock::now();
    const unsigned maximum_irls = options.max_irls_iterations;
    double total_solve_seconds = 0.0;
    bool have_usable_solution = false;
    IrlsUpdate final_update;
    for (unsigned pass = 0; pass <= maximum_irls; ++pass) {
        double remaining_seconds = options.max_solver_time_sec;
        if (options.max_solver_time_sec > 0.0) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - solve_started).count();
            remaining_seconds = options.max_solver_time_sec - elapsed;
            if (remaining_seconds <= 0.05) break;
            const unsigned remaining_passes = maximum_irls - pass + 1;
            const double pass_budget = pass == 0
                ? 0.6 * remaining_seconds
                : remaining_seconds / static_cast<double>(remaining_passes);
            solver_options.max_solver_time_in_seconds =
                std::max(0.05, pass_budget);
        }
        solver_options.max_num_iterations = static_cast<int>(
            pass == 0
            ? std::min(options.max_num_iterations, 30U)
            : std::max(1U, std::min(
                options.irls_inner_iterations, options.max_num_iterations)));

        ceres::Solver::Summary pass_summary;
        ceres::Solve(solver_options, &built.problem, &pass_summary);
        total_solve_seconds += pass_summary.total_time_in_seconds;
        result.iterations +=
            static_cast<unsigned>(pass_summary.iterations.size());
        if (!pass_summary.IsSolutionUsable()) {
            if (!have_usable_solution) {
                core::Logger::instance().warning(
                    "global positioning attempt failed: ", attempt.label,
                    " message=", pass_summary.message);
                return false;
            }
            break;
        }
        have_usable_solution = true;

        if (maximum_irls == 0) break;

        final_update = update_irls_weights(
            built.problem, built.residuals, options);
        if (!final_update.valid) break;
        result.irls_iterations = pass;
        result.downweighted_constraints = final_update.downweighted;
        result.quarantined_constraints = final_update.quarantined;
        result.median_residual = final_update.median;
        result.p90_residual = final_update.p90;
        core::Logger::instance().info(
            "global positioning IRLS: pass=", pass,
            " median=", final_update.median,
            " p90=", final_update.p90,
            " downweighted=", final_update.downweighted,
            " quarantined=", final_update.quarantined,
            " max_weight_change=", final_update.maximum_weight_change);
        if (pass == maximum_irls ||
            final_update.maximum_weight_change <
                options.irls_weight_convergence)
            break;
        result.irls_iterations = pass + 1;
    }
    if (!have_usable_solution) return false;

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
        " irls=", result.irls_iterations,
        " downweighted=", result.downweighted_constraints,
        " quarantined=", result.quarantined_constraints,
        " solve_s=", total_solve_seconds,
        " median=", result.median_residual,
        " p90=", result.p90_residual,
        " residual=", result.final_residual);
    return true;
}

std::vector<PositioningAttempt> fallback_attempts(
    const GlobalPositioningOptions& options,
    const bool allow_camera_fallback) {
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
        // Relative translations remain a last-resort initializer, never a
        // scene-size-driven primary path.
        if (allow_camera_fallback) {
            push(
                GlobalPositioningConstraint::only_cameras,
                LinearSolverStrategy::sparse_normal_cholesky,
                false, false, false,
                "fallback/only_cameras");
        }
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
         fallback_attempts(options, !selected_tracks.empty())) {
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
            if (options.constraint ==
                GlobalPositioningConstraint::only_cameras)
                return attempt_summary;
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
