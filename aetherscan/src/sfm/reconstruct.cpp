#include "sfm/reconstruct.hpp"
#include "sfm/appearance.hpp"

#include "core/logging.hpp"
#include "io/image.hpp"
#include "parallel/thread_pool.hpp"
#include "sfm/bundle.hpp"
#include "sfm/pair_weighting.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"
#include "sfm/submap_recovery.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace aetherscan::sfm {
namespace {

double median_value(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const auto middle = values.begin() +
        static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

double percentile_value(
    std::vector<double> values, const double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(std::clamp(
        fraction * static_cast<double>(values.size() - 1),
        0.0, static_cast<double>(values.size() - 1)));
    return values[index];
}

bool regularize_complete_orbit(
    Scene& scene, const GlobalPositioningOptions& options,
    bool& preserve_orbit_centers) {
    preserve_orbit_centers = false;
    if (!options.regularize_complete_orbits || scene.images.size() < 16 ||
        scene.registered_count() != scene.images.size())
        return false;

    unsigned verified_neighbors = 0;
    for (Index image_id = 0; image_id + 1 < scene.images.size(); ++image_id) {
        const ImagePair* pair = scene.find_pair(image_id, image_id + 1);
        // This is sequence evidence only. Rotation averaging may already have
        // deactivated a locally inconsistent edge, but the verified relative
        // pose still proves that the consecutive images overlap.
        if (pair && pair->relative_pose.has_value())
            ++verified_neighbors;
    }
    const double neighbor_ratio = static_cast<double>(verified_neighbors) /
        static_cast<double>(scene.images.size() - 1);
    Vec3 mean = Vec3::Zero();
    for (const Image& image : scene.images) mean += image.pose.C;
    mean /= static_cast<double>(scene.images.size());
    Mat3 covariance = Mat3::Zero();
    for (const Image& image : scene.images) {
        const Vec3 centered = image.pose.C - mean;
        covariance += centered * centered.transpose();
    }
    const Eigen::SelfAdjointEigenSolver<Mat3> eigensolver(covariance);
    if (eigensolver.info() != Eigen::Success) return false;
    Vec3 plane_normal = eigensolver.eigenvectors().col(0).normalized();
    const Vec3 axis1 = eigensolver.eigenvectors().col(2).normalized();
    double normal_score = 0.0;
    for (const Image& image : scene.images)
        normal_score += image.pose.R.row(1).dot(plane_normal);
    if (normal_score < 0.0) plane_normal = -plane_normal;
    const Vec3 axis2 = plane_normal.cross(axis1).normalized();

    Eigen::MatrixXd circle_system(scene.images.size(), 3);
    Eigen::VectorXd circle_rhs(scene.images.size());
    for (std::size_t index = 0; index < scene.images.size(); ++index) {
        const Vec3 centered = scene.images[index].pose.C - mean;
        const double x = centered.dot(axis1);
        const double y = centered.dot(axis2);
        circle_system.row(index) << 2.0 * x, 2.0 * y, 1.0;
        circle_rhs[static_cast<Eigen::Index>(index)] = x * x + y * y;
    }
    const Eigen::Vector3d circle =
        circle_system.colPivHouseholderQr().solve(circle_rhs);
    if (!circle.allFinite()) return false;
    const Vec3 orbit_center =
        mean + circle[0] * axis1 + circle[1] * axis2;

    std::vector<double> radii;
    std::vector<double> plane_distances;
    std::vector<double> angles;
    radii.reserve(scene.images.size());
    plane_distances.reserve(scene.images.size());
    angles.reserve(scene.images.size());
    for (const Image& image : scene.images) {
        const Vec3 radial = image.pose.C - orbit_center;
        const double x = radial.dot(axis1);
        const double y = radial.dot(axis2);
        radii.push_back(std::hypot(x, y));
        plane_distances.push_back(std::abs((image.pose.C - mean).dot(
            plane_normal)));
        double angle = std::atan2(y, x);
        if (!angles.empty()) {
            while (angle - angles.back() > std::numbers::pi)
                angle -= 2.0 * std::numbers::pi;
            while (angle - angles.back() < -std::numbers::pi)
                angle += 2.0 * std::numbers::pi;
        }
        angles.push_back(angle);
    }
    const double radius = median_value(radii);
    if (!(radius > 1e-8) || !std::isfinite(radius)) return false;
    std::vector<double> radius_errors;
    radius_errors.reserve(radii.size());
    for (const double value : radii)
        radius_errors.push_back(std::abs(value - radius));

    const double mean_index =
        0.5 * static_cast<double>(scene.images.size() - 1);
    const double mean_angle =
        std::accumulate(angles.begin(), angles.end(), 0.0) /
        static_cast<double>(angles.size());
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t index = 0; index < angles.size(); ++index) {
        const double centered_index =
            static_cast<double>(index) - mean_index;
        numerator += centered_index * (angles[index] - mean_angle);
        denominator += centered_index * centered_index;
    }
    if (!(denominator > 0.0)) return false;
    const double angular_step = numerator / denominator;
    const double angular_intercept = mean_angle - angular_step * mean_index;
    const double angular_coverage = std::abs(angular_step) *
        static_cast<double>(scene.images.size() - 1);
    std::vector<double> angular_residuals;
    angular_residuals.reserve(angles.size());
    unsigned consistent_steps = 0;
    for (std::size_t index = 0; index < angles.size(); ++index) {
        angular_residuals.push_back(std::abs(
            angles[index] -
            (angular_intercept + angular_step * static_cast<double>(index))));
        if (index > 0 &&
            (angles[index] - angles[index - 1]) * angular_step > 0.0)
            ++consistent_steps;
    }
    const double direction_ratio = static_cast<double>(consistent_steps) /
        static_cast<double>(scene.images.size() - 1);

    std::vector<double> roll_errors;
    roll_errors.reserve(scene.images.size());
    for (const Image& image : scene.images) {
        roll_errors.push_back(std::acos(std::clamp(
            image.pose.R.row(1).dot(plane_normal), -1.0, 1.0)));
    }
    const double planarity_ratio =
        percentile_value(plane_distances, 0.95) / radius;
    const double radius_error_ratio =
        percentile_value(radius_errors, 0.95) / radius;
    const double angular_residual_p95 =
        percentile_value(angular_residuals, 0.95);
    const double roll_error_p95 = percentile_value(roll_errors, 0.95);
    std::vector<double> center_steps;
    center_steps.reserve(scene.images.size() - 1);
    for (std::size_t index = 1; index < scene.images.size(); ++index)
        center_steps.push_back((
            scene.images[index].pose.C -
            scene.images[index - 1].pose.C).norm());
    const double median_center_step = median_value(center_steps);
    const double center_step_p95_ratio = median_center_step > 1e-10
        ? percentile_value(center_steps, 0.95) / median_center_step
        : std::numeric_limits<double>::infinity();
    const double center_step_max_ratio = median_center_step > 1e-10
        ? *std::max_element(center_steps.begin(), center_steps.end()) /
            median_center_step
        : std::numeric_limits<double>::infinity();
    const bool trajectory_needs_repair =
        radius_error_ratio > 0.08 || center_step_p95_ratio > 2.0 ||
        center_step_max_ratio > 1.5 ||
        angular_residual_p95 > 10.0 * std::numbers::pi / 180.0;
    core::Logger::instance().debug(
        "global orbit probe: views=", scene.images.size(),
        " neighbor_ratio=", neighbor_ratio,
        " planarity_ratio=", planarity_ratio,
        " radius_error_ratio=", radius_error_ratio,
        " fitted_coverage_deg=", angular_coverage * 180.0 /
            std::numbers::pi,
        " direction_ratio=", direction_ratio,
        " angular_residual_p95_deg=", angular_residual_p95 * 180.0 /
            std::numbers::pi,
        " roll_error_p95_deg=", roll_error_p95 * 180.0 /
            std::numbers::pi,
        " center_step_p95_ratio=", center_step_p95_ratio,
        " center_step_max_ratio=", center_step_max_ratio,
        " needs_repair=", trajectory_needs_repair ? 1 : 0);
    if (neighbor_ratio < 0.50 ||
        planarity_ratio > 0.03 || radius_error_ratio > 0.30 ||
        angular_coverage < 300.0 * std::numbers::pi / 180.0 ||
        angular_coverage > 420.0 * std::numbers::pi / 180.0 ||
        std::abs(angular_step) < 0.2 * std::numbers::pi / 180.0 ||
        std::abs(angular_step) > 20.0 * std::numbers::pi / 180.0 ||
        direction_ratio < 0.70 ||
        angular_residual_p95 > 35.0 * std::numbers::pi / 180.0 ||
        roll_error_p95 > 15.0 * std::numbers::pi / 180.0)
        return false;

    // A clean orbit needs no projection, but its centers should remain fixed
    // during the short distortion polish. Otherwise a small set of repetitive
    // observations can pull one otherwise well-conditioned camera off-track.
    preserve_orbit_centers = true;
    if (!trajectory_needs_repair) {
        core::Logger::instance().info(
            "global: preserving well-conditioned circular orbit centers views=",
            scene.images.size(),
            " center_step_p95_ratio=", center_step_p95_ratio,
            " center_step_max_ratio=", center_step_max_ratio);
        return false;
    }

    std::vector<Vec3> regularized_centers(scene.images.size());
    for (std::size_t index = 0; index < scene.images.size(); ++index) {
        const double angle = angular_intercept +
            angular_step * static_cast<double>(index);
        regularized_centers[index] = orbit_center + radius *
            (std::cos(angle) * axis1 + std::sin(angle) * axis2);
    }

    double best_height_fraction = 0.0;
    double best_focus_score = std::numeric_limits<double>::infinity();
    for (int step = -20; step <= 20; ++step) {
        const double fraction = 0.01 * static_cast<double>(step);
        const Vec3 focus =
            orbit_center + fraction * radius * plane_normal;
        std::vector<double> errors;
        errors.reserve(scene.images.size());
        for (std::size_t index = 0; index < scene.images.size(); ++index) {
            const Vec3 direction =
                (focus - regularized_centers[index]).normalized();
            const Vec3 optical_axis =
                scene.images[index].pose.R.row(2).transpose();
            errors.push_back(std::acos(std::clamp(
                direction.dot(optical_axis), -1.0, 1.0)));
        }
        const double score = median_value(std::move(errors));
        if (score < best_focus_score) {
            best_focus_score = score;
            best_height_fraction = fraction;
        }
    }
    if (best_focus_score > 15.0 * std::numbers::pi / 180.0) {
        preserve_orbit_centers = false;
        return false;
    }

    const Vec3 focus = orbit_center +
        best_height_fraction * radius * plane_normal;
    const double tangent_sign = angular_step >= 0.0 ? 1.0 : -1.0;
    for (std::size_t index = 0; index < scene.images.size(); ++index) {
        Image& image = scene.images[index];
        image.pose.C = regularized_centers[index];
        Vec3 x_axis = tangent_sign * plane_normal.cross(
            image.pose.C - orbit_center).normalized();
        Vec3 z_axis = focus - image.pose.C;
        z_axis -= x_axis * x_axis.dot(z_axis);
        if (z_axis.norm() <= 1e-10) return false;
        z_axis.normalize();
        const Vec3 y_axis = z_axis.cross(x_axis).normalized();
        image.pose.R.row(0) = x_axis.transpose();
        image.pose.R.row(1) = y_axis.transpose();
        image.pose.R.row(2) = z_axis.transpose();
    }
    core::Logger::instance().info(
        "global: regularized complete circular orbit views=",
        scene.images.size(),
        " angular_step_deg=", angular_step * 180.0 / std::numbers::pi,
        " coverage_deg=", angular_coverage * 180.0 / std::numbers::pi,
        " focus_height_ratio=", best_height_fraction);
    return true;
}

#if !defined(AETHERSCAN_RECONSTRUCTION_CACHE_BUILD_ID)
#define AETHERSCAN_RECONSTRUCTION_CACHE_BUILD_ID "unconfigured"
#endif

void append_optimizer(
    FingerprintBuilder& key, const ba::OptimizerOptions& options) {
    key.append(options.maximum_iterations);
    key.append(options.maximum_pcg_iterations);
    key.append(options.huber_delta);
    key.append(options.minimum_depth);
    key.append(options.initial_damping);
    key.append(options.minimum_damping);
    key.append(options.maximum_damping);
    key.append(options.function_tolerance);
    key.append(options.step_tolerance);
    key.append(options.pcg_tolerance);
    key.append(options.fix_first_pose);
    key.append(options.fix_first_point);
    key.append(options.optimize_rotations);
    key.append(options.optimize_translations);
    key.append(options.optimize_points);
    key.append(options.optimize_focal);
    key.append(options.optimize_aspect_ratio);
    key.append(options.optimize_principal_point);
    key.append(options.optimize_distortion);
    key.append(options.focal_prior_weight);
    key.append(options.min_focal_ratio);
    key.append(options.max_focal_ratio);
}

void append_resection(
    FingerprintBuilder& key, const ResectionConfig& options) {
    key.append(options.min_correspondences);
    key.append(options.min_inliers);
    key.append(options.max_local_window);
    key.append(options.local_ba_every);
    key.append(options.max_pose_wave);
    for (const unsigned value : options.full_ba_every) key.append(value);
    key.append(options.periodic_full_ba_probe_iterations);
    key.append(options.periodic_full_ba_tail_window);
    key.append(options.periodic_full_ba_tail_relative_improvement);
    key.append(options.final_ba_additional_iterations);
    key.append(options.final_ba_tail_window);
    key.append(options.final_ba_tail_relative_improvement);
    key.append(options.min_force_full_ba_samples);
    key.append(options.min_force_full_ba_interval);
    key.append(options.ratio_correspondences);
    key.append(options.avg_inliers_ratio_force_ba);
    key.append(options.min_inlier_ratio);
    key.append(options.inlier_grid_size);
    key.append(options.min_inlier_grid_cells);
    key.append(options.coverage_bypass_min_inliers);
    key.append(options.coverage_bypass_inlier_ratio);
    key.append(options.consistency_bypass_inlier_ratio);
    key.append(options.min_consistency_pair_weight);
    key.append(options.min_rotation_consistency_neighbors);
    key.append(options.max_median_rotation_error_deg);
    key.append(options.min_translation_consistency_neighbors);
    key.append(options.max_median_translation_error_deg);
    key.append(options.max_reproj_error);
    key.append(options.min_angle_deg);
    key.append(options.mult_depth_near);
    key.append(options.mult_depth_far);
    key.append(options.use_pair_match_correspondences);
    key.append(options.ransac.max_reproj_error_px);
    key.append(options.ransac.confidence);
    key.append(options.ransac.max_iterations);
    key.append(options.ransac.min_iterations);
    key.append(options.ransac.min_inliers);
    append_optimizer(key, options.local_ba);
    append_optimizer(key, options.full_ba);
}

std::uint64_t reconstruction_key(
    const std::uint64_t tracks_key,
    const ReconstructionConfig& config) {
    FingerprintBuilder key;
    key.append_string("reconstruction");
    // Bump when the single-cluster hierarchy path changes from star-only to
    // global-first mapping; old hierarchy checkpoints must not bypass it.
    key.append_string("scaled-weights-20k-v34");
    key.append_string(AETHERSCAN_RECONSTRUCTION_CACHE_BUILD_ID);
    key.append(static_cast<std::uint64_t>(__cplusplus));
#if defined(_MSC_VER)
    key.append(static_cast<std::uint32_t>(_MSC_VER));
#endif
    key.append(tracks_key);
    key.append(static_cast<std::uint32_t>(config.mode));
    key.append(config.star.min_views);
    key.append(config.star.max_views);
    key.append(config.star.min_tracks_per_view);
    key.append(config.star.max_reproj_error);
    key.append(config.star.min_angle_deg);
    key.append(config.star.min_initial_tracks);
    append_resection(key, config.resection);
    key.append(config.hierarchical.cluster.max_views_per_cluster);
    key.append(config.hierarchical.cluster.min_views_per_cluster);
    key.append(config.hierarchical.cluster.max_over_capacity);
    key.append(config.hierarchical.cluster.min_common_tracks);
    key.append(config.hierarchical.cluster.min_pair_weight);
    key.append(config.hierarchical.cluster.refine_weak_edges);
    key.append(config.hierarchical.cluster.edge_weight_percentile);
    key.append(config.hierarchical.alignment.min_pair_weight);
    key.append(config.hierarchical.alignment.min_common_tracks);
    key.append(config.hierarchical.alignment.merge_track_inliers_only);
    key.append(config.hierarchical.alignment.ransac_relative_threshold);
    key.append(config.hierarchical.alignment.minimum_inlier_ratio);
    key.append(
        config.hierarchical.alignment.merge_proximity_relative_threshold);
    key.append(config.hierarchical.alignment.ransac_iterations);
    key.append(config.hierarchical.alignment.random_seed);
    key.append(config.hierarchical.final_bundle_adjustment);
    key.append(config.incremental_hierarchical_rescue);
    key.append(config.incremental_hierarchical_rescue_min_missing);
    key.append(config.incremental_hierarchical_rescue_min_missing_ratio);
    key.append(config.minimum_final_observations_per_image);
    key.append(config.maximum_final_reprojection_error_pixels);
    key.append(config.global_rotation.max_l1_iterations);
    key.append(config.global_rotation.max_irls_iterations);
    key.append(config.global_rotation.step_convergence_threshold);
    key.append(config.global_rotation.irls_sigma_deg);
    key.append(config.global_rotation.max_relative_rotation_error_deg);
    key.append(config.global_rotation.use_pair_weights);
    key.append(config.global_rotation.reject_planar_pairs);
    key.append(
        static_cast<std::uint32_t>(config.global_rotation.weight_type));
    key.append(config.global_positioning.min_views_per_track);
    key.append(config.global_positioning.min_tracks_for_positioning);
    key.append(config.global_positioning.tracks_per_registered_image);
    key.append(config.global_positioning.max_tracks_for_positioning);
    key.append(config.global_positioning.coverage_grid_size);
    key.append(config.global_positioning.max_irls_iterations);
    key.append(config.global_positioning.irls_inner_iterations);
    key.append(config.global_positioning.irls_tuning_constant);
    key.append(config.global_positioning.irls_min_weight);
    key.append(config.global_positioning.irls_quarantine_after);
    key.append(config.global_positioning.irls_weight_convergence);
    key.append(config.global_positioning.max_num_iterations);
    key.append(config.global_positioning.max_solver_time_sec);
    key.append(config.global_positioning.function_tolerance);
    key.append(config.global_positioning.huber_threshold);
    key.append(config.global_positioning.random_seed);
    key.append(config.global_positioning.generate_random_positions);
    key.append(config.global_positioning.generate_random_points);
    key.append(config.global_positioning.generate_scales);
    key.append(config.global_positioning.optimize_positions);
    key.append(config.global_positioning.optimize_points);
    key.append(config.global_positioning.optimize_scales);
    key.append(config.global_positioning.ray_initialize_points);
    key.append(config.global_positioning.camera_warm_start_first);
    key.append(config.global_positioning.camera_warm_start_min_images);
    key.append(static_cast<std::uint32_t>(
        config.global_positioning.constraint));
    key.append(config.global_positioning.constraint_reweight_scale);
    key.append(config.global_positioning.regularize_complete_orbits);
    return key.value();
}

unsigned repair_position_outliers(
    Scene& scene, std::vector<Index>& reseeded_images) {
    struct PositionEdge {
        Index first{};
        Index second{};
        double length{};
    };
    std::vector<PositionEdge> edges;
    std::vector<double> lengths;
    edges.reserve(scene.pairs.size());
    lengths.reserve(scene.pairs.size());
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || pair.id1 >= scene.images.size() ||
            pair.id2 >= scene.images.size())
            continue;
        const Image& first = scene.images[pair.id1];
        const Image& second = scene.images[pair.id2];
        if (!first.registered || !second.registered) continue;
        const double length = (first.pose.C - second.pose.C).norm();
        if (!(length > 1e-8) || !std::isfinite(length)) continue;
        edges.push_back({pair.id1, pair.id2, length});
        lengths.push_back(length);
    }
    if (lengths.size() < 3) return 0;

    const auto middle = lengths.begin() + lengths.size() / 2;
    std::nth_element(lengths.begin(), middle, lengths.end());
    const double median_length = *middle;
    if (!(median_length > 1e-8) || !std::isfinite(median_length)) return 0;

    // A bearing-only solve may satisfy every reprojection constraint while a
    // weakly constrained camera group drifts to a different scale. Remove
    // those scale-breaking links, then keep the largest position-rigid
    // component for BA and let robust resection recover the excluded views.
    const double maximum_length = 8.0 * median_length;
    std::vector<std::vector<Index>> adjacency(scene.images.size());
    for (const PositionEdge& edge : edges) {
        if (edge.length > maximum_length) continue;
        adjacency[edge.first].push_back(edge.second);
        adjacency[edge.second].push_back(edge.first);
    }

    std::vector<std::uint8_t> visited(scene.images.size(), 0);
    std::vector<Index> largest;
    for (Index seed = 0; seed < scene.images.size(); ++seed) {
        if (visited[seed] || !scene.images[seed].registered) continue;
        std::vector<Index> component;
        std::queue<Index> pending;
        pending.push(seed);
        visited[seed] = 1;
        while (!pending.empty()) {
            const Index image_id = pending.front();
            pending.pop();
            component.push_back(image_id);
            for (const Index neighbor : adjacency[image_id]) {
                if (visited[neighbor]) continue;
                visited[neighbor] = 1;
                pending.push(neighbor);
            }
        }
        if (component.size() > largest.size())
            largest = std::move(component);
    }

    const unsigned registered = scene.registered_count();
    if (largest.size() < 3 || largest.size() * 2 < registered) return 0;
    std::vector<std::uint8_t> keep(scene.images.size(), 0);
    for (const Index image_id : largest) keep[image_id] = 1;
    std::vector<std::uint8_t> reachable = keep;
    std::vector<std::vector<Index>> recovery_adjacency(scene.images.size());
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || !pair.relative_pose.has_value() ||
            pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size() ||
            !scene.images[pair.id1].registered ||
            !scene.images[pair.id2].registered)
            continue;
        recovery_adjacency[pair.id1].push_back(pair.id2);
        recovery_adjacency[pair.id2].push_back(pair.id1);
    }
    std::queue<Index> recovery_queue;
    for (const Index image_id : largest)
        recovery_queue.push(image_id);
    while (!recovery_queue.empty()) {
        const Index image_id = recovery_queue.front();
        recovery_queue.pop();
        for (const Index neighbor : recovery_adjacency[image_id]) {
            if (reachable[neighbor]) continue;
            reachable[neighbor] = 1;
            recovery_queue.push(neighbor);
        }
    }

    std::vector<Index> recoverable;
    unsigned quarantined = 0;
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        Image& image = scene.images[image_id];
        if (!image.registered || keep[image_id]) continue;
        if (reachable[image_id])
            recoverable.push_back(image_id);
        else {
            image.registered = false;
            ++quarantined;
        }
    }
    unsigned reseeded = 0;
    if (!recoverable.empty()) {
        std::unordered_map<Index, Eigen::Index> columns;
        columns.reserve(recoverable.size());
        for (Eigen::Index column = 0;
             column < static_cast<Eigen::Index>(recoverable.size());
             ++column)
            columns.emplace(recoverable[static_cast<std::size_t>(column)],
                            column);

        struct DirectionConstraint {
            Index first{};
            Index second{};
            Vec3 offset{Vec3::Zero()};
            double weight{1.0};
        };
        std::vector<DirectionConstraint> constraints;
        constraints.reserve(scene.pairs.size());
        for (const ImagePair& pair : scene.pairs) {
            if (!pair.active || !pair.relative_pose.has_value() ||
                pair.id1 >= scene.images.size() ||
                pair.id2 >= scene.images.size() ||
                !scene.images[pair.id1].registered ||
                !scene.images[pair.id2].registered)
                continue;
            if (!columns.count(pair.id1) && !columns.count(pair.id2))
                continue;
            Vec3 direction =
                scene.images[pair.id1].pose.R.transpose() *
                pair.relative_pose->C;
            const double norm = direction.norm();
            if (!(norm > 1e-8) || !direction.allFinite()) continue;
            direction /= norm;
            constraints.push_back({
                pair.id1, pair.id2, median_length * direction,
                std::sqrt(std::max(
                    0.1, static_cast<double>(pair.composite_weight())))});
        }

        if (constraints.size() >= recoverable.size()) {
            Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(
                static_cast<Eigen::Index>(constraints.size()),
                static_cast<Eigen::Index>(recoverable.size()));
            Eigen::MatrixXd right_hand_side = Eigen::MatrixXd::Zero(
                static_cast<Eigen::Index>(constraints.size()), 3);
            for (Eigen::Index row = 0;
                 row < static_cast<Eigen::Index>(constraints.size()); ++row) {
                const DirectionConstraint& constraint =
                    constraints[static_cast<std::size_t>(row)];
                Vec3 target = constraint.offset;
                if (const auto found = columns.find(constraint.second);
                    found != columns.end())
                    coefficients(row, found->second) += constraint.weight;
                else
                    target -= scene.images[constraint.second].pose.C;
                if (const auto found = columns.find(constraint.first);
                    found != columns.end())
                    coefficients(row, found->second) -= constraint.weight;
                else
                    target += scene.images[constraint.first].pose.C;
                right_hand_side.row(row) =
                    (constraint.weight * target).transpose();
            }
            const Eigen::MatrixXd solution =
                coefficients.colPivHouseholderQr().solve(right_hand_side);
            for (Eigen::Index column = 0;
                 column < solution.rows(); ++column) {
                const Index image_id =
                    recoverable[static_cast<std::size_t>(column)];
                const Vec3 center = solution.row(column).transpose();
                if (!center.allFinite()) {
                    scene.images[image_id].registered = false;
                    ++quarantined;
                    continue;
                }
                scene.images[image_id].pose.C = center;
                // Keep the repaired pose aside while the rigid component is
                // refined. It is restored against that stable map below.
                scene.images[image_id].registered = false;
                reseeded_images.push_back(image_id);
                ++reseeded;
            }
        } else {
            for (const Index image_id : recoverable) {
                scene.images[image_id].registered = false;
                ++quarantined;
            }
        }
    }
    if (quarantined == 0 && reseeded == 0) return 0;

    // Point positions estimated together with a drifting component are not a
    // safe BA seed. Re-triangulate all structure from the retained cameras.
    for (Track& track : scene.tracks) {
        track.position = Vec3::Zero();
        track.num_inliers = 0;
    }
    core::Logger::instance().warning(
        "global position graph: reseeded=", reseeded,
        " quarantined=", quarantined,
        " retained=", largest.size(),
        " median_edge=", median_length,
        " maximum_edge=", maximum_length);
    return quarantined + reseeded;
}

// Global layout is computed with one robust shared-intrinsic group so a small
// number of zoom-transition pairs cannot be rejected before a connected graph
// exists. Once poses are stable, EXIF-identified zoom/lens states become
// separate BA groups: they share the proven layout as an initialization and
// then adapt focal/aspect/distortion independently.
unsigned split_intrinsics_by_camera_identity(Scene& scene) {
    if (scene.cameras.empty() || scene.images.empty()) return 0;

    std::vector<double> focal_lengths;
    focal_lengths.reserve(scene.images.size());
    for (const Image& image : scene.images)
        if (image.focal_length_mm > 0.0)
            focal_lengths.push_back(image.focal_length_mm);
    const double median_focal_length =
        focal_lengths.size() >= 4 ? median_value(focal_lengths) : 0.0;

    std::unordered_map<std::string, Index> group_of_identity;
    std::vector<PinholeCamera> grouped_cameras;
    grouped_cameras.reserve(scene.cameras.size());
    std::vector<unsigned> group_images;

    for (Image& image : scene.images) {
        const std::string& identity = image.camera_identity;
        auto [iterator, inserted] = group_of_identity.emplace(
            identity, static_cast<Index>(grouped_cameras.size()));
        if (inserted) {
            PinholeCamera camera = scene.cameras[image.camera_id];
            if (median_focal_length > 0.0 && image.focal_length_mm > 0.0) {
                const double focal =
                    camera.fx * image.focal_length_mm / median_focal_length;
                if (std::isfinite(focal) && focal > 0.0) {
                    const double ratio = focal / camera.fx;
                    camera.fx *= ratio;
                    camera.fy *= ratio;
                    camera.focal_prior = 0.5 * (camera.fx + camera.fy);
                }
            }
            camera.id = static_cast<Index>(grouped_cameras.size());
            grouped_cameras.push_back(camera);
            group_images.push_back(0);
        }
        image.camera_id = iterator->second;
        ++group_images[iterator->second];
    }

    if (grouped_cameras.size() <= 1) return 0;
    scene.cameras = std::move(grouped_cameras);
    std::string groups;
    for (std::size_t i = 0; i < scene.cameras.size(); ++i) {
        if (!groups.empty()) groups += ',';
        groups += std::to_string(group_images[i]);
    }
    core::Logger::instance().info(
        "global: split EXIF intrinsic groups=", scene.cameras.size(),
        " images_per_group=", groups,
        " median_focal_length_mm=", median_focal_length);
    return static_cast<unsigned>(scene.cameras.size());
}

unsigned enforce_zero_baseline_pose_groups(Scene& scene) {
    if (scene.images.empty() || scene.pairs.empty()) return 0;

    std::vector<Index> parent(scene.images.size());
    std::iota(parent.begin(), parent.end(), Index{0});
    const auto find = [&](const auto self, const Index node) -> Index {
        if (parent[node] == node) return node;
        return parent[node] = self(self, parent[node]);
    };
    const auto unite = [&](Index first, Index second) {
        first = find(find, first);
        second = find(find, second);
        if (first == second) return;
        if (first < second) parent[second] = first;
        else parent[first] = second;
    };

    bool has_zero_baseline = false;
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.zero_baseline || pair.id1 >= parent.size() ||
            pair.id2 >= parent.size() || pair.id1 == pair.id2)
            continue;
        unite(pair.id1, pair.id2);
        has_zero_baseline = true;
    }
    if (!has_zero_baseline) return 0;

    std::unordered_map<Index, std::vector<Index>> groups;
    for (Index image_id = 0; image_id < parent.size(); ++image_id) {
        const Index root = find(find, image_id);
        groups[root].push_back(image_id);
    }

    std::vector<double> reprojection_rms(scene.images.size(),
                                         std::numeric_limits<double>::max());
    std::vector<unsigned> reprojection_count(scene.images.size(), 0);
    std::vector<double> squared_errors(scene.images.size(), 0.0);
    for (const Track& track : scene.tracks) {
        if (!track.is_triangulated() || !track.position.allFinite()) continue;
        const std::size_t inlier_count = std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        for (std::size_t i = 0; i < inlier_count; ++i) {
            const Observation& observation = track.observations[i];
            if (observation.image_id >= scene.images.size()) continue;
            const Image& image = scene.images[observation.image_id];
            if (!image.registered || image.camera_id >= scene.cameras.size() ||
                observation.feature_id >= image.features.keypoints.size())
                continue;
            const Vec3 camera_point =
                image.pose.transform_world_to_camera(track.position);
            if (!camera_point.allFinite() || camera_point.z() <= 0.0) continue;
            const Vec2 projected =
                scene.cameras[image.camera_id].project(camera_point);
            const auto& keypoint =
                image.features.keypoints[observation.feature_id];
            const double error =
                (projected - Vec2(keypoint.x, keypoint.y)).norm();
            if (!std::isfinite(error)) continue;
            squared_errors[observation.image_id] += error * error;
            ++reprojection_count[observation.image_id];
        }
    }
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        if (reprojection_count[image_id] >= 10) {
            reprojection_rms[image_id] =
                std::sqrt(squared_errors[image_id] /
                          static_cast<double>(reprojection_count[image_id]));
        }
    }

    unsigned updated = 0;
    unsigned group_count = 0;
    for (auto& [root, image_ids] : groups) {
        Index representative = k_invalid;
        double best_score = std::numeric_limits<double>::infinity();
        for (const Index image_id : image_ids) {
            const Image& image = scene.images[image_id];
            if (!image.registered || !image.pose.R.allFinite() ||
                !image.pose.C.allFinite())
                continue;
            // Prefer the copy with the strongest low-error support. Views with
            // too few observations only win when no supported copy exists.
            const double score = reprojection_rms[image_id];
            if (score < best_score ||
                (score == best_score && image_id < representative)) {
                representative = image_id;
                best_score = score;
            }
        }
        if (representative == k_invalid) continue;
        ++group_count;
        const Pose3D fused = scene.images[representative].pose;
        for (const Index image_id : image_ids) {
            Image& image = scene.images[image_id];
            if (image.registered &&
                image.pose.R.isApprox(fused.R) &&
                image.pose.C.isApprox(fused.C))
                continue;
            image.registered = true;
            image.pose = fused;
            ++updated;
        }
    }
    core::Logger::instance().info(
        "zero-baseline pose groups: groups=", group_count,
        " cloned_views=", updated);
    return updated;
}

void populate_reprojection_stats(
    const Scene& scene, ReconstructionSummary& summary) {
    double error_sum = 0.0;
    double squared_error_sum = 0.0;
    std::uint64_t observation_count = 0;
    for (const Track& track : scene.tracks) {
        if (!track.is_triangulated() || !track.position.allFinite()) continue;
        const std::size_t inlier_count = std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        for (std::size_t i = 0; i < inlier_count; ++i) {
            const Observation& observation = track.observations[i];
            if (observation.image_id >= scene.images.size()) continue;
            const Image& image = scene.images[observation.image_id];
            if (!image.registered || image.camera_id >= scene.cameras.size() ||
                observation.feature_id >= image.features.keypoints.size())
                continue;
            const Vec3 camera_point =
                image.pose.transform_world_to_camera(track.position);
            if (!camera_point.allFinite() || camera_point.z() <= 0.0) continue;
            const Vec2 projected =
                scene.cameras[image.camera_id].project(camera_point);
            const auto& keypoint =
                image.features.keypoints[observation.feature_id];
            const Vec2 measured(
                static_cast<double>(keypoint.x),
                static_cast<double>(keypoint.y));
            const double error = (projected - measured).norm();
            if (!std::isfinite(error)) continue;
            error_sum += error;
            squared_error_sum += error * error;
            ++observation_count;
        }
    }
    summary.reprojection_observations = observation_count;
    if (observation_count == 0) return;
    const double inverse_count =
        1.0 / static_cast<double>(observation_count);
    summary.mean_reprojection_error_pixels = error_sum * inverse_count;
    summary.rms_reprojection_error_pixels =
        std::sqrt(squared_error_sum * inverse_count);
}

ReconstructionSummary summarize_scene(const Scene& scene) {
    ReconstructionSummary summary;
    summary.registered_views = scene.registered_count();
    summary.failed_views =
        static_cast<unsigned>(scene.images.size()) - summary.registered_views;
    const AlignmentObservability observability =
        analyze_alignment_observability(scene);
    summary.alignment_reliable_views = observability.reliable_views;
    summary.alignment_unreliable_views = observability.unreliable_views;
    for (const Track& track : scene.tracks)
        if (track.is_triangulated()) ++summary.landmarks;
    summary.valid =
        summary.registered_views >= 2 && summary.landmarks > 0;
    populate_reprojection_stats(scene, summary);
    return summary;
}

}  // namespace

AlignmentObservability analyze_alignment_observability(const Scene& scene) {
    AlignmentObservability result;
    result.reliable.assign(scene.images.size(), 0);
    if (scene.images.empty()) return result;

    struct Edge { Index first{}; Index second{}; };
    struct Adjacent { Index neighbor{}; std::size_t edge{}; };
    std::vector<Edge> edges;
    std::vector<std::vector<Adjacent>> adjacency(scene.images.size());
    std::unordered_set<std::uint64_t> edge_keys;
    edge_keys.reserve(scene.pairs.size() * 2);
    const auto add_edge = [&](Index first, Index second) {
        if (first == second || first >= scene.images.size() ||
            second >= scene.images.size() ||
            !scene.images[first].registered ||
            !scene.images[second].registered)
            return;
        if (first > second) std::swap(first, second);
        const std::uint64_t key =
            (static_cast<std::uint64_t>(first) << 32U) | second;
        if (!edge_keys.insert(key).second) return;
        const std::size_t edge_id = edges.size();
        edges.push_back({first, second});
        adjacency[first].push_back({second, edge_id});
        adjacency[second].push_back({first, edge_id});
    };
    for (const ImagePair& pair : scene.pairs) {
        if (pair.active && !pair.zero_baseline &&
            pair.relative_pose.has_value())
            add_edge(pair.id1, pair.id2);
    }
    const Index invalid = k_invalid;
    std::vector<Index> discovery(scene.images.size(), invalid);
    std::vector<Index> low(scene.images.size(), invalid);
    std::vector<Index> parent(scene.images.size(), invalid);
    std::vector<std::size_t> parent_edge(
        scene.images.size(), std::numeric_limits<std::size_t>::max());
    std::vector<std::uint8_t> bridge(edges.size(), 0);
    Index timestamp = 0;
    struct Frame { Index node{}; std::size_t next{}; };
    for (Index root = 0; root < scene.images.size(); ++root) {
        if (!scene.images[root].registered || discovery[root] != invalid)
            continue;
        discovery[root] = low[root] = timestamp++;
        std::vector<Frame> stack{{root, 0}};
        while (!stack.empty()) {
            Frame& frame = stack.back();
            if (frame.next < adjacency[frame.node].size()) {
                const Adjacent adjacent = adjacency[frame.node][frame.next++];
                if (adjacent.edge == parent_edge[frame.node]) continue;
                if (discovery[adjacent.neighbor] == invalid) {
                    parent[adjacent.neighbor] = frame.node;
                    parent_edge[adjacent.neighbor] = adjacent.edge;
                    discovery[adjacent.neighbor] =
                        low[adjacent.neighbor] = timestamp++;
                    stack.push_back({adjacent.neighbor, 0});
                } else {
                    low[frame.node] = std::min(
                        low[frame.node], discovery[adjacent.neighbor]);
                }
                continue;
            }
            const Index node = frame.node;
            stack.pop_back();
            if (parent[node] == invalid) continue;
            const Index ancestor = parent[node];
            low[ancestor] = std::min(low[ancestor], low[node]);
            if (low[node] > discovery[ancestor]) {
                bridge[parent_edge[node]] = 1;
            }
        }
    }

    std::vector<std::uint8_t> visited(scene.images.size(), 0);
    std::vector<Index> camera_block(scene.images.size(), invalid);
    std::vector<std::vector<Index>> blocks;
    for (Index root = 0; root < scene.images.size(); ++root) {
        if (!scene.images[root].registered || visited[root]) continue;
        const Index block_id = static_cast<Index>(blocks.size());
        blocks.emplace_back();
        std::queue<Index> pending;
        pending.push(root);
        visited[root] = 1;
        while (!pending.empty()) {
            const Index node = pending.front();
            pending.pop();
            camera_block[node] = block_id;
            blocks.back().push_back(node);
            for (const Adjacent adjacent : adjacency[node]) {
                if (bridge[adjacent.edge] || visited[adjacent.neighbor]) continue;
                visited[adjacent.neighbor] = 1;
                pending.push(adjacent.neighbor);
            }
        }
    }

    struct PointScatter {
        unsigned count{};
        Vec3 mean{Vec3::Zero()};
        Eigen::Matrix3d scatter{Eigen::Matrix3d::Zero()};

        void add(const Vec3& point) {
            ++count;
            const Vec3 delta = point - mean;
            mean += delta / static_cast<double>(count);
            scatter += delta * (point - mean).transpose();
        }

        [[nodiscard]] bool constrains_similarity() const {
            if (count < 3 || !scatter.allFinite()) return false;
            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(scatter);
            if (solver.info() != Eigen::Success) return false;
            const auto eigenvalues = solver.eigenvalues();
            const double scale = eigenvalues.cwiseAbs().maxCoeff();
            if (!(scale > 0.0)) return false;
            return eigenvalues[1] > 1e-4 * scale;
        }
    };
    std::unordered_map<std::uint64_t, PointScatter> shared_points;
    std::vector<Index> track_blocks;
    std::vector<std::pair<Index, Index>> block_observations;
    for (const Track& track : scene.tracks) {
        if (!track.is_triangulated() || !track.position.allFinite()) continue;
        track_blocks.clear();
        block_observations.clear();
        const std::size_t count = std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        for (std::size_t i = 0; i < count; ++i) {
            const Index image_id = track.observations[i].image_id;
            if (image_id < camera_block.size() &&
                camera_block[image_id] != invalid)
                block_observations.emplace_back(camera_block[image_id], image_id);
        }
        std::sort(block_observations.begin(), block_observations.end());
        block_observations.erase(
            std::unique(block_observations.begin(), block_observations.end()),
            block_observations.end());
        for (std::size_t begin = 0; begin < block_observations.size();) {
            std::size_t end = begin + 1;
            while (end < block_observations.size() &&
                   block_observations[end].first == block_observations[begin].first)
                ++end;
            const Vec3 first_ray = track.position -
                scene.images[block_observations[begin].second].pose.C;
            bool independent_depth = false;
            for (std::size_t observation = begin + 1; observation < end; ++observation) {
                const Vec3 ray = track.position -
                    scene.images[block_observations[observation].second].pose.C;
                const double scale = first_ray.squaredNorm() * ray.squaredNorm();
                if (std::isfinite(scale) && scale > 0.0 &&
                    // Match the one-degree depth threshold used by stable
                    // triangulation. Numerical non-parallelism alone does not
                    // make a weak branch's scale observable.
                    first_ray.cross(ray).squaredNorm() >
                        0.000304586490452135 * scale) {
                    independent_depth = true;
                    break;
                }
            }
            if (independent_depth)
                track_blocks.push_back(block_observations[begin].first);
            begin = end;
        }
        for (std::size_t i = 0; i < track_blocks.size(); ++i) {
            for (std::size_t j = i + 1; j < track_blocks.size(); ++j) {
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(track_blocks[i]) << 32U) |
                    track_blocks[j];
                shared_points[key].add(track.position);
            }
        }
    }

    std::vector<Index> block_parent(blocks.size());
    std::iota(block_parent.begin(), block_parent.end(), Index{0});
    const auto find_block = [&](const auto self, const Index node) -> Index {
        if (block_parent[node] == node) return node;
        return block_parent[node] = self(self, block_parent[node]);
    };
    unsigned similarity_constraints = 0;
    for (const auto& [key, points] : shared_points) {
        if (!points.constrains_similarity()) continue;
        const Index first = static_cast<Index>(key >> 32U);
        const Index second = static_cast<Index>(key & 0xffffffffU);
        const Index first_root = find_block(find_block, first);
        const Index second_root = find_block(find_block, second);
        if (first_root != second_root) block_parent[second_root] = first_root;
        ++similarity_constraints;
    }

    std::unordered_map<Index, unsigned> group_sizes;
    Index largest_group = invalid;
    unsigned largest_size = 0;
    for (Index block_id = 0; block_id < blocks.size(); ++block_id) {
        const Index group = find_block(find_block, block_id);
        const unsigned size =
            (group_sizes[group] += static_cast<unsigned>(blocks[block_id].size()));
        if (size > largest_size) {
            largest_size = size;
            largest_group = group;
        }
    }
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        if (camera_block[image_id] != invalid &&
            find_block(find_block, camera_block[image_id]) == largest_group)
            result.reliable[image_id] = 1;
    }
    for (std::size_t edge_id = 0; edge_id < edges.size(); ++edge_id) {
        if (!bridge[edge_id]) continue;
        const Edge& edge = edges[edge_id];
        if (find_block(find_block, camera_block[edge.first]) !=
            find_block(find_block, camera_block[edge.second]))
            ++result.bridge_edges;
    }
    result.constraint_edges =
        static_cast<unsigned>(edges.size()) + similarity_constraints;
    result.reliable_views = largest_size;
    // A resected singleton need not have independent local depth. Certify it
    // only from depths recomputed exclusively in the frozen reliable map,
    // never from a jointly triangulated parent point or newly certified view.
    const auto stable_mask = result.reliable;
    std::vector<PointScatter> resection_support(scene.images.size());
    std::vector<std::unordered_set<Index>> supported_features(scene.images.size());
    std::vector<std::unordered_set<unsigned>> supported_cells(scene.images.size());
    TriangulationOptions tri;
    tri.reproj_threshold_px = 2.F;
    tri.min_angle_deg = 1.F;
    tri.split_tracks = false;
    if (largest_size < scene.registered_count()) for (const auto& source : scene.tracks) {
        Track anchor;
        std::vector<Observation> targets;
        std::unordered_set<Index> stable_images;
        for (std::size_t o = 0; o < std::min<std::size_t>(source.num_inliers, source.observations.size()); ++o) {
            const auto& obs = source.observations[o];
            if (obs.image_id >= scene.images.size() || !scene.images[obs.image_id].registered) continue;
            if (stable_mask[obs.image_id]) {
                if (stable_images.insert(obs.image_id).second) anchor.observations.push_back(obs);
            } else targets.push_back(obs);
        }
        if (targets.empty() || anchor.observations.size() < 2 || triangulate_track(anchor, scene, tri) < 2) continue;
        for (const auto& obs : targets) {
            const auto& image = scene.images[obs.image_id];
            if (image.camera_id >= scene.cameras.size() || obs.feature_id >= image.features.keypoints.size()) continue;
            const auto& camera = scene.camera_of(image);
            const auto& measured = image.features.keypoints[obs.feature_id];
            Vec2 pixel;
            if (!camera.project_checked(image.pose.transform_world_to_camera(anchor.position), pixel) ||
                (pixel - Vec2(measured.x, measured.y)).norm() > 2.0 ||
                !supported_features[obs.image_id].insert(obs.feature_id).second) continue;
            resection_support[obs.image_id].add(anchor.position);
            const auto x = static_cast<unsigned>(std::clamp(4.F * measured.x / std::max(1u, camera.width), 0.F, 3.F));
            const auto y = static_cast<unsigned>(std::clamp(4.F * measured.y / std::max(1u, camera.height), 0.F, 3.F));
            supported_cells[obs.image_id].insert(y * 4 + x);
        }
    }
    for (Index id = 0; id < scene.images.size(); ++id) {
        if (!result.reliable[id] && resection_support[id].count >= 40 &&
            supported_cells[id].size() >= 3 && resection_support[id].constrains_similarity()) {
            result.reliable[id] = 1;
            ++result.reliable_views;
        }
    }
    result.unreliable_views = scene.registered_count() - result.reliable_views;
    return result;
}

unsigned prune_unsupported_registrations(
    Scene& scene, const unsigned minimum_observations,
    const double maximum_reprojection_error_pixels) {
    if (minimum_observations == 0) return 0;
    const double threshold =
        std::max(maximum_reprojection_error_pixels, 0.0);
    std::vector<unsigned> support(scene.images.size(), 0);
    for (const Track& track : scene.tracks) {
        if (!track.is_triangulated() || !track.position.allFinite()) continue;
        const std::size_t inlier_count = std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        for (std::size_t index = 0; index < inlier_count; ++index) {
            const Observation& observation = track.observations[index];
            if (observation.image_id >= scene.images.size()) continue;
            const Image& image = scene.images[observation.image_id];
            if (!image.registered || image.camera_id >= scene.cameras.size() ||
                observation.feature_id >= image.features.keypoints.size())
                continue;
            const Vec3 camera_point =
                image.pose.transform_world_to_camera(track.position);
            if (!camera_point.allFinite() || camera_point.z() <= 0.0) continue;
            const Vec2 projected =
                scene.cameras[image.camera_id].project(camera_point);
            const auto& feature = image.features.keypoints[observation.feature_id];
            const double error =
                (projected - Vec2(feature.x, feature.y)).norm();
            if (std::isfinite(error) && error <= threshold)
                ++support[observation.image_id];
        }
    }

    std::vector<std::uint8_t> removed(scene.images.size(), 0);
    unsigned invalidated = 0;
    for (std::size_t image_id = 0; image_id < scene.images.size(); ++image_id) {
        Image& image = scene.images[image_id];
        if (!image.registered || support[image_id] >= minimum_observations)
            continue;
        removed[image_id] = 1;
        image.registered = false;
        image.pose = Pose3D::identity();
        ++invalidated;
        core::Logger::instance().warning(
            "final registration audit: invalidated image=", image.path.filename(),
            " support=", support[image_id],
            " required=", minimum_observations);
    }
    if (invalidated == 0) return 0;

    for (Track& track : scene.tracks) {
        const std::size_t old_inliers = std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        unsigned retained_inliers = 0;
        for (std::size_t index = 0; index < old_inliers; ++index)
            if (track.observations[index].image_id < removed.size() &&
                !removed[track.observations[index].image_id])
                ++retained_inliers;
        std::erase_if(track.observations, [&](const Observation& observation) {
            return observation.image_id < removed.size() &&
                   removed[observation.image_id];
        });
        track.num_inliers = static_cast<std::uint8_t>(
            std::min<std::size_t>(retained_inliers, 255));
        if (track.num_inliers < 2) track.num_inliers = 0;
    }
    rebuild_track_index(scene);
    core::Logger::instance().warning(
        "final registration audit: invalidated=", invalidated,
        " remaining=", scene.registered_count());
    return invalidated;
}

ReconstructionSummary run_incremental_mapping(
    Scene& scene,
    const StarInitConfig& star,
    const ResectionConfig& resection) {
    core::StageScope stage("sfm.incremental_mapping");
    ReconstructionSummary summary;
    if (scene.registered_count() < 2) {
        if (!star_initialize(scene, star)) return summary;
        if (resection.checkpoint_callback)
            resection.checkpoint_callback(scene);
    }
    register_images(scene, resection);

    return summarize_scene(scene);
}

ReconstructionSummary run_global_mapping(
    Scene& scene,
    const GlobalRotationOptions& rotation,
    const GlobalPositioningOptions& positioning,
    const ResectionConfig& fallback_resection) {
    core::StageScope stage("sfm.global_mapping");
    ReconstructionSummary summary;

    GlobalRotationSummary rotation_summary =
        estimate_global_rotations(scene, rotation);
    if (!rotation_summary.success) {
        core::Logger::instance().error("global: rotation averaging pass 1 failed");
        return summary;
    }
    const unsigned total_filtered_pairs = rotation_summary.filtered_pairs;
    if (total_filtered_pairs > 0) {
        rotation_summary = estimate_global_rotations(scene, rotation);
        if (!rotation_summary.success) {
            core::Logger::instance().error("global: rotation averaging pass 2 failed");
            return summary;
        }
    }
    rotation_summary.filtered_pairs = total_filtered_pairs;
    // Rotation filtering changes triangle support. Refresh graph-local weights
    // before rebuilding tracks so rejected edges cannot dominate unions.
    compute_pair_weights(scene);
    // openMVS rebuilds tracks after relative-rotation filtering with
    // ReconstructionConfig::minPairWeight (default 3).
    build_tracks(scene, 3.F);
    core::Logger::instance().info(
        "global: rotation_images=", rotation_summary.estimated_images,
        " used_pairs=", rotation_summary.used_pairs,
        " filtered_pairs=", rotation_summary.filtered_pairs,
        " tracks=", scene.tracks.size());

    const GlobalPositioningSummary position_summary =
        solve_global_positions(scene, positioning);
    if (!position_summary.success) {
        core::Logger::instance().error(
            "global: positioning failed after ", position_summary.iterations,
            " iterations, observations=", position_summary.observations);
        return summary;
    }

    std::vector<Index> reseeded_images;
    const unsigned repaired_positions =
        repair_position_outliers(scene, reseeded_images);

    // Densify structure for BA: camera-only needs a full triangulation; a capped
    // only_points solve only marks a subset, so triangulate the remaining tracks.
    if (position_summary.positioned_tracks == 0 ||
        repaired_positions > 0) {
        triangulate_tracks(scene, false, 6.F, 1.F);
        core::Logger::instance().info(
            repaired_positions > 0
                ? "global: triangulated after position-graph repair"
                : "global: triangulated after camera-only positioning");
    } else {
        triangulate_tracks(scene, true, 6.F, 1.F);
        core::Logger::instance().info(
            "global: densified tracks after only_points (",
            position_summary.positioned_tracks, " positioned)");
    }

    filter_tracks(scene, 6.F, 1.F, 0.F, 0.F);

    if (fallback_resection.checkpoint_callback)
        fallback_resection.checkpoint_callback(scene);

    // Stage 1: structure/translation only. Opening focal this early lets a
    // weak global layout absorb into intrinsics (cloth3: 900 -> ~810).
    BundleOptions bundle;
    bundle.optimizer.maximum_iterations = 12;
    bundle.optimizer.huber_delta = 2.0;
    bundle.optimizer.optimize_rotations = false;
    bundle.optimizer.optimize_focal = false;
    bundle.optimizer.optimize_distortion = false;
    split_intrinsics_by_camera_identity(scene);
    if (!run_bundle_adjustment(scene, bundle).success) {
        core::Logger::instance().error(
            "global: position/structure bundle adjustment failed");
        return summary;
    }
    filter_tracks(
        scene,
        fallback_resection.max_reproj_error,
        fallback_resection.min_angle_deg,
        fallback_resection.mult_depth_near,
        fallback_resection.mult_depth_far);

    // Stage 2: free rotations + focal with prior/bounds; keep distortion fixed.
    bundle.optimizer.maximum_iterations = 25;
    bundle.optimizer.optimize_rotations = true;
    bundle.optimizer.optimize_focal = true;
    bundle.optimizer.optimize_aspect_ratio = true;
    bundle.optimizer.optimize_distortion = false;
    if (!run_bundle_adjustment(scene, bundle).success) {
        core::Logger::instance().error("global: full bundle adjustment failed");
        return summary;
    }
    bool preserve_orbit_centers = false;
    if (fallback_resection.checkpoint_callback)
        fallback_resection.checkpoint_callback(scene);
    // A temporarily disconnected camera is recovered below. Delay the orbit
    // probe until that camera has an anchored pose; otherwise the strict
    // complete-sequence check would silently skip hierarchical single-cluster
    // reconstructions.
    bool orbit_regularized = false;
    if (reseeded_images.empty()) {
        orbit_regularized = regularize_complete_orbit(
            scene, positioning, preserve_orbit_centers);
    }
    triangulate_tracks(
        scene,
        !orbit_regularized,
        fallback_resection.max_reproj_error,
        fallback_resection.min_angle_deg);
    filter_tracks(
        scene,
        fallback_resection.max_reproj_error,
        fallback_resection.min_angle_deg,
        fallback_resection.mult_depth_near,
        fallback_resection.mult_depth_far);

    // Stage 3: open distortion once geometry is stable. Eight iterations are
    // a probe, not proof of convergence: large scenes can still be correcting
    // local drift when this budget expires. Continue while the objective's
    // tail is improving, retaining a bounded total budget.
    bundle.optimizer.maximum_iterations = 8;
    bundle.optimizer.optimize_rotations = true;
    bundle.optimizer.optimize_translations = !preserve_orbit_centers;
    bundle.optimizer.optimize_focal = true;
    bundle.optimizer.optimize_aspect_ratio = true;
    bundle.optimizer.optimize_distortion = true;
    auto polish = run_bundle_adjustment(scene, bundle);
    if (!polish.success) {
        core::Logger::instance().warning(
            "global: final bundle polish failed; keeping previous solution");
    }
    for (unsigned continuation = 0; continuation < 3 && polish.success;
         ++continuation) {
        const auto& optimizer = polish.optimizer;
        if (optimizer.termination != ba::TerminationReason::maximum_iterations ||
            optimizer.iterations.size() < 3) break;
        const double earlier = optimizer.iterations[optimizer.iterations.size()-3].cost;
        const double improvement = (earlier-optimizer.final_cost)/std::max(earlier, 1e-12);
        if (!std::isfinite(improvement) || improvement < 0.001) break;
        core::Logger::instance().info(
            "global: continuing final BA pass=", continuation+1,
            " tail_relative_improvement=", improvement);
        bundle.optimizer.maximum_iterations = 16;
        polish = run_bundle_adjustment(scene, bundle);
        if (!polish.success)
            core::Logger::instance().warning(
                "global: final BA continuation failed; retaining last usable solution");
    }
    const float fine_reproj_error =
        std::min(fallback_resection.max_reproj_error, 2.F);
    filter_tracks(
        scene,
        fine_reproj_error,
        fallback_resection.min_angle_deg,
        fallback_resection.mult_depth_near,
        fallback_resection.mult_depth_far);

    if (!reseeded_images.empty()) {
        std::vector<std::uint8_t> is_reseeded(scene.images.size(), 0);
        for (const Index image_id : reseeded_images) {
            scene.images[image_id].registered = true;
            is_reseeded[image_id] = 1;
        }
        triangulate_tracks(
            scene, false, fallback_resection.max_reproj_error,
            fallback_resection.min_angle_deg);
        filter_tracks(
            scene, fallback_resection.max_reproj_error,
            fallback_resection.min_angle_deg,
            fallback_resection.mult_depth_near,
            fallback_resection.mult_depth_far);

        BundleOptions recovery_bundle;
        recovery_bundle.optimizer = fallback_resection.full_ba;
        recovery_bundle.optimizer.maximum_iterations = 40;
        recovery_bundle.optimizer.optimize_focal = false;
        recovery_bundle.optimizer.optimize_aspect_ratio = false;
        recovery_bundle.optimizer.optimize_distortion = false;
        recovery_bundle.optimize_all_registered = false;
        recovery_bundle.free_image_ids = reseeded_images;
        for (Index image_id = 0; image_id < scene.images.size(); ++image_id)
            if (scene.images[image_id].registered &&
                !is_reseeded[image_id])
                recovery_bundle.fixed_image_ids.push_back(image_id);
        if (!run_bundle_adjustment(scene, recovery_bundle).success)
            core::Logger::instance().warning(
                "global: anchored component bundle adjustment failed");
        triangulate_tracks(
            scene, false, fallback_resection.max_reproj_error,
            fallback_resection.min_angle_deg);
        filter_tracks(
            scene, fine_reproj_error,
            fallback_resection.min_angle_deg,
            fallback_resection.mult_depth_near,
            fallback_resection.mult_depth_far);

        bool recovered_orbit_centers = false;
        const bool recovered_orbit_regularized = regularize_complete_orbit(
            scene, positioning, recovered_orbit_centers);
        if (recovered_orbit_regularized) {
            triangulate_tracks(
                scene, false, fallback_resection.max_reproj_error,
                fallback_resection.min_angle_deg);
            filter_tracks(
                scene, fallback_resection.max_reproj_error,
                fallback_resection.min_angle_deg,
                fallback_resection.mult_depth_near,
                fallback_resection.mult_depth_far);

            BundleOptions orbit_bundle;
            orbit_bundle.optimizer = bundle.optimizer;
            orbit_bundle.optimizer.maximum_iterations = 8;
            orbit_bundle.optimizer.optimize_rotations = true;
            orbit_bundle.optimizer.optimize_translations = false;
            orbit_bundle.optimizer.optimize_focal = true;
            orbit_bundle.optimizer.optimize_aspect_ratio = true;
            orbit_bundle.optimizer.optimize_distortion = true;
            if (!run_bundle_adjustment(scene, orbit_bundle).success)
                core::Logger::instance().warning(
                    "global: recovered-orbit bundle polish failed");
            filter_tracks(
                scene, fine_reproj_error,
                fallback_resection.min_angle_deg,
                fallback_resection.mult_depth_near,
                fallback_resection.mult_depth_far);
        }
    }

    // Match openMVS final behavior: retry images excluded from the largest
    // rotation component through robust incremental resection.
    if (scene.registered_count() < scene.images.size())
        register_images(scene, fallback_resection);

    if (enforce_zero_baseline_pose_groups(scene) > 0) {
        // The poses are now exact observations. Re-triangulate affected/all
        // tracks without another free-camera BA, which could split the copies
        // again despite their identical pixels.
        triangulate_tracks(
            scene, false, fallback_resection.max_reproj_error,
            fallback_resection.min_angle_deg);
        filter_tracks(
            scene,
            std::min(fallback_resection.max_reproj_error, 2.F),
            fallback_resection.min_angle_deg,
            fallback_resection.mult_depth_near,
            fallback_resection.mult_depth_far);
    }

    summary.registered_views = scene.registered_count();
    summary.failed_views =
        static_cast<unsigned>(scene.images.size()) - summary.registered_views;
    const AlignmentObservability observability =
        analyze_alignment_observability(scene);
    summary.alignment_reliable_views = observability.reliable_views;
    summary.alignment_unreliable_views = observability.unreliable_views;
    for (const Track& track : scene.tracks) {
        if (track.is_triangulated()) ++summary.landmarks;
    }
    summary.valid = summary.registered_views >= 2 && summary.landmarks > 0;
    populate_reprojection_stats(scene, summary);
    core::Logger::instance().info(
        "global: rotations=", rotation_summary.estimated_images,
        " filtered_pairs=", rotation_summary.filtered_pairs,
        " positioned=", position_summary.positioned_images,
        " position_tracks=", position_summary.positioned_tracks,
        " position_irls=", position_summary.irls_iterations,
        " position_downweighted=",
            position_summary.downweighted_constraints,
        " position_quarantined=",
            position_summary.quarantined_constraints,
        " position_median=", position_summary.median_residual,
        " position_p90=", position_summary.p90_residual,
        " residual=", position_summary.final_residual);
    return summary;
}

namespace {
void finish_alignment_recovery(Scene& scene, const ReconstructionConfig& config) {
    prune_unsupported_registrations(scene, config.minimum_final_observations_per_image,
                                   config.maximum_final_reprojection_error_pixels);
    bool changed = !recover_stable_resections(scene).empty();
    changed = recover_weakly_connected_views(scene) != 0 || changed;
    if (!changed) return;
    // Recovery rebuilds tracks and introduces new cross-branch constraints.
    // Reconcile their points and poses before reporting a final alignment.
    // Calibration stays fixed, so weak branches cannot change the lens model.
    BundleOptions bundle;
    bundle.optimizer.maximum_iterations = 32;
    bundle.write_intrinsics = false;
    for (unsigned pass = 0; pass < 2; ++pass) {
        if (!run_bundle_adjustment(scene, bundle).success) break;
        filter_tracks(scene, 2.F, 1.F, 0.F, 0.F);
        if (pass == 1 || recover_stable_resections(scene).empty()) break;
    }
    prune_unsupported_registrations(scene, config.minimum_final_observations_per_image,
                                   config.maximum_final_reprojection_error_pixels);
}
// Frontend checkpoints restore the compact scene but not EXIF metadata. The
// late intrinsic split therefore refreshes it from the source files before
// mapping, so cold starts and resumed runs take the same code path.
void ensure_camera_identity_metadata(
    Scene& scene, const FrontEndOptions& options) {
    if (options.trust_focal_pixels) return;
    for (Image& image : scene.images) {
        if (!image.camera_identity.empty()) continue;
        image.camera_identity = io::load_camera_identity(
            image.path, &image.focal_length_mm);
    }
}

}  // namespace

ReconstructionSummary reconstruct(
    Scene& scene_out,
    const std::vector<std::filesystem::path>& image_paths,
    const ReconstructionConfig& config) {
    core::StageScope stage("sfm.reconstruct");
    FrontEndResult frontend = run_frontend(image_paths, config.frontend);
    scene_out = std::move(frontend.scene);
    ensure_camera_identity_metadata(scene_out, config.frontend);
    // Small immutable snapshot used only if a large incremental run later
    // needs an independent hierarchical retry. Incremental BA mutates these
    // shared intrinsics in place.
    const std::vector<PinholeCamera> frontend_cameras = scene_out.cameras;
    CheckpointStore checkpoints(config.frontend.checkpoint);
    const std::uint64_t mapping_key =
        reconstruction_key(frontend.tracks_checkpoint_key, config);
    if (checkpoints.load_scene(
            CheckpointStage::reconstruction, mapping_key, scene_out)) {
        scene_out.thread_count =
            parallel::resolve_thread_count(config.frontend.thread_count);
        core::Logger::instance().info("checkpoint hit: reconstruction");
        if (config.mode != ReconstructionMode::incremental) {
            finish_alignment_recovery(scene_out, config);
            if (config.resection.checkpoint_callback)
                config.resection.checkpoint_callback(scene_out);
            const ReconstructionSummary summary = summarize_scene(scene_out);
            if (summary.valid) colour_triangulated_tracks(scene_out);
            return summary;
        }
    }
    ReconstructionSummary summary;
    if (config.mode == ReconstructionMode::hierarchical) {
        HierarchicalConfig hierarchical = config.hierarchical;
        hierarchical.star = config.star;
        hierarchical.resection = config.resection;
        summary = run_hierarchical_mapping(scene_out, hierarchical);
    } else if (config.mode == ReconstructionMode::global) {
        summary = run_global_mapping(
            scene_out,
            config.global_rotation,
            config.global_positioning,
            config.resection);
    } else {
        ResectionConfig resection = config.resection;
        resection.checkpoint_interval =
            config.frontend.checkpoint.reconstruction_interval;
        if (checkpoints.writable()) {
            const auto application_checkpoint =
                resection.checkpoint_callback;
            resection.checkpoint_callback =
                [&, application_checkpoint](const Scene& scene) {
                    if (application_checkpoint)
                        application_checkpoint(scene);
                    checkpoints.save_scene(
                        CheckpointStage::reconstruction,
                        mapping_key, scene);
                };
        }
        summary = run_incremental_mapping(
            scene_out, config.star, resection);
        const unsigned missing_views = static_cast<unsigned>(
            scene_out.images.size() - summary.registered_views);
        const unsigned rescue_missing_threshold = std::max(
            config.incremental_hierarchical_rescue_min_missing,
            static_cast<unsigned>(std::ceil(
                std::max(
                    0.0,
                    config.incremental_hierarchical_rescue_min_missing_ratio) *
                static_cast<double>(scene_out.images.size()))));
        if (config.incremental_hierarchical_rescue &&
            missing_views > 0 &&
            missing_views >= rescue_missing_threshold &&
            scene_out.images.size() >
                config.hierarchical.cluster.max_views_per_cluster) {
            const ReconstructionSummary incremental_summary = summary;
            const unsigned incremental_registered = summary.registered_views;
            core::Logger::instance().warning(
                "incremental stalled: registered=", summary.registered_views,
                '/', scene_out.images.size(),
                "; starting hierarchical submap rescue");

            // Keep only the mapping state that the rescue overwrites. Moving
            // the large track arrays avoids duplicating feature descriptors
            // and makes rollback independent of the checkpoint/cache policy.
            std::vector<Track> incremental_tracks =
                std::move(scene_out.tracks);
            std::vector<std::vector<ImageTrackRef>> incremental_image_tracks =
                std::move(scene_out.image_tracks);
            const std::vector<PinholeCamera> incremental_cameras =
                scene_out.cameras;
            std::vector<Pose3D> incremental_poses;
            std::vector<bool> incremental_registered_flags;
            incremental_poses.reserve(scene_out.images.size());
            incremental_registered_flags.reserve(scene_out.images.size());
            for (const Image& image : scene_out.images) {
                incremental_poses.push_back(image.pose);
                incremental_registered_flags.push_back(image.registered);
            }
            std::vector<Vec3> incremental_relative_centers;
            incremental_relative_centers.reserve(scene_out.pairs.size());
            for (const ImagePair& pair : scene_out.pairs) {
                incremental_relative_centers.push_back(
                    pair.relative_pose.has_value()
                        ? pair.relative_pose->C
                        : Vec3::Zero());
            }
            const ResectionProgress incremental_progress =
                scene_out.resection_progress;
            const std::uint32_t incremental_generation =
                scene_out.registration_generation;
            for (Image& image : scene_out.images) {
                image.registered = false;
                image.pose = Pose3D::identity();
            }
            scene_out.cameras = frontend_cameras;
            for (ImagePair& pair : scene_out.pairs) {
                if (!pair.relative_pose.has_value()) continue;
                Vec3& relative_center = pair.relative_pose->C;
                const double norm = relative_center.norm();
                if (std::isfinite(norm) && norm > 1e-12)
                    relative_center /= norm;
            }
            scene_out.resection_progress = {};
            scene_out.registration_generation = 0;
            build_tracks(
                scene_out,
                config.hierarchical.cluster.min_pair_weight);
            HierarchicalConfig hierarchical = config.hierarchical;
            hierarchical.star = config.star;
            hierarchical.resection = config.resection;
            // Subscene checkpoints are not valid replacements for the parent
            // incremental scene. Save only the merged result below.
            hierarchical.resection.checkpoint_callback = {};
            ReconstructionSummary rescue =
                run_hierarchical_mapping(scene_out, hierarchical);
            core::Logger::instance().info(
                "incremental hierarchical rescue: registered=",
                rescue.registered_views, '/', scene_out.images.size(),
                " valid=", rescue.valid);
            if (rescue.valid &&
                rescue.registered_views > incremental_registered) {
                summary = rescue;
            } else {
                scene_out.tracks = std::move(incremental_tracks);
                scene_out.image_tracks = std::move(incremental_image_tracks);
                scene_out.cameras = incremental_cameras;
                for (std::size_t i = 0; i < scene_out.images.size(); ++i) {
                    scene_out.images[i].pose = incremental_poses[i];
                    scene_out.images[i].registered =
                        incremental_registered_flags[i];
                }
                for (std::size_t i = 0; i < scene_out.pairs.size(); ++i) {
                    if (scene_out.pairs[i].relative_pose.has_value())
                        scene_out.pairs[i].relative_pose->C =
                            incremental_relative_centers[i];
                }
                scene_out.resection_progress = incremental_progress;
                scene_out.registration_generation = incremental_generation;
                summary = incremental_summary;
                core::Logger::instance().warning(
                    "hierarchical rescue did not improve coverage; restored incremental state registered=",
                    summary.registered_views);
            }
        }
    }
    finish_alignment_recovery(scene_out, config);
    summary = summarize_scene(scene_out);
    if (config.resection.checkpoint_callback)
        config.resection.checkpoint_callback(scene_out);
    if (summary.valid)
        checkpoints.save_scene(
            CheckpointStage::reconstruction, mapping_key, scene_out);
    if (summary.valid) colour_triangulated_tracks(scene_out);
    return summary;
}

}  // namespace aetherscan::sfm
