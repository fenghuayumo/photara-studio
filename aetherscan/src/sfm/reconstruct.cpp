#include "sfm/reconstruct.hpp"

#include "core/logging.hpp"
#include "parallel/thread_pool.hpp"
#include "sfm/bundle.hpp"
#include "sfm/pair_weighting.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"

#include <cmath>
#include <iostream>

namespace aetherscan::sfm {
namespace {

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
    key.append(options.optimize_points);
    key.append(options.optimize_focal);
    key.append(options.optimize_principal_point);
    key.append(options.optimize_distortion);
}

void append_resection(
    FingerprintBuilder& key, const ResectionConfig& options) {
    key.append(options.min_correspondences);
    key.append(options.min_inliers);
    key.append(options.max_local_window);
    key.append(options.local_ba_every);
    key.append(options.max_pose_wave);
    for (const unsigned value : options.full_ba_every) key.append(value);
    key.append(options.ratio_correspondences);
    key.append(options.avg_inliers_ratio_force_ba);
    key.append(options.max_reproj_error);
    key.append(options.min_angle_deg);
    key.append(options.mult_depth_near);
    key.append(options.mult_depth_far);
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
    key.append_string("aetherscan-reconstruction-v5");
    key.append_string("aetherscan-cache-abi-20260712-1");
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
    key.append(config.global_rotation.max_l1_iterations);
    key.append(config.global_rotation.max_irls_iterations);
    key.append(config.global_rotation.step_convergence_threshold);
    key.append(config.global_rotation.irls_sigma_deg);
    key.append(config.global_rotation.max_relative_rotation_error_deg);
    key.append(config.global_rotation.use_pair_weights);
    key.append(
        static_cast<std::uint32_t>(config.global_rotation.weight_type));
    key.append(config.global_positioning.min_views_per_track);
    key.append(config.global_positioning.min_tracks_for_positioning);
    key.append(config.global_positioning.tracks_per_registered_image);
    key.append(config.global_positioning.max_tracks_for_positioning);
    key.append(config.global_positioning.coverage_grid_size);
    key.append(config.global_positioning.max_irls_iterations);
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
    key.append(static_cast<std::uint32_t>(
        config.global_positioning.constraint));
    key.append(config.global_positioning.constraint_reweight_scale);
    return key.value();
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
    for (const Track& track : scene.tracks)
        if (track.is_triangulated()) ++summary.landmarks;
    summary.valid =
        summary.registered_views >= 2 && summary.landmarks > 0;
    populate_reprojection_stats(scene, summary);
    return summary;
}

}  // namespace

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

    // Densify structure for BA: camera-only needs a full triangulation; a capped
    // only_points solve only marks a subset, so triangulate the remaining tracks.
    if (position_summary.positioned_tracks == 0) {
        triangulate_tracks(scene, false, 6.F, 1.F);
        core::Logger::instance().info(
            "global: triangulated after camera-only positioning");
    } else {
        triangulate_tracks(scene, true, 6.F, 1.F);
        core::Logger::instance().info(
            "global: densified tracks after only_points (",
            position_summary.positioned_tracks, " positioned)");
    }

    filter_tracks(scene, 6.F, 1.F, 0.F, 0.F);

    BundleOptions bundle;
    bundle.optimizer.maximum_iterations = 12;
    bundle.optimizer.huber_delta = 2.0;
    bundle.optimizer.optimize_rotations = false;
    bundle.optimizer.optimize_focal = true;
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

    bundle.optimizer.maximum_iterations = 25;
    bundle.optimizer.optimize_rotations = true;
    bundle.optimizer.optimize_focal = true;
    bundle.optimizer.optimize_distortion = true;
    if (!run_bundle_adjustment(scene, bundle).success) {
        core::Logger::instance().error("global: full bundle adjustment failed");
        return summary;
    }
    triangulate_tracks(
        scene,
        true,
        fallback_resection.max_reproj_error,
        fallback_resection.min_angle_deg);
    filter_tracks(
        scene,
        fallback_resection.max_reproj_error,
        fallback_resection.min_angle_deg,
        fallback_resection.mult_depth_near,
        fallback_resection.mult_depth_far);

    // Newly triangulated tracks were not part of the full BA above. A short
    // polish pass lowers their residuals, then a 2 px fine filter matches the
    // quality-oriented final stage used by production SfM pipelines.
    bundle.optimizer.maximum_iterations = 8;
    bundle.optimizer.optimize_rotations = true;
    bundle.optimizer.optimize_focal = true;
    bundle.optimizer.optimize_distortion = true;
    if (!run_bundle_adjustment(scene, bundle).success) {
        core::Logger::instance().warning(
            "global: final bundle polish failed; keeping previous solution");
    }
    const float fine_reproj_error =
        std::min(fallback_resection.max_reproj_error, 2.F);
    filter_tracks(
        scene,
        fine_reproj_error,
        fallback_resection.min_angle_deg,
        fallback_resection.mult_depth_near,
        fallback_resection.mult_depth_far);

    // Match openMVS final behavior: retry images excluded from the largest
    // rotation component through robust incremental resection.
    if (scene.registered_count() < scene.images.size())
        register_images(scene, fallback_resection);

    summary.registered_views = scene.registered_count();
    summary.failed_views =
        static_cast<unsigned>(scene.images.size()) - summary.registered_views;
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
        " residual=", position_summary.final_residual);
    return summary;
}

ReconstructionSummary reconstruct(
    Scene& scene_out,
    const std::vector<std::filesystem::path>& image_paths,
    const ReconstructionConfig& config) {
    core::StageScope stage("sfm.reconstruct");
    FrontEndResult frontend = run_frontend(image_paths, config.frontend);
    scene_out = std::move(frontend.scene);
    CheckpointStore checkpoints(config.frontend.checkpoint);
    const std::uint64_t mapping_key =
        reconstruction_key(frontend.tracks_checkpoint_key, config);
    if (checkpoints.load_scene(
            CheckpointStage::reconstruction, mapping_key, scene_out)) {
        scene_out.thread_count =
            parallel::resolve_thread_count(config.frontend.thread_count);
        core::Logger::instance().info("checkpoint hit: reconstruction");
        if (config.mode != ReconstructionMode::incremental)
            return summarize_scene(scene_out);
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
    }
    if (summary.valid)
        checkpoints.save_scene(
            CheckpointStage::reconstruction, mapping_key, scene_out);
    populate_reprojection_stats(scene_out, summary);
    return summary;
}

}  // namespace aetherscan::sfm
