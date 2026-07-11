#include "aetherscan/sfm/reconstruct.hpp"

#include "aetherscan/sfm/bundle.hpp"
#include "aetherscan/sfm/tracks.hpp"
#include "aetherscan/sfm/triangulation.hpp"

#include <iostream>

namespace aetherscan::sfm {

ReconstructionSummary run_incremental_mapping(
    Scene& scene,
    const StarInitConfig& star,
    const ResectionConfig& resection) {
    ReconstructionSummary summary;
    if (!star_initialize(scene, star)) return summary;
    register_images(scene, resection);

    summary.registered_views = scene.registered_count();
    summary.failed_views =
        static_cast<unsigned>(scene.images.size()) - summary.registered_views;
    for (const Track& track : scene.tracks) {
        if (track.is_triangulated()) ++summary.landmarks;
    }
    summary.valid = summary.registered_views >= 2 && summary.landmarks > 0;
    return summary;
}

ReconstructionSummary run_global_mapping(
    Scene& scene,
    const GlobalRotationOptions& rotation,
    const GlobalPositioningOptions& positioning,
    const ResectionConfig& fallback_resection) {
    ReconstructionSummary summary;

    GlobalRotationSummary rotation_summary;
    unsigned total_filtered_pairs = 0;
    bool rotation_filter_converged = false;
    for (unsigned pass = 0; pass < 4; ++pass) {
        rotation_summary = estimate_global_rotations(scene, rotation);
        if (!rotation_summary.success) {
            std::cerr << "global: rotation averaging pass " << (pass + 1)
                      << " failed\n";
            return summary;
        }
        total_filtered_pairs += rotation_summary.filtered_pairs;
        if (rotation_summary.filtered_pairs == 0) {
            rotation_filter_converged = true;
            break;
        }
    }
    if (!rotation_filter_converged) {
        GlobalRotationOptions final_rotation = rotation;
        final_rotation.max_relative_rotation_error_deg = 0.0;
        rotation_summary = estimate_global_rotations(scene, final_rotation);
        if (!rotation_summary.success) {
            std::cerr << "global: final rotation averaging pass failed\n";
            return summary;
        }
    }
    rotation_summary.filtered_pairs = total_filtered_pairs;
    if (total_filtered_pairs > 0) build_tracks(scene);
    std::cout << "global: rotation_images=" << rotation_summary.estimated_images
              << " used_pairs=" << rotation_summary.used_pairs
              << " filtered_pairs=" << rotation_summary.filtered_pairs
              << " tracks=" << scene.tracks.size() << '\n';

    const GlobalPositioningSummary position_summary =
        solve_global_positions(scene, positioning);
    if (!position_summary.success) {
        std::cerr << "global: positioning failed after "
                  << position_summary.iterations << " iterations, observations="
                  << position_summary.observations << '\n';
        return summary;
    }

    triangulate_tracks(scene, false, 6.F, 1.F);
    filter_tracks(scene, 6.F, 1.F, 0.F, 0.F);

    BundleOptions bundle;
    bundle.optimizer.maximum_iterations = 12;
    bundle.optimizer.huber_delta = 2.0;
    if (!run_bundle_adjustment(scene, bundle).success) {
        std::cerr << "global: position/structure bundle adjustment failed\n";
        return summary;
    }
    filter_tracks(
        scene,
        fallback_resection.max_reproj_error,
        fallback_resection.min_angle_deg,
        fallback_resection.mult_depth_near,
        fallback_resection.mult_depth_far);

    bundle.optimizer.maximum_iterations = 25;
    if (!run_bundle_adjustment(scene, bundle).success) {
        std::cerr << "global: full bundle adjustment failed\n";
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
    std::cout << "global: rotations=" << rotation_summary.estimated_images
              << " filtered_pairs=" << rotation_summary.filtered_pairs
              << " positioned=" << position_summary.positioned_images
              << " position_tracks=" << position_summary.positioned_tracks
              << " residual=" << position_summary.final_residual << '\n';
    return summary;
}

ReconstructionSummary reconstruct(
    Scene& scene_out,
    const std::vector<std::filesystem::path>& image_paths,
    const ReconstructionConfig& config) {
    FrontEndResult frontend = run_frontend(image_paths, config.frontend);
    scene_out = std::move(frontend.scene);
    if (config.mode == ReconstructionMode::global) {
        return run_global_mapping(
            scene_out,
            config.global_rotation,
            config.global_positioning,
            config.resection);
    }
    return run_incremental_mapping(scene_out, config.star, config.resection);
}

}  // namespace aetherscan::sfm
