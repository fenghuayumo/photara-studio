#include "sfm/hierarchical.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/rotation_recovery.hpp"
#include "sfm/triangulation.hpp"
#include "core/logging.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <random>
#include <vector>

namespace {

using namespace photara::sfm;

int failures = 0;

void expect(const bool condition, const char* message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

Scene make_scene(const int k_views = 6, const int k_points = 80) {
    Scene scene;
    PinholeCamera camera;
    camera.width = 640;
    camera.height = 480;
    camera.fx = camera.fy = 500.0;
    camera.cx = 320.0;
    camera.cy = 240.0;

    for (int i = 0; i < k_views; ++i) {
        camera.id = static_cast<Index>(i);
        scene.cameras.push_back(camera);
        Image image;
        image.id = static_cast<Index>(i);
        image.camera_id = static_cast<Index>(i);
        image.features.keypoints.resize(k_points);
        const double angle = i * 2.0 * 3.14159265358979323846 / k_views;
        image.pose.C = Vec3(2.0 * std::cos(angle), 0.2 * std::sin(2.0 * angle),
                            2.0 * std::sin(angle));
        const Vec3 forward = (-image.pose.C).normalized();
        const Vec3 right = Vec3(0, 1, 0).cross(forward).normalized();
        image.pose.R.row(0) = right.transpose();
        image.pose.R.row(1) = forward.cross(right).transpose();
        image.pose.R.row(2) = forward.transpose();
        image.registered = true;
        scene.images.push_back(std::move(image));
    }

    std::mt19937 random(7);
    std::uniform_real_distribution<double> coordinate(-0.45, 0.45);
    for (Index point_id = 0; point_id < k_points; ++point_id) {
        const Vec3 point(coordinate(random), coordinate(random), coordinate(random));
        for (Image& image : scene.images) {
            const Vec2 pixel =
                camera.project(image.pose.transform_world_to_camera(point));
            image.features.keypoints[point_id].x = static_cast<float>(pixel.x());
            image.features.keypoints[point_id].y = static_cast<float>(pixel.y());
        }
    }

    for (Index first = 0; first < k_views; ++first) {
        for (Index second = first + 1; second < k_views; ++second) {
            ImagePair pair(first, second);
            pair.weight_spatial = 1.F;
            pair.relative_pose =
                scene.images[second].pose / scene.images[first].pose;
            if (pair.relative_pose->C.norm() > 1e-9)
                pair.relative_pose->C.normalize();
            for (Index point_id = 0; point_id < k_points; ++point_id)
                pair.matches.push_back({point_id, point_id});
            scene.pairs.push_back(std::move(pair));
        }
    }

    for (Image& image : scene.images) {
        image.registered = false;
        image.pose = Pose3D::identity();
    }
    return scene;
}

Scene make_track_split_scene() {
    Scene scene;
    PinholeCamera camera;
    camera.id = 0;
    camera.width = 640;
    camera.height = 480;
    camera.fx = camera.fy = 500.0;
    camera.cx = 320.0;
    camera.cy = 240.0;
    scene.cameras.push_back(camera);

    const Vec3 first_point(0.0, 0.0, 5.0);
    const Vec3 second_point(1.0, 0.0, 5.0);
    Track contaminated;
    for (Index image_id = 0; image_id < 6; ++image_id) {
        Image image;
        image.id = image_id;
        image.camera_id = 0;
        image.pose.C = Vec3(static_cast<double>(image_id) - 2.0, 0.0, 0.0);
        image.registered = true;
        const Vec3& point = image_id < 3 ? first_point : second_point;
        const Vec2 pixel =
            camera.project(image.pose.transform_world_to_camera(point));
        image.features.keypoints.resize(1);
        image.features.keypoints[0].x = static_cast<float>(pixel.x());
        image.features.keypoints[0].y = static_cast<float>(pixel.y());
        scene.images.push_back(std::move(image));
        contaminated.observations.push_back({image_id, 0});
    }

    // Multiple parents guarantee that appending children grows the vector,
    // exercising reference invalidation in the split loop.
    scene.tracks = std::vector<Track>(16, contaminated);
    return scene;
}

}  // namespace

int main() {
    {
        const auto directory = std::filesystem::temp_directory_path() / "photara-hierarchical-parameters";
        const auto log_path = photara::core::Logger::instance().configure(
            directory, "forwarding", photara::core::LogLevel::warning);
        ReconstructionConfig options;
        options.mode = ReconstructionMode::hierarchical;
        options.global_positioning.backend = PositioningBackend::cpu;
        options.global_positioning.max_solver_time_sec = 0.001;
        options.global_rotation.max_l1_iterations = 0;
        // Exercise the Global rejection path without completing a new star seed.
        options.star.min_initial_tracks = 100000;
        Scene limited = make_scene();
        reconstruct_from_frontend(limited, options);
        options.hierarchical.cluster.max_views_per_cluster = 3;
        options.hierarchical.cluster.min_views_per_cluster = 2;
        options.hierarchical.cluster.min_common_tracks = 10;
        options.star.max_views = 3;
        Scene clustered = make_scene();
        reconstruct_from_frontend(clustered, options);
        photara::core::Logger::instance().configure({}, "tests", photara::core::LogLevel::warning);
        std::ifstream input(log_path);
        const std::string trace((std::istreambuf_iterator<char>(input)), {});
        std::size_t budget_calls=0, offset=0;
        while ((offset=trace.find("max_solver_time_s=0.001", offset))!=std::string::npos) {
            ++budget_calls;++offset;
        }
        expect(budget_calls >= 3,
               "custom positioning budget reaches fast path and both cluster retries");
        expect(trace.find("global mapping unavailable; falling back") != std::string::npos &&
               trace.find("hierarchical subscene accepted global retry") == std::string::npos,
               "budget-limited Global attempts follow fallback paths instead of ignoring the budget");
        expect(trace.find("LAD initialized") == std::string::npos,
               "custom rotation iteration limit reaches the hierarchical Global solve");
        input.close();
        std::filesystem::remove(log_path);
    }
    Scene split_tracks_scene = make_track_split_scene();
    TriangulationOptions split_options;
    split_options.reproj_threshold_px = 0.25F;
    split_options.min_angle_deg = 0.F;
    split_options.ransac_iterations = 256;
    split_options.refine_nonlinear = false;
    split_options.max_splits_per_track = 1;
    triangulate_tracks(split_tracks_scene, false, split_options);
    expect(
        split_tracks_scene.tracks.size() == 32,
        "contaminated tracks split without invalidating parent references");
    for (const Track& track : split_tracks_scene.tracks)
        expect(track.is_triangulated(), "split track remains triangulated");
    for (const Track& track : split_tracks_scene.tracks)
        expect(
            track.split_generation ==
                split_tracks_scene.registration_generation,
            "split generation records the completed outlier check");
    const std::size_t checked_track_count = split_tracks_scene.tracks.size();
    triangulate_tracks(split_tracks_scene, true, split_options);
    expect(
        split_tracks_scene.tracks.size() == checked_track_count,
        "same-generation outlier checks do not split tracks again");

    Scene split_scene = make_scene();
    ClusterConfig cluster;
    cluster.max_views_per_cluster = 3;
    cluster.min_views_per_cluster = 2;
    cluster.min_common_tracks = 10;
    const auto first_split = split_hierarchical_scene(split_scene, cluster);
    const auto second_split = split_hierarchical_scene(split_scene, cluster);
    expect(first_split.size() == 2, "scene splits into two clusters");
    expect(first_split.size() == second_split.size(), "split is deterministic");
    if (first_split.size() == second_split.size()) {
        for (std::size_t i = 0; i < first_split.size(); ++i)
            expect(
                first_split[i].local_to_global == second_split[i].local_to_global,
                "cluster membership is deterministic");
    }
    expect(
        split_hierarchical_scene(split_scene).size() == 1,
        "ordinary capture sizes stay in one global map");

    Scene scene = make_scene();
    HierarchicalConfig config;
    config.cluster = cluster;
    config.star.min_views = 3;
    config.star.max_views = 3;
    config.star.min_initial_tracks = 20;
    config.star.min_angle_deg = 0.5F;
    config.resection.min_angle_deg = 0.5F;
    config.alignment.min_common_tracks = 10;
    config.final_bundle_adjustment = true;
    const ReconstructionSummary summary = run_hierarchical_mapping(scene, config);
    expect(summary.valid, "hierarchical reconstruction succeeds");
    expect(summary.registered_views == 6, "all hierarchical views registered");
    expect(summary.landmarks >= 40, "hierarchical reconstruction keeps landmarks");
    expect(
        scene.image_tracks.size() == scene.images.size(),
        "hierarchical merge rebuilds track inverted index");
    for (const auto& references : scene.image_tracks)
        for (const ImageTrackRef& reference : references)
            expect(
                reference.track_id < scene.tracks.size(),
                "hierarchical track index references valid track");

    Scene failed_stars_scene = make_scene(8);
    HierarchicalConfig retry_config = config;
    retry_config.cluster.max_views_per_cluster = 4;
    retry_config.star.min_initial_tracks = 10000;
    const auto retry_summary = run_hierarchical_mapping(failed_stars_scene, retry_config);
    expect(retry_summary.valid && retry_summary.registered_views == 8,
        "failed local stars recover through audited cluster-global initialization");
    expect(retry_summary.landmarks >= 40,
        "cluster retry retains independently supported structure");

    Scene mixed_camera_scene = make_scene(8);
    mixed_camera_scene.cameras.resize(1);
    mixed_camera_scene.cameras[0].trust_intrinsics = false;
    for (Image& image : mixed_camera_scene.images) {
        image.camera_id = 0;
        image.camera_identity = image.id < 4 ? "camera-a" : "camera-b";
    }
    HierarchicalConfig mixed_camera_config = config;
    mixed_camera_config.cluster.max_views_per_cluster = 4;
    const auto mixed_summary = run_hierarchical_mapping(mixed_camera_scene, mixed_camera_config);
    expect(mixed_summary.valid && mixed_summary.registered_views == 8,
        "multi-cluster reconstruction registers views from both physical cameras");
    expect(mixed_camera_scene.cameras.size() == 2,
        "merge preserves physical camera groups initially sharing one intrinsic group");
    expect(mixed_camera_scene.images[0].camera_id != mixed_camera_scene.images[4].camera_id,
        "different physical cameras retain separate calibration after merge");
    for (const Image& image : mixed_camera_scene.images)
        expect(image.camera_id == mixed_camera_scene.images[image.id < 4 ? 0 : 4].camera_id,
            "views from the same physical camera share their merged calibration");

    Scene conflict_scene = make_scene(6, 120);
    const auto conflict_summary = run_hierarchical_mapping(conflict_scene, config);
    expect(conflict_summary.registered_views == 6, "rotation audit fixture reconstructs all views");
    for (ImagePair& pair : conflict_scene.pairs) pair.weight_triplet = 0.5F;
    conflict_scene.images[0].pose.R =
        Eigen::AngleAxisd(0.2, Vec3::UnitY()).toRotationMatrix() * conflict_scene.images[0].pose.R;
    Scene repeated_neighbor = conflict_scene;
    repeated_neighbor.pairs = {conflict_scene.pairs.front(), conflict_scene.pairs.front()};
    const auto repeated_conflicts = verified_rotation_conflicts(repeated_neighbor);
    expect(!repeated_conflicts[0] && !repeated_conflicts[1],
        "repeated edges to one neighbor cannot reject either camera");
    expect(prune_verified_rotation_conflicts(conflict_scene) == 1,
        "final audit rejects only the pose contradicting multiple strong neighbors");
    expect(!conflict_scene.images[0].registered && conflict_scene.registered_count() == 5,
        "verified rotation conflict is excluded from effective alignment");
    for (const Track& track : conflict_scene.tracks)
        for (unsigned i = 0; i < track.num_inliers; ++i)
            expect(track.observations[i].image_id != 0,
                "rejected camera observations leave the triangulated inlier prefix");

    Scene single_cluster_scene = make_scene();
    HierarchicalConfig single_cluster_config = config;
    single_cluster_config.cluster.max_views_per_cluster = 200;
    single_cluster_config.star.max_views = 6;
    const ReconstructionSummary single_cluster_summary =
        run_hierarchical_mapping(single_cluster_scene, single_cluster_config);
    expect(
        single_cluster_summary.valid,
        "single-cluster hierarchical fast path succeeds");
    expect(
        single_cluster_summary.registered_views == 6,
        "single-cluster fast path registers every view");
    expect(
        single_cluster_scene.image_tracks.size() ==
            single_cluster_scene.images.size(),
        "single-cluster fast path keeps the track index valid");

    if (failures == 0) {
        std::cout << "sfm hierarchical tests passed\n";
        return 0;
    }
    std::cerr << failures << " failures\n";
    return 1;
}
