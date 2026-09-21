#include "sfm/hierarchical.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/triangulation.hpp"

#include <cmath>
#include <iostream>
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

Scene make_scene() {
    constexpr int k_views = 6;
    constexpr int k_points = 80;
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
