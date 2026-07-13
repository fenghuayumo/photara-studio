#include "sfm/star_init.hpp"
#include "sfm/bundle.hpp"
#include "sfm/global_positioning.hpp"
#include "sfm/global_rotation.hpp"
#include "sfm/pair_weighting.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/retrieval.hpp"
#include "sfm/resection.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

namespace {

using namespace aetherscan::sfm;

int failures = 0;
void expect(bool c, const char* m) {
    if (!c) {
        std::cerr << "FAIL: " << m << '\n';
        ++failures;
    }
}

PinholeCamera cam() {
    PinholeCamera c;
    c.width = 640;
    c.height = 480;
    c.fx = c.fy = 500;
    c.cx = 320;
    c.cy = 240;
    return c;
}

Mat3 axis_rotation(const Vec3& axis, double degrees) {
    constexpr double k_pi = 3.14159265358979323846;
    return Eigen::AngleAxisd(
               degrees * k_pi / 180.0, axis.normalized())
        .toRotationMatrix();
}

double rotation_error(const Mat3& measured, const Mat3& expected) {
    return Eigen::AngleAxisd(measured * expected.transpose()).angle();
}

void test_pair_cycle_weighting() {
    Scene scene;
    scene.images.resize(4);
    for (Index i = 0; i < scene.images.size(); ++i)
        scene.images[i].id = i;

    const Mat3 rotations[] = {
        Mat3::Identity(),
        axis_rotation(Vec3::UnitX(), 10.0),
        axis_rotation(Vec3::UnitY(), 15.0),
        axis_rotation(Vec3::UnitZ(), -12.0)};
    auto add_pair = [&](Index first, Index second, Mat3 relative) {
        ImagePair pair(first, second);
        pair.relative_pose = Pose3D{relative, Vec3::Zero()};
        pair.weight_spatial = 1.F;
        pair.matches.resize(100);
        scene.pairs.push_back(std::move(pair));
    };
    for (Index first = 0; first < 4; ++first) {
        for (Index second = first + 1; second < 4; ++second) {
            Mat3 relative = rotations[second] * rotations[first].transpose();
            if (first == 0 && second == 1)
                relative = axis_rotation(Vec3::UnitZ(), 25.0) * relative;
            add_pair(first, second, relative);
        }
    }

    PairWeightingOptions options;
    options.min_triplets_for_penalty = 2;
    options.max_inconsistent_triplet_ratio = 0.5F;
    const PairWeightingSummary summary = compute_pair_weights(scene, options);
    expect(summary.tested_triplets == 4, "pair weighting counts K4 triplets");
    expect(summary.inconsistent_pairs == 1, "pair weighting isolates bad edge");
    expect(scene.pairs[0].weight_cycle < 0.2F,
           "bad cycle edge receives strong penalty");
    expect(scene.pairs[5].weight_triplet > 0.F &&
               scene.pairs[5].weight_cycle == 1.F,
           "consistent edge retains and gains support");
    expect(scene.pairs[0].composite_weight() < scene.pairs[5].composite_weight(),
           "cycle-supported edge outranks inconsistent edge");
}

void test_global_rotation_weighting() {
    Scene scene;
    scene.images.resize(3);
    for (Index i = 0; i < scene.images.size(); ++i)
        scene.images[i].id = i;

    const Mat3 r0 = Mat3::Identity();
    const Mat3 r1 = axis_rotation(Vec3(1, 0, 0), 30.0);
    const Mat3 r2 =
        axis_rotation(Vec3(0, 1, 0), 25.0) * r1;
    const Mat3 rotations[] = {r0, r1, r2};
    auto add_pair = [&](Index a, Index b, const Mat3& relative, float spatial) {
        ImagePair pair(a, b);
        pair.relative_pose = Pose3D{relative, Vec3::Zero()};
        pair.weight_spatial = spatial;
        pair.matches.resize(1000);
        scene.pairs.push_back(std::move(pair));
    };
    add_pair(0, 1, r1 * r0.transpose(), 1.F);
    add_pair(1, 2, r2 * r1.transpose(), 1.F);
    add_pair(
        0,
        2,
        axis_rotation(Vec3(0, 0, 1), 20.0) *
            r2 * r0.transpose(),
        0.5F);

    GlobalRotationOptions options;
    options.max_l1_iterations = 0;
    options.max_irls_iterations = 1;
    options.irls_sigma_deg = 1e6;
    options.max_relative_rotation_error_deg = 0.0;
    const GlobalRotationSummary summary =
        estimate_global_rotations(scene, options);
    expect(summary.success, "global rotation weighting solve");

    const double true_edge_error =
        rotation_error(
            scene.images[1].pose.R * scene.images[0].pose.R.transpose(),
            rotations[1] * rotations[0].transpose()) +
        rotation_error(
            scene.images[2].pose.R * scene.images[1].pose.R.transpose(),
            rotations[2] * rotations[1].transpose());
    const double bad_edge_error = rotation_error(
        scene.images[2].pose.R * scene.images[0].pose.R.transpose(),
        scene.pairs[2].relative_pose->R);
    expect(true_edge_error < 0.20, "raw pair weights dominate rotation solve");
    expect(bad_edge_error > 0.14, "rotation outlier retains expected residual");
}

void test_global_positioning_points_only() {
    constexpr int k_views = 4;
    constexpr int k_points = 30;
    Scene scene;
    const PinholeCamera camera = cam();
    std::vector<Vec3> true_centers;
    for (int view = 0; view < k_views; ++view) {
        scene.cameras.push_back(camera);
        Image image;
        image.id = static_cast<Index>(view);
        image.camera_id = static_cast<Index>(view);
        image.registered = true;
        image.features.keypoints.resize(k_points);
        image.pose.C = Vec3(
            1.5 * view,
            0.2 * (view % 2),
            0.1 * view);
        true_centers.push_back(image.pose.C);
        scene.images.push_back(std::move(image));
    }

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> xy(-1.0, 1.0);
    for (int point = 0; point < k_points; ++point) {
        const Vec3 position(xy(rng) + 2.0, xy(rng), 5.0 + 0.2 * point);
        Track track;
        for (int view = 0; view < k_views; ++view) {
            const Vec3 camera_point =
                scene.images[view].pose.transform_world_to_camera(position);
            const Vec2 pixel = camera.project(camera_point);
            scene.images[view].features.keypoints[point].x =
                static_cast<float>(pixel.x());
            scene.images[view].features.keypoints[point].y =
                static_cast<float>(pixel.y());
            track.observations.push_back(
                {static_cast<Index>(view), static_cast<Index>(point)});
        }
        scene.tracks.push_back(std::move(track));
    }

    // Deliberately inconsistent camera constraints must not affect the
    // openMVS default (ONLY_POINTS).
    for (Index view = 0; view + 1 < k_views; ++view) {
        ImagePair pair(view, view + 1);
        pair.weight_spatial = 1.F;
        pair.matches.resize(100);
        pair.relative_pose = Pose3D{
            Mat3::Identity(), Vec3(0.0, 10.0, 0.0)};
        scene.pairs.push_back(std::move(pair));
    }

    const GlobalPositioningSummary summary = solve_global_positions(scene);
    expect(summary.success, "global positioning converges");
    expect(summary.positioned_images == k_views, "global positioning uses all images");
    expect(summary.positioned_tracks == k_points, "global positioning uses all tracks");

    Vec3 estimated_mean = Vec3::Zero();
    Vec3 true_mean = Vec3::Zero();
    for (int view = 0; view < k_views; ++view) {
        estimated_mean += scene.images[view].pose.C;
        true_mean += true_centers[view];
    }
    estimated_mean /= k_views;
    true_mean /= k_views;
    double numerator = 0.0;
    double denominator = 0.0;
    for (int view = 0; view < k_views; ++view) {
        const Vec3 estimated = scene.images[view].pose.C - estimated_mean;
        const Vec3 truth = true_centers[view] - true_mean;
        numerator += estimated.dot(truth);
        denominator += estimated.squaredNorm();
    }
    const double scale = numerator / denominator;
    double squared_error = 0.0;
    for (int view = 0; view < k_views; ++view) {
        const Vec3 aligned =
            scale * (scene.images[view].pose.C - estimated_mean);
        squared_error +=
            (aligned - (true_centers[view] - true_mean)).squaredNorm();
    }
    expect(
        std::sqrt(squared_error / k_views) < 1e-2,
        "point-only positioning recovers camera layout");
    double minimum_center_norm = std::numeric_limits<double>::max();
    for (const Image& image : scene.images)
        minimum_center_norm = std::min(minimum_center_norm, image.pose.C.norm());
    expect(minimum_center_norm < 1e-12,
           "point-only positioning explicitly fixes translation gauge");
}

void test_large_point_only_positioning_does_not_require_pairs() {
    constexpr Index k_tracks = 2000;
    Scene scene;
    scene.cameras.assign(2, cam());
    scene.images.resize(2);
    const Vec3 centers[] = {Vec3(-0.5, 0.0, 0.0), Vec3(0.5, 0.0, 0.0)};
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        Image& image = scene.images[image_id];
        image.id = image_id;
        image.camera_id = image_id;
        image.registered = true;
        image.pose.C = centers[image_id];
        image.features.keypoints.resize(k_tracks);
    }
    scene.tracks.reserve(k_tracks);
    for (Index track_id = 0; track_id < k_tracks; ++track_id) {
        Track track;
        track.position = Vec3(
            (static_cast<double>(track_id % 40) - 20.0) * 0.02,
            (static_cast<double>((track_id / 40) % 25) - 12.0) * 0.02,
            4.0 + 0.001 * track_id);
        for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
            const Vec2 pixel = scene.cameras[image_id].project(
                scene.images[image_id].pose.transform_world_to_camera(
                    track.position));
            scene.images[image_id].features.keypoints[track_id].x =
                static_cast<float>(pixel.x());
            scene.images[image_id].features.keypoints[track_id].y =
                static_cast<float>(pixel.y());
            track.observations.push_back({image_id, track_id});
        }
        track.num_inliers = 2;
        scene.tracks.push_back(std::move(track));
    }

    GlobalPositioningOptions options;
    options.min_views_per_track = 2;
    options.max_tracks_for_positioning = 0;
    options.max_num_iterations = 2;
    options.max_solver_time_sec = 5.0;
    options.generate_random_positions = false;
    options.generate_random_points = false;
    options.generate_scales = false;
    options.optimize_positions = false;
    options.optimize_points = false;
    const GlobalPositioningSummary summary =
        solve_global_positions(scene, options);
    expect(summary.success,
           "large point-only positioning does not require camera pairs");
    expect(summary.positioned_tracks == k_tracks,
           "large point-only positioning keeps track constraints primary");
}

void test_global_positioning_irls_downweights_bad_direction() {
    Scene scene;
    scene.cameras.assign(4, cam());
    scene.images.resize(4);
    const Vec3 centers[] = {
        Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0), Vec3(1.0, 1.0, 0.5)};
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        scene.images[image_id].id = image_id;
        scene.images[image_id].camera_id = image_id;
        scene.images[image_id].registered = true;
    }
    for (Index first = 0; first < scene.images.size(); ++first) {
        for (Index second = first + 1; second < scene.images.size(); ++second) {
            Vec3 direction = (centers[second] - centers[first]).normalized();
            if (first == 0 && second == 1) direction = Vec3::UnitZ();
            ImagePair pair(first, second);
            pair.weight_spatial = 1.F;
            pair.matches.resize(100);
            pair.relative_pose = Pose3D{
                Mat3::Identity(), -direction};
            scene.pairs.push_back(std::move(pair));
        }
    }

    GlobalPositioningOptions options;
    options.constraint = GlobalPositioningConstraint::only_cameras;
    options.max_num_iterations = 30;
    options.max_irls_iterations = 3;
    options.irls_inner_iterations = 8;
    options.irls_tuning_constant = 2.0;
    options.irls_quarantine_after = 1;
    options.huber_threshold = 0.02;
    options.max_solver_time_sec = 10.0;
    const GlobalPositioningSummary summary =
        solve_global_positions(scene, options);
    expect(summary.success, "camera positioning IRLS succeeds");
    expect(summary.irls_iterations > 0,
           "camera positioning executes outer IRLS");
    expect(summary.downweighted_constraints > 0,
           "camera positioning downweights inconsistent direction");
    expect(summary.quarantined_constraints > 0,
           "persistent direction outlier enters quarantine");
    expect(summary.p90_residual >= summary.median_residual,
           "positioning reports robust residual quantiles");
}

void test_global_positioning_preserves_fixed_initial_positions() {
    Scene scene;
    scene.cameras.assign(2, cam());
    scene.images.resize(2);
    const Vec3 centers[] = {Vec3(-0.5, 0.0, 0.0), Vec3(0.5, 0.0, 0.0)};
    const Vec3 point(0.0, 0.1, 4.0);
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        Image& image = scene.images[image_id];
        image.id = image_id;
        image.camera_id = image_id;
        image.registered = true;
        image.pose.C = centers[image_id];
        image.features.keypoints.resize(1);
        const Vec2 pixel = scene.cameras[image_id].project(
            image.pose.transform_world_to_camera(point));
        image.features.keypoints[0].x = static_cast<float>(pixel.x());
        image.features.keypoints[0].y = static_cast<float>(pixel.y());
    }
    Track track;
    track.position = point;
    track.observations = {{0, 0}, {1, 0}};
    scene.tracks.push_back(track);

    GlobalPositioningOptions options;
    options.min_views_per_track = 2;
    options.generate_random_positions = false;
    options.generate_random_points = false;
    options.generate_scales = false;
    options.optimize_positions = false;
    options.optimize_points = false;
    options.optimize_scales = true;
    const GlobalPositioningSummary summary =
        solve_global_positions(scene, options);
    expect(summary.success, "fixed-position global solve succeeds");
    expect(
        (scene.images[0].pose.C - centers[0]).norm() == 0.0 &&
            (scene.images[1].pose.C - centers[1]).norm() == 0.0,
        "global positioning does not randomize fixed initial centers");
}

void test_global_positioning_rejects_empty_point_constraints() {
    Scene scene;
    scene.cameras.assign(2, cam());
    scene.images.resize(2);
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        scene.images[image_id].id = image_id;
        scene.images[image_id].camera_id = image_id;
        scene.images[image_id].registered = true;
        scene.images[image_id].pose.C = Vec3(static_cast<double>(image_id), 0.0, 0.0);
    }
    ImagePair pair(0, 1);
    pair.relative_pose = Pose3D{Mat3::Identity(), Vec3(1.0, 0.0, 0.0)};
    pair.weight_spatial = 1.F;
    pair.matches.resize(20);
    scene.pairs.push_back(std::move(pair));
    scene.tracks.emplace_back();
    const Vec3 first_center = scene.images[0].pose.C;
    const Vec3 second_center = scene.images[1].pose.C;

    GlobalPositioningOptions options;
    options.constraint =
        GlobalPositioningConstraint::points_and_cameras_balanced;
    const GlobalPositioningSummary summary =
        solve_global_positions(scene, options);
    expect(!summary.success, "empty point constraints fail cleanly");
    expect(
        scene.images[0].pose.C == first_center &&
            scene.images[1].pose.C == second_center,
        "failed balanced positioning preserves input centers");
}

void test_long_track_merge() {
    constexpr Index k_views = 8;
    Scene scene;
    scene.images.resize(k_views);
    for (Index image_id = 0; image_id < k_views; ++image_id) {
        scene.images[image_id].id = image_id;
        scene.images[image_id].features.keypoints.resize(1);
    }
    for (Index image_id = 1; image_id < k_views; ++image_id) {
        ImagePair pair(0, image_id);
        pair.weight_spatial = 1.F;
        pair.matches.push_back({0, 0});
        scene.pairs.push_back(std::move(pair));
    }
    for (Index image_id = 1; image_id + 1 < k_views; ++image_id) {
        ImagePair pair(image_id, image_id + 1);
        pair.weight_spatial = 1.F;
        pair.matches.push_back({0, 0});
        scene.pairs.push_back(std::move(pair));
    }

    build_tracks(scene);
    expect(scene.tracks.size() == 1, "repeated pair edges keep one long track");
    expect(
        !scene.tracks.empty() &&
            scene.tracks.front().observations.size() == k_views,
        "long track retains one observation per image");
}

void test_retrieval_inverted_index() {
    std::vector<Image> images(6);
    std::mt19937 random(77);
    std::normal_distribution<float> noise(0.F, 0.01F);
    std::vector<std::vector<float>> prototypes(3, std::vector<float>(128, 0.F));
    for (std::size_t cluster = 0; cluster < prototypes.size(); ++cluster) {
        for (std::size_t column = 0; column < 32; ++column)
            prototypes[cluster][cluster * 32 + column] = 1.F;
    }

    for (Index image_id = 0; image_id < images.size(); ++image_id) {
        Image& image = images[image_id];
        image.id = image_id;
        image.features.image_width = 640;
        image.features.image_height = 480;
        image.features.descriptor_dimension = 128;
        image.features.keypoints.resize(300);
        image.features.descriptors.resize(300 * 128);
        const auto& prototype = prototypes[image_id / 2];
        for (std::size_t row = 0; row < 300; ++row) {
            image.features.keypoints[row].x =
                static_cast<float>((row % 20) * 30 + 10);
            image.features.keypoints[row].y =
                static_cast<float>((row / 20) * 30 + 10);
            image.features.keypoints[row].response = 1.F;
            image.features.keypoints[row].scale = 4.F;
            for (std::size_t column = 0; column < 128; ++column)
                image.features.descriptors[row * 128 + column] =
                    prototype[column] + noise(random);
        }
    }

    RetrievalOptions options;
    options.top_k = 2;
    options.max_descriptors_per_image = 300;
    options.sample_grid = 3;
    options.vocabulary.branching = 2;
    options.vocabulary.depth = 4;
    options.vocabulary.max_iterations = 8;
    options.vocabulary.max_training_descriptors = 2000;
    const auto pairs = retrieve_image_pairs(images, options);
    const auto contains = [&](const Index first, const Index second) {
        return std::any_of(
            pairs.begin(), pairs.end(),
            [&](const RetrievedPair& pair) {
                return pair.first == first && pair.second == second;
            });
    };
    expect(contains(0, 1), "retrieval links first visual cluster");
    expect(contains(2, 3), "retrieval links second visual cluster");
    expect(contains(4, 5), "retrieval links third visual cluster");
}

void test_local_ba_boundary_selection() {
    Scene scene;
    scene.cameras.assign(3, cam());
    scene.images.resize(3);
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        scene.images[image_id].id = image_id;
        scene.images[image_id].camera_id = image_id;
        scene.images[image_id].registered = true;
        scene.images[image_id].pose.C =
            Vec3(0.5 * static_cast<double>(image_id), 0.0, 0.0);
    }
    const auto add_observation = [&](Track& track, const Index image_id,
                                     const Vec3& true_point) {
        Image& image = scene.images[image_id];
        const Vec2 pixel = scene.cameras[image_id].project(
            image.pose.transform_world_to_camera(true_point));
        const Index feature_id =
            static_cast<Index>(image.features.keypoints.size());
        aetherscan::features::Keypoint keypoint;
        keypoint.x = static_cast<float>(pixel.x());
        keypoint.y = static_cast<float>(pixel.y());
        image.features.keypoints.push_back(keypoint);
        track.observations.push_back({image_id, feature_id});
    };

    for (int point_id = 0; point_id < 20; ++point_id) {
        const Vec3 truth(
            -0.5 + 0.05 * point_id,
            -0.2 + 0.02 * (point_id % 5), 4.0 + 0.03 * point_id);
        Track track;
        track.position = truth + Vec3(0.01, -0.01, 0.02);
        add_observation(track, 0, truth);
        add_observation(track, 1, truth);
        track.num_inliers = 2;
        scene.tracks.push_back(std::move(track));
    }
    Track boundary_only;
    const Vec3 boundary_truth(0.4, 0.1, 5.0);
    boundary_only.position = boundary_truth + Vec3(2.0, 0.0, 0.0);
    add_observation(boundary_only, 1, boundary_truth);
    add_observation(boundary_only, 2, boundary_truth);
    boundary_only.num_inliers = 2;
    scene.tracks.push_back(std::move(boundary_only));
    const Vec3 unchanged = scene.tracks.back().position;
    std::vector<Vec3> local_points_before;
    local_points_before.reserve(20);
    for (std::size_t point_id = 0; point_id < 20; ++point_id)
        local_points_before.push_back(scene.tracks[point_id].position);

    BundleOptions options;
    options.free_image_ids = {0};
    options.fixed_image_ids = {1, 2};
    options.optimize_all_registered = false;
    options.optimize_points = false;
    options.optimizer.maximum_iterations = 3;
    const BundleSummary summary = run_bundle_adjustment(scene, options);
    expect(summary.success, "fixed-point local BA succeeds");
    expect(summary.num_points == 20, "local BA excludes boundary-only tracks");
    bool local_points_unchanged = true;
    for (std::size_t point_id = 0; point_id < local_points_before.size(); ++point_id)
        local_points_unchanged = local_points_unchanged &&
            scene.tracks[point_id].position == local_points_before[point_id];
    expect(local_points_unchanged, "local BA honors optimize_points=false");
    expect(
        (scene.tracks.back().position - unchanged).norm() == 0.0,
        "local BA does not modify boundary-only landmarks");
}

void test_robust_triangulation_and_track_split() {
    Scene scene;
    scene.cameras.assign(6, cam());
    scene.images.resize(6);
    const Vec3 centers[] = {
        Vec3(-1.5, 0.0, 0.0), Vec3(-0.5, 0.0, 0.0), Vec3(0.5, 0.0, 0.0),
        Vec3(1.5, 0.0, 0.0), Vec3(-1.0, 1.0, 0.0), Vec3(1.0, 1.0, 0.0)};
    const Vec3 true_points[] = {
        Vec3(0.0, 0.0, 4.0),
        Vec3(0.4, 0.3, 5.5)};
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        Image& image = scene.images[image_id];
        image.id = image_id;
        image.camera_id = image_id;
        image.registered = true;
        image.pose.C = centers[image_id];
        image.features.keypoints.resize(2);
        for (Index point_id = 0; point_id < 2; ++point_id) {
            const Vec2 pixel = scene.cameras[image_id].project(
                image.pose.transform_world_to_camera(true_points[point_id]));
            image.features.keypoints[point_id].x = static_cast<float>(pixel.x());
            image.features.keypoints[point_id].y = static_cast<float>(pixel.y());
        }
    }

    // One union-find track mixes two landmarks: views 0-3 see point A, views
    // 4-5 see point B. LO-RANSAC keeps A; split recovers B.
    Track contaminated;
    for (Index image_id = 0; image_id < 4; ++image_id)
        contaminated.observations.push_back({image_id, 0});
    contaminated.observations.push_back({4, 1});
    contaminated.observations.push_back({5, 1});
    scene.tracks.push_back(std::move(contaminated));

    TriangulationOptions options;
    options.reproj_threshold_px = 2.F;
    options.min_angle_deg = 0.5F;
    options.min_observations_for_ransac = 4;
    options.ransac_iterations = 48;
    options.use_lo_ransac = true;
    options.refine_nonlinear = true;
    options.split_tracks = true;
    options.max_splits_per_track = 2;

    const unsigned triangulated = triangulate_tracks(scene, false, options);
    expect(triangulated >= 2, "robust triangulation recovers both consensus sets");
    expect(scene.tracks.size() >= 2, "contaminated track is split");

    unsigned good_tracks = 0;
    for (const Track& track : scene.tracks) {
        if (!track.is_triangulated()) continue;
        const double err0 = (track.position - true_points[0]).norm();
        const double err1 = (track.position - true_points[1]).norm();
        if (std::min(err0, err1) < 0.05) ++good_tracks;
    }
    expect(good_tracks >= 2, "split tracks recover both true 3D points");
}

void test_dirty_track_updates_are_isolated() {
    Scene scene;
    scene.cameras.assign(3, cam());
    scene.images.resize(3);
    const Vec3 centers[] = {
        Vec3(-1.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0)};
    const Vec3 points[] = {
        Vec3(0.1, 0.2, 4.0), Vec3(-0.3, 0.1, 5.0)};
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        Image& image = scene.images[image_id];
        image.id = image_id;
        image.camera_id = image_id;
        image.registered = true;
        image.pose.C = centers[image_id];
        image.features.keypoints.resize(2);
        for (Index point_id = 0; point_id < 2; ++point_id) {
            const Vec2 pixel = scene.cameras[image_id].project(
                image.pose.transform_world_to_camera(points[point_id]));
            image.features.keypoints[point_id].x =
                static_cast<float>(pixel.x());
            image.features.keypoints[point_id].y =
                static_cast<float>(pixel.y());
        }
    }
    for (Index point_id = 0; point_id < 2; ++point_id) {
        Track track;
        for (Index image_id = 0; image_id < scene.images.size(); ++image_id)
            track.observations.push_back({image_id, point_id});
        scene.tracks.push_back(std::move(track));
    }
    scene.tracks[1].position = points[1];
    scene.tracks[1].num_inliers = 3;
    const Track untouched_before = scene.tracks[1];
    const auto observations_equal = [](const Track& left, const Track& right) {
        if (left.observations.size() != right.observations.size()) return false;
        for (std::size_t i = 0; i < left.observations.size(); ++i) {
            if (left.observations[i].image_id != right.observations[i].image_id ||
                left.observations[i].feature_id !=
                    right.observations[i].feature_id)
                return false;
        }
        return true;
    };

    TriangulationOptions options;
    options.reproj_threshold_px = 1.F;
    options.min_angle_deg = 0.F;
    options.split_tracks = false;
    triangulate_tracks(
        scene, std::vector<Index>{0, 0, k_invalid}, false, options);
    expect(scene.tracks[0].is_triangulated(),
           "dirty triangulation updates the selected track");
    expect(
        scene.tracks[1].num_inliers == untouched_before.num_inliers &&
            scene.tracks[1].position == untouched_before.position &&
            observations_equal(scene.tracks[1], untouched_before),
        "dirty triangulation leaves unselected tracks unchanged");

    filter_tracks(scene, std::vector<Index>{0}, 1.F, 0.F);
    expect(
        scene.tracks[1].num_inliers == untouched_before.num_inliers &&
            scene.tracks[1].position == untouched_before.position &&
            observations_equal(scene.tracks[1], untouched_before),
        "dirty filtering leaves unselected tracks unchanged");
}

}  // namespace

int main() {
    test_pair_cycle_weighting();
    test_global_rotation_weighting();
    test_global_positioning_points_only();
    test_large_point_only_positioning_does_not_require_pairs();
    test_global_positioning_irls_downweights_bad_direction();
    test_global_positioning_preserves_fixed_initial_positions();
    test_global_positioning_rejects_empty_point_constraints();
    test_long_track_merge();
    test_robust_triangulation_and_track_split();
    test_dirty_track_updates_are_isolated();
    test_retrieval_inverted_index();
    test_local_ba_boundary_selection();

    // 4-camera ring looking at a point cloud; build pairs + tracks, star-init.
    constexpr int k_views = 4;
    constexpr int k_points = 100;
    Scene scene;
    const PinholeCamera camera = cam();
    for (int i = 0; i < k_views; ++i) {
        scene.cameras.push_back(camera);
        Image image;
        image.id = static_cast<Index>(i);
        image.camera_id = static_cast<Index>(i);
        image.features.keypoints.resize(k_points);
        // Place cameras on a circle
        const double angle = i * 2.0 * 3.14159265358979323846 / k_views;
        Pose3D pose;
        pose.C = Vec3(std::cos(angle), 0.0, std::sin(angle));
        // Look toward origin: R such that camera forward (~row2) points to -C
        const Vec3 forward = (-pose.C).normalized();
        const Vec3 right = Vec3(0, 1, 0).cross(forward).normalized();
        const Vec3 down = forward.cross(right);
        pose.R.row(0) = right.transpose();
        pose.R.row(1) = down.transpose();
        pose.R.row(2) = forward.transpose();
        image.pose = pose;
        scene.images.push_back(std::move(image));
    }

    std::mt19937 rng(2);
    std::uniform_real_distribution<double> xyz(-0.4, 0.4);
    std::vector<Vec3> world_points;
    for (int p = 0; p < k_points; ++p) {
        const Vec3 X(xyz(rng), xyz(rng), xyz(rng));
        world_points.push_back(X);
        for (int v = 0; v < k_views; ++v) {
            const Vec3 Xc = scene.images[v].pose.transform_world_to_camera(X);
            const Vec2 uv = camera.project(Xc);
            scene.images[v].features.keypoints[p].x = static_cast<float>(uv.x());
            scene.images[v].features.keypoints[p].y = static_cast<float>(uv.y());
        }
    }

    // Verified pairs between consecutive views (+ closing pair)
    for (int i = 0; i < k_views; ++i) {
        const int j = (i + 1) % k_views;
        Index a = static_cast<Index>(i);
        Index b = static_cast<Index>(j);
        if (a > b) std::swap(a, b);
        ImagePair pair(a, b);
        pair.weight_spatial = 1.F;
        // Relative pose of b w.r.t a
        pair.relative_pose = scene.images[b].pose / scene.images[a].pose;
        // Unit-scale the translation like a real essential-matrix estimate
        if (pair.relative_pose->C.norm() > 1e-9)
            pair.relative_pose->C.normalize();
        for (Index p = 0; p < k_points; ++p) pair.matches.push_back({p, p});
        scene.pairs.push_back(std::move(pair));
    }

    // Clear absolute poses — reconstruction must rediscover them
    for (auto& image : scene.images) {
        image.registered = false;
        image.pose = Pose3D::identity();
    }

    build_tracks(scene);
    expect(scene.tracks.size() == static_cast<std::size_t>(k_points), "tracks built");
    expect(
        scene.image_tracks.size() == scene.images.size() &&
            scene.image_tracks.front().size() == static_cast<std::size_t>(k_points),
        "track inverted index built");
    Scene global_scene = scene;

    StarInitConfig star;
    star.min_views = 3;
    star.max_views = 4;
    star.min_initial_tracks = 30;
    star.min_angle_deg = 0.5F;
    const bool ok = star_initialize(scene, star);
    expect(ok, "star initialize");
    expect(scene.registered_count() >= 3, "enough views registered");

    unsigned landmarks = 0;
    for (const Track& t : scene.tracks) {
        if (t.is_triangulated()) ++landmarks;
    }
    expect(landmarks >= 30, "enough landmarks after star init");

    GlobalPositioningOptions positioning;
    positioning.min_views_per_track = 3;
    positioning.max_tracks_for_positioning = 0;
    positioning.max_solver_time_sec = 30.0;
    positioning.max_irls_iterations = 4;
    ResectionConfig fallback;
    fallback.min_angle_deg = 0.5F;
    const ReconstructionSummary global =
        run_global_mapping(global_scene, {}, positioning, fallback);
    expect(global.valid, "global mapping");
    expect(global.registered_views == k_views, "global mapping registers every view");
    expect(global.landmarks >= 30, "global mapping reconstructs enough landmarks");
    expect(
        global.reprojection_observations > 0 &&
            std::isfinite(global.mean_reprojection_error_pixels) &&
            std::isfinite(global.rms_reprojection_error_pixels) &&
            global.rms_reprojection_error_pixels >=
                global.mean_reprojection_error_pixels,
        "global mapping reports reprojection statistics");

    if (failures == 0) {
        std::cout << "sfm mapping tests passed\n";
        return 0;
    }
    std::cerr << failures << " failures\n";
    return 1;
}
