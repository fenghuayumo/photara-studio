#include "sfm/star_init.hpp"
#include "sfm/bundle.hpp"
#include "sfm/global_positioning.hpp"
#include "sfm/positioning_control.hpp"
#include "../src/sfm/positioning_validation.hpp"
#include "../src/sfm/resection_validation.hpp"
#include "sfm/global_rotation.hpp"
#include "sfm/pair_weighting.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/retrieval.hpp"
#include "sfm/resection.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"
#include "sfm/appearance.hpp"
#include "io/image.hpp"
#include "parallel/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <chrono>

namespace {

using namespace photara::sfm;

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
    // This test exercises the optional composite-weight path explicitly;
    // production defaults use robust inlier-count weighting.
    options.use_pair_weights = true;
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

void test_global_rotation_dense_graph_iterative_solve() {
    constexpr Index k_views = 76;
    Scene scene;
    scene.images.resize(k_views);
    std::vector<Mat3> rotations(k_views);
    for (Index image_id = 0; image_id < k_views; ++image_id) {
        scene.images[image_id].id = image_id;
        rotations[image_id] =
            axis_rotation(Vec3::UnitY(), 360.0 * image_id / k_views) *
            axis_rotation(Vec3::UnitX(), 2.0 * std::sin(0.3 * image_id));
    }

    for (Index first = 0; first < k_views; ++first) {
        for (Index offset = 1; offset <= 8; ++offset) {
            const Index second = first + offset;
            if (second >= k_views) break;
            Mat3 relative =
                rotations[second] * rotations[first].transpose();
            if ((first * 11 + second) % 53 == 0)
                relative =
                    axis_rotation(Vec3::UnitZ(), 18.0) * relative;
            ImagePair pair(first, second);
            pair.relative_pose = Pose3D{relative, Vec3::Zero()};
            pair.weight_spatial =
                0.5F + 0.5F * static_cast<float>((first + second) % 7) / 6.F;
            pair.matches.resize(200 + (first * 17 + second) % 800);
            scene.pairs.push_back(std::move(pair));
        }
    }

    // A corrupt checkpoint must be rejected before it reaches graph storage.
    ImagePair invalid(0, k_views + 5);
    invalid.relative_pose = Pose3D{Mat3::Identity(), Vec3::Zero()};
    invalid.weight_spatial = 1.F;
    invalid.matches.resize(500);
    scene.pairs.push_back(std::move(invalid));

    GlobalRotationOptions options;
    options.max_relative_rotation_error_deg = 0.0;
    const GlobalRotationSummary summary =
        estimate_global_rotations(scene, options);
    expect(summary.success, "dense global rotation graph solves iteratively");
    expect(summary.estimated_images == k_views,
           "dense global rotation graph estimates every image");
    expect(summary.used_pairs > 500,
           "dense global rotation graph retains broad connectivity");

    double mean_error = 0.0;
    for (Index image_id = 1; image_id < k_views; ++image_id) {
        mean_error += rotation_error(
            scene.images[image_id].pose.R *
                scene.images[image_id - 1].pose.R.transpose(),
            rotations[image_id] * rotations[image_id - 1].transpose());
    }
    mean_error /= static_cast<double>(k_views - 1);
    expect(mean_error < 0.03,
           "dense global rotation graph remains accurate with outliers");
}

void test_warm_start_geometry_acceptance() {
    Scene scene;
    scene.cameras.push_back(cam());
    constexpr Index views = 6, points = 48;
    for (Index view = 0; view < views; ++view) {
        Image image;
        image.id = view;
        image.camera_id = 0;
        image.registered = true;
        image.pose.C = Vec3(0.6 * view, 0.1 * (view % 2), 0);
        image.features.keypoints.resize(points);
        scene.images.push_back(std::move(image));
    }
    for (Index point = 0; point < points; ++point) {
        Track track;
        const Vec3 position(0.2 + 0.04 * point, 0.1 * (point % 5), 5 + 0.05 * point);
        for (Index view = 0; view < views; ++view) {
            const Vec2 pixel = scene.cameras[0].project(position - scene.images[view].pose.C);
            auto& feature = scene.images[view].features.keypoints[point];
            feature.x = static_cast<float>(pixel.x());
            feature.y = static_cast<float>(pixel.y());
            track.observations.push_back({view, point});
        }
        scene.tracks.push_back(std::move(track));
    }
    for (Index view = 0; view + 1 < views; ++view) {
        ImagePair pair(view, view + 1);
        pair.matches.resize(40);
        pair.weight_spatial = 1;
        pair.relative_pose = Pose3D{Mat3::Identity(),
            scene.images[view + 1].pose.C - scene.images[view].pose.C};
        scene.pairs.push_back(std::move(pair));
    }
    const std::vector<Index> optimized{0, 1, 2, 3, 4, 5, 6, 7};
    const auto baseline = detail::evaluate_warm_start_geometry(scene, optimized);
    expect(baseline.heldout_tracks == 40 && baseline.positive_observations == 240 &&
           baseline.camera_heldout_observations.front() == 40 &&
           baseline.camera_heldout_observations.back() == 40,
           "validation excludes optimized tracks and scores anchors using alternate fitted observations");
    expect(baseline.pair_p90_radians < 1e-6 && baseline.heldout_p90_radians < 1e-5,
           "exact geometry has low independent direction and heldout errors");

    Scene gauge = scene;
    for (auto& image : gauge.images) image.pose.C = 7 * image.pose.C + Vec3(3, -4, 2);
    for (auto& track : gauge.tracks) track.position.setConstant(std::numeric_limits<double>::quiet_NaN());
    const auto gauge_quality = detail::evaluate_warm_start_geometry(gauge, optimized);
    expect(!detail::warm_start_geometry_rejection(baseline, gauge_quality),
           "acceptance is scale/translation invariant and ignores optimized point coordinates");

    Scene noisy_initialization = scene;
    noisy_initialization.images[3].pose.C.y() += 0.4;
    for (auto& pair : noisy_initialization.pairs)
        pair.relative_pose = Pose3D{Mat3::Identity(),
            noisy_initialization.images[pair.id2].pose.C - noisy_initialization.images[pair.id1].pose.C};
    Scene corrected = noisy_initialization;
    corrected.images[3].pose.C = scene.images[3].pose.C;
    const auto noisy_quality = detail::evaluate_warm_start_geometry(noisy_initialization, optimized);
    const auto corrected_quality = detail::evaluate_warm_start_geometry(corrected, optimized);
    expect(corrected_quality.pair_p90_radians > noisy_quality.pair_p90_radians + 0.01 &&
           corrected_quality.heldout_p90_radians < noisy_quality.heldout_p90_radians &&
           !detail::warm_start_geometry_rejection(noisy_quality, corrected_quality),
           "independent per-camera ray evidence can accept correction of noisy two-view translations");
    Scene sparse_noisy = noisy_initialization, sparse_corrected = corrected;
    for (auto& feature : sparse_noisy.images[3].features.keypoints)
        feature.x = std::numeric_limits<float>::quiet_NaN();
    sparse_corrected.images[3].features = sparse_noisy.images[3].features;
    const auto sparse_before = detail::evaluate_warm_start_geometry(sparse_noisy, optimized);
    const auto sparse_after = detail::evaluate_warm_start_geometry(sparse_corrected, optimized);
    expect(!detail::warm_start_geometry_rejection(sparse_before, sparse_after) &&
           detail::warm_start_unverified_direction_changes(sparse_before, sparse_after) > 0,
           "missing per-camera evidence is reported as uncertainty instead of a false geometric regression");

    Scene drift = scene;
    drift.images[3].pose.C.y() += 4;
    const auto drift_quality = detail::evaluate_warm_start_geometry(drift, optimized);
    expect(detail::warm_start_geometry_rejection(baseline, drift_quality) != nullptr,
           "camera drift is rejected despite an otherwise usable solver result");
    drift.pairs.clear();
    Scene no_pairs = scene;
    no_pairs.pairs.clear();
    const auto heldout_baseline = detail::evaluate_warm_start_geometry(no_pairs, optimized);
    auto pixel_regression = heldout_baseline;
    pixel_regression.heldout_p90_radians += 2.5*heldout_baseline.pixel_noise_radians;
    expect(detail::warm_start_geometry_rejection(heldout_baseline, pixel_regression) != nullptr,
           "held-out degradation above pixel noise cannot hide under a half-degree floor");
    auto broad_before = heldout_baseline, broad_after = heldout_baseline;
    broad_before.pair_constraints = broad_after.pair_constraints = 100;
    broad_before.pair_p90_radians = 0.04;
    broad_after.pair_p90_radians = 0.08;
    broad_before.heldout_p90_radians = 0.01;
    broad_after.heldout_p90_radians = 0.009;
    broad_after.layout_change_p90_baselines = 2;
    expect(detail::warm_start_geometry_rejection(broad_before, broad_after) != nullptr,
           "small point-fit gains cannot conceal a broad translation graph regression");
    broad_after.layout_change_p90_baselines = 0.1;
    expect(!detail::warm_start_geometry_rejection(broad_before, broad_after),
           "small layout refinements may tolerate noisy relative directions");
    broad_after.layout_change_p90_baselines = 2;
    broad_before.heldout_p90_radians = 0.05;
    broad_after.heldout_p90_radians = 0.005;
    expect(!detail::warm_start_geometry_rejection(broad_before, broad_after),
           "strong independent point correction may override noisy relative translations");
    const auto heldout_drift = detail::evaluate_warm_start_geometry(drift, optimized);
    expect(detail::warm_start_geometry_rejection(heldout_baseline, heldout_drift) != nullptr,
           "heldout rays reject drift independently of relative translation constraints");
    for (auto& image : no_pairs.images) image.pose.C = -image.pose.C;
    const auto reversed = detail::evaluate_warm_start_geometry(no_pairs, optimized);
    expect(reversed.heldout_observations == heldout_baseline.heldout_observations &&
           reversed.positive_observations == 0 &&
           detail::warm_start_geometry_rejection(heldout_baseline, reversed) != nullptr,
           "behind-camera predictions remain in validation and trigger rejection");
    Scene collapsed = scene;
    collapsed.images[1].pose.C = collapsed.images[0].pose.C;
    expect(detail::warm_start_geometry_rejection(baseline,
               detail::evaluate_warm_start_geometry(collapsed, optimized)) != nullptr,
           "collapsed camera baselines are rejected");
    collapsed.pairs.clear();
    for (auto& image : collapsed.images) image.pose.C.setZero();
    expect(detail::warm_start_geometry_rejection(heldout_baseline,
               detail::evaluate_warm_start_geometry(collapsed, optimized)) != nullptr,
           "collapsed camera geometry cannot pass by omitting pair evidence");
    Scene bulk = scene;
    bulk.pairs.clear();
    // Many correct cameras dilute one bad view in whole-scene P90. Retain the
    // usable bulk initializer, but explicitly nominate the local discrepancy
    // for the final geometry audit rather than restarting every camera.
    for (Index repeat = 1; repeat < 5; ++repeat)
        for (Index view = 0; view < views; ++view) {
            Image copy = scene.images[view];
            copy.id = static_cast<Index>(bulk.images.size());
            bulk.images.push_back(std::move(copy));
            for (Index point = 0; point < points; ++point)
                bulk.tracks[point].observations.push_back({bulk.images.back().id, point});
        }
    const auto bulk_before = detail::evaluate_warm_start_geometry(bulk, optimized);
    bulk.images[3].pose.C.y() += 4;
    const auto bulk_after = detail::evaluate_warm_start_geometry(bulk, optimized);
    const auto local_risks = detail::warm_start_local_geometry_risks(bulk_before, bulk_after);
    expect(!detail::warm_start_geometry_rejection(bulk_before, bulk_after) &&
           std::find(local_risks.begin(), local_risks.end(), 3) != local_risks.end(),
           "local risk is retained without discarding independently healthy bulk geometry");
    scene.images[0].pose.C.x() = std::numeric_limits<double>::quiet_NaN();
    expect(detail::warm_start_geometry_rejection(baseline,
               detail::evaluate_warm_start_geometry(scene, optimized)) != nullptr,
           "nonfinite camera centers are rejected");
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

    {
        Scene limited = scene;
        GlobalPositioningOptions options;
        options.backend = PositioningBackend::cpu;
        options.prefer_cuda = false;
        options.max_solver_time_sec = 0.001;
        const auto stopped = solve_global_positions(limited, options);
        expect(!stopped.success && stopped.termination == GlobalPositioningTermination::time_limit &&
               stopped.cpu_accepted_updates == 0 && stopped.irls_iterations == 0,
               "exhausted CPU budget cannot report a usable solve or IRLS convergence");
        for (std::size_t i = 0; i < scene.images.size(); ++i)
            expect(limited.images[i].pose.C == scene.images[i].pose.C,
                   "failed budget-limited fallback restores the original camera centers");
        for (std::size_t i = 0; i < scene.tracks.size(); ++i)
            expect(limited.tracks[i].position == scene.tracks[i].position &&
                   limited.tracks[i].num_inliers == scene.tracks[i].num_inliers,
                   "failed positioning restores both points and inlier metadata");
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

void test_resection_candidate_failures() {
    const auto fixture = [] {
        Scene scene;
        scene.thread_count = 1;
        scene.cameras.push_back(cam());
        scene.images.resize(5);
        for (Index id = 0; id < 5; ++id) {
            auto& image = scene.images[id];
            image.id = id; image.camera_id = 0; image.registered = id < 3;
            image.pose.C = Vec3(0.4*id,0.1*(id%2),0);
        }
        std::mt19937 rng(9721);
        std::uniform_real_distribution<double> x(-1.5,1.5), y(-1,1), z(4,7);
        for (Index t = 0; t < 200; ++t) {
            Track track;
            track.position = Vec3(x(rng),y(rng),z(rng));
            for (Index id = 0; id < 3; ++id) {
                const auto pixel = scene.cameras[0].project(scene.images[id].pose.transform_world_to_camera(track.position));
                photara::features::Keypoint kp;
                kp.x = static_cast<float>(pixel.x()); kp.y = static_cast<float>(pixel.y());
                scene.images[id].features.keypoints.push_back(kp);
                track.observations.push_back({id,t});
            }
            track.num_inliers = 3;
            scene.tracks.push_back(track);
        }
        return scene;
    };
    const auto options = [] {
        ResectionConfig config;
        config.local_ba_every = 0;
        config.full_ba_every = {0,0,0};
        config.final_ba_additional_iterations = 0;
        config.full_ba.maximum_iterations = 3;
        config.full_ba.optimize_focal = false;
        config.full_ba.optimize_aspect_ratio = false;
        config.full_ba.optimize_distortion = false;
        config.ransac.max_iterations = 500;
        return config;
    };
    {
        Scene source = fixture();
        const auto truth = source.images[3].pose;
        for (Index f = 0; f < 30; ++f) {
            const auto pixel = source.cameras[0].project(truth.transform_world_to_camera(source.tracks[f].position));
            photara::features::Keypoint kp;
            kp.x = static_cast<float>(pixel.x()); kp.y = static_cast<float>(pixel.y());
            source.images[3].features.keypoints.push_back(kp);
        }
        // Two anchors propose a contested landmark first; a third supplies
        // the valid alternative for each feature. No direct target track IDs.
        for (Index anchor = 0; anchor < 3; ++anchor) {
            ImagePair pair(anchor,3);
            for (Index f = 0; f < 30; ++f) pair.matches.push_back({anchor < 2 ? 0U : f,f});
            source.pairs.push_back(pair);
        }
        auto legacy = source;
        auto config = options();
        config.preserve_conflict_alternatives = false;
        register_images(legacy,config);
        expect(!legacy.images[3].registered,"legacy conflict claims lose valid alternate landmark links");
        config.preserve_conflict_alternatives = true;
        register_images(source,config);
        expect(source.images[3].registered && (source.images[3].pose.C-truth.C).norm() < 1e-3,
            "conflicting landmark proposals cannot consume a feature needed by a correct pose");
    }
    {
        Scene source = fixture();
        const auto truth = source.images[4].pose;
        for (Index t = 0; t < 200; ++t) {
            photara::features::Keypoint bad;
            bad.x = 320; bad.y = 240;
            source.images[3].features.keypoints.push_back(bad);
            source.tracks[t].observations.push_back({3,t});
            if (t >= 30) continue;
            const auto pixel = source.cameras[0].project(truth.transform_world_to_camera(source.tracks[t].position));
            photara::features::Keypoint good;
            good.x = static_cast<float>(pixel.x()); good.y = static_cast<float>(pixel.y());
            source.images[4].features.keypoints.push_back(good);
            source.tracks[t].observations.push_back({4,t});
        }
        auto legacy = source;
        auto config = options();
        config.retry_lower_support_candidates = false;
        register_images(legacy,config);
        expect(!legacy.images[4].registered,"failed high-count band reproduces starvation of valid weaker candidates");
        config.retry_lower_support_candidates = true;
        register_images(source,config);
        expect(source.images[4].registered && (source.images[4].pose.C-truth.C).norm() < 1e-3,
            "band exhaustion must try a geometrically valid lower-count camera");
        expect(!source.images[3].registered,"fallback must still reject the unsupported high-count camera");
    }
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
        photara::features::Keypoint keypoint;
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
        add_observation(track, 2, truth);
        track.num_inliers = 3;
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
    Scene global_scene = scene;

    BundleOptions options;
    options.free_image_ids = {0};
    options.fixed_image_ids = {1, 2};
    options.optimize_all_registered = false;
    options.optimize_points = false;
    options.optimizer.maximum_iterations = 3;
    // Fixed boundary cameras are intentionally unsupported by the CUDA BA
    // backend and must retain the exact CPU local-BA semantics.
    options.cuda_min_observations = 0;
    const BundleSummary summary = run_bundle_adjustment(scene, options);
    expect(summary.success, "fixed-point local BA succeeds");
    expect(summary.backend == BundleBackend::cpu,
           "fixed-boundary local BA stays on CPU");
    expect(summary.num_points == 20, "local BA excludes boundary-only tracks");
    bool local_points_unchanged = true;
    for (std::size_t point_id = 0; point_id < local_points_before.size(); ++point_id)
        local_points_unchanged = local_points_unchanged &&
            scene.tracks[point_id].position == local_points_before[point_id];
    expect(local_points_unchanged, "local BA honors optimize_points=false");
    expect(
        (scene.tracks.back().position - unchanged).norm() == 0.0,
        "local BA does not modify boundary-only landmarks");

    BundleOptions global_options;
    global_options.optimizer.maximum_iterations = 5;
    global_options.prefer_cuda = true;
    global_options.cuda_min_observations = 0;
    const BundleSummary global_summary =
        run_bundle_adjustment(global_scene, global_options);
    expect(global_summary.success, "automatic BA backend succeeds");
    BundleBackend expected_backend = BundleBackend::cpu;
#if defined(PHOTARA_HAS_CUDA)
    if (photara::ba::CudaOptimizer::is_available())
        expected_backend = BundleBackend::cuda;
#endif
#if defined(PHOTARA_HAS_VULKAN_BA)
    if (expected_backend == BundleBackend::cpu &&
        photara::ba::VulkanOptimizer::is_available())
        expected_backend = BundleBackend::vulkan;
#endif
    expect(global_summary.backend == expected_backend,
           "automatic BA prefers CUDA, then Vulkan, then CPU");
}

void test_final_registration_audit_prunes_unsupported_pose() {
    Scene scene;
    scene.cameras.push_back(cam());
    scene.images.resize(3);
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        Image& image = scene.images[image_id];
        image.id = image_id;
        image.camera_id = 0;
        image.registered = true;
        image.path = std::filesystem::path(
            "audit_" + std::to_string(image_id) + ".png");
        image.features.keypoints.resize(3);
    }
    for (Index point_id = 0; point_id < 3; ++point_id) {
        Track track;
        track.position = Vec3(0.05 * point_id, 0.02 * point_id, 4.0);
        for (Index image_id = 0; image_id < 2; ++image_id) {
            const Vec2 pixel = scene.cameras[0].project(track.position);
            scene.images[image_id].features.keypoints[point_id].x =
                static_cast<float>(pixel.x());
            scene.images[image_id].features.keypoints[point_id].y =
                static_cast<float>(pixel.y());
            track.observations.push_back({image_id, point_id});
        }
        if (point_id == 0) {
            const Vec2 pixel = scene.cameras[0].project(track.position);
            scene.images[2].features.keypoints[point_id].x =
                static_cast<float>(pixel.x());
            scene.images[2].features.keypoints[point_id].y =
                static_cast<float>(pixel.y());
            track.observations.push_back({2, point_id});
        }
        track.num_inliers = static_cast<std::uint8_t>(
            track.observations.size());
        scene.tracks.push_back(std::move(track));
    }
    rebuild_track_index(scene);
    expect(
        prune_unsupported_registrations(scene, 2, 1.0) == 1,
        "final registration audit invalidates weak pose");
    expect(
        scene.images[0].registered && scene.images[1].registered &&
            !scene.images[2].registered,
        "final registration audit keeps supported poses");
    expect(
        std::none_of(
            scene.tracks.front().observations.begin(),
            scene.tracks.front().observations.end(),
            [](const Observation& observation) {
                return observation.image_id == 2;
            }),
        "final registration audit removes weak observations");
}

void test_full_sphere_registration_audit() {
    Scene scene;
    auto camera = cam();
    camera.model = photara::CameraModel::equirectangular;
    camera.width = 2048;
    camera.height = 1024;
    camera.fx = camera.fy = 2048.0 / (2.0 * std::numbers::pi);
    camera.cx = 1024.0;
    camera.cy = 512.0;
    scene.cameras.push_back(camera);
    scene.images.resize(2);
    for (Index id = 0; id < 2; ++id) {
        scene.images[id].id = id;
        scene.images[id].camera_id = 0;
        scene.images[id].registered = true;
        scene.images[id].features.keypoints.resize(3);
    }
    // Rear hemisphere, seam crossing and a near-pole observation.
    const std::vector<Vec3> points = {
        Vec3(0.001, 0.0, -4.0), Vec3(-1.0, 0.5, -4.0), Vec3(0.01, 4.0, -0.01)};
    for (Index id = 0; id < points.size(); ++id) {
        Track track;
        track.position = points[id];
        Vec2 pixel = camera.project(points[id]);
        if (id == 0) pixel.x() -= camera.width;  // Same ray across the seam.
        for (Index image = 0; image < 2; ++image) {
            const Vec2 observed = image == 0 ? pixel : camera.project(-points[id]);
            auto& feature = scene.images[image].features.keypoints[id];
            feature.x = static_cast<float>(observed.x());
            feature.y = static_cast<float>(observed.y());
            track.observations.push_back({image, id});
        }
        track.num_inliers = 2;
        scene.tracks.push_back(std::move(track));
        expect(camera.checked_reprojection_error(points[id], pixel) < 1e-8,
            "full-sphere validation accepts rear, seam and pole observations");
        expect(!std::isfinite(camera.checked_reprojection_error(
            points[id], camera.project(-points[id]))),
            "full-sphere validation rejects antipodal tangent minima");
    }
    rebuild_track_index(scene);
    expect(prune_unsupported_registrations(scene, 3, 1.0) == 1,
        "full-sphere audit rejects only the antipodal registration");
    expect(scene.images[0].registered && !scene.images[1].registered,
        "full-sphere audit retains a supported rear-facing view");
}

void test_alignment_observability_reports_bridge_branch() {
    Scene scene;
    scene.images.resize(7);
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        scene.images[image_id].id = image_id;
        scene.images[image_id].registered = true;
    }
    const auto add_pair = [&](const Index first, const Index second) {
        ImagePair pair(first, second);
        pair.relative_pose = Pose3D::identity();
        scene.pairs.push_back(std::move(pair));
    };
    add_pair(0, 1); add_pair(1, 2); add_pair(2, 3);
    add_pair(3, 0); add_pair(0, 2);
    add_pair(3, 4);
    add_pair(4, 5); add_pair(5, 6); add_pair(6, 4);
    const AlignmentObservability audit =
        analyze_alignment_observability(scene);
    expect(audit.bridge_edges == 1,
           "alignment observability finds scale bridge");
    expect(audit.reliable_views == 4 && audit.unreliable_views == 3,
           "alignment observability retains largest rigid block");

    Track cross_track;
    cross_track.position = Vec3(0.0, 0.0, 4.0);
    cross_track.observations = {{2, 0}, {4, 0}, {5, 0}};
    cross_track.num_inliers = 3;
    scene.tracks.push_back(std::move(cross_track));
    const AlignmentObservability one_point =
        analyze_alignment_observability(scene);
    expect(one_point.bridge_edges == 1 && one_point.reliable_views == 4,
           "one shared point does not constrain branch scale");

    Track cross_track_2;
    cross_track_2.position = Vec3(1.0, 0.0, 4.0);
    cross_track_2.observations = {{1, 0}, {5, 0}};
    cross_track_2.num_inliers = 2;
    scene.tracks.push_back(std::move(cross_track_2));
    Track cross_track_3;
    cross_track_3.position = Vec3(0.0, 1.0, 4.0);
    cross_track_3.observations = {{0, 0}, {6, 0}};
    cross_track_3.num_inliers = 2;
    scene.tracks.push_back(std::move(cross_track_3));
    expect(analyze_alignment_observability(scene).unreliable_views == 3,
           "jointly triangulated points cannot certify independent block depth");
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id)
        scene.images[image_id].pose.C = Vec3(static_cast<double>(image_id), 0.0, 0.0);
    for (Track& track : scene.tracks) {
        track.observations = {{1, 0}, {2, 0}, {4, 0}, {5, 0}};
        track.num_inliers = 4;
    }
    const AlignmentObservability anchored =
        analyze_alignment_observability(scene);
    expect(anchored.bridge_edges == 0 && anchored.reliable_views == 7,
           "three non-collinear shared points make branch scale observable");
    for (Index image_id=0;image_id<scene.images.size();++image_id)
        scene.images[image_id].pose.C=Vec3(1e-5*image_id,0,0);
    expect(analyze_alignment_observability(scene).unreliable_views==3,
           "numerically distinct rays without useful parallax cannot certify branch depth");
    for (Index image_id=0;image_id<scene.images.size();++image_id)
        scene.images[image_id].pose.C=Vec3(static_cast<double>(image_id),0,0);
    scene.tracks.back().position = Vec3(2.0, 0.0, 4.0);
    expect(analyze_alignment_observability(scene).unreliable_views == 3,
           "collinear shared landmarks cannot certify a similarity");
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

void test_fast_split_gate() {
    Scene scene;
    scene.cameras.assign(6, cam());
    scene.images.resize(6);
    const Vec3 first_point(0.0, 0.0, 5.0);
    const Vec3 second_point(0.8, 0.4, 6.0);
    const Vec3 unrelated_points[] = {
        Vec3(-1.5, -1.2, 3.0), Vec3(1.8, -0.8, 8.0),
        Vec3(-2.0, 1.3, 4.0), Vec3(2.2, 1.1, 7.0)};
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        Image& image = scene.images[image_id];
        image.id = image_id;
        image.camera_id = image_id;
        image.registered = true;
        image.pose.C = Vec3(static_cast<double>(image_id) - 2.5, 0.0, 0.0);
        image.features.keypoints.resize(3);
        const Vec3 points[] = {
            first_point, second_point,
            unrelated_points[image_id >= 2 ? image_id - 2 : 0]};
        for (Index feature_id = 0; feature_id < 3; ++feature_id) {
            const Vec2 pixel = scene.cameras[image_id].project(
                image.pose.transform_world_to_camera(points[feature_id]));
            image.features.keypoints[feature_id].x =
                static_cast<float>(pixel.x());
            image.features.keypoints[feature_id].y =
                static_cast<float>(pixel.y());
        }
    }

    const auto make_parent = [&](const Index suffix_feature) {
        Track track;
        track.position = first_point;
        track.observations.push_back({0, 0});
        track.observations.push_back({1, 0});
        track.num_inliers = 2;
        for (Index image_id = 2; image_id < 6; ++image_id)
            track.observations.push_back({image_id, suffix_feature});
        return track;
    };
    scene.tracks.push_back(make_parent(0));  // Existing point: absorb all four.
    scene.tracks.push_back(make_parent(2));  // Incoherent: reject before RANSAC.
    scene.tracks.push_back(make_parent(1));  // Coherent second point: split.

    TriangulationOptions options;
    options.reproj_threshold_px = 0.25F;
    options.min_angle_deg = 0.5F;
    options.min_observations_for_ransac = 4;
    options.ransac_iterations = 48;
    options.refine_nonlinear = false;
    options.max_splits_per_track = 1;
    Scene serial = scene;
    for (unsigned repeat = 1; repeat < 512; ++repeat)
        serial.tracks.insert(serial.tracks.end(), scene.tracks.begin(), scene.tracks.end());
    Scene parallel = serial;
    serial.thread_count = 1;
    parallel.thread_count = 4;
    triangulate_tracks(serial, true, options);
    triangulate_tracks(parallel, true, options);
    expect(serial.tracks.size() == parallel.tracks.size(), "parallel split preserves child count");
    for (std::size_t t = 0; t < std::min(serial.tracks.size(), parallel.tracks.size()); ++t) {
        const auto& a = serial.tracks[t];
        const auto& b = parallel.tracks[t];
        expect(a.position == b.position && a.num_inliers == b.num_inliers &&
               a.split_generation == b.split_generation && a.observations.size() == b.observations.size(),
            "parallel split preserves ordered geometry exactly");
        for (std::size_t o = 0; o < std::min(a.observations.size(), b.observations.size()); ++o)
            expect(a.observations[o].image_id == b.observations[o].image_id &&
                   a.observations[o].feature_id == b.observations[o].feature_id,
                "parallel split preserves observation order and child IDs");
    }
    triangulate_tracks(scene, true, options);

    expect(
        scene.tracks[0].num_inliers == 6 && scene.tracks[0].observations.size() == 6,
        "fast split gate absorbs observations supported by the parent point");
    expect(
        scene.tracks[1].num_inliers == 2 && scene.tracks[1].observations.size() == 6,
        "fast split gate rejects incoherent registered outliers");
    expect(
        scene.tracks.size() == 4,
        "fast split gate launches consensus only for the coherent candidate");
    bool recovered_second_point = false;
    for (std::size_t track_id = 3; track_id < scene.tracks.size(); ++track_id)
        recovered_second_point = recovered_second_point ||
            (scene.tracks[track_id].position - second_point).norm() < 0.05;
    expect(
        recovered_second_point,
        "fast split gate preserves a real alternate geometric consensus");
}

void expect_same_split_scene(const Scene& a, const Scene& b) {
    expect(a.tracks.size() == b.tracks.size(), "recursive split preserves track count");
    for (std::size_t i = 0; i < std::min(a.tracks.size(), b.tracks.size()); ++i) {
        const auto& x = a.tracks[i];
        const auto& y = b.tracks[i];
        expect(x.position == y.position && x.num_inliers == y.num_inliers &&
               x.split_generation == y.split_generation && x.color_r == y.color_r &&
               x.color_g == y.color_g && x.color_b == y.color_b && x.has_color == y.has_color,
               "recursive split preserves ordered geometry and metadata exactly");
        expect(x.observations.size() == y.observations.size(), "recursive split preserves observation count");
        for (std::size_t o = 0; o < std::min(x.observations.size(), y.observations.size()); ++o)
            expect(x.observations[o].image_id == y.observations[o].image_id &&
                   x.observations[o].feature_id == y.observations[o].feature_id,
                   "recursive split preserves ordered child observations");
    }
    expect(a.image_tracks.size() == b.image_tracks.size(), "recursive split preserves image index size");
    for (std::size_t i = 0; i < std::min(a.image_tracks.size(), b.image_tracks.size()); ++i) {
        expect(a.image_tracks[i].size() == b.image_tracks[i].size(), "recursive split preserves image index count");
        for (std::size_t j = 0; j < std::min(a.image_tracks[i].size(), b.image_tracks[i].size()); ++j)
            expect(a.image_tracks[i][j].track_id == b.image_tracks[i][j].track_id &&
                   a.image_tracks[i][j].feature_id == b.image_tracks[i][j].feature_id,
                   "recursive split preserves global child IDs in image index");
    }
}

void test_parallel_recursive_track_split() {
    constexpr Index roots = 160;
    Scene input;
    input.cameras.assign(13, cam());
    input.images.resize(13);
    const Vec3 points[] = {Vec3(0, 0, 5), Vec3(0.8, 0.4, 6), Vec3(-0.9, -0.6, 4)};
    for (Index i = 0; i < input.images.size(); ++i) {
        auto& image = input.images[i];
        image.id = image.camera_id = i;
        image.registered = i < 12;
        image.pose.C = Vec3(double(i) * 0.6 - 3, double(i % 3) * 0.2, 0);
        image.features.keypoints.resize(roots * 3);
        for (Index root = 0; root < roots; ++root)
            for (Index point = 0; point < 3; ++point) {
                const Vec3 shifted = points[point] + Vec3(root * 0.001, 0, 0);
                const Vec2 pixel = input.cameras[i].project(image.pose.transform_world_to_camera(shifted));
                auto& keypoint = image.features.keypoints[root * 3 + point];
                keypoint.x = static_cast<float>(pixel.x());
                keypoint.y = static_cast<float>(pixel.y());
            }
    }
    for (Index root = 0; root < roots; ++root) {
        Track track;
        track.position = points[0] + Vec3(root * 0.001, 0, 0);
        track.num_inliers = 6;
        track.has_color = true;
        track.color_r = static_cast<std::uint8_t>(root);
        for (Index image = 0; image < 12; ++image) {
            const Index point = image < 6 ? 0 : image < 10 ? 1 : 2;
            track.observations.push_back({image, root * 3 + point});
        }
        track.observations.push_back({12, root * 3}); // Retained unregistered observation.
        input.tracks.push_back(std::move(track));
    }
    for (bool fast_gate : {false, true}) {
        for (unsigned depth : {1U, 2U}) {
            for (bool selected : {false, true}) {
                TriangulationOptions options;
                options.reproj_threshold_px = 0.25F;
                options.min_angle_deg = 0.5F;
                options.ransac_iterations = 48;
                options.refine_nonlinear = false;
                options.max_splits_per_track = depth;
                options.use_fast_split_gate = fast_gate;
                std::vector<Index> ids;
                for (Index i = roots; i-- > 0;)
                    if (i % 2 == 0) { ids.push_back(i); ids.push_back(i); }
                ids.push_back(k_invalid);
                const auto run = [&](Scene& scene) {
                    return selected ? triangulate_tracks(scene, ids, true, options)
                                    : triangulate_tracks(scene, true, options);
                };
                Scene serial = input;
                serial.thread_count = 1;
                const auto expected = run(serial);
                const auto processed = selected ? roots / 2 : roots;
                expect(serial.tracks.size() == roots + processed * depth,
                       "recursive split exercises the requested depth");
                for (unsigned threads : {2U, 4U}) {
                    Scene threaded = input;
                    threaded.thread_count = threads;
                    expect(run(threaded) == expected, "parallel split preserves return count");
                    expect_same_split_scene(serial, threaded);
                    run(serial);
                    run(threaded);
                    expect_same_split_scene(serial, threaded);
                }
                Scene nested = input;
                nested.thread_count = 4;
                {
                    const photara::parallel::ScopedParallelRegion region(true);
                    run(nested);
                }
                expect_same_split_scene(serial, nested);
            }
        }
    }
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

void test_parallel_track_colours() {
    const auto directory=std::filesystem::temp_directory_path() /
        ("photara-colours-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    Scene scene;
    scene.images.resize(13);
    scene.tracks.resize(3);
    for (Index i=0;i<13;++i) {
        auto& image=scene.images[i];
        image.path=directory/(std::to_string(i)+".png");
        image.features.image_width=8;
        image.features.image_height=8;
        image.features.keypoints.resize(2);
        image.features.keypoints[0].x=1.5F;
        image.features.keypoints[0].y=2.F;
        image.features.keypoints[1].x=-10.F;
        if (i<12) {
            photara::io::RgbImage rgb{16,16,std::vector<std::uint8_t>(16*16*3)};
            for (unsigned y=0;y<16;++y)
                for (unsigned x=0;x<16;++x) {
                    rgb.pixels[(y*16+x)*3]=static_cast<std::uint8_t>(i*10);
                    rgb.pixels[(y*16+x)*3+1]=static_cast<std::uint8_t>(x);
                    rgb.pixels[(y*16+x)*3+2]=static_cast<std::uint8_t>(y);
                }
            photara::io::save_rgb_png(rgb,image.path);
        }
        scene.tracks[0].observations.push_back({i,0});
        scene.tracks[1].observations.push_back({i,1});
        scene.tracks[2].observations.push_back({i,99});
    }
    for (auto& track : scene.tracks) track.num_inliers=13;
    Scene serial=scene;
    serial.thread_count=1;
    scene.thread_count=8;
    expect(colour_triangulated_tracks(serial)==1,"serial colour valid samples only");
    expect(colour_triangulated_tracks(scene)==1,"parallel colour valid samples only");
    expect(scene.tracks[0].color_r==55 && scene.tracks[0].color_g==3 && scene.tracks[0].color_b==4,
        "colour averages all loaded photos at scaled keypoints");
    for (std::size_t i=0;i<scene.tracks.size();++i) {
        const auto& a=scene.tracks[i];
        const auto& b=serial.tracks[i];
        expect(a.has_color==b.has_color && a.color_r==b.color_r && a.color_g==b.color_g && a.color_b==b.color_b,
            "parallel colour exactly matches serial with missing photos and invalid samples");
    }
    expect(colour_triangulated_tracks(scene)==0,"already coloured tracks remain unchanged");
    std::filesystem::remove_all(directory);
}

void test_missing_view_with_coincident_anchors() {
    const auto directory = std::filesystem::temp_directory_path() /
        ("photara-coincident-anchors-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    Scene scene;
    auto camera = cam();
    camera.width = 320; camera.height = 240;
    camera.fx = camera.fy = 250; camera.cx = 160; camera.cy = 120;
    scene.cameras.push_back(camera);
    scene.thread_count = 2;
    const Vec3 centers[] = {Vec3(-0.6,0,0), Vec3(0,0.6,0), Vec3(0.6,0,0),
                            Vec3::Zero(), Vec3::Zero(), Vec3(0.3,0.2,0)};
    std::mt19937 random(913);
    std::vector<unsigned> noise(161*121);
    for (auto& value : noise) value = 20 + random()%216;
    for (Index id = 0; id < 6; ++id) {
        Image image;
        image.id = id; image.camera_id = 0; image.registered = id < 5;
        image.pose.C = id < 5 ? centers[id] : Vec3::Zero();
        image.path = directory/(std::to_string(id)+".png");
        photara::io::GrayImage gray{320,240,std::vector<std::uint8_t>(320*240)};
        for (unsigned y = 0; y < 240; ++y) for (unsigned x = 0; x < 320; ++x) {
            const unsigned tx = x+160+int(50*centers[id].x());
            const unsigned ty = y+120+int(50*centers[id].y());
            const unsigned gx = tx/4, gy = ty/4, dx = tx%4, dy = ty%4;
            gray.pixels[y*320+x] = static_cast<std::uint8_t>(((4-dy)*((4-dx)*noise[gy*161+gx] +
                dx*noise[gy*161+gx+1]) + dy*((4-dx)*noise[(gy+1)*161+gx] + dx*noise[(gy+1)*161+gx+1]))/16);
        }
        photara::io::save_gray_png(gray,image.path);
        scene.images.push_back(std::move(image));
    }
    FrontEndOptions recovery;
    recovery.extractor = "sift"; recovery.matcher = "mutual_ratio";
    recovery.sift_contrast_threshold = 0.001;
    recovery.max_features = recovery.progressive_rescue_max_features = 4000;
    unsigned recovered = 0;
    try { recovered = recover_missing_views(scene,recovery); }
    catch (const std::exception& error) { std::cerr << "coincident anchor recovery: " << error.what() << '\n'; }
    expect(recovered == 1 && scene.images.back().registered &&
           (scene.images.back().pose.C-centers[5]).norm() < 0.02,
           "coincident anchors cannot shift batched matches onto the wrong image pairs");
    for (Index id = 0; id < 5; ++id)
        expect(scene.images[id].registered && (scene.images[id].pose.C-centers[id]).norm() == 0,
               "descriptor recovery preserves every frozen anchor including duplicate centers");
    std::filesystem::remove_all(directory);
}

}  // namespace

void test_intrinsic_group_units() {
    Scene scene;
    scene.cameras={cam()};
    scene.cameras[0].trust_intrinsics=false;
    scene.cameras[0].fx=scene.cameras[0].fy=2000;
    scene.images.resize(6);
    for (Index i=0;i<6;++i) {
        auto& image=scene.images[i];
        image.id=i;
        image.camera_id=0;
        image.camera_identity="same-body";
        image.focal_length_mm=i<4?20.0:(i==4?17.0:19.0);
        image.exif_focal_px=i<4?2100.0:0.0;
    }
    expect(split_intrinsics_by_camera_identity(scene)==3,"split sparse zoom groups");
    expect(std::abs(scene.camera_of(scene.images[4]).fx-1700.0)<1e-9,
        "millimetre focal uses a ratio, not a pixel value");
    expect(std::abs(scene.camera_of(scene.images[5]).fx-1900.0)<1e-9,
        "singleton zoom keeps a physically scaled pixel initialization");
    expect(std::abs(scene.camera_of(scene.images[0]).fx-2100.0)<1e-9,
        "pixel focal group uses its measured initialization");
    Scene distinct;
    distinct.cameras={cam(),cam()};
    distinct.cameras[1].fx=700;
    distinct.cameras[1].fy=710;
    distinct.images.resize(2);
    for (Index i=0;i<2;++i) {
        distinct.images[i].id=i;
        distinct.images[i].camera_id=i;
    }
    split_intrinsics_by_camera_identity(distinct);
    expect(distinct.images[0].camera_id!=distinct.images[1].camera_id &&
        distinct.camera_of(distinct.images[1]).fx==700 &&
        distinct.camera_of(distinct.images[1]).fy==710,"missing EXIF cannot merge existing calibrations");
    Scene locked=scene;
    for (auto& camera:locked.cameras) camera.trust_intrinsics=true;
    const auto before=locked.cameras;
    for (auto& image:locked.images) image.exif_focal_px=9999;
    split_intrinsics_by_camera_identity(locked);
    for (std::size_t i=0;i<before.size();++i)
        expect(locked.cameras[i].fx==before[i].fx && locked.cameras[i].fy==before[i].fy,
            "trusted calibration is never overwritten by EXIF grouping");
}

void test_positioning_budget_progress() {
    ceres::Solver::Summary summary;
    summary.termination_type = ceres::NO_CONVERGENCE;
    summary.total_time_in_seconds = 2.5;
    summary.preprocessor_time_in_seconds = 2.0;
    summary.linear_solver_time_in_seconds = 0.4;
    summary.iterations.resize(2);
    summary.iterations[0].iteration = 0;
    summary.iterations[0].step_is_successful = true;
    summary.iterations[1].iteration = 1;
    summary.iterations[1].step_is_successful = false;
    auto progress = detail::positioning_pass_progress(summary, 1.0);
    expect(progress.accepted_updates == 0 && progress.time_limited && !progress.converged,
           "evaluation-only timeout is not effective update or convergence");
    expect(detail::can_retry_positioning_pass(10.0, progress) &&
           !detail::can_retry_positioning_pass(0.2, progress),
           "unchanged pass retries only when measured step fits remaining budget");
    const double budget = detail::positioning_pass_budget(10.0, 8, false,
                                                        progress.minimum_next_budget);
    expect(budget > summary.preprocessor_time_in_seconds + 0.4 && budget <= 10.0,
           "next CPU slice reserves preprocessing and an attempted update");
    expect(detail::positioning_pass_budget(0.2, 8, false, 3.0) == 0.2,
           "adaptive slice never exceeds remaining attempt budget");
    progress = detail::positioning_pass_progress(summary, 10.0);
    expect(!progress.time_limited && !progress.converged && progress.accepted_updates == 0,
           "rejected steps without timeout are stalled, not convergence");
    expect(!detail::can_retry_positioning_pass(10.0, progress),
           "non-timeout stall does not spin retries");
    summary.termination_type = ceres::CONVERGENCE;
    progress = detail::positioning_pass_progress(summary, 1.0);
    expect(progress.converged && !progress.time_limited,
           "a stationary optimum can legitimately converge without an update");
    summary.iterations[1].step_is_successful = true;
    expect(detail::positioning_pass_progress(summary, 10.0).accepted_updates == 1,
           "iteration zero is excluded from accepted parameter updates");
}

void test_position_audit_does_not_certify_a_weak_clique() {
    Scene scene;
    scene.cameras.push_back(cam());scene.cameras[0].id=0;
    scene.images.resize(7);
    for(Index i=0;i<7;++i) {
        auto& image=scene.images[i];image.id=i;image.camera_id=0;image.registered=true;
        image.pose.C=Vec3(0.5*i,0,0);
        for(unsigned p=0;p<64;++p) {
            const Vec3 point(-2.0+0.5*(p%8),-1.5+0.4*(p/8),8.0+0.1*(p%3));
            const Vec2 pixel=scene.cameras[0].project(point-image.pose.C);
            photara::features::Keypoint kp;kp.x=static_cast<float>(pixel.x());kp.y=static_cast<float>(pixel.y());
            image.features.keypoints.push_back(kp);
        }
    }
    const auto edge=[&](Index a,Index b,float weight) {
        ImagePair pair(a,b);pair.relative_pose=scene.images[b].pose/scene.images[a].pose;
        pair.weight_spatial=weight;
        for(Index p=0;p<64;++p)pair.matches.push_back({p,p});
        scene.pairs.push_back(std::move(pair));
    };
    for(Index a=0;a<4;++a)for(Index b=a+1;b<4;++b)edge(a,b,1);
    for(Index a=4;a<7;++a)for(Index b=a+1;b<7;++b)edge(a,b,1);
    edge(0,4,0.02F);edge(1,5,0.02F);
    // The weak branch is internally connected but must not become its own witness.
    for(Index i=4;i<7;++i)scene.images[i].pose.C+=Vec3(3,1,0);
    const auto before=scene.images;
    expect(audit_position_support(scene)==3,"weak clique is reported as position-unverified");
    for(Index i=0;i<7;++i) {
        expect(scene.images[i].position_audit==(i<4?PositionAudit::graph_supported:PositionAudit::unverified),
               "position audit distinguishes core support from weak branch");
        expect(scene.images[i].registered && scene.images[i].pose.C==before[i].pose.C &&
               scene.images[i].pose.R==before[i].pose.R,"nomination never changes registration or pose");
    }
    const Mat3 gauge=axis_rotation(Vec3(1,2,3),40);
    for(auto& image:scene.images) {
        image.pose.C=gauge*image.pose.C+Vec3(7,8,9);
        image.pose.R=image.pose.R*gauge.transpose();
    }
    expect(audit_position_support(scene)==3,"position support audit is world-gauge invariant");
    for(auto& image:scene.images) {
        image.pose.R=Mat3::Identity();
        image.features.keypoints=scene.images[0].features.keypoints;
    }
    expect(analyze_position_observability(scene).reliable_views==0,
           "connected zero-parallax matches cannot certify translation support");
}

// A complete circular capture may contain deliberately uneven sampling. The
// image observations, rather than a constant-speed orbit prior, must determine
// its final camera geometry. The old default flattened this trajectory.
void test_global_mapping_preserves_nonuniform_orbit(bool self_calibrate = false) {
    constexpr Index views = 24;
    constexpr Index points = 160;
    Scene scene;
    auto camera = cam();
    camera.id = 0;
    camera.trust_intrinsics = true;
    scene.cameras.push_back(camera);
    std::vector<Pose3D> truth(views);
    for (Index i = 0; i < views; ++i) {
        double angle = 2.0 * std::numbers::pi * i / views;
        if (i == 21) angle += 0.25;
        if (i == 22) angle += 0.16;
        if (i == 23) angle += 0.10;
        auto& pose = truth[i];
        // Vary height in the calibration case: a perfectly planar yaw-only
        // orbit cannot distinguish fy from a vertical scale of the scene.
        pose.C = Vec3(3.0 * std::cos(angle),
                      self_calibrate ? 0.8 * std::sin(0.7 * i) : 0.0,
                      3.0 * std::sin(angle));
        const Vec3 forward = -pose.C.normalized();
        const Vec3 right = Vec3::UnitY().cross(forward).normalized();
        pose.R.row(0) = right.transpose();
        pose.R.row(1) = forward.cross(right).transpose();
        pose.R.row(2) = forward.transpose();
        Image image;
        image.id = i;
        image.camera_id = 0;
        image.features.keypoints.resize(points);
        scene.images.push_back(std::move(image));
    }
    std::mt19937 rng(31);
    const double point_extent = self_calibrate ? 1.0 : 0.5;
    std::uniform_real_distribution<double> xyz(-point_extent, point_extent);
    for (Index p = 0; p < points; ++p) {
        Vec3 point;
        bool visible = false;
        while (!visible) {
            point = Vec3(xyz(rng), xyz(rng), xyz(rng));
            visible = true;
            for (const auto& pose : truth) {
                const Vec3 local = pose.transform_world_to_camera(point);
                const Vec2 pixel = camera.project(local);
                if (local.z() <= 0 || pixel.x() < 1 || pixel.y() < 1 ||
                    pixel.x() >= camera.width-1 || pixel.y() >= camera.height-1) {
                    visible = false;
                    break;
                }
            }
        }
        for (Index i = 0; i < views; ++i) {
            const Vec2 pixel = camera.project(truth[i].transform_world_to_camera(point));
            auto& kp = scene.images[i].features.keypoints[p];
            kp.x = static_cast<float>(pixel.x());
            kp.y = static_cast<float>(pixel.y());
        }
    }
    for (Index i = 0; i < views; ++i) {
        for (Index j = i + 1; j < views; ++j) {
            ImagePair pair(i, j);
            pair.relative_pose = truth[j] / truth[i];
            pair.relative_pose->C.normalize();
            pair.weight_spatial = 1.F;
            for (Index p = 0; p < points; ++p) pair.matches.push_back({p, p});
            scene.pairs.push_back(std::move(pair));
        }
    }
    // Leave orbit settings at production defaults. Only bound the synthetic
    // positioning problem and force CPU so this regression also runs headless.
    if (self_calibrate) {
        // A view-graph/EXIF focal is an initialization, not a calibration.
        // Precise multi-view observations must be allowed to correct it.
        scene.cameras[0].trust_intrinsics = false;
        scene.cameras[0].fx = scene.cameras[0].fy = 560;
        scene.cameras[0].focal_prior = 560;
    }
    GlobalPositioningOptions options;
    options.backend = PositioningBackend::cpu;
    options.min_tracks_for_positioning = points;
    options.max_tracks_for_positioning = points;
    const auto result = run_global_mapping(scene, {}, options);
    expect(result.valid && result.registered_views == views,
           "nonuniform orbit registers all cameras");
    expect(result.reprojection_observations >= views * points * 9 / 10 &&
               result.rms_reprojection_error_pixels < (self_calibrate ? 0.2 : 0.05),
           "nonuniform orbit retains observations with accurate reprojection");
    std::vector<double> steps;
    double max_rotation_error = 0.0;
    for (Index i = 1; i < views; ++i) {
        steps.push_back((scene.images[i].pose.C-scene.images[i-1].pose.C).norm());
        max_rotation_error = std::max(max_rotation_error, rotation_error(
            scene.images[i].pose.R * scene.images[i-1].pose.R.transpose(),
            truth[i].R * truth[i-1].R.transpose()));
    }
    std::sort(steps.begin(), steps.end());
    expect(steps.back() > 1.7 * steps[steps.size()/2],
           "nonuniform orbit preserves unequal camera steps");
    expect(max_rotation_error < (self_calibrate ? 0.2 : 0.05) * std::numbers::pi / 180.0,
           "nonuniform orbit preserves measured relative rotations");
    if (self_calibrate) {
        std::cout << "self_calibration fx=" << scene.cameras[0].fx
                  << " fy=" << scene.cameras[0].fy
                  << " rms=" << result.rms_reprojection_error_pixels
                  << " max_rotation_deg=" << max_rotation_error*180/std::numbers::pi << '\n';
        expect(std::abs(scene.cameras[0].focal()/camera.focal()-1.0) < 0.01,
               "late self-calibration corrects an untrusted initial focal");
    }
}

void test_boundary_track_quota() {
    Scene scene; scene.thread_count=1; auto camera=cam(); camera.id=0; scene.cameras.push_back(camera);
    for (Index id=0;id<12;++id) {
        Image image; image.id=id; image.camera_id=0; image.registered=true;
        image.features.image_width=640; image.features.image_height=480;
        image.features.keypoints.resize(21);
        for (auto& kp:image.features.keypoints) { kp.x=320; kp.y=240; }
        scene.images.push_back(image);
    }
    for (Index a=0;a<12;++a) for (Index b=a+1;b<12;++b) {
        if ((a<8 && b>=8) && !(a==7 && b==8)) continue;
        ImagePair pair(a,b); pair.relative_pose=Pose3D::identity(); pair.weight_spatial=1.F;
        for (Index p=0;p<100;++p) pair.matches.push_back({0,0});
        scene.pairs.push_back(pair);
    }
    for (Index t=0;t<21;++t) {
        Track track;
        if (t<8) for (Index id=0;id<8;++id) track.observations.push_back({id,t});
        else if (t<16) for (Index id=8;id<12;++id) track.observations.push_back({id,t});
        else if (t<20) for (Index id:{0u,1u,8u,9u}) track.observations.push_back({id,t});
        else for (Index id:{0u,8u}) track.observations.push_back({id,t});
        scene.tracks.push_back(track);
    }
    struct Selected {};
    const auto selection=[&](unsigned quota) {
        auto copy=scene; std::vector<Index> ids;
        GlobalPositioningOptions options; options.max_tracks_for_positioning=8;
        options.cross_branch_tracks_per_image=quota;
        options.selection_callback=[&](const Scene&,const std::vector<Index>& selected) { ids=selected; throw Selected{}; };
        try { solve_global_positions(copy,options); } catch (const Selected&) {}
        return ids;
    };
    const auto old=selection(0), improved=selection(20);
    expect(old.size()==8 && improved.size()==8,"boundary quota must preserve the hard budget");
    expect(std::none_of(old.begin(),old.end(),[](Index id){return id>=16;}),"control must reproduce boundary starvation");
    expect(std::count_if(improved.begin(),improved.end(),[](Index id){return id>=16 && id<20;})>=2,
           "quota must preserve eligible core-to-branch tracks");
    expect(std::find(improved.begin(),improved.end(),20)==improved.end(),"quota must not admit a two-view track");
}

int main() {
    test_boundary_track_quota();
    {
        Scene layout;
        layout.images.resize(80);
        std::vector<Vec3> before;
        const Mat3 gauge_rotation = Eigen::AngleAxisd(0.4, Vec3::UnitZ()).toRotationMatrix();
        for (Index id = 0; id < layout.images.size(); ++id) {
            before.emplace_back(id, 0.5*std::sin(0.2*id), 0.3*std::cos(0.1*id));
            layout.images[id].registered = true;
            layout.images[id].pose.C = 3*gauge_rotation*before.back()+Vec3(20,-5,7);
            if (id) {
                ImagePair pair(id-1,id); pair.relative_pose = Pose3D::identity();
                layout.pairs.push_back(pair);
            }
        }
        expect(detail::warm_start_layout_change(layout, before) < 1e-8,
               "bulk layout screening ignores similarity gauge changes");
        layout.images[20].pose.C += Vec3(1000,0,0);
        expect(detail::warm_start_layout_change(layout, before) < 1e-8,
               "one local outlier cannot masquerade as bulk layout regression");
        for (Index id = 0; id < layout.images.size(); ++id)
            layout.images[id].pose.C.y() += 0.05*id*id;
        expect(detail::warm_start_layout_change(layout, before) > 1,
               "large nonuniform layout distortion is measured in local baseline units");
    }
    {
        std::vector<std::vector<Index>> ancestry(47, {0, 1});
        ancestry[45] = ancestry[46] = {1, 2};
        const auto heldout = detail::stratified_resection_holdout(ancestry, 16);
        expect(std::count(heldout.begin(), heldout.end(), true) == 16 && heldout[45] != heldout[46],
               "rare independent depth ancestry remains represented in fit and holdout");
        expect(heldout == detail::stratified_resection_holdout(ancestry, 16),
               "ancestry stratification is deterministic without residual-dependent retries");
    }
    {
        Scene weak_leaf;
        weak_leaf.images.resize(40);
        for (Index i = 0; i < weak_leaf.images.size(); ++i) {
            weak_leaf.images[i].registered = true;
            weak_leaf.images[i].pose.C = Vec3(i, 0, 0);
            if (i) {
                ImagePair pair(i-1, i);
                pair.relative_pose = Pose3D::identity();
                pair.matches.resize(40);
                weak_leaf.pairs.push_back(pair);
            }
        }
        weak_leaf.images.back().pose.C.x() = 1000;
        const auto candidates = select_weak_position_repairs(weak_leaf);
        expect(std::find(candidates.begin(), candidates.end(), 39) != candidates.end(),
               "an unsupported scale-breaking leaf is nominated independently of references");
        expect(std::find(candidates.begin(), candidates.end(), 0) == candidates.end(),
               "ordinary camera spacing is not nominated for position repair");
        expect(recover_alignment_views(weak_leaf) == 0 && weak_leaf.registered_count() == 40 &&
               weak_leaf.images.back().pose.C.x() == 1000 && weak_leaf.images[8].pose.C.x() == 8,
               "failed independent re-localization restores registration and preserves poses");
    }
    {
        Scene neighbors;
        neighbors.images.resize(80);
        for (auto& image : neighbors.images) image.registered = true;
        neighbors.images[40].registered = false;
        ImagePair revisit(2, 40);
        revisit.relative_pose = Pose3D::identity();
        revisit.matches.resize(60);
        // A mapping-rejected edge is useful for searching, not accepting a pose.
        revisit.active = false;
        neighbors.pairs.push_back(revisit);
        ImagePair unavailable(3, 40);
        unavailable.relative_pose = Pose3D::identity();
        unavailable.matches.resize(100);
        neighbors.images[3].registered = false;
        neighbors.pairs.push_back(unavailable);
        const auto anchors = select_missing_view_anchors(neighbors, 40);
        expect(anchors.size() == 16, "recovery retains a bounded anchor budget");
        expect(std::find(anchors.begin(), anchors.end(), 2) != anchors.end(),
               "recovery includes distant verified visual overlap");
        expect(std::find(anchors.begin(), anchors.end(), 39) != anchors.end() &&
               std::find(anchors.begin(), anchors.end(), 41) != anchors.end(),
               "recovery retains temporal anchors on both sides");
        expect(std::find(anchors.begin(), anchors.end(), 3) == anchors.end() &&
               std::find(anchors.begin(), anchors.end(), 40) == anchors.end(),
               "recovery excludes unregistered anchors and its target");
        ImagePair continuation(3, 4);
        continuation.relative_pose = Pose3D::identity();
        continuation.matches.resize(100);
        neighbors.pairs.push_back(continuation);
        const auto extended = select_missing_view_anchors(neighbors, 40);
        expect(extended.size() == 16 && std::find(extended.begin(), extended.end(), 4) != extended.end(),
               "missing-run graph bridges nominate distant registered anchors within the same budget");
        expect(std::find(extended.begin(), extended.end(), 3) == extended.end(),
               "a missing bridge never becomes a depth anchor");
    }
    test_resection_candidate_failures();
    test_global_mapping_preserves_nonuniform_orbit();
    test_global_mapping_preserves_nonuniform_orbit(true);
    test_positioning_budget_progress();
    test_position_audit_does_not_certify_a_weak_clique();
    test_intrinsic_group_units();
    test_parallel_track_colours();
    test_missing_view_with_coincident_anchors();
    test_pair_cycle_weighting();
    test_global_rotation_weighting();
    test_global_rotation_dense_graph_iterative_solve();
    test_global_positioning_points_only();
    test_warm_start_geometry_acceptance();
    test_large_point_only_positioning_does_not_require_pairs();
    test_final_registration_audit_prunes_unsupported_pose();
    test_full_sphere_registration_audit();
    test_alignment_observability_reports_bridge_branch();
    test_global_positioning_irls_downweights_bad_direction();
    test_global_positioning_preserves_fixed_initial_positions();
    test_global_positioning_rejects_empty_point_constraints();
    test_long_track_merge();
    test_robust_triangulation_and_track_split();
    test_fast_split_gate();
    test_parallel_recursive_track_split();
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
