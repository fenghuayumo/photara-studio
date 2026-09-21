// Equirectangular (360) camera support: projection conventions, the
// seam/pole-safe local residual, two-view geometry, triangulation, resection
// and bundle adjustment on the full sphere.
#include "ba/optimizer.hpp"
#include "ba/linearizer.hpp"
#include "core/camera_projection.hpp"
#include "sfm/asfm.hpp"
#include "sfm/export_colmap.hpp"
#include "sfm/export_mvs.hpp"
#include "sfm/geometry.hpp"
#include "sfm/camera_selection.hpp"
#include "sfm/scene.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace {

using photara::CameraModel;
using photara::sfm::Index;
using photara::sfm::Mat3;
using photara::sfm::PinholeCamera;
using photara::sfm::Pose3D;
using photara::sfm::Scene;
using photara::sfm::Track;
using photara::sfm::Vec2;
using photara::sfm::Vec3;

int failures = 0;

void expect(const bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

constexpr double k_pi = 3.14159265358979323846;

PinholeCamera make_equirect(const std::uint32_t width = 2048,
                            const std::uint32_t height = 1024) {
    PinholeCamera camera;
    camera.width = width;
    camera.height = height;
    camera.set_equirectangular_intrinsics();
    return camera;
}

Pose3D make_pose(const Vec3& axis, const double angle, const Vec3& center) {
    Pose3D pose;
    pose.R = Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
    pose.C = center;
    return pose;
}

// Deterministic points on a shell around the origin so that a large fraction
// falls behind each camera: that back hemisphere is exactly what the pinhole
// pipelines cannot represent.
std::vector<Vec3> make_sphere_points(const int count, const unsigned seed,
                                     const double radius_min,
                                     const double radius_max) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> radius(radius_min, radius_max);
    std::vector<Vec3> points;
    points.reserve(static_cast<std::size_t>(count));
    while (static_cast<int>(points.size()) < count) {
        const Vec3 candidate(unit(rng), unit(rng), unit(rng));
        if (candidate.norm() < 1e-3) continue;
        points.push_back(candidate.normalized() * radius(rng));
    }
    return points;
}

void test_projection_conventions() {
    const PinholeCamera camera = make_equirect(1024, 512);
    expect(camera.fx == camera.fy, "2:1 equirect has equal pixel scales");
    expect(std::abs(camera.fx - 1024.0 / (2.0 * k_pi)) < 1e-12,
           "equirect fx = width / 2pi");
    expect(std::abs(camera.fy - 512.0 / k_pi) < 1e-12,
           "equirect fy = height / pi");
    expect(std::abs(camera.cx - 512.0) < 1e-12 &&
               std::abs(camera.cy - 256.0) < 1e-12,
           "equirect principal point is the image centre");
    expect(camera.trust_intrinsics && camera.is_equirectangular(),
           "equirect intrinsics are fully determined by the size");

    const Vec2 forward = camera.project(Vec3(0.0, 0.0, 1.0));
    expect((forward - Vec2(512.0, 256.0)).norm() < 1e-9,
           "optical axis projects to the image centre");
    const Vec2 right = camera.project(Vec3(1.0, 0.0, 0.0));
    expect((right - Vec2(768.0, 256.0)).norm() < 1e-9,
           "+X azimuth is a quarter turn");
    const Vec2 up = camera.project(Vec3(0.0, -1.0, 0.0));
    expect((up - Vec2(512.0, 0.0)).norm() < 1e-9,
           "-Y (image up) projects to the top row");
    const Vec2 down = camera.project(Vec3(0.0, 1.0, 0.0));
    expect((down - Vec2(512.0, 512.0)).norm() < 1e-9,
           "+Y projects to the bottom row");

    // Full-sphere round trip, including both hemispheres and both poles.
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    double worst = 0.0;
    for (int i = 0; i < 500; ++i) {
        const Vec3 bearing = Vec3(unit(rng), unit(rng), unit(rng));
        if (bearing.norm() < 1e-3) continue;
        const Vec3 direction = bearing.normalized();
        const Vec2 pixel = camera.project(direction);
        const Vec3 restored = camera.unproject_normalized(pixel);
        worst = std::max(worst, (restored - direction).norm());
    }
    expect(worst < 1e-9, "equirect project/unproject round trip over the sphere");

    const std::vector<Vec3> poles = {
        Vec3(0.0, 1.0, 0.0), Vec3(0.0, -1.0, 0.0), Vec3(0.0, 0.0, -1.0)};
    for (const Vec3& pole : poles) {
        const Vec2 pixel = camera.project(pole);
        expect((camera.unproject_normalized(pixel) - pole).norm() < 1e-9,
               "pole and back-hemisphere round trip");
    }

    // Back-hemisphere points must be observable: no cheirality check.
    Vec2 behind{};
    expect(camera.project_checked(Vec3(0.0, 0.0, -2.0), behind),
           "points behind an equirect camera are observable");
    expect(!camera.project_checked(Vec3::Zero(), behind),
           "a point at the optical centre is not observable");

    // Pixel-space thresholds are converted with the panorama sampling scale
    // (2 pi / width) tightened by k_equirect_threshold_scale, so that a pixel
    // budget means the same angular precision as it does for a rectilinear lens
    // of the same width (the chart samples the sphere ~6x more coarsely).
    expect(std::abs(camera.pixel_error_to_angular(1.0) -
                    (2.0 * k_pi / 1024.0) *
                        photara::k_equirect_threshold_scale) < 1e-15,
           "equirect pixel to angle conversion");
}

void test_local_reprojection() {
    const PinholeCamera camera = make_equirect(2048, 1024);
    const double pixel_scale = camera.fx;

    // Zero residual on the observed ray, anywhere on the sphere.
    for (double azimuth : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
        for (double elevation : {-1.4, -0.3, 0.0, 0.3, 1.4}) {
            const Vec3 direction(std::cos(elevation) * std::sin(azimuth),
                                 std::sin(elevation),
                                 std::cos(elevation) * std::cos(azimuth));
            const Vec2 pixel = camera.project(direction);
            const Vec3 point = direction * 4.5;
            const auto local = camera.local_reprojection(point, pixel);
            expect(local.valid && local.residual.norm() < 1e-9,
                   "tangent residual vanishes on the observed ray");
        }
    }

    // A small angular perturbation shows up as a proportional pixel residual,
    // and the analytic Jacobian matches finite differences.
    const Vec3 direction(Vec3(1.0, 0.4, -0.7).normalized());
    const Vec2 pixel = camera.project(direction);
    const Vec3 point = direction * 3.0;
    const auto local = camera.local_reprojection(point, pixel);
    expect(local.valid, "local reprojection is valid off axis");
    for (int column = 0; column < 3; ++column) {
        const double step = 1e-6 * point.norm();
        Vec3 shifted = point;
        shifted(column) += step;
        const auto moved = camera.local_reprojection(shifted, pixel);
        if (!moved.valid) continue;
        const Eigen::Vector2d difference = (moved.residual - local.residual) / step;
        const double error =
            (difference - local.jacobian.col(column)).norm();
        expect(error < 1e-4, "equirect residual Jacobian matches finite differences");
    }

    // Seam safety: an observation just before the seam and a predicted ray just
    // after it are two pixels apart in angle, while their image coordinates are
    // a whole width apart - exactly the case a pixel residual cannot express.
    const Vec2 seam_pixel(1.0, 512.0);
    const Vec3 seam_observation = camera.unproject_normalized(seam_pixel);
    const double seam_azimuth =
        (seam_pixel.x() / static_cast<double>(camera.width) - 0.5) * 2.0 * k_pi;
    const double mirrored_azimuth = -seam_azimuth;
    const Vec3 seam_direction(std::sin(mirrored_azimuth), 0.0,
                              std::cos(mirrored_azimuth));
    const auto seam_local =
        camera.local_reprojection(seam_direction * 2.0, seam_pixel);
    expect(seam_local.valid && seam_local.residual.norm() < 3.0,
           "tangent residual does not wrap across the azimuth seam");
    const double naive =
        (camera.project(seam_direction * 2.0) - seam_pixel).norm();
    expect(naive > 0.5 * camera.width,
           "naive pixel difference would wrap across the seam");

    // Pole safety: the residual stays finite and small near the image poles.
    const Vec2 pole_pixel(1024.0, 1.0);
    const Vec3 pole_direction = camera.unproject_normalized(pole_pixel);
    const double pole_epsilon = 4.0 * k_pi / 2048.0;
    const Vec3 pole_offset =
        (Eigen::AngleAxisd(pole_epsilon, Vec3::UnitX()) * pole_direction)
            .normalized();
    const auto pole_local =
        camera.local_reprojection(pole_offset * 2.0, pole_pixel);
    expect(pole_local.valid && pole_local.residual.norm() < 2.5,
           "tangent residual stays bounded at the pole");
    expect(std::isfinite(pixel_scale), "pixel scale is finite");
}

void test_relative_pose_equirect() {
    const PinholeCamera camera = make_equirect();
    const Pose3D second = make_pose(Vec3(0.2, 1.0, 0.1), 0.22, Vec3(1.1, 0.15, 0.05));
    const std::vector<Vec3> points = make_sphere_points(400, 23, 2.0, 7.0);

    std::vector<Vec2> first_pixels;
    std::vector<Vec2> second_pixels;
    for (const Vec3& point : points) {
        first_pixels.push_back(camera.project(point));
        second_pixels.push_back(
            camera.project(second.transform_world_to_camera(point)));
    }
    // Half of the points are behind one of the two cameras.
    unsigned behind = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (points[i].z() < 0.0) ++behind;
        if (second.transform_world_to_camera(points[i]).z() < 0.0) ++behind;
    }
    expect(behind > 100, "the synthetic pair really exercises the back hemisphere");

    const auto relative = photara::sfm::estimate_relative_pose(
        first_pixels, second_pixels, camera, camera);
    expect(relative.success, "equirect two-view estimation succeeds");
    if (relative.success) {
        expect((relative.pose.R - second.R).norm() < 5e-3,
               "equirect relative rotation");
        expect(relative.pose.C.normalized().dot(second.C.normalized()) > 0.999,
               "equirect translation direction");
        if (relative.num_inliers < 380) {
            std::cerr << "  [debug] equirect pair inliers=" << relative.num_inliers
                      << " ransac=" << relative.num_ransac_inliers << '\n';
            expect(false, "equirect pair keeps the back-hemisphere inliers");
        }
        expect(relative.F.isZero(1e-12),
               "no fundamental matrix is produced for a spherical pair");
        expect(!relative.degenerate_planar,
               "a real baseline is not flagged as a rotation-only pair");
    }

    // A pure-rotation panorama pair has no baseline at all, so it must never be
    // accepted as an initialization edge: either the estimation fails (nothing
    // to triangulate) or the rotation-degeneracy test (the spherical analogue of
    // the homography test) flags it.
    const Pose3D rotation_only = make_pose(Vec3(0.0, 1.0, 0.0), 0.3, Vec3::Zero());
    std::vector<Vec2> rotated_pixels;
    for (const Vec3& point : points) {
        rotated_pixels.push_back(
            camera.project(rotation_only.transform_world_to_camera(point)));
    }
    const auto degenerate = photara::sfm::estimate_relative_pose(
        first_pixels, rotated_pixels, camera, camera);
    expect(!degenerate.success || degenerate.degenerate_planar,
           "rotation-only panorama pair is never usable for initialization");
    if (degenerate.success) {
        expect((degenerate.pose.R - rotation_only.R).norm() < 5e-3,
               "rotation-only pair still recovers the rotation");
        expect(degenerate.degenerate_planar,
               "rotation-only pair is flagged as planar/degenerate");
    }

    // A wide-baseline pair must not be flagged: the degeneracy test has to
    // discriminate a rotation-only capture from a real one.
    expect(!relative.degenerate_planar,
           "a genuine panorama baseline is not flagged degenerate");
}

void test_absolute_pose_equirect() {
    const PinholeCamera camera = make_equirect();
    const Pose3D pose =
        make_pose(Vec3(-0.3, 1.0, 0.2), 0.35, Vec3(0.4, -0.2, 0.9));
    const std::vector<Vec3> points = make_sphere_points(300, 31, 2.0, 6.0);

    std::vector<Vec3> bearings;
    std::vector<Vec3> world;
    for (const Vec3& point : points) {
        const Vec3 camera_point = pose.transform_world_to_camera(point);
        bearings.push_back(camera_point.normalized());
        world.push_back(point);
    }

    const auto absolute = photara::sfm::estimate_absolute_pose(
        bearings, world, camera, {});
    expect(absolute.success, "equirect absolute pose succeeds");
    if (absolute.success) {
        expect((absolute.pose.R - pose.R).norm() < 1e-3,
               "equirect resection rotation");
        expect((absolute.pose.C - pose.C).norm() < 1e-2,
               "equirect resection center");
        expect(absolute.num_inliers >= 295,
               "equirect resection keeps back-hemisphere points");
    }

    // A pinhole camera fed the same bearings must still use the historical
    // front-hemisphere PoseLib path (no behavior change for existing cameras).
    PinholeCamera pinhole;
    pinhole.width = 2048;
    pinhole.height = 1024;
    pinhole.fx = pinhole.fy = 900.0;
    pinhole.cx = 1024.0;
    pinhole.cy = 512.0;
    std::vector<Vec3> front_bearings;
    std::vector<Vec3> front_world;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (bearings[i].z() <= 0.05) continue;
        front_bearings.push_back(bearings[i]);
        front_world.push_back(world[i]);
    }
    const auto pinhole_pose = photara::sfm::estimate_absolute_pose(
        front_bearings, front_world, pinhole, {});
    expect(pinhole_pose.success &&
               (pinhole_pose.pose.R - pose.R).norm() < 1e-3,
           "pinhole resection is unchanged");
}

void test_triangulation_equirect() {
    const PinholeCamera camera = make_equirect();
    Scene scene;
    scene.cameras = {camera, camera, camera};
    const std::vector<Pose3D> poses = {
        Pose3D::identity(),
        make_pose(Vec3(0.0, 1.0, 0.0), 0.5, Vec3(2.0, 0.1, 0.0)),
        make_pose(Vec3(1.0, 0.0, 0.3), -0.6, Vec3(-0.5, 0.4, 1.8))};

    scene.images.resize(poses.size());
    const std::vector<Vec3> points = make_sphere_points(120, 47, 2.5, 6.0);
    for (std::size_t image = 0; image < poses.size(); ++image) {
        scene.images[image].id = static_cast<Index>(image);
        scene.images[image].camera_id = static_cast<Index>(image);
        scene.images[image].registered = true;
        scene.images[image].pose = poses[image];
        scene.images[image].features.keypoints.resize(points.size());
        for (std::size_t i = 0; i < points.size(); ++i) {
            const Vec3 camera_point =
                poses[image].transform_world_to_camera(points[i]);
            const Vec2 pixel = camera.project(camera_point);
            scene.images[image].features.keypoints[i].x =
                static_cast<float>(pixel.x());
            scene.images[image].features.keypoints[i].y =
                static_cast<float>(pixel.y());
        }
    }
    for (std::size_t i = 0; i < points.size(); ++i) {
        Track track;
        track.position = points[i] + Vec3(0.4, -0.3, 0.2);
        for (std::size_t image = 0; image < poses.size(); ++image) {
            track.observations.push_back(
                {static_cast<Index>(image), static_cast<Index>(i)});
        }
        scene.tracks.push_back(track);
    }

    const unsigned triangulated = photara::sfm::triangulate_tracks(
        scene, false, photara::sfm::TriangulationOptions{});
    expect(triangulated >= 115,
           "multi-view equirect triangulation keeps nearly every track");

    double worst = 0.0;
    unsigned compared = 0;
    for (std::size_t i = 0; i < scene.tracks.size(); ++i) {
        if (!scene.tracks[i].is_triangulated()) continue;
        worst = std::max(worst, (scene.tracks[i].position - points[i]).norm());
        ++compared;
    }
    expect(compared > 100 && worst < 1e-4,
           "equirect triangulated positions are accurate");

    // Track filtering must accept the full-sphere observations, including the
    // ones behind a camera, and reject a deliberately wrong position.
    const auto stats = photara::sfm::filter_tracks(scene, 3.F, 0.5F);
    expect(stats.first < 1e-3, "equirect track filtering keeps small residuals");
    unsigned behind_observations = 0;
    for (std::size_t image = 0; image < poses.size(); ++image) {
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (poses[image].transform_world_to_camera(points[i]).z() < 0.0)
                ++behind_observations;
        }
    }
    expect(behind_observations > 50,
           "the filtered tracks do contain back-hemisphere observations");
    std::size_t surviving_inliers = 0;
    for (const Track& track : scene.tracks) surviving_inliers += track.num_inliers;
    expect(surviving_inliers >=
               static_cast<std::size_t>(3) * 115 - 5,
           "back-hemisphere observations survive filtering");
}

void test_bundle_adjustment_equirect() {
    const PinholeCamera camera = make_equirect();
    // The cloud sits well outside the camera cluster: points passing close to a
    // camera centre are geometrically ill-conditioned for any camera model and
    // would dominate the cost instead of exercising the spherical residual.
    const std::vector<Vec3> points = make_sphere_points(150, 61, 9.0, 15.0);
    const std::vector<Pose3D> poses = {
        Pose3D::identity(),
        make_pose(Vec3(0.1, 1.0, 0.0), 0.4, Vec3(3.0, 0.4, 0.2)),
        make_pose(Vec3(1.0, 0.2, 0.4), -0.5, Vec3(-1.2, 0.8, 3.0))};

    photara::ba::Problem problem;
    for (const Pose3D& pose : poses) {
        const Eigen::Quaterniond quaternion(pose.R);
        problem.poses.push_back({quaternion.w(), quaternion.x(),
                                 quaternion.y(), quaternion.z(), pose.C.x(),
                                 pose.C.y(), pose.C.z()});
        problem.pose_intrinsic.push_back(0);
        problem.pose_constant.push_back(0);
    }
    problem.intrinsics.push_back({camera.fx, camera.fy, camera.cx, camera.cy,
                                  camera.k1, camera.k2, camera.p1, camera.p2,
                                  CameraModel::equirectangular});
    problem.initial_intrinsics = problem.intrinsics;
    problem.intrinsic_constant.assign(1, 1);
    for (const Vec3& point : points)
        problem.points.push_back({point.x(), point.y(), point.z()});
    for (std::size_t image = 0; image < poses.size(); ++image) {
        for (std::size_t i = 0; i < points.size(); ++i) {
            const Vec2 pixel = camera.project(
                poses[image].transform_world_to_camera(points[i]));
            problem.observations.push_back(static_cast<photara::ba::Index>(image),
                                           static_cast<photara::ba::Index>(i),
                                           pixel.x(), pixel.y());
        }
    }
    problem.validate();

    const double exact_cost = photara::ba::evaluate_cost(problem, 2.0);
    expect(exact_cost < 1e-12,
           "the exact equirect solution has zero cost");

    // Perturb the initial solution the way an incremental initialization would:
    // a fraction of a degree of rotation and a small fraction of a pixel of
    // position error. (Bigger errors are handled by the pipeline's staged BA;
    // a panorama's azimuth coordinate amplifies them near the poles, which is a
    // property of the chart rather than of the optimizer.)
    for (std::size_t image = 1; image < problem.poses.size(); ++image) {
        photara::ba::Pose& pose = problem.poses[image];
        const Eigen::Quaterniond bump =
            Eigen::Quaterniond(Eigen::AngleAxisd(
                0.004 * static_cast<double>(image), Vec3::UnitX()));
        const Eigen::Quaterniond shifted(
            bump * Eigen::Quaterniond(pose.qw, pose.qx, pose.qy, pose.qz));
        pose.qw = shifted.w();
        pose.qx = shifted.x();
        pose.qy = shifted.y();
        pose.qz = shifted.z();
    }
    for (std::size_t i = 0; i < problem.points.size(); ++i) {
        // Signed: (i % 5) - 2 on an unsigned index would wrap around.
        const int index = static_cast<int>(i);
        problem.points[i].x += 0.002 * static_cast<double>(index % 5 - 2);
        problem.points[i].y += 0.002 * static_cast<double>(index % 7 - 3);
        problem.points[i].z += 0.002 * static_cast<double>(index % 3 - 1);
    }

    const double initial_cost = photara::ba::evaluate_cost(problem, 2.0);
    expect(std::isfinite(initial_cost) && initial_cost > 0.0,
           "equirect BA starts from a finite cost");
    photara::ba::OptimizerOptions options;
    options.fix_first_pose = true;
    const auto summary = photara::ba::optimize_cpu(problem, options);
    if (!summary.usable() || !(summary.final_cost < initial_cost * 0.05)) {
        std::cerr << "  equirect BA cost: " << initial_cost << " -> "
                  << summary.final_cost << " status="
                  << static_cast<int>(summary.termination) << '\n';
    }
    expect(summary.usable(), "equirect BA converges");
    expect(summary.final_cost < initial_cost * 0.05,
           "equirect BA drives the cost close to zero");
    expect(problem.intrinsics[0].fx == camera.fx,
           "equirect intrinsics stay frozen");

    // Only one pose is anchored, so monocular BA keeps the global scale gauge
    // free: the recovered structure is compared after removing it, which is the
    // meaningful invariant (the optimizer starts near scale 1 and stays there).
    Vec3 true_centroid = Vec3::Zero();
    Vec3 recovered_centroid = Vec3::Zero();
    std::vector<Vec3> recovered_points(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        recovered_points[i] = Vec3(problem.points[i].x, problem.points[i].y,
                                   problem.points[i].z);
        true_centroid += points[i];
        recovered_centroid += recovered_points[i];
    }
    true_centroid /= static_cast<double>(points.size());
    recovered_centroid /= static_cast<double>(points.size());
    double true_radius = 0.0;
    double recovered_radius = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        true_radius += (points[i] - true_centroid).squaredNorm();
        recovered_radius +=
            (recovered_points[i] - recovered_centroid).squaredNorm();
    }
    true_radius = std::sqrt(true_radius / static_cast<double>(points.size()));
    recovered_radius =
        std::sqrt(recovered_radius / static_cast<double>(points.size()));
    expect(true_radius > 1e-6 && recovered_radius > 1e-6,
           "equirect BA keeps a finite scene scale");
    const double scale = recovered_radius / std::max(true_radius, 1e-12);
    expect(std::abs(scale - 1.0) < 0.02,
           "equirect BA does not drift off the initialization scale");
    double worst_point = 0.0;
    double worst_rotation = 0.0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Vec3 normalized_true = (points[i] - true_centroid) / true_radius;
        const Vec3 normalized_recovered =
            (recovered_points[i] - recovered_centroid) / recovered_radius;
        worst_point = std::max(
            worst_point, (normalized_recovered - normalized_true).norm());
    }
    for (std::size_t image = 1; image < poses.size(); ++image) {
        const photara::ba::Pose& pose = problem.poses[image];
        const Mat3 rotation =
            Eigen::Quaterniond(pose.qw, pose.qx, pose.qy, pose.qz)
                .normalized()
                .toRotationMatrix();
        worst_rotation = std::max(worst_rotation, (rotation - poses[image].R).norm());
    }
    expect(worst_point < 2e-3,
           "equirect BA recovers the point cloud (up to the scale gauge)");
    expect(worst_rotation < 1e-3, "equirect BA recovers the camera rotations");
    if (worst_point >= 2e-3 || worst_rotation >= 1e-3) {
        std::cerr << "  [debug] worst_point=" << worst_point
                  << " worst_rotation=" << worst_rotation
                  << " scale=" << scale
                  << " final_cost=" << summary.final_cost << '\n';
    }

    // A deliberately large initialization error must still improve the cost
    // monotonically even though the azimuth chart makes it a hard case.
    photara::ba::Problem coarse = problem;
    for (std::size_t image = 1; image < coarse.poses.size(); ++image) {
        photara::ba::Pose& pose = coarse.poses[image];
        const Eigen::Quaterniond bump = Eigen::Quaterniond(
            Eigen::AngleAxisd(0.05 * static_cast<double>(image), Vec3::UnitX()));
        const Eigen::Quaterniond shifted(
            bump * Eigen::Quaterniond(pose.qw, pose.qx, pose.qy, pose.qz));
        pose.qw = shifted.w();
        pose.qx = shifted.x();
        pose.qy = shifted.y();
        pose.qz = shifted.z();
    }
    const double coarse_cost = photara::ba::evaluate_cost(coarse, 2.0);
    photara::ba::OptimizerOptions coarse_options = options;
    coarse_options.maximum_iterations = 60;
    const auto coarse_summary = photara::ba::optimize_cpu(coarse, coarse_options);
    expect(coarse_summary.usable() && coarse_summary.final_cost < coarse_cost * 0.9,
           "equirect BA still improves a coarse initialization");

#if defined(PHOTARA_HAS_CUDA)
    // The linearizer is shared by the CPU and CUDA backends, so the spherical
    // branch must produce bit-comparable residuals on both. This catches the
    // class of bug where a device-only helper silently returns an invalid
    // observation (which then looks like a converged zero-cost problem).
    if (photara::ba::CudaLinearizer::is_available()) {
        photara::ba::LinearizationOutput cpu_output;
        photara::ba::linearize_cpu(coarse, cpu_output);
        photara::ba::CudaLinearizer device;
        device.upload(coarse);
        device.evaluate();
        photara::ba::LinearizationOutput gpu_output;
        device.download(gpu_output);
        expect(cpu_output.observations.size() == gpu_output.observations.size(),
               "equirect CPU/CUDA linearizer observation counts match");
        unsigned mismatches = 0;
        unsigned valid_mismatches = 0;
        for (std::size_t i = 0; i < cpu_output.observations.size() &&
                                i < gpu_output.observations.size();
             ++i) {
            if (cpu_output.observations[i].valid !=
                gpu_output.observations[i].valid)
                ++valid_mismatches;
            for (int axis = 0; axis < 2; ++axis) {
                if (std::abs(cpu_output.observations[i].residual[axis] -
                             gpu_output.observations[i].residual[axis]) > 1e-8)
                    ++mismatches;
            }
        }
        expect(valid_mismatches == 0,
               "equirect CPU/CUDA linearizer validity agrees");
        expect(mismatches == 0, "equirect CPU/CUDA linearizer residuals agree");
    }
#endif
}

// Scene persistence and export guards for panoramas: the native .asfm scene must
// round-trip the model and its chart intrinsics, the COLMAP export must declare
// the spherical model, and the OpenMVS interface must refuse (a sphere cannot be
// resampled into a pinhole working camera).
void test_storage_and_export() {
    const PinholeCamera camera = make_equirect(2048, 1024);
    Scene scene;
    scene.cameras = {camera};
    scene.images.resize(2);
    const std::vector<Pose3D> poses = {
        Pose3D::identity(),
        make_pose(Vec3(0.0, 1.0, 0.0), 0.2, Vec3(0.5, 0.0, 0.0))};
    for (std::size_t i = 0; i < poses.size(); ++i) {
        scene.images[i].id = static_cast<Index>(i);
        scene.images[i].camera_id = 0;
        scene.images[i].registered = true;
        scene.images[i].pose = poses[i];
        scene.images[i].path = "frame_0000" + std::to_string(i) + ".jpg";
        scene.images[i].features.image_width = camera.width;
        scene.images[i].features.image_height = camera.height;
    }
    Track track;
    track.position = Vec3(0.0, 0.0, 5.0);
    for (std::size_t i = 0; i < poses.size(); ++i) {
        const Vec2 pixel = camera.project(
            poses[i].transform_world_to_camera(track.position));
        scene.images[i].features.keypoints.resize(1);
        scene.images[i].features.keypoints[0].x = static_cast<float>(pixel.x());
        scene.images[i].features.keypoints[0].y = static_cast<float>(pixel.y());
        track.observations.push_back({static_cast<Index>(i), 0});
    }
    track.num_inliers = 2;
    scene.tracks.push_back(track);

    const auto encoded = photara::sfm::encode_asfm(scene);
    const Scene decoded = photara::sfm::decode_asfm(encoded);
    expect(decoded.cameras.size() == 1 &&
               decoded.cameras[0].model == CameraModel::equirectangular,
           ".asfm round trip preserves the panorama camera model");
    expect(decoded.cameras[0].fx == camera.fx &&
               decoded.cameras[0].fy == camera.fy &&
               decoded.cameras[0].cx == camera.cx,
           ".asfm round trip preserves the panorama chart intrinsics");

    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "photara_equirect_export_test";
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    photara::sfm::save_colmap_text(scene, directory / "colmap", {}, false);
    std::ifstream cameras_txt(directory / "colmap" / "cameras.txt");
    std::string line;
    bool declared = false;
    while (std::getline(cameras_txt, line))
        if (line.find("EQUIRECTANGULAR") != std::string::npos) declared = true;
    expect(declared, "COLMAP export declares the equirectangular model");

    bool rejected = false;
    try {
        photara::sfm::export_openmvs_interface(scene, directory / "scene.mvs");
    } catch (const std::exception&) {
        rejected = true;
    }
    expect(rejected, "OpenMVS export refuses an equirectangular alignment");
    std::filesystem::remove_all(directory, error);
}

// Auto model selection: a 2:1 image size selects the panorama chart when the
// panorama hypotheses are geometrically sound, and must not fire for a 2:1
// rectilinear image (whose pixels would span only a few degrees of the sphere).
void test_camera_model_selection() {
    expect(photara::sfm::is_equirectangular_image_size(2048, 1024),
           "2:1 image size is recognized");
    expect(!photara::sfm::is_equirectangular_image_size(1920, 1080),
           "16:9 image size is not a panorama");
    expect(!photara::sfm::is_equirectangular_image_size(0, 0),
           "empty image size is not a panorama");

    // Probe pairs are taken from a genuine panorama capture: two 360 views of a
    // point cloud, the second one rotated and shifted slightly (a hand-held
    // panorama sampled from a video).
    const PinholeCamera panorama = make_equirect(2048, 1024);
    const std::vector<Vec3> points = make_sphere_points(600, 91, 6.0, 20.0);
    std::vector<photara::sfm::CameraModelProbe> probes;
    std::mt19937 rng(5);
    std::normal_distribution<double> noise(0.0, 0.5);
    for (int pair = 0; pair < 6; ++pair) {
        const Pose3D second = make_pose(
            Vec3(0.1 * pair, 1.0, 0.05 * pair), 0.05 + 0.02 * pair,
            Vec3(0.15 * pair, 0.02 * pair, 0.05));
        photara::sfm::CameraModelProbe probe;
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (i % 3 == 0) continue;  // training subset
            const Vec2 first_pixel = panorama.project(points[i]);
            const Vec3 camera_point =
                second.transform_world_to_camera(points[i]);
            if (camera_point.norm() < 1e-6) continue;
            const Vec2 second_pixel = panorama.project(camera_point);
            probe.first.emplace_back(first_pixel.x() + noise(rng),
                                     first_pixel.y() + noise(rng));
            probe.second.emplace_back(second_pixel.x() + noise(rng),
                                      second_pixel.y() + noise(rng));
        }
        probes.push_back(std::move(probe));
    }

    PinholeCamera uncalibrated = panorama;
    uncalibrated.model = CameraModel::pinhole;
    uncalibrated.fx = uncalibrated.fy = 1.2 * 2048.0;
    uncalibrated.cx = 1024.0;
    uncalibrated.cy = 512.0;
    uncalibrated.trust_intrinsics = false;
    const auto selected = photara::sfm::select_camera_model(
        uncalibrated, probes, 0.0, false, CameraModel::automatic, true);
    if (selected.model != CameraModel::equirectangular) {
        std::cerr << "  [debug] panorama fixture: pano_score=" << selected.equirect_score
                  << " pinhole_score=" << selected.pinhole_score
                  << " fisheye_score=" << selected.fisheye_score
                  << " informative=" << selected.informative_pairs
                  << " reason=" << selected.reason << '\n';
    }
    expect(selected.model == CameraModel::equirectangular,
           "auto selection picks the panorama for 2:1 panorama imagery");
    expect(selected.reason.find("2:1") != std::string::npos,
           "auto selection reports the size-driven reason");
    expect(std::abs(selected.focal_pixels - panorama.fx) < 1e-9,
           "auto selection reports the panorama pixel scale");

    // The same probe data without the 2:1 size hint must not select a panorama.
    const auto no_hint = photara::sfm::select_camera_model(
        uncalibrated, probes, 0.0, false, CameraModel::automatic, false);
    expect(no_hint.model != CameraModel::equirectangular,
           "the panorama model is only considered for 2:1 imagery");

    // An explicit request always wins and needs no probes.
    const auto explicit_request = photara::sfm::select_camera_model(
        uncalibrated, {}, 0.0, false, CameraModel::equirectangular, true);
    expect(explicit_request.model == CameraModel::equirectangular,
           "an explicit panorama request is honoured");

    // A pinhole capture rendered into a 2:1 image: the panorama chart must be
    // rejected because its pixels cover far less than the sphere.
    PinholeCamera pinhole = make_equirect(2048, 1024);
    pinhole.model = CameraModel::pinhole;
    pinhole.fx = pinhole.fy = 700.0;
    pinhole.trust_intrinsics = true;
    std::vector<photara::sfm::CameraModelProbe> pinhole_probes;
    for (int pair = 0; pair < 6; ++pair) {
        const Pose3D second = make_pose(
            Vec3(0.0, 1.0, 0.0), 0.03 * (pair + 1), Vec3(0.4, 0.0, 0.0));
        photara::sfm::CameraModelProbe probe;
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (i % 3 == 0) continue;
            const Vec3 first_camera_point = points[i];
            if (first_camera_point.z() < 1.0) continue;
            const Vec2 first_pixel = pinhole.project(first_camera_point);
            if (first_pixel.x() < 0.0 || first_pixel.x() > 2048.0 ||
                first_pixel.y() < 0.0 || first_pixel.y() > 1024.0)
                continue;
            const Vec3 second_camera_point =
                second.transform_world_to_camera(points[i]);
            if (second_camera_point.z() < 1.0) continue;
            const Vec2 second_pixel = pinhole.project(second_camera_point);
            probe.first.emplace_back(first_pixel.x() + noise(rng) * 0.2,
                                     first_pixel.y() + noise(rng) * 0.2);
            probe.second.emplace_back(second_pixel.x() + noise(rng) * 0.2,
                                      second_pixel.y() + noise(rng) * 0.2);
        }
        pinhole_probes.push_back(std::move(probe));
    }
    const auto rectilinear = photara::sfm::select_camera_model(
        pinhole, pinhole_probes, 0.0, false, CameraModel::automatic, true);
    if (rectilinear.model == CameraModel::equirectangular) {
        std::cerr << "  [debug] rectilinear: pano_score=" << rectilinear.equirect_score
                  << " pinhole_score=" << rectilinear.pinhole_score
                  << " fisheye_score=" << rectilinear.fisheye_score
                  << " informative=" << rectilinear.informative_pairs << '\n';
    }
    expect(rectilinear.model != CameraModel::equirectangular,
           "a 2:1 rectilinear capture is not mistaken for a panorama");
}

}  // namespace

int main() {
    test_projection_conventions();
    test_local_reprojection();
    test_relative_pose_equirect();
    test_absolute_pose_equirect();
    test_triangulation_equirect();
    test_bundle_adjustment_equirect();
    test_storage_and_export();
    test_camera_model_selection();

    if (failures == 0) {
        std::cout << "equirect tests passed\n";
        return 0;
    }
    std::cerr << failures << " equirect test(s) failed\n";
    return 1;
}
