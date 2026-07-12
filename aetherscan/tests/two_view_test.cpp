#include "sfm/geometry.hpp"
#include "sfm/triangulation.hpp"
#include "sfm/tracks.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {

using aetherscan::sfm::Index;
using aetherscan::sfm::Mat3;
using aetherscan::sfm::PinholeCamera;
using aetherscan::sfm::Pose3D;
using aetherscan::sfm::Scene;
using aetherscan::sfm::Track;
using aetherscan::sfm::Vec2;
using aetherscan::sfm::Vec3;

int failures = 0;

void expect(const bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

PinholeCamera make_camera(const double focal = 800.0) {
    PinholeCamera camera;
    camera.width = 1280;
    camera.height = 720;
    camera.fx = focal;
    camera.fy = focal;
    camera.cx = 640.0;
    camera.cy = 360.0;
    return camera;
}

}  // namespace

int main() {
    PinholeCamera distorted = make_camera();
    distorted.k1 = 0.08;
    distorted.k2 = -0.03;
    distorted.p1 = 0.001;
    distorted.p2 = -0.002;
    const Vec3 distorted_ray(0.45, -0.2, 2.0);
    const Vec2 distorted_pixel = distorted.project(distorted_ray);
    expect(
        (distorted.unproject_normalized(distorted_pixel) -
         distorted_ray.normalized()).norm() < 1e-7,
        "distorted project/unproject should be consistent");

    // Synthetic two-view relative pose
    const PinholeCamera cam = make_camera();
    Pose3D pose2;
    pose2.R = Mat3::Identity();
    pose2.C = Vec3(0.3, 0.0, 0.0);

    std::mt19937 rng(1);
    std::uniform_real_distribution<double> xy(-1.0, 1.0);
    std::uniform_real_distribution<double> z(3.0, 8.0);

    std::vector<Vec2> pixels1, pixels2;
    for (int i = 0; i < 120; ++i) {
        const Vec3 X(xy(rng), xy(rng), z(rng));
        const Vec3 X1 = X;
        const Vec3 X2 = pose2.transform_world_to_camera(X);
        pixels1.push_back(cam.project(X1));
        pixels2.push_back(cam.project(X2));
    }

    aetherscan::sfm::RelativePoseOptions options;
    options.min_inliers = 40;
    options.max_epipolar_error_px = 2.0;
    options.max_reproj_error_px = 2.0;
    options.min_ray_angle_deg = 0.5;
    const auto result =
        aetherscan::sfm::estimate_relative_pose(pixels1, pixels2, cam, cam, options);
    expect(result.success, "relative pose should succeed");
    expect(result.num_inliers > 80, "relative pose should have many inliers");
    expect(result.pose.C.norm() > 0.1, "baseline should be non-trivial");
    // Synthetic scene has strong parallax — should not be flagged planar.
    expect(!result.degenerate_planar, "parallax scene should not be H-degenerate");

    std::vector<Vec2> distorted_pixels1, distorted_pixels2;
    for (int i = 0; i < 120; ++i) {
        const Vec3 X(xy(rng), xy(rng), z(rng));
        distorted_pixels1.push_back(distorted.project(X));
        distorted_pixels2.push_back(
            distorted.project(pose2.transform_world_to_camera(X)));
    }
    const auto distorted_result = aetherscan::sfm::estimate_relative_pose(
        distorted_pixels1, distorted_pixels2, distorted, distorted, options);
    expect(
        distorted_result.success && distorted_result.num_inliers > 80,
        "distortion-aware relative pose should succeed");
    auto fundamental_options = options;
    fundamental_options.force_fundamental = true;
    fundamental_options.decompose_fundamental = true;
    const auto distorted_fundamental =
        aetherscan::sfm::estimate_relative_pose(
            distorted_pixels1, distorted_pixels2, distorted, distorted,
            fundamental_options);
    expect(
        distorted_fundamental.success &&
            distorted_fundamental.num_inliers > 80,
        "distortion-aware fundamental estimation should succeed");

    // Absolute pose
    std::vector<Vec3> bearings, points;
    for (int i = 0; i < 80; ++i) {
        const Vec3 X(xy(rng), xy(rng), z(rng));
        points.push_back(X);
        bearings.push_back(pose2.transform_world_to_camera(X).normalized());
    }
    aetherscan::sfm::AbsolutePoseOptions abs_opts;
    abs_opts.min_inliers = 20;
    const auto abs =
        aetherscan::sfm::estimate_absolute_pose(bearings, points, cam, abs_opts);
    expect(abs.success, "absolute pose should succeed");
    expect((abs.pose.C - pose2.C).norm() < 0.05, "absolute pose center close");

    // Triangulation + tracks
    Scene scene;
    scene.cameras = {cam, cam};
    scene.images.resize(2);
    scene.images[0].id = 0;
    scene.images[0].camera_id = 0;
    scene.images[0].registered = true;
    scene.images[0].pose = Pose3D::identity();
    scene.images[0].features.keypoints.resize(pixels1.size());
    scene.images[1].id = 1;
    scene.images[1].camera_id = 1;
    scene.images[1].registered = true;
    scene.images[1].pose = pose2;
    scene.images[1].features.keypoints.resize(pixels2.size());
    for (std::size_t i = 0; i < pixels1.size(); ++i) {
        scene.images[0].features.keypoints[i].x = static_cast<float>(pixels1[i].x());
        scene.images[0].features.keypoints[i].y = static_cast<float>(pixels1[i].y());
        scene.images[1].features.keypoints[i].x = static_cast<float>(pixels2[i].x());
        scene.images[1].features.keypoints[i].y = static_cast<float>(pixels2[i].y());
    }
    aetherscan::sfm::ImagePair pair(0, 1);
    pair.relative_pose = pose2;
    pair.weight_spatial = 1.F;
    for (Index i = 0; i < pixels1.size(); ++i) pair.matches.push_back({i, i});
    scene.pairs.push_back(pair);
    aetherscan::sfm::build_tracks(scene);
    expect(scene.tracks.size() == pixels1.size(), "one track per correspondence");
    const unsigned triangulated =
        aetherscan::sfm::triangulate_tracks(scene, false, 2.F, 0.5F);
    expect(triangulated > 80, "most tracks triangulate");

    if (failures == 0) {
        std::cout << "sfm geometry/triangulation tests passed\n";
        return 0;
    }
    std::cerr << failures << " failures\n";
    return 1;
}
