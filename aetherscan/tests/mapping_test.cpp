#include "aetherscan/sfm/star_init.hpp"
#include "aetherscan/sfm/resection.hpp"
#include "aetherscan/sfm/tracks.hpp"
#include "aetherscan/sfm/triangulation.hpp"

#include <cmath>
#include <iostream>
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

}  // namespace

int main() {
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

    if (failures == 0) {
        std::cout << "sfm mapping tests passed\n";
        return 0;
    }
    std::cerr << failures << " failures\n";
    return 1;
}
