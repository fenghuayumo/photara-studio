#include "sfm/geometry.hpp"
#include "sfm/camera_selection.hpp"
#include "sfm/triangulation.hpp"
#include "sfm/tracks.hpp"

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {

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

    PinholeCamera fisheye = make_camera(400.0);
    fisheye.model = photara::CameraModel::opencv_fisheye;
    fisheye.k1 = 0.025; fisheye.k2 = -0.002;
    fisheye.p1 = 0.0002; fisheye.p2 = -0.00001;
    for (double angle : {0.0, 1e-9, 0.4, 0.9, 1.4, 1.55}) {
        const Vec3 ray(std::sin(angle)*0.8, std::sin(angle)*0.6, std::cos(angle));
        expect((fisheye.unproject_normalized(fisheye.project(ray))-ray).norm() < 1e-8,
               "fisheye optical axis and wide-angle round trip");
        const double t2 = angle*angle;
        const double radius = angle*(1+t2*(0.025+t2*(-0.002+t2*(0.0002-t2*0.00001))));
        expect((fisheye.project(ray)-Vec2(640+400*radius*0.8,360+400*radius*0.6)).norm()<1e-8,
               "fisheye projection matches angular polynomial");
    }
    std::vector<Vec2> fish1, fish2;
    std::vector<Vec3> fish_points;
    Pose3D fish_pose;
    fish_pose.R = Eigen::AngleAxisd(0.07, Vec3::UnitY()).toRotationMatrix();
    fish_pose.C = Vec3(0.5, 0.03, 0.0);
    std::mt19937 fish_rng(82);
    std::uniform_real_distribution<double> fish_xy(-3.0,3.0), fish_z(2.0,5.0);
    for (int i=0; i<200; ++i) {
        Vec3 point(fish_xy(fish_rng),fish_xy(fish_rng),fish_z(fish_rng));
        fish_points.push_back(point);
        fish1.push_back(fisheye.project(point));
        fish2.push_back(fisheye.project(fish_pose.transform_world_to_camera(point)));
    }
    // An unlocked focal still uses the fisheye E solver, never pinhole self-calibration.
    fisheye.trust_intrinsics = false;
    const auto fish_relative = photara::sfm::estimate_relative_pose(fish1,fish2,fisheye,fisheye);
    expect(fish_relative.success && fish_relative.num_inliers >= 190,
           "fisheye two-view estimation");
    if (fish_relative.success) {
        expect((fish_relative.pose.R-fish_pose.R).norm()<1e-4, "fisheye relative rotation");
        expect(fish_relative.pose.C.normalized().dot(fish_pose.C.normalized())>0.9999,
               "fisheye translation direction");
    }
    std::vector<Vec3> fish_bearings;
    for (const auto& pixel : fish2) fish_bearings.push_back(fisheye.unproject_normalized(pixel));
    const auto fish_absolute = photara::sfm::estimate_absolute_pose(fish_bearings,fish_points,fisheye);
    expect(fish_absolute.success, "fisheye absolute pose estimation");

    // Automatic model selection uses identical raw correspondences for each hypothesis.
    for (const auto model : {photara::CameraModel::pinhole,
                             photara::CameraModel::opencv_fisheye}) {
        auto truth = make_camera(model == photara::CameraModel::opencv_fisheye ? 448 : 640);
        truth.model = model;
        std::vector<photara::sfm::CameraModelProbe> probes;
        for (int pair=0; pair<3; ++pair) {
            photara::sfm::CameraModelProbe probe;
            Pose3D other;
            other.C = Vec3(0.4+0.15*pair,0.05*pair,0.1);
            other.R = Eigen::AngleAxisd(0.04*pair,Vec3::UnitY()).toRotationMatrix();
            std::mt19937 random(50+pair);
            std::uniform_real_distribution<double> xy(-2.8,2.8), depth(2,5);
            for (int i=0; i<500; ++i) {
                const Vec3 point(xy(random),xy(random),depth(random));
                const Vec2 a=truth.project(point), b=truth.project(other.transform_world_to_camera(point));
                if (a.x()<0 || a.x()>=1280 || b.x()<0 || b.x()>=1280 ||
                    a.y()<0 || a.y()>=720 || b.y()<0 || b.y()>=720) continue;
                probe.first.push_back(a); probe.second.push_back(b);
            }
            probes.push_back(std::move(probe));
        }
        const auto selection = photara::sfm::select_camera_model(truth,probes);
        std::cout << "auto model=" << static_cast<int>(model) << " selected=" << static_cast<int>(selection.model)
                  << " pinhole=" << selection.pinhole_score << " fish=" << selection.fisheye_score << '\n';
        expect(selection.model == model, "automatic camera model from synthetic geometry");
        if (model == photara::CameraModel::opencv_fisheye) {
            expect(selection.confident, "wide-angle fisheye evidence is decisive");
            auto with_planes=probes;
            for (int pair=0;pair<3;++pair) {
                photara::sfm::CameraModelProbe plane;
                Pose3D pose;
                pose.C=Vec3(0.3+0.1*pair,0.03*pair,0);
                for (int x=-9;x<=9;++x) for (int y=-5;y<=5;++y) {
                    const Vec3 point(0.2*x,0.15*y,3);
                    plane.first.push_back(truth.project(point));
                    plane.second.push_back(truth.project(pose.transform_world_to_camera(point)));
                }
                with_planes.push_back(std::move(plane));
            }
            const auto mixed_selection=photara::sfm::select_camera_model(truth,with_planes);
            expect(mixed_selection.model==model && mixed_selection.confident,
                   "excluded planar pairs must not dilute valid model evidence");
            auto initial = make_camera();
            const auto explicit_fish = photara::sfm::select_camera_model(
                initial, probes, 0, false, photara::CameraModel::opencv_fisheye);
            expect(explicit_fish.model == model &&
                   std::abs(explicit_fish.focal_pixels/truth.focal()-1) < 0.05,
                   "explicit fisheye estimates focal without a supplied calibration");
            auto distorted_truth = truth;
            distorted_truth.k1 = 0.035;
            distorted_truth.k2 = 0.01;
            distorted_truth.p1 = -0.003;
            distorted_truth.p2 = 0.001;
            auto distorted_probes = probes;
            for (auto& probe : distorted_probes) {
                for (auto& pixel : probe.first)
                    pixel = distorted_truth.project(truth.unproject_normalized(pixel));
                for (auto& pixel : probe.second)
                    pixel = distorted_truth.project(truth.unproject_normalized(pixel));
            }
            const auto distortion_fit = photara::sfm::select_camera_model(
                initial, distorted_probes, 0, false, model);
            auto fitted = truth;
            fitted.fx = fitted.fy = distortion_fit.focal_pixels;
            fitted.k1 = distortion_fit.distortion[0];
            fitted.k2 = distortion_fit.distortion[1];
            fitted.p1 = distortion_fit.distortion[2];
            fitted.p2 = distortion_fit.distortion[3];
            for (double angle : {0.1, 0.4, 0.7, 0.9}) {
                const Vec3 ray(std::sin(angle), 0, std::cos(angle));
                expect((fitted.project(ray)-distorted_truth.project(ray)).norm() < 2,
                       "joint fisheye focal/distortion fit predicts unseen rays");
            }
            // Some probe poses can be excluded from joint BA by a planar
            // hypothesis. Changing shared intrinsics must still leave a
            // calibration that predicts rays beyond the fitted point set.
            for (auto& probe : with_planes) {
                for (auto& pixel : probe.first)
                    pixel = distorted_truth.project(truth.unproject_normalized(pixel));
                for (auto& pixel : probe.second)
                    pixel = distorted_truth.project(truth.unproject_normalized(pixel));
            }
            const auto mixed_distortion_fit = photara::sfm::select_camera_model(
                initial, with_planes, 0, false, model);
            fitted.fx = fitted.fy = mixed_distortion_fit.focal_pixels;
            fitted.k1 = mixed_distortion_fit.distortion[0];
            fitted.k2 = mixed_distortion_fit.distortion[1];
            fitted.p1 = mixed_distortion_fit.distortion[2];
            fitted.p2 = mixed_distortion_fit.distortion[3];
            for (double angle : {0.1, 0.4, 0.7, 0.9}) {
                const Vec3 ray(std::sin(angle), 0, std::cos(angle));
                expect((fitted.project(ray)-distorted_truth.project(ray)).norm() < 2,
                       "mixed planar probes preserve fisheye ray calibration");
            }
        }
        else {
            const auto calibrated = photara::sfm::select_camera_model(truth,probes,640);
            expect(calibrated.model == model && std::abs(calibrated.focal_pixels-640)<1e-6,
                   "known focal is preserved when model evidence is ambiguous");
            if (!calibrated.confident)
                expect(calibrated.distortion == std::array<double,4>{},
                       "ambiguous pinhole calibration must not inject distortion");
        }
    }
    const auto ambiguous = photara::sfm::select_camera_model(make_camera(),{});
    expect(ambiguous.model == photara::CameraModel::pinhole && !ambiguous.confident,
           "insufficient model evidence falls back to pinhole");
    std::vector<photara::sfm::CameraModelProbe> planar_probes;
    const auto planar_camera = make_camera();
    for (int pair=0; pair<3; ++pair) {
        photara::sfm::CameraModelProbe probe;
        Pose3D pose;
        pose.C = Vec3(0.2+0.1*pair, 0.04*pair, 0.03);
        pose.R = Eigen::AngleAxisd(0.03*pair, Vec3::UnitY()).toRotationMatrix();
        for (int x=-9; x<=9; ++x) for (int y=-5; y<=5; ++y) {
            const Vec3 point(0.12*x, 0.12*y, 3);
            probe.first.push_back(planar_camera.project(point));
            probe.second.push_back(planar_camera.project(pose.transform_world_to_camera(point)));
        }
        planar_probes.push_back(std::move(probe));
    }
    const auto planar_selection = photara::sfm::select_camera_model(planar_camera, planar_probes);
    expect(planar_selection.model == photara::CameraModel::pinhole && !planar_selection.confident,
           "planar support cannot manufacture confidence for another projection model");
    photara::sfm::CameraModelProbe malformed;
    malformed.first.assign(60, Vec2(100, 100));
    malformed.second.assign(59, Vec2(110, 100));
    const auto explicit_empty = photara::sfm::select_camera_model(
        make_camera(), {malformed, malformed}, 0, false,
        photara::CameraModel::opencv_fisheye);
    expect(explicit_empty.model == photara::CameraModel::opencv_fisheye &&
           !explicit_empty.confident && explicit_empty.focal_pixels == 640,
           "invalid probe lengths preserve explicit fisheye initialization");

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

    photara::sfm::RelativePoseOptions options;
    options.min_inliers = 40;
    options.max_epipolar_error_px = 2.0;
    options.max_reproj_error_px = 2.0;
    options.min_ray_angle_deg = 0.5;
    const auto result =
        photara::sfm::estimate_relative_pose(pixels1, pixels2, cam, cam, options);
    expect(result.success, "relative pose should succeed");
    expect(result.num_inliers > 80, "relative pose should have many inliers");
    expect(result.pose.C.norm() > 0.1, "baseline should be non-trivial");
    // Synthetic scene has strong parallax — should not be flagged planar.
    expect(!result.degenerate_planar, "parallax scene should not be H-degenerate");

    auto untrusted_cam = cam;
    untrusted_cam.fx = untrusted_cam.fy = 950.0;
    untrusted_cam.trust_intrinsics = false;
    Pose3D self_cal_pose = pose2;
    self_cal_pose.R =
        Eigen::AngleAxisd(8.0 * 3.14159265358979323846 / 180.0, Vec3::UnitY())
            .toRotationMatrix();
    std::vector<Vec2> self_pixels1, self_pixels2;
    for (int i = 0; i < 200; ++i) {
        const Vec3 X(xy(rng), xy(rng), z(rng));
        self_pixels1.push_back(cam.project(X));
        self_pixels2.push_back(
            cam.project(self_cal_pose.transform_world_to_camera(X)));
    }
    const auto self_calibrated = photara::sfm::estimate_relative_pose(
        self_pixels1, self_pixels2, untrusted_cam, untrusted_cam, options);
    expect(self_calibrated.success, "shared-focal self-calibration should succeed");
    expect(
        self_calibrated.estimated_focal.has_value(),
        "shared-focal result should report its focal");
    expect(
        std::abs(*self_calibrated.estimated_focal - cam.fx) < 0.25 * cam.fx,
        "shared-focal estimate should be close to truth");
    expect(
        self_calibrated.num_inliers == self_calibrated.num_ransac_inliers,
        "unknown-focal pass must preserve F inliers for calibrated rerun");

    std::vector<Vec2> distorted_pixels1, distorted_pixels2;
    for (int i = 0; i < 120; ++i) {
        const Vec3 X(xy(rng), xy(rng), z(rng));
        distorted_pixels1.push_back(distorted.project(X));
        distorted_pixels2.push_back(
            distorted.project(pose2.transform_world_to_camera(X)));
    }
    const auto distorted_result = photara::sfm::estimate_relative_pose(
        distorted_pixels1, distorted_pixels2, distorted, distorted, options);
    expect(
        distorted_result.success && distorted_result.num_inliers > 80,
        "distortion-aware relative pose should succeed");
    auto fundamental_options = options;
    fundamental_options.force_fundamental = true;
    fundamental_options.decompose_fundamental = true;
    const auto distorted_fundamental =
        photara::sfm::estimate_relative_pose(
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
    photara::sfm::AbsolutePoseOptions abs_opts;
    abs_opts.min_inliers = 20;
    const auto abs =
        photara::sfm::estimate_absolute_pose(bearings, points, cam, abs_opts);
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
    photara::sfm::ImagePair pair(0, 1);
    pair.relative_pose = pose2;
    pair.weight_spatial = 1.F;
    for (Index i = 0; i < pixels1.size(); ++i) pair.matches.push_back({i, i});
    scene.pairs.push_back(pair);
    photara::sfm::build_tracks(scene);
    expect(scene.tracks.size() == pixels1.size(), "one track per correspondence");
    const unsigned triangulated =
        photara::sfm::triangulate_tracks(scene, false, 2.F, 0.5F);
    expect(triangulated > 80, "most tracks triangulate");

    if (failures == 0) {
        std::cout << "sfm geometry/triangulation tests passed\n";
        return 0;
    }
    std::cerr << failures << " failures\n";
    return 1;
}
