#include "sfm/bundle.hpp"

#include <unordered_map>
#include <unordered_set>

namespace aetherscan::sfm {

BundleSummary run_bundle_adjustment(Scene& scene, const BundleOptions& options) {
    BundleSummary summary;

    std::unordered_set<Index> free_set(
        options.free_image_ids.begin(), options.free_image_ids.end());
    const bool use_filter = !options.free_image_ids.empty();

    std::vector<Index> camera_images;
    std::unordered_map<Index, Index> image_to_ba;
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        if (!scene.images[image_id].registered) continue;
        if (use_filter && !free_set.count(image_id) && !options.optimize_all_registered) {
            // Still include fixed cameras that observe optimized points — for v1
            // we only include free cameras when filtering is on.
            if (!free_set.count(image_id)) continue;
        }
        if (use_filter && !options.optimize_all_registered && !free_set.count(image_id))
            continue;
        image_to_ba[image_id] = static_cast<Index>(camera_images.size());
        camera_images.push_back(image_id);
    }
    // When free list is provided for local BA, also include them explicitly.
    if (use_filter) {
        camera_images.clear();
        image_to_ba.clear();
        for (Index image_id : options.free_image_ids) {
            if (image_id >= scene.images.size() || !scene.images[image_id].registered)
                continue;
            if (image_to_ba.count(image_id)) continue;
            image_to_ba[image_id] = static_cast<Index>(camera_images.size());
            camera_images.push_back(image_id);
        }
    }

    if (camera_images.size() < 2) return summary;

    ba::Problem problem;
    problem.poses.resize(camera_images.size());
    problem.intrinsics.resize(camera_images.size());
    for (std::size_t i = 0; i < camera_images.size(); ++i) {
        const Image& image = scene.images[camera_images[i]];
        const PinholeCamera& camera = scene.camera_of(image);
        const Quat q = image.pose.quaternion();
        problem.poses[i] = {q.w(), q.x(), q.y(), q.z(),
                            image.pose.C.x(), image.pose.C.y(), image.pose.C.z()};
        problem.intrinsics[i] = {
            camera.fx, camera.fy, camera.cx, camera.cy,
            camera.k1, camera.k2, camera.p1, camera.p2};
    }

    std::unordered_map<Index, Index> track_to_point;
    for (Index track_id = 0; track_id < scene.tracks.size(); ++track_id) {
        const Track& track = scene.tracks[track_id];
        if (!track.is_triangulated()) continue;
        bool visible = false;
        for (unsigned o = 0; o < track.num_inliers; ++o) {
            if (image_to_ba.count(track.observations[o].image_id)) {
                visible = true;
                break;
            }
        }
        if (!visible) continue;
        track_to_point[track_id] = static_cast<Index>(problem.points.size());
        problem.points.push_back({track.position.x(), track.position.y(), track.position.z()});
    }
    if (problem.points.empty()) return summary;

    for (const auto& [track_id, point_id] : track_to_point) {
        const Track& track = scene.tracks[track_id];
        for (unsigned o = 0; o < track.num_inliers; ++o) {
            const Observation& obs = track.observations[o];
            const auto cam_it = image_to_ba.find(obs.image_id);
            if (cam_it == image_to_ba.end()) continue;
            const auto& kp = scene.images[obs.image_id].features.keypoints[obs.feature_id];
            problem.observations.push_back(cam_it->second, point_id, kp.x, kp.y);
        }
    }
    if (problem.observations.size() == 0) return summary;

    ba::OptimizerOptions opt = options.optimizer;
    summary.optimizer = ba::optimize_cpu(problem, opt);
    summary.success = summary.optimizer.usable();
    summary.num_cameras = static_cast<unsigned>(problem.poses.size());
    summary.num_points = static_cast<unsigned>(problem.points.size());
    summary.num_observations = static_cast<unsigned>(problem.observations.size());
    if (!summary.success) return summary;

    for (std::size_t i = 0; i < camera_images.size(); ++i) {
        const ba::Pose& p = problem.poses[i];
        Pose3D pose;
        pose.R = Quat(p.qw, p.qx, p.qy, p.qz).normalized().toRotationMatrix();
        pose.C = Vec3(p.cx, p.cy, p.cz);
        scene.images[camera_images[i]].pose = pose;
    }
    if (options.optimize_points) {
        for (const auto& [track_id, point_id] : track_to_point) {
            const ba::Point3& p = problem.points[point_id];
            scene.tracks[track_id].position = Vec3(p.x, p.y, p.z);
        }
    }
    return summary;
}

}  // namespace aetherscan::sfm
