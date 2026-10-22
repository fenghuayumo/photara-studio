#pragma once

#include "ba/optimizer.hpp"
#include "sfm/scene.hpp"
#include <algorithm>
#include <map>
#include <set>

namespace photara::sfm::detail {
struct ResectionFitPoint {
    Index feature, track;
    Vec3 position;
    std::vector<Index> anchors;
};

// This disposable problem contains fit observations only. Its point updates
// are nuisance variables, never replacements for the frozen scene landmarks.
inline Pose3D refine_resection_depths(
    const Scene& scene, Index target, const Pose3D& initial,
    const std::vector<ResectionFitPoint>& fit) {
    if (fit.size() < 18) return initial;
    ba::Problem problem;
    std::map<Index, Index> cameras;
    const auto add_camera = [&](Index id) {
        auto [it, added] = cameras.emplace(id, static_cast<Index>(problem.poses.size()));
        if (added) {
            const auto& pose = id == target ? initial : scene.images[id].pose;
            const auto q = pose.quaternion();
            problem.poses.push_back({q.w(),q.x(),q.y(),q.z(),pose.C.x(),pose.C.y(),pose.C.z()});
            const auto& camera = scene.camera_of(scene.images[id]);
            problem.intrinsics.push_back({camera.fx,camera.fy,camera.cx,camera.cy,
                camera.k1,camera.k2,camera.p1,camera.p2,camera.model});
            problem.pose_constant.push_back(id != target);
            problem.intrinsic_constant.push_back(1);
        }
        return it->second;
    };
    const Index candidate = add_camera(target);
    for (const auto& point : fit) {
        const Index p = static_cast<Index>(problem.points.size());
        problem.points.push_back({point.position.x(),point.position.y(),point.position.z()});
        const auto& measured = scene.images[target].features.keypoints[point.feature];
        problem.observations.push_back(candidate,p,measured.x,measured.y);
        std::set<Index> included;
        for (const auto& obs : scene.tracks[point.track].observations) {
            if (obs.image_id == target || !scene.images[obs.image_id].registered ||
                std::find(point.anchors.begin(),point.anchors.end(),obs.image_id) == point.anchors.end() ||
                !included.insert(obs.image_id).second) continue;
            const auto& kp = scene.images[obs.image_id].features.keypoints[obs.feature_id];
            const auto camera = add_camera(obs.image_id);
            problem.observations.push_back(camera,p,kp.x,kp.y);
        }
    }
    ba::OptimizerOptions options;
    options.maximum_iterations = 40;
    options.huber_delta = 1;
    options.fix_first_pose = false;
    options.fix_first_point = false;
    const auto result = ba::optimize_cpu(problem, options);
    if (!result.usable() || result.final_cost > result.initial_cost) return initial;
    const auto& optimized = problem.poses[candidate];
    Pose3D pose;
    pose.R = Quat(optimized.qw,optimized.qx,optimized.qy,optimized.qz).normalized().toRotationMatrix();
    pose.C = Vec3(optimized.cx,optimized.cy,optimized.cz);
    return pose.R.allFinite() && pose.C.allFinite() ? pose : initial;
}
} // namespace photara::sfm::detail
