#pragma once

#include "sfm/geometry.hpp"
#include "sfm/scene.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace photara::sfm::detail {

// Search proposal only. Two independently verified anchor-to-target edges
// supply rotation and intersecting center rays when descriptor PnP cannot
// find a seed. No inherited target pose or main-map point is used here.
inline AbsolutePoseResult relative_resection_search_seed(
    const Scene& scene, Index target, const std::vector<Vec3>& bearings,
    const std::vector<Vec3>& points) {
    AbsolutePoseResult best;
    if (target >= scene.images.size() || bearings.size() != points.size()) return best;
    struct Ray { Vec3 origin, direction; Mat3 rotation; };
    std::vector<Ray> rays;
    for (const auto& pair : scene.pairs) {
        if (pair.id2 != target || pair.id1 >= scene.images.size() ||
            !scene.images[pair.id1].registered || !pair.active ||
            !pair.relative_pose || pair.zero_baseline || pair.matches.size() < 15) continue;
        const auto& anchor = scene.images[pair.id1].pose;
        const auto& relative = *pair.relative_pose;
        if (!relative.C.allFinite() || relative.C.norm() <= 1e-10) continue;
        rays.push_back({anchor.C, anchor.R.transpose()*relative.C.normalized(), relative.R*anchor.R});
    }
    const auto& camera = scene.camera_of(scene.images[target]);
    for (std::size_t a = 0; a < rays.size(); ++a) for (std::size_t b = a+1; b < rays.size(); ++b) {
        const auto& first = rays[a]; const auto& second = rays[b];
        const double cosine = first.direction.dot(second.direction);
        const double denominator = 1-cosine*cosine;
        if (denominator < 0.0027) continue; // about three degrees
        const double rotation_cosine = ((first.rotation*second.rotation.transpose()).trace()-1)*0.5;
        if (rotation_cosine < std::cos(3.0*3.141592653589793/180.0)) continue;
        const Vec3 baseline = second.origin-first.origin;
        if (baseline.norm() <= 1e-10) continue;
        const double along_first = first.direction.dot(baseline);
        const double along_second = second.direction.dot(baseline);
        const double depth_second = (cosine*along_first-along_second)/denominator;
        const double depth_first = along_first+cosine*depth_second;
        if (depth_first <= 0 || depth_second <= 0) continue;
        const Vec3 first_center = first.origin+depth_first*first.direction;
        const Vec3 second_center = second.origin+depth_second*second.direction;
        if ((first_center-second_center).norm() > 0.1*baseline.norm()) continue;
        Pose3D pose;
        pose.C = 0.5*(first_center+second_center);
        pose.R = Eigen::Quaterniond(first.rotation).slerp(0.5, Eigen::Quaterniond(second.rotation)).toRotationMatrix();
        unsigned support = 0;
        for (std::size_t i = 0; i < points.size(); ++i) {
            Vec2 measured;
            if (camera.project_checked(bearings[i], measured) &&
                camera.checked_reprojection_error(pose.transform_world_to_camera(points[i]), measured) <= 8)
                ++support;
        }
        if (support >= 6 && support > best.num_inliers) {
            best.success = true;
            best.pose = pose;
            best.num_inliers = support;
        }
    }
    return best;
}
} // namespace photara::sfm::detail
