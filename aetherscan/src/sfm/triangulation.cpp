#include "sfm/triangulation.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <vector>

namespace aetherscan::sfm {
namespace {

constexpr double k_rad2deg = 180.0 / 3.14159265358979323846;

Vec2 keypoint_xy(const Image& image, const Index feature_id) {
    const auto& kp = image.features.keypoints[feature_id];
    return {kp.x, kp.y};
}

}  // namespace

float track_min_ray_angle_deg(const Track& track, const Scene& scene) {
    if (track.num_inliers < 2) return 0.F;
    double best_cos = 1.0;
    for (unsigned i = 0; i + 1 < track.num_inliers; ++i) {
        const Image& img_i = scene.images[track.observations[i].image_id];
        const Vec3 ray_i = (track.position - img_i.pose.C).normalized();
        for (unsigned j = i + 1; j < track.num_inliers; ++j) {
            const Image& img_j = scene.images[track.observations[j].image_id];
            const Vec3 ray_j = (track.position - img_j.pose.C).normalized();
            best_cos = std::min(best_cos, ray_i.dot(ray_j));
        }
    }
    return static_cast<float>(std::acos(std::clamp(best_cos, -1.0, 1.0)) * k_rad2deg);
}

unsigned triangulate_track(
    Track& track,
    const Scene& scene,
    const float reproj_threshold_px,
    const float min_angle_deg,
    const unsigned min_inliers) {
    if (!track.is_valid()) return 0;

    struct CamRow {
        Mat3 DR;
        Vec3 Dt;
        Index obs_index{};
    };
    std::vector<CamRow> cams;
    cams.reserve(track.observations.size());

    for (Index obs_index = 0; obs_index < track.observations.size(); ++obs_index) {
        const Observation& obs = track.observations[obs_index];
        const Image& image = scene.images[obs.image_id];
        if (!image.registered) continue;
        if (obs.feature_id >= image.features.keypoints.size()) continue;
        const PinholeCamera& camera = scene.camera_of(image);
        const Vec3 dir = camera.unproject_normalized(keypoint_xy(image, obs.feature_id));
        Mat3 Dcross;
        Dcross << 0, -dir.z(), dir.y(), dir.z(), 0, -dir.x(), -dir.y(), dir.x(), 0;
        cams.push_back({Dcross * image.pose.R, -Dcross * image.pose.translation(), obs_index});
    }
    if (cams.size() < min_inliers) {
        track.num_inliers = 0;
        return 0;
    }

    Eigen::MatrixXd A(2 * static_cast<Eigen::Index>(cams.size()), 3);
    Eigen::VectorXd b(2 * static_cast<Eigen::Index>(cams.size()));
    for (std::size_t i = 0; i < cams.size(); ++i) {
        A.row(static_cast<Eigen::Index>(2 * i)) = cams[i].DR.row(0);
        b(static_cast<Eigen::Index>(2 * i)) = cams[i].Dt(0);
        A.row(static_cast<Eigen::Index>(2 * i + 1)) = cams[i].DR.row(1);
        b(static_cast<Eigen::Index>(2 * i + 1)) = cams[i].Dt(1);
    }
    track.position = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    if (!track.position.allFinite()) {
        track.num_inliers = 0;
        return 0;
    }

    std::vector<Index> inlier_obs;
    std::vector<CamRow> inlier_cams;
    inlier_obs.reserve(cams.size());
    inlier_cams.reserve(cams.size());
    for (const CamRow& cam : cams) {
        const Observation& obs = track.observations[cam.obs_index];
        const Image& image = scene.images[obs.image_id];
        const PinholeCamera& camera = scene.camera_of(image);
        Vec2 proj;
        if (!camera.project_checked(image.pose.transform_world_to_camera(track.position), proj))
            continue;
        if ((proj - keypoint_xy(image, obs.feature_id)).norm() > reproj_threshold_px) continue;
        inlier_obs.push_back(cam.obs_index);
        inlier_cams.push_back(cam);
    }
    if (inlier_obs.size() < min_inliers) {
        track.num_inliers = 0;
        return 0;
    }

    std::vector<char> used(track.observations.size(), 0);
    std::vector<Observation> ordered;
    ordered.reserve(track.observations.size());
    for (Index idx : inlier_obs) {
        ordered.push_back(track.observations[idx]);
        used[idx] = 1;
    }
    for (std::size_t i = 0; i < track.observations.size(); ++i) {
        if (!used[i]) ordered.push_back(track.observations[i]);
    }
    track.observations = std::move(ordered);
    track.num_inliers =
        static_cast<std::uint8_t>(std::min<std::size_t>(inlier_obs.size(), 255));

    if (track_min_ray_angle_deg(track, scene) < min_angle_deg) {
        track.num_inliers = 0;
        return 0;
    }

    if (inlier_cams.size() != cams.size()) {
        Eigen::MatrixXd A2(2 * static_cast<Eigen::Index>(inlier_cams.size()), 3);
        Eigen::VectorXd b2(2 * static_cast<Eigen::Index>(inlier_cams.size()));
        for (std::size_t k = 0; k < inlier_cams.size(); ++k) {
            A2.row(static_cast<Eigen::Index>(2 * k)) = inlier_cams[k].DR.row(0);
            b2(static_cast<Eigen::Index>(2 * k)) = inlier_cams[k].Dt(0);
            A2.row(static_cast<Eigen::Index>(2 * k + 1)) = inlier_cams[k].DR.row(1);
            b2(static_cast<Eigen::Index>(2 * k + 1)) = inlier_cams[k].Dt(1);
        }
        track.position = A2.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b2);
        if (!track.position.allFinite()) {
            track.num_inliers = 0;
            return 0;
        }
    }
    return track.num_inliers;
}

unsigned triangulate_tracks(
    Scene& scene,
    const bool outliers_only,
    const float reproj_threshold_px,
    const float min_angle_deg) {
    unsigned inlier_tracks = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : inlier_tracks) schedule(dynamic)
#endif
    for (int i = 0; i < static_cast<int>(scene.tracks.size()); ++i) {
        Track& track = scene.tracks[static_cast<std::size_t>(i)];
        if (outliers_only && track.is_triangulated()) {
            ++inlier_tracks;
            continue;
        }
        if (triangulate_track(track, scene, reproj_threshold_px, min_angle_deg) >= 2)
            ++inlier_tracks;
    }
    return inlier_tracks;
}

}  // namespace aetherscan::sfm
