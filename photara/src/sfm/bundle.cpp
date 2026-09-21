#include "sfm/bundle.hpp"
#include "core/logging.hpp"
#include "sfm/triangulation.hpp"

#include <atomic>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace photara::sfm {
namespace {

float median_parallax_deg_for_camera(
    const Scene& scene,
    const Index camera_id,
    const std::unordered_set<Index>& active_images) {
    std::vector<float> angles;
    angles.reserve(256);
    for (const Track& track : scene.tracks) {
        if (!track.is_triangulated() || track.num_inliers < 2) continue;
        bool sees_camera = false;
        for (unsigned o = 0; o < track.num_inliers; ++o) {
            const Index image_id = track.observations[o].image_id;
            if (!active_images.count(image_id)) continue;
            if (scene.images[image_id].camera_id == camera_id) {
                sees_camera = true;
                break;
            }
        }
        if (!sees_camera) continue;
        const float angle = track_min_ray_angle_deg(track, scene);
        if (angle > 0.F) angles.push_back(angle);
    }
    if (angles.empty()) return 0.F;
    const std::size_t mid = angles.size() / 2;
    std::nth_element(angles.begin(), angles.begin() + mid, angles.end());
    return angles[mid];
}

std::atomic<BundleBackendPreference>& backend_preference_storage() {
    static std::atomic<BundleBackendPreference> value{
        BundleBackendPreference::automatic};
    return value;
}

}  // namespace

void set_bundle_backend_preference(const BundleBackendPreference preference) {
    backend_preference_storage().store(preference, std::memory_order_relaxed);
}

BundleBackendPreference bundle_backend_preference() {
    return backend_preference_storage().load(std::memory_order_relaxed);
}

BundleSummary run_bundle_adjustment(Scene& scene, const BundleOptions& options) {
    BundleSummary summary;

    std::unordered_set<Index> free_set(
        options.free_image_ids.begin(), options.free_image_ids.end());
    std::unordered_set<Index> fixed_set(
        options.fixed_image_ids.begin(), options.fixed_image_ids.end());
    const bool use_filter = !free_set.empty() || !fixed_set.empty();

    std::vector<Index> camera_images;
    std::unordered_map<Index, Index> image_to_ba;
    if (use_filter) {
        const auto append = [&](const std::vector<Index>& ids) {
            for (Index image_id : ids) {
                if (image_id >= scene.images.size() ||
                    !scene.images[image_id].registered ||
                    image_to_ba.count(image_id))
                    continue;
                image_to_ba[image_id] = static_cast<Index>(camera_images.size());
                camera_images.push_back(image_id);
            }
        };
        append(options.free_image_ids);
        append(options.fixed_image_ids);
    } else {
        for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
            if (!scene.images[image_id].registered) continue;
            image_to_ba[image_id] = static_cast<Index>(camera_images.size());
            camera_images.push_back(image_id);
        }
    }

    if (camera_images.size() < 2) return summary;

    ba::Problem problem;
    problem.poses.resize(camera_images.size());
    problem.pose_intrinsic.resize(camera_images.size());
    problem.pose_constant.resize(camera_images.size(), 0);
    std::unordered_map<Index, Index> scene_camera_to_group;
    std::vector<Index> group_to_scene_camera;
    for (std::size_t i = 0; i < camera_images.size(); ++i) {
        const Image& image = scene.images[camera_images[i]];
        const PinholeCamera& camera = scene.camera_of(image);
        const Quat q = image.pose.quaternion();
        problem.poses[i] = {q.w(), q.x(), q.y(), q.z(),
                            image.pose.C.x(), image.pose.C.y(), image.pose.C.z()};
        auto [group_it, inserted] = scene_camera_to_group.try_emplace(
            image.camera_id, static_cast<Index>(problem.intrinsics.size()));
        if (inserted) {
            problem.intrinsics.push_back({
                camera.fx, camera.fy, camera.cx, camera.cy,
                camera.k1, camera.k2, camera.p1, camera.p2, camera.model});
            group_to_scene_camera.push_back(image.camera_id);
        }
        problem.pose_intrinsic[i] = group_it->second;
        problem.pose_constant[i] =
            fixed_set.count(camera_images[i]) ? std::uint8_t{1} : std::uint8_t{0};
    }
    problem.initial_intrinsics = problem.intrinsics;
    problem.intrinsic_constant.assign(problem.intrinsics.size(), 0);
    for (std::size_t group = 0; group < group_to_scene_camera.size(); ++group) {
        const PinholeCamera& camera =
            scene.cameras[group_to_scene_camera[group]];
        if (camera.focal_prior > 1.0) {
            problem.initial_intrinsics[group].fx = camera.focal_prior;
            problem.initial_intrinsics[group].fy = camera.focal_prior;
        }
        if (camera.trust_intrinsics)
            problem.intrinsic_constant[group] = 1;
    }

    std::unordered_map<Index, Index> track_to_point;
    for (Index track_id = 0; track_id < scene.tracks.size(); ++track_id) {
        const Track& track = scene.tracks[track_id];
        if (!track.is_triangulated()) continue;
        bool visible = false;
        for (unsigned o = 0; o < track.num_inliers; ++o) {
            const Index image_id = track.observations[o].image_id;
            const bool activates_point =
                use_filter ? free_set.count(image_id) != 0
                           : image_to_ba.count(image_id) != 0;
            if (activates_point) {
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
    opt.optimize_points = opt.optimize_points && options.optimize_points;
    if (!fixed_set.empty()) opt.fix_first_pose = false;

    const bool optimize_any_intrinsics =
        opt.optimize_focal || opt.optimize_principal_point ||
        opt.optimize_distortion;
    if (optimize_any_intrinsics && options.gate_intrinsics_by_observability) {
        std::unordered_set<Index> active_images(
            camera_images.begin(), camera_images.end());
        std::vector<unsigned> views_per_group(problem.intrinsics.size(), 0);
        for (const Index image_id : camera_images) {
            if (fixed_set.count(image_id)) continue;
            const Index group =
                problem.pose_intrinsic[image_to_ba[image_id]];
            ++views_per_group[group];
        }
        unsigned frozen = 0;
        for (std::size_t group = 0; group < problem.intrinsics.size(); ++group) {
            const Index camera_id = group_to_scene_camera[group];
            const float parallax = median_parallax_deg_for_camera(
                scene, camera_id, active_images);
            if (views_per_group[group] < options.min_views_for_intrinsics ||
                parallax < options.min_median_parallax_deg) {
                problem.intrinsic_constant[group] = 1;
                ++frozen;
            }
        }
        if (frozen > 0) {
            core::Logger::instance().info(
                "bundle intrinsics gated: frozen=", frozen, '/',
                problem.intrinsics.size(),
                " min_views=", options.min_views_for_intrinsics,
                " min_parallax_deg=", options.min_median_parallax_deg);
        }
        bool any_free = false;
        for (const std::uint8_t flag : problem.intrinsic_constant)
            if (flag == 0) any_free = true;
        if (!any_free) {
            opt.optimize_focal = false;
            opt.optimize_principal_point = false;
            opt.optimize_distortion = false;
        }
    }

    bool use_cuda = false;
#if defined(PHOTARA_HAS_CUDA)
    const bool has_partial_pose_locks = std::any_of(
        problem.pose_constant.begin(), problem.pose_constant.end(),
        [](const std::uint8_t value) { return value != 0; });
    // The CUDA joint Schur/PCG solver now estimates shared intrinsic groups
    // (focal / aspect / principal point / distortion) together with poses and
    // points. Partial pose locks (local BA with fixed boundary views) still
    // use the CPU backend.
    const bool supported_parameterization =
        (opt.optimize_rotations || opt.optimize_translations) &&
        !has_partial_pose_locks;
    const BundleBackendPreference backend = bundle_backend_preference();
    use_cuda = options.prefer_cuda && backend != BundleBackendPreference::cpu &&
        supported_parameterization &&
        problem.observations.size() >= options.cuda_min_observations &&
        ba::CudaOptimizer::is_available();
    if (backend == BundleBackendPreference::cuda && !use_cuda &&
        options.prefer_cuda && supported_parameterization) {
        core::Logger::instance().warning(
            "ba-backend=cuda requested but this solve stays on the CPU: "
            "observations=", problem.observations.size(),
            " minimum=", options.cuda_min_observations,
            " cuda_available=", ba::CudaOptimizer::is_available() ? 1 : 0);
    }
    if (use_cuda) {
        try {
            core::Logger::instance().info(
                "bundle backend=cuda device=", ba::CudaOptimizer::device_name(),
                " observations=", problem.observations.size());
            summary.optimizer = ba::optimize_cuda(problem, opt);
            summary.backend = BundleBackend::cuda;
            if (!summary.optimizer.usable()) {
                core::Logger::instance().warning(
                    "CUDA bundle adjustment produced an unusable step; "
                    "continuing with CPU from the last accepted state");
                use_cuda = false;
            }
        } catch (const std::exception& error) {
            core::Logger::instance().warning(
                "CUDA bundle adjustment unavailable at runtime; falling back "
                "to CPU: ", error.what());
            use_cuda = false;
        }
    }
#endif
    if (!use_cuda) {
        summary.optimizer = ba::optimize_cpu(problem, opt);
        summary.backend = BundleBackend::cpu;
    }
    summary.success = summary.optimizer.usable();
    summary.num_cameras = static_cast<unsigned>(problem.poses.size());
    summary.num_points = static_cast<unsigned>(problem.points.size());
    summary.num_observations = static_cast<unsigned>(problem.observations.size());
    core::Logger::instance().info(
        "bundle adjustment: cameras=", summary.num_cameras,
        " points=", summary.num_points,
        " observations=", summary.num_observations,
        " backend=", summary.backend == BundleBackend::cuda ? "cuda" : "cpu",
        ' ',
        summary.optimizer.brief_report());
    if (!summary.success) return summary;

    for (std::size_t i = 0; i < camera_images.size(); ++i) {
        const ba::Pose& p = problem.poses[i];
        Pose3D pose;
        pose.R = Quat(p.qw, p.qx, p.qy, p.qz).normalized().toRotationMatrix();
        pose.C = Vec3(p.cx, p.cy, p.cz);
        scene.images[camera_images[i]].pose = pose;
    }
    if (options.write_intrinsics &&
        (opt.optimize_focal || opt.optimize_principal_point || opt.optimize_distortion) &&
        !problem.intrinsics.empty()) {
        for (std::size_t group = 0; group < problem.intrinsics.size(); ++group) {
            if (problem.is_intrinsic_constant(group)) continue;
            const Index camera_id = group_to_scene_camera[group];
            const ba::PinholeIntrinsics& optimized =
                problem.intrinsics[group];
            PinholeCamera& camera = scene.cameras[camera_id];
            camera.fx = optimized.fx;
            camera.fy = optimized.fy;
            camera.cx = optimized.cx;
            camera.cy = optimized.cy;
            camera.k1 = optimized.k1;
            camera.k2 = optimized.k2;
            camera.p1 = optimized.p1;
            camera.p2 = optimized.p2;
        }
    }
    if (options.optimize_points) {
        for (const auto& [track_id, point_id] : track_to_point) {
            const ba::Point3& p = problem.points[point_id];
            scene.tracks[track_id].position = Vec3(p.x, p.y, p.z);
        }
    }
    return summary;
}

}  // namespace photara::sfm
