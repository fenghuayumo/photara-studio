#pragma once
#include "sfm/geometry.hpp"
#include "sfm/triangulation.hpp"
#include "ba/optimizer.hpp"
#include <map>

// Offline hypothesis test: reference poses are never read. Reconstruct depths
// from at least three cameras outside the tested intrinsic group, fit on 80%
// of those observations, and report the remaining 20% separately.
inline void probe_small_group_focals(aetherscan::sfm::Scene& scene) {
    using namespace aetherscan::sfm;
    namespace ba = aetherscan::ba;
    struct Sample { Index image, feature; Vec3 point; bool heldout; };
    std::vector<unsigned> counts(scene.cameras.size());
    for (const auto& image : scene.images)
        if (image.registered) ++counts[image.camera_id];
    for (Index group = 0; group < scene.cameras.size(); ++group) {
        const auto initial_camera = scene.cameras[group];
        if (initial_camera.trust_intrinsics || !counts[group] || counts[group] > 8) continue;
        std::vector<Sample> samples;
        for (Index t = 0; t < scene.tracks.size(); ++t) {
            const auto& source = scene.tracks[t];
            Track depth;
            std::vector<Observation> targets;
            for (const auto& o : source.observations) {
                const auto& image = scene.images[o.image_id];
                if (!image.registered) continue;
                if (image.camera_id == group) targets.push_back(o);
                else if (counts[image.camera_id] > 8) depth.observations.push_back(o);
            }
            if (targets.empty() || depth.observations.size() < 3) continue;
            if (triangulate_track(depth, scene, 2.F, 1.F) < 3) continue;
            for (const auto& o : targets)
                samples.push_back({o.image_id, o.feature_id, depth.position, t % 5 == 0});
        }
        std::map<Index, Index> local_ids;
        for (const auto& sample : samples)
            if (!local_ids.contains(sample.image))
                local_ids.emplace(sample.image, static_cast<Index>(local_ids.size()));
        if (local_ids.size() != counts[group] || samples.size() < 100) continue;
        const auto score = [&](const PinholeCamera& camera, const std::vector<Pose3D>& poses) {
            double cost = 0;
            unsigned inliers = 0, total = 0;
            for (const auto& sample : samples) {
                if (!sample.heldout) continue;
                const auto& kp = scene.images[sample.image].features.keypoints[sample.feature];
                const Vec3 p = poses[local_ids.at(sample.image)].transform_world_to_camera(sample.point);
                double error = 4;
                if (p.z() > 0) {
                    const double e = (camera.project(p) - Vec2(kp.x,kp.y)).squaredNorm();
                    if (std::isfinite(e)) error = std::min(e,4.0);
                }
                cost += error;
                inliers += error < 4;
                ++total;
            }
            return std::tuple{total ? cost/total : 4.0, inliers, total};
        };
        std::vector<Pose3D> original(local_ids.size());
        for (const auto& [image,local] : local_ids) original[local] = scene.images[image].pose;
        auto [best_score, original_inliers, heldout_count] = score(initial_camera,original);
        PinholeCamera best_camera = initial_camera;
        auto best_poses = original;
        std::cout << "focal baseline group=" << group << " images=" << counts[group]
                  << " samples=" << samples.size() << " focal=" << initial_camera.fx
                  << " heldout=" << original_inliers << '/' << heldout_count << " score=" << best_score << '\n';
        for (const double ratio : {0.7,0.85,1.0,1.15,1.3}) {
            PinholeCamera camera = initial_camera;
            camera.fx *= ratio; camera.fy *= ratio;
            ba::Problem problem;
            problem.intrinsics.push_back({camera.fx,camera.fy,camera.cx,camera.cy,
                camera.k1,camera.k2,camera.p1,camera.p2,camera.model});
            problem.poses.resize(local_ids.size());
            problem.pose_intrinsic.resize(local_ids.size(),0);
            bool valid = true;
            for (const auto& [image,local] : local_ids) {
                std::vector<Vec3> points,bearings;
                for (const auto& sample : samples) {
                    if (sample.image != image || sample.heldout) continue;
                    const auto& kp = scene.images[image].features.keypoints[sample.feature];
                    points.push_back(sample.point); bearings.push_back(camera.unproject(Vec2(kp.x,kp.y)));
                }
                AbsolutePoseOptions options;
                options.min_inliers=30; options.max_reproj_error_px=2; options.max_iterations=5000;
                const auto fit = estimate_absolute_pose(bearings,points,camera,options);
                if (!fit.success || fit.num_inliers < points.size()/2) { valid=false; break; }
                const auto q = fit.pose.quaternion();
                problem.poses[local] = {q.w(),q.x(),q.y(),q.z(),fit.pose.C.x(),fit.pose.C.y(),fit.pose.C.z()};
            }
            if (!valid) { std::cout << "focal candidate group=" << group << " ratio=" << ratio << " rejected=pnp\n"; continue; }
            for (const auto& sample : samples) {
                if (sample.heldout) continue;
                const auto& kp = scene.images[sample.image].features.keypoints[sample.feature];
                const auto point = static_cast<ba::Index>(problem.points.size());
                problem.points.push_back({sample.point.x(),sample.point.y(),sample.point.z()});
                problem.observations.push_back(local_ids.at(sample.image),point,kp.x,kp.y);
            }
            ba::OptimizerOptions options;
            options.optimize_points=false; options.fix_first_point=false; options.fix_first_pose=false;
            options.optimize_focal=true; options.focal_prior_weight=0.05;
            options.maximum_iterations=60; options.huber_delta=1;
            const auto fit = ba::optimize_cpu(problem,options);
            if (!fit.usable()) continue;
            camera.fx=problem.intrinsics[0].fx; camera.fy=problem.intrinsics[0].fy;
            std::vector<Pose3D> poses(problem.poses.size());
            for (Index i=0;i<poses.size();++i) {
                const auto& p=problem.poses[i];
                poses[i] = {Quat(p.qw,p.qx,p.qy,p.qz).normalized().toRotationMatrix(),Vec3(p.cx,p.cy,p.cz)};
            }
            const auto [candidate_score,inliers,total] = score(camera,poses);
            std::cout << "focal candidate group=" << group << " ratio=" << ratio << " focal=" << camera.fx
                      << " heldout=" << inliers << '/' << total << " score=" << candidate_score << '\n';
            if (inliers >= 30 && candidate_score < best_score * 0.95) {
                best_score=candidate_score; best_camera=camera; best_poses=std::move(poses);
            }
        }
        scene.cameras[group]=best_camera;
        for (const auto& [image,local] : local_ids) scene.images[image].pose=best_poses[local];
    }
    // Rebuild depths after the experiment so the diagnostic reprojection uses
    // the candidate calibration, rather than stale point coordinates.
    triangulate_tracks(scene, true, 2.F, 1.F);
    filter_tracks(scene,2.F,1.F,0.F,0.F);
}
