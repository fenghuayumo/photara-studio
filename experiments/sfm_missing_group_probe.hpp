#pragma once

#include "sfm/frontend.hpp"
#include "sfm/star_init.hpp"
#include "sfm/resection.hpp"
#include "sfm/submap_recovery.hpp"
#include "sfm/tracks.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <numeric>
#include <set>

// Offline experiment: jointly reconstruct missing connected runs without any
// inherited target poses or points, then use the existing independent Sim(3)
// fit/holdout and final support gates. Never called by production reconstruction.
inline unsigned probe_missing_groups(photara::sfm::Scene& scene, std::ostream& reports) {
    using namespace photara::sfm;
    std::vector<Index> roots(scene.images.size());
    std::iota(roots.begin(), roots.end(), 0);
    const auto root = [&](Index id) { while (roots[id] != id) id = roots[id]; return id; };
    const auto join = [&](Index a, Index b) {
        if (a >= roots.size() || b >= roots.size() || scene.images[a].registered || scene.images[b].registered) return;
        const auto ra = root(a), rb = root(b); if (ra != rb) roots[rb] = ra;
    };
    for (Index id = 1; id < scene.images.size(); ++id) join(id-1, id);
    for (const auto& pair : scene.pairs)
        if (pair.relative_pose && pair.matches.size() >= 15 && !pair.zero_baseline) join(pair.id1, pair.id2);
    std::map<Index, std::vector<Index>> groups;
    for (Index id = 0; id < scene.images.size(); ++id)
        if (!scene.images[id].registered) groups[root(id)].push_back(id);
    reports << "group,missing,local_registered,candidates,shared,fit,heldout,heldout_inliers,accepted,reason\n";
    unsigned recovered = 0, attempted = 0;
    for (const auto& [group_id, group] : groups) {
        if (group.size() < 4 || group.size() > 32 || attempted >= 2) continue;
        std::map<Index, double> scores;
        for (Index id : group) {
            const auto anchors = select_missing_view_anchors(scene, id);
            for (std::size_t j = 0; j < anchors.size(); ++j) scores[anchors[j]] += 1.0/(1+j);
        }
        std::vector<Index> anchors;
        for (const auto& [id, score] : scores) anchors.push_back(id);
        std::stable_sort(anchors.begin(), anchors.end(), [&](Index a, Index b) { return scores[a] > scores[b]; });
        if (anchors.size() > 16) anchors.resize(16);
        if (anchors.size() < 4) continue;
        auto globals = anchors;
        globals.insert(globals.end(), group.begin(), group.end());
        if (std::any_of(globals.begin(), globals.end(), [&](Index id) {
            return !std::filesystem::is_regular_file(scene.images[id].path); })) continue;
        ++attempted;
        std::vector<std::filesystem::path> paths;
        for (Index id : globals) paths.push_back(scene.images[id].path);
        FrontEndOptions options;
        options.thread_count = scene.thread_count;
        options.neighbor_window = 0;
        options.augment_sequential_with_retrieval = false;
        options.progressive_pair_expansion = false;
        options.max_features = 54000;
        options.sift_contrast_threshold = 0.001;
        options.match_ratio = 0.85F;
        const auto& calibration = scene.camera_of(scene.images[globals.front()]);
        options.camera_model = calibration.model;
        options.focal_pixels = calibration.fx;
        options.trust_focal_pixels = true;
        auto local = run_frontend(paths, options).scene;
        // Match by full path rather than assuming frontend image enumeration.
        std::map<std::filesystem::path, Index> by_path;
        for (Index id : globals) by_path.emplace(scene.images[id].path, id);
        globals.clear();
        std::vector<std::uint8_t> stable(local.images.size());
        std::vector<Index> targets;
        local.cameras = scene.cameras;
        for (auto& camera : local.cameras) camera.trust_intrinsics = true;
        for (Index id = 0; id < local.images.size(); ++id) {
            const Index global = by_path.at(local.images[id].path);
            globals.push_back(global);
            local.images[id].camera_id = scene.images[global].camera_id;
            stable[id] = scene.images[global].registered;
            local.images[id].registered = stable[id];
            if (stable[id]) local.images[id].pose = scene.images[global].pose;
            else targets.push_back(id);
        }
        // Frontend focal initialization does not carry native aspect/distortion.
        // Reverify its proposed correspondences with the exact frozen native
        // calibration before using any relative pose or rebuilding tracks.
        for (auto& pair : local.pairs) {
            std::vector<Vec2> first, second;
            for (const auto& match : pair.matches) {
                const auto& a = local.images[pair.id1].features.keypoints[match.query];
                const auto& b = local.images[pair.id2].features.keypoints[match.train];
                first.emplace_back(a.x,a.y); second.emplace_back(b.x,b.y);
            }
            RelativePoseOptions verification;
            verification.min_inliers = 15;
            const auto result = estimate_relative_pose(first,second,
                local.camera_of(local.images[pair.id1]),local.camera_of(local.images[pair.id2]),verification);
            pair.active = result.success;
            if (!result.success) continue;
            std::vector<FeatureMatch> inliers;
            for (std::size_t i = 0; i < pair.matches.size(); ++i)
                if (i < result.inlier_mask.size() && result.inlier_mask[i]) inliers.push_back(pair.matches[i]);
            pair.matches = std::move(inliers);
            pair.relative_pose = result.pose;
            pair.degenerate_planar = result.degenerate_planar;
            pair.weight_geometry = result.degenerate_planar ? 0.05F : 1.F;
            pair.weight_spatial = result.weight_spatial;
            pair.mean_ray_angle = result.mean_ray_angle;
        }
        build_tracks(local);
        auto draft = make_independent_submap(local, targets);
        build_tracks(draft.scene);
        StarInitConfig star;
        if (star_initialize(draft.scene, star)) {
            ResectionConfig resection;
            resection.full_ba.optimize_focal = false;
            resection.full_ba.optimize_aspect_ratio = false;
            resection.full_ba.optimize_distortion = false;
            register_images(draft.scene, resection);
        }
        std::vector<Index> candidates;
        for (Index id = 0; id < draft.scene.images.size(); ++id)
            if (draft.scene.images[id].registered) candidates.push_back(targets[id]);
        if (candidates.size() < 4) {
            reports << group_id << ',' << group.size() << ',' << draft.scene.registered_count()
                << ",0,0,0,0,0,0,independent_initialization_incomplete\n"; reports.flush(); continue;
        }
        for (Index id : candidates) local.images[id].registered = true;
        const auto result = recover_independent_submap(local, candidates, stable);
        reports << group_id << ',' << group.size() << ',' << draft.scene.registered_count() << ','
            << candidates.size() << ',' << result.shared_points << ',' << result.fit_inliers << ','
            << result.validation_points << ',' << result.validation_inliers << ',' << result.accepted << ','
            << result.reason << '\n'; reports.flush();
        if (!result.accepted) continue;
        std::set<Index> accepted(candidates.begin(), candidates.end());
        std::vector<Index> offsets(local.images.size());
        for (Index id = 0; id < local.images.size(); ++id) {
            auto& destination = scene.images[globals[id]].features.keypoints;
            offsets[id] = static_cast<Index>(destination.size());
            destination.insert(destination.end(), local.images[id].features.keypoints.begin(),
                               local.images[id].features.keypoints.end());
        }
        for (Index id : candidates) {
            scene.images[globals[id]].pose = local.images[id].pose;
            scene.images[globals[id]].registered = true;
            ++recovered;
        }
        for (const auto& source : local.tracks) {
            bool needed = false;
            for (std::size_t o = 0; o < std::min<std::size_t>(source.num_inliers, source.observations.size()); ++o)
                needed = needed || accepted.contains(source.observations[o].image_id);
            if (!needed || !source.is_triangulated()) continue;
            Track track; track.position = source.position;
            for (std::size_t o = 0; o < std::min<std::size_t>(source.num_inliers, source.observations.size()); ++o) {
                const auto& obs = source.observations[o];
                if (stable[obs.image_id] || accepted.contains(obs.image_id))
                    track.observations.push_back({globals[obs.image_id], offsets[obs.image_id]+obs.feature_id});
            }
            track.num_inliers = static_cast<std::uint8_t>(track.observations.size());
            if (track.num_inliers >= 2) scene.tracks.push_back(std::move(track));
        }
    }
    if (recovered) rebuild_track_index(scene);
    return recovered;
}
