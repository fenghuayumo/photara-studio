#pragma once

#include "sfm/frontend.hpp"
#include "sfm/star_init.hpp"
#include "sfm/resection.hpp"
#include "sfm/submap_recovery.hpp"
#include "sfm/tracks.hpp"
#include "sfm_pair_group_init.hpp"
#include "sfm_local_proposals.hpp"
#include "sfm_ray_group_alignment.hpp"
#include <cstdlib>
#include <unordered_set>

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
    reports << "group,missing,local_registered,candidates,shared,fit,heldout,heldout_inliers,accepted,reason,boundary_matches,candidate_tracks,stable_multiview_tracks,stable_triangulated_tracks,ray_attempted,ray_tracks,ray_fit,ray_holdout,ray_anchors,ray_fit_inliers,ray_holdout_inliers,ray_mean,ray_after_mean,ray_condition,ray_fit_observations,ray_holdout_observations\n";
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
        if (std::getenv("PHOTARA_GROUP_COVISIBLE_ANCHORS")) {
            const std::set<Index> targets(group.begin(),group.end());
            const auto feature_key=[](Index image,Index feature) { return (std::uint64_t{image}<<32)|feature; };
            std::unordered_set<std::uint64_t> boundary_features;
            for (const auto& pair:scene.pairs) {
                if (!pair.relative_pose || pair.zero_baseline) continue;
                const bool forward=targets.contains(pair.id1) && scene.images[pair.id2].registered;
                const bool reverse=targets.contains(pair.id2) && scene.images[pair.id1].registered;
                if (!forward && !reverse) continue;
                for (const auto& match:pair.matches)
                    boundary_features.insert(feature_key(forward?pair.id2:pair.id1,forward?match.train:match.query));
            }
            std::map<Index,unsigned> votes;
            for (const auto& track:scene.tracks) {
                if (!track.is_triangulated()) continue;
                std::set<Index> observers; bool touches=false;
                for (std::size_t o=0;o<std::min<std::size_t>(track.num_inliers,track.observations.size());++o) {
                    const auto& obs=track.observations[o];
                    if (!scene.images[obs.image_id].registered) continue;
                    observers.insert(obs.image_id);
                    touches=touches || boundary_features.contains(feature_key(obs.image_id,obs.feature_id));
                }
                if (touches && observers.size()>=3) for (Index id:observers) ++votes[id];
            }
            std::vector<Index> extra;
            for (const auto& [id,vote]:votes) extra.push_back(id);
            std::stable_sort(extra.begin(),extra.end(),[&](Index a,Index b) {
                return votes[a]!=votes[b]?votes[a]>votes[b]:a<b;
            });
            const auto add=[&](Index id) {
                if (anchors.size()<32 && scene.images[id].registered && std::find(anchors.begin(),anchors.end(),id)==anchors.end())
                    anchors.push_back(id);
            };
            for (Index id:extra) add(id);
            // If preserved boundary features cannot nominate enough observers,
            // retain temporal coverage at both segment ends as fresh proposals.
            for (unsigned distance=1;distance<=24 && anchors.size()<32;++distance) {
                if (group.front()>=distance) add(group.front()-distance);
                if (group.back()+distance<scene.images.size()) add(group.back()+distance);
            }
            photara::core::Logger::instance().info("group depth anchors: group=",group_id," voted=",votes.size()," chosen=",anchors.size());
        }
        for (Index id:anchors) photara::core::Logger::instance().info("group anchor: group=",group_id," image=",id," name=",scene.images[id].path.filename().string());

        if (anchors.size() < 4) continue;
        auto globals = anchors;
        globals.insert(globals.end(), group.begin(), group.end());
        if (std::any_of(globals.begin(), globals.end(), [&](Index id) {
            return !std::filesystem::is_regular_file(scene.images[id].path); })) continue;
        ++attempted;
        for (Index id:group) photara::core::Logger::instance().info("group member: group=",group_id," image=",id," name=",scene.images[id].path.filename().string());
        std::vector<std::filesystem::path> paths;
        for (Index id : globals) paths.push_back(scene.images[id].path);
        FrontEndOptions options;
        options.thread_count = scene.thread_count;
        if (const auto* cache=std::getenv("PHOTARA_GROUP_FRONTEND_CACHE"))
            options.checkpoint.directory=std::filesystem::path(cache)/std::to_string(group_id);
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
        std::vector<RawPairMatches> replay_raw;
        const auto* replay=std::getenv("PHOTARA_GROUP_REPLAY_ROOT");
        auto local = replay ? load_local_proposal_snapshot(std::filesystem::path(replay)/std::to_string(group_id),replay_raw)
                            : run_frontend(paths, options).scene;
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
        if (std::getenv("PHOTARA_GROUP_NATIVE_RAW")) {
            if (!replay) throw std::invalid_argument("Native raw verification requires a fixed replay snapshot");
            const auto summary=verify_local_raw_proposals(local,stable,replay_raw,
                std::getenv("PHOTARA_GROUP_NO_WEAK_CYCLES")==nullptr);
            photara::core::Logger::instance().info("group native raw proposals: group=",group_id,
                " raw=",summary.raw_pairs," accepted=",summary.accepted_pairs," boundary=",summary.boundary_pairs,
                " weak_boundary=",summary.weak_boundary_pairs," cycle_matches=",summary.cycle_supported_matches,
                " rejected_weak=",summary.rejected_weak_pairs);
        } else {
        if (std::getenv("PHOTARA_GROUP_FROZEN_ANCHOR_GEOMETRY")) {
            if (!options.checkpoint.enabled())
                throw std::runtime_error("Frozen-anchor experiment requires its own frontend cache");
            std::vector<std::filesystem::path> matches;
            for (const auto& entry : std::filesystem::directory_iterator(options.checkpoint.directory))
                if (entry.path().extension() == ".bin" && entry.path().stem().string().starts_with("matches-"))
                    matches.push_back(entry.path());
            if (matches.size() != 1)
                throw std::runtime_error("Frozen-anchor experiment requires one unambiguous raw-match snapshot");
            const auto match_key = std::stoull(matches.front().stem().string().substr(8), nullptr, 16);
            std::vector<RawPairMatches> raw;
            if (!CheckpointStore(options.checkpoint).load_matches(match_key, raw))
                throw std::runtime_error("Cannot load raw anchor correspondences");
            std::erase_if(local.pairs, [&](const auto& pair) { return stable[pair.id1] && stable[pair.id2]; });
            unsigned added = 0;
            for (const auto& proposal : raw) {
                if (proposal.id1 >= stable.size() || proposal.id2 >= stable.size())
                    throw std::runtime_error("Raw anchor snapshot image index is invalid");
                if (!stable[proposal.id1] || !stable[proposal.id2]) continue;
                ImagePair pair(proposal.id1, proposal.id2);
                for (const auto& match : proposal.matches) {
                    if (match.query >= local.images[pair.id1].features.keypoints.size() ||
                        match.train >= local.images[pair.id2].features.keypoints.size())
                        throw std::runtime_error("Raw anchor snapshot feature index is invalid");
                    pair.matches.push_back({match.query,match.train});
                }
                local.pairs.push_back(std::move(pair)); ++added;
            }
            photara::core::Logger::instance().info("group raw anchor proposals: group=",group_id," pairs=",added);
        }
        // Frontend focal initialization does not carry native aspect/distortion.
        // Reverify its proposed correspondences with the exact frozen native
        // calibration before using any relative pose or rebuilding tracks.
        for (auto& pair : local.pairs) {
            // Frozen anchor geometry can verify anchor-only correspondences
            // without estimating a fresh essential matrix on a weak baseline.
            // This does not supply any missing-view pose or depth.
            if (std::getenv("PHOTARA_GROUP_FROZEN_ANCHOR_GEOMETRY") && stable[pair.id1] && stable[pair.id2]) {
                const auto& first = local.images[pair.id1];
                const auto& second = local.images[pair.id2];
                const Vec3 baseline = second.pose.C - first.pose.C;
                const double angular = std::max(local.camera_of(first).pixel_error_to_angular(4),
                                                local.camera_of(second).pixel_error_to_angular(4));
                std::vector<FeatureMatch> inliers;
                if (baseline.norm() > 1e-10) for (const auto& match : pair.matches) {
                    const auto& a = first.features.keypoints[match.query];
                    const auto& b = second.features.keypoints[match.train];
                    const Vec3 u = first.pose.R.transpose() * local.camera_of(first).unproject_normalized(Vec2(a.x,a.y));
                    const Vec3 v = second.pose.R.transpose() * local.camera_of(second).unproject_normalized(Vec2(b.x,b.y));
                    const Vec3 n1 = baseline.cross(u), n2 = baseline.cross(v);
                    if (n1.norm() > 1e-10 && n2.norm() > 1e-10 &&
                        std::abs(n1.normalized().dot(v)) <= std::sin(angular) &&
                        std::abs(n2.normalized().dot(u)) <= std::sin(angular)) inliers.push_back(match);
                }
                pair.matches = std::move(inliers);
                pair.active = pair.matches.size() >= 8;
                pair.relative_pose = second.pose / first.pose;
                continue;
            }
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
        }
        build_tracks(local);
        auto draft = make_independent_submap(local, targets);
        build_tracks(draft.scene);
        StarInitConfig star;
        const bool pair_init = std::getenv("PHOTARA_GROUP_PAIR_INITIALIZER") != nullptr;
        const bool initialized = pair_init ? initialize_missing_group_pairs(draft.scene) : star_initialize(draft.scene, star);
        if (initialized) {
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
                << ",0,0,0,0,0,0,independent_initialization_incomplete,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"; reports.flush(); continue;
        }
        for (Index id : candidates) local.images[id].registered = true;
        SubmapRecoveryOptions recovery;
        if (pair_init) recovery.independent_initializer = initialize_missing_group_pairs;
        const auto result = recover_independent_submap(local, candidates, stable, recovery);
        bool accepted=result.accepted;
        std::string reason=result.reason;
        experiment::RayGroupReport ray;
        const bool ray_attempted=!result.accepted && std::getenv("PHOTARA_GROUP_RAY_ALIGNMENT");
        if (ray_attempted) {
            ray=experiment::recover_ray_group(local,candidates,stable,
                std::getenv("PHOTARA_GROUP_NO_SEGMENT_BA")==nullptr);
            photara::core::Logger::instance().info("group ray alignment: group=",group_id,
                " tracks=",ray.tracks," fit_tracks=",ray.fit_tracks," heldout_tracks=",ray.heldout_tracks,
                " fit_observations=",ray.fit_observations," heldout_observations=",ray.heldout_observations,
                " anchors=",ray.anchors," fit_inliers=",ray.fit_inliers," heldout_inliers=",ray.heldout_inliers,
                " mean=",ray.heldout_mean," after_mean=",ray.heldout_after_mean," condition=",ray.condition_ratio,
                " accepted=",ray.accepted," reason=",ray.reason);
            accepted=ray.accepted;reason="generalized_"+ray.reason;
        }
        reports << group_id << ',' << group.size() << ',' << draft.scene.registered_count() << ','
            << candidates.size() << ',' << result.shared_points << ',' << result.fit_inliers << ','
            << result.validation_points << ',' << result.validation_inliers << ',' << accepted << ','
            << reason << ',' << result.boundary_matches << ',' << result.candidate_tracks << ','
            << result.stable_multiview_tracks << ',' << result.stable_triangulated_tracks << ','
            << ray_attempted << ',' << ray.tracks << ',' << ray.fit_tracks << ',' << ray.heldout_tracks << ','
            << ray.anchors << ',' << ray.fit_inliers << ',' << ray.heldout_inliers << ','
            << ray.heldout_mean << ',' << ray.heldout_after_mean << ',' << ray.condition_ratio << ','
            << ray.fit_observations << ',' << ray.heldout_observations << '\n'; reports.flush();
        if (!accepted) continue;
        std::set<Index> accepted_ids(candidates.begin(), candidates.end());
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
                needed = needed || accepted_ids.contains(source.observations[o].image_id);
            if (!needed || !source.is_triangulated()) continue;
            Track track; track.position = source.position;
            for (std::size_t o = 0; o < std::min<std::size_t>(source.num_inliers, source.observations.size()); ++o) {
                const auto& obs = source.observations[o];
                if (stable[obs.image_id] || accepted_ids.contains(obs.image_id))
                    track.observations.push_back({globals[obs.image_id], offsets[obs.image_id]+obs.feature_id});
            }
            track.num_inliers = static_cast<std::uint8_t>(track.observations.size());
            if (track.num_inliers >= 2) scene.tracks.push_back(std::move(track));
        }
    }
    if (recovered) rebuild_track_index(scene);
    return recovered;
}
