#include "sfm/submap_recovery.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"
#include "core/logging.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace aetherscan::sfm {
namespace {
std::uint64_t key(Index image, Index feature) {
    return (std::uint64_t{image} << 32U) | feature;
}

bool noncollinear(const std::vector<Vec3>& points) {
    if (points.size() < 3) return false;
    Vec3 mean = Vec3::Zero();
    for (const auto& p : points) mean += p;
    mean /= static_cast<double>(points.size());
    Mat3 scatter = Mat3::Zero();
    for (const auto& p : points) scatter += (p - mean) * (p - mean).transpose();
    Eigen::SelfAdjointEigenSolver<Mat3> eig(scatter);
    // Reject practically ill-conditioned sets, not just exactly singular ones.
    return eig.info() == Eigen::Success && eig.eigenvalues().allFinite() &&
        eig.eigenvalues()[2] > 0 &&
        eig.eigenvalues()[1] > 1e-4 * eig.eigenvalues()[2];
}

bool supports(const Scene& scene, const Observation& obs, const Vec3& point,
              const Pose3D& pose, double threshold) {
    if (obs.image_id >= scene.images.size()) return false;
    const Image& image = scene.images[obs.image_id];
    if (image.camera_id >= scene.cameras.size() ||
        obs.feature_id >= image.features.keypoints.size()) return false;
    Vec2 pixel;
    if (!scene.camera_of(image).project_checked(pose.transform_world_to_camera(point), pixel))
        return false;
    const auto& measured = image.features.keypoints[obs.feature_id];
    return (pixel - Vec2(measured.x, measured.y)).norm() <= threshold;
}
}  // namespace

std::vector<std::uint8_t> find_structural_pair_risks(const Scene& scene) {
    Scene graph;
    graph.images.resize(scene.images.size());
    for (Index id = 0; id < graph.images.size(); ++id) {
        graph.images[id].id = id;
        graph.images[id].registered = true;
    }
    graph.pairs.reserve(scene.pairs.size());
    for (const auto& pair : scene.pairs) {
        if (!pair.active || !pair.relative_pose || pair.zero_baseline) continue;
        ImagePair edge(pair.id1, pair.id2);
        edge.relative_pose = pair.relative_pose;
        graph.pairs.push_back(std::move(edge));
    }
    auto audit = analyze_alignment_observability(graph);
    for (auto& reliable : audit.reliable) reliable = reliable ? 0 : 1;
    return audit.reliable;
}

HierarchicalSubscene make_independent_submap(
    const Scene& parent, const std::vector<Index>& images) {
    HierarchicalSubscene sub;
    sub.local_to_global = images;
    sub.scene.thread_count = parent.thread_count;
    std::vector<Index> local(parent.images.size(), k_invalid);
    std::map<Index, Index> cameras;
    for (Index id : images) {
        if (id >= parent.images.size() || local[id] != k_invalid)
            throw std::invalid_argument("invalid or duplicate submap image");
        const Image& source = parent.images[id];
        if (source.camera_id >= parent.cameras.size())
            throw std::invalid_argument("invalid submap camera");
        Image image;
        image.id = static_cast<Index>(sub.scene.images.size());
        local[id] = image.id;
        auto [it, added] = cameras.emplace(source.camera_id,
            static_cast<Index>(sub.scene.cameras.size()));
        if (added) {
            auto camera = parent.cameras[source.camera_id];
            camera.id = it->second;
            camera.trust_intrinsics = true;
            sub.scene.cameras.push_back(camera);
        }
        image.camera_id = it->second;
        image.path = source.path;
        image.features.keypoints = source.features.keypoints;
        image.features.image_width = source.features.image_width;
        image.features.image_height = source.features.image_height;
        sub.scene.images.push_back(std::move(image));
    }
    for (const auto& pair : parent.pairs) {
        if (!pair.active || !pair.relative_pose || pair.zero_baseline ||
            pair.id1 >= local.size() || pair.id2 >= local.size() ||
            local[pair.id1] == k_invalid || local[pair.id2] == k_invalid) continue;
        auto copy = pair;
        copy.id1 = local[pair.id1];
        copy.id2 = local[pair.id2];
        sub.scene.pairs.push_back(std::move(copy));
    }
    return sub;
}

SubmapRecoveryReport recover_independent_submap(
    Scene& parent, const std::vector<Index>& images,
    const std::vector<std::uint8_t>& stable, const SubmapRecoveryOptions& options) {
    if (stable.size() != parent.images.size() || options.minimum_shared_points < 10 ||
        options.minimum_camera_observations < 2 ||
        !std::isfinite(options.maximum_reprojection_error) || options.maximum_reprojection_error <= 0 ||
        !std::isfinite(options.relative_alignment_error) || options.relative_alignment_error <= 0 ||
        !std::isfinite(options.minimum_inlier_ratio) || options.minimum_inlier_ratio <= 0 ||
        options.minimum_inlier_ratio > 1 || !std::isfinite(options.minimum_angle_degrees) ||
        options.minimum_angle_degrees <= 0)
        throw std::invalid_argument("invalid submap recovery options/mask");
    SubmapRecoveryReport report;
    report.images = images;
    const auto reject = [&](const char* reason) {
        report.reason = reason;
        return report;
    };
    auto sub = make_independent_submap(parent, images);
    std::vector<Index> local(parent.images.size(), k_invalid);
    for (Index i = 0; i < images.size(); ++i) {
        if (stable[images[i]] || !parent.images[images[i]].registered)
            throw std::invalid_argument("submap must contain registered non-stable cameras");
        local[images[i]] = i;
    }
    if (images.size() < 4) return reject("too_few_local_views");
    build_tracks(sub.scene);
    StarInitConfig star;
    if (!star_initialize(sub.scene, star)) {
        report.reconstructed_views = sub.scene.registered_count();
        return reject("independent_initialization_failed");
    }
    ResectionConfig resection;
    resection.full_ba.optimize_focal = false;
    resection.full_ba.optimize_aspect_ratio = false;
    resection.full_ba.optimize_distortion = false;
    register_images(sub.scene, resection);
    report.reconstructed_views = sub.scene.registered_count();
    if (report.reconstructed_views != images.size()) return reject("incomplete_local_reconstruction");

    TriangulationOptions tri;
    tri.reproj_threshold_px = static_cast<float>(options.maximum_reprojection_error);
    tri.min_angle_deg = options.minimum_angle_degrees;
    tri.split_tracks = false;
    triangulate_tracks(sub.scene, false, tri);
    if (options.independent_model_callback) options.independent_model_callback(sub);
    std::unordered_map<std::uint64_t, Index> local_points;
    for (Index t = 0; t < sub.scene.tracks.size(); ++t) {
        const auto& track = sub.scene.tracks[t];
        if (track.is_triangulated()) ++report.local_landmarks;
        for (std::size_t o = 0; track.is_triangulated() && o < track.num_inliers; ++o) {
            const auto& obs = track.observations[o];
            const auto k = key(images[obs.image_id], obs.feature_id);
            auto [it, inserted] = local_points.emplace(k, t);
            if (!inserted && it->second != t) it->second = k_invalid;
        }
    }

    // Index only observations that can connect to this submap. Depth is solved
    // lazily and exclusively from stable cameras, never from the parent point.
    std::unordered_map<std::uint64_t, std::vector<Index>> links;
    for (const auto& pair : parent.pairs) {
        if (!pair.active || !pair.relative_pose || pair.zero_baseline ||
            pair.id1 >= local.size() || pair.id2 >= local.size()) continue;
        bool forward = local[pair.id1] != k_invalid && stable[pair.id2];
        bool reverse = local[pair.id2] != k_invalid && stable[pair.id1];
        if (!forward && !reverse) continue;
        report.boundary_matches += static_cast<unsigned>(pair.matches.size());
        for (const auto& match : pair.matches) {
            const auto it = local_points.find(key(forward ? pair.id1 : pair.id2,
                                                 forward ? match.query : match.train));
            if (it != local_points.end() && it->second != k_invalid)
                links[key(forward ? pair.id2 : pair.id1,
                          forward ? match.train : match.query)].push_back(it->second);
        }
    }
    struct Correspondence { Index local_track; Vec3 point; };
    std::vector<Correspondence> candidates;
    std::map<Index, unsigned> multiplicity;
    for (const auto& source : parent.tracks) {
        Track independent;
        std::set<Index> targets;
        std::set<Index> observed_images;
        for (const auto& obs : source.observations) {
            if (obs.image_id >= stable.size()) continue;
            if (stable[obs.image_id] && parent.images[obs.image_id].registered &&
                observed_images.insert(obs.image_id).second)
                independent.observations.push_back(obs);
            const auto k = key(obs.image_id, obs.feature_id);
            if (const auto it = local_points.find(k); it != local_points.end() && it->second != k_invalid)
                targets.insert(it->second);
            if (const auto it = links.find(k); it != links.end())
                targets.insert(it->second.begin(), it->second.end());
        }
        if (targets.empty()) continue;
        ++report.candidate_tracks;
        if (independent.observations.size() < 2) continue;
        ++report.stable_multiview_tracks;
        if (targets.size() != 1) continue;
        if (triangulate_track(independent, parent, tri) < 2) continue;
        ++report.stable_triangulated_tracks;
        candidates.push_back({*targets.begin(), independent.position});
        ++multiplicity[*targets.begin()];
    }
    std::vector<Correspondence> shared;
    for (const auto& c : candidates)
        if (multiplicity[c.local_track] == 1) shared.push_back(c);
    std::sort(shared.begin(), shared.end(), [](const auto& a, const auto& b) {
        return a.local_track < b.local_track;
    });
    report.shared_points = static_cast<unsigned>(shared.size());
    if (shared.size() < options.minimum_shared_points) return reject("insufficient_independent_shared_points");

    std::vector<Vec3> src, dst;
    std::vector<Correspondence> heldout;
    for (std::size_t i = 0; i < shared.size(); ++i) {
        if (i % 5 == 0) heldout.push_back(shared[i]);
        else {
            src.push_back(sub.scene.tracks[shared[i].local_track].position);
            dst.push_back(shared[i].point);
        }
    }
    if (!noncollinear(src) || !noncollinear(dst)) return reject("ill_conditioned_shared_geometry");
    Vec3 mean = Vec3::Zero();
    for (const auto& p : dst) mean += p;
    mean /= static_cast<double>(dst.size());
    std::vector<double> radii;
    for (const auto& p : dst) radii.push_back((p - mean).norm());
    std::sort(radii.begin(), radii.end());
    const double threshold = options.relative_alignment_error * radii[radii.size() / 2];
    if (!(threshold > 0)) return reject("zero_alignment_extent");
    std::vector<std::size_t> inliers;
    report.fit_inliers = estimate_similarity_transform(src, dst, report.transform,
        threshold, 2048, 0xA37E5CA1u, &inliers);
    if (report.fit_inliers < 3 || report.fit_inliers < options.minimum_inlier_ratio * src.size())
        return reject("similarity_consensus_failed");
    std::vector<Vec3> inlier_src, inlier_dst;
    for (auto i : inliers) { inlier_src.push_back(src[i]); inlier_dst.push_back(dst[i]); }
    if (!noncollinear(inlier_src) || !noncollinear(inlier_dst))
        return reject("ill_conditioned_consensus");
    std::vector<Pose3D> poses;
    for (const auto& image : sub.scene.images)
        poses.push_back({image.pose.R * report.transform.R.transpose(),
                         report.transform.apply(image.pose.C)});
    std::set<Index> validation_cameras;
    std::vector<Vec3> validated_points;
    report.validation_points = static_cast<unsigned>(heldout.size());
    for (const auto& c : heldout) {
        const auto& track = sub.scene.tracks[c.local_track];
        if ((report.transform.apply(track.position) - c.point).norm() > threshold) continue;
        bool good = true;
        for (std::size_t o = 0; o < track.num_inliers; ++o) {
            const auto& obs = track.observations[o];
            if (!supports(parent, {images[obs.image_id], obs.feature_id}, c.point,
                          poses[obs.image_id], options.maximum_reprojection_error)) good = false;
        }
        if (!good) continue;
        ++report.validation_inliers;
        validated_points.push_back(c.point);
        for (std::size_t o = 0; o < track.num_inliers; ++o)
            validation_cameras.insert(track.observations[o].image_id);
    }
    if (report.validation_inliers < 3 ||
        report.validation_inliers < options.minimum_inlier_ratio * heldout.size() ||
        validation_cameras.size() < 2 || !noncollinear(validated_points))
        return reject("heldout_geometry_failed");

    // Stage all affected tracks against candidate poses. Roll back poses before
    // returning on rejection; stable poses/intrinsics are never optimized.
    std::vector<Pose3D> previous;
    for (Index i = 0; i < images.size(); ++i) {
        previous.push_back(parent.images[images[i]].pose);
        parent.images[images[i]].pose = poses[i];
    }
    std::vector<std::pair<Index, Track>> updated;
    std::vector<std::set<Index>> support(images.size());
    bool stable_support_preserved = true;
    try {
        for (Index t = 0; t < parent.tracks.size(); ++t) {
            const auto& track = parent.tracks[t];
            bool affected = false;
            for (const auto& obs : track.observations)
                if (obs.image_id < local.size() && local[obs.image_id] != k_invalid) affected = true;
            if (!affected) continue;
            Track next = track;
            next.position.setZero();
            next.num_inliers = 0;
            next.split_generation = 0;
            Track geometry;
            for (const auto& obs : track.observations)
                if (obs.image_id < local.size() &&
                    (stable[obs.image_id] || local[obs.image_id] != k_invalid))
                    geometry.observations.push_back(obs);
            if (triangulate_track(geometry, parent, tri) >= 2) {
                next.position = geometry.position;
                std::stable_partition(next.observations.begin(), next.observations.end(),
                    [&](const Observation& obs) {
                        return obs.image_id < parent.images.size() && parent.images[obs.image_id].registered &&
                            supports(parent, obs, next.position, parent.images[obs.image_id].pose,
                                     options.maximum_reprojection_error);
                    });
                for (const auto& obs : next.observations) {
                    if (next.num_inliers == 255 || obs.image_id >= parent.images.size() ||
                        !parent.images[obs.image_id].registered ||
                        !supports(parent, obs, next.position, parent.images[obs.image_id].pose,
                                  options.maximum_reprojection_error)) break;
                    ++next.num_inliers;
                }
            }
            for (std::size_t o = 0; o < std::min<std::size_t>(track.num_inliers, track.observations.size()); ++o) {
                const auto& obs = track.observations[o];
                if (obs.image_id >= stable.size() || !stable[obs.image_id]) continue;
                const Pose3D& pose = parent.images[obs.image_id].pose;
                if (supports(parent, obs, track.position, pose, options.maximum_reprojection_error) &&
                    (!next.is_triangulated() || !supports(parent, obs, next.position, pose,
                                                         options.maximum_reprojection_error)))
                    stable_support_preserved = false;
            }
            for (std::size_t o = 0; next.is_triangulated() && o < next.num_inliers; ++o) {
                const auto& obs = next.observations[o];
                if (local[obs.image_id] != k_invalid)
                    support[local[obs.image_id]].insert(obs.feature_id);
            }
            updated.emplace_back(t, std::move(next));
        }
    } catch (...) {
        for (Index i = 0; i < images.size(); ++i) parent.images[images[i]].pose = previous[i];
        throw;
    }
    if (!stable_support_preserved) {
        for (Index i = 0; i < images.size(); ++i) parent.images[images[i]].pose = previous[i];
        return reject("stable_reprojection_support_lost");
    }
    for (const auto& s : support) {
        if (s.size() >= options.minimum_camera_observations) continue;
        for (Index i = 0; i < images.size(); ++i) parent.images[images[i]].pose = previous[i];
        return reject("insufficient_final_camera_support");
    }
    for (auto& [id, track] : updated) parent.tracks[id] = std::move(track);
    rebuild_track_index(parent);
    report.accepted = true;
    report.reason = "accepted";
    return report;
}
std::vector<Index> recover_stable_resections(Scene& scene) {
    const auto audit = analyze_alignment_observability(scene);
    if (audit.reliable_views == scene.images.size() || audit.reliable_views < 2) return {};
    struct Match { Index feature; Index track; Vec3 point; };
    std::vector<std::vector<Match>> matches(scene.images.size());
    std::unordered_map<std::uint64_t, std::vector<Observation>> boundary;
    for (const auto& pair : scene.pairs) {
        if (!pair.active || !pair.relative_pose || pair.zero_baseline ||
            pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size()) continue;
        const bool forward = audit.reliable[pair.id1] && !audit.reliable[pair.id2];
        const bool reverse = audit.reliable[pair.id2] && !audit.reliable[pair.id1];
        if (!forward && !reverse) continue;
        for (const auto& m : pair.matches)
            boundary[key(forward ? pair.id1 : pair.id2, forward ? m.query : m.train)]
                .push_back({forward ? pair.id2 : pair.id1, forward ? m.train : m.query});
    }
    TriangulationOptions tri;
    tri.reproj_threshold_px = 2;
    tri.min_angle_deg = 1;
    tri.split_tracks = false;
    for (Index t = 0; t < scene.tracks.size(); ++t) {
        const auto& source = scene.tracks[t];
        Track stable;
        std::map<Index, Index> stable_observations;
        std::set<std::pair<Index, Index>> targets;
        for (const auto& obs : source.observations) {
            if (obs.image_id >= scene.images.size()) continue;
            if (audit.reliable[obs.image_id]) {
                stable_observations.emplace(obs.image_id, obs.feature_id);
                if (const auto it = boundary.find(key(obs.image_id, obs.feature_id)); it != boundary.end())
                    for (const auto& target : it->second) targets.emplace(target.image_id, target.feature_id);
            } else targets.emplace(obs.image_id, obs.feature_id);
        }
        if (targets.empty() || stable_observations.size() < 2) continue;
        for (const auto& [image, feature] : stable_observations) stable.observations.push_back({image, feature});
        if (triangulate_track(stable, scene, tri) < 2) continue;
        // Replacing a point must not discard valid observations in the frozen
        // map. A locally successful PnP is insufficient if its anchors damage
        // the geometry that made the main map reliable.
        bool preserves_stable_support = true;
        for (std::size_t o = 0; o < std::min<std::size_t>(source.num_inliers, source.observations.size()); ++o) {
            const auto& obs = source.observations[o];
            if (obs.image_id >= scene.images.size() || !audit.reliable[obs.image_id]) continue;
            const auto& pose = scene.images[obs.image_id].pose;
            if (supports(scene, obs, source.position, pose, 2) &&
                !supports(scene, obs, stable.position, pose, 2)) {
                preserves_stable_support = false;
                break;
            }
        }
        if (!preserves_stable_support) continue;
        for (const auto& [image, feature] : targets)
            if (feature < scene.images[image].features.keypoints.size())
                matches[image].push_back({feature, t, stable.position});
    }
    std::vector<Index> recovered;
    std::vector<std::vector<Match>> accepted(scene.images.size());
    for (Index id = 0; id < scene.images.size(); ++id) {
        if (audit.reliable[id]) continue;
        const auto& image = scene.images[id];
        if (image.camera_id >= scene.cameras.size()) continue;
        const auto& camera = scene.camera_of(image);
        std::map<Index, std::set<Index>> by_feature, by_track;
        for (const auto& m : matches[id]) { by_feature[m.feature].insert(m.track); by_track[m.track].insert(m.feature); }
        std::map<Index, Match> unique;
        for (const auto& m : matches[id])
            if (by_feature[m.feature].size() == 1 && by_track[m.track].size() == 1) unique.emplace(m.feature, m);
        core::Logger::instance().info("stable resection: image=", image.path.filename(), " correspondences=", unique.size());
        const auto reject = [&](const char* reason) {
            core::Logger::instance().info("stable resection rejected: image=", image.path.filename(), " reason=", reason);
        };
        if (unique.size() < 40) { reject("insufficient_independent_depths"); continue; }
        std::vector<Vec3> bearings, points;
        std::vector<Match> validation;
        std::size_t n = 0;
        for (const auto& [feature, m] : unique) {
            if (n++ % 5 == 0) { validation.push_back(m); continue; }
            const auto& p = image.features.keypoints[feature];
            bearings.push_back(camera.unproject(Vec2(p.x, p.y)));
            points.push_back(m.point);
        }
        if (!noncollinear(points)) { reject("ill_conditioned_fit_points"); continue; }
        AbsolutePoseOptions options;
        options.min_inliers = 30;
        options.max_reproj_error_px = 2;
        options.max_iterations = 10000;
        const auto pose = estimate_absolute_pose(bearings, points, camera, options);
        if (!pose.success || pose.num_inliers < 0.8 * points.size()) { reject("pose_consensus_failed"); continue; }
        unsigned heldout = 0;
        std::set<std::pair<int, int>> cells;
        std::vector<Vec3> heldout_points;
        for (const auto& m : validation) {
            if (!supports(scene, {id, m.feature}, m.point, pose.pose, 2)) continue;
            ++heldout;
            heldout_points.push_back(m.point);
            const auto& p = image.features.keypoints[m.feature];
            cells.emplace(static_cast<int>(std::clamp(4*p.x/std::max(1u,camera.width), 0.F, 3.F)),
                          static_cast<int>(std::clamp(4*p.y/std::max(1u,camera.height), 0.F, 3.F)));
        }
        if (heldout < 8 || heldout < 0.8 * validation.size()) { reject("heldout_consensus_failed"); continue; }
        if (cells.size() < 3 || !noncollinear(heldout_points)) { reject("ill_conditioned_validation"); continue; }
        for (const auto& [feature, m] : unique)
            if (supports(scene, {id, feature}, m.point, pose.pose, 2)) accepted[id].push_back(m);
        scene.images[id].pose = pose.pose;
        scene.images[id].registered = true;
        recovered.push_back(id);
        core::Logger::instance().info("stable resection accepted: image=", image.path.filename(),
            " fit=", pose.num_inliers, " heldout=", heldout, '/', validation.size());
    }
    if (recovered.empty()) return recovered;
    std::vector<bool> moved(scene.images.size(), false);
    for (Index id : recovered) moved[id] = true;
    // Reattach only verified 2D-3D observations. Unsupported old target tracks
    // must not keep stale geometry after replacing a camera pose.
    for (auto& track : scene.tracks) {
        std::vector<Observation> observations;
        std::uint8_t count = 0;
        for (std::size_t o = 0; o < track.observations.size(); ++o) {
            const auto& obs = track.observations[o];
            if (obs.image_id < moved.size() && moved[obs.image_id]) continue;
            observations.push_back(obs);
            if (o < track.num_inliers) ++count;
        }
        track.observations = std::move(observations);
        track.num_inliers = count >= 2 ? count : 0;
    }
    std::set<Index> changed;
    for (Index id : recovered) for (const auto& m : accepted[id]) {
        auto& track = scene.tracks[m.track];
        track.position = m.point;
        track.observations.push_back({id, m.feature});
        changed.insert(m.track);
    }
    for (Index t : changed) {
        auto& track = scene.tracks[t];
        const auto end = std::stable_partition(track.observations.begin(), track.observations.end(), [&](const Observation& obs) {
            return obs.image_id < scene.images.size() && scene.images[obs.image_id].registered &&
                supports(scene, obs, track.position, scene.images[obs.image_id].pose, 2);
        });
        track.num_inliers = static_cast<std::uint8_t>(std::min<std::size_t>(255, end - track.observations.begin()));
    }
    rebuild_track_index(scene);
    return recovered;
}
}  // namespace aetherscan::sfm
