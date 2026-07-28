#include "sfm/resection.hpp"

#include "core/logging.hpp"
#include "parallel/thread_pool.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aetherscan::sfm {
namespace {

struct PoseProposal {
    Index image_id{k_invalid};
    AbsolutePoseResult pose{};
    std::vector<ImageTrackRef> correspondences;
    unsigned num_points{0};
    double inlier_ratio{0.0};
    unsigned inlier_grid_cells{0};
    unsigned rotation_neighbors{0};
    double median_rotation_error_deg{0.0};
    unsigned translation_neighbors{0};
    double median_translation_error_deg{0.0};
};

struct PoseCandidate {
    Index image_id{k_invalid};
    std::vector<ImageTrackRef> correspondences;
};

using RegisteredFeatureTracks =
    std::vector<std::unordered_map<Index, Index>>;

RegisteredFeatureTracks build_registered_feature_tracks(
    const Scene& scene) {
    RegisteredFeatureTracks index(scene.images.size());
    for (Index track_id = 0; track_id < scene.tracks.size(); ++track_id) {
        const Track& track = scene.tracks[track_id];
        if (!track.is_triangulated()) continue;
        const std::size_t count = std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        for (std::size_t i = 0; i < count; ++i) {
            const Observation& observation = track.observations[i];
            if (observation.image_id >= scene.images.size() ||
                !scene.images[observation.image_id].registered)
                continue;
            index[observation.image_id].try_emplace(
                observation.feature_id, track_id);
        }
    }
    return index;
}

std::vector<ImageTrackRef> collect_pose_correspondences(
    const Scene& scene, const RegisteredFeatureTracks& registered_tracks,
    const Index image_id, const ResectionConfig& config) {
    std::unordered_map<std::uint64_t, unsigned> votes;
    if (image_id < scene.image_tracks.size()) {
        votes.reserve(scene.image_tracks[image_id].size() * 2);
        for (const ImageTrackRef& reference : scene.image_tracks[image_id]) {
            if (reference.track_id >= scene.tracks.size() ||
                !scene.tracks[reference.track_id].is_triangulated())
                continue;
            const std::uint64_t key =
                (static_cast<std::uint64_t>(reference.feature_id) << 32U) |
                reference.track_id;
            votes[key] += 1U << 20U;
        }
    }

    if (config.use_pair_match_correspondences) {
        for (const ImagePair& pair : scene.pairs) {
            if (!pair.active || pair.matches.empty()) continue;
            Index other_id = k_invalid;
            bool target_is_first = false;
            if (pair.id1 == image_id) {
                other_id = pair.id2;
                target_is_first = true;
            } else if (pair.id2 == image_id) {
                other_id = pair.id1;
            } else {
                continue;
            }
            if (other_id >= scene.images.size() ||
                !scene.images[other_id].registered ||
                other_id >= registered_tracks.size())
                continue;
            const auto& other_tracks = registered_tracks[other_id];
            for (const FeatureMatch& match : pair.matches) {
                const Index target_feature =
                    target_is_first ? match.query : match.train;
                const Index other_feature =
                    target_is_first ? match.train : match.query;
                const auto position = other_tracks.find(other_feature);
                if (position == other_tracks.end()) continue;
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(target_feature) << 32U) |
                    position->second;
                ++votes[key];
            }
        }
    }

    struct RankedReference {
        ImageTrackRef reference;
        unsigned votes{};
    };
    std::vector<RankedReference> ranked;
    ranked.reserve(votes.size());
    for (const auto& [key, count] : votes) {
        ranked.push_back({
            {static_cast<Index>(key & 0xffffffffULL),
             static_cast<Index>(key >> 32U)},
            count});
    }
    std::sort(
        ranked.begin(), ranked.end(),
        [](const RankedReference& left, const RankedReference& right) {
            if (left.votes != right.votes)
                return left.votes > right.votes;
            if (left.reference.feature_id != right.reference.feature_id)
                return left.reference.feature_id <
                       right.reference.feature_id;
            return left.reference.track_id < right.reference.track_id;
        });

    std::unordered_set<Index> used_features;
    std::unordered_set<Index> used_tracks;
    used_features.reserve(ranked.size());
    used_tracks.reserve(ranked.size());
    std::vector<ImageTrackRef> result;
    result.reserve(ranked.size());
    for (const RankedReference& entry : ranked) {
        if (!used_features.insert(entry.reference.feature_id).second ||
            !used_tracks.insert(entry.reference.track_id).second)
            continue;
        result.push_back(entry.reference);
    }
    return result;
}

double rotation_error_deg(const Mat3& first, const Mat3& second) {
    const Mat3 delta = first * second.transpose();
    const double cosine = std::clamp((delta.trace() - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(cosine) * 180.0 / 3.14159265358979323846;
}

std::pair<unsigned, double> pose_rotation_consistency(
    const Scene& scene, const Index image_id, const Pose3D& proposed,
    const float min_pair_weight) {
    std::vector<double> errors;
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || !pair.relative_pose.has_value() ||
            pair.composite_weight() < min_pair_weight) {
            continue;
        }
        Index other_id = k_invalid;
        Mat3 actual = Mat3::Identity();
        if (pair.id1 == image_id) {
            other_id = pair.id2;
            if (other_id >= scene.images.size() ||
                !scene.images[other_id].registered) {
                continue;
            }
            actual = scene.images[other_id].pose.R * proposed.R.transpose();
        } else if (pair.id2 == image_id) {
            other_id = pair.id1;
            if (other_id >= scene.images.size() ||
                !scene.images[other_id].registered) {
                continue;
            }
            actual = proposed.R * scene.images[other_id].pose.R.transpose();
        } else {
            continue;
        }
        errors.push_back(rotation_error_deg(actual, pair.relative_pose->R));
    }
    if (errors.empty()) return {0U, 0.0};
    const std::size_t middle = errors.size() / 2;
    std::nth_element(errors.begin(), errors.begin() + middle, errors.end());
    return {static_cast<unsigned>(errors.size()), errors[middle]};
}

double vector_error_deg(const Vec3& first, const Vec3& second) {
    const double denominator = first.norm() * second.norm();
    if (denominator <= 1e-12) return 180.0;
    const double cosine = std::clamp(first.dot(second) / denominator, -1.0, 1.0);
    return std::acos(cosine) * 180.0 / 3.14159265358979323846;
}

std::pair<unsigned, double> pose_translation_consistency(
    const Scene& scene, const Index image_id, const Pose3D& proposed,
    const float min_pair_weight) {
    std::vector<double> errors;
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || !pair.relative_pose.has_value() ||
            pair.composite_weight() < min_pair_weight) {
            continue;
        }
        Vec3 expected = Vec3::Zero();
        Vec3 actual = Vec3::Zero();
        if (pair.id1 == image_id) {
            if (pair.id2 >= scene.images.size() ||
                !scene.images[pair.id2].registered) {
                continue;
            }
            expected = proposed.R.transpose() * pair.relative_pose->C;
            actual = scene.images[pair.id2].pose.C - proposed.C;
        } else if (pair.id2 == image_id) {
            if (pair.id1 >= scene.images.size() ||
                !scene.images[pair.id1].registered) {
                continue;
            }
            const Pose3D& first = scene.images[pair.id1].pose;
            expected = first.R.transpose() * pair.relative_pose->C;
            actual = proposed.C - first.C;
        } else {
            continue;
        }
        errors.push_back(vector_error_deg(expected, actual));
    }
    if (errors.empty()) return {0U, 0.0};
    const std::size_t middle = errors.size() / 2;
    std::nth_element(errors.begin(), errors.begin() + middle, errors.end());
    return {static_cast<unsigned>(errors.size()), errors[middle]};
}

std::vector<PoseCandidate> select_next_images(
    const Scene& scene,
    const RegisteredFeatureTracks& registered_tracks,
    const std::unordered_map<Index, unsigned>& unregistered,
    const ResectionConfig& config) {
    std::vector<PoseCandidate> candidates;
    candidates.reserve(unregistered.size());
    for (const auto& [image_id, score] : unregistered) {
        (void)score;
        candidates.push_back({image_id, {}});
    }
    if (candidates.empty()) return {};

    const unsigned threads = parallel::resolve_thread_count(scene.thread_count);
    parallel::parallel_for(
        candidates.size(), threads, [&](const std::size_t index) {
            candidates[index].correspondences =
                collect_pose_correspondences(
                    scene, registered_tracks,
                    candidates[index].image_id, config);
        });

    std::vector<PoseCandidate> next;
    next.reserve(candidates.size());
    for (PoseCandidate& candidate : candidates)
        if (candidate.correspondences.size() >= config.min_correspondences)
            next.push_back(std::move(candidate));
    if (next.empty()) return next;

    std::sort(next.begin(), next.end(), [](const auto& a, const auto& b) {
        return a.correspondences.size() > b.correspondences.size();
    });
    const unsigned best =
        static_cast<unsigned>(next.front().correspondences.size());
    const unsigned threshold = static_cast<unsigned>(
        config.ratio_correspondences * static_cast<float>(best));
    next.erase(
        std::remove_if(
            next.begin(), next.end(),
            [&](const PoseCandidate& candidate) {
                return candidate.correspondences.size() <
                       std::max(threshold, config.min_correspondences);
            }),
        next.end());
    return next;
}

PoseProposal estimate_image_pose(
    const Scene& scene, const PoseCandidate& candidate,
    const ResectionConfig& config) {
    PoseProposal proposal;
    proposal.image_id = candidate.image_id;
    proposal.correspondences.reserve(candidate.correspondences.size());
    std::vector<Vec3> bearings;
    std::vector<Vec3> points;
    std::vector<Vec2> pixels;
    const Index image_id = candidate.image_id;
    if (image_id >= scene.images.size()) return proposal;
    const Image& image = scene.images[image_id];
    const PinholeCamera& camera = scene.camera_of(image);

    for (const ImageTrackRef& reference : candidate.correspondences) {
        if (reference.track_id >= scene.tracks.size()) continue;
        const Track& track = scene.tracks[reference.track_id];
        if (!track.is_triangulated()) continue;
        if (reference.feature_id >= image.features.keypoints.size()) continue;
        const auto& kp = image.features.keypoints[reference.feature_id];
        proposal.correspondences.push_back(reference);
        bearings.push_back(camera.unproject_normalized({kp.x, kp.y}));
        points.push_back(track.position);
        pixels.emplace_back(kp.x, kp.y);
    }

    proposal.num_points = static_cast<unsigned>(bearings.size());
    if (proposal.num_points < config.min_inliers) return proposal;

    AbsolutePoseOptions ransac = config.ransac;
    ransac.min_inliers = config.min_inliers;
    proposal.pose =
        estimate_absolute_pose(bearings, points, camera, ransac);
    if (!proposal.pose.success) return proposal;

    proposal.inlier_ratio = static_cast<double>(proposal.pose.num_inliers) /
                            static_cast<double>(proposal.num_points);
    const unsigned grid_size = std::max(1U, config.inlier_grid_size);
    std::vector<std::uint8_t> occupied(
        static_cast<std::size_t>(grid_size) * grid_size, 0);
    const std::size_t mask_count = std::min(
        pixels.size(), proposal.pose.inlier_mask.size());
    for (std::size_t i = 0; i < mask_count; ++i) {
        if (!proposal.pose.inlier_mask[i]) continue;
        const double normalized_x = std::clamp(
            pixels[i].x() / static_cast<double>(std::max(camera.width, 1U)),
            0.0, std::nextafter(1.0, 0.0));
        const double normalized_y = std::clamp(
            pixels[i].y() / static_cast<double>(std::max(camera.height, 1U)),
            0.0, std::nextafter(1.0, 0.0));
        const unsigned x = std::min(
            grid_size - 1,
            static_cast<unsigned>(normalized_x * grid_size));
        const unsigned y = std::min(
            grid_size - 1,
            static_cast<unsigned>(normalized_y * grid_size));
        occupied[static_cast<std::size_t>(y) * grid_size + x] = 1;
    }
    proposal.inlier_grid_cells = static_cast<unsigned>(
        std::count(occupied.begin(), occupied.end(), std::uint8_t{1}));
    std::tie(
        proposal.rotation_neighbors,
        proposal.median_rotation_error_deg) =
        pose_rotation_consistency(
            scene, image_id, proposal.pose.pose,
            config.min_consistency_pair_weight);
    std::tie(
        proposal.translation_neighbors,
        proposal.median_translation_error_deg) =
        pose_translation_consistency(
            scene, image_id, proposal.pose.pose,
            config.min_consistency_pair_weight);

    const bool ratio_ok =
        config.min_inlier_ratio <= 0.F ||
        proposal.inlier_ratio >= config.min_inlier_ratio;
    const bool bypass_coverage =
        config.coverage_bypass_inlier_ratio > 0.F &&
        proposal.pose.num_inliers >=
            config.coverage_bypass_min_inliers &&
        proposal.inlier_ratio >=
            config.coverage_bypass_inlier_ratio;
    const bool coverage_ok =
        bypass_coverage ||
        config.min_inlier_grid_cells == 0 ||
        proposal.inlier_grid_cells >= config.min_inlier_grid_cells;
    const bool bypass_consistency =
        config.consistency_bypass_inlier_ratio > 0.F &&
        proposal.inlier_ratio >= config.consistency_bypass_inlier_ratio;
    const bool rotation_ok =
        bypass_consistency ||
        config.max_median_rotation_error_deg <= 0.F ||
        proposal.rotation_neighbors < config.min_rotation_consistency_neighbors ||
        proposal.median_rotation_error_deg <=
            config.max_median_rotation_error_deg;
    const bool translation_ok =
        bypass_consistency ||
        config.max_median_translation_error_deg <= 0.F ||
        proposal.translation_neighbors <
            config.min_translation_consistency_neighbors ||
        proposal.median_translation_error_deg <=
            config.max_median_translation_error_deg;
    if (!ratio_ok || !coverage_ok || !rotation_ok || !translation_ok)
        proposal.pose.success = false;
    return proposal;
}

void attach_inlier_observations(
    Scene& scene, const PoseProposal& proposal) {
    const std::size_t count = std::min(
        proposal.correspondences.size(),
        proposal.pose.inlier_mask.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (!proposal.pose.inlier_mask[i]) continue;
        const ImageTrackRef& reference = proposal.correspondences[i];
        if (reference.track_id >= scene.tracks.size()) continue;
        Track& track = scene.tracks[reference.track_id];
        const auto existing = std::find_if(
            track.observations.begin(), track.observations.end(),
            [&](const Observation& observation) {
                return observation.image_id == proposal.image_id;
            });
        const bool observation_existed =
            existing != track.observations.end();
        const std::size_t inlier_count = std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        if (observation_existed) {
            const std::size_t existing_index = static_cast<std::size_t>(
                std::distance(track.observations.begin(), existing));
            if (existing_index < inlier_count) continue;
            const Observation promoted = *existing;
            track.observations.erase(existing);
            track.observations.insert(
                track.observations.begin() +
                    static_cast<std::ptrdiff_t>(inlier_count),
                promoted);
        } else {
            track.observations.insert(
                track.observations.begin() +
                    static_cast<std::ptrdiff_t>(inlier_count),
                {proposal.image_id, reference.feature_id});
        }
        track.num_inliers = static_cast<std::uint8_t>(
            std::min<std::size_t>(inlier_count + 1U, 255U));
        if (!observation_existed &&
            proposal.image_id < scene.image_tracks.size())
            scene.image_tracks[proposal.image_id].push_back(reference);
    }
}

std::vector<Index> build_local_window(
    const Scene& scene,
    const std::vector<Index>& image_ids,
    const ResectionConfig& config) {
    std::unordered_set<Index> target(image_ids.begin(), image_ids.end());
    std::unordered_set<Index> candidate_tracks;
    for (Index image_id : image_ids) {
        if (image_id >= scene.image_tracks.size()) continue;
        for (const ImageTrackRef& reference : scene.image_tracks[image_id])
            candidate_tracks.insert(reference.track_id);
    }
    std::unordered_map<Index, unsigned> counts;
    for (Index track_id : candidate_tracks) {
        if (track_id >= scene.tracks.size()) continue;
        const Track& track = scene.tracks[track_id];
        if (!track.is_triangulated()) continue;
        for (unsigned o = 0; o < track.num_inliers; ++o) {
            const Index id = track.observations[o].image_id;
            if (!target.count(id) && scene.images[id].registered) ++counts[id];
        }
    }
    std::vector<Index> neighbors;
    neighbors.reserve(counts.size());
    for (const auto& [id, score] : counts) {
        (void)score;
        neighbors.push_back(id);
    }
    std::sort(neighbors.begin(), neighbors.end(), [&](Index a, Index b) {
        return counts[a] > counts[b];
    });
    const std::size_t max_neighbors =
        config.max_local_window == 0
            ? neighbors.size()
            : std::min(
                  neighbors.size(),
                  static_cast<std::size_t>(
                      std::max(0, static_cast<int>(config.max_local_window) -
                                      static_cast<int>(image_ids.size()))));
    if (neighbors.size() > max_neighbors) neighbors.resize(max_neighbors);
    return neighbors;
}

class RunningAverage {
public:
    RunningAverage(
        const std::size_t window, std::vector<double>& values)
        : window_(window), values_(values) {}
    void add(const double value) {
        values_.push_back(value);
        if (values_.size() > window_) values_.erase(values_.begin());
    }
    void clear() { values_.clear(); }
    [[nodiscard]] double average() const {
        if (values_.empty()) return 1.0;
        double sum = 0.0;
        for (double v : values_) sum += v;
        return sum / static_cast<double>(values_.size());
    }
    [[nodiscard]] std::size_t size() const { return values_.size(); }

private:
    std::size_t window_;
    std::vector<double>& values_;
};

BundleSummary run_required_bundle_adjustment(
    Scene& scene, const BundleOptions& options, const char* stage) {
    const BundleSummary summary = run_bundle_adjustment(scene, options);
    if (!summary.success)
        throw std::runtime_error(
            std::string("Incremental ") + stage +
            " bundle adjustment failed; reconstruction stopped before filtering/checkpointing");
    return summary;
}

bool bundle_tail_is_still_improving(
    const BundleSummary& summary, const unsigned tail_window,
    const double threshold, const char* label) {
    if (threshold <= 0.0 ||
        summary.optimizer.termination !=
            ba::TerminationReason::maximum_iterations ||
        summary.optimizer.iterations.empty()) {
        return false;
    }
    const std::size_t window = std::min<std::size_t>(
        std::max(1U, tail_window),
        summary.optimizer.iterations.size());
    const double earlier = summary.optimizer.iterations[
        summary.optimizer.iterations.size() - window].cost;
    const double final = summary.optimizer.final_cost;
    if (!std::isfinite(earlier) || !std::isfinite(final) || earlier <= 0.0)
        return false;
    const double relative_improvement =
        std::max(0.0, earlier - final) / earlier;
    core::Logger::instance().debug(
        label, " BA tail: window=", window,
        " relative_improvement=", relative_improvement,
        " continuation_threshold=", threshold);
    return relative_improvement >= threshold;
}

void run_periodic_full_bundle_adjustment(
    Scene& scene, const ResectionConfig& config, const bool force_full) {
    BundleOptions ba;
    ba.optimizer = config.full_ba;
    const unsigned full_iterations = static_cast<unsigned>(
        ba.optimizer.maximum_iterations);
    const unsigned probe_iterations = std::min(
        config.periodic_full_ba_probe_iterations, full_iterations);
    if (force_full || probe_iterations == 0 ||
        probe_iterations >= full_iterations) {
        run_required_bundle_adjustment(scene, ba, "full");
        return;
    }

    ba.optimizer.maximum_iterations = probe_iterations;
    const BundleSummary probe =
        run_required_bundle_adjustment(scene, ba, "full probe");
    if (!bundle_tail_is_still_improving(
            probe, config.periodic_full_ba_tail_window,
            config.periodic_full_ba_tail_relative_improvement,
            "periodic full")) {
        core::Logger::instance().info(
            "periodic full BA stopped after probe iterations=",
            probe_iterations, " cameras=", probe.num_cameras,
            " points=", probe.num_points);
        return;
    }

    ba.optimizer = config.full_ba;
    ba.optimizer.maximum_iterations = full_iterations - probe_iterations;
    run_required_bundle_adjustment(scene, ba, "full continuation");
}

std::vector<Index> collect_tracks_for_images(
    const Scene& scene, const std::vector<Index>& image_ids) {
    std::vector<Index> tracks;
    std::size_t reference_count = 0;
    for (const Index image_id : image_ids)
        if (image_id < scene.image_tracks.size())
            reference_count += scene.image_tracks[image_id].size();
    tracks.reserve(reference_count);
    for (const Index image_id : image_ids) {
        if (image_id >= scene.image_tracks.size()) continue;
        const auto& references = scene.image_tracks[image_id];
        for (const ImageTrackRef& reference : references)
            if (reference.track_id < scene.tracks.size())
                tracks.push_back(reference.track_id);
    }
    std::sort(tracks.begin(), tracks.end());
    tracks.erase(std::unique(tracks.begin(), tracks.end()), tracks.end());
    return tracks;
}

std::vector<Index> triangulate_dirty_tracks(
    Scene& scene, const std::vector<Index>& dirty_images,
    const ResectionConfig& config) {
    std::vector<Index> dirty_tracks =
        collect_tracks_for_images(scene, dirty_images);
    if (dirty_tracks.empty()) return dirty_tracks;
    const std::size_t tracks_before = scene.tracks.size();
    triangulate_tracks(
        scene, dirty_tracks, true, config.max_reproj_error,
        config.min_angle_deg);
    // Splitting rebuilds image_tracks and may add children observed by a dirty
    // image. Recollect so local BA/filtering sees the updated topology.
    if (scene.tracks.size() != tracks_before)
        dirty_tracks = collect_tracks_for_images(scene, dirty_images);
    core::Logger::instance().debug(
        "dirty tracks: images=", dirty_images.size(),
        " tracks=", dirty_tracks.size(), '/', scene.tracks.size());
    return dirty_tracks;
}

}  // namespace

unsigned register_images(Scene& scene, const ResectionConfig& config) {
    core::StageScope stage("sfm.resection");
    rebuild_track_index(scene);
    scene.registration_generation = std::max<std::uint32_t>(
        scene.registration_generation, scene.registered_count());
    std::unordered_map<Index, unsigned> unregistered;
    for (Index i = 0; i < scene.images.size(); ++i) {
        if (!scene.images[i].registered) unregistered[i] = 0;
    }
    core::ProgressReporter progress("register images", unregistered.size());

    unsigned registered_count = 0;
    unsigned checkpointed_count = 0;
    unsigned& since_full_ba = scene.resection_progress.since_full_ba;
    unsigned& n_ba =
        scene.resection_progress.bundle_adjustment_stage;
    std::vector<Index>& last_registered =
        scene.resection_progress.last_registered;
    RunningAverage avg_inliers(
        10, scene.resection_progress.recent_inlier_ratios);
    const unsigned threads = parallel::resolve_thread_count(scene.thread_count);
    const unsigned wave_limit = std::max(1U, config.max_pose_wave);
    std::vector<Index> dirty_images;

    while (!unregistered.empty()) {
        const RegisteredFeatureTracks registered_tracks =
            build_registered_feature_tracks(scene);
        std::vector<PoseCandidate> next_candidates =
            select_next_images(
                scene, registered_tracks, unregistered, config);
        if (next_candidates.empty()) break;

        const unsigned start_count = registered_count;
        bool stop_candidate_band = false;
        for (std::size_t wave_begin = 0;
             wave_begin < next_candidates.size() && !stop_candidate_band;
             wave_begin += wave_limit) {
            const std::size_t wave_end = std::min(
                wave_begin + static_cast<std::size_t>(wave_limit),
                next_candidates.size());
            const std::size_t wave_count = wave_end - wave_begin;
            std::vector<PoseProposal> proposals(wave_count);
            parallel::parallel_for(
                wave_count, threads, [&](const std::size_t index) {
                    proposals[index] = estimate_image_pose(
                        scene, next_candidates[wave_begin + index],
                        config);
                });

            for (std::size_t n = 0; n < proposals.size();) {
                PoseProposal& proposal = proposals[n];
                const unsigned num_inliers =
                    proposal.pose.success ? proposal.pose.num_inliers : 0U;
                const unsigned num_points = proposal.num_points;

                if (num_inliers == 0) {
                    core::Logger::instance().debug(
                        "resection rejected image=", proposal.image_id,
                        " correspondences=", num_points,
                        " pnp_inliers=", proposal.pose.num_inliers,
                        " ratio=", proposal.inlier_ratio,
                        " grid_cells=", proposal.inlier_grid_cells,
                        " rotation_neighbors=", proposal.rotation_neighbors,
                        " rotation_median_deg=",
                            proposal.median_rotation_error_deg,
                        " translation_neighbors=",
                            proposal.translation_neighbors,
                        " translation_median_deg=",
                            proposal.median_translation_error_deg);
                    ++n;
                    continue;
                }

                avg_inliers.add(proposal.inlier_ratio);

                scene.images[proposal.image_id].pose = proposal.pose.pose;
                scene.images[proposal.image_id].registered = true;
                attach_inlier_observations(scene, proposal);
                ++scene.registration_generation;
                if (scene.registration_generation == 0) {
                    // Generation zero means "unchecked". A wrap is extremely
                    // unlikely, but resetting preserves correctness.
                    scene.registration_generation = 1;
                    for (Track& track : scene.tracks)
                        track.split_generation = 0;
                }
                last_registered.push_back(proposal.image_id);
                dirty_images.push_back(proposal.image_id);
                unregistered.erase(proposal.image_id);
                ++registered_count;
                progress.advance();
                core::Logger::instance().debug(
                    "resection registered image=", proposal.image_id,
                    " inliers=", num_inliers, '/', num_points,
                    " ratio=", num_points == 0 ? 0.0
                        : static_cast<double>(num_inliers) / num_points,
                    " grid_cells=", proposal.inlier_grid_cells,
                    " rotation_median_deg=",
                        proposal.median_rotation_error_deg,
                    " translation_median_deg=",
                        proposal.median_translation_error_deg);
                ++since_full_ba;
                ++n;

                const bool force_full =
                    config.avg_inliers_ratio_force_ba > 0 &&
                    avg_inliers.size() >= config.min_force_full_ba_samples &&
                    since_full_ba >= config.min_force_full_ba_interval &&
                    avg_inliers.average() < config.avg_inliers_ratio_force_ba;
                const unsigned full_every =
                    n_ba < config.full_ba_every.size()
                        ? config.full_ba_every[n_ba]
                        : 0;
                const bool do_full =
                    (full_every > 0 && since_full_ba >= full_every) ||
                    force_full;

                if (do_full) {
                    triangulate_tracks(
                        scene, false, config.max_reproj_error,
                        config.min_angle_deg);
                    run_periodic_full_bundle_adjustment(
                        scene, config, force_full);
                    filter_tracks(
                        scene, config.max_reproj_error, config.min_angle_deg,
                        config.mult_depth_near, config.mult_depth_far);
                    last_registered.clear();
                    dirty_images.clear();
                    avg_inliers.clear();
                    since_full_ba = 0;
                    if (n_ba + 1 < config.full_ba_every.size()) ++n_ba;
                    stop_candidate_band = true;
                    break;
                }

                if (config.local_ba_every > 0 &&
                    last_registered.size() >= config.local_ba_every) {
                    triangulate_dirty_tracks(scene, dirty_images, config);
                    BundleOptions ba;
                    ba.optimizer = config.local_ba;
                    ba.optimize_all_registered = false;
                    ba.free_image_ids = last_registered;
                    ba.fixed_image_ids =
                        build_local_window(scene, last_registered, config);
                    run_required_bundle_adjustment(scene, ba, "local");
                    const std::vector<Index> affected_tracks =
                        collect_tracks_for_images(scene, last_registered);
                    filter_tracks(
                        scene, affected_tracks, config.max_reproj_error,
                        config.min_angle_deg);
                    last_registered.clear();
                    dirty_images.clear();
                    stop_candidate_band = true;
                    break;
                }
            }
        }
        if (!stop_candidate_band && registered_count > start_count) {
            const std::vector<Index> dirty_tracks =
                triangulate_dirty_tracks(scene, dirty_images, config);
            filter_tracks(
                scene, dirty_tracks, config.max_reproj_error,
                config.min_angle_deg);
            dirty_images.clear();
        }

        if (registered_count == start_count) break;
        if (config.checkpoint_callback &&
            registered_count - checkpointed_count >=
                std::max(1U, config.checkpoint_interval)) {
            config.checkpoint_callback(scene);
            checkpointed_count = registered_count;
        }
    }
    progress.finish();

    if (registered_count > 0) {
        triangulate_tracks(scene, false, config.max_reproj_error, config.min_angle_deg);
        BundleOptions ba;
        ba.optimizer = config.full_ba;
        const BundleSummary final_summary =
            run_required_bundle_adjustment(scene, ba, "final");
        if (config.final_ba_additional_iterations > 0 &&
            bundle_tail_is_still_improving(
                final_summary, config.final_ba_tail_window,
                config.final_ba_tail_relative_improvement, "final")) {
            ba.optimizer.maximum_iterations =
                config.final_ba_additional_iterations;
            run_required_bundle_adjustment(scene, ba, "final polish");
        }
        filter_tracks(
            scene, config.max_reproj_error, config.min_angle_deg, config.mult_depth_near,
            config.mult_depth_far);
        since_full_ba = 0;
        n_ba = 0;
        last_registered.clear();
        dirty_images.clear();
        avg_inliers.clear();
    }
    if (config.checkpoint_callback &&
        registered_count != checkpointed_count)
        config.checkpoint_callback(scene);

    core::Logger::instance().info(
        "resection: newly_registered=", registered_count,
        " total=", scene.registered_count());
    return registered_count;
}

}  // namespace aetherscan::sfm
