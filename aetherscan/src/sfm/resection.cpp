#include "sfm/resection.hpp"

#include "core/logging.hpp"
#include "parallel/thread_pool.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aetherscan::sfm {
namespace {

struct PoseProposal {
    Index image_id{k_invalid};
    AbsolutePoseResult pose{};
    unsigned num_points{0};
};

std::vector<Index> select_next_images(
    const Scene& scene,
    const std::unordered_map<Index, unsigned>& unregistered,
    const ResectionConfig& config) {
    std::vector<Index> candidates;
    candidates.reserve(unregistered.size());
    for (const auto& [image_id, score] : unregistered) {
        (void)score;
        candidates.push_back(image_id);
    }
    if (candidates.empty()) return {};

    std::vector<unsigned> scores(candidates.size(), 0);
    const unsigned threads = parallel::resolve_thread_count(scene.thread_count);
    parallel::parallel_for(
        candidates.size(), threads, [&](const std::size_t index) {
            const Index image_id = candidates[index];
            if (image_id >= scene.image_tracks.size()) return;
            unsigned score = 0;
            for (const ImageTrackRef& reference : scene.image_tracks[image_id]) {
                if (reference.track_id < scene.tracks.size() &&
                    scene.tracks[reference.track_id].is_triangulated())
                    ++score;
            }
            scores[index] = score;
        });

    std::vector<Index> next;
    next.reserve(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (scores[index] >= config.min_correspondences)
            next.push_back(candidates[index]);
    }
    if (next.empty()) return next;

    std::unordered_map<Index, unsigned> score_by_id;
    score_by_id.reserve(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index)
        score_by_id.emplace(candidates[index], scores[index]);

    std::sort(next.begin(), next.end(), [&](Index a, Index b) {
        return score_by_id[a] > score_by_id[b];
    });
    const unsigned best = score_by_id[next.front()];
    const unsigned threshold = static_cast<unsigned>(
        config.ratio_correspondences * static_cast<float>(best));
    next.erase(
        std::remove_if(
            next.begin(), next.end(),
            [&](Index id) {
                return score_by_id[id] <
                       std::max(threshold, config.min_correspondences);
            }),
        next.end());
    return next;
}

PoseProposal estimate_image_pose(
    const Scene& scene, const Index image_id, const ResectionConfig& config) {
    PoseProposal proposal;
    proposal.image_id = image_id;
    std::vector<Vec3> bearings;
    std::vector<Vec3> points;
    if (image_id >= scene.images.size()) return proposal;
    const Image& image = scene.images[image_id];
    const PinholeCamera& camera = scene.camera_of(image);

    if (image_id >= scene.image_tracks.size()) return proposal;
    for (const ImageTrackRef& reference : scene.image_tracks[image_id]) {
        if (reference.track_id >= scene.tracks.size()) continue;
        const Track& track = scene.tracks[reference.track_id];
        if (!track.is_triangulated()) continue;
        if (reference.feature_id >= image.features.keypoints.size()) continue;
        const auto& kp = image.features.keypoints[reference.feature_id];
        bearings.push_back(camera.unproject_normalized({kp.x, kp.y}));
        points.push_back(track.position);
    }

    proposal.num_points = static_cast<unsigned>(bearings.size());
    if (proposal.num_points < config.min_inliers) return proposal;

    AbsolutePoseOptions ransac = config.ransac;
    ransac.min_inliers = config.min_inliers;
    proposal.pose =
        estimate_absolute_pose(bearings, points, camera, ransac);
    return proposal;
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

private:
    std::size_t window_;
    std::vector<double>& values_;
};

void run_required_bundle_adjustment(
    Scene& scene, const BundleOptions& options, const char* stage) {
    const BundleSummary summary = run_bundle_adjustment(scene, options);
    if (!summary.success)
        throw std::runtime_error(
            std::string("Incremental ") + stage +
            " bundle adjustment failed; reconstruction stopped before filtering/checkpointing");
}

}  // namespace

unsigned register_images(Scene& scene, const ResectionConfig& config) {
    core::StageScope stage("sfm.resection");
    rebuild_track_index(scene);
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

    while (!unregistered.empty()) {
        std::vector<Index> next_ids = select_next_images(scene, unregistered, config);
        if (next_ids.empty()) break;

        const unsigned start_count = registered_count;
        bool stop_candidate_band = false;
        for (std::size_t wave_begin = 0;
             wave_begin < next_ids.size() && !stop_candidate_band;
             wave_begin += wave_limit) {
            const std::size_t wave_end = std::min(
                wave_begin + static_cast<std::size_t>(wave_limit),
                next_ids.size());
            const std::size_t wave_count = wave_end - wave_begin;
            std::vector<PoseProposal> proposals(wave_count);
            parallel::parallel_for(
                wave_count, threads, [&](const std::size_t index) {
                    proposals[index] = estimate_image_pose(
                        scene, next_ids[wave_begin + index], config);
                });

            for (std::size_t n = 0; n < proposals.size();) {
                PoseProposal& proposal = proposals[n];
                const unsigned num_inliers =
                    proposal.pose.success ? proposal.pose.num_inliers : 0U;
                const unsigned num_points = proposal.num_points;
                if (num_points > 0)
                    avg_inliers.add(
                        static_cast<double>(num_inliers) /
                        static_cast<double>(num_points));

                if (num_inliers == 0) {
                    core::Logger::instance().debug(
                        "resection rejected image=", proposal.image_id,
                        " correspondences=", num_points);
                    ++n;
                    continue;
                }

                scene.images[proposal.image_id].pose = proposal.pose.pose;
                scene.images[proposal.image_id].registered = true;
                last_registered.push_back(proposal.image_id);
                unregistered.erase(proposal.image_id);
                ++registered_count;
                progress.advance();
                core::Logger::instance().debug(
                    "resection registered image=", proposal.image_id,
                    " inliers=", num_inliers, '/', num_points,
                    " ratio=", num_points == 0 ? 0.0
                        : static_cast<double>(num_inliers) / num_points);
                ++since_full_ba;
                ++n;

                const bool force_full =
                    config.avg_inliers_ratio_force_ba > 0 &&
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
                    BundleOptions ba;
                    ba.optimizer = config.full_ba;
                    run_required_bundle_adjustment(scene, ba, "full");
                    filter_tracks(
                        scene, config.max_reproj_error, config.min_angle_deg,
                        config.mult_depth_near, config.mult_depth_far);
                    last_registered.clear();
                    avg_inliers.clear();
                    since_full_ba = 0;
                    if (n_ba + 1 < config.full_ba_every.size()) ++n_ba;
                    stop_candidate_band = true;
                    break;
                }

                if (config.local_ba_every > 0 &&
                    last_registered.size() >= config.local_ba_every) {
                    triangulate_tracks(
                        scene, true, config.max_reproj_error,
                        config.min_angle_deg);
                    BundleOptions ba;
                    ba.optimizer = config.local_ba;
                    ba.optimize_all_registered = false;
                    ba.free_image_ids = last_registered;
                    ba.fixed_image_ids =
                        build_local_window(scene, last_registered, config);
                    run_required_bundle_adjustment(scene, ba, "local");
                    filter_tracks(
                        scene, config.max_reproj_error, config.min_angle_deg,
                        config.mult_depth_near, config.mult_depth_far);
                    last_registered.clear();
                    stop_candidate_band = true;
                    break;
                }
            }
        }
        if (!stop_candidate_band && registered_count > start_count) {
            triangulate_tracks(
                scene, true, config.max_reproj_error, config.min_angle_deg);
            filter_tracks(
                scene, config.max_reproj_error, config.min_angle_deg,
                config.mult_depth_near, config.mult_depth_far);
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
        ba.optimizer.maximum_iterations = 100;
        run_required_bundle_adjustment(scene, ba, "final");
        filter_tracks(
            scene, config.max_reproj_error, config.min_angle_deg, config.mult_depth_near,
            config.mult_depth_far);
        since_full_ba = 0;
        n_ba = 0;
        last_registered.clear();
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
