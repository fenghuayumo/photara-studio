#include "sfm/resection.hpp"

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

std::vector<Index> select_next_images(
    const Scene& scene,
    const std::unordered_map<Index, unsigned>& unregistered,
    const ResectionConfig& config) {
    std::unordered_map<Index, unsigned> scores = unregistered;
    for (auto& [id, score] : scores) score = 0;

    // Score only the tracks referenced by each candidate image.
    for (auto& [image_id, score] : scores) {
        if (image_id >= scene.image_tracks.size()) continue;
        for (const ImageTrackRef& reference : scene.image_tracks[image_id]) {
            if (reference.track_id < scene.tracks.size() &&
                scene.tracks[reference.track_id].is_triangulated())
                ++score;
        }
    }

    std::vector<Index> next;
    for (const auto& [id, score] : scores) {
        if (score >= config.min_correspondences) next.push_back(id);
    }
    if (next.empty()) return next;

    std::sort(next.begin(), next.end(), [&](Index a, Index b) {
        return scores[a] > scores[b];
    });
    const unsigned best = scores[next.front()];
    const unsigned threshold = static_cast<unsigned>(
        config.ratio_correspondences * static_cast<float>(best));
    next.erase(
        std::remove_if(
            next.begin(), next.end(),
            [&](Index id) {
                return scores[id] < std::max(threshold, config.min_correspondences);
            }),
        next.end());
    return next;
}

std::pair<unsigned, unsigned> register_image(
    Scene& scene, const Index image_id, const ResectionConfig& config) {
    std::vector<Vec3> bearings;
    std::vector<Vec3> points;
    const Image& image = scene.images[image_id];
    const PinholeCamera& camera = scene.camera_of(image);

    if (image_id >= scene.image_tracks.size()) return {0, 0};
    for (const ImageTrackRef& reference : scene.image_tracks[image_id]) {
        if (reference.track_id >= scene.tracks.size()) continue;
        const Track& track = scene.tracks[reference.track_id];
        if (!track.is_triangulated()) continue;
        if (reference.feature_id >= image.features.keypoints.size()) continue;
        const auto& kp = image.features.keypoints[reference.feature_id];
        bearings.push_back(camera.unproject_normalized({kp.x, kp.y}));
        points.push_back(track.position);
    }

    const unsigned n = static_cast<unsigned>(bearings.size());
    if (n < config.min_inliers) return {0, n};

    AbsolutePoseOptions ransac = config.ransac;
    ransac.min_inliers = config.min_inliers;
    const AbsolutePoseResult pose =
        estimate_absolute_pose(bearings, points, camera, ransac);
    if (!pose.success) return {0, n};

    scene.images[image_id].pose = pose.pose;
    scene.images[image_id].registered = true;
    return {pose.num_inliers, n};
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
    rebuild_track_index(scene);
    std::unordered_map<Index, unsigned> unregistered;
    for (Index i = 0; i < scene.images.size(); ++i) {
        if (!scene.images[i].registered) unregistered[i] = 0;
    }

    unsigned registered_count = 0;
    unsigned checkpointed_count = 0;
    unsigned& since_full_ba = scene.resection_progress.since_full_ba;
    unsigned& n_ba =
        scene.resection_progress.bundle_adjustment_stage;
    std::vector<Index>& last_registered =
        scene.resection_progress.last_registered;
    RunningAverage avg_inliers(
        10, scene.resection_progress.recent_inlier_ratios);

    while (!unregistered.empty()) {
        std::vector<Index> next_ids = select_next_images(scene, unregistered, config);
        if (next_ids.empty()) break;

        const unsigned start_count = registered_count;
        for (std::size_t n = 0; n < next_ids.size();) {
            const Index next_id = next_ids[n];
            const auto [num_inliers, num_points] = register_image(scene, next_id, config);
            if (num_points > 0)
                avg_inliers.add(static_cast<double>(num_inliers) / static_cast<double>(num_points));

            if (num_inliers == 0) {
                next_ids.erase(next_ids.begin() + static_cast<std::ptrdiff_t>(n));
                continue;
            }

            last_registered.push_back(next_id);
            unregistered.erase(next_id);
            ++registered_count;
            ++since_full_ba;
            ++n;

            const bool force_full =
                config.avg_inliers_ratio_force_ba > 0 &&
                avg_inliers.average() < config.avg_inliers_ratio_force_ba;
            const unsigned full_every =
                n_ba < config.full_ba_every.size() ? config.full_ba_every[n_ba] : 0;
            const bool do_full =
                (full_every > 0 && since_full_ba >= full_every) || force_full;

            if (do_full) {
                triangulate_tracks(
                    scene, false, config.max_reproj_error, config.min_angle_deg);
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
                break;
            }

            if (config.local_ba_every > 0 &&
                last_registered.size() >= config.local_ba_every) {
                triangulate_tracks(
                    scene, true, config.max_reproj_error, config.min_angle_deg);
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
                break;
            }

            if (n == next_ids.size()) {
                triangulate_tracks(
                    scene, true, config.max_reproj_error, config.min_angle_deg);
                filter_tracks(
                    scene, config.max_reproj_error, config.min_angle_deg,
                    config.mult_depth_near, config.mult_depth_far);
                break;
            }
        }

        if (registered_count == start_count) break;
        if (config.checkpoint_callback &&
            registered_count - checkpointed_count >=
                std::max(1U, config.checkpoint_interval)) {
            config.checkpoint_callback(scene);
            checkpointed_count = registered_count;
        }
    }

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

    std::cout << "resection: newly_registered=" << registered_count
              << " total=" << scene.registered_count() << '\n';
    return registered_count;
}

}  // namespace aetherscan::sfm
