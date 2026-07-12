#include "sfm/tracks.hpp"

#include "core/logging.hpp"
#include "sfm/triangulation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aetherscan::sfm {
namespace {

struct DisjointSet {
    std::vector<Index> parent;
    std::vector<Index> rank;

    explicit DisjointSet(const std::size_t n) : parent(n), rank(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    Index find(Index x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    bool unite(Index a, Index b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) ++rank[a];
        return true;
    }
};

}  // namespace

void rebuild_track_index(Scene& scene) {
    scene.image_tracks.assign(scene.images.size(), {});
    std::vector<std::size_t> counts(scene.images.size(), 0);
    for (const Track& track : scene.tracks)
        for (const Observation& observation : track.observations)
            if (observation.image_id < counts.size())
                ++counts[observation.image_id];
    for (std::size_t image_id = 0; image_id < counts.size(); ++image_id)
        scene.image_tracks[image_id].reserve(counts[image_id]);
    for (Index track_id = 0; track_id < scene.tracks.size(); ++track_id) {
        for (const Observation& observation :
             scene.tracks[track_id].observations) {
            if (observation.image_id >= scene.image_tracks.size()) continue;
            scene.image_tracks[observation.image_id].push_back(
                {track_id, observation.feature_id});
        }
    }
}

void build_tracks(Scene& scene, const float min_pair_weight) {
    core::StageScope stage("sfm.build_tracks");
    scene.tracks.clear();
    scene.image_tracks.clear();
    std::size_t total_features = 0;
    for (const auto& image : scene.images)
        total_features += image.features.keypoints.size();
    if (total_features == 0) return;

    std::vector<Index> feature_offsets(scene.images.size() + 1, 0);
    for (std::size_t image_id = 0; image_id < scene.images.size(); ++image_id) {
        feature_offsets[image_id + 1] =
            feature_offsets[image_id] +
            static_cast<Index>(
                scene.images[image_id].features.keypoints.size());
    }

    DisjointSet ds(total_features);
    // A valid track contains at most one feature from each image. Store image
    // membership, not the number of pair edges incident on a feature: the same
    // observation commonly appears in several verified image pairs.
    std::unordered_map<Index, std::unordered_set<Index>> component_images;

    const auto accumulate = [&](const Index global_id, const Index image_id) {
        const Index root = ds.find(global_id);
        auto& images = component_images[root];
        images.insert(image_id);
    };

    core::ProgressReporter progress("build tracks", scene.pairs.size());
    for (const ImagePair& pair : scene.pairs) {
        progress.advance();
        if (!pair.active || pair.matches.empty()) continue;
        if (pair.composite_weight() <= min_pair_weight) continue;
        const Index off1 = feature_offsets[pair.id1];
        const Index off2 = feature_offsets[pair.id2];
        for (const FeatureMatch& match : pair.matches) {
            const Index g1 = off1 + match.query;
            const Index g2 = off2 + match.train;
            accumulate(g1, pair.id1);
            accumulate(g2, pair.id2);

            const Index r1 = ds.find(g1);
            const Index r2 = ds.find(g2);
            if (r1 == r2) continue;

            auto& map1 = component_images[r1];
            auto& map2 = component_images[r2];
            bool conflict = false;
            const auto& smaller = map1.size() <= map2.size() ? map1 : map2;
            const auto& larger = map1.size() <= map2.size() ? map2 : map1;
            for (const Index image_id : smaller) {
                if (larger.contains(image_id)) {
                    conflict = true;
                    break;
                }
            }
            if (conflict) continue;

            ds.unite(r1, r2);
            const Index root = ds.find(r1);
            auto& merged = component_images[root];
            if (root != r1) {
                merged.insert(map1.begin(), map1.end());
            }
            if (root != r2) {
                merged.insert(map2.begin(), map2.end());
            }
            if (root != r1) component_images.erase(r1);
            if (root != r2) component_images.erase(r2);
        }
    }

    std::unordered_map<Index, Track> tracks_by_root;
    Index running = 0;
    for (Index image_id = 0; image_id < scene.images.size(); ++image_id) {
        const auto& image = scene.images[image_id];
        for (Index feat = 0; feat < image.features.keypoints.size(); ++feat, ++running) {
            const Index root = ds.find(running);
            tracks_by_root[root].observations.push_back({image_id, feat});
        }
    }

    scene.tracks.reserve(tracks_by_root.size());
    for (auto& [root, track] : tracks_by_root) {
        (void)root;
        if (track.observations.size() < 2) continue;
        std::sort(
            track.observations.begin(), track.observations.end(),
            [](const Observation& a, const Observation& b) {
                return a.image_id < b.image_id ||
                       (a.image_id == b.image_id && a.feature_id < b.feature_id);
            });
        scene.tracks.push_back(std::move(track));
    }
    rebuild_track_index(scene);
}

std::pair<float, float> filter_tracks(
    Scene& scene,
    const float max_reproj_error_px,
    const float min_angle_deg,
    const float mult_depth_near,
    const float mult_depth_far) {
    core::StageScope stage("sfm.filter_tracks", core::LogLevel::debug);
    std::vector<double> average_distances;
    average_distances.reserve(scene.tracks.size());
    double sum_px = 0.0;
    double sum_cos_angle = 0.0;
    unsigned counted = 0;
    for (Track& track : scene.tracks) {
        track.num_inliers = 0;
        if (!track.is_valid()) continue;
        double track_px = 0.0;
        double track_cos_angle = 0.0;
        double track_distance = 0.0;
        unsigned kept = 0;
        for (unsigned i = 0; i < track.observations.size(); ++i) {
            const Observation& obs = track.observations[i];
            if (obs.image_id >= scene.images.size()) continue;
            const Image& image = scene.images[obs.image_id];
            if (!image.registered ||
                obs.feature_id >= image.features.keypoints.size())
                continue;
            const PinholeCamera& camera = scene.camera_of(image);
            const Vec3 Xc = image.pose.transform_world_to_camera(track.position);
            const auto& kp = image.features.keypoints[obs.feature_id];
            const Vec3 observed =
                camera.unproject_normalized({kp.x, kp.y});
            const double norm = Xc.norm();
            if (!(norm > 1e-12)) continue;
            const double cosine =
                observed.dot(Xc) / norm;
            const double minimum_cosine =
                std::cos(camera.pixel_error_to_angular(max_reproj_error_px));
            if (cosine < minimum_cosine) continue;
            const Vec2 projected = camera.project(Xc);
            const double error =
                (projected - Vec2(kp.x, kp.y)).norm();
            if (!std::isfinite(error) || error > max_reproj_error_px) continue;
            if (kept != i) std::swap(track.observations[kept], track.observations[i]);
            ++kept;
            track_px += error;
            track_cos_angle += cosine;
            track_distance += norm;
        }
        track.num_inliers = static_cast<std::uint8_t>(std::min<unsigned>(kept, 255));
        if (track.num_inliers < 2 ||
            track_min_ray_angle_deg(track, scene) < min_angle_deg) {
            track.num_inliers = 0;
            continue;
        }
        sum_px += track_px;
        sum_cos_angle += track_cos_angle;
        counted += track.num_inliers;
        average_distances.push_back(
            track_distance / static_cast<double>(track.num_inliers));
    }

    if (average_distances.size() > 1000 &&
        (mult_depth_near > 0.F || mult_depth_far > 0.F)) {
        std::vector<double> sorted = average_distances;
        std::nth_element(
            sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
        const double median = sorted[sorted.size() / 2];
        const double minimum =
            mult_depth_near > 0.F ? mult_depth_near * median : 0.0;
        const double maximum =
            mult_depth_far > 0.F
            ? mult_depth_far * median
            : std::numeric_limits<double>::max();
        std::size_t distance_index = 0;
        for (Track& track : scene.tracks) {
            if (!track.is_triangulated()) continue;
            const double distance = average_distances[distance_index++];
            if (distance < minimum || distance > maximum)
                track.num_inliers = 0;
        }
    }
    if (counted == 0) return {0.F, 0.F};
    unsigned kept_tracks = 0;
    for (const Track& track : scene.tracks)
        kept_tracks += track.is_triangulated() ? 1U : 0U;
    const std::pair<float, float> result{
        static_cast<float>(sum_px / counted),
        static_cast<float>(
            std::acos(std::clamp(
                sum_cos_angle / counted, -1.0, 1.0)) *
            180.0 / 3.14159265358979323846)};
    core::Logger::instance().debug(
        "track filter: kept=", kept_tracks, '/', scene.tracks.size(),
        " observations=", counted, " mean_reproj_px=", result.first,
        " mean_angle_deg=", result.second);
    return result;
}

}  // namespace aetherscan::sfm
