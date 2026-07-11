#include "aetherscan/sfm/tracks.hpp"

#include "aetherscan/sfm/triangulation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
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

Index feature_offset(const Scene& scene, const Index image_id) {
    Index offset = 0;
    for (Index i = 0; i < image_id; ++i)
        offset += static_cast<Index>(scene.images[i].features.keypoints.size());
    return offset;
}

}  // namespace

void build_tracks(Scene& scene, const float min_pair_weight) {
    scene.tracks.clear();
    std::size_t total_features = 0;
    for (const auto& image : scene.images)
        total_features += image.features.keypoints.size();
    if (total_features == 0) return;

    DisjointSet ds(total_features);
    // Per-component: image_id -> observation count (must stay <= 1)
    std::unordered_map<Index, std::unordered_map<Index, unsigned>> component_images;

    const auto accumulate = [&](const Index global_id, const Index image_id) {
        const Index root = ds.find(global_id);
        auto& images = component_images[root];
        ++images[image_id];
    };

    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || pair.matches.empty()) continue;
        if (pair.composite_weight() <= min_pair_weight) continue;
        const Index off1 = feature_offset(scene, pair.id1);
        const Index off2 = feature_offset(scene, pair.id2);
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
            for (const auto& [image_id, count] : map1) {
                if (map2.count(image_id) && map2[image_id] + count > 1) {
                    conflict = true;
                    break;
                }
            }
            if (conflict) continue;

            ds.unite(r1, r2);
            const Index root = ds.find(r1);
            auto& merged = component_images[root];
            if (root != r1) {
                for (const auto& [image_id, count] : map1) merged[image_id] += count;
            }
            if (root != r2) {
                for (const auto& [image_id, count] : map2) merged[image_id] += count;
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
}

std::pair<float, float> filter_tracks(
    Scene& scene,
    const float max_reproj_error_px,
    const float min_angle_deg,
    const float mult_depth_near,
    const float mult_depth_far) {
    std::vector<double> depths;
    depths.reserve(scene.tracks.size() * 2);
    for (const Track& track : scene.tracks) {
        if (!track.is_triangulated()) continue;
        for (unsigned i = 0; i < track.num_inliers; ++i) {
            const Image& image = scene.images[track.observations[i].image_id];
            depths.push_back(image.pose.transform_world_to_camera(track.position).z());
        }
    }
    double median_depth = 1.0;
    if (!depths.empty()) {
        std::nth_element(depths.begin(), depths.begin() + depths.size() / 2, depths.end());
        median_depth = std::max(1e-6, depths[depths.size() / 2]);
    }
    const double near_z = mult_depth_near > 0 ? mult_depth_near * median_depth : 0.0;
    const double far_z = mult_depth_far > 0 ? mult_depth_far * median_depth : 1e12;

    double sum_px = 0.0;
    double sum_deg = 0.0;
    unsigned counted = 0;
    for (Track& track : scene.tracks) {
        if (!track.is_triangulated()) continue;
        unsigned kept = 0;
        for (unsigned i = 0; i < track.num_inliers; ++i) {
            const Observation& obs = track.observations[i];
            const Image& image = scene.images[obs.image_id];
            const PinholeCamera& camera = scene.camera_of(image);
            const Vec3 Xc = image.pose.transform_world_to_camera(track.position);
            Vec2 proj;
            if (!camera.project_checked(Xc, proj)) continue;
            if (Xc.z() < near_z || Xc.z() > far_z) continue;
            const auto& kp = image.features.keypoints[obs.feature_id];
            const double err = (proj - Vec2(kp.x, kp.y)).norm();
            if (err > max_reproj_error_px) continue;
            if (kept != i) std::swap(track.observations[kept], track.observations[i]);
            ++kept;
            sum_px += err;
            sum_deg += camera.pixel_error_to_angular(err) * 180.0 / 3.14159265358979323846;
            ++counted;
        }
        track.num_inliers = static_cast<std::uint8_t>(std::min<unsigned>(kept, 255));
        if (track.num_inliers < 2 ||
            track_min_ray_angle_deg(track, scene) < min_angle_deg) {
            track.num_inliers = 0;
        }
    }
    if (counted == 0) return {0.F, 0.F};
    return {
        static_cast<float>(sum_px / counted),
        static_cast<float>(sum_deg / counted)};
}

}  // namespace aetherscan::sfm
