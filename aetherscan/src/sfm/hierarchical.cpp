#include "sfm/hierarchical.hpp"

#include "core/logging.hpp"
#include "parallel/thread_pool.hpp"
#include "sfm/bundle.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"

#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace aetherscan::sfm {
namespace {

constexpr double k_weight_multiplier = 10.0;

using Cluster = std::vector<Index>;

struct Graph {
    std::vector<std::vector<std::pair<Index, int>>> adjacency;
};

Graph build_graph(const Scene& scene, const ClusterConfig& config) {
    Graph graph;
    graph.adjacency.resize(scene.images.size());
    for (const ImagePair& pair : scene.pairs) {
        if (!pair.active || pair.id1 >= scene.images.size() ||
            pair.id2 >= scene.images.size() ||
            pair.num_inliers() < config.min_common_tracks ||
            pair.composite_weight() < config.min_pair_weight) {
            continue;
        }
        const int weight = std::max(
            1, static_cast<int>(std::lround(
                   static_cast<double>(pair.composite_weight()) * k_weight_multiplier)));
        graph.adjacency[pair.id1].emplace_back(pair.id2, weight);
        graph.adjacency[pair.id2].emplace_back(pair.id1, weight);
    }
    for (auto& neighbors : graph.adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
    }
    return graph;
}

std::vector<int> cluster_owners(
    const std::vector<Cluster>& clusters,
    const std::size_t node_count) {
    std::vector<int> owner(node_count, -1);
    for (std::size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        for (const Index node : clusters[cluster_id])
            owner[node] = static_cast<int>(cluster_id);
    }
    return owner;
}

void erase_empty_clusters(std::vector<Cluster>& clusters) {
    clusters.erase(
        std::remove_if(
            clusters.begin(), clusters.end(),
            [](const Cluster& cluster) { return cluster.empty(); }),
        clusters.end());
}

void refine_local_search(
    std::vector<Cluster>& clusters,
    const Graph& graph,
    const ClusterConfig& config) {
    for (unsigned iteration = 0; iteration < 20; ++iteration) {
        bool changed = false;
        std::vector<int> owner = cluster_owners(clusters, graph.adjacency.size());
        for (Index node = 0; node < graph.adjacency.size(); ++node) {
            const int current = owner[node];
            if (current < 0) continue;
            std::map<int, int> weights;
            for (const auto [neighbor, weight] : graph.adjacency[node]) {
                const int target = owner[neighbor];
                if (target >= 0) weights[target] += weight;
            }
            const int internal = weights[current];
            int best = current;
            int best_gain = 0;
            for (const auto [target, weight] : weights) {
                if (target == current ||
                    clusters[static_cast<std::size_t>(target)].size() >=
                        config.max_views_per_cluster) {
                    continue;
                }
                const int gain = weight - internal;
                if (gain > best_gain ||
                    (gain == best_gain && gain > 0 && target < best)) {
                    best = target;
                    best_gain = gain;
                }
            }
            if (best == current) continue;
            Cluster& source = clusters[static_cast<std::size_t>(current)];
            const auto it = std::find(source.begin(), source.end(), node);
            if (it == source.end()) continue;
            source.erase(it);
            clusters[static_cast<std::size_t>(best)].push_back(node);
            owner[node] = best;
            changed = true;
        }
        if (!changed) break;
    }
    erase_empty_clusters(clusters);
}

void merge_small_clusters(
    std::vector<Cluster>& clusters,
    const Graph& graph,
    const ClusterConfig& config) {
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<int> owner = cluster_owners(clusters, graph.adjacency.size());
        for (std::size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
            Cluster& cluster = clusters[cluster_id];
            if (cluster.empty() || cluster.size() >= config.min_views_per_cluster)
                continue;
            std::map<int, std::int64_t> weights;
            for (const Index node : cluster) {
                for (const auto [neighbor, weight] : graph.adjacency[node]) {
                    const int target = owner[neighbor];
                    if (target >= 0 && target != static_cast<int>(cluster_id))
                        weights[target] += weight;
                }
            }
            int best = -1;
            std::int64_t best_weight = -1;
            for (const auto [target, weight] : weights) {
                const Cluster& destination = clusters[static_cast<std::size_t>(target)];
                if (destination.size() + cluster.size() >
                    config.max_views_per_cluster + config.max_over_capacity) {
                    continue;
                }
                if (weight > best_weight || (weight == best_weight && target < best)) {
                    best = target;
                    best_weight = weight;
                }
            }
            if (best < 0) continue;
            Cluster moved = std::move(cluster);
            cluster.clear();
            Cluster& destination = clusters[static_cast<std::size_t>(best)];
            destination.insert(destination.end(), moved.begin(), moved.end());
            changed = true;
        }
    }
    erase_empty_clusters(clusters);
}

void split_disconnected(std::vector<Cluster>& clusters, const Graph& graph) {
    std::vector<Cluster> split;
    for (const Cluster& cluster : clusters) {
        std::unordered_set<Index> remaining(cluster.begin(), cluster.end());
        while (!remaining.empty()) {
            const Index start = *std::min_element(remaining.begin(), remaining.end());
            remaining.erase(start);
            Cluster component;
            std::queue<Index> queue;
            queue.push(start);
            while (!queue.empty()) {
                const Index node = queue.front();
                queue.pop();
                component.push_back(node);
                for (const auto [neighbor, weight] : graph.adjacency[node]) {
                    (void)weight;
                    const auto it = remaining.find(neighbor);
                    if (it == remaining.end()) continue;
                    remaining.erase(it);
                    queue.push(neighbor);
                }
            }
            std::sort(component.begin(), component.end());
            split.push_back(std::move(component));
        }
    }
    clusters = std::move(split);
}

void rescue_orphans(
    std::vector<Cluster>& clusters,
    const Graph& graph,
    const ClusterConfig& config) {
    std::vector<int> owner = cluster_owners(clusters, graph.adjacency.size());
    for (std::size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        if (clusters[cluster_id].empty() ||
            clusters[cluster_id].size() >= config.min_views_per_cluster) {
            continue;
        }
        Cluster nodes = std::move(clusters[cluster_id]);
        clusters[cluster_id].clear();
        std::sort(nodes.begin(), nodes.end());
        for (const Index node : nodes) {
            std::map<int, std::int64_t> weights;
            for (const auto [neighbor, weight] : graph.adjacency[node]) {
                const int target = owner[neighbor];
                if (target >= 0 && target != static_cast<int>(cluster_id))
                    weights[target] += weight;
            }
            int best = -1;
            std::int64_t best_weight = -1;
            for (const auto [target, weight] : weights) {
                if (clusters[static_cast<std::size_t>(target)].size() >=
                    config.max_views_per_cluster + config.max_over_capacity) {
                    continue;
                }
                if (weight > best_weight || (weight == best_weight && target < best)) {
                    best = target;
                    best_weight = weight;
                }
            }
            if (best < 0) {
                for (std::size_t target = 0; target < clusters.size(); ++target) {
                    if (target != cluster_id && !clusters[target].empty() &&
                        clusters[target].size() <
                            config.max_views_per_cluster + config.max_over_capacity) {
                        best = static_cast<int>(target);
                        break;
                    }
                }
            }
            if (best < 0) {
                clusters[cluster_id].push_back(node);
                owner[node] = static_cast<int>(cluster_id);
            } else {
                clusters[static_cast<std::size_t>(best)].push_back(node);
                owner[node] = best;
            }
        }
    }
    erase_empty_clusters(clusters);
}

std::vector<Cluster> cluster_scene(const Scene& scene, const ClusterConfig& config) {
    const std::size_t count = scene.images.size();
    if (count == 0) return {};
    if (config.max_views_per_cluster == 0 ||
        count <= config.max_views_per_cluster) {
        Cluster all(count);
        std::iota(all.begin(), all.end(), Index{0});
        return {std::move(all)};
    }

    const Graph graph = build_graph(scene, config);
    std::vector<Cluster> clusters(count);
    std::vector<int> owner(count);
    for (Index node = 0; node < count; ++node) {
        clusters[node].push_back(node);
        owner[node] = static_cast<int>(node);
    }

    struct Edge {
        int first{};
        int second{};
        std::int64_t weight{};
    };
    struct EdgeLess {
        bool operator()(const Edge& a, const Edge& b) const {
            if (a.weight != b.weight) return a.weight < b.weight;
            if (a.first != b.first) return a.first > b.first;
            return a.second > b.second;
        }
    };

    auto rebuild = [&]() {
        std::map<std::pair<int, int>, std::int64_t> weights;
        for (Index node = 0; node < count; ++node) {
            for (const auto [neighbor, weight] : graph.adjacency[node]) {
                if (node >= neighbor) continue;
                int a = owner[node];
                int b = owner[neighbor];
                if (a == b) continue;
                if (a > b) std::swap(a, b);
                weights[{a, b}] += weight;
            }
        }
        std::priority_queue<Edge, std::vector<Edge>, EdgeLess> queue;
        for (const auto& [ends, weight] : weights)
            queue.push({ends.first, ends.second, weight});
        return std::pair{std::move(weights), std::move(queue)};
    };

    auto [weights, queue] = rebuild();
    unsigned merges_since_refine = 0;
    while (!queue.empty()) {
        const Edge edge = queue.top();
        queue.pop();
        const auto weight_it = weights.find({edge.first, edge.second});
        if (weight_it == weights.end() || weight_it->second != edge.weight ||
            clusters[static_cast<std::size_t>(edge.first)].empty() ||
            clusters[static_cast<std::size_t>(edge.second)].empty()) {
            continue;
        }
        Cluster& first = clusters[static_cast<std::size_t>(edge.first)];
        Cluster& second = clusters[static_cast<std::size_t>(edge.second)];
        if (first.size() + second.size() > config.max_views_per_cluster) continue;
        for (const Index node : second) owner[node] = edge.first;
        first.insert(first.end(), second.begin(), second.end());
        second.clear();
        ++merges_since_refine;

        const unsigned refine_interval =
            std::max(10u, config.max_views_per_cluster / 10u);
        if (config.refine_weak_edges && merges_since_refine >= refine_interval) {
            refine_local_search(clusters, graph, config);
            owner = cluster_owners(clusters, count);
            std::tie(weights, queue) = rebuild();
            merges_since_refine = 0;
        } else {
            owner = cluster_owners(clusters, count);
            std::tie(weights, queue) = rebuild();
        }
    }

    if (config.refine_weak_edges) refine_local_search(clusters, graph, config);
    merge_small_clusters(clusters, graph, config);
    split_disconnected(clusters, graph);
    rescue_orphans(clusters, graph, config);
    for (Cluster& cluster : clusters) std::sort(cluster.begin(), cluster.end());
    std::sort(clusters.begin(), clusters.end(), [](const Cluster& a, const Cluster& b) {
        return a.front() < b.front();
    });
    return clusters;
}

HierarchicalSubscene extract_subscene(
    const Scene& parent,
    const Cluster& global_images,
    const unsigned thread_count) {
    HierarchicalSubscene output;
    output.scene.thread_count = thread_count;
    output.local_to_global = global_images;
    std::vector<Index> global_to_local(parent.images.size(), k_invalid);
    std::unordered_map<Index, Index> camera_map;

    for (Index local_id = 0; local_id < global_images.size(); ++local_id) {
        const Index global_id = global_images[local_id];
        global_to_local[global_id] = local_id;
        Image image = parent.images[global_id];
        const Index global_camera = image.camera_id;
        auto [it, inserted] =
            camera_map.emplace(global_camera, static_cast<Index>(output.scene.cameras.size()));
        if (inserted) {
            PinholeCamera camera = parent.cameras[global_camera];
            camera.id = it->second;
            output.scene.cameras.push_back(std::move(camera));
        }
        image.id = local_id;
        image.camera_id = it->second;
        image.registered = false;
        image.pose = Pose3D::identity();
        output.scene.images.push_back(std::move(image));
    }

    for (const ImagePair& pair : parent.pairs) {
        if (pair.id1 >= global_to_local.size() || pair.id2 >= global_to_local.size())
            continue;
        const Index local1 = global_to_local[pair.id1];
        const Index local2 = global_to_local[pair.id2];
        if (local1 == k_invalid || local2 == k_invalid) continue;
        ImagePair local_pair = pair;
        local_pair.id1 = local1;
        local_pair.id2 = local2;
        output.scene.pairs.push_back(std::move(local_pair));
    }

    for (const Track& source : parent.tracks) {
        Track destination;
        destination.position = source.position;
        for (std::size_t i = 0; i < source.observations.size(); ++i) {
            const Observation& observation = source.observations[i];
            if (observation.image_id >= global_to_local.size()) continue;
            const Index local = global_to_local[observation.image_id];
            if (local == k_invalid) continue;
            destination.observations.push_back({local, observation.feature_id});
            if (i < source.num_inliers && destination.num_inliers < 255)
                ++destination.num_inliers;
        }
        if (destination.observations.size() >= 2)
            output.scene.tracks.push_back(std::move(destination));
    }
    return output;
}

bool fit_similarity(
    const std::vector<Vec3>& source,
    const std::vector<Vec3>& destination,
    const std::vector<std::size_t>& indices,
    Similarity3& transform) {
    if (indices.size() < 3) return false;
    Vec3 source_mean = Vec3::Zero();
    Vec3 destination_mean = Vec3::Zero();
    for (const std::size_t index : indices) {
        source_mean += source[index];
        destination_mean += destination[index];
    }
    const double inverse_count = 1.0 / static_cast<double>(indices.size());
    source_mean *= inverse_count;
    destination_mean *= inverse_count;

    Mat3 covariance = Mat3::Zero();
    double variance = 0.0;
    for (const std::size_t index : indices) {
        const Vec3 source_centered = source[index] - source_mean;
        const Vec3 destination_centered = destination[index] - destination_mean;
        covariance += destination_centered * source_centered.transpose();
        variance += source_centered.squaredNorm();
    }
    covariance *= inverse_count;
    variance *= inverse_count;
    if (!std::isfinite(variance) || variance <= 1e-15) return false;

    Eigen::JacobiSVD<Mat3> svd(
        covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 correction = Mat3::Identity();
    if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0)
        correction(2, 2) = -1.0;
    transform.R = svd.matrixU() * correction * svd.matrixV().transpose();
    transform.scale =
        (svd.singularValues().array() * correction.diagonal().array()).sum() /
        variance;
    transform.t = destination_mean - transform.scale * transform.R * source_mean;
    return transform.R.allFinite() && transform.t.allFinite() &&
           std::isfinite(transform.scale) && transform.scale > 1e-12;
}

Mat3 project_rotation(const Mat3& matrix) {
    Eigen::JacobiSVD<Mat3> svd(
        matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 correction = Mat3::Identity();
    if ((svd.matrixU() * svd.matrixV().transpose()).determinant() < 0.0)
        correction(2, 2) = -1.0;
    return svd.matrixU() * correction * svd.matrixV().transpose();
}

std::uint64_t observation_key(const Index image, const Index feature) {
    return (static_cast<std::uint64_t>(image) << 32u) |
           static_cast<std::uint64_t>(feature);
}

struct ScenePair {
    std::size_t first{};
    std::size_t second{};
    Similarity3 relative;
    unsigned inliers{};
};

struct CachedTrackPoint {
    Index track_id{k_invalid};
    Vec3 position{Vec3::Zero()};
};

std::vector<ScenePair> estimate_scene_pairs(
    const Scene& parent,
    const std::vector<HierarchicalSubscene>& subscenes,
    const std::vector<int>& global_owner,
    const std::vector<Index>& global_local,
    const GlobalAlignmentConfig& config) {
    std::vector<std::unordered_map<std::uint64_t, CachedTrackPoint>> caches(
        subscenes.size());
    for (std::size_t scene_id = 0; scene_id < subscenes.size(); ++scene_id) {
        const Scene& subscene = subscenes[scene_id].scene;
        for (Index track_id = 0; track_id < subscene.tracks.size(); ++track_id) {
            const Track& track = subscene.tracks[track_id];
            if (!track.is_triangulated()) continue;
            const unsigned count = std::min<unsigned>(
                track.num_inliers,
                static_cast<unsigned>(track.observations.size()));
            for (unsigned i = 0; i < count; ++i) {
                const Observation& observation = track.observations[i];
                caches[scene_id].emplace(
                    observation_key(observation.image_id, observation.feature_id),
                    CachedTrackPoint{track_id, track.position});
            }
        }
    }

    struct Link {
        const ImagePair* pair{};
        Index local_first{k_invalid};
        Index local_second{k_invalid};
        bool first_is_query{true};
    };
    std::map<std::pair<std::size_t, std::size_t>, std::vector<Link>> grouped;
    for (const ImagePair& pair : parent.pairs) {
        if (!pair.active || pair.id1 >= global_owner.size() ||
            pair.id2 >= global_owner.size() ||
            pair.num_inliers() < config.min_common_tracks ||
            pair.composite_weight() < config.min_pair_weight) {
            continue;
        }
        const int owner1 = global_owner[pair.id1];
        const int owner2 = global_owner[pair.id2];
        if (owner1 < 0 || owner2 < 0 || owner1 == owner2) continue;
        Link link;
        link.pair = &pair;
        if (owner1 < owner2) {
            link.local_first = global_local[pair.id1];
            link.local_second = global_local[pair.id2];
            link.first_is_query = true;
            grouped[{static_cast<std::size_t>(owner1),
                     static_cast<std::size_t>(owner2)}]
                .push_back(link);
        } else {
            link.local_first = global_local[pair.id2];
            link.local_second = global_local[pair.id1];
            link.first_is_query = false;
            grouped[{static_cast<std::size_t>(owner2),
                     static_cast<std::size_t>(owner1)}]
                .push_back(link);
        }
    }

    std::vector<ScenePair> result;
    for (const auto& [scene_ids, links] : grouped) {
        std::vector<Vec3> source;
        std::vector<Vec3> destination;
        std::vector<Index> source_track_ids;
        std::vector<Index> destination_track_ids;
        std::unordered_set<std::uint64_t> seen_track_pairs;
        for (const Link& link : links) {
            for (const FeatureMatch& match : link.pair->matches) {
                const Index feature_first =
                    link.first_is_query ? match.query : match.train;
                const Index feature_second =
                    link.first_is_query ? match.train : match.query;
                const auto first_it = caches[scene_ids.first].find(
                    observation_key(link.local_first, feature_first));
                const auto second_it = caches[scene_ids.second].find(
                    observation_key(link.local_second, feature_second));
                if (first_it == caches[scene_ids.first].end() ||
                    second_it == caches[scene_ids.second].end()) {
                    continue;
                }
                const std::uint64_t track_pair =
                    (static_cast<std::uint64_t>(first_it->second.track_id) << 32u) |
                    static_cast<std::uint64_t>(second_it->second.track_id);
                if (!seen_track_pairs.insert(track_pair).second) continue;
                source.push_back(first_it->second.position);
                destination.push_back(second_it->second.position);
                source_track_ids.push_back(first_it->second.track_id);
                destination_track_ids.push_back(second_it->second.track_id);
            }
        }
        if (source.size() < config.min_common_tracks) {
            core::Logger::instance().debug(
                "hierarchical align skip: scenes=", scene_ids.first, '-',
                scene_ids.second, " common_points=", source.size(),
                " < min=", config.min_common_tracks);
            continue;
        }
        Vec3 minimum = source.front();
        Vec3 maximum = source.front();
        for (const Vec3& point : source) {
            minimum = minimum.cwiseMin(point);
            maximum = maximum.cwiseMax(point);
        }
        const double threshold =
            config.ransac_relative_threshold * (maximum - minimum).norm();
        if (!std::isfinite(threshold) || threshold <= 1e-12) continue;
        Similarity3 relative;
        const std::uint32_t seed =
            config.random_seed ^
            static_cast<std::uint32_t>(scene_ids.first * 0x9E3779B1u) ^
            static_cast<std::uint32_t>(scene_ids.second * 0x85EBCA77u);
        std::vector<std::size_t> final_inlier_ids;
        const unsigned inliers = estimate_similarity_transform(
            source, destination, relative, threshold,
            config.ransac_iterations, seed, &final_inlier_ids);
        const double inlier_ratio =
            static_cast<double>(inliers) / static_cast<double>(source.size());
        std::unordered_set<Index> unique_source_inliers;
        std::unordered_set<Index> unique_destination_inliers;
        for (const std::size_t inlier : final_inlier_ids) {
            unique_source_inliers.insert(source_track_ids[inlier]);
            unique_destination_inliers.insert(destination_track_ids[inlier]);
        }
        const unsigned unique_support = static_cast<unsigned>(std::min(
            unique_source_inliers.size(), unique_destination_inliers.size()));
        // A large one-to-one-equivalent consensus remains reliable when track
        // splitting depresses the raw correspondence inlier ratio.
        const unsigned strong_consensus =
            std::max(config.min_common_tracks * 2U, 50U);
        if (inliers < config.min_common_tracks ||
            (inlier_ratio < config.minimum_inlier_ratio &&
             unique_support < strong_consensus)) {
            core::Logger::instance().info(
                "hierarchical align reject: scenes=", scene_ids.first, '-',
                scene_ids.second, " points=", source.size(),
                " inliers=", inliers, " ratio=", inlier_ratio,
                " unique_support=", unique_support,
                " threshold=", threshold);
            continue;
        }
        if (inlier_ratio < config.minimum_inlier_ratio) {
            core::Logger::instance().info(
                "hierarchical align accept strong consensus: scenes=",
                scene_ids.first, '-', scene_ids.second,
                " inliers=", inliers, " ratio=", inlier_ratio,
                " unique_support=", unique_support);
        }
        result.push_back(
            {scene_ids.first, scene_ids.second, relative, inliers});
    }
    return result;
}

std::vector<bool> largest_connected_component(
    const std::size_t count,
    const std::vector<ScenePair>& pairs) {
    std::vector<std::vector<std::size_t>> adjacency(count);
    for (const ScenePair& pair : pairs) {
        adjacency[pair.first].push_back(pair.second);
        adjacency[pair.second].push_back(pair.first);
    }
    std::vector<bool> best(count, false);
    std::vector<bool> visited(count, false);
    for (std::size_t start = 0; start < count; ++start) {
        if (visited[start]) continue;
        std::vector<std::size_t> component;
        std::queue<std::size_t> queue;
        queue.push(start);
        visited[start] = true;
        while (!queue.empty()) {
            const std::size_t node = queue.front();
            queue.pop();
            component.push_back(node);
            for (const std::size_t neighbor : adjacency[node]) {
                if (visited[neighbor]) continue;
                visited[neighbor] = true;
                queue.push(neighbor);
            }
        }
        if (component.size() <= static_cast<std::size_t>(
                                    std::count(best.begin(), best.end(), true))) {
            continue;
        }
        std::fill(best.begin(), best.end(), false);
        for (const std::size_t node : component) best[node] = true;
    }
    return best;
}

bool estimate_global_transforms(
    const std::size_t count,
    const std::vector<ScenePair>& pairs,
    const std::vector<bool>& included,
    std::vector<Similarity3>& transforms) {
    transforms.assign(count, {});
    std::size_t anchor = count;
    double anchor_weight = -1.0;
    std::vector<double> connection_weight(count, 0.0);
    for (const ScenePair& pair : pairs) {
        if (!included[pair.first] || !included[pair.second]) continue;
        connection_weight[pair.first] += pair.inliers;
        connection_weight[pair.second] += pair.inliers;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (included[i] &&
            (connection_weight[i] > anchor_weight ||
             (connection_weight[i] == anchor_weight && i < anchor))) {
            anchor = i;
            anchor_weight = connection_weight[i];
        }
    }
    if (anchor == count) return false;

    std::vector<bool> initialized(count, false);
    initialized[anchor] = true;
    std::queue<std::size_t> queue;
    queue.push(anchor);
    while (!queue.empty()) {
        const std::size_t node = queue.front();
        queue.pop();
        for (const ScenePair& pair : pairs) {
            std::size_t neighbor = count;
            Mat3 neighbor_rotation = Mat3::Identity();
            if (pair.first == node && included[pair.second]) {
                neighbor = pair.second;
                neighbor_rotation = transforms[node].R * pair.relative.R.transpose();
            } else if (pair.second == node && included[pair.first]) {
                neighbor = pair.first;
                neighbor_rotation = transforms[node].R * pair.relative.R;
            }
            if (neighbor == count || initialized[neighbor]) continue;
            transforms[neighbor].R = neighbor_rotation;
            initialized[neighbor] = true;
            queue.push(neighbor);
        }
    }

    for (unsigned iteration = 0; iteration < 30; ++iteration) {
        std::vector<Mat3> updated(count, Mat3::Zero());
        std::vector<double> weights(count, 0.0);
        for (const ScenePair& pair : pairs) {
            if (!included[pair.first] || !included[pair.second]) continue;
            const double weight = static_cast<double>(pair.inliers);
            updated[pair.first] +=
                weight * transforms[pair.second].R * pair.relative.R;
            updated[pair.second] +=
                weight * transforms[pair.first].R * pair.relative.R.transpose();
            weights[pair.first] += weight;
            weights[pair.second] += weight;
        }
        for (std::size_t i = 0; i < count; ++i) {
            if (!included[i] || i == anchor || weights[i] <= 0.0) continue;
            transforms[i].R = project_rotation(updated[i]);
        }
        transforms[anchor].R = Mat3::Identity();
    }

    std::vector<std::size_t> variables;
    std::vector<int> variable_index(count, -1);
    for (std::size_t i = 0; i < count; ++i) {
        if (included[i] && i != anchor) {
            variable_index[i] = static_cast<int>(variables.size());
            variables.push_back(i);
        }
    }
    std::vector<const ScenePair*> active_pairs;
    for (const ScenePair& pair : pairs) {
        if (included[pair.first] && included[pair.second])
            active_pairs.push_back(&pair);
    }
    if (variables.empty()) return true;

    Eigen::MatrixXd A =
        Eigen::MatrixXd::Zero(active_pairs.size(), variables.size());
    Eigen::VectorXd scale_rhs = Eigen::VectorXd::Zero(active_pairs.size());
    Eigen::MatrixXd translation_rhs =
        Eigen::MatrixXd::Zero(active_pairs.size(), 3);
    for (std::size_t row = 0; row < active_pairs.size(); ++row) {
        const ScenePair& pair = *active_pairs[row];
        const double weight = std::sqrt(static_cast<double>(pair.inliers));
        if (pair.first != anchor)
            A(static_cast<Eigen::Index>(row), variable_index[pair.first]) = -weight;
        if (pair.second != anchor)
            A(static_cast<Eigen::Index>(row), variable_index[pair.second]) = weight;
        scale_rhs(static_cast<Eigen::Index>(row)) =
            -weight * std::log(pair.relative.scale);
    }
    const Eigen::VectorXd log_scales = A.colPivHouseholderQr().solve(scale_rhs);
    if (!log_scales.allFinite()) return false;
    for (std::size_t i = 0; i < variables.size(); ++i)
        transforms[variables[i]].scale =
            std::exp(log_scales(static_cast<Eigen::Index>(i)));
    transforms[anchor].scale = 1.0;

    for (std::size_t row = 0; row < active_pairs.size(); ++row) {
        const ScenePair& pair = *active_pairs[row];
        const double weight = std::sqrt(static_cast<double>(pair.inliers));
        const Vec3 origin_second_in_first =
            -(pair.relative.R.transpose() * pair.relative.t) /
            pair.relative.scale;
        const Vec3 difference =
            transforms[pair.first].scale * transforms[pair.first].R *
            origin_second_in_first;
        translation_rhs.row(static_cast<Eigen::Index>(row)) =
            (weight * difference).transpose();
    }
    const Eigen::MatrixXd translations =
        A.colPivHouseholderQr().solve(translation_rhs);
    if (!translations.allFinite()) return false;
    transforms[anchor].t = Vec3::Zero();
    for (std::size_t i = 0; i < variables.size(); ++i)
        transforms[variables[i]].t =
            translations.row(static_cast<Eigen::Index>(i)).transpose();
    return true;
}

void apply_similarity(Scene& scene, const Similarity3& transform) {
    for (Image& image : scene.images) {
        if (!image.registered) continue;
        image.pose.C = transform.apply(image.pose.C);
        image.pose.R = image.pose.R * transform.R.transpose();
    }
    for (Track& track : scene.tracks) {
        if (track.is_triangulated())
            track.position = transform.apply(track.position);
    }
}

struct MergeDisjointSet {
    std::vector<Index> parent;
    std::vector<unsigned> rank;

    explicit MergeDisjointSet(const std::size_t count) : parent(count), rank(count, 0) {
        std::iota(parent.begin(), parent.end(), Index{0});
    }

    Index find(Index node) {
        while (parent[node] != node) {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        return node;
    }
};

struct RootMetadata {
    Vec3 position{Vec3::Zero()};
    unsigned inlier_weight{0};
    bool has_position{false};
    std::unordered_set<Index> images;
    bool active{false};
};

void merge_tracks(
    Scene& parent,
    const std::vector<HierarchicalSubscene>& subscenes,
    const std::vector<bool>& included,
    const std::vector<int>& global_owner,
    const GlobalAlignmentConfig& config) {
    std::vector<Index> offsets(parent.images.size() + 1, 0);
    std::size_t total_features = 0;
    for (std::size_t i = 0; i < parent.images.size(); ++i) {
        offsets[i] = static_cast<Index>(total_features);
        total_features += parent.images[i].features.keypoints.size();
    }
    if (total_features > std::numeric_limits<Index>::max()) {
        parent.tracks.clear();
        return;
    }
    offsets.back() = static_cast<Index>(total_features);
    MergeDisjointSet sets(total_features);
    std::vector<RootMetadata> metadata(total_features);
    std::vector<bool> counted(total_features, false);

    auto unite = [&](Index first, Index second, const bool guarded, const double proximity) {
        Index root_first = sets.find(first);
        Index root_second = sets.find(second);
        if (root_first == root_second) return true;
        if (sets.rank[root_first] < sets.rank[root_second])
            std::swap(root_first, root_second);
        RootMetadata& first_meta = metadata[root_first];
        RootMetadata& second_meta = metadata[root_second];
        if (guarded) {
            for (const Index image : second_meta.images) {
                if (first_meta.images.contains(image)) return false;
            }
            if (first_meta.has_position && second_meta.has_position &&
                proximity > 0.0 &&
                (first_meta.position - second_meta.position).norm() > proximity) {
                return false;
            }
        }
        sets.parent[root_second] = root_first;
        if (sets.rank[root_first] == sets.rank[root_second])
            ++sets.rank[root_first];
        if (first_meta.has_position && second_meta.has_position) {
            const double first_weight =
                static_cast<double>(std::max(first_meta.inlier_weight, 1u));
            const double second_weight =
                static_cast<double>(std::max(second_meta.inlier_weight, 1u));
            first_meta.position =
                (first_weight * first_meta.position +
                 second_weight * second_meta.position) /
                (first_weight + second_weight);
        } else if (second_meta.has_position) {
            first_meta.position = second_meta.position;
            first_meta.has_position = true;
        }
        first_meta.inlier_weight += second_meta.inlier_weight;
        first_meta.images.insert(
            second_meta.images.begin(), second_meta.images.end());
        first_meta.active = first_meta.active || second_meta.active;
        second_meta = {};
        return true;
    };

    Vec3 bbox_min = Vec3::Constant(std::numeric_limits<double>::infinity());
    Vec3 bbox_max = Vec3::Constant(-std::numeric_limits<double>::infinity());
    for (std::size_t scene_id = 0; scene_id < subscenes.size(); ++scene_id) {
        if (!included[scene_id]) continue;
        const HierarchicalSubscene& subscene = subscenes[scene_id];
        for (const Track& track : subscene.scene.tracks) {
            const unsigned observation_count = config.merge_track_inliers_only
                ? std::min<unsigned>(
                      track.num_inliers,
                      static_cast<unsigned>(track.observations.size()))
                : static_cast<unsigned>(track.observations.size());
            if (observation_count < 2) continue;
            Index first_gid = k_invalid;
            std::vector<std::pair<Index, Index>> observations;
            for (unsigned i = 0; i < observation_count; ++i) {
                const Observation& observation = track.observations[i];
                if (observation.image_id >= subscene.local_to_global.size()) continue;
                const Index global_image =
                    subscene.local_to_global[observation.image_id];
                if (global_image >= parent.images.size() ||
                    observation.feature_id >=
                        parent.images[global_image].features.keypoints.size()) {
                    continue;
                }
                const Index gid = offsets[global_image] + observation.feature_id;
                counted[gid] = true;
                metadata[gid].active = true;
                metadata[gid].images.insert(global_image);
                observations.emplace_back(gid, global_image);
                if (first_gid == k_invalid) first_gid = gid;
                else unite(first_gid, gid, false, 0.0);
            }
            if (first_gid == k_invalid) continue;
            const Index root = sets.find(first_gid);
            RootMetadata& root_meta = metadata[root];
            for (const auto [gid, image] : observations) {
                (void)gid;
                root_meta.images.insert(image);
            }
            if (track.is_triangulated()) {
                root_meta.position = track.position;
                root_meta.inlier_weight = track.num_inliers;
                root_meta.has_position = true;
                bbox_min = bbox_min.cwiseMin(track.position);
                bbox_max = bbox_max.cwiseMax(track.position);
            }
            root_meta.active = true;
        }
    }
    const double bbox_diagonal =
        bbox_min.allFinite() && bbox_max.allFinite() ? (bbox_max - bbox_min).norm() : 0.0;
    const double proximity =
        config.merge_proximity_relative_threshold * bbox_diagonal;

    auto accumulate_feature = [&](const Index gid, const Index image) {
        if (counted[gid]) return;
        counted[gid] = true;
        const Index root = sets.find(gid);
        metadata[root].active = true;
        metadata[root].images.insert(image);
    };
    for (const ImagePair& pair : parent.pairs) {
        if (!pair.active || pair.id1 >= global_owner.size() ||
            pair.id2 >= global_owner.size()) {
            continue;
        }
        const int owner1 = global_owner[pair.id1];
        const int owner2 = global_owner[pair.id2];
        if (owner1 < 0 || owner2 < 0 || owner1 == owner2 ||
            !included[static_cast<std::size_t>(owner1)] ||
            !included[static_cast<std::size_t>(owner2)]) {
            continue;
        }
        for (const FeatureMatch& match : pair.matches) {
            if (match.query >= parent.images[pair.id1].features.keypoints.size() ||
                match.train >= parent.images[pair.id2].features.keypoints.size()) {
                continue;
            }
            const Index first = offsets[pair.id1] + match.query;
            const Index second = offsets[pair.id2] + match.train;
            accumulate_feature(first, pair.id1);
            accumulate_feature(second, pair.id2);
            unite(first, second, true, proximity);
        }
    }

    std::map<Index, std::vector<Observation>> groups;
    for (Index image = 0; image < parent.images.size(); ++image) {
        const Index feature_count = static_cast<Index>(
            parent.images[image].features.keypoints.size());
        for (Index feature = 0; feature < feature_count; ++feature) {
            const Index gid = offsets[image] + feature;
            if (!counted[gid]) continue;
            const Index root = sets.find(gid);
            if (!metadata[root].active) continue;
            groups[root].push_back({image, feature});
        }
    }

    parent.tracks.clear();
    parent.image_tracks.clear();
    parent.tracks.reserve(groups.size());
    for (auto& [root, observations] : groups) {
        if (observations.size() < 2) continue;
        std::sort(
            observations.begin(), observations.end(),
            [](const Observation& a, const Observation& b) {
                return a.image_id < b.image_id ||
                       (a.image_id == b.image_id && a.feature_id < b.feature_id);
            });
        Track track;
        track.observations = std::move(observations);
        const RootMetadata& root_meta = metadata[root];
        if (root_meta.has_position) {
            track.position = root_meta.position;
            track.num_inliers = static_cast<std::uint8_t>(std::min<std::size_t>(
                {root_meta.inlier_weight, track.observations.size(), 255u}));
        } else {
            triangulate_track(track, parent);
        }
        parent.tracks.push_back(std::move(track));
    }
    rebuild_track_index(parent);
}

ReconstructionSummary summarize(const Scene& scene) {
    ReconstructionSummary summary;
    summary.registered_views = scene.registered_count();
    summary.failed_views =
        static_cast<unsigned>(scene.images.size()) - summary.registered_views;
    for (const Track& track : scene.tracks) {
        if (track.is_triangulated()) ++summary.landmarks;
    }
    summary.valid = summary.registered_views >= 2 && summary.landmarks > 0;
    return summary;
}

}  // namespace

std::vector<HierarchicalSubscene> split_hierarchical_scene(
    const Scene& scene,
    const ClusterConfig& config) {
    const std::vector<Cluster> clusters = cluster_scene(scene, config);
    std::vector<HierarchicalSubscene> output;
    const unsigned parent_threads = scene.thread_count == 0
        ? std::max(1u, std::thread::hardware_concurrency())
        : scene.thread_count;
    const unsigned child_threads =
        std::max(1u, parent_threads / std::max(1u, static_cast<unsigned>(clusters.size())));
    for (const Cluster& cluster : clusters) {
        if (cluster.size() < config.min_views_per_cluster &&
            clusters.size() > 1) {
            continue;
        }
        output.push_back(extract_subscene(scene, cluster, child_threads));
    }
    return output;
}

unsigned estimate_similarity_transform(
    const std::vector<Vec3>& source,
    const std::vector<Vec3>& destination,
    Similarity3& transform,
    const double inlier_threshold,
    const unsigned max_iterations,
    const std::uint32_t random_seed,
    std::vector<std::size_t>* final_inlier_ids) {
    if (source.size() != destination.size() || source.size() < 3) return 0;
    std::vector<std::size_t> all(source.size());
    std::iota(all.begin(), all.end(), std::size_t{0});
    if (inlier_threshold <= 0.0) {
        if (!fit_similarity(source, destination, all, transform)) return 0u;
        if (final_inlier_ids) *final_inlier_ids = all;
        return static_cast<unsigned>(source.size());
    }

    std::mt19937 generator(random_seed);
    std::uniform_int_distribution<std::size_t> distribution(0, source.size() - 1);
    std::vector<std::size_t> best_inliers;
    double best_error = std::numeric_limits<double>::infinity();
    for (unsigned iteration = 0; iteration < max_iterations; ++iteration) {
        std::size_t first = distribution(generator);
        std::size_t second = distribution(generator);
        std::size_t third = distribution(generator);
        while (second == first) second = distribution(generator);
        while (third == first || third == second) third = distribution(generator);
        const std::vector<std::size_t> sample{first, second, third};
        Similarity3 candidate;
        if (!fit_similarity(source, destination, sample, candidate)) continue;
        std::vector<std::size_t> inliers;
        double error = 0.0;
        for (std::size_t i = 0; i < source.size(); ++i) {
            const double residual =
                (candidate.apply(source[i]) - destination[i]).norm();
            if (residual > inlier_threshold) continue;
            inliers.push_back(i);
            error += residual * residual;
        }
        if (inliers.size() > best_inliers.size() ||
            (inliers.size() == best_inliers.size() && error < best_error)) {
            best_inliers = std::move(inliers);
            best_error = error;
        }
    }
    if (best_inliers.size() < 3 ||
        !fit_similarity(source, destination, best_inliers, transform)) {
        return 0;
    }
    std::vector<std::size_t> final_inliers;
    for (std::size_t i = 0; i < source.size(); ++i) {
        if ((transform.apply(source[i]) - destination[i]).norm() <= inlier_threshold)
            final_inliers.push_back(i);
    }
    if (final_inliers.size() >= 3)
        fit_similarity(source, destination, final_inliers, transform);
    if (final_inlier_ids) *final_inlier_ids = final_inliers;
    return static_cast<unsigned>(final_inliers.size());
}

bool align_and_merge_hierarchical(
    Scene& parent,
    std::vector<HierarchicalSubscene>& subscenes,
    const GlobalAlignmentConfig& config) {
    if (subscenes.empty()) return false;
    std::vector<int> global_owner(parent.images.size(), -1);
    std::vector<Index> global_local(parent.images.size(), k_invalid);
    for (std::size_t scene_id = 0; scene_id < subscenes.size(); ++scene_id) {
        const auto& mapping = subscenes[scene_id].local_to_global;
        for (Index local = 0; local < mapping.size(); ++local) {
            const Index global = mapping[local];
            if (global >= parent.images.size() || global_owner[global] >= 0) {
                core::Logger::instance().error(
                    "hierarchical align: overlapping or invalid image mapping "
                    "at global=",
                    global, " scene=", scene_id);
                return false;
            }
            global_owner[global] = static_cast<int>(scene_id);
            global_local[global] = local;
        }
    }

    std::vector<ScenePair> pairs =
        estimate_scene_pairs(parent, subscenes, global_owner, global_local, config);
    core::Logger::instance().info(
        "hierarchical align: subscenes=", subscenes.size(),
        " relative_pairs=", pairs.size());
    for (const ScenePair& pair : pairs) {
        core::Logger::instance().info(
            "hierarchical align pair: ", pair.first, '-', pair.second,
            " inliers=", pair.inliers, " scale=", pair.relative.scale);
    }
    std::vector<bool> included;
    if (subscenes.size() == 1) {
        included.assign(1, true);
    } else {
        if (pairs.empty()) {
            core::Logger::instance().warning(
                "hierarchical align: no valid Sim(3) pairs between subscenes");
            return false;
        }
        included = largest_connected_component(subscenes.size(), pairs);
    }
    unsigned included_count = 0;
    for (const bool value : included) included_count += value ? 1U : 0U;
    core::Logger::instance().info(
        "hierarchical align: connected_subscenes=", included_count, '/',
        subscenes.size());
    std::vector<Similarity3> transforms;
    if (!estimate_global_transforms(
            subscenes.size(), pairs, included, transforms)) {
        core::Logger::instance().warning(
            "hierarchical align: global transform solve failed");
        return false;
    }
    for (std::size_t i = 0; i < subscenes.size(); ++i) {
        if (included[i]) apply_similarity(subscenes[i].scene, transforms[i]);
    }

    std::vector<PinholeCamera> camera_sums(parent.cameras.size());
    std::vector<unsigned> camera_counts(parent.cameras.size(), 0);
    for (std::size_t scene_id = 0; scene_id < subscenes.size(); ++scene_id) {
        if (!included[scene_id]) continue;
        const HierarchicalSubscene& subscene = subscenes[scene_id];
        for (Index local = 0; local < subscene.local_to_global.size(); ++local) {
            const Index global = subscene.local_to_global[local];
            Image& destination = parent.images[global];
            const Image& source = subscene.scene.images[local];
            destination.registered = source.registered;
            if (source.registered) destination.pose = source.pose;
            if (destination.camera_id >= parent.cameras.size() ||
                source.camera_id >= subscene.scene.cameras.size()) {
                continue;
            }
            const PinholeCamera& camera = subscene.scene.cameras[source.camera_id];
            PinholeCamera& sum = camera_sums[destination.camera_id];
            if (camera_counts[destination.camera_id] == 0) {
                sum = camera;
                sum.fx = sum.fy = sum.cx = sum.cy = 0.0;
                sum.k1 = sum.k2 = sum.p1 = sum.p2 = 0.0;
            }
            sum.fx += camera.fx;
            sum.fy += camera.fy;
            sum.cx += camera.cx;
            sum.cy += camera.cy;
            sum.k1 += camera.k1;
            sum.k2 += camera.k2;
            sum.p1 += camera.p1;
            sum.p2 += camera.p2;
            ++camera_counts[destination.camera_id];
        }
    }
    for (std::size_t camera_id = 0; camera_id < parent.cameras.size(); ++camera_id) {
        if (camera_counts[camera_id] == 0) continue;
        const double inverse = 1.0 / static_cast<double>(camera_counts[camera_id]);
        PinholeCamera& destination = parent.cameras[camera_id];
        const PinholeCamera& sum = camera_sums[camera_id];
        destination.fx = sum.fx * inverse;
        destination.fy = sum.fy * inverse;
        destination.cx = sum.cx * inverse;
        destination.cy = sum.cy * inverse;
        destination.k1 = sum.k1 * inverse;
        destination.k2 = sum.k2 * inverse;
        destination.p1 = sum.p1 * inverse;
        destination.p2 = sum.p2 * inverse;
    }
    merge_tracks(parent, subscenes, included, global_owner, config);
    filter_tracks(parent, 16.F, 0.5F);
    const unsigned registered = parent.registered_count();
    core::Logger::instance().info(
        "hierarchical merge: registered=", registered, '/', parent.images.size(),
        " tracks=", parent.tracks.size());
    return registered >= 2;
}

ReconstructionSummary run_hierarchical_mapping(
    Scene& scene,
    const HierarchicalConfig& config) {
    core::StageScope stage("sfm.hierarchical_mapping");
    std::vector<HierarchicalSubscene> subscenes =
        split_hierarchical_scene(scene, config.cluster);
    if (subscenes.empty()) return {};

    const unsigned available_threads = scene.thread_count == 0
        ? std::max(1u, std::thread::hardware_concurrency())
        : scene.thread_count;
    const unsigned worker_count = std::min<unsigned>(
        available_threads, static_cast<unsigned>(subscenes.size()));
    const unsigned threads_per_subscene =
        std::max(1U, available_threads / std::max(1U, worker_count));
    for (HierarchicalSubscene& subscene : subscenes)
        subscene.scene.thread_count = threads_per_subscene;
    std::vector<bool> succeeded(subscenes.size(), false);
    std::mutex result_mutex;
    core::ProgressReporter progress(
        "reconstruct subscenes", subscenes.size());
    parallel::parallel_for(
        subscenes.size(), worker_count,
        [&](const std::size_t index) {
            Scene& subscene = subscenes[index].scene;
            const parallel::ScopedOpenMpThreads openmp_budget(
                subscene.thread_count);
            build_tracks(subscene, config.cluster.min_pair_weight);
            bool success = star_initialize(subscene, config.star);
            if (success) {
                register_images(subscene, config.resection);
                BundleOptions bundle;
                bundle.optimizer = config.resection.full_ba;
                success = run_bundle_adjustment(subscene, bundle).success;
                filter_tracks(
                    subscene,
                    config.resection.max_reproj_error,
                    config.resection.min_angle_deg,
                    config.resection.mult_depth_near,
                    config.resection.mult_depth_far);
            }
            std::scoped_lock lock(result_mutex);
            succeeded[index] = success;
            progress.advance();
        });
    progress.finish();

    std::vector<HierarchicalSubscene> reconstructed;
    reconstructed.reserve(subscenes.size());
    for (std::size_t i = 0; i < subscenes.size(); ++i) {
        if (succeeded[i]) reconstructed.push_back(std::move(subscenes[i]));
    }
    if (reconstructed.empty()) return {};
    if (!align_and_merge_hierarchical(scene, reconstructed, config.alignment)) {
        std::size_t best = reconstructed.size();
        unsigned best_registered = 0;
        for (std::size_t i = 0; i < reconstructed.size(); ++i) {
            const unsigned registered =
                reconstructed[i].scene.registered_count();
            if (registered > best_registered) {
                best_registered = registered;
                best = i;
            }
        }
        if (best >= reconstructed.size() || best_registered < 2)
            return summarize(scene);
        core::Logger::instance().warning(
            "hierarchical align failed; adopting largest subscene registered=",
            best_registered, '/', scene.images.size());
        std::vector<HierarchicalSubscene> singleton;
        singleton.push_back(std::move(reconstructed[best]));
        if (!align_and_merge_hierarchical(scene, singleton, config.alignment))
            return summarize(scene);
    }

    if (config.final_bundle_adjustment) {
        BundleOptions bundle;
        bundle.optimizer = config.resection.full_ba;
        run_bundle_adjustment(scene, bundle);
        filter_tracks(
            scene,
            config.resection.max_reproj_error,
            config.resection.min_angle_deg,
            config.resection.mult_depth_near,
            config.resection.mult_depth_far);
        triangulate_tracks(
            scene, true,
            config.resection.max_reproj_error,
            config.resection.min_angle_deg);
        filter_tracks(
            scene,
            config.resection.max_reproj_error,
            config.resection.min_angle_deg,
            config.resection.mult_depth_near,
            config.resection.mult_depth_far);
    }
    if (scene.registered_count() < scene.images.size())
        register_images(scene, config.resection);
    return summarize(scene);
}

}  // namespace aetherscan::sfm
