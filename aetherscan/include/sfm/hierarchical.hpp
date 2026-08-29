#pragma once

#include "sfm/resection.hpp"
#include "sfm/scene.hpp"
#include "sfm/star_init.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aetherscan::sfm {

struct ReconstructionSummary;

struct ClusterConfig {
    unsigned max_views_per_cluster{200};
    unsigned min_views_per_cluster{10};
    unsigned max_over_capacity{20};
    unsigned min_common_tracks{25};
    float min_pair_weight{3.F};
    bool refine_weak_edges{true};
    float edge_weight_percentile{0.9F};
};

struct GlobalAlignmentConfig {
    float min_pair_weight{3.F};
    unsigned min_common_tracks{25};
    bool merge_track_inliers_only{true};
    double ransac_relative_threshold{0.001};
    double minimum_inlier_ratio{0.3};
    double merge_proximity_relative_threshold{0.02};
    unsigned ransac_iterations{2048};
    std::uint32_t random_seed{0xA37E5CA1u};
};

struct HierarchicalConfig {
    ClusterConfig cluster{};
    GlobalAlignmentConfig alignment{};
    StarInitConfig star{};
    ResectionConfig resection{};
    bool final_bundle_adjustment{true};
};

struct Similarity3 {
    double scale{1.0};
    Mat3 R{Mat3::Identity()};
    Vec3 t{Vec3::Zero()};

    [[nodiscard]] Vec3 apply(const Vec3& point) const {
        return scale * (R * point) + t;
    }
};

struct HierarchicalSubscene {
    Scene scene;
    // local_to_global[local image ID] = image ID in the parent scene.
    std::vector<Index> local_to_global;
};

// Deterministic weighted-covisibility clustering and copy-based sub-scene extraction.
std::vector<HierarchicalSubscene> split_hierarchical_scene(
    const Scene& scene,
    const ClusterConfig& config = {});

// Deterministic 3-point Sim(3) RANSAC followed by an all-inlier Umeyama fit.
unsigned estimate_similarity_transform(
    const std::vector<Vec3>& source,
    const std::vector<Vec3>& destination,
    Similarity3& transform,
    double inlier_threshold = 0.0,
    unsigned max_iterations = 2048,
    std::uint32_t random_seed = 0xA37E5CA1u,
    std::vector<std::size_t>* final_inlier_ids = nullptr);

// Align reconstructed sub-scenes, transform poses/points and merge guarded tracks.
bool align_and_merge_hierarchical(
    Scene& parent,
    std::vector<HierarchicalSubscene>& subscenes,
    const GlobalAlignmentConfig& config = {});

// Mapping-only entry point for a scene containing features and verified pairs.
ReconstructionSummary run_hierarchical_mapping(
    Scene& scene,
    const HierarchicalConfig& config = {});

}  // namespace aetherscan::sfm
