#pragma once

#include "aetherscan/ba/optimizer.hpp"
#include "aetherscan/sfm/triangulation.hpp"

#include <array>
#include <optional>
#include <vector>

namespace aetherscan::sfm {

// p_target = R_target_source * p_source + t_target_source.
struct RelativePoseEdge {
    Id source{invalid_id};
    Id target{invalid_id};
    std::array<double, 9> rotation{};
    std::array<double, 3> translation_direction{};
    double weight{1.0};
    std::size_t inlier_count{};
};

struct PnPOptions {
    std::size_t minimum_correspondences{20};
    std::size_t minimum_inliers{15};
    std::size_t maximum_iterations{10000};
    double maximum_reprojection_error{4.0};
    double confidence{0.999};
};

struct PnPResult {
    bool valid{false};
    std::size_t correspondence_count{};
    std::size_t inlier_count{};
    double mean_reprojection_error{};
};

PnPResult register_view_pnp(Scene& scene, Id view_id, const PnPOptions& options = {});

struct AveragingOptions {
    std::size_t maximum_iterations{20};
    double huber_threshold{0.05};
    double convergence_tolerance{1e-8};
};

struct AveragingResult {
    bool valid{false};
    std::vector<ba::Pose> poses;
    double mean_residual{};
    std::size_t iterations{};
};

AveragingResult average_global_poses(
    std::size_t view_count,
    const std::vector<RelativePoseEdge>& edges,
    Id anchor_view = 0,
    const AveragingOptions& options = {});

struct SceneBundleOptions {
    ba::OptimizerOptions optimizer;
    std::size_t maximum_local_views{12};
};

ba::OptimizerSummary bundle_adjust_scene(
    Scene& scene,
    const std::vector<Id>& view_ids,
    const std::vector<Id>& landmark_ids,
    const ba::OptimizerOptions& options = {});

struct IncrementalMapperOptions {
    PnPOptions pnp;
    TriangulationOptions triangulation;
    SceneBundleOptions bundle;
    std::size_t local_ba_interval{1};
};

struct MapperSummary {
    bool valid{false};
    std::size_t registered_views{};
    std::size_t landmarks{};
    std::size_t failed_views{};
    std::vector<ba::OptimizerSummary> bundle_summaries;
};

MapperSummary run_incremental_mapping(
    Scene& scene,
    const RelativePoseEdge& seed,
    const IncrementalMapperOptions& options = {});

struct GlobalMapperOptions {
    AveragingOptions averaging;
    TriangulationOptions triangulation;
    ba::OptimizerOptions bundle;
};

MapperSummary run_global_mapping(
    Scene& scene,
    const std::vector<RelativePoseEdge>& edges,
    const GlobalMapperOptions& options = {});

}  // namespace aetherscan::sfm
