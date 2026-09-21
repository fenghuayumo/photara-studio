#pragma once

#include "sfm/frontend.hpp"
#include "sfm/global_positioning.hpp"
#include "sfm/global_rotation.hpp"
#include "sfm/hierarchical.hpp"
#include "sfm/resection.hpp"
#include "sfm/star_init.hpp"

#include <cstdint>

namespace photara::sfm {

enum class ReconstructionMode {
    incremental,
    hierarchical,
    global,
};

struct ReconstructionConfig {
    ReconstructionMode mode{ReconstructionMode::global};
    FrontEndOptions frontend{};
    StarInitConfig star{};
    ResectionConfig resection{};
    HierarchicalConfig hierarchical{};
    // If a large incremental reconstruction stalls with missing views, retry
    // from independent submaps and merge them. Small/successful captures stay
    // on the ordinary incremental path.
    bool incremental_hierarchical_rescue{true};
    unsigned incremental_hierarchical_rescue_min_missing{8};
    double incremental_hierarchical_rescue_min_missing_ratio{0.02};
    // Final product audit: a registered pose must retain this many finite,
    // positive-depth inlier observations within the reprojection threshold.
    unsigned minimum_final_observations_per_image{30};
    double maximum_final_reprojection_error_pixels{2.0};
    GlobalRotationOptions global_rotation{};
    GlobalPositioningOptions global_positioning{};
};

struct ReconstructionSummary {
    bool valid{false};
    unsigned registered_views{0};
    unsigned landmarks{0};
    unsigned failed_views{0};
    unsigned alignment_reliable_views{0};
    unsigned alignment_unreliable_views{0};
    std::uint64_t reprojection_observations{0};
    double mean_reprojection_error_pixels{0.0};
    double rms_reprojection_error_pixels{0.0};
};

struct AlignmentObservability {
    std::vector<std::uint8_t> reliable;
    unsigned reliable_views{0};
    unsigned unreliable_views{0};
    unsigned constraint_edges{0};
    unsigned bridge_edges{0};
};

AlignmentObservability analyze_alignment_observability(const Scene& scene);

// Split untrusted camera/lens states for late BA. Millimetre focal lengths
// only scale a pixel-space initialization; existing camera groups stay distinct.
unsigned split_intrinsics_by_camera_identity(Scene& scene);

// Remove registration flags from cameras that no longer have enough valid
// landmark support after final filtering/BA. Returns the number invalidated.
unsigned prune_unsupported_registrations(
    Scene& scene, unsigned minimum_observations = 30,
    double maximum_reprojection_error_pixels = 2.0);

// Full reconstruction using the mode selected in ReconstructionConfig.
ReconstructionSummary reconstruct(
    Scene& scene_out,
    const std::vector<std::filesystem::path>& image_paths,
    const ReconstructionConfig& config = {});

// Run mapping only on an already-prepared scene (features + verified pairs + tracks).
ReconstructionSummary run_incremental_mapping(
    Scene& scene,
    const StarInitConfig& star = {},
    const ResectionConfig& resection = {});

// Global rotation averaging -> fixed-rotation positioning -> triangulation -> BA.
ReconstructionSummary run_global_mapping(
    Scene& scene,
    const GlobalRotationOptions& rotation = {},
    const GlobalPositioningOptions& positioning = {},
    const ResectionConfig& fallback_resection = {});

}  // namespace photara::sfm
