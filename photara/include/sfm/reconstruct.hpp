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
    hierarchical = 1,
    global = 2,
};

struct ReconstructionConfig {
    ReconstructionMode mode{ReconstructionMode::global};
    FrontEndOptions frontend{};
    StarInitConfig star{};
    ResectionConfig resection{};
    HierarchicalConfig hierarchical{};
    // Final product audit: a registered pose must retain this many finite,
    // oriented inlier observations within the reprojection threshold (a full
    // sphere camera also observes points behind its optical axis).
    unsigned minimum_final_observations_per_image{30};
    double maximum_final_reprojection_error_pixels{2.0};
    GlobalRotationOptions global_rotation{};
    GlobalPositioningOptions global_positioning{};
    // Optional final repair of weak registered poses; other poses/calibration
    // stay fixed. Independent held-out improvement is required for admission.
    bool refine_weak_cameras{false};
};

struct ReconstructionSummary {
    // A usable reconstruction; neither a precision guarantee nor solver convergence.
    bool valid{false};
    unsigned registered_views{0};
    unsigned landmarks{0};
    unsigned failed_views{0};
    unsigned alignment_reliable_views{0};
    unsigned alignment_unreliable_views{0};
    // Graph-support screening is conservative and does not prove position accuracy.
    unsigned position_audited_views{0};
    unsigned position_unverified_views{0};
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

// Screen translation support using strong, nonplanar, parallax-bearing pairs.
// Does not use optimized landmark coordinates or an external reference.
AlignmentObservability analyze_position_observability(const Scene& scene);
// Persist uncertainty separately from registration; nomination never changes poses.
unsigned audit_position_support(Scene& scene);

// A well-constrained main component must not certify a leaf/block from its
// jointly optimized depths alone. Clear unsupported registrations and their
// track observations; do nothing when no established main component exists.
unsigned prune_unobservable_registrations(Scene& scene);

// Split untrusted camera/lens states for late BA. Millimetre focal lengths
// only scale a pixel-space initialization; existing camera groups stay distinct.
unsigned split_intrinsics_by_camera_identity(Scene& scene);

// Remove registration flags from cameras that no longer have enough valid
// landmark support after final filtering/BA. Returns the number invalidated.
unsigned prune_unsupported_registrations(
    Scene& scene, unsigned minimum_observations = 30,
    double maximum_reprojection_error_pixels = 2.0);

// Final native track refresh and fixed-calibration joint pose/point BA.
// Uses all live associations, including observations excluded by earlier BA.
// Registration stays fixed; a rejected/failed solve restores the input geometry.
bool refine_alignment_geometry(Scene& scene);

// Mapping and final audits for an already extracted/verified frontend scene.
ReconstructionSummary reconstruct_from_frontend(
    Scene& scene, const ReconstructionConfig& config = {});

// Full reconstruction using the mode selected in ReconstructionConfig.
ReconstructionSummary reconstruct(
    Scene& scene_out,
    const std::vector<std::filesystem::path>& image_paths,
    const ReconstructionConfig& config = {});

// Global rotation averaging -> fixed-rotation positioning -> triangulation -> BA.
ReconstructionSummary run_global_mapping(
    Scene& scene,
    const GlobalRotationOptions& rotation = {},
    const GlobalPositioningOptions& positioning = {},
    const ResectionConfig& fallback_resection = {});

}  // namespace photara::sfm
