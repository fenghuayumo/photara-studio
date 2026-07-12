#pragma once

#include "sfm/frontend.hpp"
#include "sfm/global_positioning.hpp"
#include "sfm/global_rotation.hpp"
#include "sfm/hierarchical.hpp"
#include "sfm/resection.hpp"
#include "sfm/star_init.hpp"

namespace aetherscan::sfm {

enum class ReconstructionMode {
    incremental,
    hierarchical,
    global,
};

struct ReconstructionConfig {
    ReconstructionMode mode{ReconstructionMode::incremental};
    FrontEndOptions frontend{};
    StarInitConfig star{};
    ResectionConfig resection{};
    HierarchicalConfig hierarchical{};
    GlobalRotationOptions global_rotation{};
    GlobalPositioningOptions global_positioning{};
};

struct ReconstructionSummary {
    bool valid{false};
    unsigned registered_views{0};
    unsigned landmarks{0};
    unsigned failed_views{0};
};

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

}  // namespace aetherscan::sfm
