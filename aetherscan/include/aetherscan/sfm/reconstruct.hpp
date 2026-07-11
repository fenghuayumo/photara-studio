#pragma once

#include "aetherscan/sfm/frontend.hpp"
#include "aetherscan/sfm/resection.hpp"
#include "aetherscan/sfm/star_init.hpp"

namespace aetherscan::sfm {

struct ReconstructionConfig {
    FrontEndOptions frontend{};
    StarInitConfig star{};
    ResectionConfig resection{};
};

struct ReconstructionSummary {
    bool valid{false};
    unsigned registered_views{0};
    unsigned landmarks{0};
    unsigned failed_views{0};
};

// Full incremental reconstruction: frontend -> star init -> resection.
ReconstructionSummary reconstruct(
    Scene& scene_out,
    const std::vector<std::filesystem::path>& image_paths,
    const ReconstructionConfig& config = {});

// Run mapping only on an already-prepared scene (features + verified pairs + tracks).
ReconstructionSummary run_incremental_mapping(
    Scene& scene,
    const StarInitConfig& star = {},
    const ResectionConfig& resection = {});

}  // namespace aetherscan::sfm
