#pragma once

#include "aetherscan/sfm/scene.hpp"

#include <filesystem>

namespace aetherscan::sfm {

struct ExportMvsOptions {
    // If empty, image paths are written as absolute paths with '/' separators.
    // If set, image paths are written relative to this directory when possible.
    std::filesystem::path image_path_base;
    // Sample RGB from source images at track keypoints and write verticesColor
    // (OpenMVS BGR). Viewer shows white points when this array is empty.
    bool sample_colors{true};
};

// Writes an OpenMVS Interface (.mvs / MVSI) scene for Viewer / densify import.
// Only registered views are exported. Landmarks need >=2 observations among
// those views. Pose convention matches OpenMVS: R world->camera, C = center.
void export_openmvs_interface(
    const Scene& scene,
    const std::filesystem::path& path,
    const ExportMvsOptions& options = {});

}  // namespace aetherscan::sfm
