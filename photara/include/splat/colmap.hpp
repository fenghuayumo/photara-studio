#pragma once

#include "mvs/types.hpp"

#include <filesystem>

namespace photara::splat {

struct ColmapLoadResult {
    mvs::MvsScene scene;
    std::filesystem::path model_directory;
    bool binary{};
};

// Load COLMAP cameras/images/points3D from either binary or text format.
// model_directory may point at the dataset root, sparse/, or sparse/0.
// image_directory is the directory containing the image names stored by COLMAP.
ColmapLoadResult load_colmap_scene(
    const std::filesystem::path& model_directory,
    const std::filesystem::path& image_directory);

}  // namespace photara::splat
