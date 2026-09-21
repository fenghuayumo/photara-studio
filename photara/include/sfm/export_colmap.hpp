#pragma once

#include "sfm/scene.hpp"

#include <filesystem>

namespace photara::sfm {

// COLMAP text model: cameras.txt, images.txt, points3D.txt in `directory`.
// Image names are relative to `image_path_base` when they do not escape it.
// `write_points` fills points3D.txt; otherwise the file is empty besides the
// header so COLMAP still accepts the model.
void save_colmap_text(
    const Scene& scene,
    const std::filesystem::path& directory,
    const std::filesystem::path& image_path_base = {},
    bool write_points = true);

}  // namespace photara::sfm
