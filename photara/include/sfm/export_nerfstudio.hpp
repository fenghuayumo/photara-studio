#pragma once

#include "sfm/scene.hpp"

#include <filesystem>

namespace photara::sfm {

// Nerfstudio / Blender `transforms.json`. Pose is camera-to-world in OpenGL
// convention (Y and Z flipped from OpenCV). `ply_file_path` is stored relative
// to the JSON when possible; pass empty to omit it.
void save_nerfstudio_transforms(
    const Scene& scene,
    const std::filesystem::path& json_path,
    const std::filesystem::path& image_path_base = {},
    const std::filesystem::path& ply_path = {});

}  // namespace photara::sfm
