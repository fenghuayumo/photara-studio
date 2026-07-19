#pragma once

#include "texture/types.hpp"

#include <filesystem>

namespace aetherscan::texture {

// Writes `<stem>.obj`, `<stem>.mtl`, and `<stem>_albedo.png`.
void save_textured_obj(
    const TexturedMesh& mesh, const std::filesystem::path& stem);

void save_atlas_png(
    const TexturedMesh& mesh, const std::filesystem::path& path);

}  // namespace aetherscan::texture
