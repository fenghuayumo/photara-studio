#pragma once

#include "mvs/types.hpp"

#include <cstdint>
#include <filesystem>
#include <span>

namespace aetherscan::io {

// Triangle-mesh interchange formats. `mvs::Mesh` is the shared reconstruction
// mesh type (MVS fusion and Gaussian surface extraction both produce it).

// glTF 2.0 binary (.glb). Coordinates match PLY/OBJ (no axis conversion).
// `uvs` uses the glTF convention (origin bottom-left, V up) and must be empty
// or one entry per vertex. A non-empty `albedo_png` is embedded as an unlit
// base-color texture; it is ignored when UVs are missing.
void save_mesh_glb(
    const mvs::Mesh& mesh, const std::filesystem::path& path,
    std::span<const mvs::Vec2f> uvs = {},
    std::span<const std::uint8_t> albedo_png = {});

}  // namespace aetherscan::io
