#pragma once

#include "mvs/types.hpp"
#include "texture/options.hpp"
#include "texture/types.hpp"

#include <filesystem>

namespace photara::texture {

// UV unwrap + multi-view projective texture bake via photara_drender.
// A non-manifold mesh is repaired with photara_mesh before unwrap when that
// backend is available. When options.delight is true, photographs are
// Intrinsic-decomposed first.
[[nodiscard]] TexturedMesh bake_mesh_texture(
    const mvs::MvsScene& scene, const TextureOptions& options = {});

// Convenience: bake and write OBJ+MTL+PNG next to `output_stem`.
void bake_and_export(
    const mvs::MvsScene& scene, const std::filesystem::path& output_stem,
    const TextureOptions& options = {});

}  // namespace photara::texture
