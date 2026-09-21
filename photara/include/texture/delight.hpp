#pragma once

#include "texture/options.hpp"
#include "texture/types.hpp"

#include <filesystem>
#include <vector>

namespace photara::texture {

// Run Intrinsic delighter via ONNX Runtime (C++). Replaces each view's rgb
// with delighted albedo (still stored as linear RGB).
void delight_texture_views(
    std::vector<TextureViewImage>& views, const TextureOptions& options);

[[nodiscard]] bool delighter_available(const TextureOptions& options);

}  // namespace photara::texture
