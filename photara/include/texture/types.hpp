#pragma once

#include "mvs/types.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace photara::texture {

struct TexturedMesh {
    // Unwrapped geometry (UVAtlas may duplicate vertices).
    std::vector<mvs::Vec3f> positions;
    std::vector<mvs::Vec3f> normals;
    std::vector<mvs::Vec2f> uvs;
    std::vector<std::uint32_t> indices;  // triangle list, 3 * face_count
    std::vector<std::uint32_t> vertex_remap;
    std::vector<std::uint32_t> face_chart_ids;
    std::uint32_t chart_count{0};
    float max_stretch{0.F};

    // Atlas RGB, row-major top-left, 3 channels, values in [0, 1].
    // linear_rgb is false for display-referred sRGB (the default bake) and
    // true when the atlas was blended in scene-linear light.
    std::uint32_t atlas_width{0};
    std::uint32_t atlas_height{0};
    std::vector<float> atlas_rgb;
    std::vector<float> atlas_confidence;
    std::vector<float> atlas_valid;  // 0/1 coverage mask
    bool used_ray_query{false};
    bool linear_rgb{false};
    bool delighted{false};
    bool optimized{false};
    std::vector<float> optimization_loss;
    std::vector<float> seam_loss;
    double optimization_precompute_seconds{0.0};
    double optimization_seconds{0.0};
};

struct TextureViewImage {
    std::uint32_t width{0};
    std::uint32_t height{0};
    // RGB in the bake color space (sRGB or linear), top-left origin,
    // 3 * width * height floats in [0, 1].
    std::vector<float> rgb;
    // Optional foreground mask in [0, 1]; empty = all foreground.
    std::vector<float> mask;
};

}  // namespace photara::texture
