#pragma once

#include "mvs/types.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace aetherscan::texture {

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

    // Atlas in linear RGB, row-major top-left, 3 channels, values in [0, 1].
    std::uint32_t atlas_width{0};
    std::uint32_t atlas_height{0};
    std::vector<float> atlas_rgb;
    std::vector<float> atlas_confidence;
    std::vector<float> atlas_valid;  // 0/1 coverage mask
    bool used_ray_query{false};
    bool delighted{false};
};

struct TextureViewImage {
    std::uint32_t width{0};
    std::uint32_t height{0};
    // Linear RGB, top-left origin, 3 * width * height floats in [0, 1].
    std::vector<float> rgb;
    // Optional foreground mask in [0, 1]; empty = all foreground.
    std::vector<float> mask;
};

}  // namespace aetherscan::texture
