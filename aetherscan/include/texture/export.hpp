#pragma once

#include "texture/types.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace aetherscan::texture {

inline constexpr std::uint32_t k_texture_chunk_version = 1;

[[nodiscard]] inline std::filesystem::path textured_obj_path(
    const std::filesystem::path& stem) {
    auto path = stem;
    path += ".obj";
    return path;
}

[[nodiscard]] inline std::filesystem::path textured_mtl_path(
    const std::filesystem::path& stem) {
    auto path = stem;
    path += ".mtl";
    return path;
}

[[nodiscard]] inline std::filesystem::path textured_albedo_path(
    const std::filesystem::path& stem) {
    auto path = stem;
    path += "_albedo.png";
    return path;
}

[[nodiscard]] bool textured_obj_exists(const std::filesystem::path& stem);

// Writes `<stem>.obj`, `<stem>.mtl`, and `<stem>_albedo.png`.
void save_textured_obj(
    const TexturedMesh& mesh, const std::filesystem::path& stem);

void save_atlas_png(
    const TexturedMesh& mesh, const std::filesystem::path& path);

// Copy a baked OBJ/MTL/PNG trio to a new stem, rewriting mtllib / map_Kd.
void copy_textured_obj(
    const std::filesystem::path& source_stem,
    const std::filesystem::path& destination_stem);

// Pack the working OBJ/MTL/PNG into the .ascan texture chunk.
[[nodiscard]] std::vector<std::uint8_t> encode_textured_obj(
    const std::filesystem::path& stem);
void decode_textured_obj(
    std::span<const std::uint8_t> bytes, const std::filesystem::path& stem);

}  // namespace aetherscan::texture
