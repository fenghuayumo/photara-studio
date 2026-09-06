#pragma once

#include "mvs/types.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace aetherscan::mvs {

// Load an ASCII or binary-little-endian PLY point cloud. Position is required;
// RGB, normal, weight, view_indices and view_weights are imported when present.
DenseCloud load_dense_ply(const std::filesystem::path& path);
// Load an ASCII or binary-little-endian polygon PLY. Polygons with more than
// three vertices are triangulated as a fan. Position is required; per-vertex
// normals and RGB are imported when all corresponding properties are present.
Mesh load_mesh_ply(const std::filesystem::path& path);
// Includes weight and per-point view_indices/view_weights for graph-cut replay.
// Replay must use the same ordered cameras and world coordinate system.
void save_dense_ply(const DenseCloud& cloud, const std::filesystem::path& path);
void save_mesh_ply(const Mesh& mesh, const std::filesystem::path& path);
void save_mesh_obj(const Mesh& mesh, const std::filesystem::path& path);
void save_subject_bounds(
    const OrientedBoundingBox& bounds, const std::filesystem::path& path);

std::vector<std::uint8_t> encode_mesh(const Mesh& mesh);
Mesh decode_mesh(std::span<const std::uint8_t> bytes);

inline constexpr std::uint32_t k_mesh_chunk_version = 1;

// Optional binary depth-map cache (simple AetherScan .admap container).
void save_depth_map(const DepthMap& depth, const std::filesystem::path& path);
bool load_depth_map(DepthMap& depth, const std::filesystem::path& path);

}  // namespace aetherscan::mvs
