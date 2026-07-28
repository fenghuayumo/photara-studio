#pragma once

#include "mvs/types.hpp"

#include <filesystem>

namespace aetherscan::mvs {

// Load an ASCII or binary-little-endian PLY point cloud. Position is required;
// RGB, normal, weight, view_indices and view_weights are imported when present.
DenseCloud load_dense_ply(const std::filesystem::path& path);
// Load an ASCII or binary-little-endian polygon PLY. Polygons with more than
// three vertices are triangulated as a fan. Position is required; per-vertex
// normals and RGB are imported when all corresponding properties are present.
Mesh load_mesh_ply(const std::filesystem::path& path);
void save_dense_ply(const DenseCloud& cloud, const std::filesystem::path& path);
void save_mesh_ply(const Mesh& mesh, const std::filesystem::path& path);
void save_mesh_obj(const Mesh& mesh, const std::filesystem::path& path);
void save_roi(const OrientedBoundingBox& roi, const std::filesystem::path& path);

// Optional binary depth-map cache (simple AetherScan .admap container).
void save_depth_map(const DepthMap& depth, const std::filesystem::path& path);
bool load_depth_map(DepthMap& depth, const std::filesystem::path& path);

}  // namespace aetherscan::mvs
