#pragma once

#include "mvs/types.hpp"

#include <filesystem>

namespace aetherscan::mvs {

void save_dense_ply(const DenseCloud& cloud, const std::filesystem::path& path);
void save_mesh_ply(const Mesh& mesh, const std::filesystem::path& path);
void save_mesh_obj(const Mesh& mesh, const std::filesystem::path& path);

// Optional binary depth-map cache (simple AetherScan .admap container).
void save_depth_map(const DepthMap& depth, const std::filesystem::path& path);
bool load_depth_map(DepthMap& depth, const std::filesystem::path& path);

}  // namespace aetherscan::mvs
