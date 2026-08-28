#pragma once

#include "project/archive.hpp"
#include "sfm/asfm.hpp"
#include "sfm/scene.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace aetherscan::project {

inline constexpr std::uint32_t k_settings_version = 1;
inline constexpr std::uint32_t k_settings_min_reader = 1;

struct Settings {
    std::string name;
    std::filesystem::path image_directory;
    int sfm_mode = 0;  // 0 global, 1 incremental, 2 hierarchical
    bool reuse_cache = true;
    unsigned max_features = 27'000;
    bool scene_mode = false;
    int iterations = 30'000;
    int preview_interval = 50;
    int strategy = 1;
    int max_resolution = 1'920;
    bool progressive_resolution = true;
    bool use_mask = false;
    bool build_mesh = false;
    int mesh_method = 0;
    float depth_normal_weight = 0.05F;
    float multi_view_geo_weight = 0.02F;
    float multi_view_ncc_weight = 0.6F;
    int geometry_from_iter = 3'000;
    bool normal_field = false;
};

std::vector<std::uint8_t> encode_settings(
    const Settings& settings,
    const std::filesystem::path& project_file);
Settings decode_settings(
    const std::vector<std::uint8_t>& bytes,
    const std::filesystem::path& project_file);

void write_settings(
    Archive& archive,
    const Settings& settings,
    const std::filesystem::path& project_file);
Settings read_settings(const Archive& archive);

void write_sfm(
    Archive& archive,
    const sfm::Scene& scene,
    const std::filesystem::path& project_file);
std::optional<sfm::Scene> read_sfm(const Archive& archive);

// Replace the SfM stage and drop later stages that depended on it.
void replace_sfm_stage(
    Archive& archive,
    const sfm::Scene& scene,
    const Settings& settings,
    const std::filesystem::path& project_file);

}  // namespace aetherscan::project
