#pragma once

#include "project/archive.hpp"
#include "sfm/asfm.hpp"
#include "sfm/scene.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace photara::project {

inline constexpr std::uint32_t k_settings_version = 3;
inline constexpr std::uint32_t k_settings_min_reader = 1;

struct Settings {
    std::string name;
    std::filesystem::path image_directory;
    // Optional external camera/dataset input used by direct splat training.
    // These fields are additive so older .ascan settings remain readable.
    std::filesystem::path dataset_source;
    std::string dataset_format{"auto"};
    std::filesystem::path dataset_initial_cloud;
    // Final trained Gaussian representation: ply, sog, or spz.
    std::string splat_output_format{"ply"};
    // Optional external trained Gaussian file used by the editor preview.
    std::filesystem::path splat_model_source;
    // Optional video capture. image_directory may itself be a video file; these
    // fields are the extraction knobs Align Photos / --images apply first.
    std::filesystem::path video_frames_dir;
    float video_fps = 2.0F;
    int video_sharp_window = 3;
    int video_max_frames = 0;
    int video_quality = 95;
    float video_scale = 1.0F;
    int video_rotate = 0;
    int camera_model = 0;  // 0 pinhole, 1 OpenCV fisheye, 2 automatic
    int sfm_mode = 0;  // 0 global, 1 incremental, 2 hierarchical
    bool reuse_cache = false;
    unsigned max_features = 27'000;
    bool scene_mode = false;
    int iterations = 30'000;
    int densification_cap = 1'000'000;
    int sh_degree = 3;
    int preview_interval = 50;
    // 0=adc_igs, 1=adc_plus, 2=dense_adaptive.
    int strategy = 0;
    int max_resolution = 1'920;
    bool progressive_resolution = true;
    bool use_mask = false;
    // SAM 3 masks generated before alignment. The checkpoint itself is not
    // stored in the project; only the prompt and whether generation is on.
    bool sam_masks = false;
    std::string sam_model;
    std::string sam_text;
    std::string sam_negative_text;
    bool sam_keep_prompted = true;
    bool sam_video = true;
    int sam_max_size = 0;
    bool build_mesh = false;
    // 0 = extract from geometry-supervised 3DGS, 1 = photogrammetry (MVS).
    int mesh_source = 0;
    int mesh_method = 0;
    // Texture bake after mesh: 0 Fast, 1 Standard, 2 High.
    int texture_quality = 1;
    int atlas_resolution = 2048;
    bool texture_delight = false;
    bool texture_optimize = true;
    float depth_normal_weight = 0.05F;
    float multi_view_geo_weight = 0.02F;
    float multi_view_ncc_weight = 0.6F;
    int geometry_from_iter = 3'000;
    bool normal_field = false;
    // Training-time colour correction for auto-exposure / auto-white-balance
    // drift: ppisp_layout 0 = off, nonzero = PPISP on. Legacy value 2
    // (exposure + white balance) is treated as on. The bilateral grid adds
    // spatially varying correction.
    int ppisp_layout = 0;
    bool bilateral_grid = false;
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

}  // namespace photara::project
