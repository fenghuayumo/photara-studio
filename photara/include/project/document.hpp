#pragma once

#include "splat/loss_settings.hpp"

#include "project/archive.hpp"
#include "project/camera_effect.hpp"
#include "sfm/asfm.hpp"
#include "sfm/scene.hpp"

#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace photara::project {

inline constexpr std::uint32_t k_settings_version = 29;
inline constexpr std::uint32_t k_settings_min_reader = 1;

struct GaussianSource {
    std::filesystem::path path;
    std::string name;
    bool visible{true};
    float offset[3]{};
    float scale{1.F};
    float rotation_degrees[3]{};
};

struct CameraKeyframe {
    float time{};
    float target[3]{};
    float yaw{}, pitch{}, distance{6.F}, fov_degrees{50.F}, ortho_height{4.F}, fisheye_k1{};
    int projection{};
    float roll{};
    int interpolation{}; // Outgoing segment: smooth, linear, hold.
    std::uint32_t thumbnail_width{}, thumbnail_height{};
    std::vector<std::uint8_t> thumbnail_rgb;
};

struct Settings : photara::splat::TrainingLossSettings {
    std::string name;
    std::vector<CameraKeyframe> camera_keyframes;
    CameraEffectTrack camera_effect;
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
    std::vector<GaussianSource> gaussian_models;
    // Optional video capture. image_directory may itself be a video file; these
    // fields are the extraction knobs Align Photos / --images apply first.
    std::filesystem::path video_frames_dir;
    float video_fps = 2.0F;
    int video_sharp_window = 3;
    int video_max_frames = 0;
    int video_quality = 95;
    float video_scale = 1.0F;
    int video_rotate = 0;
    // Capture ranges in source time, `start:end;start:end` with `-` as the
    // open end. Empty keeps the whole video.
    std::string video_segments;
    int camera_model = 0;  // 0 pinhole, 1 OpenCV fisheye, 2 automatic
    int alignment_quality = 1; // 0 fast, 1 medium, 2 high.
    int alignment_capture_type = 0; // 0 unordered photos, 1 ordered/video frames.
    int camera_sharing = 0; // 0 folder, 1 all images, 2 each image.
    int matching_strategy = 0; // 0 automatic, 1 sequential, 2 exhaustive.
    double initial_focal_pixels = 0.0; // 0 uses EXIF / automatic initialization.
    bool trust_initial_focal = false;
    std::size_t sequential_overlap = 6;
    bool retrieval_augmentation = true;
    std::string feature_extractor{"siftgpu"};
    std::string feature_matcher{"gpu_mutual_ratio"};
    std::filesystem::path extractor_model;
    std::filesystem::path lightglue_model;
    int sfm_mode = 0;  // 0 global, 2 hierarchical (1 is a legacy project value)
    bool reuse_cache = false;
    bool refine_weak_cameras = false;
    unsigned max_features = 27'000;
    bool scene_mode = false;
    bool focus_region = false;
    int iterations = 30'000;
    int densification_cap = 3'000'000;
    int training_mode = 1; // 0 Fast, 1 Standard, 2 High quality.
    bool speedy_pruning = false;
    int sh_degree = 3;
    int preview_interval = 50;
    // 0=adc_igs, 1=adc_plus, 2=dense_adaptive.
    int strategy = 0;
    int render_method = 0;
    int filter_mode = 0; // 0 Classic, 1 Mip2D, 2 Mip3D.
    int max_resolution = 1'920;
    bool progressive_resolution = true;
    bool use_mask = false;
    // 0 = masked (invalid rays are missing observations),
    // 1 = transparent (the mask is the target output alpha).
    int mask_mode = 0;
    // SAM 3 masks generated before alignment. The checkpoint itself is not
    // stored in the project; only the prompt and whether generation is on.
    bool sam_masks = false;
    // Opt-in: coarse SAM masks may discard useful alignment features.
    bool sam_use_mask_for_sfm = false;
    // 0 = SAM 3 text path, 1 = text detector + visual segmentation.
    int sam_engine = 0;
    std::string sam_model;
    std::string sam_detector_model;
    std::string sam_text;
    std::string sam_negative_text;
    bool sam_keep_prompted = true;
    bool sam_video = true;
    bool sam_sfm_guided = false;
    int sam_max_size = 0;
    float sam_threshold = 0.5F;
    float sam_detector_threshold = 0.35F;
    float sam_guided_threshold = 0.2F;
    float sam_nms = 0.1F;
    float sam_min_area = 0.001F;
    float sam_max_area = 0.65F;
    int sam_close_kernel = 5;
    bool sam_fill_holes = true;
    bool build_mesh = false;
    // 0 = extract from geometry-supervised 3DGS, 1 = photogrammetry (MVS).
    int mesh_source = 0;
    int mesh_method = 0;
    int mesh_quality = 1; // 0 Fast, 1 Standard, 2 High.
    int mesh_postprocess_quality = 1;
    bool mesh_simplify = false;
    std::uint64_t mesh_target_faces = 1'000'000;
    bool mesh_remesh = false;
    std::uint64_t mesh_remesh_target_faces = 1'000'000;
    int mesh_remesh_smooth_iterations = 1;
    // Texture bake after mesh: 0 Fast, 1 Standard, 2 High.
    int texture_quality = 1;
    int atlas_resolution = 2048;
    int uv_parallel_partitions = 8;
    float uv_max_stretch = 1.F / 3.F;
    bool texture_delight = false;
    bool texture_optimize = true;
    int texture_photometric_steps = 3000;
    int texture_seam_steps = 60;
    // false = blend and export in sRGB (matches photographs). true = linear.
    bool texture_blend_linear = false;
    int geometry_from_iter = 7'000;
    bool normal_field = false;
    bool ai_priors = false;
    int ai_prior_source = 0;
    std::filesystem::path ai_prior_model;
    std::filesystem::path ai_prior_directory;
    float ai_depth_weight = 0.01F;
    float ai_normal_weight = 0.01F;
    int ai_prior_tokens = 1200;
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
void write_gaussian_sources(Archive& archive,
    const std::vector<std::vector<std::uint8_t>>& sources);
std::vector<std::vector<std::uint8_t>> read_gaussian_sources(const Archive& archive);

void write_sfm(
    Archive& archive,
    const sfm::Scene& scene,
    const std::filesystem::path& project_file);
// Cheap identity of a working ASFM generation and its path interpretation.
// Store alongside sfm only after successfully importing that generation.
std::vector<std::uint8_t> sfm_source_signature(
    const std::filesystem::path& working_sfm,
    const std::filesystem::path& image_directory,
    const std::filesystem::path& project_file);
std::optional<sfm::Scene> read_sfm(const Archive& archive);

// Replace the SfM stage and drop later stages that depended on it.
void replace_sfm_stage(
    Archive& archive,
    const sfm::Scene& scene,
    const Settings& settings,
    const std::filesystem::path& project_file);

}  // namespace photara::project
