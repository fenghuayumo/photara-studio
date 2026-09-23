#include "sfm/reconstruct.hpp"
#include "sfm/bundle.hpp"
#include "sfm/appearance.hpp"
#include "sfm/asfm.hpp"
#include "sfm/preview.hpp"
#include "sfm/align_live.hpp"
#include "sfm/export_mvs.hpp"
#include "sfm/export_colmap.hpp"
#include "project/archive.hpp"
#include "project/document.hpp"
#include "mvs/densify.hpp"
#include "mvs/export.hpp"
#include "mvs/internal.hpp"
#include "core/logging.hpp"
#include "core/version.hpp"
#include "io/image.hpp"
#include "io/video_frames.hpp"
#include "sam/masks.hpp"
#include "sam/model_cache.hpp"
#if defined(PHOTARA_HAS_SPLAT)
#include "splat/dataset.hpp"
#include "splat/cuda_vulkan_preview.hpp"
#include "splat/formats.hpp"
#include "splat/trainer.hpp"
#endif
#if defined(PHOTARA_HAS_TEXTURE)
#include "texture/bake.hpp"
#include "texture/export.hpp"
#include "texture/mask.hpp"
#include "texture/options.hpp"
#endif
#if defined(PHOTARA_HAS_MESH_TOOLS)
#include "photara_mesh/mesh_ops.hpp"
#endif

#include <cxxopts.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <crtdbg.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {

// Atlas size used when neither --atlas-resolution nor a quality preset picks
// one; kept in sync with photara::texture::TextureOptions.
constexpr std::uint32_t kDefaultAtlasResolution{2048U};

struct ReconstructCli {
    std::filesystem::path images_dir;
    double focal_pixels{};
    photara::CameraModel camera_model{photara::CameraModel::automatic};
    std::string mode{"global"};
    bool trust_focal{false};
    bool structural_pair_expansion{false};
    bool positioning_cuda{true};
    std::filesystem::path output;
    // Optional COLMAP text model written next to the native output, so the
    // alignment can be inspected by COLMAP-family tools (and by scripts that
    // validate poses against a reference reconstruction).
    std::filesystem::path export_colmap_dir;
    // Editor/interactive runs skip user-facing ascan/asfm/PLY sidecars, eval
    // PNG dumps, and the extra timestamped log file. The caller already
    // captures stdout. Working copies stay in cache for preview/export.
    bool gui{false};
    std::filesystem::path working_sfm;
    std::filesystem::path working_splat;
    std::filesystem::path working_mesh;
    std::filesystem::path working_dense;
    std::filesystem::path working_texture;
    // Handshake files the editor consumes while Align runs. Empty keeps the
    // historical behaviour of deriving them next to the working copy.
    std::filesystem::path align_live;
    std::filesystem::path align_preview;
    bool texture_only{false};
    bool export_mvs_requested{false};
    std::filesystem::path export_mvs_path;
    std::size_t neighbor_window{3};
    float match_ratio{0.8F};
    bool mutual_check{true};
    double sift_contrast{0.005};
    std::filesystem::path cache_dir;
    std::string extractor{"siftgpu"};
    std::string matcher{"gpu_mutual_ratio"};
    std::string pipeline;  // empty / none | lightglue_end2end
    unsigned max_features{27000U};
    std::filesystem::path extractor_model;
    std::uint32_t extractor_width{1024U};
    std::uint32_t extractor_height{1024U};
    float extractor_min_score{-1.F};
    bool extractor_cpu{false};
    std::filesystem::path lightglue_model;
    std::string lightglue_extractor{"disk"};
    std::uint32_t lightglue_width{1024U};
    std::uint32_t lightglue_height{1024U};
    float lightglue_min_score{0.0F};
    unsigned hybrid_lightglue_max_features{2048U};
    bool lightglue_cpu{false};
    bool dense{false};
    bool splat{false};
    bool splat_view{false};
    std::string capture_mode{"object"};
    std::filesystem::path subject_bounds;
    std::filesystem::path splat_dataset;
    std::filesystem::path dense_ply;
    unsigned splat_iterations{10'000};
    unsigned splat_log_interval{100};
    unsigned splat_preview_interval{0};
    unsigned splat_preview_view{0};
    std::filesystem::path splat_preview_view_file;
    std::filesystem::path splat_preview_camera_file;
    std::filesystem::path splat_preview_vis_file;
    std::filesystem::path splat_preview_ack_file;
    std::filesystem::path splat_preview_dir;
    std::uint64_t splat_preview_vk_memory_handle{};
    std::uint64_t splat_preview_vk_semaphore_handle{};
    std::uint64_t splat_preview_vk_allocation_size{};
    unsigned splat_preview_vk_width{};
    unsigned splat_preview_vk_height{};
    std::uint64_t splat_preview_vk_device_luid{};
    unsigned splat_preview_vk_device_node_mask{};
    bool splat_profile_cuda{false};
    bool splat_fuse_sh_adam{true};
    unsigned splat_profile_interval{100};
    unsigned splat_sh_degree{3};
    unsigned splat_max_resolution{1'920};
    bool splat_undistort{false};
    float splat_kernel_size{0.F};
    bool splat_progressive_resolution{true};
    unsigned splat_progressive_interval{3'000};
    float splat_progressive_initial_scale{0.25F};
    std::uint64_t splat_view_cache_mb{6'144};
    std::uint64_t splat_device_cache_mb{512};
    std::uint64_t splat_device_cache_max_mb{0};
    bool splat_async_upload{true};
    bool splat_cache_auto{true};
    unsigned splat_prefetch_views{4};
    bool splat_prefetch_adaptive{true};
    unsigned splat_eval_split_every{8};
    bool splat_use_mask{true};
    bool splat_bilateral_grid{false};
    bool splat_bilateral_grid_shared{true};
    unsigned splat_bilateral_grid_width{16};
    unsigned splat_bilateral_grid_height{16};
    unsigned splat_bilateral_grid_luma{8};
    float splat_bilateral_grid_lr{2e-3F};
    float splat_bilateral_grid_tv{10.F};
    bool splat_ppisp{false};
    std::string splat_ppisp_type{"no_crf_no_vig"};
    float splat_ppisp_lr{2e-3F};
    bool splat_ppisp_before_bilagrid{true};
    std::string splat_alpha_mode{"masked"};
    float splat_match_alpha_weight{0.25F};
    float splat_ssim_weight{0.2F};
    float splat_opacity_reg{0.F};
    float splat_log_scale_reg{0.F};
    float splat_depth_normal_weight{0.05F};
    bool splat_normal_field{false};
    float splat_normal_field_weight{0.05F};
    float splat_normal_field_depth_ratio{0.6F};
    unsigned splat_normal_field_from_iter{8'001};
    float splat_multi_view_geo_weight{0.02F};
    float splat_multi_view_ncc_weight{0.6F};
    float splat_multi_view_depth_bracket{0.F};
    float splat_multi_view_depth_tolerance{0.F};
    unsigned splat_multi_view_num{8};
    unsigned splat_multi_view_tail_interval{1};
    bool splat_multi_view_adaptive{false};
    unsigned splat_multi_view_adaptive_max_interval{2};
    unsigned splat_multi_view_stable_refinements{5};
    float splat_multi_view_stable_count_threshold{0.005F};
    float splat_multi_view_stable_churn_threshold{0.01F};
    float splat_multi_view_stable_depth_threshold{0.02F};
    float splat_multi_view_min_depth_consistency{0.5F};
    float splat_multi_view_stable_distribution_threshold{0.025F};
    float splat_multi_view_pixel_noise{1.F};
    unsigned splat_geometry_from_iter{3'000};
    float splat_min_scale_fraction{1e-4F};
    float splat_max_scale_fraction{0.002F};
    float splat_max_scale_ratio{0.F};
    bool splat_constrain_scales{false};
    std::string splat_strategy{"adc_igs"};
    float splat_growth_factor{0.F};
    unsigned splat_seed{42};
    bool splat_densification{true};
    unsigned splat_structure_freeze_iter{0};
    std::uint64_t splat_densification_cap{1'000'000};
    std::uint64_t splat_init_point_budget{0};
    bool mesh{false};
    bool mvs_mesh_only{false};
    std::filesystem::path mask_mesh;
    bool mesh_obj{false};
    std::string mesh_method{"auto"};
    std::uint64_t mesh_max_points{2'000'000};
    std::uint64_t mesh_target_faces{0};
    bool mesh_remesh{true};
    float mesh_tsdf_voxel_scale{-1.F};
    float mesh_tsdf_bounds_padding{2.F};
    unsigned mesh_tsdf_pixel_step{4};
    unsigned mesh_tsdf_support_closing_axes{2};
    std::filesystem::path mesh_tsdf_frame_export_dir;
    unsigned mesh_tsdf_smooth_iters{2};
    float mesh_tsdf_smooth_lambda{0.5F};
    float mesh_tsdf_smooth_mu{-0.53F};
    float mesh_dist_insert_px{-1.F};
    bool mesh_free_space_support{true};
    bool mesh_adaptive_sigma{true};
    float mesh_max_edge_scale{4.F};
    float mesh_free_space_quantile{0.95F};
    std::uint64_t pam_max_points{1'000'000};
    std::uint64_t pam_pivot_max_points{1'000'000};
    float pam_pivot_std_factor{3.F};
    float pam_gaussian_seed_fraction{0.F};
    unsigned pam_refinement_steps{10};
    unsigned pam_neighbors{32};
    unsigned pam_points_per_tetrahedron{10};
    float pam_occupancy_iso_value{0.5F};
    float pam_vacancy_threshold{0.1F};
    float pam_mask_background_threshold{0.01F};
    unsigned patchmatch_tile_rows{8};
    unsigned patchmatch_concurrent_views{8};
    photara::mvs::DensifyQuality dense_quality{
        photara::mvs::DensifyQuality::default_quality};
    // Unset follows --dense-quality instead of forcing a working resolution.
    std::optional<unsigned> dense_resolution_level;
    std::filesystem::path masks_dir;
    bool masks_auto{true};
    std::filesystem::path sam_model;
    std::string sam_backend{"auto"};
    std::string sam_text;
    std::string sam_negative_text;
    bool sam_keep_prompted{true};
    bool sam_video{true};
    int sam_max_size{0};
    bool sam_refresh{false};
    bool texture{false};
    bool delight{false};
    // Unset keeps the default atlas size (kDefaultAtlasResolution).
    std::optional<std::uint32_t> atlas_resolution;
    std::uint32_t uv_parallel_partitions{8};
    bool texture_optimize{true};
    std::uint32_t texture_optimize_steps{1000};
    std::uint32_t texture_optimize_batch_size{4};
    std::uint32_t texture_seam_samples{4};
    float video_fps{2.F};
    int video_sharp_window{3};
    int video_max_frames{0};
    int video_quality{95};
    float video_scale{1.F};
    int video_rotate{0};
    std::filesystem::path video_frames_dir;
    std::filesystem::path video_source;
    std::filesystem::path ffmpeg{"ffmpeg"};
    bool video_redo{false};
};

std::uint64_t peak_working_set_bytes() noexcept {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (!K32GetProcessMemoryInfo(
            GetCurrentProcess(), &counters, sizeof(counters)))
        return 0;
    return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
#endif
}

#if defined(_WIN32)
std::string wide_to_utf8(const wchar_t* text) {
    if (*text == L'\0') return {};
    const int size = WideCharToMultiByte(
        CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) return {};
    std::string result(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(
        CP_UTF8, 0, text, -1, result.data(), size, nullptr, nullptr);
    result.pop_back();
    return result;
}

std::filesystem::path utf8_to_path(const std::string& utf8) {
    if (utf8.empty()) return {};
    const int wide_size = MultiByteToWideChar(
        CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (wide_size <= 1) return {};
    std::wstring wide(static_cast<std::size_t>(wide_size), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, utf8.c_str(), -1, wide.data(), wide_size);
    wide.pop_back();
    return std::filesystem::path(wide);
}
#else
std::filesystem::path utf8_to_path(const std::string& utf8) {
    return std::filesystem::path(utf8);
}
#endif

struct Utf8Argv {
    std::vector<std::string> storage;
    std::vector<char*> pointers;

#if defined(_WIN32)
    explicit Utf8Argv(int argc, wchar_t** argv) {
        storage.reserve(static_cast<std::size_t>(argc));
        for (int i = 0; i < argc; ++i)
            storage.push_back(wide_to_utf8(argv[i]));
        pointers.reserve(storage.size());
        for (auto& argument : storage) pointers.push_back(argument.data());
    }
#else
    explicit Utf8Argv(int argc, char** argv) {
        storage.reserve(static_cast<std::size_t>(argc));
        for (int i = 0; i < argc; ++i) storage.emplace_back(argv[i]);
        pointers.reserve(storage.size());
        for (auto& argument : storage) pointers.push_back(argument.data());
    }
#endif

    int argc() const { return static_cast<int>(pointers.size()); }
    char** argv() { return pointers.data(); }
};

void print_help(const cxxopts::Options& options) {
    std::cout << PHOTARA_VERSION_STRING << '\n'
              << options.help() << '\n'
              << "Feature backends:\n"
              << "  default   --extractor siftgpu --matcher gpu_mutual_ratio\n"
              << "  compose   any compatible --extractor × --matcher\n"
              << "  fused     --pipeline lightglue_end2end (optional recipe)\n"
              << "Modes:\n"
              << "  incremental  star initialization + PnP resection\n"
              << "  hierarchical clustered incremental SfM + Sim(3) merge\n"
              << "  global       rotation averaging + global positioning + BA\n"
              << "Dense (optional Stage A Fast MVS after SfM):\n"
              << "  --dense      PatchMatch depth + fuse -> dense.ply; with --mesh, an MVS surface\n"
              << "               (does not train splats unless --splat is also set)\n"
              << "  --splat       train CUDA Gaussian splats -> *_splat.ply, or --output .sog/.spz/.glb\n"
              << "  --splat-view  orbit-preview a trained splat from the camera sidecar\n"
              << "  --splat-dataset PATH  external COLMAP/RealityCapture/OpenMVS cameras (auto-detected)\n"
              << "  --dense-ply PATH  replace initial points; without camera data, use internal SfM\n"
              << "  --splat-iterations N  splat optimizer steps (default 10000)\n"
              << "  --splat-log-interval N  training-stat log every N steps (default 100, 0 = first/last)\n"
              << "  --splat-preview-interval N  emit a live preview every N steps (0 disables)\n"
              << "  --splat-preview-view N  camera index for live preview (default 0, first frame)\n"
              << "  --splat-preview-view-file PATH  optional file the editor updates to switch cameras\n"
              << "  --splat-preview-camera-file PATH  optional orbit-camera sidecar (overrides view index)\n"
              << "  --splat-preview-vis-file PATH  optional splat/points/rings visualization sidecar\n"
              << "  --splat-preview-ack-file PATH  optional sidecar the editor updates with the frames it copied\n"
              << "  --splat-preview-dir PATH  editor preview PNG directory\n"
              << "  --splat-profile-cuda BOOL  CUDA-event timings for training stages (default false)\n"
              << "  --splat-profile-interval N  profiling aggregation window (default 100, max 1000)\n"
              << "  --splat-device-cache-mb N  packed CUDA image cache budget (default 512, 0 disables)\n"
              << "  --splat-device-cache-max-mb N  ceiling for the adaptive device cache\n"
              << "                              (default 0 = keep --splat-device-cache-mb)\n"
              << "  --splat-async-upload BOOL  overlap future packed-view H2D copies (default true)\n"
              << "  --splat-cache-auto BOOL  grow cache budgets safely for large datasets (default true)\n"
        << "  --splat-prefetch-views N  minimum concurrent host image prefetch count (default 4)\n"
        << "  --splat-prefetch-adaptive BOOL  grow the prefetch lookahead from the measured\n"
        << "                              host-load/iteration times (default true)\n"
              << "  --splat-sh-degree N  spherical-harmonic bands 0..3 (default 3)\n"
              << "  --splat-kernel-size V  screen covariance low-pass variance; "
                 "0 disables, 0.1 matches Brush Mip\n"
              << "  --splat-progressive-resolution BOOL  1/4 -> 1/2 -> full schedule (default true)\n"
              << "  --splat-progressive-interval N  iterations per resolution level (default 3000)\n"
              << "  --splat-eval-split-every N  hold out every Nth view for PSNR/SSIM (default 0 = every view trains; metrics then come from three training views)\n"
              << "  --splat-use-mask BOOL  apply masks/ or source alpha to training (default true)\n"
              << "  --splat-alpha-mode masked|transparent (default masked)\n"
              << "  --splat-match-alpha-weight W  transparent alpha BCE weight (default 0.25)\n"
              << "  --splat-ssim-weight W  structural loss blend (default 0.2)\n"
              << "  --splat-opacity-reg W  per-Gaussian opacity prior (default 0)\n"
              << "  --splat-log-scale-reg W  per-Gaussian log-scale prior (default 0)\n"
              << "  --splat-depth-normal-weight W  median-depth/normal consistency (default 0.05)\n"
              << "  --splat-normal-field BOOL  learn GaussianWrapping normal features (default false)\n"
              << "  --splat-normal-field-weight W  normal-field consistency weight (default 0.05)\n"
              << "  --splat-normal-field-depth-ratio R  median-depth alignment ratio (default 0.6)\n"
              << "  --splat-normal-field-from-iter N  start normal-field loss (default 8001)\n"
              << "  --splat-mv-geo-weight W  multi-view round-trip loss (default 0.02)\n"
              << "  --splat-mv-ncc-weight W  plane-warp NCC loss (default 0.6)\n"
              << "  --splat-mv-neighbors N  nearest camera candidates (default 8)\n"
              << "  --splat-mv-tail-interval N  multi-view interval after ADC growth stops (default 1;\n"
              << "                              N>1 lowers geometry/mesh quality - previews only)\n"
              << "  --splat-mv-adaptive BOOL  lower multi-view frequency only after geometry stabilizes\n"
              << "  --splat-mv-adaptive-max-interval N  adaptive interval ceiling (default 2)\n"
              << "  --splat-mv-stable-refinements N  stable refine windows before each reduction (default 5)\n"
              << "  --splat-mv-pixel-noise P  geometry reprojection gate (default 1px)\n"
              << "  --splat-geometry-from-iter N  start geometry loss (default 3000)\n"
              << "  --splat-min-scale-fraction F  minimum scale / scene extent (default 1e-4)\n"
              << "  --splat-max-scale-fraction F  maximum scale / scene extent (default 0.002)\n"
              << "  --splat-max-scale-ratio R  hard anisotropy clamp (default 0 = no limit)\n"
              << "  --splat-constrain-scales=BOOL  clamp sparse KNN scales (default false)\n"
              << "  --splat-bilateral-grid=BOOL  spatially-varying affine colour "
                 "correction (default false)\n"
              << "  --splat-ppisp=BOOL  PPISP exposure/white-balance correction "
                 "(default false)\n"
              << "  --splat-ppisp-type TYPE  no_crf_no_vig (default), no_crf, or original\n"
              << "  --splat-strategy adc_plus|adc_igs|emc\n"
              << "  --splat-densification=BOOL  enable split/prune (default true)\n"
              << "  --splat-structure-freeze-iter N  freeze geometry/opacity after N (default 0)\n"
              << "  --splat-densification-cap N  densify growth ceiling (default 1000000)\n"
              << "  --splat-growth-factor W  EMC per-refine count multiplier; 0 keeps the strategy preset\n"
              << "  --splat-init-point-budget N  cap the initialization cloud (default 0 = keep all)\n"
              << "  --mesh       also build a surface mesh -> mesh.ply\n"
              << "  --mask-mesh PATH  load an existing PLY mesh and render masks/previews\n"
              << "  --mesh-method auto|tsdf|delaunay|pam\n"
              << "               auto uses TSDF for splats, otherwise the quality preset\n"
              << "  --pam-max-points N  PAM refined surface candidates (default 1000000)\n"
              << "  --pam-pivot-max-points N  GaussianWrapping pivot vertices before tetra_triangulation\n"
              << "  --pam-pivot-std-factor F  learned-normal pivot displacement in sigma (default 3)\n"
              << "  --pam-gaussian-seed-fraction F  optional direct Gaussian candidate fraction (default 0)\n"
              << "  --pam-refinement-steps N  PAM occupancy/vector-field steps (default 10)\n"
              << "  --pam-neighbors N  Gaussian neighbors in the PAM field (default 32)\n"
              << "  --pam-occupancy-iso-value V  occupied threshold (0.3 matches GW iso shift 0.2)\n"
              << "  --mesh-dist-insert-px N  global Delaunay projection spacing\n"
              << "  --mesh-free-space-support BOOL  weak-surface beta/gamma cut\n"
              << "  --mesh-adaptive-sigma BOOL  adapt visibility scale to local point density\n"
              << "  --mesh-max-edge-scale F  reject gap-spanning cut facets (default 4; 0 disables)\n"
              << "  --mesh-free-space-quantile Q  support-scale calibration (0 disables)\n"
              << "  --mesh-target-faces N  photara/CGAL repair + decimate target (0 disables)\n"
              << "  --mesh-remesh BOOL  Instant Meshes before CGAL repair (default true)\n"
              << "  --mesh-tsdf-voxel-scale F  inferred voxel multiplier (-1 = auto)\n"
              << "  --mesh-tsdf-bounds-padding F  point-cloud bounds multiplier (default 2)\n"
              << "  --mesh-tsdf-pixel-step N  sparse allocation stride (1 preserves thin wires; default 4)\n"
              << "  --mesh-tsdf-support-closing-axes N  0 disables; 2 = conservative default\n"
              << "  --mesh-tsdf-frame-export-dir DIR  export exact TSDF input frames for A/B\n"
              << "  --mesh-tsdf-smooth-iters N  boundary-locked Taubin passes (default 2)\n"
              << "  --mesh-obj   additionally write the much slower ASCII OBJ\n"
              << "  --dense-quality preview|default|high (whole-pipeline preset)\n"
              << "  --dense-resolution-level N  MVS downscale steps; unset follows\n"
                 "                              --dense-quality (0=full, 1~=half)\n"
              << "  --splat-dataset PATH --dense --mesh  MVS with fixed imported cameras (no splat training)\n"
              << "  --capture-mode object|scene  object uses SfM SubjectBounds; with --dense\n"
              << "               and without --splat this only selects the bounds\n"
              << "  --subject-bounds PATH  load object focus region (SubjectBounds txt)\n"
              << "  --masks DIR optional valid-region masks for default SfM/MVS/training; "
                 "black pixels are ignored (auto: sibling masks/)\n"
              << "Texture (Stage B after --mesh; requires Vulkan + UVAtlas):\n"
              << "  --texture    UV unwrap + projective bake -> textured OBJ/MTL/PNG\n"
              << "  --texture --working-mesh PATH  bake only; reuse an existing mesh\n"
              << "  --delight    Intrinsic image delighter before bake (albedo)\n"
              << "  --atlas-resolution N  atlas size (default 2048)\n"
              << "  --uv-parallel-partitions N  concurrent UVAtlas partitioning (default 8)\n"
              << "  --texture-optimize BOOL  photara_drender photometric + seam optimization (default true)\n"
              << "  --texture-optimize-steps N  native Vulkan Adam steps (default 1000)\n"
              << "  --texture-optimize-batch-size N  calibrated views per optimizer step (default 4)\n"
              << "  --texture-seam-samples N  samples per UV chart seam edge (default 4)\n"
              << "Output formats:\n"
              << "  .ascan  Photara project (settings + completed stages)\n"
              << "  .asfm   native SfM scene (cameras, keypoints, tracks)\n"
              << "  .ply    sparse XYZRGB point cloud; splat sidecar is {stem}_splat.ply\n"
              << "  .sog/.spz/.glb  trained Gaussian file (format from the suffix)\n"
              << "  .mvs    OpenMVS Interface (interop export)\n"
              << "  --export-mvs [path]  also write OpenMVS Interface after SfM\n"
              << "  --gui  editor: no ascan/asfm/PLY sidecars, eval PNG, extra log\n"
              << "  --working-sfm PATH  compact SfM working copy for --gui\n"
              << "  --working-splat PATH  trained Gaussian working copy for --gui\n"
              << "  --working-mesh PATH  mesh working copy for --gui\n"
              << "  --working-dense PATH  dense cloud working copy for --gui\n"
              << "  --working-texture STEM  textured OBJ/MTL/PNG working copy for --gui\n"
              << "  --align-live PATH  alignment live frame (default <working-sfm>.live)\n"
              << "  --align-preview PATH  alignment preview snapshot\n"
              << "Video (when --images is a video file, frames are extracted first):\n"
              << "  --video-fps F  kept frames per second (default 2)\n"
              << "  --video-sharp-window N  keep the sharpest of N candidates (default 3; 1 = off)\n"
              << "  --video-max-frames N  cap extracted stills (0 = no cap)\n"
              << "  --video-quality Q  JPEG quality 0-100 (default 95; outside writes PNG)\n"
              << "  --video-scale S  resize extracted frames (default 1)\n"
              << "  --video-rotate D  clockwise degrees 0/90/180/270\n"
              << "  --video-frames-dir PATH  stills folder (default <video_stem>/images)\n"
              << "  --ffmpeg PATH  ffmpeg executable (default ffmpeg on PATH)\n"
              << "  --video-redo  ignore a previous extract and write frames again\n"
              << "  with --dense: also writes dense.ply next to --output\n"
              << "  with --texture: also writes *_textured.obj/.mtl/_albedo.png\n"
              << "Log level: set PHOTARA_LOG_LEVEL=error|warning|info|debug|trace|off\n";
}

ReconstructCli parse_cli(int argc, char** argv) {
    cxxopts::Options options(
        "photara", "High-performance Structure from Motion reconstruction");
    options.custom_help("[options]");
    options.add_options()
        ("h,help", "Print usage")
        ("v,version", "Print version")
        ("i,images", "Image directory or video file", cxxopts::value<std::string>())
        ("video-fps", "Kept frames per second when --images is a video",
         cxxopts::value<float>()->default_value("2"))
        ("video-sharp-window",
         "Keep the sharpest of N candidate frames (1 disables blur selection)",
         cxxopts::value<int>()->default_value("3"))
        ("video-max-frames", "Maximum extracted stills (0 = no cap)",
         cxxopts::value<int>()->default_value("0"))
        ("video-quality", "JPEG quality for extracted frames (0-100; else PNG)",
         cxxopts::value<int>()->default_value("95"))
        ("video-scale", "Scale factor applied to extracted frames",
         cxxopts::value<float>()->default_value("1"))
        ("video-rotate", "Clockwise rotation in degrees (0/90/180/270)",
         cxxopts::value<int>()->default_value("0"))
        ("video-frames-dir", "Directory for extracted stills",
         cxxopts::value<std::string>()->default_value(""))
        ("ffmpeg", "ffmpeg executable used to decode video",
         cxxopts::value<std::string>()->default_value("ffmpeg"))
        ("video-redo", "Re-extract video frames even if a matching set exists",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("f,focal",
         "Initial focal in pixels (0 = max dimension * 1.2, fisheye * 0.5; "
         "refined by view-graph consensus + BA unless trusted)",
         cxxopts::value<double>()->default_value("0"))
        ("m,mode",
         "Reconstruction mode: global (default), incremental, or hierarchical",
         cxxopts::value<std::string>()->default_value("global"))
        ("camera-model", "Camera model: auto | pinhole | opencv_fisheye (alias fisheye) | equirectangular (aliases panorama, equirect)",
         cxxopts::value<std::string>()->default_value("auto"))
        ("trust-focal", "Lock externally calibrated --focal and zero distortion",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("sfm-structural-rescue", "Experimental bridge-branch pair expansion (not validated for production)",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("export-colmap",
         "Also write the aligned scene as a COLMAP text model into this folder",
         cxxopts::value<std::string>()->default_value(""))
        ("o,output",
         "Output path (.ascan, .asfm, .ply, .mvs, or .sog/.spz/.glb for Gaussians)",
         cxxopts::value<std::string>())
        ("gui",
         "Editor run: skip ascan/asfm/PLY sidecars, eval dumps, and extra log file",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("working-sfm",
         "Compact SfM working copy used by --gui (read on --splat, write after SfM)",
         cxxopts::value<std::string>()->default_value(""))
        ("working-splat",
         "Gaussian file used by --gui training and --splat-view",
         cxxopts::value<std::string>()->default_value(""))
        ("working-mesh",
         "Mesh working copy used by --gui instead of a sidecar PLY",
         cxxopts::value<std::string>()->default_value(""))
        ("working-dense",
         "Dense cloud working copy used by --gui instead of a sidecar PLY",
         cxxopts::value<std::string>()->default_value(""))
        ("working-texture",
         "Textured OBJ/MTL/PNG working-copy stem used by --gui",
         cxxopts::value<std::string>()->default_value(""))
        ("align-live",
         "Editor alignment live-frame path (default: <working-sfm>.live)",
         cxxopts::value<std::string>()->default_value(""))
        ("align-preview",
         "Editor alignment preview snapshot path "
         "(default: <working-sfm>.preview.asfm)",
         cxxopts::value<std::string>()->default_value(""))
        ("export-mvs",
         "Write OpenMVS Interface after SfM. Optional path; default is "
         "<output-stem>.mvs. .ply output does not write .mvs unless this is set.",
         cxxopts::value<std::string>()->implicit_value(""))
        ("window", "Sequential neighbor window",
         cxxopts::value<std::size_t>()->default_value("3"))
        ("match-ratio", "Lowe ratio test threshold",
         cxxopts::value<float>()->default_value("0.8"))
        ("mutual-check", "Mutual match consistency check",
         cxxopts::value<bool>()->default_value("true"))
        ("sift-contrast", "SIFT contrast threshold",
         cxxopts::value<double>()->default_value("0.005"))
        ("cache-dir", "Feature cache directory (- to disable)",
         cxxopts::value<std::string>()->default_value(""))
        ("ba-backend",
         "Bundle adjustment backend: automatic (default), cpu, or cuda "
         "(cuda warns when a solve falls back)",
         cxxopts::value<std::string>()->default_value("automatic"))
        ("positioning-cuda", "Use CUDA for supported global bearing positioning solves (default: true)",
         cxxopts::value<bool>()->default_value("true"))
        ("extractor",
         "Feature extractor: siftgpu (default), sift, superpoint, disk, aliked",
         cxxopts::value<std::string>()->default_value("siftgpu"))
        ("matcher",
         "Feature matcher: gpu_mutual_ratio (default), mutual_ratio, "
         "lightglue, hybrid_lightglue. "
         "Legacy alias: siftgpu→gpu_mutual_ratio",
         cxxopts::value<std::string>()->default_value("gpu_mutual_ratio"))
        ("pipeline",
         "Optional fused pair recipe: none (default) or lightglue_end2end",
         cxxopts::value<std::string>()->default_value(""))
        ("max-features", "Maximum features per image",
         cxxopts::value<unsigned>()->default_value("27000"))
        ("extractor-model",
         "ONNX weights for --extractor superpoint|disk|aliked",
         cxxopts::value<std::string>()->default_value(""))
        ("extractor-width", "Learned extractor network input width",
         cxxopts::value<std::uint32_t>()->default_value("1024"))
        ("extractor-height", "Learned extractor network input height",
         cxxopts::value<std::uint32_t>()->default_value("1024"))
        ("extractor-min-score",
         "Learned keypoint score threshold (-1 = backend default)",
         cxxopts::value<float>()->default_value("-1"))
        ("extractor-cpu", "Force learned extractor ONNX on CPU",
         cxxopts::value<bool>()->default_value("false"))
        ("lightglue-model",
         "ONNX for --matcher lightglue|hybrid_lightglue or "
         "--pipeline lightglue_end2end",
         cxxopts::value<std::string>()->default_value(""))
        ("lightglue-extractor",
         "End2end pipeline head only: disk or superpoint",
         cxxopts::value<std::string>()->default_value("disk"))
        ("lightglue-width", "End2end pipeline network width",
         cxxopts::value<std::uint32_t>()->default_value("1024"))
        ("lightglue-height", "End2end pipeline network height",
         cxxopts::value<std::uint32_t>()->default_value("1024"))
        ("lightglue-min-score", "Drop LightGlue matches below this score",
         cxxopts::value<float>()->default_value("0"))
        ("hybrid-lightglue-max-features",
         "Maximum descriptors per image in hybrid LightGlue rescue (0 = all)",
         cxxopts::value<unsigned>()->default_value("2048"))
        ("lightglue-cpu", "Force LightGlue matcher / end2end ONNX on CPU",
         cxxopts::value<bool>()->default_value("false"))
        ("dense", "Run Fast MVS densify after SfM",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("splat", "Train CUDA Gaussian splats directly from SfM sparse points",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("splat-view",
         "Orbit-preview a trained splat from --splat-preview-camera-file",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("capture-mode",
         "Capture type: object uses SfM SubjectBounds; scene is unbounded",
         cxxopts::value<std::string>()->default_value("object"))
        ("subject-bounds",
         "Load object-mode SubjectBounds / focus region from a text file",
         cxxopts::value<std::string>()->default_value(""))
        ("splat-dataset",
         "External camera dataset: COLMAP root, RealityCapture CSV/dir, or OpenMVS .mvs",
         cxxopts::value<std::string>()->default_value(""))
        ("dense-ply", "Dense PLY initializer for external or internal-SfM cameras",
         cxxopts::value<std::string>()->default_value(""))
        ("splat-iterations", "Splat optimizer iterations",
         cxxopts::value<unsigned>()->default_value("10000"))
        ("splat-log-interval",
         "Log splat training stats every N iterations (0 = first and last only)",
         cxxopts::value<unsigned>()->default_value("100"))
        ("splat-preview-interval",
         "Emit a live preview from a fixed camera every N iterations",
         cxxopts::value<unsigned>()->default_value("0"))
        ("splat-preview-view",
         "Camera index for live preview (0 is the first captured frame)",
         cxxopts::value<unsigned>()->default_value("0"))
        ("splat-preview-view-file",
         "Sidecar file whose integer contents select the live preview camera",
         cxxopts::value<std::string>()->default_value(""))
        ("splat-preview-camera-file",
         "Sidecar file with an orbit-camera W2C pose for the live preview",
         cxxopts::value<std::string>()->default_value(""))
        ("splat-preview-vis-file",
         "Sidecar file selecting live preview shading: splat, points, or rings",
         cxxopts::value<std::string>()->default_value(""))
        ("splat-preview-ack-file",
         "Sidecar file the editor updates with the preview frames it consumed",
         cxxopts::value<std::string>()->default_value(""))
        ("splat-preview-dir", "Directory for live training preview PNGs",
         cxxopts::value<std::string>()->default_value(""))
        ("splat-preview-vk-memory-handle",
         "Inherited Win32 Vulkan external-memory handle",
         cxxopts::value<std::uint64_t>()->default_value("0"))
        ("splat-preview-vk-semaphore-handle",
         "Inherited Win32 Vulkan timeline-semaphore handle",
         cxxopts::value<std::uint64_t>()->default_value("0"))
        ("splat-preview-vk-allocation-size",
         "Vulkan external image allocation size",
         cxxopts::value<std::uint64_t>()->default_value("0"))
        ("splat-preview-vk-width", "Vulkan external image width",
         cxxopts::value<unsigned>()->default_value("0"))
        ("splat-preview-vk-height", "Vulkan external image height",
         cxxopts::value<unsigned>()->default_value("0"))
        ("splat-preview-vk-device-luid", "Vulkan physical-device Win32 LUID",
         cxxopts::value<std::uint64_t>()->default_value("0"))
        ("splat-preview-vk-device-node-mask", "Vulkan device-node mask",
         cxxopts::value<unsigned>()->default_value("0"))
        ("splat-profile-cuda",
         "Record windowed CUDA-event timings for splat training stages",
         cxxopts::value<bool>()->default_value("false")
             ->implicit_value("true"))
        ("splat-profile-interval",
         "CUDA profiling aggregation window (1..1000 iterations)",
         cxxopts::value<unsigned>()->default_value("100"))
        ("splat-sh-degree", "Spherical-harmonic colour bands (0..3)",
         cxxopts::value<unsigned>()->default_value("3"))
        ("splat-max-resolution", "Maximum splat training image dimension (0 = source)",
         cxxopts::value<unsigned>()->default_value("1920"))
        ("splat-undistort",
         "Resample OpenCV fisheye views onto a pinhole camera before splat training",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("splat-kernel-size",
         "Screen covariance low-pass variance (0 disables; Brush Mip uses 0.1)",
         cxxopts::value<float>()->default_value("0"))
        ("splat-progressive-resolution",
         "Enable 1/4 -> 1/2 -> full coarse-to-fine splat training",
         cxxopts::value<bool>()->default_value("true")
             ->implicit_value("true"))
        ("splat-progressive-interval",
         "Iterations per splat resolution level",
         cxxopts::value<unsigned>()->default_value("3000"))
        ("splat-progressive-initial-scale",
         "Initial splat linear image scale",
         cxxopts::value<float>()->default_value("0.25"))
        ("splat-view-cache-mb", "Packed RGBA8 splat host-view LRU budget (0 = no cache)",
         cxxopts::value<std::uint64_t>()->default_value("6144"))
        ("splat-device-cache-mb", "Packed splat CUDA-view LRU budget, capped by free VRAM (0 = disabled)",
         cxxopts::value<std::uint64_t>()->default_value("512"))
        ("splat-device-cache-max-mb",
         "Ceiling the adaptive splat CUDA-view cache may grow to (0 = no growth)",
         cxxopts::value<std::uint64_t>()->default_value("0"))
        ("splat-async-upload",
         "Upload decoded future splat views on a non-blocking CUDA copy stream",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("splat-fuse-sh-adam", "Fuse SH projection gradients into Adam",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("splat-cache-auto",
         "Grow host cache; shrink CUDA cache within its explicit budget and VRAM safety limits",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("splat-prefetch-views",
         "Minimum concurrent host decode/pack prefetch views (0 disables; host-side only)",
         cxxopts::value<unsigned>()->default_value("4"))
        ("splat-prefetch-adaptive",
         "Grow the prefetch lookahead to cover one measured host load plus one iteration",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("splat-eval-split-every",
         "Hold out every Nth view for PSNR/SSIM evaluation (0 = train all)",
         cxxopts::value<unsigned>()->default_value("0"))
        ("splat-use-mask", "Enable mask-aware training",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("splat-alpha-mode",
         "Mask semantics: masked ignores invalid rays; transparent trains output alpha",
         cxxopts::value<std::string>()->default_value("masked"))
        ("splat-match-alpha-weight", "Alpha BCE weight in transparent mode",
         cxxopts::value<float>()->default_value("0.25"))
        ("splat-ssim-weight", "SSIM blend in the photometric loss",
         cxxopts::value<float>()->default_value("0.2"))
        ("splat-opacity-reg", "Per-Gaussian opacity regularization weight",
         cxxopts::value<float>()->default_value("0"))
        ("splat-log-scale-reg", "Per-Gaussian log-scale regularization weight",
         cxxopts::value<float>()->default_value("0"))
        ("splat-depth-normal-weight",
         "Splat median-depth/raster-normal consistency weight",
         cxxopts::value<float>()->default_value("0.05"))
        ("splat-normal-field",
         "Learn GaussianWrapping four-channel normal-field features",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("splat-normal-field-weight", "Normal-field consistency loss weight",
         cxxopts::value<float>()->default_value("0.05"))
        ("splat-normal-field-depth-ratio",
         "Median-depth fraction of normal-field alignment",
         cxxopts::value<float>()->default_value("0.6"))
        ("splat-normal-field-from-iter", "Iteration to start normal-field loss",
         cxxopts::value<unsigned>()->default_value("8001"))
        ("splat-mv-geo-weight", "Multi-view depth round-trip loss weight",
         cxxopts::value<float>()->default_value("0.02"))
        ("splat-mv-depth-bracket",
         "Multi-view point-query median-depth seed window in world units "
         "(0 = derive from the scene extent, < 0 = reference +/-200)",
         cxxopts::value<float>()->default_value("0"))
        ("splat-mv-depth-tolerance",
         "Multi-view point-query depth precision in world units "
         "(0 = derive from the scene extent, < 0 = reference 8 refinements)",
         cxxopts::value<float>()->default_value("0"))
        ("splat-mv-ncc-weight", "Multi-view plane-warp NCC loss weight",
         cxxopts::value<float>()->default_value("0.6"))
        ("splat-mv-neighbors", "Number of nearest multi-view candidates",
         cxxopts::value<unsigned>()->default_value("8"))
        ("splat-mv-tail-interval",
         "Multi-view sampling interval after ADC growth stops (N>1 lowers geometry/mesh quality; "
         "keep 1 when meshing)",
         cxxopts::value<unsigned>()->default_value("1"))
        ("splat-mv-adaptive",
         "Adapt multi-view frequency from geometry stability",
         cxxopts::value<bool>()->default_value("false")
             ->implicit_value("true"))
        ("splat-mv-adaptive-max-interval",
         "Maximum geometry-stable multi-view interval",
         cxxopts::value<unsigned>()->default_value("2"))
        ("splat-mv-stable-refinements",
         "Stable refinement windows required before reducing frequency",
         cxxopts::value<unsigned>()->default_value("5"))
        ("splat-mv-stable-count-threshold",
         "Maximum relative Gaussian-count drift per stable window",
         cxxopts::value<float>()->default_value("0.005"))
        ("splat-mv-stable-churn-threshold",
         "Maximum grow+prune fraction per stable window",
         cxxopts::value<float>()->default_value("0.01"))
        ("splat-mv-stable-depth-threshold",
         "Maximum depth-consistency-rate drift per stable window",
         cxxopts::value<float>()->default_value("0.02"))
        ("splat-mv-min-depth-consistency",
         "Minimum depth round-trip consistency for stable geometry",
         cxxopts::value<float>()->default_value("0.5"))
        ("splat-mv-stable-distribution-threshold",
         "Maximum opacity/scale distribution drift per stable window",
         cxxopts::value<float>()->default_value("0.025"))
        ("splat-mv-pixel-noise", "Multi-view reprojection threshold in pixels",
         cxxopts::value<float>()->default_value("1"))
        ("splat-geometry-from-iter", "Iteration to start splat geometry loss",
         cxxopts::value<unsigned>()->default_value("3000"))
        ("splat-min-scale-fraction", "Minimum Gaussian scale / scene extent",
         cxxopts::value<float>()->default_value("0.0001"))
        ("splat-max-scale-fraction", "Maximum Gaussian scale / scene extent",
         cxxopts::value<float>()->default_value("0.002"))
        ("splat-max-scale-ratio",
         "Maximum Gaussian axis ratio (0 disables; default 0 = no limit)",
         cxxopts::value<float>()->default_value("0"))
        ("splat-constrain-scales", "Clamp sparse KNN scales to configured fractions",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("splat-bilateral-grid",
         "Spatially-varying affine bilateral-grid colour correction",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("splat-bilateral-grid-width", "Bilateral grid spatial width",
         cxxopts::value<unsigned>()->default_value("16"))
        ("splat-bilateral-grid-per-view",
         "Give every view its own grid instead of one shared grid",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("splat-bilateral-grid-height", "Bilateral grid spatial height",
         cxxopts::value<unsigned>()->default_value("16"))
        ("splat-bilateral-grid-luma", "Bilateral grid luma bins",
         cxxopts::value<unsigned>()->default_value("8"))
        ("splat-bilateral-grid-lr", "Bilateral grid Adam learning rate",
         cxxopts::value<float>()->default_value("0.0002"))
        ("splat-bilateral-grid-tv", "Bilateral grid total-variation weight",
         cxxopts::value<float>()->default_value("10"))
        ("splat-ppisp",
         "PPISP per-view exposure/white-balance colour correction",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("splat-ppisp-type",
         "PPISP layout: no_crf_no_vig, no_crf, or original",
         cxxopts::value<std::string>()->default_value("no_crf_no_vig"))
        ("splat-ppisp-lr", "PPISP Adam learning rate",
         cxxopts::value<float>()->default_value("0.002"))
        ("splat-ppisp-before-bilagrid",
         "Run PPISP before the bilateral grid when both are enabled",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("splat-strategy", "Densification: adc_plus, adc_igs",
         cxxopts::value<std::string>()->default_value("adc_igs"))
        ("splat-densification", "Enable splat split/prune",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("splat-growth-factor",
         "EMC per-refine growth multiplier (0 = strategy preset)",
         cxxopts::value<float>()->default_value("0"))
        ("splat-seed", "Splat training RNG seed",
         cxxopts::value<unsigned>()->default_value("42"))
        ("splat-structure-freeze-iter", "Freeze means/scale/quaternion/opacity after N",
         cxxopts::value<unsigned>()->default_value("0"))
        ("splat-densification-cap", "Densify growth ceiling",
         cxxopts::value<std::uint64_t>()->default_value("1000000"))
        ("splat-init-point-budget",
         "Cap on the initialization Gaussians built from the input cloud "
         "(0 keeps every point)",
         cxxopts::value<std::uint64_t>()->default_value("0"))
        ("mesh", "Build TSDF mesh from splats (or MVS mesh with --dense)",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("mvs-mesh-only",
         "External-dataset quality gate: mesh --dense-ply directly and "
         "render source-resolution masks without splat training",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("mask-mesh",
         "Existing PLY mesh for --mvs-mesh-only; bypasses Delaunay and renders "
         "photara_drender masks plus normal-shaded previews",
         cxxopts::value<std::string>()->default_value(""))
        ("mesh-method",
         "Mesh backend: auto, tsdf, delaunay, or pam",
         cxxopts::value<std::string>()->default_value("auto"))
        ("pam-max-points", "Maximum PAM refined surface candidates",
         cxxopts::value<std::uint64_t>()->default_value("1000000"))
        ("pam-pivot-max-points",
         "Maximum Gaussian pivot vertices for initial tetra triangulation",
         cxxopts::value<std::uint64_t>()->default_value("1000000"))
        ("pam-pivot-std-factor",
         "Learned-normal Gaussian pivot displacement in sigma",
         cxxopts::value<float>()->default_value("3"))
        ("pam-gaussian-seed-fraction",
         "Fraction of PAM candidates seeded from learned Gaussians",
         cxxopts::value<float>()->default_value("0"))
        ("pam-refinement-steps", "PAM occupancy/vector-field refinement steps",
         cxxopts::value<unsigned>()->default_value("10"))
        ("pam-neighbors", "Nearest Gaussians used by the PAM vector field",
         cxxopts::value<unsigned>()->default_value("32"))
        ("pam-points-per-tetrahedron", "PAM occupancy samples per tetrahedron",
         cxxopts::value<unsigned>()->default_value("10"))
        ("pam-occupancy-iso-value",
         "PAM integrated occupancy isosurface threshold",
         cxxopts::value<float>()->default_value("0.5"))
        ("pam-vacancy-threshold", "PAM candidate distance from occupancy 0.5",
         cxxopts::value<float>()->default_value("0.1"))
        ("pam-mask-background-threshold",
         "Mask alpha below this value votes empty in PAM",
         cxxopts::value<float>()->default_value("0.01"))
        ("mesh-max-points",
         "Maximum samples inserted into global Delaunay (0 = unlimited)",
         cxxopts::value<std::uint64_t>()->default_value("2000000"))
        ("mesh-target-faces",
         "Optional photara/CGAL repair and decimation target (0 disables)",
         cxxopts::value<std::uint64_t>()->default_value("0"))
        ("mesh-remesh", "Run Instant Meshes before CGAL repair",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("mesh-tsdf-voxel-scale",
         "Automatic TSDF voxel multiplier (-1 = gs2mesh default 1x)",
         cxxopts::value<float>()->default_value("-1"))
        ("mesh-tsdf-bounds-padding",
         "Point-cloud TSDF bounds multiplier",
         cxxopts::value<float>()->default_value("2"))
        ("mesh-tsdf-pixel-step",
         "Sparse TSDF block allocation pixel stride",
         cxxopts::value<unsigned>()->default_value("4"))
        ("mesh-tsdf-support-closing-axes",
         "Required bilateral support axes before filling a zero-weight voxel",
         cxxopts::value<unsigned>()->default_value("2"))
        ("mesh-tsdf-frame-export-dir",
         "Export exact uint16-mm TSDF depth frames and camera manifest",
         cxxopts::value<std::string>()->default_value(""))
        ("mesh-tsdf-smooth-iters",
         "Boundary-locked TSDF Taubin smoothing iterations (0 disables)",
         cxxopts::value<unsigned>()->default_value("2"))
        ("mesh-tsdf-smooth-lambda", "TSDF Taubin positive coefficient",
         cxxopts::value<float>()->default_value("0.5"))
        ("mesh-tsdf-smooth-mu", "TSDF Taubin negative coefficient",
         cxxopts::value<float>()->default_value("-0.53"))
        ("mesh-dist-insert-px",
         "Minimum projection spacing for global Delaunay (-1 = preset)",
         cxxopts::value<float>()->default_value("-1"))
        ("mesh-free-space-support",
         "OpenMVS weak-surface beta/gamma sink reinforcement",
         cxxopts::value<bool>()->default_value("true"))
        ("mesh-adaptive-sigma",
         "OpenMVS local-density adaptive graph-cut uncertainty",
         cxxopts::value<bool>()->default_value("true"))
        ("mesh-max-edge-scale",
         "Maximum cut-facet edge relative to the cut-facet median",
         cxxopts::value<float>()->default_value("4"))
        ("mesh-free-space-quantile",
         "Quantile used to calibrate fused support to OpenMVS scale",
         cxxopts::value<float>()->default_value("0.95"))
        ("texture-optimize",
         "Refine projected atlas with photara_drender's native optimizer",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("texture-optimize-steps", "Native texture optimization steps",
         cxxopts::value<std::uint32_t>()->default_value("1000"))
        ("texture-optimize-batch-size", "Views per texture optimization step",
         cxxopts::value<std::uint32_t>()->default_value("4"))
        ("texture-seam-samples", "Samples per UV chart seam edge",
         cxxopts::value<std::uint32_t>()->default_value("4"))
        ("patchmatch-tile-rows",
         "Rows per CPU PatchMatch scheduling tile",
         cxxopts::value<unsigned>()->default_value("8"))
        ("patchmatch-concurrent-views",
         "Reference views concurrently sharing the PatchMatch CPU budget",
         cxxopts::value<unsigned>()->default_value("8"))
        ("mesh-obj", "Additionally export mesh as ASCII OBJ",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("dense-quality",
         "MVS quality preset: preview, default, or high",
         cxxopts::value<std::string>()->default_value("default"))
        ("dense-resolution-level",
         "MVS image downscale steps (0=full, 1~=half); unset follows "
         "--dense-quality",
         cxxopts::value<unsigned>())
        ("masks",
         "Valid-region mask directory; black pixels are ignored by default SfM and "
         "downstream stages (auto, - to disable, or explicit path)",
         cxxopts::value<std::string>()->default_value("auto"))
        ("sam-model",
         "SAM 3 GGML checkpoint. Empty uses the Photara model cache",
         cxxopts::value<std::string>()->default_value(""))
        ("sam-backend",
         "SAM 3 backend: auto, cpu, cuda, vulkan, or metal",
         cxxopts::value<std::string>()->default_value("auto"))
        ("sam-text",
         "SAM 3 prompt. Semicolons separate phrases. Generates masks before SfM",
         cxxopts::value<std::string>()->default_value(""))
        ("sam-neg-text",
         "SAM 3 phrases to remove from the mask",
         cxxopts::value<std::string>()->default_value(""))
        ("sam-keep-prompted",
         "Keep the prompted subject (true) or remove the prompted distractors",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("sam-video",
         "Track the prompt across the ordered frames",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("sam-max-size",
         "SAM inference size (0 = checkpoint native); the mask is written at "
         "the source resolution",
         cxxopts::value<int>()->default_value("0"))
        ("sam-refresh",
         "Regenerate SAM masks even when the mask directory already has files",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("texture",
         "UV unwrap + projective texture bake. Implies --mesh unless "
         "--working-mesh is set without --dense/--splat",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("delight",
         "Run Intrinsic delighter before texture bake (implies --texture)",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("atlas-resolution", "Texture atlas resolution (default 2048)",
         cxxopts::value<std::uint32_t>())
        ("uv-parallel-partitions",
         "Spatial UVAtlas partitioning level (1 = serial)",
         cxxopts::value<std::uint32_t>()->default_value("8"));

    const auto result = options.parse(argc, argv);
    if (result.count("version")) {
        std::cout << PHOTARA_VERSION_STRING << '\n';
        std::exit(0);
    }
    if (result.count("help") || argc <= 1) {
        print_help(options);
        std::exit(0);
    }

    if (!result.count("images") || !result.count("output")) {
        throw std::invalid_argument(
            "Missing required options: --images, --output");
    }

    ReconstructCli cli;
    cli.images_dir = utf8_to_path(result["images"].as<std::string>());
    cli.video_fps = result["video-fps"].as<float>();
    cli.video_sharp_window = result["video-sharp-window"].as<int>();
    cli.video_max_frames = result["video-max-frames"].as<int>();
    cli.video_quality = result["video-quality"].as<int>();
    cli.video_scale = result["video-scale"].as<float>();
    cli.video_rotate = result["video-rotate"].as<int>();
    const std::string video_frames_text =
        result["video-frames-dir"].as<std::string>();
    if (!video_frames_text.empty())
        cli.video_frames_dir = utf8_to_path(video_frames_text);
    cli.ffmpeg = utf8_to_path(result["ffmpeg"].as<std::string>());
    if (cli.ffmpeg.empty()) cli.ffmpeg = "ffmpeg";
    cli.video_redo = result["video-redo"].as<bool>();
    if (!std::isfinite(cli.video_fps) || cli.video_fps <= 0.0F)
        throw std::invalid_argument("--video-fps must be positive");
    if (cli.video_sharp_window < 1)
        throw std::invalid_argument("--video-sharp-window must be >= 1");
    if (cli.video_max_frames < 0)
        throw std::invalid_argument("--video-max-frames must be >= 0");
    if (!std::isfinite(cli.video_scale) || cli.video_scale <= 0.0F)
        throw std::invalid_argument("--video-scale must be positive");
    if (cli.video_rotate % 90 != 0)
        throw std::invalid_argument(
            "--video-rotate must be a multiple of 90 degrees");
    const auto camera_model = result["camera-model"].as<std::string>();
    if (camera_model == "fisheye" || camera_model == "opencv_fisheye")
        cli.camera_model = photara::CameraModel::opencv_fisheye;
    else if (camera_model == "equirectangular" || camera_model == "equirect" ||
             camera_model == "panorama" || camera_model == "spherical")
        cli.camera_model = photara::CameraModel::equirectangular;
    else if (camera_model == "auto")
        cli.camera_model = photara::CameraModel::automatic;
    else if (camera_model == "pinhole")
        cli.camera_model = photara::CameraModel::pinhole;
    else
        throw std::invalid_argument(
            "--camera-model must be auto, pinhole, opencv_fisheye or equirectangular");
    cli.focal_pixels = result["focal"].as<double>();
    cli.export_colmap_dir = result["export-colmap"].as<std::string>();
    cli.mode = result["mode"].as<std::string>();
    cli.trust_focal = result["trust-focal"].as<bool>();
    cli.structural_pair_expansion = result["sfm-structural-rescue"].as<bool>();
    if (!std::isfinite(cli.focal_pixels) || cli.focal_pixels < 0.0 ||
        (cli.trust_focal && cli.focal_pixels <= 0.0))
        throw std::invalid_argument(
            "--focal must be finite and nonnegative; --trust-focal requires --focal > 0");
    if (cli.mode != "global" && cli.mode != "incremental" &&
        cli.mode != "hierarchical")
        throw std::invalid_argument(
            "--mode must be global, incremental, or hierarchical");
    const std::string ba_backend = result["ba-backend"].as<std::string>();
    cli.positioning_cuda = result["positioning-cuda"].as<bool>();
    if (ba_backend == "automatic")
        photara::sfm::set_bundle_backend_preference(
            photara::sfm::BundleBackendPreference::automatic);
    else if (ba_backend == "cpu")
        photara::sfm::set_bundle_backend_preference(
            photara::sfm::BundleBackendPreference::cpu);
    else if (ba_backend == "cuda")
        photara::sfm::set_bundle_backend_preference(
            photara::sfm::BundleBackendPreference::cuda);
    else
        throw std::invalid_argument(
            "--ba-backend must be automatic, cpu, or cuda");
    cli.output = utf8_to_path(result["output"].as<std::string>());
    cli.gui = result["gui"].as<bool>();
    const std::string working_sfm_text =
        result["working-sfm"].as<std::string>();
    if (!working_sfm_text.empty())
        cli.working_sfm = utf8_to_path(working_sfm_text);
    const std::string working_splat_text =
        result["working-splat"].as<std::string>();
    if (!working_splat_text.empty())
        cli.working_splat = utf8_to_path(working_splat_text);
    const std::string working_mesh_text =
        result["working-mesh"].as<std::string>();
    if (!working_mesh_text.empty())
        cli.working_mesh = utf8_to_path(working_mesh_text);
    const std::string working_dense_text =
        result["working-dense"].as<std::string>();
    if (!working_dense_text.empty())
        cli.working_dense = utf8_to_path(working_dense_text);
    const std::string working_texture_text =
        result["working-texture"].as<std::string>();
    if (!working_texture_text.empty())
        cli.working_texture = utf8_to_path(working_texture_text);
    const std::string align_live_text = result["align-live"].as<std::string>();
    if (!align_live_text.empty())
        cli.align_live = utf8_to_path(align_live_text);
    const std::string align_preview_text =
        result["align-preview"].as<std::string>();
    if (!align_preview_text.empty())
        cli.align_preview = utf8_to_path(align_preview_text);
    if (result.count("export-mvs") != 0) {
        cli.export_mvs_requested = true;
        const auto export_mvs_text = result["export-mvs"].as<std::string>();
        if (!export_mvs_text.empty())
            cli.export_mvs_path = utf8_to_path(export_mvs_text);
    }
    cli.neighbor_window = result["window"].as<std::size_t>();
    cli.match_ratio = result["match-ratio"].as<float>();
    cli.mutual_check = result["mutual-check"].as<bool>();
    cli.sift_contrast = result["sift-contrast"].as<double>();
    cli.extractor = result["extractor"].as<std::string>();
    cli.matcher = result["matcher"].as<std::string>();
    cli.pipeline = result["pipeline"].as<std::string>();
    cli.max_features = result["max-features"].as<unsigned>();
    const auto extractor_model_text =
        result["extractor-model"].as<std::string>();
    if (!extractor_model_text.empty())
        cli.extractor_model = utf8_to_path(extractor_model_text);
    cli.extractor_width = result["extractor-width"].as<std::uint32_t>();
    cli.extractor_height = result["extractor-height"].as<std::uint32_t>();
    cli.extractor_min_score = result["extractor-min-score"].as<float>();
    cli.extractor_cpu = result["extractor-cpu"].as<bool>();
    const auto lightglue_model_text =
        result["lightglue-model"].as<std::string>();
    if (!lightglue_model_text.empty())
        cli.lightglue_model = utf8_to_path(lightglue_model_text);
    cli.lightglue_extractor = result["lightglue-extractor"].as<std::string>();
    cli.lightglue_width = result["lightglue-width"].as<std::uint32_t>();
    cli.lightglue_height = result["lightglue-height"].as<std::uint32_t>();
    cli.lightglue_min_score = result["lightglue-min-score"].as<float>();
    cli.hybrid_lightglue_max_features =
        result["hybrid-lightglue-max-features"].as<unsigned>();
    cli.lightglue_cpu = result["lightglue-cpu"].as<bool>();
    cli.dense = result["dense"].as<bool>();
    cli.splat = result["splat"].as<bool>();
    cli.splat_view = result["splat-view"].as<bool>();
    cli.capture_mode = result["capture-mode"].as<std::string>();
    if (cli.capture_mode != "object" && cli.capture_mode != "scene")
        throw std::invalid_argument(
            "--capture-mode must be object or scene");
    const std::string subject_bounds_text =
        result["subject-bounds"].as<std::string>();
    if (!subject_bounds_text.empty())
        cli.subject_bounds = utf8_to_path(subject_bounds_text);
    const std::string splat_dataset_text =
        result["splat-dataset"].as<std::string>();
    if (!splat_dataset_text.empty())
        cli.splat_dataset = utf8_to_path(splat_dataset_text);
    const std::string dense_ply_text = result["dense-ply"].as<std::string>();
    if (!dense_ply_text.empty()) cli.dense_ply = utf8_to_path(dense_ply_text);
    cli.splat_iterations = result["splat-iterations"].as<unsigned>();
    cli.splat_log_interval = result["splat-log-interval"].as<unsigned>();
    cli.splat_preview_interval =
        result["splat-preview-interval"].as<unsigned>();
    cli.splat_preview_view = result["splat-preview-view"].as<unsigned>();
    const std::string splat_preview_view_file_text =
        result["splat-preview-view-file"].as<std::string>();
    if (!splat_preview_view_file_text.empty())
        cli.splat_preview_view_file = utf8_to_path(splat_preview_view_file_text);
    const std::string splat_preview_camera_file_text =
        result["splat-preview-camera-file"].as<std::string>();
    if (!splat_preview_camera_file_text.empty())
        cli.splat_preview_camera_file =
            utf8_to_path(splat_preview_camera_file_text);
    const std::string splat_preview_vis_file_text =
        result["splat-preview-vis-file"].as<std::string>();
    if (!splat_preview_vis_file_text.empty())
        cli.splat_preview_vis_file = utf8_to_path(splat_preview_vis_file_text);
    const std::string splat_preview_ack_file_text =
        result["splat-preview-ack-file"].as<std::string>();
    if (!splat_preview_ack_file_text.empty())
        cli.splat_preview_ack_file = utf8_to_path(splat_preview_ack_file_text);
    const std::string splat_preview_dir_text =
        result["splat-preview-dir"].as<std::string>();
    if (!splat_preview_dir_text.empty())
        cli.splat_preview_dir = utf8_to_path(splat_preview_dir_text);
    cli.splat_preview_vk_memory_handle =
        result["splat-preview-vk-memory-handle"].as<std::uint64_t>();
    cli.splat_preview_vk_semaphore_handle =
        result["splat-preview-vk-semaphore-handle"].as<std::uint64_t>();
    cli.splat_preview_vk_allocation_size =
        result["splat-preview-vk-allocation-size"].as<std::uint64_t>();
    cli.splat_preview_vk_width =
        result["splat-preview-vk-width"].as<unsigned>();
    cli.splat_preview_vk_height =
        result["splat-preview-vk-height"].as<unsigned>();
    cli.splat_preview_vk_device_luid =
        result["splat-preview-vk-device-luid"].as<std::uint64_t>();
    cli.splat_preview_vk_device_node_mask =
        result["splat-preview-vk-device-node-mask"].as<unsigned>();
    cli.splat_profile_cuda = result["splat-profile-cuda"].as<bool>();
    cli.splat_fuse_sh_adam = result["splat-fuse-sh-adam"].as<bool>();
    cli.splat_profile_interval =
        result["splat-profile-interval"].as<unsigned>();
    if (cli.splat_profile_interval == 0 ||
        cli.splat_profile_interval > 1'000)
        throw std::invalid_argument(
            "--splat-profile-interval must be in [1, 1000]");
    cli.splat_sh_degree = result["splat-sh-degree"].as<unsigned>();
    if (cli.splat_sh_degree > 3)
        throw std::invalid_argument("--splat-sh-degree must be in [0, 3]");
    cli.splat_max_resolution =
        result["splat-max-resolution"].as<unsigned>();
    cli.splat_undistort = result["splat-undistort"].as<bool>();
    cli.splat_kernel_size = result["splat-kernel-size"].as<float>();
    cli.splat_progressive_resolution =
        result["splat-progressive-resolution"].as<bool>();
    cli.splat_progressive_interval =
        result["splat-progressive-interval"].as<unsigned>();
    cli.splat_progressive_initial_scale =
        result["splat-progressive-initial-scale"].as<float>();
    cli.splat_view_cache_mb =
        result["splat-view-cache-mb"].as<std::uint64_t>();
    cli.splat_device_cache_mb =
        result["splat-device-cache-mb"].as<std::uint64_t>();
    cli.splat_device_cache_max_mb =
        result["splat-device-cache-max-mb"].as<std::uint64_t>();
    cli.splat_async_upload = result["splat-async-upload"].as<bool>();
    cli.splat_cache_auto = result["splat-cache-auto"].as<bool>();
    cli.splat_prefetch_views =
        result["splat-prefetch-views"].as<unsigned>();
    if (cli.splat_prefetch_views > 16)
        throw std::invalid_argument(
            "--splat-prefetch-views must be in [0, 16]");
    cli.splat_prefetch_adaptive = result["splat-prefetch-adaptive"].as<bool>();
    cli.splat_eval_split_every =
        result["splat-eval-split-every"].as<unsigned>();
    cli.splat_use_mask = result["splat-use-mask"].as<bool>();
    cli.splat_alpha_mode = result["splat-alpha-mode"].as<std::string>();
    cli.splat_match_alpha_weight =
        result["splat-match-alpha-weight"].as<float>();
    cli.splat_ssim_weight = result["splat-ssim-weight"].as<float>();
    cli.splat_opacity_reg = result["splat-opacity-reg"].as<float>();
    cli.splat_log_scale_reg = result["splat-log-scale-reg"].as<float>();
    cli.splat_depth_normal_weight =
        result["splat-depth-normal-weight"].as<float>();
    cli.splat_multi_view_depth_bracket =
        result["splat-mv-depth-bracket"].as<float>();
    cli.splat_multi_view_depth_tolerance =
        result["splat-mv-depth-tolerance"].as<float>();
    cli.splat_normal_field = result["splat-normal-field"].as<bool>();
    cli.splat_normal_field_weight =
        result["splat-normal-field-weight"].as<float>();
    cli.splat_normal_field_depth_ratio =
        result["splat-normal-field-depth-ratio"].as<float>();
    cli.splat_normal_field_from_iter =
        result["splat-normal-field-from-iter"].as<unsigned>();
    cli.splat_multi_view_geo_weight =
        result["splat-mv-geo-weight"].as<float>();
    cli.splat_multi_view_ncc_weight =
        result["splat-mv-ncc-weight"].as<float>();
    cli.splat_multi_view_num = result["splat-mv-neighbors"].as<unsigned>();
    cli.splat_multi_view_tail_interval =
        result["splat-mv-tail-interval"].as<unsigned>();
    cli.splat_multi_view_adaptive =
        result["splat-mv-adaptive"].as<bool>();
    cli.splat_multi_view_adaptive_max_interval =
        result["splat-mv-adaptive-max-interval"].as<unsigned>();
    cli.splat_multi_view_stable_refinements =
        result["splat-mv-stable-refinements"].as<unsigned>();
    cli.splat_multi_view_stable_count_threshold =
        result["splat-mv-stable-count-threshold"].as<float>();
    cli.splat_multi_view_stable_churn_threshold =
        result["splat-mv-stable-churn-threshold"].as<float>();
    cli.splat_multi_view_stable_depth_threshold =
        result["splat-mv-stable-depth-threshold"].as<float>();
    cli.splat_multi_view_min_depth_consistency =
        result["splat-mv-min-depth-consistency"].as<float>();
    cli.splat_multi_view_stable_distribution_threshold =
        result["splat-mv-stable-distribution-threshold"].as<float>();
    cli.splat_multi_view_pixel_noise =
        result["splat-mv-pixel-noise"].as<float>();
    cli.splat_geometry_from_iter =
        result["splat-geometry-from-iter"].as<unsigned>();
    cli.splat_min_scale_fraction =
        result["splat-min-scale-fraction"].as<float>();
    cli.splat_max_scale_fraction =
        result["splat-max-scale-fraction"].as<float>();
    cli.splat_max_scale_ratio =
        result["splat-max-scale-ratio"].as<float>();
    cli.splat_constrain_scales = result["splat-constrain-scales"].as<bool>();
    cli.splat_bilateral_grid = result["splat-bilateral-grid"].as<bool>();
    cli.splat_bilateral_grid_shared =
        !result["splat-bilateral-grid-per-view"].as<bool>();
    cli.splat_bilateral_grid_width =
        result["splat-bilateral-grid-width"].as<unsigned>();
    cli.splat_bilateral_grid_height =
        result["splat-bilateral-grid-height"].as<unsigned>();
    cli.splat_bilateral_grid_luma =
        result["splat-bilateral-grid-luma"].as<unsigned>();
    cli.splat_bilateral_grid_lr =
        result["splat-bilateral-grid-lr"].as<float>();
    cli.splat_bilateral_grid_tv =
        result["splat-bilateral-grid-tv"].as<float>();
    cli.splat_ppisp = result["splat-ppisp"].as<bool>();
    cli.splat_ppisp_type = result["splat-ppisp-type"].as<std::string>();
    cli.splat_ppisp_lr = result["splat-ppisp-lr"].as<float>();
    cli.splat_ppisp_before_bilagrid =
        result["splat-ppisp-before-bilagrid"].as<bool>();
    cli.splat_strategy = result["splat-strategy"].as<std::string>();
    cli.splat_densification = result["splat-densification"].as<bool>();
    cli.splat_growth_factor = result["splat-growth-factor"].as<float>();
    if (!std::isfinite(cli.splat_growth_factor) || cli.splat_growth_factor < 0.F)
        throw std::invalid_argument("--splat-growth-factor must be finite and >= 0");
    cli.splat_seed = result["splat-seed"].as<unsigned>();
    cli.splat_structure_freeze_iter =
        result["splat-structure-freeze-iter"].as<unsigned>();
    cli.splat_densification_cap =
        result["splat-densification-cap"].as<std::uint64_t>();
    cli.splat_init_point_budget =
        result["splat-init-point-budget"].as<std::uint64_t>();
    cli.mesh = result["mesh"].as<bool>();
    cli.mvs_mesh_only = result["mvs-mesh-only"].as<bool>();
    const std::string mask_mesh_text = result["mask-mesh"].as<std::string>();
    if (!mask_mesh_text.empty())
        cli.mask_mesh = utf8_to_path(mask_mesh_text);
    cli.mesh_obj = result["mesh-obj"].as<bool>();
    cli.texture = result["texture"].as<bool>();
    cli.delight = result["delight"].as<bool>();
    if (result.count("atlas-resolution") != 0)
        cli.atlas_resolution = result["atlas-resolution"].as<std::uint32_t>();
    cli.uv_parallel_partitions =
        result["uv-parallel-partitions"].as<std::uint32_t>();
    cli.texture_optimize = result["texture-optimize"].as<bool>();
    cli.texture_optimize_steps =
        result["texture-optimize-steps"].as<std::uint32_t>();
    cli.texture_optimize_batch_size =
        result["texture-optimize-batch-size"].as<std::uint32_t>();
    cli.texture_seam_samples =
        result["texture-seam-samples"].as<std::uint32_t>();
    cli.mesh_method = result["mesh-method"].as<std::string>();
    cli.pam_max_points = result["pam-max-points"].as<std::uint64_t>();
    cli.pam_pivot_max_points =
        result["pam-pivot-max-points"].as<std::uint64_t>();
    cli.pam_pivot_std_factor =
        result["pam-pivot-std-factor"].as<float>();
    cli.pam_gaussian_seed_fraction =
        result["pam-gaussian-seed-fraction"].as<float>();
    cli.pam_refinement_steps =
        result["pam-refinement-steps"].as<unsigned>();
    cli.pam_neighbors = result["pam-neighbors"].as<unsigned>();
    cli.pam_points_per_tetrahedron =
        result["pam-points-per-tetrahedron"].as<unsigned>();
    cli.pam_occupancy_iso_value =
        result["pam-occupancy-iso-value"].as<float>();
    cli.pam_vacancy_threshold =
        result["pam-vacancy-threshold"].as<float>();
    cli.pam_mask_background_threshold =
        result["pam-mask-background-threshold"].as<float>();
    cli.mesh_max_points = result["mesh-max-points"].as<std::uint64_t>();
    cli.mesh_target_faces =
        result["mesh-target-faces"].as<std::uint64_t>();
    cli.mesh_remesh = result["mesh-remesh"].as<bool>();
    cli.mesh_tsdf_voxel_scale =
        result["mesh-tsdf-voxel-scale"].as<float>();
    cli.mesh_tsdf_bounds_padding =
        result["mesh-tsdf-bounds-padding"].as<float>();
    cli.mesh_tsdf_pixel_step =
        result["mesh-tsdf-pixel-step"].as<unsigned>();
    cli.mesh_tsdf_support_closing_axes =
        result["mesh-tsdf-support-closing-axes"].as<unsigned>();
    const std::string mesh_tsdf_frame_export_dir =
        result["mesh-tsdf-frame-export-dir"].as<std::string>();
    if (!mesh_tsdf_frame_export_dir.empty())
        cli.mesh_tsdf_frame_export_dir =
            utf8_to_path(mesh_tsdf_frame_export_dir);
    cli.mesh_tsdf_smooth_iters =
        result["mesh-tsdf-smooth-iters"].as<unsigned>();
    cli.mesh_tsdf_smooth_lambda =
        result["mesh-tsdf-smooth-lambda"].as<float>();
    cli.mesh_tsdf_smooth_mu =
        result["mesh-tsdf-smooth-mu"].as<float>();
    cli.mesh_dist_insert_px =
        result["mesh-dist-insert-px"].as<float>();
    cli.mesh_free_space_support =
        result["mesh-free-space-support"].as<bool>();
    cli.mesh_adaptive_sigma = result["mesh-adaptive-sigma"].as<bool>();
    cli.mesh_max_edge_scale = result["mesh-max-edge-scale"].as<float>();
    cli.mesh_free_space_quantile =
        result["mesh-free-space-quantile"].as<float>();
    cli.patchmatch_tile_rows =
        result["patchmatch-tile-rows"].as<unsigned>();
    cli.patchmatch_concurrent_views =
        result["patchmatch-concurrent-views"].as<unsigned>();
    const std::string dense_quality = result["dense-quality"].as<std::string>();
    if (dense_quality == "preview") {
        cli.dense_quality = photara::mvs::DensifyQuality::preview;
    } else if (dense_quality == "default") {
        cli.dense_quality = photara::mvs::DensifyQuality::default_quality;
    } else if (dense_quality == "high") {
        cli.dense_quality = photara::mvs::DensifyQuality::high;
    } else {
        throw std::invalid_argument(
            "--dense-quality must be preview, default, or high");
    }
    if (result.count("dense-resolution-level") != 0)
        cli.dense_resolution_level =
            result["dense-resolution-level"].as<unsigned>();
    const std::string masks_text = result["masks"].as<std::string>();
    cli.masks_auto = masks_text == "auto";
    if (cli.masks_auto) {
        const std::filesystem::path candidate =
            cli.images_dir.parent_path() / "masks";
        if (std::filesystem::is_directory(candidate)) cli.masks_dir = candidate;
    } else if (!masks_text.empty() && masks_text != "-") {
        cli.masks_dir = utf8_to_path(masks_text);
    }
    const std::string sam_model_text = result["sam-model"].as<std::string>();
    if (!sam_model_text.empty())
        cli.sam_model = utf8_to_path(sam_model_text);
    cli.sam_backend = result["sam-backend"].as<std::string>();
    if (cli.sam_backend != "auto" && cli.sam_backend != "cpu" &&
        cli.sam_backend != "cuda" && cli.sam_backend != "vulkan" &&
        cli.sam_backend != "metal")
        throw std::invalid_argument(
            "--sam-backend must be auto, cpu, cuda, vulkan, or metal");
    cli.sam_text = result["sam-text"].as<std::string>();
    cli.sam_negative_text = result["sam-neg-text"].as<std::string>();
    cli.sam_keep_prompted = result["sam-keep-prompted"].as<bool>();
    cli.sam_video = result["sam-video"].as<bool>();
    cli.sam_max_size = result["sam-max-size"].as<int>();
    cli.sam_refresh = result["sam-refresh"].as<bool>();
    if (cli.sam_max_size < 0)
        throw std::invalid_argument("--sam-max-size must be non-negative");
    if (cli.delight) cli.texture = true;
    const bool have_existing_mesh =
        !cli.working_mesh.empty() || !cli.mask_mesh.empty();
    cli.texture_only = cli.texture && have_existing_mesh && !cli.splat &&
                       !cli.dense && result.count("mesh") == 0;
    if (cli.texture && !cli.texture_only) cli.mesh = true;
    if (cli.mesh_obj) cli.mesh = true;
    // An explicitly selected capture mode is the product-level full rebuild
    // preset. Omitting it keeps the low-level SfM-only developer workflow.
    // An explicit --mesh still wins, so a caller can request appearance-only
    // splat training in either capture mode.
    //
    // --dense without an explicit --splat is MVS (dense cloud, and a mesh when
    // --mesh is set). Capture mode then only chooses object bounds versus an
    // unbounded scene. Dense-cloud initialization of 3DGS stays --dense --splat.
    if (result.count("capture-mode") != 0) {
        const bool mvs_request = result.count("dense") != 0 &&
            (result.count("splat") == 0 || !cli.splat);
        if (!mvs_request) cli.splat = true;
        if (!mvs_request && result.count("mesh") == 0) cli.mesh = true;
    }
    if (!cli.mask_mesh.empty()) cli.mvs_mesh_only = true;
    if (cli.mvs_mesh_only) cli.mesh = true;
    // The GGGS-derived path extracts its mesh from learned median depth
    // through TSDF and does not require PatchMatch or an MVS dense cloud.
    if (cli.mesh && !cli.splat) cli.dense = true;
    const bool external_splat_dataset = !cli.splat_dataset.empty();
    if (cli.mvs_mesh_only) {
        if (!external_splat_dataset ||
            (cli.dense_ply.empty() && cli.mask_mesh.empty()))
            throw std::invalid_argument(
                "--mvs-mesh-only requires --splat-dataset and either "
                "--dense-ply or --mask-mesh");
        cli.splat = false;
    } else if ((external_splat_dataset && !cli.dense) ||
               (!external_splat_dataset && !cli.dense_ply.empty())) {
        cli.splat = true;
    }
    // Internal SfM now follows the same sparse initialization path as direct
    // COLMAP/OpenMVS datasets. --dense remains an explicit MVS diagnostic.
    if (external_splat_dataset && cli.splat && cli.texture && !cli.texture_only)
        throw std::invalid_argument(
            "--texture is not yet available in the direct external splat path");
#if !defined(PHOTARA_HAS_SPLAT)
    if (cli.splat || cli.splat_view || external_splat_dataset) {
        throw std::invalid_argument(
            "--splat requires CUDA and PHOTARA_ENABLE_SPLAT=ON");
    }
#endif
    if (!cli.splat_view && cli.splat_iterations == 0)
        throw std::invalid_argument("--splat-iterations must be positive");
    if (cli.splat_view && cli.splat_preview_camera_file.empty())
        throw std::invalid_argument(
            "--splat-view requires --splat-preview-camera-file");
    if (!std::isfinite(cli.splat_kernel_size) ||
        cli.splat_kernel_size < 0.F)
        throw std::invalid_argument(
            "--splat-kernel-size must be finite and non-negative");
    if (cli.splat_alpha_mode != "masked" &&
        cli.splat_alpha_mode != "transparent")
        throw std::invalid_argument(
            "--splat-alpha-mode must be masked or transparent");
    if (cli.splat_ppisp_type != "no_crf_no_vig" &&
        cli.splat_ppisp_type != "no_crf" &&
        cli.splat_ppisp_type != "original")
        throw std::invalid_argument(
            "--splat-ppisp-type must be no_crf_no_vig, no_crf, or original");
    if (cli.splat_bilateral_grid_width == 0 ||
        cli.splat_bilateral_grid_height == 0 ||
        cli.splat_bilateral_grid_luma == 0)
        throw std::invalid_argument(
            "--splat-bilateral-grid-width/height/luma must be positive");
    if (!std::isfinite(cli.splat_bilateral_grid_lr) ||
        cli.splat_bilateral_grid_lr < 0.F)
        throw std::invalid_argument(
            "--splat-bilateral-grid-lr must be finite and non-negative");
    if (!std::isfinite(cli.splat_bilateral_grid_tv) ||
        cli.splat_bilateral_grid_tv < 0.F)
        throw std::invalid_argument(
            "--splat-bilateral-grid-tv must be finite and non-negative");
    if (!std::isfinite(cli.splat_ppisp_lr) || cli.splat_ppisp_lr < 0.F)
        throw std::invalid_argument(
            "--splat-ppisp-lr must be finite and non-negative");
    if (cli.splat_match_alpha_weight < 0.F)
        throw std::invalid_argument(
            "--splat-match-alpha-weight must be non-negative");
    if (cli.splat_ssim_weight < 0.F || cli.splat_ssim_weight > 1.F)
        throw std::invalid_argument("--splat-ssim-weight must be in [0,1]");
    if (!std::isfinite(cli.splat_opacity_reg) || cli.splat_opacity_reg < 0.F ||
        !std::isfinite(cli.splat_log_scale_reg) || cli.splat_log_scale_reg < 0.F)
        throw std::invalid_argument("Splat geometry regularization weights must be finite and non-negative");
    if (cli.splat_depth_normal_weight < 0.F)
        throw std::invalid_argument(
            "--splat-depth-normal-weight must be non-negative");
    if (!std::isfinite(cli.splat_normal_field_weight) ||
        cli.splat_normal_field_weight < 0.F)
        throw std::invalid_argument(
            "--splat-normal-field-weight must be finite and non-negative");
    if (!std::isfinite(cli.splat_normal_field_depth_ratio) ||
        cli.splat_normal_field_depth_ratio < 0.F ||
        cli.splat_normal_field_depth_ratio > 1.F)
        throw std::invalid_argument(
            "--splat-normal-field-depth-ratio must be in [0,1]");
    if (cli.splat_multi_view_geo_weight < 0.F ||
        cli.splat_multi_view_ncc_weight < 0.F)
        throw std::invalid_argument(
            "Splat multi-view loss weights must be non-negative");
    if (cli.splat_multi_view_num == 0)
        throw std::invalid_argument("--splat-mv-neighbors must be positive");
    if (cli.splat_multi_view_tail_interval == 0)
        throw std::invalid_argument(
            "--splat-mv-tail-interval must be positive");
    if (cli.splat_multi_view_adaptive_max_interval == 0 ||
        cli.splat_multi_view_adaptive_max_interval > 16)
        throw std::invalid_argument(
            "--splat-mv-adaptive-max-interval must be in [1,16]");
    if (cli.splat_multi_view_stable_refinements == 0)
        throw std::invalid_argument(
            "--splat-mv-stable-refinements must be positive");
    const auto valid_unit_threshold = [](const float value) {
        return std::isfinite(value) && value >= 0.F && value <= 1.F;
    };
    if (!valid_unit_threshold(
            cli.splat_multi_view_stable_count_threshold) ||
        !valid_unit_threshold(
            cli.splat_multi_view_stable_churn_threshold) ||
        !valid_unit_threshold(
            cli.splat_multi_view_stable_depth_threshold) ||
        !valid_unit_threshold(
            cli.splat_multi_view_min_depth_consistency) ||
        !valid_unit_threshold(
            cli.splat_multi_view_stable_distribution_threshold))
        throw std::invalid_argument(
            "Splat adaptive multi-view thresholds must be finite in [0,1]");
    if (!(cli.splat_multi_view_pixel_noise > 0.F))
        throw std::invalid_argument("--splat-mv-pixel-noise must be positive");
    if (cli.splat_min_scale_fraction <= 0.F ||
        cli.splat_max_scale_fraction < cli.splat_min_scale_fraction)
        throw std::invalid_argument(
            "Splat scale fractions must satisfy 0 < min <= max");
    if (cli.splat_max_scale_ratio != 0.F && cli.splat_max_scale_ratio < 1.F)
        throw std::invalid_argument(
            "--splat-max-scale-ratio must be 0 or >= 1");
    if (cli.splat_strategy != "adc_plus" &&
        cli.splat_strategy != "adc_igs" &&
        cli.splat_strategy != "emc")
        throw std::invalid_argument(
            "--splat-strategy must be adc_plus, adc_igs or emc");
    if (cli.splat_densification_cap == 0)
        throw std::invalid_argument(
            "--splat-densification-cap must be positive");
#if !defined(PHOTARA_HAS_TEXTURE)
    if (cli.texture || cli.delight) {
        throw std::invalid_argument(
            "--texture/--delight require a build with PHOTARA_ENABLE_TEXTURE "
            "(Vulkan SDK + photara_drender)");
    }
#endif
    if (cli.atlas_resolution && *cli.atlas_resolution < 64) {
        throw std::invalid_argument("--atlas-resolution must be >= 64");
    }
    if (cli.uv_parallel_partitions == 0)
        throw std::invalid_argument(
            "--uv-parallel-partitions must be positive");
    if (cli.texture_optimize &&
        (cli.texture_optimize_steps == 0 ||
         cli.texture_optimize_batch_size == 0))
        throw std::invalid_argument(
            "--texture-optimize-steps and --texture-optimize-batch-size must "
            "be positive when optimization is enabled");
    if (cli.mesh_method != "auto" && cli.mesh_method != "tsdf" &&
        cli.mesh_method != "delaunay" && cli.mesh_method != "pam") {
        throw std::invalid_argument(
            "--mesh-method must be auto, tsdf, delaunay, or pam");
    }
    if (cli.mesh_method == "pam" && !cli.splat)
        throw std::invalid_argument(
            "--mesh-method pam requires splat training");
    if (cli.pam_max_points < 4 || cli.pam_pivot_max_points < 4 ||
        cli.pam_neighbors == 0 ||
        cli.pam_points_per_tetrahedron == 0 ||
        !std::isfinite(cli.pam_pivot_std_factor) ||
        !(cli.pam_pivot_std_factor > 0.F) ||
        !std::isfinite(cli.pam_gaussian_seed_fraction) ||
        cli.pam_gaussian_seed_fraction < 0.F ||
        cli.pam_gaussian_seed_fraction > 1.F ||
        !std::isfinite(cli.pam_occupancy_iso_value) ||
        !(cli.pam_occupancy_iso_value > 0.F &&
          cli.pam_occupancy_iso_value < 1.F) ||
        !std::isfinite(cli.pam_vacancy_threshold) ||
        cli.pam_vacancy_threshold < 0.F ||
        !std::isfinite(cli.pam_mask_background_threshold) ||
        cli.pam_mask_background_threshold < 0.F ||
        cli.pam_mask_background_threshold > 1.F)
        throw std::invalid_argument("Invalid PAM extraction options");
    if (cli.patchmatch_tile_rows == 0)
        throw std::invalid_argument("--patchmatch-tile-rows must be positive");
    if (cli.patchmatch_concurrent_views == 0)
        throw std::invalid_argument(
            "--patchmatch-concurrent-views must be positive");
    if (cli.mesh_dist_insert_px < -1.F || cli.mesh_dist_insert_px > 16.F)
        throw std::invalid_argument(
            "--mesh-dist-insert-px must be -1 or in [0,16]");
    if (!std::isfinite(cli.mesh_max_edge_scale) ||
        cli.mesh_max_edge_scale < 0.F)
        throw std::invalid_argument(
            "--mesh-max-edge-scale must be finite and >= 0");
    if (cli.mesh_target_faces > 0 && cli.mesh_target_faces < 4)
        throw std::invalid_argument(
            "--mesh-target-faces must be 0 or at least 4");
    if (cli.mesh_tsdf_voxel_scale != -1.F &&
        (!(cli.mesh_tsdf_voxel_scale > 0.F) ||
         !std::isfinite(cli.mesh_tsdf_voxel_scale)))
        throw std::invalid_argument(
            "--mesh-tsdf-voxel-scale must be -1 or positive");
    if (!(cli.mesh_tsdf_bounds_padding >= 1.F) ||
        !std::isfinite(cli.mesh_tsdf_bounds_padding))
        throw std::invalid_argument(
            "--mesh-tsdf-bounds-padding must be finite and >= 1");
    if (cli.mesh_tsdf_pixel_step == 0 || cli.mesh_tsdf_pixel_step > 16)
        throw std::invalid_argument(
            "--mesh-tsdf-pixel-step must be in [1,16]");
    if (cli.mesh_tsdf_support_closing_axes > 3)
        throw std::invalid_argument(
            "--mesh-tsdf-support-closing-axes must be in [0,3]");
    if (!std::isfinite(cli.mesh_tsdf_smooth_lambda) ||
        cli.mesh_tsdf_smooth_lambda < 0.F ||
        cli.mesh_tsdf_smooth_lambda > 1.F)
        throw std::invalid_argument(
            "--mesh-tsdf-smooth-lambda must be in [0,1]");
    if (!std::isfinite(cli.mesh_tsdf_smooth_mu) ||
        cli.mesh_tsdf_smooth_mu < -1.F ||
        cli.mesh_tsdf_smooth_mu > 0.F)
        throw std::invalid_argument(
            "--mesh-tsdf-smooth-mu must be in [-1,0]");

    const auto cache_text = result["cache-dir"].as<std::string>();
    if (!cache_text.empty() && cache_text != "-")
        cli.cache_dir = utf8_to_path(cache_text);

    if (cli.focal_pixels < 0.0)
        throw std::invalid_argument("--focal must be >= 0");
    if (cli.mode != "incremental" && cli.mode != "hierarchical" &&
        cli.mode != "global") {
        throw std::invalid_argument(
            "--mode must be incremental, hierarchical, or global");
    }
    if (cli.match_ratio <= 0.F || cli.match_ratio > 1.F)
        throw std::invalid_argument("--match-ratio must be in (0, 1]");
    if (cli.neighbor_window == 0)
        throw std::invalid_argument("--window must be positive");
    if (cli.sift_contrast <= 0.0)
        throw std::invalid_argument("--sift-contrast must be positive");
    if (cli.max_features == 0U)
        throw std::invalid_argument("--max-features must be positive");

    if (cli.pipeline == "none") cli.pipeline.clear();
    if (!cli.pipeline.empty() && cli.pipeline != "lightglue_end2end")
        throw std::invalid_argument(
            "--pipeline must be empty/none or lightglue_end2end");
    if (!cli.pipeline.empty() &&
        (cli.matcher == "lightglue" ||
         cli.matcher == "hybrid_lightglue"))
        throw std::invalid_argument(
            "Use either a LightGlue matcher or --pipeline lightglue_end2end");

    const bool need_lg_model =
        cli.pipeline == "lightglue_end2end" || cli.matcher == "lightglue" ||
        cli.matcher == "hybrid_lightglue";
    if (need_lg_model) {
        if (cli.lightglue_model.empty())
            throw std::invalid_argument(
                "--lightglue-model is required for a LightGlue matcher or "
                "pipeline lightglue_end2end");
        if (cli.lightglue_width == 0U || cli.lightglue_height == 0U)
            throw std::invalid_argument(
                "--lightglue-width/height must be positive");
        if (cli.lightglue_min_score < 0.F || cli.lightglue_min_score > 1.F)
            throw std::invalid_argument(
                "--lightglue-min-score must be in [0, 1]");
    }
    if (cli.pipeline == "lightglue_end2end") {
        if (cli.lightglue_extractor != "disk" &&
            cli.lightglue_extractor != "superpoint")
            throw std::invalid_argument(
                "--lightglue-extractor must be disk or superpoint");
    }
    if (cli.matcher == "lightglue") {
        if (cli.extractor != "superpoint" && cli.extractor != "disk" &&
            cli.extractor != "aliked" && cli.extractor != "sift" &&
            cli.extractor != "siftgpu")
            throw std::invalid_argument(
                "--matcher lightglue accepts superpoint, disk, aliked, sift, "
                "or siftgpu descriptors");
        if ((cli.extractor == "superpoint" || cli.extractor == "disk" ||
             cli.extractor == "aliked") && cli.extractor_model.empty())
            throw std::invalid_argument(
                "--extractor superpoint|disk|aliked requires --extractor-model");
    }
    if (cli.matcher == "hybrid_lightglue" && cli.extractor != "siftgpu")
        throw std::invalid_argument(
            "--matcher hybrid_lightglue requires --extractor siftgpu");
    if (cli.extractor == "superpoint" || cli.extractor == "disk" ||
        cli.extractor == "aliked") {
        if (cli.extractor_model.empty() && cli.pipeline.empty())
            throw std::invalid_argument(
                "--extractor superpoint|disk|aliked requires --extractor-model");
        if (cli.extractor != "aliked" &&
            (cli.extractor_width == 0U || cli.extractor_height == 0U))
            throw std::invalid_argument(
                "--extractor-width/height must be positive");
    }
    if (cli.extractor_min_score < -1.F || cli.extractor_min_score > 1.F)
        throw std::invalid_argument(
            "--extractor-min-score must be -1 or in [0, 1]");
    return cli;
}

std::string lower_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](const unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension;
}

#if defined(PHOTARA_HAS_SPLAT)
bool is_gaussian_output(const std::filesystem::path& path) {
    const auto extension = lower_extension(path);
    return extension == ".sog" || extension == ".spz" || extension == ".glb";
}

std::filesystem::path splat_file_path(const ReconstructCli& cli) {
    if (cli.gui && !cli.working_splat.empty()) return cli.working_splat;
    if (is_gaussian_output(cli.output)) return cli.output;
    const auto parent = cli.output.parent_path().empty()
        ? std::filesystem::current_path()
        : cli.output.parent_path();
    return parent / (cli.output.stem().string() + "_splat.ply");
}

void apply_cli_subject_bounds(
    const ReconstructCli& cli, photara::mvs::MvsScene& scene) {
    if (!cli.subject_bounds.empty()) {
        if (!photara::mvs::load_subject_bounds(
                scene.subject_bounds, cli.subject_bounds))
            throw std::runtime_error(
                "Failed to load SubjectBounds: " +
                cli.subject_bounds.string());
        photara::core::Logger::instance().info(
            "subject_bounds=", cli.subject_bounds.string(), " source=file");
        return;
    }
    if (cli.capture_mode == "scene") scene.subject_bounds = {};
}

void run_splat_view(
    const ReconstructCli& cli, photara::project::Archive& archive) {
    const bool has_vulkan_preview =
        cli.splat_preview_vk_memory_handle != 0 &&
        cli.splat_preview_vk_semaphore_handle != 0 &&
        cli.splat_preview_vk_allocation_size != 0 &&
        cli.splat_preview_vk_width != 0 &&
        cli.splat_preview_vk_height != 0;
    if (!has_vulkan_preview)
        throw std::invalid_argument(
            "--splat-view requires Vulkan preview handles");

    photara::splat::GaussianModel model;
    std::filesystem::path model_path;
    std::error_code exists_error;
    std::vector<std::filesystem::path> candidates;
    if (!cli.working_splat.empty())
        candidates.push_back(cli.working_splat);
    if (is_gaussian_output(cli.output))
        candidates.push_back(cli.output);
    const std::filesystem::path parent = cli.output.parent_path().empty()
        ? std::filesystem::current_path()
        : cli.output.parent_path();
    const std::string stem = cli.output.stem().string() + "_splat";
    candidates.push_back(parent / (stem + ".sog"));
    candidates.push_back(parent / (stem + ".spz"));
    candidates.push_back(parent / (stem + ".glb"));
    candidates.push_back(parent / (stem + ".ply"));
    for (const auto& candidate : candidates) {
        if (!std::filesystem::exists(candidate, exists_error)) continue;
        model_path = candidate;
        model = photara::splat::load_gaussians(model_path);
        photara::core::Logger::instance().info(
            "splat_view_model=", model_path, " gaussians=", model.size());
        break;
    }
    if (model_path.empty()) {
        if (!archive.has(photara::project::ChunkType::gaussians))
            throw std::runtime_error(
                "No trained splat model found for --splat-view");
        model = photara::splat::decode_gaussians(
            archive.chunk(photara::project::ChunkType::gaussians));
        photara::core::Logger::instance().info(
            "splat_view_model=ascan gaussians=", model.size());
    }
    if (model.size() == 0)
        throw std::runtime_error("--splat-view loaded an empty Gaussian model");

    auto vulkan_preview = std::make_unique<photara::splat::CudaVulkanPreview>(
        photara::splat::CudaVulkanPreviewOptions{
            cli.splat_preview_vk_memory_handle,
            cli.splat_preview_vk_semaphore_handle,
            cli.splat_preview_vk_allocation_size,
            cli.splat_preview_vk_width,
            cli.splat_preview_vk_height,
            cli.splat_preview_vk_device_luid,
            cli.splat_preview_vk_device_node_mask});
    photara::core::Logger::instance().info(
        "splat_view_transport=cuda_vulkan_external_memory extent=",
        cli.splat_preview_vk_width, 'x', cli.splat_preview_vk_height);
    photara::splat::run_orbit_preview(
        model, cli.splat_preview_camera_file,
        [&vulkan_preview](
            unsigned, std::size_t, const photara::splat::Camera& camera,
            const tinytensor::Tensor& color) {
            vulkan_preview->submit(color, camera.width, camera.height);
        },
        cli.splat_kernel_size, cli.splat_preview_vis_file);
}
#endif

photara::io::VideoExtractOptions video_extract_options(
    const ReconstructCli& cli) {
    photara::io::VideoExtractOptions options;
    options.video = cli.images_dir;
    options.output_dir = cli.video_frames_dir.empty()
        ? photara::io::default_video_frames_dir(cli.images_dir)
        : cli.video_frames_dir;
    options.ffmpeg = cli.ffmpeg;
    options.fps = cli.video_fps;
    options.sharp_window = cli.video_sharp_window;
    options.max_frames = cli.video_max_frames;
    options.quality = cli.video_quality;
    options.scale = cli.video_scale;
    options.rotate = cli.video_rotate;
    options.resume = !cli.video_redo;
    return options;
}

void adopt_video_frames(
    ReconstructCli& cli, const std::filesystem::path& frames_dir) {
    cli.video_source = cli.images_dir;
    cli.images_dir = frames_dir;
}

bool is_still_image(const std::filesystem::path& path) {
    const auto extension = lower_extension(path);
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
           extension == ".tif" || extension == ".tiff";
}

std::vector<std::filesystem::path> list_still_images(
    const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> files;
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) return files;
    for (const auto& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (error || !entry.is_regular_file() || !is_still_image(entry.path()))
            continue;
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

bool directory_has_mask(const std::filesystem::path& directory) {
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) return false;
    for (const auto& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (error || !entry.is_regular_file()) continue;
        const auto extension = lower_extension(entry.path());
        if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
            return true;
    }
    return false;
}

void refresh_auto_masks(ReconstructCli& cli) {
    if (!cli.masks_auto) return;
    cli.masks_dir.clear();
    const auto candidate = cli.images_dir.parent_path() / "masks";
    std::error_code error;
    if (std::filesystem::is_directory(candidate, error))
        cli.masks_dir = candidate;
}

void ensure_sam_masks(ReconstructCli& cli) {
    if (cli.sam_text.empty()) return;
    auto output = cli.masks_dir.empty()
        ? cli.images_dir.parent_path() / "masks"
        : cli.masks_dir;
    if (!cli.sam_refresh && directory_has_mask(output)) {
        cli.masks_dir = output;
        photara::core::Logger::instance().info(
            "sam_masks=reused dir=", output);
        return;
    }
    photara::sam::GenerateOptions options;
    options.model = cli.sam_model;
    options.backend = cli.sam_backend;
    options.output_dir = output;
    options.images = list_still_images(cli.images_dir);
    options.text = cli.sam_text;
    options.negative_text = cli.sam_negative_text;
    options.keep_prompted = cli.sam_keep_prompted;
    options.video = cli.sam_video;
    options.max_size = cli.sam_max_size;
    if (options.images.size() < 1)
        throw std::runtime_error(
            "SAM mask generation needs images in " + cli.images_dir.string());
    photara::sam::generate_masks(options);
    cli.masks_dir = output;
}

void ensure_video_frames(ReconstructCli& cli, const bool preserve_cameras) {
    if (!photara::io::is_video_path(cli.images_dir)) return;
    auto options = video_extract_options(cli);
    if (preserve_cameras) {
        if (cli.video_redo) {
            throw std::runtime_error(
                "Cannot redo video extraction while reusing an existing "
                "alignment. Re-run Align Photos.");
        }
        if (photara::io::has_matching_video_extract(options)) {
            adopt_video_frames(cli, options.output_dir);
            photara::core::Logger::instance().info(
                "video_frames_dir=", options.output_dir, " reused=true");
            return;
        }
        throw std::runtime_error(
            "Video extract settings do not match the existing alignment. "
            "Re-run Align Photos before Train 3DGS or Dense MVS.");
    }
    const auto result = photara::io::extract_video_frames(options);
    adopt_video_frames(cli, result.image_dir);
    photara::core::Logger::instance().info(
        "video_frames_dir=", result.image_dir,
        " frames=", result.frames_written,
        " reused=", result.reused ? "true" : "false");
}

photara::project::Settings settings_from_cli(const ReconstructCli& cli) {
    photara::project::Settings settings;
    settings.name = cli.output.stem().string();
    settings.image_directory =
        cli.video_source.empty() ? cli.images_dir : cli.video_source;
    settings.dataset_source = cli.splat_dataset;
    settings.dataset_format = "auto";
    settings.dataset_initial_cloud = cli.dense_ply;
#if defined(PHOTARA_HAS_SPLAT)
    settings.splat_output_format =
        photara::splat::gaussian_format_name(
            photara::splat::gaussian_format_from_path(cli.output));
#endif
    settings.video_frames_dir =
        cli.video_source.empty() ? cli.video_frames_dir : cli.images_dir;
    settings.video_fps = cli.video_fps;
    settings.video_sharp_window = cli.video_sharp_window;
    settings.video_max_frames = cli.video_max_frames;
    settings.video_quality = cli.video_quality;
    settings.video_scale = cli.video_scale;
    settings.video_rotate = cli.video_rotate;
    settings.camera_model = static_cast<int>(cli.camera_model);
    if (cli.mode == "incremental") settings.sfm_mode = 1;
    else if (cli.mode == "hierarchical") settings.sfm_mode = 2;
    settings.max_features = cli.max_features;
    settings.scene_mode = cli.capture_mode == "scene";
    settings.iterations = static_cast<int>(cli.splat_iterations);
    settings.preview_interval = static_cast<int>(cli.splat_preview_interval);
    if (cli.splat_strategy == "adc_plus") settings.strategy = 1;
    else settings.strategy = 0;
    settings.max_resolution = static_cast<int>(cli.splat_max_resolution);
    settings.progressive_resolution = cli.splat_progressive_resolution;
    settings.use_mask = cli.splat_use_mask;
    settings.mask_mode = cli.splat_alpha_mode == "transparent" ? 1 : 0;
    settings.build_mesh = cli.mesh;
    settings.mesh_source = (cli.dense && cli.mesh && !cli.splat) ? 1 : 0;
    if (cli.mesh_method == "tsdf") settings.mesh_method = 1;
    else if (cli.mesh_method == "pam") settings.mesh_method = 2;
    else settings.mesh_method = 0;
    settings.depth_normal_weight = cli.splat_depth_normal_weight;
    settings.multi_view_geo_weight = cli.splat_multi_view_geo_weight;
    settings.multi_view_ncc_weight = cli.splat_multi_view_ncc_weight;
    settings.geometry_from_iter = static_cast<int>(cli.splat_geometry_from_iter);
    settings.normal_field = cli.splat_normal_field;
    settings.ppisp_layout = cli.splat_ppisp ? 1 : 0;
    settings.bilateral_grid = cli.splat_bilateral_grid;
    const std::uint32_t atlas_resolution =
        cli.atlas_resolution.value_or(kDefaultAtlasResolution);
    settings.atlas_resolution = static_cast<int>(atlas_resolution);
    settings.texture_delight = cli.delight;
    settings.texture_optimize = cli.texture_optimize;
    if (atlas_resolution <= 1024) settings.texture_quality = 0;
    else if (atlas_resolution >= 4096) settings.texture_quality = 2;
    else settings.texture_quality = 1;
    return settings;
}

void save_working_sfm(
    const ReconstructCli& cli, const photara::sfm::Scene& scene) {
    if (cli.working_sfm.empty() || scene.registered_count() < 2) return;
    std::error_code error;
    std::filesystem::create_directories(cli.working_sfm.parent_path(), error);
    photara::sfm::AsfmOptions options;
    options.path_base = cli.images_dir;
    photara::sfm::save_asfm(scene, cli.working_sfm, options);
    photara::core::Logger::instance().info(
        "working_sfm=", cli.working_sfm,
        " images=", scene.images.size(),
        " registered=", scene.registered_count());
}

void ensure_artifact_parent(const std::filesystem::path& path) {
    if (path.empty() || path.parent_path().empty()) return;
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
}

std::filesystem::path artifact_path(
    const ReconstructCli& cli, const std::filesystem::path& working,
    const std::filesystem::path& sidecar) {
    return cli.gui ? working : sidecar;
}

void write_dense_artifact(
    const ReconstructCli& cli, const photara::mvs::DenseCloud& cloud,
    const std::filesystem::path& sidecar) {
    const auto path = artifact_path(cli, cli.working_dense, sidecar);
    if (path.empty()) return;
    ensure_artifact_parent(path);
    photara::mvs::save_dense_ply(cloud, path);
    photara::core::Logger::instance().info(
        "dense_ply=", path, " points=", cloud.points.size());
}

void write_mesh_artifact(
    const ReconstructCli& cli, const photara::mvs::Mesh& mesh,
    const std::filesystem::path& sidecar, const bool write_obj = false) {
    const auto path = artifact_path(cli, cli.working_mesh, sidecar);
    if (path.empty() || mesh.faces.empty()) return;
    ensure_artifact_parent(path);
    photara::mvs::save_mesh_ply(mesh, path);
    if (!cli.gui && write_obj)
        photara::mvs::save_mesh_obj(
            mesh, path.parent_path() / (path.stem().string() + ".obj"));
    photara::core::Logger::instance().info(
        "mesh_ply=", path,
        " vertices=", mesh.vertices.size(),
        " faces=", mesh.faces.size());
}

#if defined(PHOTARA_HAS_TEXTURE)
photara::texture::TextureOptions texture_options_from_cli(
    const ReconstructCli& cli) {
    photara::texture::TextureOptions options;
    options.atlas_resolution =
        cli.atlas_resolution.value_or(options.atlas_resolution);
    options.uv_parallel_partitions = cli.uv_parallel_partitions;
    options.optimize = cli.texture_optimize;
    options.optimize_steps = cli.texture_optimize_steps;
    options.optimize_batch_size = cli.texture_optimize_batch_size;
    options.seam_samples_per_edge = cli.texture_seam_samples;
    options.delight = cli.delight;
    options.mask_dir = cli.masks_dir;
    return options;
}

photara::mvs::Mesh load_texture_source_mesh(
    const ReconstructCli& cli, const photara::project::Archive& archive) {
    std::error_code error;
    if (!cli.working_mesh.empty() &&
        std::filesystem::exists(cli.working_mesh, error))
        return photara::mvs::load_mesh_ply(cli.working_mesh);
    if (!cli.mask_mesh.empty() &&
        std::filesystem::exists(cli.mask_mesh, error))
        return photara::mvs::load_mesh_ply(cli.mask_mesh);
    if (archive.has(photara::project::ChunkType::mesh))
        return photara::mvs::decode_mesh(
            archive.chunk(photara::project::ChunkType::mesh));
    const auto parent = cli.output.parent_path().empty()
        ? std::filesystem::current_path()
        : cli.output.parent_path();
    const std::string stem = cli.output.stem().string();
    const std::array<std::filesystem::path, 3> sidecars = {
        parent / (stem + "_mesh.ply"),
        parent / (stem + "_splat_mesh.ply"),
        parent / (stem + "_mvs_mesh.ply")};
    for (const auto& candidate : sidecars)
        if (std::filesystem::exists(candidate, error))
            return photara::mvs::load_mesh_ply(candidate);
    throw std::runtime_error(
        "Bake Texture needs a reconstructed mesh (--working-mesh or a mesh "
        "already in the project)");
}

void write_texture_artifact(
    const ReconstructCli& cli, photara::mvs::MvsScene& scene,
    const std::filesystem::path& sidecar_stem,
    photara::project::Archive* archive, const bool write_project) {
    if (scene.mesh.faces.empty())
        throw std::runtime_error("Cannot bake texture for an empty mesh");
    auto options = texture_options_from_cli(cli);
    if (!cli.texture_only) {
        if (cli.dense_quality == photara::mvs::DensifyQuality::high) {
            options.blend_mode =
                photara::texture::BlendMode::weighted_average;
            options.visibility_mode =
                photara::texture::VisibilityMode::hybrid_ray_query;
        } else if (
            cli.dense_quality == photara::mvs::DensifyQuality::preview) {
            if (!cli.atlas_resolution)
                options.atlas_resolution = 1024U;
            options.visibility_mode =
                photara::texture::VisibilityMode::shadow_map;
        }
    }
    const auto stem = artifact_path(cli, cli.working_texture, sidecar_stem);
    if (stem.empty())
        throw std::runtime_error("No textured-mesh output path");
    ensure_artifact_parent(stem);
    const auto started = std::chrono::steady_clock::now();
    photara::texture::bake_and_export(scene, stem, options);
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    photara::core::Logger::instance().info(
        "textured_obj=", photara::texture::textured_obj_path(stem),
        " delight=", cli.delight,
        " atlas=", options.atlas_resolution,
        " optimized=", options.optimize,
        " texture_s=", elapsed);
    if (write_project && archive != nullptr) {
        archive->set_chunk(
            photara::project::ChunkType::texture,
            photara::texture::encode_textured_obj(stem));
        archive->save(cli.output);
        photara::core::Logger::instance().info(
            "ascan_texture=", cli.output);
    }
}

void bake_texture_only(
    const ReconstructCli& cli, photara::mvs::MvsScene& scene,
    photara::project::Archive& archive, const bool write_project) {
    scene.mesh = load_texture_source_mesh(cli, archive);
    const auto parent = cli.output.parent_path().empty()
        ? std::filesystem::current_path()
        : cli.output.parent_path();
    write_texture_artifact(
        cli, scene, parent / (cli.output.stem().string() + "_textured"),
        write_project ? &archive : nullptr, write_project);
}
#endif

bool load_working_sfm(
    const ReconstructCli& cli, photara::sfm::Scene& scene) {
    std::error_code error;
    if (cli.working_sfm.empty() ||
        !std::filesystem::exists(cli.working_sfm, error))
        return false;
    photara::sfm::AsfmOptions options;
    options.path_base = cli.images_dir;
    auto loaded = photara::sfm::load_asfm(cli.working_sfm, options);
    if (loaded.registered_count() < 2) return false;
    scene = std::move(loaded);
    photara::core::Logger::instance().info(
        "working_sfm_loaded=", cli.working_sfm,
        " images=", scene.images.size(),
        " registered=", scene.registered_count(),
        " tracks=", scene.tracks.size());
    return true;
}

void ensure_scene_appearance(
    const ReconstructCli& cli, photara::sfm::Scene& scene) {
    if (photara::sfm::triangulated_tracks_have_color(scene)) return;
    const std::size_t colored =
        photara::sfm::colour_triangulated_tracks(scene);
    if (colored > 0) save_working_sfm(cli, scene);
}

void save_ply(const photara::sfm::Scene& scene, const std::filesystem::path& path) {
    std::size_t count = 0;
    for (const auto& track : scene.tracks) {
        if (track.is_triangulated()) ++count;
    }
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Failed to create PLY: " + path.string());
    output << "ply\nformat ascii 1.0\nelement vertex " << count
           << "\nproperty float x\nproperty float y\nproperty float z\n"
           << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           << "end_header\n";

    // Sample each landmark from its inlier observations so the editor can
    // show the reconstruction in the photos' own colours instead of a ramp.
    std::unordered_map<photara::sfm::Index, std::unique_ptr<photara::io::RgbImage>>
        rgb_cache;
    const auto cached_rgb =
        [&](const photara::sfm::Index image_id) -> const photara::io::RgbImage* {
        const auto existing = rgb_cache.find(image_id);
        if (existing != rgb_cache.end()) return existing->second.get();
        try {
            auto image = std::make_unique<photara::io::RgbImage>(
                photara::io::load_rgb(scene.images[image_id].path));
            const photara::io::RgbImage* pointer = image.get();
            rgb_cache.emplace(image_id, std::move(image));
            return pointer;
        } catch (...) {
            rgb_cache.emplace(image_id, nullptr);
            return nullptr;
        }
    };

    for (const auto& track : scene.tracks) {
        if (!track.is_triangulated()) continue;
        if (track.has_color) {
            output << track.position.x() << ' ' << track.position.y() << ' '
                   << track.position.z() << ' '
                   << static_cast<int>(track.color_r) << ' '
                   << static_cast<int>(track.color_g) << ' '
                   << static_cast<int>(track.color_b) << '\n';
            continue;
        }
        double sum_r = 0.0;
        double sum_g = 0.0;
        double sum_b = 0.0;
        std::size_t samples = 0;
        const std::size_t inliers = std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        for (std::size_t i = 0; i < inliers; ++i) {
            const auto& observation = track.observations[i];
            if (observation.image_id >= scene.images.size()) continue;
            const auto& image = scene.images[observation.image_id];
            if (!image.registered ||
                observation.feature_id >= image.features.keypoints.size())
                continue;
            const auto* rgb = cached_rgb(observation.image_id);
            if (!rgb || rgb->width == 0 || rgb->height == 0) continue;
            const auto& keypoint = image.features.keypoints[observation.feature_id];
            const int xi = static_cast<int>(std::lround(keypoint.x));
            const int yi = static_cast<int>(std::lround(keypoint.y));
            if (xi < 0 || yi < 0 ||
                xi >= static_cast<int>(rgb->width) ||
                yi >= static_cast<int>(rgb->height))
                continue;
            const std::size_t offset =
                (static_cast<std::size_t>(yi) * rgb->width +
                 static_cast<std::size_t>(xi)) *
                3;
            if (offset + 2 >= rgb->pixels.size()) continue;
            sum_r += rgb->pixels[offset + 0];
            sum_g += rgb->pixels[offset + 1];
            sum_b += rgb->pixels[offset + 2];
            ++samples;
        }
        const int red = samples > 0
            ? static_cast<int>(std::lround(sum_r / static_cast<double>(samples)))
            : 200;
        const int green = samples > 0
            ? static_cast<int>(std::lround(sum_g / static_cast<double>(samples)))
            : 200;
        const int blue = samples > 0
            ? static_cast<int>(std::lround(sum_b / static_cast<double>(samples)))
            : 200;
        output << track.position.x() << ' ' << track.position.y() << ' '
               << track.position.z() << ' ' << red << ' ' << green << ' '
               << blue << '\n';
    }
}

double percentile(std::vector<double> values, const double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = std::clamp(fraction, 0.0, 1.0) *
                            static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double alpha = position - static_cast<double>(lower);
    return values[lower] * (1.0 - alpha) + values[upper] * alpha;
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped{"\""};
    for (const char character : value) {
        if (character == '"') escaped += '"';
        escaped += character;
    }
    escaped += '"';
    return escaped;
}

std::filesystem::path write_sfm_diagnostics(
    const photara::sfm::Scene& scene,
    const std::filesystem::path& reconstruction_path) {
    using photara::sfm::Vec2;
    using photara::sfm::Vec3;
    using photara::sfm::Mat3;

    struct ImageStats {
        std::vector<double> errors;
        double squared_sum{0.0};
    };
    struct GraphStats {
        unsigned active_pairs{0};
        unsigned cycle_supported_pairs{0};
        unsigned cycle_inconsistent_pairs{0};
        unsigned degenerate_pairs{0};
        double composite_weight_sum{0.0};
        double maximum_composite_weight{0.0};
        double weighted_ray_angle_sum{0.0};
    };
    std::vector<ImageStats> stats(scene.images.size());
    std::vector<GraphStats> graph_stats(scene.images.size());
    const auto alignment_observability =
        photara::sfm::analyze_alignment_observability(scene);
    std::array<std::vector<double>, 5> radial_errors;
    std::array<double, 5> radial_signed_sum{};

    for (const auto& pair : scene.pairs) {
        if (!pair.active || !pair.relative_pose.has_value() ||
            pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size())
            continue;
        const double weight = (std::max)(
            0.0, static_cast<double>(pair.composite_weight()));
        for (const auto image_id : {pair.id1, pair.id2}) {
            auto& graph = graph_stats[image_id];
            ++graph.active_pairs;
            if (pair.weight_triplet > 0.F) ++graph.cycle_supported_pairs;
            if (pair.weight_cycle < 0.999F) ++graph.cycle_inconsistent_pairs;
            if (pair.degenerate_planar) ++graph.degenerate_pairs;
            graph.composite_weight_sum += weight;
            graph.maximum_composite_weight =
                (std::max)(graph.maximum_composite_weight, weight);
            graph.weighted_ray_angle_sum +=
                weight * static_cast<double>(pair.mean_ray_angle);
        }
    }

    for (const auto& track : scene.tracks) {
        if (!track.is_triangulated() || !track.position.allFinite()) continue;
        const std::size_t inlier_count = std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        for (std::size_t i = 0; i < inlier_count; ++i) {
            const auto& observation = track.observations[i];
            if (observation.image_id >= scene.images.size()) continue;
            const auto& image = scene.images[observation.image_id];
            if (!image.registered || image.camera_id >= scene.cameras.size() ||
                observation.feature_id >= image.features.keypoints.size())
                continue;
            const auto& camera = scene.cameras[image.camera_id];
            const Vec3 camera_point =
                image.pose.transform_world_to_camera(track.position);
            if (!camera_point.allFinite() || camera_point.z() <= 0.0) continue;
            const Vec2 projected = camera.project(camera_point);
            const auto& keypoint = image.features.keypoints[observation.feature_id];
            const Vec2 measured(keypoint.x, keypoint.y);
            const Vec2 residual = projected - measured;
            const double error = residual.norm();
            if (!std::isfinite(error)) continue;
            stats[observation.image_id].errors.push_back(error);
            stats[observation.image_id].squared_sum += error * error;

            const Vec2 normalized(
                (measured.x() - camera.cx) / camera.fx,
                (measured.y() - camera.cy) / camera.fy);
            const double radius = normalized.norm();
            const std::size_t bin = std::min<std::size_t>(
                4, static_cast<std::size_t>(radius / 0.2));
            radial_errors[bin].push_back(error);
            if (radius > 1e-8) {
                const Vec2 radial_pixel(
                    normalized.x() * camera.fx,
                    normalized.y() * camera.fy);
                radial_signed_sum[bin] += residual.dot(radial_pixel.normalized());
            }
        }
    }

    auto csv_path = reconstruction_path.parent_path() /
                    (reconstruction_path.stem().string() +
                     "_sfm_diagnostics.csv");
    std::ofstream output(csv_path);
    if (!output)
        throw std::runtime_error(
            "Failed to create SfM diagnostics: " + csv_path.string());
    output << std::setprecision(12)
           << "image_id,name,registered,camera_id,width,height,fx,fy,cx,cy,"
              "k1,k2,p1,p2,center_x,center_y,center_z,qw,qx,qy,qz,"
              "observations,reprojection_mean_px,reprojection_rms_px,"
              "reprojection_p95_px,reprojection_max_px,previous_center_step,"
              "previous_rotation_deg,alignment_reliable,active_pairs,cycle_supported_pairs,"
              "cycle_inconsistent_pairs,degenerate_pairs,pair_weight_sum,"
              "pair_weight_max,weighted_mean_ray_angle_deg,camera_model\n";

    std::vector<std::tuple<double, std::size_t, double, double>> worst_images;
    std::vector<double> trajectory_steps;
    std::vector<double> rotation_steps;
    const double radians_to_degrees = 180.0 / 3.14159265358979323846;
    for (std::size_t image_index = 0; image_index < scene.images.size(); ++image_index) {
        const auto& image = scene.images[image_index];
        const auto& image_stats = stats[image_index];
        const bool camera_valid = image.camera_id < scene.cameras.size();
        const auto* camera = camera_valid ? &scene.cameras[image.camera_id] : nullptr;
        const std::size_t count = image_stats.errors.size();
        double sum = 0.0;
        for (const double error : image_stats.errors) sum += error;
        const double mean = count == 0 ? 0.0 : sum / static_cast<double>(count);
        const double rms = count == 0
            ? 0.0
            : std::sqrt(image_stats.squared_sum / static_cast<double>(count));
        const double p95 = percentile(image_stats.errors, 0.95);
        const double maximum = image_stats.errors.empty()
            ? 0.0
            : *std::max_element(image_stats.errors.begin(), image_stats.errors.end());
        worst_images.emplace_back(p95, image_index, rms, maximum);

        double center_step = std::numeric_limits<double>::quiet_NaN();
        double rotation_step = std::numeric_limits<double>::quiet_NaN();
        if (image_index > 0 && image.registered &&
            scene.images[image_index - 1].registered) {
            const auto& previous = scene.images[image_index - 1];
            center_step = (image.pose.C - previous.pose.C).norm();
            const double cosine = std::clamp(
                0.5 * ((image.pose.R * previous.pose.R.transpose()).trace() - 1.0),
                -1.0, 1.0);
            rotation_step = std::acos(cosine) * radians_to_degrees;
            trajectory_steps.push_back(center_step);
            rotation_steps.push_back(rotation_step);
        }
        const auto quaternion = image.pose.quaternion();
        const auto& graph = graph_stats[image_index];
        const double mean_ray_angle_deg = graph.composite_weight_sum > 0.0
            ? graph.weighted_ray_angle_sum / graph.composite_weight_sum *
                  radians_to_degrees
            : 0.0;
        output << image_index << ','
               << csv_escape(image.path.filename().string()) << ','
               << (image.registered ? 1 : 0) << ',' << image.camera_id << ','
               << (camera ? camera->width : 0) << ','
               << (camera ? camera->height : 0) << ','
               << (camera ? camera->fx : 0.0) << ','
               << (camera ? camera->fy : 0.0) << ','
               << (camera ? camera->cx : 0.0) << ','
               << (camera ? camera->cy : 0.0) << ','
               << (camera ? camera->k1 : 0.0) << ','
               << (camera ? camera->k2 : 0.0) << ','
               << (camera ? camera->p1 : 0.0) << ','
               << (camera ? camera->p2 : 0.0) << ','
               << image.pose.C.x() << ',' << image.pose.C.y() << ','
               << image.pose.C.z() << ',' << quaternion.w() << ','
               << quaternion.x() << ',' << quaternion.y() << ','
               << quaternion.z() << ',' << count << ',' << mean << ',' << rms
               << ',' << p95 << ',' << maximum << ',' << center_step << ','
               << rotation_step << ','
               << static_cast<unsigned>(
                      alignment_observability.reliable[image_index]) << ','
               << graph.active_pairs << ','
               << graph.cycle_supported_pairs << ','
               << graph.cycle_inconsistent_pairs << ','
               << graph.degenerate_pairs << ',' << graph.composite_weight_sum
               << ',' << graph.maximum_composite_weight << ','
               << mean_ray_angle_deg << ','
               << (camera && camera->model == photara::CameraModel::opencv_fisheye
                       ? "opencv_fisheye"
                       : (camera && camera->model == photara::CameraModel::equirectangular
                              ? "equirectangular"
                              : "pinhole")) << '\n';
    }

    const auto pair_csv_path = reconstruction_path.parent_path() /
        (reconstruction_path.stem().string() + "_sfm_pairs.csv");
    std::ofstream pair_output(pair_csv_path);
    if (!pair_output)
        throw std::runtime_error(
            "Failed to create SfM pair diagnostics: " +
            pair_csv_path.string());
    pair_output << std::setprecision(12)
        << "image_id1,name1,image_id2,name2,active,inliers,"
           "composite_weight,spatial_weight,geometry_weight,"
           "connectivity_weight,triplet_weight,cycle_weight,"
           "mean_ray_angle_deg,degenerate_planar,zero_baseline,"
           "center_distance,direction_error_deg,rotation_error_deg\n";
    for (const auto& pair : scene.pairs) {
        if (pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size())
            continue;
        const auto& first = scene.images[pair.id1];
        const auto& second = scene.images[pair.id2];
        double center_distance = std::numeric_limits<double>::quiet_NaN();
        double direction_error = std::numeric_limits<double>::quiet_NaN();
        double rotation_error = std::numeric_limits<double>::quiet_NaN();
        if (first.registered && second.registered &&
            pair.relative_pose.has_value()) {
            const Vec3 actual = second.pose.C - first.pose.C;
            const Vec3 expected = -(
                second.pose.R.transpose() *
                pair.relative_pose->translation());
            center_distance = actual.norm();
            if (center_distance > 1e-12 && expected.norm() > 1e-12) {
                direction_error = std::acos(std::clamp(
                    actual.normalized().dot(expected.normalized()), -1.0, 1.0)) *
                    radians_to_degrees;
            }
            const Mat3 rotation_delta =
                (second.pose.R * first.pose.R.transpose()) *
                pair.relative_pose->R.transpose();
            rotation_error = std::acos(std::clamp(
                0.5 * (rotation_delta.trace() - 1.0), -1.0, 1.0)) *
                radians_to_degrees;
        }
        pair_output << pair.id1 << ','
            << csv_escape(first.path.filename().string()) << ','
            << pair.id2 << ',' << csv_escape(second.path.filename().string())
            << ',' << (pair.active ? 1 : 0) << ',' << pair.num_inliers() << ','
            << pair.composite_weight() << ',' << pair.weight_spatial << ','
            << pair.weight_geometry << ',' << pair.weight_connectivity << ','
            << pair.weight_triplet << ',' << pair.weight_cycle << ','
            << static_cast<double>(pair.mean_ray_angle) * radians_to_degrees
            << ',' << (pair.degenerate_planar ? 1 : 0) << ','
            << (pair.zero_baseline ? 1 : 0) << ',' << center_distance << ','
            << direction_error << ',' << rotation_error << '\n';
    }

    std::sort(worst_images.begin(), worst_images.end(), std::greater<>());
    const std::size_t reported = std::min<std::size_t>(8, worst_images.size());
    for (std::size_t i = 0; i < reported; ++i) {
        const auto [p95, image_index, rms, maximum] = worst_images[i];
        photara::core::Logger::instance().info(
            "sfm audit worst[", i, "] image=",
            scene.images[image_index].path.filename(), " observations=",
            stats[image_index].errors.size(), " rms_px=", rms,
            " p95_px=", p95, " max_px=", maximum);
    }
    photara::core::Logger::instance().info(
        "sfm audit trajectory: step_median=", percentile(trajectory_steps, 0.5),
        " step_p95=", percentile(trajectory_steps, 0.95),
        " rotation_median_deg=", percentile(rotation_steps, 0.5),
        " rotation_p95_deg=", percentile(rotation_steps, 0.95));
    photara::core::Logger::instance().info(
        "sfm audit alignment observability: reliable=",
        alignment_observability.reliable_views,
        " unreliable=", alignment_observability.unreliable_views,
        " constraint_edges=", alignment_observability.constraint_edges,
        " bridges=", alignment_observability.bridge_edges);
    for (std::size_t bin = 0; bin < radial_errors.size(); ++bin) {
        const std::size_t count = radial_errors[bin].size();
        photara::core::Logger::instance().info(
            "sfm audit radius_bin=", bin, " observations=", count,
            " mean_abs_px=",
            count == 0 ? 0.0
                       : [&] {
                             double sum = 0.0;
                             for (const double value : radial_errors[bin]) sum += value;
                             return sum / static_cast<double>(count);
                         }(),
            " p95_px=", percentile(radial_errors[bin], 0.95),
            " mean_signed_radial_px=",
            count == 0 ? 0.0
                       : radial_signed_sum[bin] / static_cast<double>(count));
    }
    return csv_path;
}

#if defined(PHOTARA_HAS_MESH_TOOLS)
void retain_largest_edge_component(
    std::vector<float>& positions, std::vector<std::uint32_t>& indices) {
    const std::size_t face_count = indices.size() / 3;
    const std::size_t vertex_count = positions.size() / 3;
    if (face_count == 0 || vertex_count == 0) return;

    std::vector<std::uint32_t> parent(face_count);
    std::vector<std::uint32_t> sizes(face_count, 1U);
    for (std::size_t face = 0; face < face_count; ++face)
        parent[face] = static_cast<std::uint32_t>(face);
    const auto find_root = [&](std::uint32_t value) {
        std::uint32_t root = value;
        while (parent[root] != root) root = parent[root];
        while (parent[value] != value) {
            const std::uint32_t next = parent[value];
            parent[value] = root;
            value = next;
        }
        return root;
    };
    const auto unite = [&](std::uint32_t a, std::uint32_t b) {
        a = find_root(a);
        b = find_root(b);
        if (a == b) return;
        if (sizes[a] < sizes[b]) std::swap(a, b);
        parent[b] = a;
        sizes[a] += sizes[b];
    };
    const auto edge_key = [](std::uint32_t a, std::uint32_t b) {
        if (a > b) std::swap(a, b);
        return (static_cast<std::uint64_t>(a) << 32U) |
            static_cast<std::uint64_t>(b);
    };

    std::unordered_map<std::uint64_t, std::uint32_t> edge_owner;
    edge_owner.reserve(face_count * 2);
    for (std::size_t face = 0; face < face_count; ++face) {
        const auto face_id = static_cast<std::uint32_t>(face);
        const std::uint32_t a = indices[face * 3];
        const std::uint32_t b = indices[face * 3 + 1];
        const std::uint32_t c = indices[face * 3 + 2];
        for (const auto edge : {edge_key(a, b), edge_key(b, c), edge_key(c, a)}) {
            const auto [found, inserted] = edge_owner.emplace(edge, face_id);
            if (!inserted) unite(face_id, found->second);
        }
    }
    std::uint32_t largest_root = find_root(0U);
    for (std::size_t face = 1; face < face_count; ++face) {
        const auto root = find_root(static_cast<std::uint32_t>(face));
        if (sizes[root] > sizes[largest_root]) largest_root = root;
    }
    const std::size_t retained_faces = sizes[largest_root];
    if (retained_faces == face_count) return;

    constexpr std::uint32_t unused = (std::numeric_limits<std::uint32_t>::max)();
    std::vector<std::uint32_t> remap(vertex_count, unused);
    std::vector<float> compact_positions;
    std::vector<std::uint32_t> compact_indices;
    compact_positions.reserve(positions.size());
    compact_indices.reserve(retained_faces * 3);
    for (std::size_t face = 0; face < face_count; ++face) {
        if (find_root(static_cast<std::uint32_t>(face)) != largest_root) continue;
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const std::uint32_t old_index = indices[face * 3 + corner];
            auto& new_index = remap[old_index];
            if (new_index == unused) {
                new_index = static_cast<std::uint32_t>(compact_positions.size() / 3);
                compact_positions.insert(
                    compact_positions.end(),
                    positions.begin() + static_cast<std::ptrdiff_t>(old_index) * 3,
                    positions.begin() + static_cast<std::ptrdiff_t>(old_index) * 3 + 3);
            }
            compact_indices.push_back(new_index);
        }
    }
    photara::core::Logger::instance().info(
        "instant mesh component clean: faces=", face_count, " -> ",
        compact_indices.size() / 3, " vertices=", vertex_count, " -> ",
        compact_positions.size() / 3);
    positions = std::move(compact_positions);
    indices = std::move(compact_indices);
}

bool repair_and_decimate_mesh(
    photara::mvs::Mesh& mesh, const std::uint64_t target_faces,
    const bool use_instant_remesh) {
    if (target_faces == 0 || mesh.faces.empty()) return false;
    if (mesh.faces.size() <= target_faces) {
        photara::core::Logger::instance().info(
            "mesh target postprocess skipped: input_faces=", mesh.faces.size(),
            " target_faces=", target_faces);
        return false;
    }
    if (mesh.vertices.size() >
        static_cast<std::size_t>(
            (std::numeric_limits<std::uint32_t>::max)()))
        throw std::runtime_error(
            "photara mesh preprocessing requires 32-bit vertex indices");

    photara::core::StageScope stage("mesh.photara_repair_decimate");
    std::vector<float> positions;
    positions.reserve(mesh.vertices.size() * 3);
    for (const auto& vertex : mesh.vertices) {
        positions.push_back(vertex.x());
        positions.push_back(vertex.y());
        positions.push_back(vertex.z());
    }
    std::vector<std::uint32_t> indices;
    indices.reserve(mesh.faces.size() * 3);
    for (const auto& face : mesh.faces) {
        if (face.minCoeff() < 0 ||
            face.maxCoeff() >= static_cast<int>(mesh.vertices.size()))
            continue;
        indices.push_back(static_cast<std::uint32_t>(face[0]));
        indices.push_back(static_cast<std::uint32_t>(face[1]));
        indices.push_back(static_cast<std::uint32_t>(face[2]));
    }

    if (use_instant_remesh && photara_mesh::has_instant_meshes_backend()) {
        photara::core::StageScope remesh_stage("mesh.instant_remesh");
        photara_mesh::RemeshOptions remesh_options;
        // PoSy=4 produces approximately one quad per requested face; the
        // binding triangulates each regular quad for the downstream pipeline.
        const std::uint64_t instant_faces =
            (std::max<std::uint64_t>)(4, (target_faces + 1) / 2);
        remesh_options.face_count = static_cast<int>(std::min<std::uint64_t>(
            instant_faces,
            static_cast<std::uint64_t>((std::numeric_limits<int>::max)())));
        remesh_options.deterministic = true;
        try {
            auto remeshed = photara_mesh::remesh_field_aligned(
                positions, indices, remesh_options);
            photara::core::Logger::instance().info(
                "instant mesh: vertices=", positions.size() / 3, " -> ",
                remeshed.positions.size() / 3, " faces=", indices.size() / 3,
                " -> ", remeshed.indices.size() / 3);
            positions = std::move(remeshed.positions);
            indices = std::move(remeshed.indices);
            retain_largest_edge_component(positions, indices);
        } catch (const std::exception& error) {
            photara::core::Logger::instance().warning(
                "Instant Meshes failed; continuing with CGAL repair: ",
                error.what());
        }
        remesh_stage.finish();
    } else if (use_instant_remesh) {
        photara::core::Logger::instance().warning(
            "Instant Meshes backend unavailable; continuing with CGAL repair");
    }

    auto result = photara_mesh::repair_and_decimate(
        positions, indices,
        photara_mesh::DecimateOptions{
            static_cast<std::size_t>(target_faces), false});
    retain_largest_edge_component(result.positions, result.indices);
    photara::mvs::Mesh processed;
    processed.vertices.reserve(result.positions.size() / 3);
    for (std::size_t vertex = 0; vertex < result.positions.size() / 3;
         ++vertex) {
        processed.vertices.emplace_back(
            result.positions[vertex * 3], result.positions[vertex * 3 + 1],
            result.positions[vertex * 3 + 2]);
    }
    processed.faces.reserve(result.indices.size() / 3);
    for (std::size_t face = 0; face < result.indices.size() / 3; ++face) {
        processed.faces.emplace_back(
            static_cast<int>(result.indices[face * 3]),
            static_cast<int>(result.indices[face * 3 + 1]),
            static_cast<int>(result.indices[face * 3 + 2]));
    }
    processed.normals.assign(
        processed.vertices.size(), photara::mvs::Vec3f::Zero());
    for (const auto& face : processed.faces) {
        const auto normal =
            (processed.vertices[static_cast<std::size_t>(face[1])] -
             processed.vertices[static_cast<std::size_t>(face[0])])
                .cross(
                    processed.vertices[static_cast<std::size_t>(face[2])] -
                    processed.vertices[static_cast<std::size_t>(face[0])]);
        for (int slot = 0; slot < 3; ++slot)
            processed.normals[static_cast<std::size_t>(face[slot])] += normal;
    }
    for (auto& normal : processed.normals)
        if (normal.squaredNorm() > 1e-12F) normal.normalize();

    photara::core::Logger::instance().info(
        "photara mesh: vertices=", mesh.vertices.size(), " -> ",
        processed.vertices.size(), " faces=", mesh.faces.size(), " -> ",
        processed.faces.size(), " target_faces=", target_faces);
    mesh = std::move(processed);
    stage.finish();
    return true;
}
#endif

#if defined(PHOTARA_HAS_SPLAT)
std::optional<photara::mvs::Mesh> run_splat_training(
    const photara::mvs::MvsScene& scene,
    const ReconstructCli& cli,
    const bool dense_input,
    const photara::mvs::DensifyOptions* mesh_options = nullptr,
    const std::filesystem::path& generated_mask_dir = {},
    photara::project::Archive* project_archive = nullptr) {
    photara::splat::TrainingOptions options;
    options.iterations = cli.splat_iterations;
    options.log_interval = cli.splat_log_interval;
    options.preview_interval = cli.splat_preview_interval;
    options.preview_view_index = cli.splat_preview_view;
    options.preview_view_file = cli.splat_preview_view_file;
    options.preview_camera_file = cli.splat_preview_camera_file;
    options.preview_vis_file = cli.splat_preview_vis_file;
    options.preview_ack_file = cli.splat_preview_ack_file;
    options.sh_degree = cli.splat_sh_degree;
    options.input_is_dense = dense_input;
    options.initialize_scale_from_knn = true;
    options.use_source_resolution = true;
    options.undistort_to_pinhole = cli.splat_undistort;
    options.max_image_dimension = cli.splat_max_resolution;
    options.kernel_size = cli.splat_kernel_size;
    options.progressive_resolution = cli.splat_progressive_resolution;
    options.progressive_resolution_interval =
        cli.splat_progressive_interval;
    options.progressive_initial_scale =
        std::clamp(cli.splat_progressive_initial_scale, 1e-3F, 1.F);
    options.evaluation_split_every =
        cli.gui ? 0U : cli.splat_eval_split_every;
    constexpr std::uint64_t bytes_per_megabyte = 1024ULL * 1024ULL;
    options.training_view_cache_bytes = static_cast<std::size_t>(
        std::min<std::uint64_t>(
            cli.splat_view_cache_mb >
                    (std::numeric_limits<std::uint64_t>::max)() /
                        bytes_per_megabyte
                ? (std::numeric_limits<std::uint64_t>::max)()
                : cli.splat_view_cache_mb * bytes_per_megabyte,
            (std::numeric_limits<std::size_t>::max)()));
    options.training_device_cache_bytes = static_cast<std::size_t>(
        std::min<std::uint64_t>(
            cli.splat_device_cache_mb,
            (std::numeric_limits<std::size_t>::max)() / bytes_per_megabyte)
        * bytes_per_megabyte);
    options.training_device_cache_max_bytes =
        cli.splat_device_cache_max_mb == 0
        ? 0
        : static_cast<std::size_t>(
              std::min<std::uint64_t>(
                  cli.splat_device_cache_max_mb,
                  (std::numeric_limits<std::size_t>::max)() /
                      bytes_per_megabyte) *
              bytes_per_megabyte);
    options.training_async_upload = cli.splat_async_upload;
    options.adaptive_training_cache = cli.splat_cache_auto;
    options.training_prefetch_views = cli.splat_prefetch_views;
    options.training_prefetch_adaptive = cli.splat_prefetch_adaptive;
    if (!cli.gui) {
        for (const unsigned milestone :
             {1'000U, 5'000U, 10'000U, 15'000U, 30'000U})
            if (milestone < options.iterations)
                options.evaluation_iterations.push_back(milestone);
        if (std::find(
                options.evaluation_iterations.begin(),
                options.evaluation_iterations.end(),
                options.iterations) == options.evaluation_iterations.end())
            options.evaluation_iterations.push_back(options.iterations);
    }
    options.densification_cap = static_cast<std::size_t>(
        std::min<std::uint64_t>(
            cli.splat_densification_cap,
            (std::numeric_limits<std::size_t>::max)()));
    options.initial_point_budget = static_cast<std::size_t>(
        std::min<std::uint64_t>(
            cli.splat_init_point_budget,
            (std::numeric_limits<std::size_t>::max)()));
    if (cli.splat_strategy == "adc_plus")
        options.densification_strategy =
            photara::splat::DensificationStrategy::adc_plus;
    else if (cli.splat_strategy == "emc")
        options.densification_strategy =
            photara::splat::DensificationStrategy::emc;
    else
        options.densification_strategy =
            photara::splat::DensificationStrategy::adc_igs;
    // ADC+ and ADC-IGS share the Brush learning rates, initial opacity and
    // growth thresholds. Apply user overrides after the strategy preset.
    photara::splat::apply_strategy_defaults(options);
    if (cli.splat_growth_factor > 0.F)
        options.densify_growth_factor = cli.splat_growth_factor;
    options.seed = cli.splat_seed;
    options.enable_densification = cli.splat_densification && !dense_input;
    options.structure_freeze_iter = cli.splat_structure_freeze_iter;
    options.use_bilateral_grid = cli.splat_bilateral_grid;
    options.bilateral_grid_shared = cli.splat_bilateral_grid_shared;
    options.bilateral_grid_width = cli.splat_bilateral_grid_width;
    options.bilateral_grid_height = cli.splat_bilateral_grid_height;
    options.bilateral_grid_luma = cli.splat_bilateral_grid_luma;
    options.bilateral_grid_lr = cli.splat_bilateral_grid_lr;
    options.bilateral_grid_tv_weight = cli.splat_bilateral_grid_tv;
    options.use_ppisp = cli.splat_ppisp;
    options.ppisp_type = cli.splat_ppisp_type == "original"
        ? photara::splat::PpispParamType::original
        : cli.splat_ppisp_type == "no_crf"
            ? photara::splat::PpispParamType::no_crf
            : photara::splat::PpispParamType::no_crf_no_vig;
    options.ppisp_lr = cli.splat_ppisp_lr;
    options.ppisp_before_bilagrid = cli.splat_ppisp_before_bilagrid;
    if (!dense_input &&
        (photara::splat::is_adc_strategy(options.densification_strategy) ||
         options.densification_strategy ==
             photara::splat::DensificationStrategy::emc)) {
        // Remaining Brush optimizer settings, shared by ADC+ and ADC-IGS so
        // the strategies differ only in densification. The means/opacity
        // learning rates and initial opacity are set above.
        options.scales_lr = 5e-3F;
        options.quaternions_lr = 2e-3F;
        options.sh0_lr = 2e-3F;
        options.sh_rest_lr = 2e-4F;
        options.beta1 = 0.9F;
        options.beta2 = 0.999F;
        // Brush explicitly constructs AdamScaled with epsilon=1e-15; the
        // means scheduler decays by 100x to 2e-7 over the configured run.
        options.adam_epsilon = 1e-15F;
        // Keep Brush's SH schedule. Both strategies honor the same explicit
        // progressive-resolution settings loaded from the CLI above.
        options.sh_degree_interval = 0;
        options.background_noise_strength = 0.1F;
    }
    const std::size_t projected_mask_views =
        static_cast<std::size_t>(std::count_if(
            scene.views.begin(), scene.views.end(),
            [](const photara::mvs::MvsView& view) {
                return view.foreground_mask.size() ==
                    static_cast<std::size_t>(view.width) * view.height;
            }));
    const bool all_views_have_projected_masks =
        !scene.views.empty() &&
        projected_mask_views == scene.views.size();
    if (projected_mask_views != 0) {
        std::uint64_t projected_mask_pixels = 0;
        std::uint64_t projected_mask_nonzero = 0;
        std::uint64_t projected_mask_foreground = 0;
        std::uint64_t projected_mask_sum = 0;
        for (const auto& view : scene.views) {
            if (view.foreground_mask.size() !=
                static_cast<std::size_t>(view.width) * view.height)
                continue;
            projected_mask_pixels += view.foreground_mask.size();
            for (const std::uint8_t value : view.foreground_mask) {
                projected_mask_nonzero += value != 0;
                projected_mask_foreground += value > 127;
                projected_mask_sum += value;
            }
        }
        const double inverse_pixels = projected_mask_pixels == 0
            ? 0.0
            : 1.0 / static_cast<double>(projected_mask_pixels);
        photara::core::Logger::instance().info(
            "splat projected mask audit: pixels=", projected_mask_pixels,
            " nonzero_fraction=",
            projected_mask_nonzero * inverse_pixels,
            " foreground_fraction=",
            projected_mask_foreground * inverse_pixels,
            " mean_coverage=",
            projected_mask_sum * inverse_pixels / 255.0);
    }
    // Respect an explicit no-mask training request. Automatically generated
    // masks are training inputs only and never constrain TSDF independently.
    options.use_mask = cli.splat_use_mask &&
        (all_views_have_projected_masks || !generated_mask_dir.empty() ||
         !cli.masks_dir.empty());
    options.mask_dir = generated_mask_dir.empty()
        ? cli.masks_dir
        : generated_mask_dir;
    options.alpha_mode = cli.splat_alpha_mode == "masked"
        ? photara::splat::AlphaMode::masked
        : photara::splat::AlphaMode::transparent;
    options.match_alpha_weight = cli.splat_match_alpha_weight;
    options.ssim_weight = cli.splat_ssim_weight;
    options.opacity_regularization_weight = cli.splat_opacity_reg;
    options.log_scale_regularization_weight = cli.splat_log_scale_reg;
    options.use_normal_field = cli.splat_normal_field;
    options.normal_field_weight = cli.splat_normal_field_weight;
    options.normal_field_depth_ratio =
        cli.splat_normal_field_depth_ratio;
    options.normal_field_from_iter = cli.splat_normal_field_from_iter;
    options.profile_cuda = cli.splat_profile_cuda;
    options.fuse_sh_adam = cli.splat_fuse_sh_adam;
    options.cuda_profile_interval = cli.splat_profile_interval;
    options.minimum_scale_fraction = cli.splat_min_scale_fraction;
    options.maximum_scale_fraction = cli.splat_max_scale_fraction;
    // pygsplat leaves Gaussian scales unconstrained for dense point-cloud
    // initialization. Retain the safety clamp only when explicitly requested;
    // forcing it for every dense input clips tangential splats and removes
    // legitimate surface coverage in sparsely sampled detail regions.
    options.constrain_scale_range = cli.splat_constrain_scales;
    // Every strategy now shares the CLI clamp: ADC+/ADC-IGS follow Brush and
    // leave the axis ratio unconstrained unless the user asks for a limit.
    options.max_scale_ratio = cli.splat_max_scale_ratio;
    options.use_mvs_depth = false;
    options.use_mvs_normals = false;
    options.use_depth_normal_loss = cli.mesh &&
        cli.splat_depth_normal_weight > 0.F;
    options.depth_normal_weight = cli.splat_depth_normal_weight;
    options.multi_view_geo_weight = cli.mesh
        ? cli.splat_multi_view_geo_weight
        : 0.F;
    options.multi_view_ncc_weight = cli.mesh
        ? cli.splat_multi_view_ncc_weight
        : 0.F;
    options.multi_view_depth_bracket = cli.splat_multi_view_depth_bracket;
    options.multi_view_depth_tolerance = cli.splat_multi_view_depth_tolerance;
    options.multi_view_num = cli.splat_multi_view_num;
    options.multi_view_tail_interval =
        cli.splat_multi_view_tail_interval;
    if (cli.mesh && cli.splat_multi_view_tail_interval > 1)
        photara::core::Logger::instance().warning(
            "splat_mv_tail_interval=", cli.splat_multi_view_tail_interval,
            " meshing=1: multi-view geometry supervision runs every ",
            cli.splat_multi_view_tail_interval,
            " steps, which lowers TSDF mesh quality. Keep interval=1 when the "
            "mesh is the deliverable.");
    options.multi_view_adaptive_frequency =
        cli.splat_multi_view_adaptive;
    options.multi_view_adaptive_max_interval =
        cli.splat_multi_view_adaptive_max_interval;
    options.multi_view_adaptive_stable_refinements =
        cli.splat_multi_view_stable_refinements;
    options.multi_view_adaptive_count_threshold =
        cli.splat_multi_view_stable_count_threshold;
    options.multi_view_adaptive_churn_threshold =
        cli.splat_multi_view_stable_churn_threshold;
    options.multi_view_adaptive_depth_threshold =
        cli.splat_multi_view_stable_depth_threshold;
    options.multi_view_adaptive_min_depth_consistency =
        cli.splat_multi_view_min_depth_consistency;
    options.multi_view_adaptive_distribution_threshold =
        cli.splat_multi_view_stable_distribution_threshold;
    options.multi_view_pixel_noise_threshold =
        cli.splat_multi_view_pixel_noise;
    options.depth_normal_from_iter = cli.splat_geometry_from_iter;
    // Keep structural parameters trainable while GGGS geometry supervision is
    // active. The 3k schedule now matches pygsplat's complete loss stack.
    if ((options.use_depth_normal_loss ||
         options.multi_view_geo_weight > 0.F ||
         options.multi_view_ncc_weight > 0.F) && dense_input)
        options.dense_structure_freeze_iter = 0;
    const char* effective_strategy = !options.enable_densification
        ? "disabled"
        : cli.splat_strategy.c_str();
    photara::core::Logger::instance().info(
        "splat training: iterations=", options.iterations,
        " log_interval=", options.log_interval,
        " input=", dense_input ? "dense_points" : "sparse_points",
        " input_points=", scene.dense_cloud.points.size(),
        " densification_strategy=", effective_strategy,
        " densification_enabled=", options.enable_densification,
        " initial_point_budget=", options.initial_point_budget,
        " structure_freeze_iter=", options.structure_freeze_iter,
        " grow_stop_iter=", options.grow_stop_iter,
        " opacity_decay=", options.opacity_decay,
        " opacity_reg=", options.opacity_regularization_weight,
        " log_scale_reg=", options.log_scale_regularization_weight,
        " scale_decay=", options.scale_decay,
        " bilateral_grid=", options.use_bilateral_grid,
        " bilateral_grid_shared=", options.bilateral_grid_shared,
        " bilateral_grid_shape=", options.bilateral_grid_width, 'x',
        options.bilateral_grid_height, 'x', options.bilateral_grid_luma,
        " bilateral_grid_lr=", options.bilateral_grid_lr,
        " bilateral_grid_tv=", options.bilateral_grid_tv_weight,
        " bilateral_grid_deviation_limit=",
        options.bilateral_grid_deviation_limit,
        " bilateral_grid_identity_projection=",
        options.bilateral_grid_identity_projection,
        " ppisp=", options.use_ppisp,
        " ppisp_type=", cli.splat_ppisp_type,
        " ppisp_lr=", options.ppisp_lr,
        " ppisp_identity_projection=", options.ppisp_identity_projection,
        " ppisp_exposure_limit=", options.ppisp_exposure_limit,
        " ppisp_color_limit=", options.ppisp_color_limit,
        " ppisp_before_bilagrid=", options.ppisp_before_bilagrid,
        " densification_cap=", options.densification_cap,
        " dense_recycle_fraction=", options.dense_recycle_fraction,
        " dense_growth_fraction=", options.dense_growth_fraction,
        " use_mask=", options.use_mask,
        " projected_mask_views=", projected_mask_views, '/',
        scene.views.size(),
        " mask_dir=", options.mask_dir,
        " alpha_mode=", cli.splat_alpha_mode,
        " match_alpha_weight=", options.match_alpha_weight,
        " background_noise=", options.background_noise_strength,
        " cuda_profile=", options.profile_cuda,
        " cuda_profile_interval=", options.cuda_profile_interval,
        " ssim=fused_11x11_valid weight=", options.ssim_weight,
        " source_resolution=", options.use_source_resolution,
        " max_image_dimension=", options.max_image_dimension,
        " screen_space_kernel=", options.kernel_size,
        " progressive_resolution=", options.progressive_resolution,
        " progressive_interval=",
        options.progressive_resolution_interval,
        " progressive_initial_scale=",
        options.progressive_initial_scale,
        " host_view_cache_mb=",
        options.training_view_cache_bytes / (1024 * 1024),
        " device_cache_mb=",
        options.training_device_cache_bytes / (1024 * 1024),
        " device_cache_max_mb=",
        options.training_device_cache_max_bytes / (1024 * 1024),
        " async_upload=", options.training_async_upload,
        " cache_auto=", options.adaptive_training_cache,
        " prefetch_views=", options.training_prefetch_views,
        " prefetch_adaptive=", options.training_prefetch_adaptive,
        " eval_split_every=", options.evaluation_split_every,
        " knn_scale=", options.initialize_scale_from_knn,
        " dense_structure_freeze_iter=",
        options.dense_structure_freeze_iter,
        " depth_normal_loss=", options.use_depth_normal_loss,
        " depth_normal_weight=", options.depth_normal_weight,
        " normal_field=", options.use_normal_field,
        " normal_field_weight=", options.normal_field_weight,
        " normal_field_depth_ratio=", options.normal_field_depth_ratio,
        " normal_field_from_iter=", options.normal_field_from_iter,
        " filter_3d=", options.use_depth_normal_loss,
        " multi_view_geo_weight=", options.multi_view_geo_weight,
        " multi_view_ncc_weight=", options.multi_view_ncc_weight,
        " multi_view_depth_bracket=", options.multi_view_depth_bracket,
        " multi_view_depth_tolerance=", options.multi_view_depth_tolerance,
        " multi_view_neighbours=", options.multi_view_num,
        " multi_view_tail_interval=", options.multi_view_tail_interval,
        " multi_view_adaptive=",
        options.multi_view_adaptive_frequency,
        " multi_view_adaptive_max_interval=",
        options.multi_view_adaptive_max_interval,
        " multi_view_stable_refinements=",
        options.multi_view_adaptive_stable_refinements,
        " multi_view_stability_thresholds=[count:",
        options.multi_view_adaptive_count_threshold,
        ",churn:", options.multi_view_adaptive_churn_threshold,
        ",depth:", options.multi_view_adaptive_depth_threshold,
        ",min_depth:",
        options.multi_view_adaptive_min_depth_consistency,
        ",distribution:",
        options.multi_view_adaptive_distribution_threshold, ']',
        " multi_view_pixel_noise=",
        options.multi_view_pixel_noise_threshold,
        " geometry_from_iter=", options.depth_normal_from_iter,
        " scale_fraction=[", options.minimum_scale_fraction,
        ',', options.maximum_scale_fraction, ']',
        " constrain_scales=", options.constrain_scale_range,
        " max_scale_ratio=", options.max_scale_ratio,
        " views=", scene.views.size());
    const std::filesystem::path out_dir = cli.output.parent_path().empty()
        ? std::filesystem::current_path()
        : cli.output.parent_path();
    std::vector<std::size_t> evaluation_views;
    photara::splat::EvaluationCallback evaluate;
    const bool held_out_eval =
        !cli.gui && options.evaluation_split_every != 0 &&
        scene.views.size() > 1;
    if (!cli.gui) {
        if (held_out_eval) {
            for (std::size_t index = 0; index < scene.views.size(); ++index)
                if (index % options.evaluation_split_every == 0)
                    evaluation_views.push_back(index);
            if (evaluation_views.size() >= scene.views.size())
                evaluation_views.pop_back();
        } else if (!scene.views.empty()) {
            evaluation_views = {
                0, scene.views.size() / 2, scene.views.size() - 1};
        }
        std::sort(evaluation_views.begin(), evaluation_views.end());
        evaluation_views.erase(
            std::unique(evaluation_views.begin(), evaluation_views.end()),
            evaluation_views.end());
        const std::size_t train_views = held_out_eval
            ? scene.views.size() - evaluation_views.size()
            : scene.views.size();
        photara::core::Logger::instance().info(
            "splat_eval_split every=", options.evaluation_split_every,
            " train_views=", train_views,
            " held_out_views=",
            held_out_eval ? evaluation_views.size() : 0,
            held_out_eval ? "" : " (eval views are trained)");
        evaluate =
            [&](const unsigned iteration,
                const photara::splat::GaussianModel& model) {
                double psnr_sum = 0.0;
                double foreground_psnr_sum = 0.0;
                double ssim_sum = 0.0;
                for (const std::size_t view_index : evaluation_views) {
                    const auto render_path = out_dir /
                        (cli.output.stem().string() + "_splat_iter_" +
                         std::to_string(iteration) + "_view_" +
                         std::to_string(view_index) + ".png");
                    const auto metrics =
                        photara::splat::render_evaluation_png(
                            model, scene.views[view_index], render_path,
                            options);
                    photara::core::Logger::instance().info(
                        "splat_eval_iteration=", iteration,
                        " view=", view_index,
                        " psnr=", metrics.psnr,
                        " ssim=", metrics.ssim,
                        " foreground_psnr=", metrics.foreground_psnr);
                    psnr_sum += metrics.psnr;
                    foreground_psnr_sum += metrics.foreground_psnr;
                    ssim_sum += metrics.ssim;
                }
                photara::core::Logger::instance().info(
                    "splat_eval_iteration=", iteration,
                    " views=", evaluation_views.size(),
                    " held_out=", held_out_eval ? 1 : 0,
                    " average_psnr=", psnr_sum / evaluation_views.size(),
                    " average_ssim=", ssim_sum / evaluation_views.size(),
                    " average_foreground_psnr=",
                    foreground_psnr_sum / evaluation_views.size());
            };
    }
    const auto started = std::chrono::steady_clock::now();
    photara::splat::PreviewCallback preview;
    photara::splat::DevicePreviewCallback device_preview;
    std::unique_ptr<photara::splat::CudaVulkanPreview> vulkan_preview;
    const bool has_vulkan_preview =
        cli.splat_preview_vk_memory_handle != 0 &&
        cli.splat_preview_vk_semaphore_handle != 0 &&
        cli.splat_preview_vk_allocation_size != 0 &&
        cli.splat_preview_vk_width != 0 &&
        cli.splat_preview_vk_height != 0;
    if (options.preview_interval != 0 && has_vulkan_preview) {
        vulkan_preview =
            std::make_unique<photara::splat::CudaVulkanPreview>(
                photara::splat::CudaVulkanPreviewOptions{
                    cli.splat_preview_vk_memory_handle,
                    cli.splat_preview_vk_semaphore_handle,
                    cli.splat_preview_vk_allocation_size,
                    cli.splat_preview_vk_width,
                    cli.splat_preview_vk_height,
                    cli.splat_preview_vk_device_luid,
                    cli.splat_preview_vk_device_node_mask});
        device_preview = [&vulkan_preview](
                             const unsigned iteration,
                             const std::size_t view_index,
                             const photara::splat::Camera& camera,
                             const tinytensor::Tensor& color) {
            vulkan_preview->submit(color, camera.width, camera.height);
            photara::core::Logger::instance().debug(
                "splat_preview_iteration=", iteration,
                " view=", view_index,
                " transport=cuda_vulkan_external_memory",
                " timeline_value=", 2 * vulkan_preview->frame_count() - 1);
        };
        photara::core::Logger::instance().info(
            "splat_preview_transport=cuda_vulkan_external_memory extent=",
            cli.splat_preview_vk_width, 'x',
            cli.splat_preview_vk_height);
        if (!cli.splat_preview_ack_file.empty())
            photara::core::Logger::instance().info(
                "splat_preview_ack_file=\"", cli.splat_preview_ack_file, '"');
    } else if (options.preview_interval != 0 && !cli.gui) {
        const std::filesystem::path preview_dir =
            cli.splat_preview_dir.empty()
                ? out_dir / "splat_previews"
                : cli.splat_preview_dir;
        std::filesystem::create_directories(preview_dir);
        preview = [preview_dir, previous = std::filesystem::path{}](
                      photara::splat::TrainingPreview frame) mutable {
            photara::io::RgbImage image;
            image.width = frame.width;
            image.height = frame.height;
            image.pixels = std::move(frame.rgb);
            std::ostringstream filename;
            filename << "preview_iter_" << std::setw(8)
                     << std::setfill('0') << frame.iteration << "_view_"
                     << frame.view_index << ".png";
            const auto path = preview_dir / filename.str();
            photara::io::save_rgb_png(image, path);
            if (!previous.empty()) {
                std::error_code remove_error;
                std::filesystem::remove(previous, remove_error);
            }
            previous = path;
            photara::core::Logger::instance().debug(
                "splat_preview_iteration=", frame.iteration,
                " view=", frame.view_index, " image=", path);
        };
    }
    photara::splat::GaussianModel gaussians =
        photara::splat::Trainer(options).train(
            scene,
            [](const photara::splat::TrainingProgress& progress) {
                std::ostringstream line;
                line << "splat iteration=" << progress.iteration << '/'
                     << progress.total_iterations
                     << " view=" << progress.view_index
                     << " gaussians=" << progress.gaussian_count;
                if (progress.grown_count != 0 || progress.pruned_count != 0)
                    line << " grown=" << progress.grown_count
                         << " pruned=" << progress.pruned_count;
                line << " loss=" << progress.loss
                     << " rgb=" << progress.rgb_loss
                     << " alpha=" << progress.alpha_loss;
                if (progress.depth_loss != 0.F)
                    line << " depth=" << progress.depth_loss;
                if (progress.normal_loss != 0.F)
                    line << " normal=" << progress.normal_loss;
                if (progress.multi_view_geometry_loss != 0.F ||
                    progress.multi_view_ncc_loss != 0.F)
                    line << " mv_geo=" << progress.multi_view_geometry_loss
                         << " mv_ncc=" << progress.multi_view_ncc_loss;
                line << " image=" << progress.image_width << 'x'
                     << progress.image_height
                     << " instances=" << progress.rendered_instances
                     << " sh_degree=" << progress.active_sh_degree
                     << " step_ms=" << progress.milliseconds;
                photara::core::Logger::instance().info(line.str());
                return true;
            },
            evaluate, preview, device_preview);
    const std::filesystem::path model_path = splat_file_path(cli);
    if (!model_path.empty()) {
        ensure_artifact_parent(model_path);
        photara::splat::save_gaussians(gaussians, model_path);
    }
    if (!cli.gui && !evaluation_views.empty()) {
    double final_psnr_sum = 0.0;
    double final_foreground_psnr_sum = 0.0;
    double final_ssim_sum = 0.0;
    for (const std::size_t view_index : evaluation_views) {
        const auto render_path = out_dir /
            (cli.output.stem().string() + "_splat_view_" +
             std::to_string(view_index) + ".png");
        const auto metrics = photara::splat::render_evaluation_png(
            gaussians, scene.views[view_index], render_path, options);
        photara::core::Logger::instance().info(
            "splat_render=", render_path,
            " view=", view_index,
            " psnr=", metrics.psnr,
            " ssim=", metrics.ssim,
            " foreground_psnr=", metrics.foreground_psnr);
        final_psnr_sum += metrics.psnr;
        final_foreground_psnr_sum += metrics.foreground_psnr;
        final_ssim_sum += metrics.ssim;
    }
    photara::core::Logger::instance().info(
        "splat_final_evaluation_views=", evaluation_views.size(),
        " held_out=", held_out_eval ? 1 : 0,
        " average_psnr=", final_psnr_sum / evaluation_views.size(),
        " average_ssim=", final_ssim_sum / evaluation_views.size(),
        " average_foreground_psnr=",
        final_foreground_psnr_sum / evaluation_views.size());
    }
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    photara::core::Logger::instance().info(
        "splat_model=", model_path,
        " gaussians=", gaussians.size(),
        " training_s=",
        elapsed);
    if (project_archive) {
        project_archive->set_chunk(
            photara::project::ChunkType::gaussians,
            photara::splat::encode_gaussians(gaussians));
        project_archive->save(cli.output);
        photara::core::Logger::instance().info(
            "ascan_gaussians=", cli.output, " count=", gaussians.size());
    }
    if (!cli.mesh) return std::nullopt;
    if (cli.mesh_method == "pam") {
        photara::splat::PamMeshOptions pam_options;
        pam_options.max_points = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                cli.pam_max_points,
                (std::numeric_limits<std::size_t>::max)()));
        pam_options.pivot_max_points = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                cli.pam_pivot_max_points,
                (std::numeric_limits<std::size_t>::max)()));
        pam_options.pivot_std_factor = cli.pam_pivot_std_factor;
        pam_options.gaussian_seed_fraction =
            cli.pam_gaussian_seed_fraction;
        pam_options.refinement_steps = cli.pam_refinement_steps;
        pam_options.vector_field_neighbors = cli.pam_neighbors;
        pam_options.points_per_tetrahedron =
            cli.pam_points_per_tetrahedron;
        pam_options.occupancy_iso_value =
            cli.pam_occupancy_iso_value;
        pam_options.vacancy_threshold = cli.pam_vacancy_threshold;
        pam_options.mask_background_threshold =
            cli.pam_mask_background_threshold;
        const auto pam_started = std::chrono::steady_clock::now();
        // An empty seed requests GaussianWrapping's native path:
        // learned-normal pivots -> tetra_triangulation -> marching tetrahedra
        // -> PAM resampling and a second occupancy-classified Delaunay.
        auto pam = photara::splat::extract_pam_mesh(
            gaussians, scene, photara::mvs::Mesh{}, options,
            pam_options);
        const auto seed_ply = out_dir /
            (cli.output.stem().string() + "_pam_pivot_mesh.ply");
        const auto candidates_ply = out_dir /
            (cli.output.stem().string() + "_pam_candidates.ply");
        if (!cli.gui) {
            if (!pam.seed_mesh.faces.empty())
                photara::mvs::save_mesh_ply(pam.seed_mesh, seed_ply);
            photara::mvs::save_dense_ply(
                pam.candidate_cloud, candidates_ply);
        }
        photara::core::Logger::instance().info(
            "pam_pivot_mesh_ply=", seed_ply,
            " pivot_vertices=", pam.seed_mesh.vertices.size(),
            " pivot_faces=", pam.seed_mesh.faces.size(),
            " pam_candidates_ply=", candidates_ply,
            " candidates=", pam.candidate_cloud.points.size(),
            " tetrahedra=", pam.tetrahedron_count,
            " occupied_tetrahedra=", pam.occupied_tetrahedron_count,
            " mesh_vertices=", pam.mesh.vertices.size(),
            " mesh_faces=", pam.mesh.faces.size(),
            " pam_s=",
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pam_started).count());
        return std::move(pam.mesh);
    }
    if (mesh_options == nullptr)
        throw std::invalid_argument(
            "Splat mesh extraction requires configured MVS mesh options");

    photara::splat::SplatMeshOptions extraction_options;
    extraction_options.fusion = *mesh_options;
    extraction_options.fusion.build_mesh = true;
    if (!cli.gui) extraction_options.diagnostics_dir = out_dir;
    const auto mesh_started = std::chrono::steady_clock::now();
    auto extraction = photara::splat::extract_splat_mesh(
        gaussians, scene, options, extraction_options);
    const auto surface_ply = out_dir /
        (cli.output.stem().string() + "_splat_surface.ply");
    if (!cli.gui && !extraction.surface_cloud.points.empty())
        photara::mvs::save_dense_ply(
            extraction.surface_cloud, surface_ply);
    const double mesh_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - mesh_started).count();
    photara::core::Logger::instance().info(
        "splat_surface_ply=",
        extraction.surface_cloud.points.empty()
            ? std::filesystem::path{"disabled_for_tsdf"}
            : surface_ply,
        " valid_depth_pixels=", extraction.valid_depth_pixels,
        " surface_points=", extraction.surface_cloud.points.size(),
        " mesh_vertices=", extraction.mesh.vertices.size(),
        " mesh_faces=", extraction.mesh.faces.size(),
        " mesh_s=", mesh_elapsed);
    return std::move(extraction.mesh);
}
#endif

}  // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    // Report CRT parameter failures (which otherwise abort with no trace).
    _set_invalid_parameter_handler(
        [](const wchar_t* expression, const wchar_t* function,
           const wchar_t* file, unsigned line, std::uintptr_t) {
            const auto narrow = [](const wchar_t* text) {
                if (text == nullptr) return std::string{"?"};
                std::string out;
                for (; *text != L'\0'; ++text)
                    out.push_back(*text < 128 ? static_cast<char>(*text) : '?');
                return out;
            };
            photara::core::Logger::instance().error(
                "invalid_parameter function=", narrow(function),
                " file=", narrow(file), " line=", line,
                " expression=", narrow(expression));
            std::_Exit(3);
        });
    // A terminate from a worker thread or a destructor used to abort without
    // leaving any trace in the log; report what is known before aborting.
    std::set_terminate([] {
        try {
            if (std::current_exception() == nullptr) {
                photara::core::Logger::instance().error(
                    "terminate: no active exception");
            } else {
                try {
                    std::rethrow_exception(std::current_exception());
                } catch (const std::exception& error) {
                    photara::core::Logger::instance().error(
                        "terminate: ", error.what());
                } catch (...) {
                    photara::core::Logger::instance().error(
                        "terminate: non-standard exception");
                }
            }
        } catch (...) {
        }
        std::abort();
    });
    try {
        Utf8Argv utf8_argv(argc, argv);
        ReconstructCli cli = parse_cli(utf8_argv.argc(), utf8_argv.argv());

        const char* configured_level = std::getenv("PHOTARA_LOG_LEVEL");
        const auto console_level = configured_level
            ? photara::core::parse_log_level(
                  configured_level, photara::core::LogLevel::info)
            : photara::core::LogLevel::info;
        std::filesystem::path log_directory = cli.output.parent_path();
        if (log_directory.empty()) log_directory = std::filesystem::current_path();
        const std::filesystem::path log_path =
            photara::core::Logger::instance().configure(
                cli.gui ? std::filesystem::path{} : log_directory,
                "photara", console_level,
                cli.gui ? photara::core::LogLevel::off
                        : photara::core::LogLevel::trace);
        photara::core::Logger::instance().info(
            PHOTARA_VERSION_STRING, " started: mode=", cli.mode,
            " images_dir=", cli.images_dir, " output=", cli.output,
            " log=", log_path);

        const auto output_ext = lower_extension(cli.output);
        const bool project_output = output_ext == ".ascan";
        const bool write_project = project_output && !cli.gui;
        photara::project::Archive archive =
            photara::project::Archive::create();
        std::error_code project_error;
        if (project_output &&
            std::filesystem::exists(cli.output, project_error)) {
            archive = photara::project::Archive::open(cli.output);
            photara::core::Logger::instance().info(
                "ascan_open=", cli.output,
                " sfm=", archive.has(photara::project::ChunkType::sfm),
                " gaussians=",
                archive.has(photara::project::ChunkType::gaussians),
                " mesh=", archive.has(photara::project::ChunkType::mesh));
        }

#if defined(PHOTARA_HAS_SPLAT)
        if (cli.splat_view) {
            run_splat_view(cli, archive);
            return 0;
        }
#endif
        const bool preserve_cameras =
            (cli.splat || cli.dense || cli.texture) &&
            ((!cli.working_sfm.empty() &&
              std::filesystem::exists(cli.working_sfm, project_error)) ||
             archive.has(photara::project::ChunkType::sfm));
        ensure_video_frames(cli, preserve_cameras);
        refresh_auto_masks(cli);
        ensure_sam_masks(cli);

#if defined(PHOTARA_HAS_SPLAT)
        if (!cli.splat_dataset.empty()) {
            if (write_project)
                photara::project::write_settings(
                    archive, settings_from_cli(cli), cli.output);
            photara::splat::DatasetLoadRequest request;
            request.source = cli.splat_dataset;
            request.image_directory = cli.images_dir;
            request.initial_point_cloud = cli.dense_ply;
            auto loaded =
                photara::splat::load_splat_dataset(request);
            for (const std::string& warning : loaded.warnings)
                photara::core::Logger::instance().warning(
                    "splat dataset: ", warning);
            apply_cli_subject_bounds(cli, loaded.scene);
            photara::core::Logger::instance().info(
                "pipeline_handoff=external_dataset");
            photara::core::Logger::instance().info(
                "splat_dataset=", loaded.resolved_source,
                " format=",
                photara::splat::dataset_format_name(loaded.format),
                " cameras=", loaded.scene.views.size(),
                loaded.initial_points_dense
                    ? " dense_points="
                    : " initial_points=",
                loaded.scene.dense_cloud.points.size(),
                loaded.generated_initial_points
                    ? " generated_initial_points=true"
                    : std::string{},
                loaded.initial_point_cloud.empty()
                    ? std::string{}
                    : " initial_ply=" +
                          loaded.initial_point_cloud.string());
            if (cli.texture_only) {
#if defined(PHOTARA_HAS_TEXTURE)
                bake_texture_only(cli, loaded.scene, archive, write_project);
                return 0;
#else
                throw std::runtime_error(
                    "Bake Texture requires PHOTARA_ENABLE_TEXTURE");
#endif
            }
            photara::mvs::DensifyOptions mesh_options;
            photara::mvs::apply_quality_preset(
                mesh_options, cli.dense_quality);
            mesh_options.mask_dir = cli.masks_dir;
            mesh_options.mesh_method = photara::mvs::MeshMethod::tsdf;
            mesh_options.mesh_tsdf_voxel_scale =
                cli.mesh_tsdf_voxel_scale > 0.F
                ? cli.mesh_tsdf_voxel_scale
                : 1.F;
            mesh_options.mesh_tsdf_bounds_padding =
                cli.mesh_tsdf_bounds_padding;
            mesh_options.mesh_tsdf_pixel_step =
                cli.mesh_tsdf_pixel_step;
            mesh_options.mesh_tsdf_support_closing_axes =
                cli.mesh_tsdf_support_closing_axes;
            mesh_options.mesh_tsdf_frame_export_dir =
                cli.mesh_tsdf_frame_export_dir;
            mesh_options.mesh_tsdf_smooth_iters =
                cli.mesh_tsdf_smooth_iters;
            mesh_options.mesh_tsdf_smooth_lambda =
                cli.mesh_tsdf_smooth_lambda;
            mesh_options.mesh_tsdf_smooth_mu = cli.mesh_tsdf_smooth_mu;
            if (!cli.splat && !cli.mvs_mesh_only) {
                // Explicit --dense/--mesh with calibrated cameras is MVS, not
                // an implicit Gaussian-training request. Keep the given poses.
                mesh_options.mesh_method = cli.mesh_method == "tsdf"
                    ? photara::mvs::MeshMethod::tsdf
                    : photara::mvs::MeshMethod::delaunay_cut;
                mesh_options.build_mesh = cli.mesh;
                mesh_options.mesh_max_points = cli.mesh_max_points;
                if (cli.dense_resolution_level)
                    mesh_options.resolution_level = *cli.dense_resolution_level;
                if (cli.mesh_dist_insert_px >= 0.F)
                    mesh_options.mesh_dist_insert_px = cli.mesh_dist_insert_px;
                mesh_options.mesh_use_free_space_support = cli.mesh_free_space_support;
                mesh_options.mesh_adaptive_sigma = cli.mesh_adaptive_sigma;
                mesh_options.mesh_max_edge_scale = cli.mesh_max_edge_scale;
                mesh_options.mesh_k_free_space_calibration_quantile =
                    std::clamp(cli.mesh_free_space_quantile, 0.F, 0.999F);
                mesh_options.patchmatch_tile_rows = cli.patchmatch_tile_rows;
                mesh_options.patchmatch_concurrent_views = cli.patchmatch_concurrent_views;
                photara::mvs::prepare_imported_scene(loaded.scene, mesh_options);
                photara::mvs::densify(loaded.scene, mesh_options);
                const auto stem = cli.output.parent_path() / cli.output.stem();
                write_dense_artifact(
                    cli, loaded.scene.dense_cloud,
                    std::filesystem::path(stem.string() + "_dense.ply"));
                if (cli.mesh) {
                    if (loaded.scene.mesh.faces.empty())
                        throw std::runtime_error("Calibrated MVS produced an empty mesh");
                    write_mesh_artifact(
                        cli, loaded.scene.mesh,
                        std::filesystem::path(stem.string() + "_mesh.ply"),
                        cli.mesh_obj);
                    if (write_project) {
                        archive.set_chunk(photara::project::ChunkType::mesh,
                            photara::mvs::encode_mesh(loaded.scene.mesh));
                        archive.save(cli.output);
                    }
                }
#if defined(PHOTARA_HAS_TEXTURE)
                if (cli.texture) {
                    write_texture_artifact(
                        cli, loaded.scene,
                        std::filesystem::path(stem.string() + "_textured"),
                        write_project ? &archive : nullptr, write_project);
                }
#endif
                return 0;
            }
            if (cli.mvs_mesh_only) {
                if (!cli.mask_mesh.empty()) {
                    loaded.scene.mesh =
                        photara::mvs::load_mesh_ply(cli.mask_mesh);
                } else {
                    mesh_options.mesh_method =
                        photara::mvs::MeshMethod::delaunay_cut;
                    mesh_options.build_mesh = true;
                    mesh_options.mesh_max_points = cli.mesh_max_points;
                    if (cli.mesh_dist_insert_px >= 0.F)
                        mesh_options.mesh_dist_insert_px =
                            cli.mesh_dist_insert_px;
                    mesh_options.mesh_use_free_space_support =
                        cli.mesh_free_space_support;
                    mesh_options.mesh_adaptive_sigma = cli.mesh_adaptive_sigma;
                    mesh_options.mesh_max_edge_scale = cli.mesh_max_edge_scale;
                    mesh_options.mesh_k_free_space_calibration_quantile =
                        std::clamp(
                            cli.mesh_free_space_quantile, 0.F, 0.999F);
                    photara::mvs::reconstruct_mesh(
                        loaded.scene, mesh_options);
                }
                photara::mvs::save_mesh_ply(
                    loaded.scene.mesh, cli.output);
#if defined(PHOTARA_HAS_TEXTURE)
                const auto mask_directory =
                    cli.output.parent_path() /
                    (cli.output.stem().string() + "_masks");
                photara::texture::MeshMaskOptions mask_options;
                mask_options.preview_directory =
                    cli.output.parent_path() /
                    (cli.output.stem().string() + "_mesh_previews");
                const auto mask_summary =
                    photara::texture::render_mesh_foreground_masks(
                        loaded.scene, mask_directory, mask_options);
                photara::core::Logger::instance().info(
                    "mvs_mesh_quality_gate=", cli.output,
                    cli.mask_mesh.empty()
                        ? std::string{}
                        : " source_mesh=" + cli.mask_mesh.string(),
                    " vertices=", loaded.scene.mesh.vertices.size(),
                    " faces=", loaded.scene.mesh.faces.size(),
                    " mask_views=", mask_summary.image_count,
                    " preview_views=", mask_summary.preview_count,
                    " mask_directory=", mask_directory);
#endif
                return 0;
            }
            auto mesh = run_splat_training(
                loaded.scene, cli, loaded.initial_points_dense,
                cli.mesh ? &mesh_options : nullptr, {},
                write_project ? &archive : nullptr);
            if (mesh) {
#if defined(PHOTARA_HAS_MESH_TOOLS)
                if (cli.mesh_target_faces > 0)
                    repair_and_decimate_mesh(
                        *mesh, cli.mesh_target_faces, cli.mesh_remesh);
#endif
                const auto mesh_path = cli.output.parent_path() /
                    (cli.output.stem().string() + "_splat_mesh.ply");
                write_mesh_artifact(cli, *mesh, mesh_path, cli.mesh_obj);
                if (write_project) {
                    archive.set_chunk(
                        photara::project::ChunkType::mesh,
                        photara::mvs::encode_mesh(*mesh));
                    archive.save(cli.output);
                    photara::core::Logger::instance().info(
                        "ascan_mesh=", cli.output,
                        " faces=", mesh->faces.size());
                }
            }
            return 0;
        }
#endif

        photara::sfm::Scene scene;
        bool loaded_project_sfm = false;
        const bool needs_cameras = cli.splat || cli.dense || cli.texture ||
            !cli.export_colmap_dir.empty();
        if (project_output && needs_cameras &&
            archive.has(photara::project::ChunkType::sfm)) {
            auto loaded = photara::project::read_sfm(archive);
            if (loaded && loaded->registered_count() >= 2) {
                scene = std::move(*loaded);
                loaded_project_sfm = true;
                photara::core::Logger::instance().info(
                    "pipeline_handoff=ascan_sfm");
                photara::core::Logger::instance().info(
                    "ascan_sfm_loaded images=", scene.images.size(),
                    " registered=", scene.registered_count(),
                    " tracks=", scene.tracks.size());
            }
        }
        if (needs_cameras && !loaded_project_sfm &&
            load_working_sfm(cli, scene)) {
            loaded_project_sfm = true;
            photara::core::Logger::instance().info(
                "pipeline_handoff=working_sfm");
        }
        if (loaded_project_sfm) ensure_scene_appearance(cli, scene);

        if (cli.texture_only) {
            if (!loaded_project_sfm)
                throw std::runtime_error(
                    "Bake Texture needs aligned cameras. Run Align Photos or "
                    "load an external camera dataset first.");
#if defined(PHOTARA_HAS_TEXTURE)
            photara::mvs::DensifyOptions texture_scene_options;
            texture_scene_options.resolution_level = 0;
            texture_scene_options.build_mesh = false;
            texture_scene_options.mesh_method =
                photara::mvs::MeshMethod::none;
            texture_scene_options.thread_count = scene.thread_count;
            auto texture_scene = photara::mvs::build_mvs_scene(
                scene, texture_scene_options);
            bake_texture_only(
                cli, texture_scene, archive, write_project);
            return 0;
#else
            throw std::runtime_error(
                "Bake Texture requires PHOTARA_ENABLE_TEXTURE");
#endif
        }

        photara::sfm::ReconstructionSummary summary{};
        double elapsed = 0.0;
        if (!loaded_project_sfm) {
        photara::core::Logger::instance().info(
            "pipeline_handoff=reconstruct");
        std::vector<std::filesystem::path> files;
        for (const auto& entry :
             std::filesystem::directory_iterator(cli.images_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string extension = entry.path().extension().string();
            std::transform(
                extension.begin(), extension.end(), extension.begin(),
                [](const unsigned char value) {
                    return static_cast<char>(std::tolower(value));
                });
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
                extension == ".tif" || extension == ".tiff")
                files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end());
        if (files.size() < 2) throw std::runtime_error("Need at least two images");

        photara::sfm::ReconstructionConfig config;
        config.global_positioning.prefer_cuda = cli.positioning_cuda;
        std::chrono::steady_clock::time_point last_alignment_preview{};
        bool has_alignment_preview = false;
        if (!cli.working_sfm.empty()) {
            config.resection.checkpoint_callback = [&](const photara::sfm::Scene& partial) {
                const auto now = std::chrono::steady_clock::now();
                if (partial.registered_count() < 2 ||
                    (has_alignment_preview &&
                     now - last_alignment_preview < std::chrono::seconds(3))) return;
                last_alignment_preview = now;
                has_alignment_preview = true;
                auto path = cli.align_preview.empty() ? cli.working_sfm
                                                      : cli.align_preview;
                if (cli.align_preview.empty()) path += ".preview.asfm";
                try {
                    photara::sfm::save_alignment_preview(partial, path);
                    photara::core::Logger::instance().info(
                        "sfm_preview=", path, " registered=", partial.registered_count());
                } catch (const std::exception& error) {
                    photara::core::Logger::instance().warning("SfM preview skipped: ", error.what());
                }
            };
        }
        if (cli.mode == "global")
            config.mode = photara::sfm::ReconstructionMode::global;
        else if (cli.mode == "hierarchical")
            config.mode = photara::sfm::ReconstructionMode::hierarchical;
        else
            config.mode = photara::sfm::ReconstructionMode::incremental;
        config.frontend.camera_model = cli.camera_model;
        config.frontend.focal_pixels = cli.focal_pixels;
        config.frontend.trust_focal_pixels = cli.trust_focal;
        config.frontend.mask_dir = cli.masks_dir;
        config.frontend.structural_pair_expansion = cli.structural_pair_expansion;
        config.frontend.neighbor_window = cli.neighbor_window;
        config.frontend.sift_contrast_threshold = cli.sift_contrast;
        config.frontend.match_ratio = cli.match_ratio;
        config.frontend.mutual_check = cli.mutual_check;
        config.frontend.extractor = cli.extractor;
        config.frontend.matcher = cli.matcher;
        config.frontend.pipeline = cli.pipeline;
        config.frontend.max_features = cli.max_features;
        config.frontend.extractor_model_path = cli.extractor_model;
        config.frontend.extractor_input_width = cli.extractor_width;
        config.frontend.extractor_input_height = cli.extractor_height;
        config.frontend.extractor_min_score = cli.extractor_min_score;
        config.frontend.extractor_use_cuda = !cli.extractor_cpu;
        config.frontend.lightglue_model_path = cli.lightglue_model;
        config.frontend.lightglue_extractor = cli.lightglue_extractor;
        config.frontend.lightglue_input_width = cli.lightglue_width;
        config.frontend.lightglue_input_height = cli.lightglue_height;
        config.frontend.lightglue_min_score = cli.lightglue_min_score;
        config.frontend.hybrid_lightglue_max_features =
            cli.hybrid_lightglue_max_features;
        config.frontend.lightglue_use_cuda = !cli.lightglue_cpu;
        // Sequential window + BoW retrieval (learned vocabulary).
        // lightglue_end2end uses a temporary SiftGPU extract for BoW only.
        config.frontend.augment_sequential_with_retrieval = true;
        config.frontend.checkpoint.directory = cli.cache_dir;
        photara::sfm::AlignLivePreview live_preview;
        if (!cli.working_sfm.empty()) {
            auto live_path =
                cli.align_live.empty() ? cli.working_sfm : cli.align_live;
            if (cli.align_live.empty()) live_path += ".live";
            live_preview.set_path(live_path);
            config.frontend.live_preview = &live_preview;
        }

        const auto started = std::chrono::steady_clock::now();
        summary = photara::sfm::reconstruct(scene, files, config);
        elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();

        if (!summary.valid) {
            photara::core::Logger::instance().error(
                "reconstruction failed: registered=", summary.registered_views,
                '/', scene.images.size(), " landmarks=", summary.landmarks);
            return 2;
        }

        if (!cli.gui) {
            const auto diagnostics_path =
                write_sfm_diagnostics(scene, cli.output);
            photara::core::Logger::instance().info(
                "sfm_diagnostics=", diagnostics_path);
        }
        } else {
            summary.valid = true;
            summary.registered_views =
                static_cast<unsigned>(scene.registered_count());
            unsigned landmarks = 0;
            for (const auto& track : scene.tracks)
                if (track.is_triangulated()) ++landmarks;
            summary.landmarks = landmarks;
            summary.failed_views = static_cast<unsigned>(
                scene.images.size() - scene.registered_count());
        }

        save_working_sfm(cli, scene);

        if (!cli.export_colmap_dir.empty() && scene.registered_count() >= 2) {
            // Diagnostics-grade export: cameras, poses and the sparse cloud in
            // the COLMAP text layout (equirectangular cameras become model 17).
            photara::sfm::save_colmap_text(
                scene, cli.export_colmap_dir, cli.images_dir, true);
            photara::core::Logger::instance().info(
                "colmap_export=", cli.export_colmap_dir);
        }

        if (output_ext == ".mvs") {
            photara::sfm::export_openmvs_interface(scene, cli.output);
        } else if (output_ext == ".ply") {
            save_ply(scene, cli.output);
        } else if (output_ext == ".asfm") {
            photara::sfm::save_asfm(scene, cli.output);
            photara::core::Logger::instance().info("asfm=", cli.output);
        } else if (output_ext == ".ascan") {
            if (write_project) {
                const auto settings = settings_from_cli(cli);
                if (!loaded_project_sfm)
                    photara::project::replace_sfm_stage(
                        archive, scene, settings, cli.output);
                else
                    photara::project::write_settings(
                        archive, settings, cli.output);
                archive.save(cli.output);
                photara::core::Logger::instance().info(
                    "ascan=", cli.output, " sfm_images=", scene.images.size());
            }
        } else {
            throw std::invalid_argument(
                "Output must end with .ascan, .asfm, .ply, or .mvs");
        }

        if (cli.export_mvs_requested) {
            std::filesystem::path mvs_path = cli.export_mvs_path;
            if (mvs_path.empty()) {
                mvs_path = cli.output.parent_path() / cli.output.stem();
                mvs_path += ".mvs";
            }
            if (lower_extension(mvs_path) != ".mvs")
                throw std::invalid_argument(
                    "--export-mvs path must end with .mvs");
            if (mvs_path != cli.output) {
                photara::sfm::export_openmvs_interface(scene, mvs_path);
                photara::core::Logger::instance().info("mvs=", mvs_path);
            }
        }

#if defined(PHOTARA_HAS_SPLAT)
        if (cli.splat && !cli.dense) {
            photara::mvs::DensifyOptions mesh_options;
            photara::mvs::apply_quality_preset(
                mesh_options, cli.dense_quality);
            mesh_options.mask_dir = cli.masks_dir;
            mesh_options.mesh_method = photara::mvs::MeshMethod::tsdf;
            mesh_options.mesh_tsdf_voxel_scale =
                cli.mesh_tsdf_voxel_scale > 0.F
                ? cli.mesh_tsdf_voxel_scale
                : 1.F;
            mesh_options.mesh_tsdf_bounds_padding =
                cli.mesh_tsdf_bounds_padding;
            mesh_options.mesh_tsdf_pixel_step =
                cli.mesh_tsdf_pixel_step;
            mesh_options.mesh_tsdf_support_closing_axes =
                cli.mesh_tsdf_support_closing_axes;
            mesh_options.mesh_tsdf_frame_export_dir =
                cli.mesh_tsdf_frame_export_dir;
            mesh_options.mesh_tsdf_smooth_iters =
                cli.mesh_tsdf_smooth_iters;
            mesh_options.mesh_tsdf_smooth_lambda =
                cli.mesh_tsdf_smooth_lambda;
            mesh_options.mesh_tsdf_smooth_mu = cli.mesh_tsdf_smooth_mu;
            mesh_options.thread_count = scene.thread_count;

            auto splat_scene =
                photara::mvs::build_mvs_scene(scene, mesh_options);
            photara::splat::initialize_scene_from_sparse_points(
                splat_scene);
            apply_cli_subject_bounds(cli, splat_scene);
            if (cli.capture_mode == "object" && cli.subject_bounds.empty() &&
                !splat_scene.subject_bounds.valid)
                throw std::runtime_error(
                    "Could not estimate SubjectBounds from SfM sparse points");

            const std::filesystem::path out_dir =
                cli.output.parent_path().empty()
                ? std::filesystem::current_path()
                : cli.output.parent_path();
            std::filesystem::path bounds_path;
            if (!cli.gui && splat_scene.subject_bounds.valid) {
                bounds_path =
                    out_dir /
                    (cli.output.stem().string() + "_subject_bounds.txt");
                photara::mvs::save_subject_bounds(
                    splat_scene.subject_bounds, bounds_path);
            }
            photara::core::Logger::instance().info(
                "splat_input=sfm_sparse",
                " initial_points=",
                splat_scene.dense_cloud.points.size(),
                " capture_mode=", cli.capture_mode,
                " subject_bounds=",
                bounds_path.empty()
                    ? std::string{"disabled"}
                    : bounds_path.string(),
                " patchmatch=false");

            auto mesh = run_splat_training(
                splat_scene, cli, false,
                cli.mesh ? &mesh_options : nullptr,
                cli.masks_dir,
                write_project ? &archive : nullptr);
            if (mesh) {
#if defined(PHOTARA_HAS_MESH_TOOLS)
                if (cli.mesh_target_faces > 0)
                    repair_and_decimate_mesh(
                        *mesh, cli.mesh_target_faces, cli.mesh_remesh);
#endif
                splat_scene.mesh = std::move(*mesh);
                const auto mesh_path =
                    out_dir /
                    (cli.output.stem().string() + "_splat_mesh.ply");
                write_mesh_artifact(
                    cli, splat_scene.mesh, mesh_path, cli.mesh_obj);
                if (write_project) {
                    archive.set_chunk(
                        photara::project::ChunkType::mesh,
                        photara::mvs::encode_mesh(splat_scene.mesh));
                    archive.save(cli.output);
                    photara::core::Logger::instance().info(
                        "ascan_mesh=", cli.output,
                        " faces=", splat_scene.mesh.faces.size());
                }

#if defined(PHOTARA_HAS_TEXTURE)
                if (cli.texture) {
                    write_texture_artifact(
                        cli, splat_scene,
                        out_dir / (cli.output.stem().string() + "_textured"),
                        write_project ? &archive : nullptr, write_project);
                }
#endif
            }
            return 0;
        }
#endif

        if (cli.dense) {
            photara::mvs::DensifyOptions densify_opts;
            photara::mvs::apply_quality_preset(
                densify_opts, cli.dense_quality);
            if (cli.dense_resolution_level)
                densify_opts.resolution_level = *cli.dense_resolution_level;
            densify_opts.mask_dir = cli.masks_dir;
            densify_opts.mesh_max_points = cli.mesh_max_points;
            // GGGS resolves the native voxel to max_depth/2048, matching
            // gs2mesh.py. A positive scale remains an explicit quality/speed
            // override; the default keeps the reference resolution.
            densify_opts.mesh_tsdf_voxel_scale =
                cli.mesh_tsdf_voxel_scale > 0.F
                ? cli.mesh_tsdf_voxel_scale
                : 1.F;
            densify_opts.mesh_tsdf_bounds_padding =
                cli.mesh_tsdf_bounds_padding;
            densify_opts.mesh_tsdf_pixel_step =
                cli.mesh_tsdf_pixel_step;
            densify_opts.mesh_tsdf_support_closing_axes =
                cli.mesh_tsdf_support_closing_axes;
            densify_opts.mesh_tsdf_frame_export_dir =
                cli.mesh_tsdf_frame_export_dir;
            densify_opts.mesh_tsdf_smooth_iters =
                cli.mesh_tsdf_smooth_iters;
            densify_opts.mesh_tsdf_smooth_lambda =
                cli.mesh_tsdf_smooth_lambda;
            densify_opts.mesh_tsdf_smooth_mu = cli.mesh_tsdf_smooth_mu;
            if (cli.mesh_dist_insert_px >= 0.F)
                densify_opts.mesh_dist_insert_px =
                    cli.mesh_dist_insert_px;
            densify_opts.mesh_use_free_space_support =
                cli.mesh_free_space_support;
            densify_opts.mesh_adaptive_sigma = cli.mesh_adaptive_sigma;
            densify_opts.mesh_max_edge_scale = cli.mesh_max_edge_scale;
            densify_opts.mesh_k_free_space_calibration_quantile =
                std::clamp(cli.mesh_free_space_quantile, 0.F, 0.999F);
            densify_opts.patchmatch_tile_rows = cli.patchmatch_tile_rows;
            densify_opts.patchmatch_concurrent_views =
                cli.patchmatch_concurrent_views;
            // MVS is now an explicit diagnostic path. Product splat training
            // starts directly from SfM sparse points and never depends on an
            // MVS-derived spatial crop or foreground mask.
            densify_opts.build_mesh = cli.texture || (!cli.splat && cli.mesh);
            if (!densify_opts.build_mesh) {
                densify_opts.mesh_method = photara::mvs::MeshMethod::none;
            } else if (cli.mesh_method == "tsdf") {
                densify_opts.mesh_method =
                    photara::mvs::MeshMethod::tsdf;
            } else {
                densify_opts.mesh_method =
                    photara::mvs::MeshMethod::delaunay_cut;
            }
            densify_opts.geometric_consistency = true;
            densify_opts.thread_count = scene.thread_count;
            photara::core::Logger::instance().info(
                "mvs config: resolution_level=", densify_opts.resolution_level,
                " pyramid_levels=", densify_opts.sub_resolution_levels + 1,
                " photo_iters=", densify_opts.estimation_iters,
                " geometric_rounds=", densify_opts.geometric_iters,
                " neighbors=", densify_opts.max_neighbors,
                " patch_views=", densify_opts.min_patch_views,
                " filter_views=", densify_opts.min_views_filter,
                " fuse_views=", densify_opts.min_views_fuse,
                " mesh_dist_insert_px=", densify_opts.mesh_dist_insert_px,
                " free_space_support=",
                densify_opts.mesh_use_free_space_support,
                " free_space_quantile=",
                densify_opts.mesh_k_free_space_calibration_quantile,
                " tile_rows=", densify_opts.patchmatch_tile_rows,
                " concurrent_views=",
                densify_opts.patchmatch_concurrent_views,
                " tsdf_voxel_scale=",
                densify_opts.mesh_tsdf_voxel_scale,
                " mesh_method=",
                densify_opts.mesh_method ==
                        photara::mvs::MeshMethod::delaunay_cut
                    ? "delaunay"
                    : densify_opts.mesh_method ==
                              photara::mvs::MeshMethod::tsdf
                          ? "tsdf"
                          : "none",
                " mask_border_px=", densify_opts.mask_border_px,
                " subject_bounds_source=sfm_sparse",
                " grazing_weight_floor=",
                densify_opts.grazing_weight_floor);
            if (!densify_opts.mask_dir.empty())
                photara::core::Logger::instance().info(
                    "mvs masks=", densify_opts.mask_dir);

            const auto dense_started = std::chrono::steady_clock::now();
            photara::mvs::MvsScene mvs_scene;
            if (!cli.dense_ply.empty()) {
                // No external camera metadata: keep the built-in SfM poses,
                // but use the caller's dense initialization directly.
                mvs_scene =
                    photara::mvs::build_mvs_scene(scene, densify_opts);
                apply_cli_subject_bounds(cli, mvs_scene);
                mvs_scene.dense_cloud =
                    photara::mvs::load_dense_ply(cli.dense_ply);
                photara::core::Logger::instance().info(
                    "splat cameras=internal_sfm dense_ply=", cli.dense_ply,
                    " points=", mvs_scene.dense_cloud.points.size());
            } else {
                mvs_scene =
                    photara::mvs::build_mvs_scene(scene, densify_opts);
                apply_cli_subject_bounds(cli, mvs_scene);
                photara::mvs::densify(mvs_scene, densify_opts);
            }
            const double dense_elapsed = std::chrono::duration<double>(
                                             std::chrono::steady_clock::now() -
                                             dense_started)
                                             .count();

            const std::filesystem::path out_dir =
                cli.output.parent_path().empty()
                    ? std::filesystem::current_path()
                    : cli.output.parent_path();
            write_dense_artifact(
                cli, mvs_scene.dense_cloud,
                out_dir / (cli.output.stem().string() + "_dense.ply"));
            photara::core::Logger::instance().info(
                "densify_s=", dense_elapsed);

            std::filesystem::path effective_mask_dir = cli.masks_dir;
            if (!cli.gui && !mvs_scene.mesh.faces.empty()) {
                const auto mvs_mesh_path =
                    out_dir /
                    (cli.output.stem().string() + "_mvs_mesh.ply");
                photara::mvs::save_mesh_ply(
                    mvs_scene.mesh, mvs_mesh_path);
                photara::core::Logger::instance().info(
                    "mvs_mesh_ply=", mvs_mesh_path,
                    " vertices=", mvs_scene.mesh.vertices.size(),
                    " faces=", mvs_scene.mesh.faces.size());
            }

#if defined(PHOTARA_HAS_SPLAT)
            if (cli.splat) {
                auto splat_mesh_options = densify_opts;
                splat_mesh_options.mesh_method =
                    photara::mvs::MeshMethod::tsdf;
                auto splat_mesh = run_splat_training(
                    mvs_scene, cli, true, &splat_mesh_options,
                    effective_mask_dir,
                    write_project ? &archive : nullptr);
                if (splat_mesh) mvs_scene.mesh = std::move(*splat_mesh);
            }
#endif

            if (cli.mesh && !mvs_scene.mesh.faces.empty()) {
#if defined(PHOTARA_HAS_MESH_TOOLS)
                if (cli.splat && cli.mesh_target_faces > 0) {
                    try {
                        repair_and_decimate_mesh(
                            mvs_scene.mesh, cli.mesh_target_faces,
                            cli.mesh_remesh);
                    } catch (const std::exception& error) {
                        photara::core::Logger::instance().warning(
                            "photara mesh postprocess failed; retaining cleaned "
                            "mesh: ", error.what());
                    }
                }
#else
                if (cli.splat && cli.mesh_target_faces > 0)
                    photara::core::Logger::instance().warning(
                        "photara mesh postprocess unavailable (CGAL mesh tools "
                        "were not built); retaining cleaned mesh");
#endif
                const std::string mesh_tag = cli.splat
                    ? "_splat_mesh"
                    : "_mesh";
                const auto mesh_ply =
                    out_dir / (cli.output.stem().string() + mesh_tag + ".ply");
                write_mesh_artifact(
                    cli, mvs_scene.mesh, mesh_ply, cli.mesh_obj);

#if defined(PHOTARA_HAS_TEXTURE)
                if (cli.texture) {
                    if (!effective_mask_dir.empty())
                        cli.masks_dir = effective_mask_dir;
                    write_texture_artifact(
                        cli, mvs_scene,
                        out_dir / (cli.output.stem().string() + "_textured"),
                        write_project ? &archive : nullptr, write_project);
                }
#endif
            }
        }

        photara::core::Logger::instance().info(
            "valid=", summary.valid, " registered=", summary.registered_views,
            '/', scene.images.size(), " landmarks=", summary.landmarks,
            " reprojection_mean_px=",
            summary.mean_reprojection_error_pixels,
            " reprojection_rms_px=",
            summary.rms_reprojection_error_pixels,
            " reprojection_observations=",
            summary.reprojection_observations,
            " peak_working_set_mb=",
            static_cast<double>(peak_working_set_bytes()) / (1024.0 * 1024.0),
            " failed=", summary.failed_views, " elapsed_s=", elapsed,
            " output=", cli.output, " log=", log_path);
        return summary.valid ? 0 : 2;
    } catch (const std::exception& error) {
        photara::core::Logger::instance().error("error: ", error.what());
        return 1;
    }
}
