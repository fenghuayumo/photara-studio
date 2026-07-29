#include "sfm/reconstruct.hpp"
#include "sfm/export_mvs.hpp"
#include "mvs/densify.hpp"
#include "mvs/export.hpp"
#include "mvs/internal.hpp"
#include "core/logging.hpp"
#if defined(AETHERSCAN_HAS_GGGS)
#include "splat/dataset.hpp"
#include "splat/trainer.hpp"
#endif
#if defined(AETHERSCAN_HAS_TEXTURE)
#include "texture/bake.hpp"
#include "texture/mask.hpp"
#include "texture/options.hpp"
#endif
#if defined(AETHERSCAN_HAS_ASDIFF_MESH)
#include "asdiff_mesh/mesh_ops.hpp"
#endif

#include <cxxopts.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
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

struct ReconstructCli {
    std::filesystem::path images_dir;
    double focal_pixels{};
    std::string mode{"global"};
    std::filesystem::path output;
    std::size_t neighbor_window{3};
    float match_ratio{0.85F};
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
    bool gggs{false};
    std::filesystem::path splat_dataset;
    std::string splat_format{"auto"};
    std::filesystem::path colmap_model;
    std::filesystem::path dense_ply;
    std::filesystem::path gggs_model;
    unsigned gggs_iterations{10'000};
    std::uint64_t gggs_max_gaussians{500'000};
    unsigned gggs_max_resolution{1'920};
    float gggs_kernel_size{0.F};
    bool gggs_progressive_resolution{true};
    unsigned gggs_progressive_interval{3'000};
    float gggs_progressive_initial_scale{0.25F};
    std::uint64_t gggs_view_cache_mb{6'144};
    unsigned gggs_eval_split_every{0};
    bool gggs_use_mask{true};
    std::string gggs_alpha_mode{"transparent"};
    float gggs_match_alpha_weight{0.25F};
    float gggs_ssim_weight{0.2F};
    float gggs_depth_normal_weight{0.05F};
    float gggs_multi_view_geo_weight{0.02F};
    float gggs_multi_view_ncc_weight{0.6F};
    unsigned gggs_multi_view_num{8};
    float gggs_multi_view_pixel_noise{1.F};
    unsigned gggs_geometry_from_iter{3'000};
    float gggs_min_scale_fraction{1e-4F};
    float gggs_max_scale_fraction{0.002F};
    float gggs_max_scale_ratio{0.F};
    bool gggs_max_scale_ratio_overridden{false};
    bool gggs_constrain_scales{false};
    std::string gggs_strategy{"default"};
    bool gggs_densification{true};
    unsigned gggs_structure_freeze_iter{0};
    std::uint64_t gggs_densification_cap{10'000'000};
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
    unsigned mesh_tsdf_support_closing_axes{2};
    std::filesystem::path mesh_tsdf_frame_export_dir;
    unsigned mesh_tsdf_smooth_iters{2};
    float mesh_tsdf_smooth_lambda{0.5F};
    float mesh_tsdf_smooth_mu{-0.53F};
    float mesh_dist_insert_px{-1.F};
    bool mesh_free_space_support{true};
    float mesh_free_space_quantile{0.95F};
    unsigned patchmatch_tile_rows{8};
    unsigned patchmatch_concurrent_views{8};
    aetherscan::mvs::DensifyQuality dense_quality{
        aetherscan::mvs::DensifyQuality::default_quality};
    unsigned dense_resolution_level{1};
    bool dense_resolution_overridden{false};
    std::filesystem::path masks_dir;
    std::string foreground_mask_source{"depth"};
    bool foreground_mask_only{false};
    std::string roi{"none"};
    float roi_margin{0.15F};
    bool roi_margin_overridden{false};
    unsigned roi_mask_dilate{5};
    unsigned roi_mask_close{0};
    unsigned roi_mask_feather{2};
    bool roi_mask_feather_overridden{false};
    bool coarse_preview_only{false};
    bool texture{false};
    bool delight{false};
    std::uint32_t atlas_resolution{2048};
    bool atlas_resolution_overridden{false};
    std::uint32_t uv_parallel_partitions{8};
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
    std::cout << options.help() << '\n'
              << "Feature backends:\n"
              << "  default   --extractor siftgpu --matcher gpu_mutual_ratio\n"
              << "  compose   any compatible --extractor × --matcher\n"
              << "  fused     --pipeline lightglue_end2end (optional recipe)\n"
              << "Modes:\n"
              << "  incremental  star initialization + PnP resection\n"
              << "  hierarchical clustered incremental SfM + Sim(3) merge\n"
              << "  global       rotation averaging + global positioning + BA\n"
              << "Dense (optional Stage A Fast MVS after SfM):\n"
              << "  --dense      PatchMatch depth + fuse -> dense.ply\n"
              << "  --gggs       train CUDA GGGS -> *_gggs.ply\n"
              << "  --splat-dataset PATH  external COLMAP/RealityCapture/OpenMVS camera data\n"
              << "  --splat-format auto|colmap|realitycapture|openmvs\n"
              << "  --colmap PATH  compatibility alias for --splat-format colmap\n"
              << "  --dense-ply PATH  replace initial points; without camera data, use internal SfM\n"
              << "  --gggs-model PATH  load a trained GGGS PLY and skip optimization\n"
              << "  --gggs-iterations N  GGGS optimizer steps (default 10000)\n"
              << "  --gggs-max-gaussians N  fixed-model cap (0 = all; default 500000)\n"
              << "  --gggs-kernel-size V  screen covariance low-pass variance; "
                 "0 disables, 0.1 matches Brush Mip\n"
              << "  --gggs-progressive-resolution BOOL  1/4 -> 1/2 -> full schedule (default true)\n"
              << "  --gggs-progressive-interval N  iterations per resolution level (default 3000)\n"
              << "  --gggs-eval-split-every N  hold out every Nth view for PSNR\n"
              << "  --gggs-use-mask BOOL  isolate the subject using masks/ or source alpha (default true)\n"
              << "  --gggs-alpha-mode masked|transparent (default transparent)\n"
              << "  --gggs-match-alpha-weight W  transparent alpha BCE weight (default 0.25)\n"
              << "  --gggs-ssim-weight W  structural loss blend (default 0.2)\n"
              << "  --gggs-depth-normal-weight W  median-depth/normal consistency (default 0.05)\n"
              << "  --gggs-mv-geo-weight W  multi-view round-trip loss (default 0.02)\n"
              << "  --gggs-mv-ncc-weight W  plane-warp NCC loss (default 0.6)\n"
              << "  --gggs-mv-neighbors N  nearest camera candidates (default 8)\n"
              << "  --gggs-mv-pixel-noise P  geometry reprojection gate (default 1px)\n"
              << "  --gggs-geometry-from-iter N  start geometry loss (default 3000)\n"
              << "  --gggs-min-scale-fraction F  minimum scale / scene extent (default 1e-4)\n"
              << "  --gggs-max-scale-fraction F  maximum scale / scene extent (default 0.002)\n"
              << "  --gggs-max-scale-ratio R  hard anisotropy clamp "
                 "(0 disables; ADC+ visual default 100)\n"
              << "  --gggs-constrain-scales=BOOL  clamp sparse KNN scales (default false)\n"
              << "  --gggs-strategy default|adc_plus|adc_igs|dense_adaptive\n"
              << "  --gggs-densification=BOOL  enable split/prune/reset (default true)\n"
              << "  --gggs-structure-freeze-iter N  freeze geometry/opacity after N (default 0)\n"
              << "  --gggs-densification-cap N  dynamic Gaussian hard cap (default 10M)\n"
              << "  --mesh       also build a surface mesh -> mesh.ply\n"
              << "  --mask-mesh PATH  load an existing PLY mesh and render masks/previews\n"
              << "  --mesh-method auto|tsdf|delaunay\n"
              << "               auto uses TSDF for GGGS, otherwise the quality preset\n"
              << "  --mesh-dist-insert-px N  global Delaunay projection spacing\n"
              << "  --mesh-free-space-support BOOL  weak-surface beta/gamma cut\n"
              << "  --mesh-free-space-quantile Q  support-scale calibration (0 disables)\n"
              << "  --mesh-target-faces N  asdiff/CGAL repair + decimate target (0 disables)\n"
              << "  --mesh-remesh BOOL  Instant Meshes before CGAL repair (default true)\n"
              << "  --mesh-tsdf-voxel-scale F  inferred voxel multiplier (-1 = auto)\n"
              << "  --mesh-tsdf-bounds-padding F  point-cloud bounds multiplier (default 2)\n"
              << "  --mesh-tsdf-support-closing-axes N  0 disables; 2 = conservative default\n"
              << "  --mesh-tsdf-frame-export-dir DIR  export exact TSDF input frames for A/B\n"
              << "  --mesh-tsdf-smooth-iters N  boundary-locked Taubin passes (default 2)\n"
              << "  --mesh-obj   additionally write the much slower ASCII OBJ\n"
              << "  --dense-quality preview|default|high (whole-pipeline preset)\n"
              << "  --masks DIR foreground masks (auto: sibling masks/ directory)\n"
              << "  --roi none|auto|FILE  OBB ROI; FILE contains 15 floats:\n"
              << "             center(3), row-major axes(9), half extents(3)\n"
              << "Texture (optional Stage B after --mesh; requires Vulkan + UVAtlas):\n"
              << "  --texture    UV unwrap + projective bake -> textured OBJ/MTL/PNG\n"
              << "  --delight    Intrinsic image delighter before bake (albedo)\n"
              << "  --atlas-resolution N  atlas size (default 2048)\n"
              << "  --uv-parallel-partitions N  concurrent UVAtlas partitioning (default 8)\n"
              << "Output formats:\n"
              << "  .mvs  OpenMVS Interface (open in Viewer)\n"
              << "  .ply  sparse XYZ point cloud\n"
              << "  with --dense: also writes dense.ply next to --output\n"
              << "  with --texture: also writes *_textured.obj/.mtl/_albedo.png\n"
              << "Log level: set AETHERSCAN_LOG_LEVEL=error|warning|info|debug|trace|off\n";
}

ReconstructCli parse_cli(int argc, char** argv) {
    cxxopts::Options options(
        "aetherscan", "High-performance Structure from Motion reconstruction");
    options.custom_help("[options]");
    options.add_options()
        ("h,help", "Print usage")
        ("i,images", "Image directory", cxxopts::value<std::string>())
        ("f,focal",
         "Initial focal length in pixels (0 = 1.2 * max(width,height); "
         "refined by view-graph consensus + BA unless trusted)",
         cxxopts::value<double>()->default_value("0"))
        ("m,mode",
         "Reconstruction mode: global (default), incremental, or hierarchical",
         cxxopts::value<std::string>()->default_value("global"))
        ("o,output", "Output path (.mvs or .ply)", cxxopts::value<std::string>())
        ("window", "Sequential neighbor window",
         cxxopts::value<std::size_t>()->default_value("3"))
        ("match-ratio", "Lowe ratio test threshold",
         cxxopts::value<float>()->default_value("0.85"))
        ("mutual-check", "Mutual match consistency check",
         cxxopts::value<bool>()->default_value("true"))
        ("sift-contrast", "SIFT contrast threshold",
         cxxopts::value<double>()->default_value("0.005"))
        ("cache-dir", "Feature cache directory (- to disable)",
         cxxopts::value<std::string>()->default_value(""))
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
        ("gggs", "Train CUDA GGGS (MVS dense input disables GS densification)",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("splat-dataset",
         "External camera dataset: COLMAP root, RealityCapture CSV/dir, or OpenMVS .mvs",
         cxxopts::value<std::string>()->default_value(""))
        ("splat-format",
         "External camera format: auto, colmap, realitycapture, or openmvs",
         cxxopts::value<std::string>()->default_value("auto"))
        ("colmap", "Compatibility alias for --splat-dataset PATH --splat-format colmap",
         cxxopts::value<std::string>()->default_value(""))
        ("dense-ply", "Dense PLY initializer for external or internal-SfM cameras",
         cxxopts::value<std::string>()->default_value(""))
        ("gggs-model", "Trained GGGS PLY to load instead of optimizing",
         cxxopts::value<std::string>()->default_value(""))
        ("gggs-iterations", "GGGS optimizer iterations",
         cxxopts::value<unsigned>()->default_value("10000"))
        ("gggs-max-gaussians", "Maximum initial Gaussians (0 = all dense points)",
         cxxopts::value<std::uint64_t>()->default_value("500000"))
        ("gggs-max-resolution", "Maximum GGGS training image dimension (0 = source)",
         cxxopts::value<unsigned>()->default_value("1920"))
        ("gggs-kernel-size",
         "Screen covariance low-pass variance (0 disables; Brush Mip uses 0.1)",
         cxxopts::value<float>()->default_value("0"))
        ("gggs-progressive-resolution",
         "Enable 1/4 -> 1/2 -> full coarse-to-fine GGGS training",
         cxxopts::value<bool>()->default_value("true")
             ->implicit_value("true"))
        ("gggs-progressive-interval",
         "Iterations per GGGS resolution level",
         cxxopts::value<unsigned>()->default_value("3000"))
        ("gggs-progressive-initial-scale",
         "Initial GGGS linear image scale",
         cxxopts::value<float>()->default_value("0.25"))
        ("gggs-view-cache-mb", "Packed RGBA8 GGGS host-view LRU budget (0 = no cache)",
         cxxopts::value<std::uint64_t>()->default_value("6144"))
        ("gggs-eval-split-every",
         "Hold out every Nth view for PSNR evaluation (0 = train all)",
         cxxopts::value<unsigned>()->default_value("0"))
        ("gggs-use-mask", "Enable pygsplat-compatible foreground-mask training",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("gggs-alpha-mode", "Mask alpha mode: masked or transparent",
         cxxopts::value<std::string>()->default_value("transparent"))
        ("gggs-match-alpha-weight", "Alpha BCE weight in transparent mode",
         cxxopts::value<float>()->default_value("0.25"))
        ("gggs-ssim-weight", "SSIM blend in the photometric loss",
         cxxopts::value<float>()->default_value("0.2"))
        ("gggs-depth-normal-weight",
         "GGGS median-depth/raster-normal consistency weight",
         cxxopts::value<float>()->default_value("0.05"))
        ("gggs-mv-geo-weight", "Multi-view depth round-trip loss weight",
         cxxopts::value<float>()->default_value("0.02"))
        ("gggs-mv-ncc-weight", "Multi-view plane-warp NCC loss weight",
         cxxopts::value<float>()->default_value("0.6"))
        ("gggs-mv-neighbors", "Number of nearest multi-view candidates",
         cxxopts::value<unsigned>()->default_value("8"))
        ("gggs-mv-pixel-noise", "Multi-view reprojection threshold in pixels",
         cxxopts::value<float>()->default_value("1"))
        ("gggs-geometry-from-iter", "Iteration to start GGGS geometry loss",
         cxxopts::value<unsigned>()->default_value("3000"))
        ("gggs-min-scale-fraction", "Minimum Gaussian scale / scene extent",
         cxxopts::value<float>()->default_value("0.0001"))
        ("gggs-max-scale-fraction", "Maximum Gaussian scale / scene extent",
         cxxopts::value<float>()->default_value("0.002"))
        ("gggs-max-scale-ratio",
         "Maximum Gaussian axis ratio (0 disables; ADC+ defaults to 100)",
         cxxopts::value<float>()->default_value("0"))
        ("gggs-constrain-scales", "Clamp sparse KNN scales to configured fractions",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("gggs-strategy", "Densification: default, adc_plus, adc_igs, dense_adaptive",
         cxxopts::value<std::string>()->default_value("default"))
        ("gggs-densification", "Enable GGGS split/prune/opacity-reset",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("gggs-structure-freeze-iter", "Freeze means/scale/quaternion/opacity after N",
         cxxopts::value<unsigned>()->default_value("0"))
        ("gggs-densification-cap", "Dynamic Gaussian hard cap",
         cxxopts::value<std::uint64_t>()->default_value("10000000"))
        ("mesh", "Build MVS mesh after densify (implies --dense)",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("mvs-mesh-only",
         "External-dataset quality gate: mesh --dense-ply directly and "
         "render source-resolution masks without GGGS",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("mask-mesh",
         "Existing PLY mesh for --mvs-mesh-only; bypasses Delaunay and renders "
         "asdiff masks plus normal-shaded previews",
         cxxopts::value<std::string>()->default_value(""))
        ("mesh-method",
         "Mesh backend: auto, tsdf, or delaunay",
         cxxopts::value<std::string>()->default_value("auto"))
        ("mesh-max-points",
         "Maximum samples inserted into global Delaunay (0 = unlimited)",
         cxxopts::value<std::uint64_t>()->default_value("2000000"))
        ("mesh-target-faces",
         "Optional asdiff/CGAL repair and decimation target (0 disables)",
         cxxopts::value<std::uint64_t>()->default_value("0"))
        ("mesh-remesh", "Run Instant Meshes before CGAL repair",
         cxxopts::value<bool>()->default_value("true")->implicit_value("true"))
        ("mesh-tsdf-voxel-scale",
         "Automatic TSDF voxel multiplier (-1 = gs2mesh default 1x)",
         cxxopts::value<float>()->default_value("-1"))
        ("mesh-tsdf-bounds-padding",
         "Point-cloud TSDF bounds multiplier",
         cxxopts::value<float>()->default_value("2"))
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
        ("mesh-free-space-quantile",
         "Quantile used to calibrate fused support to OpenMVS scale",
         cxxopts::value<float>()->default_value("0.95"))
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
         "MVS image downscale steps (0=full, 1~=half)",
         cxxopts::value<unsigned>()->default_value("1"))
        ("masks",
         "Foreground mask directory (auto, - to disable, or explicit path)",
         cxxopts::value<std::string>()->default_value("auto"))
        ("foreground-mask-source",
         "Generated GGGS mask source when --masks is absent: depth, mesh, or none",
         cxxopts::value<std::string>()->default_value("depth"))
        ("foreground-mask-only",
         "Stop after MVS depth/ROI foreground masks; do not train GGGS",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("roi", "Reconstruction ROI: none, auto, or 15-float OBB file",
         cxxopts::value<std::string>()->default_value("none"))
        ("roi-margin", "Automatic OBB fractional extent padding",
         cxxopts::value<float>()->default_value("0.15"))
        ("roi-mask-dilate", "Coarse-mesh mask dilation in working pixels",
         cxxopts::value<unsigned>()->default_value("5"))
        ("roi-mask-close",
         "Coarse-mesh mask closing radius (0 = adaptive)",
         cxxopts::value<unsigned>()->default_value("0"))
        ("roi-mask-feather",
         "Coarse-mesh mask soft-edge radius in working pixels",
         cxxopts::value<unsigned>()->default_value("2"))
        ("coarse-preview-only",
         "Deprecated compatibility flag; single-pass MVS is always used",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("texture",
         "UV unwrap + projective texture bake on MVS mesh (implies --mesh)",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("delight",
         "Run Intrinsic delighter before texture bake (implies --texture)",
         cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
        ("atlas-resolution", "Texture atlas resolution",
         cxxopts::value<std::uint32_t>()->default_value("2048"))
        ("uv-parallel-partitions",
         "Spatial UVAtlas partitioning level (1 = serial)",
         cxxopts::value<std::uint32_t>()->default_value("8"));

    const auto result = options.parse(argc, argv);
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
    cli.focal_pixels = result["focal"].as<double>();
    cli.mode = result["mode"].as<std::string>();
    cli.output = utf8_to_path(result["output"].as<std::string>());
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
    cli.gggs = result["gggs"].as<bool>();
    const std::string splat_dataset_text =
        result["splat-dataset"].as<std::string>();
    if (!splat_dataset_text.empty())
        cli.splat_dataset = utf8_to_path(splat_dataset_text);
    cli.splat_format = result["splat-format"].as<std::string>();
    const std::string colmap_text = result["colmap"].as<std::string>();
    if (!colmap_text.empty()) cli.colmap_model = utf8_to_path(colmap_text);
    if (!cli.colmap_model.empty()) {
        if (!cli.splat_dataset.empty() &&
            cli.splat_dataset != cli.colmap_model)
            throw std::invalid_argument(
                "--colmap and --splat-dataset cannot name different inputs");
        if (cli.splat_format != "auto" && cli.splat_format != "colmap")
            throw std::invalid_argument(
                "--colmap conflicts with non-COLMAP --splat-format");
        cli.splat_dataset = cli.colmap_model;
        cli.splat_format = "colmap";
    }
    const std::string dense_ply_text = result["dense-ply"].as<std::string>();
    if (!dense_ply_text.empty()) cli.dense_ply = utf8_to_path(dense_ply_text);
    const std::string gggs_model_text =
        result["gggs-model"].as<std::string>();
    if (!gggs_model_text.empty())
        cli.gggs_model = utf8_to_path(gggs_model_text);
    cli.gggs_iterations = result["gggs-iterations"].as<unsigned>();
    cli.gggs_max_gaussians =
        result["gggs-max-gaussians"].as<std::uint64_t>();
    cli.gggs_max_resolution =
        result["gggs-max-resolution"].as<unsigned>();
    cli.gggs_kernel_size = result["gggs-kernel-size"].as<float>();
    cli.gggs_progressive_resolution =
        result["gggs-progressive-resolution"].as<bool>();
    cli.gggs_progressive_interval =
        result["gggs-progressive-interval"].as<unsigned>();
    cli.gggs_progressive_initial_scale =
        result["gggs-progressive-initial-scale"].as<float>();
    cli.gggs_view_cache_mb =
        result["gggs-view-cache-mb"].as<std::uint64_t>();
    cli.gggs_eval_split_every =
        result["gggs-eval-split-every"].as<unsigned>();
    cli.gggs_use_mask = result["gggs-use-mask"].as<bool>();
    cli.gggs_alpha_mode = result["gggs-alpha-mode"].as<std::string>();
    cli.gggs_match_alpha_weight =
        result["gggs-match-alpha-weight"].as<float>();
    cli.gggs_ssim_weight = result["gggs-ssim-weight"].as<float>();
    cli.gggs_depth_normal_weight =
        result["gggs-depth-normal-weight"].as<float>();
    cli.gggs_multi_view_geo_weight =
        result["gggs-mv-geo-weight"].as<float>();
    cli.gggs_multi_view_ncc_weight =
        result["gggs-mv-ncc-weight"].as<float>();
    cli.gggs_multi_view_num = result["gggs-mv-neighbors"].as<unsigned>();
    cli.gggs_multi_view_pixel_noise =
        result["gggs-mv-pixel-noise"].as<float>();
    cli.gggs_geometry_from_iter =
        result["gggs-geometry-from-iter"].as<unsigned>();
    cli.gggs_min_scale_fraction =
        result["gggs-min-scale-fraction"].as<float>();
    cli.gggs_max_scale_fraction =
        result["gggs-max-scale-fraction"].as<float>();
    cli.gggs_max_scale_ratio =
        result["gggs-max-scale-ratio"].as<float>();
    cli.gggs_max_scale_ratio_overridden =
        result.count("gggs-max-scale-ratio") != 0;
    cli.gggs_constrain_scales = result["gggs-constrain-scales"].as<bool>();
    cli.gggs_strategy = result["gggs-strategy"].as<std::string>();
    cli.gggs_densification = result["gggs-densification"].as<bool>();
    cli.gggs_structure_freeze_iter =
        result["gggs-structure-freeze-iter"].as<unsigned>();
    cli.gggs_densification_cap =
        result["gggs-densification-cap"].as<std::uint64_t>();
    cli.mesh = result["mesh"].as<bool>();
    cli.mvs_mesh_only = result["mvs-mesh-only"].as<bool>();
    const std::string mask_mesh_text = result["mask-mesh"].as<std::string>();
    if (!mask_mesh_text.empty())
        cli.mask_mesh = utf8_to_path(mask_mesh_text);
    cli.mesh_obj = result["mesh-obj"].as<bool>();
    cli.texture = result["texture"].as<bool>();
    cli.delight = result["delight"].as<bool>();
    cli.atlas_resolution = result["atlas-resolution"].as<std::uint32_t>();
    cli.atlas_resolution_overridden =
        result.count("atlas-resolution") != 0;
    cli.uv_parallel_partitions =
        result["uv-parallel-partitions"].as<std::uint32_t>();
    cli.mesh_method = result["mesh-method"].as<std::string>();
    cli.mesh_max_points = result["mesh-max-points"].as<std::uint64_t>();
    cli.mesh_target_faces =
        result["mesh-target-faces"].as<std::uint64_t>();
    cli.mesh_remesh = result["mesh-remesh"].as<bool>();
    cli.mesh_tsdf_voxel_scale =
        result["mesh-tsdf-voxel-scale"].as<float>();
    cli.mesh_tsdf_bounds_padding =
        result["mesh-tsdf-bounds-padding"].as<float>();
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
    cli.mesh_free_space_quantile =
        result["mesh-free-space-quantile"].as<float>();
    cli.patchmatch_tile_rows =
        result["patchmatch-tile-rows"].as<unsigned>();
    cli.patchmatch_concurrent_views =
        result["patchmatch-concurrent-views"].as<unsigned>();
    const std::string dense_quality = result["dense-quality"].as<std::string>();
    if (dense_quality == "preview") {
        cli.dense_quality = aetherscan::mvs::DensifyQuality::preview;
    } else if (dense_quality == "default") {
        cli.dense_quality = aetherscan::mvs::DensifyQuality::default_quality;
    } else if (dense_quality == "high") {
        cli.dense_quality = aetherscan::mvs::DensifyQuality::high;
    } else {
        throw std::invalid_argument(
            "--dense-quality must be preview, default, or high");
    }
    cli.dense_resolution_level =
        result["dense-resolution-level"].as<unsigned>();
    cli.dense_resolution_overridden =
        result.count("dense-resolution-level") != 0;
    const std::string masks_text = result["masks"].as<std::string>();
    if (masks_text == "auto") {
        const std::filesystem::path candidate =
            cli.images_dir.parent_path() / "masks";
        if (std::filesystem::is_directory(candidate)) cli.masks_dir = candidate;
    } else if (!masks_text.empty() && masks_text != "-") {
        cli.masks_dir = utf8_to_path(masks_text);
    }
    cli.foreground_mask_source =
        result["foreground-mask-source"].as<std::string>();
    cli.foreground_mask_only =
        result["foreground-mask-only"].as<bool>();
    if (cli.foreground_mask_source != "depth" &&
        cli.foreground_mask_source != "mesh" &&
        cli.foreground_mask_source != "none")
        throw std::invalid_argument(
            "--foreground-mask-source must be depth, mesh, or none");
    if (cli.foreground_mask_source == "none" &&
        cli.masks_dir.empty())
        cli.gggs_use_mask = false;
    cli.roi = result["roi"].as<std::string>();
    cli.roi_margin = result["roi-margin"].as<float>();
    cli.roi_margin_overridden = result.count("roi-margin") != 0;
    cli.roi_mask_dilate = result["roi-mask-dilate"].as<unsigned>();
    cli.roi_mask_close = result["roi-mask-close"].as<unsigned>();
    cli.roi_mask_feather = result["roi-mask-feather"].as<unsigned>();
    cli.roi_mask_feather_overridden =
        result.count("roi-mask-feather") != 0;
    cli.coarse_preview_only = result["coarse-preview-only"].as<bool>();
    if (cli.roi_margin < 0.F || cli.roi_margin > 1.F)
        throw std::invalid_argument("--roi-margin must be in [0,1]");
    if (cli.delight) cli.texture = true;
    if (cli.texture) cli.mesh = true;
    if (cli.mesh_obj) cli.mesh = true;
    if (!cli.mask_mesh.empty()) cli.mvs_mesh_only = true;
    if (cli.mvs_mesh_only) cli.mesh = true;
    if (cli.foreground_mask_only) cli.dense = true;
    if (cli.mesh) cli.dense = true;
    const bool external_splat_dataset = !cli.splat_dataset.empty();
    if (cli.mvs_mesh_only) {
        if (!external_splat_dataset ||
            (cli.dense_ply.empty() && cli.mask_mesh.empty()))
            throw std::invalid_argument(
                "--mvs-mesh-only requires --splat-dataset and either "
                "--dense-ply or --mask-mesh");
        cli.gggs = false;
    } else if (external_splat_dataset || !cli.dense_ply.empty()) {
        cli.gggs = true;
    }
    if (cli.gggs && !external_splat_dataset) cli.dense = true;
    if (external_splat_dataset && cli.texture)
        throw std::invalid_argument(
            "--texture is not yet available in the direct external GGGS path");
#if !defined(AETHERSCAN_HAS_GGGS)
    if (cli.gggs || external_splat_dataset) {
        throw std::invalid_argument(
            "--gggs requires CUDA and AETHERSCAN_ENABLE_GGGS=ON");
    }
#else
    static_cast<void>(
        aetherscan::splat::parse_dataset_format(cli.splat_format));
#endif
    if (cli.gggs_iterations == 0)
        throw std::invalid_argument("--gggs-iterations must be positive");
    if (!std::isfinite(cli.gggs_kernel_size) ||
        cli.gggs_kernel_size < 0.F)
        throw std::invalid_argument(
            "--gggs-kernel-size must be finite and non-negative");
    if (cli.gggs_alpha_mode != "masked" &&
        cli.gggs_alpha_mode != "transparent")
        throw std::invalid_argument(
            "--gggs-alpha-mode must be masked or transparent");
    if (cli.gggs_match_alpha_weight < 0.F)
        throw std::invalid_argument(
            "--gggs-match-alpha-weight must be non-negative");
    if (cli.gggs_ssim_weight < 0.F || cli.gggs_ssim_weight > 1.F)
        throw std::invalid_argument("--gggs-ssim-weight must be in [0,1]");
    if (cli.gggs_depth_normal_weight < 0.F)
        throw std::invalid_argument(
            "--gggs-depth-normal-weight must be non-negative");
    if (cli.gggs_multi_view_geo_weight < 0.F ||
        cli.gggs_multi_view_ncc_weight < 0.F)
        throw std::invalid_argument(
            "GGGS multi-view loss weights must be non-negative");
    if (cli.gggs_multi_view_num == 0)
        throw std::invalid_argument("--gggs-mv-neighbors must be positive");
    if (!(cli.gggs_multi_view_pixel_noise > 0.F))
        throw std::invalid_argument("--gggs-mv-pixel-noise must be positive");
    if (cli.gggs_min_scale_fraction <= 0.F ||
        cli.gggs_max_scale_fraction < cli.gggs_min_scale_fraction)
        throw std::invalid_argument(
            "GGGS scale fractions must satisfy 0 < min <= max");
    if (cli.gggs_max_scale_ratio != 0.F && cli.gggs_max_scale_ratio < 1.F)
        throw std::invalid_argument(
            "--gggs-max-scale-ratio must be 0 or >= 1");
    if (cli.gggs_strategy != "default" &&
        cli.gggs_strategy != "adc_plus" &&
        cli.gggs_strategy != "adc_igs" &&
        cli.gggs_strategy != "dense_adaptive")
        throw std::invalid_argument(
            "--gggs-strategy must be default, adc_plus, adc_igs, or "
            "dense_adaptive");
    if (cli.gggs_densification_cap == 0)
        throw std::invalid_argument(
            "--gggs-densification-cap must be positive");
#if !defined(AETHERSCAN_HAS_TEXTURE)
    if (cli.texture || cli.delight) {
        throw std::invalid_argument(
            "--texture/--delight require a build with AETHERSCAN_ENABLE_TEXTURE "
            "(Vulkan SDK + asdiff_render)");
    }
#endif
    if (cli.atlas_resolution < 64) {
        throw std::invalid_argument("--atlas-resolution must be >= 64");
    }
    if (cli.uv_parallel_partitions == 0)
        throw std::invalid_argument(
            "--uv-parallel-partitions must be positive");
    if (cli.mesh_method != "auto" && cli.mesh_method != "tsdf" &&
        cli.mesh_method != "delaunay") {
        throw std::invalid_argument(
            "--mesh-method must be auto, tsdf, or delaunay");
    }
    if (cli.patchmatch_tile_rows == 0)
        throw std::invalid_argument("--patchmatch-tile-rows must be positive");
    if (cli.patchmatch_concurrent_views == 0)
        throw std::invalid_argument(
            "--patchmatch-concurrent-views must be positive");
    if (cli.mesh_dist_insert_px < -1.F || cli.mesh_dist_insert_px > 16.F)
        throw std::invalid_argument(
            "--mesh-dist-insert-px must be -1 or in [0,16]");
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

void save_ply(const aetherscan::sfm::Scene& scene, const std::filesystem::path& path) {
    std::size_t count = 0;
    for (const auto& track : scene.tracks) {
        if (track.is_triangulated()) ++count;
    }
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Failed to create PLY: " + path.string());
    output << "ply\nformat ascii 1.0\nelement vertex " << count
           << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    for (const auto& track : scene.tracks) {
        if (!track.is_triangulated()) continue;
        output << track.position.x() << ' ' << track.position.y() << ' '
               << track.position.z() << '\n';
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
    const aetherscan::sfm::Scene& scene,
    const std::filesystem::path& reconstruction_path) {
    using aetherscan::sfm::Vec2;
    using aetherscan::sfm::Vec3;

    struct ImageStats {
        std::vector<double> errors;
        double squared_sum{0.0};
    };
    std::vector<ImageStats> stats(scene.images.size());
    std::array<std::vector<double>, 5> radial_errors;
    std::array<double, 5> radial_signed_sum{};

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
              "previous_rotation_deg\n";

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
               << rotation_step << '\n';
    }

    std::sort(worst_images.begin(), worst_images.end(), std::greater<>());
    const std::size_t reported = std::min<std::size_t>(8, worst_images.size());
    for (std::size_t i = 0; i < reported; ++i) {
        const auto [p95, image_index, rms, maximum] = worst_images[i];
        aetherscan::core::Logger::instance().info(
            "sfm audit worst[", i, "] image=",
            scene.images[image_index].path.filename(), " observations=",
            stats[image_index].errors.size(), " rms_px=", rms,
            " p95_px=", p95, " max_px=", maximum);
    }
    aetherscan::core::Logger::instance().info(
        "sfm audit trajectory: step_median=", percentile(trajectory_steps, 0.5),
        " step_p95=", percentile(trajectory_steps, 0.95),
        " rotation_median_deg=", percentile(rotation_steps, 0.5),
        " rotation_p95_deg=", percentile(rotation_steps, 0.95));
    for (std::size_t bin = 0; bin < radial_errors.size(); ++bin) {
        const std::size_t count = radial_errors[bin].size();
        aetherscan::core::Logger::instance().info(
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

#if defined(AETHERSCAN_HAS_ASDIFF_MESH)
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
    aetherscan::core::Logger::instance().info(
        "instant mesh component clean: faces=", face_count, " -> ",
        compact_indices.size() / 3, " vertices=", vertex_count, " -> ",
        compact_positions.size() / 3);
    positions = std::move(compact_positions);
    indices = std::move(compact_indices);
}

bool repair_and_decimate_mesh(
    aetherscan::mvs::Mesh& mesh, const std::uint64_t target_faces,
    const bool use_instant_remesh) {
    if (target_faces == 0 || mesh.faces.empty()) return false;
    if (mesh.faces.size() <= target_faces) {
        aetherscan::core::Logger::instance().info(
            "mesh target postprocess skipped: input_faces=", mesh.faces.size(),
            " target_faces=", target_faces);
        return false;
    }
    if (mesh.vertices.size() >
        static_cast<std::size_t>(
            (std::numeric_limits<std::uint32_t>::max)()))
        throw std::runtime_error(
            "asdiff mesh preprocessing requires 32-bit vertex indices");

    aetherscan::core::StageScope stage("mesh.asdiff_repair_decimate");
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

    if (use_instant_remesh && asdiff_mesh::has_instant_meshes_backend()) {
        aetherscan::core::StageScope remesh_stage("mesh.instant_remesh");
        asdiff_mesh::RemeshOptions remesh_options;
        // PoSy=4 produces approximately one quad per requested face; the
        // binding triangulates each regular quad for the downstream pipeline.
        const std::uint64_t instant_faces =
            (std::max<std::uint64_t>)(4, (target_faces + 1) / 2);
        remesh_options.face_count = static_cast<int>(std::min<std::uint64_t>(
            instant_faces,
            static_cast<std::uint64_t>((std::numeric_limits<int>::max)())));
        remesh_options.deterministic = true;
        try {
            auto remeshed = asdiff_mesh::remesh_field_aligned(
                positions, indices, remesh_options);
            aetherscan::core::Logger::instance().info(
                "instant mesh: vertices=", positions.size() / 3, " -> ",
                remeshed.positions.size() / 3, " faces=", indices.size() / 3,
                " -> ", remeshed.indices.size() / 3);
            positions = std::move(remeshed.positions);
            indices = std::move(remeshed.indices);
            retain_largest_edge_component(positions, indices);
        } catch (const std::exception& error) {
            aetherscan::core::Logger::instance().warning(
                "Instant Meshes failed; continuing with CGAL repair: ",
                error.what());
        }
        remesh_stage.finish();
    } else if (use_instant_remesh) {
        aetherscan::core::Logger::instance().warning(
            "Instant Meshes backend unavailable; continuing with CGAL repair");
    }

    auto result = asdiff_mesh::repair_and_decimate(
        positions, indices,
        asdiff_mesh::DecimateOptions{
            static_cast<std::size_t>(target_faces), false});
    retain_largest_edge_component(result.positions, result.indices);
    aetherscan::mvs::Mesh processed;
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
        processed.vertices.size(), aetherscan::mvs::Vec3f::Zero());
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

    aetherscan::core::Logger::instance().info(
        "asdiff mesh: vertices=", mesh.vertices.size(), " -> ",
        processed.vertices.size(), " faces=", mesh.faces.size(), " -> ",
        processed.faces.size(), " target_faces=", target_faces);
    mesh = std::move(processed);
    stage.finish();
    return true;
}
#endif

#if defined(AETHERSCAN_HAS_GGGS)
bool prepare_loaded_point_cloud_roi(
    aetherscan::mvs::MvsScene& scene,
    const aetherscan::mvs::DensifyOptions& options) {
    bool configured = scene.roi.valid;
    if (!options.roi_path.empty()) {
        configured = aetherscan::mvs::detail::load_manual_roi(
            options.roi_path, scene.roi);
        if (!configured)
            throw std::runtime_error(
                "Invalid manual ROI file: " +
                options.roi_path.string());
        scene.roi_automatic = false;
    } else if (options.auto_roi && !scene.roi.valid) {
        configured =
            aetherscan::mvs::detail::estimate_automatic_roi(scene, options);
        if (!configured)
            aetherscan::core::Logger::instance().warning(
                "automatic ROI failed for loaded point cloud; "
                "GGGS/TSDF will use the uncropped scene");
    }
    if (scene.roi.valid) {
        aetherscan::mvs::detail::build_projected_foreground_masks(
            scene, options);
        std::size_t masked_views = 0;
        for (const auto& view : scene.views)
            if (view.foreground_mask.size() ==
                static_cast<std::size_t>(view.width) * view.height)
                ++masked_views;
        aetherscan::core::Logger::instance().info(
            "point-cloud ROI prepared: automatic=", scene.roi_automatic,
            " center=", scene.roi.center.transpose(),
            " half_extent=", scene.roi.half_extent.transpose(),
            " projected_mask_views=", masked_views, '/', scene.views.size());
    }
    return configured;
}

std::optional<aetherscan::mvs::Mesh> run_gggs_training(
    const aetherscan::mvs::MvsScene& scene,
    const ReconstructCli& cli,
    const bool dense_input,
    const aetherscan::mvs::DensifyOptions* mesh_options = nullptr,
    const std::filesystem::path& generated_mask_dir = {}) {
    aetherscan::splat::TrainingOptions options;
    options.iterations = cli.gggs_iterations;
    options.max_gaussians = static_cast<std::size_t>(
        std::min<std::uint64_t>(
            cli.gggs_max_gaussians,
            (std::numeric_limits<std::size_t>::max)()));
    options.input_is_dense = dense_input;
    options.initialize_scale_from_knn = true;
    options.use_source_resolution = true;
    options.max_image_dimension = cli.gggs_max_resolution;
    options.kernel_size = cli.gggs_kernel_size;
    options.progressive_resolution = cli.gggs_progressive_resolution;
    options.progressive_resolution_interval =
        cli.gggs_progressive_interval;
    options.progressive_initial_scale =
        std::clamp(cli.gggs_progressive_initial_scale, 1e-3F, 1.F);
    options.evaluation_split_every = cli.gggs_eval_split_every;
    constexpr std::uint64_t bytes_per_megabyte = 1024ULL * 1024ULL;
    options.training_view_cache_bytes = static_cast<std::size_t>(
        std::min<std::uint64_t>(
            cli.gggs_view_cache_mb >
                    (std::numeric_limits<std::uint64_t>::max)() /
                        bytes_per_megabyte
                ? (std::numeric_limits<std::uint64_t>::max)()
                : cli.gggs_view_cache_mb * bytes_per_megabyte,
            (std::numeric_limits<std::size_t>::max)()));
    for (const unsigned milestone : {1'000U, 5'000U, 10'000U, 15'000U, 30'000U})
        if (milestone < options.iterations)
            options.evaluation_iterations.push_back(milestone);
    options.densification_cap = static_cast<std::size_t>(
        std::min<std::uint64_t>(
            cli.gggs_densification_cap,
            (std::numeric_limits<std::size_t>::max)()));
    if (!dense_input)
        options.max_gaussians = options.max_gaussians == 0
            ? options.densification_cap
            : (std::min)(options.max_gaussians, options.densification_cap);
    if (cli.gggs_strategy == "adc_plus")
        options.densification_strategy =
            aetherscan::splat::DensificationStrategy::adc_plus;
    else if (cli.gggs_strategy == "adc_igs")
        options.densification_strategy =
            aetherscan::splat::DensificationStrategy::adc_igs;
    else if (cli.gggs_strategy == "dense_adaptive")
        options.densification_strategy =
            aetherscan::splat::DensificationStrategy::dense_adaptive;
    else
        options.densification_strategy =
            aetherscan::splat::DensificationStrategy::default_strategy;
    const bool dense_adaptive = dense_input &&
        options.densification_strategy ==
            aetherscan::splat::DensificationStrategy::dense_adaptive;
    if (!dense_input && options.densification_strategy ==
            aetherscan::splat::DensificationStrategy::dense_adaptive)
        throw std::invalid_argument(
            "--gggs-strategy dense_adaptive requires dense MVS input");
    options.enable_densification =
        cli.gggs_densification && (!dense_input || dense_adaptive);
    options.structure_freeze_iter = cli.gggs_structure_freeze_iter;
    if (dense_adaptive) {
        options.max_gaussians = options.max_gaussians == 0
            ? options.densification_cap
            : (std::min)(options.max_gaussians, options.densification_cap);
    }
    if (!dense_input &&
        options.densification_strategy ==
            aetherscan::splat::DensificationStrategy::adc_plus) {
        // brush-train optimizer defaults. The trainer also switches ADC+ to
        // brush's 2-NN/identity/0.5-opacity sparse initialization.
        options.means_lr = 2e-5F;
        options.scales_lr = 5e-3F;
        options.opacities_lr = 0.012F;
        options.quaternions_lr = 2e-3F;
        options.sh0_lr = 2e-3F;
        options.sh_rest_lr = 2e-4F;
        options.beta1 = 0.9F;
        options.beta2 = 0.999F;
        // Brush explicitly constructs AdamScaled with epsilon=1e-15; the
        // means scheduler decays by 100x to 2e-7 over the configured run.
        options.adam_epsilon = 1e-15F;
        // brush trains every configured SH band from the first step and keeps
        // a fixed image scale.  Letting geometry first fit quarter-resolution
        // images with only DC color gives ADC+ a strong incentive to create
        // large view-dependent sheets; later stages can recover training-view
        // PSNR without removing that erroneous geometry, so it appears as
        // floaters in extrapolated views.
        options.sh_degree_interval = 0;
        options.progressive_resolution = false;
        options.background_noise_strength = 0.1F;
    } else if (
        !dense_input &&
        options.densification_strategy !=
            aetherscan::splat::DensificationStrategy::default_strategy) {
        options.opacities_lr = 0.025F;
    }
    const std::size_t projected_mask_views =
        static_cast<std::size_t>(std::count_if(
            scene.views.begin(), scene.views.end(),
            [](const aetherscan::mvs::MvsView& view) {
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
        aetherscan::core::Logger::instance().info(
            "gggs projected mask audit: pixels=", projected_mask_pixels,
            " nonzero_fraction=",
            projected_mask_nonzero * inverse_pixels,
            " foreground_fraction=",
            projected_mask_foreground * inverse_pixels,
            " mean_coverage=",
            projected_mask_sum * inverse_pixels / 255.0);
    }
    // Respect an explicit no-mask training request. Automatically generated
    // masks are training inputs only and never constrain TSDF independently.
    options.use_mask = cli.gggs_use_mask &&
        (all_views_have_projected_masks || !generated_mask_dir.empty() ||
         !cli.masks_dir.empty());
    options.mask_dir = generated_mask_dir.empty()
        ? cli.masks_dir
        : generated_mask_dir;
    options.alpha_mode = cli.gggs_alpha_mode == "masked"
        ? aetherscan::splat::AlphaMode::masked
        : aetherscan::splat::AlphaMode::transparent;
    options.match_alpha_weight = cli.gggs_match_alpha_weight;
    options.ssim_weight = cli.gggs_ssim_weight;
    options.minimum_scale_fraction = cli.gggs_min_scale_fraction;
    options.maximum_scale_fraction = cli.gggs_max_scale_fraction;
    // pygsplat leaves Gaussian scales unconstrained for dense point-cloud
    // initialization. Retain the safety clamp only when explicitly requested;
    // forcing it for every dense input clips tangential splats and removes
    // legitimate surface coverage in sparsely sampled detail regions.
    options.constrain_scale_range = cli.gggs_constrain_scales;
    options.max_scale_ratio =
        !dense_input &&
                options.densification_strategy ==
                    aetherscan::splat::DensificationStrategy::adc_plus &&
                !cli.gggs_max_scale_ratio_overridden
            ? 0.F
            : cli.gggs_max_scale_ratio;
    options.use_mvs_depth = false;
    options.use_mvs_normals = false;
    options.use_depth_normal_loss = cli.mesh &&
        cli.gggs_depth_normal_weight > 0.F;
    options.depth_normal_weight = cli.gggs_depth_normal_weight;
    options.multi_view_geo_weight = cli.mesh
        ? cli.gggs_multi_view_geo_weight
        : 0.F;
    options.multi_view_ncc_weight = cli.mesh
        ? cli.gggs_multi_view_ncc_weight
        : 0.F;
    options.multi_view_num = cli.gggs_multi_view_num;
    options.multi_view_pixel_noise_threshold =
        cli.gggs_multi_view_pixel_noise;
    options.depth_normal_from_iter = cli.gggs_geometry_from_iter;
    // Keep structural parameters trainable while GGGS geometry supervision is
    // active. The 3k schedule now matches pygsplat's complete loss stack.
    if ((options.use_depth_normal_loss ||
         options.multi_view_geo_weight > 0.F ||
         options.multi_view_ncc_weight > 0.F) && dense_input)
        options.dense_structure_freeze_iter = 0;
    const char* effective_strategy = !options.enable_densification
        ? "disabled"
        : cli.gggs_strategy.c_str();
    aetherscan::core::Logger::instance().info(
        "gggs training: iterations=", options.iterations,
        " input=", dense_input ? "dense_points" : "sparse_points",
        " input_points=", scene.dense_cloud.points.size(),
        " max_initial_gaussians=", options.max_gaussians,
        " densification_strategy=", effective_strategy,
        " densification_enabled=", options.enable_densification,
        " structure_freeze_iter=", options.structure_freeze_iter,
        " densification_cap=", options.densification_cap,
        " dense_recycle_fraction=", options.dense_recycle_fraction,
        " dense_growth_fraction=", options.dense_growth_fraction,
        " use_mask=", options.use_mask,
        " projected_mask_views=", projected_mask_views, '/',
        scene.views.size(),
        " mask_dir=", options.mask_dir,
        " alpha_mode=", cli.gggs_alpha_mode,
        " match_alpha_weight=", options.match_alpha_weight,
        " background_noise=", options.background_noise_strength,
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
        " prefetch_views=", options.training_prefetch_views,
        " eval_split_every=", options.evaluation_split_every,
        " knn_scale=", options.initialize_scale_from_knn,
        " dense_structure_freeze_iter=",
        options.dense_structure_freeze_iter,
        " depth_normal_loss=", options.use_depth_normal_loss,
        " depth_normal_weight=", options.depth_normal_weight,
        " filter_3d=", options.use_depth_normal_loss,
        " multi_view_geo_weight=", options.multi_view_geo_weight,
        " multi_view_ncc_weight=", options.multi_view_ncc_weight,
        " multi_view_neighbours=", options.multi_view_num,
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
    if (options.evaluation_split_every != 0) {
        for (std::size_t index = 0; index < scene.views.size(); ++index)
            if (index % options.evaluation_split_every == 0)
                evaluation_views.push_back(index);
    } else {
        evaluation_views = {
            0, scene.views.size() / 2, scene.views.size() - 1};
    }
    std::sort(evaluation_views.begin(), evaluation_views.end());
    evaluation_views.erase(
        std::unique(evaluation_views.begin(), evaluation_views.end()),
        evaluation_views.end());
    const auto evaluate =
        [&](const unsigned iteration,
            const aetherscan::splat::GaussianModel& model) {
            double psnr_sum = 0.0;
            double masked_psnr_sum = 0.0;
            for (const std::size_t view_index : evaluation_views) {
                const auto render_path = out_dir /
                    (cli.output.stem().string() + "_gggs_iter_" +
                     std::to_string(iteration) + "_view_" +
                     std::to_string(view_index) + ".png");
                const auto metrics = aetherscan::splat::render_evaluation_png(
                    model, scene.views[view_index], render_path, options);
                aetherscan::core::Logger::instance().info(
                    "gggs_eval_iteration=", iteration,
                    " view=", view_index,
                    " foreground_psnr=", metrics.psnr,
                    " masked_psnr=", metrics.masked_psnr,
                    " mae=", metrics.mae,
                    " alpha_bce=", metrics.alpha_bce,
                    " alpha_coverage=", metrics.alpha_coverage,
                    " render=", render_path);
                psnr_sum += metrics.psnr;
                masked_psnr_sum += metrics.masked_psnr;
            }
            aetherscan::core::Logger::instance().info(
                "gggs_eval_iteration=", iteration,
                " views=", evaluation_views.size(),
                " average_psnr=", psnr_sum / evaluation_views.size(),
                " average_masked_psnr=",
                masked_psnr_sum / evaluation_views.size());
        };
    const auto started = std::chrono::steady_clock::now();
    aetherscan::splat::GaussianModel gaussians;
    if (!cli.gggs_model.empty()) {
        gaussians =
            aetherscan::splat::load_gaussians_ply(cli.gggs_model);
    } else {
        gaussians = aetherscan::splat::Trainer(options).train(
            scene,
            [](const aetherscan::splat::TrainingProgress& progress) {
                aetherscan::core::Logger::instance().info(
                    "gggs iteration=", progress.iteration, '/',
                    progress.total_iterations,
                    " view=", progress.view_index,
                    " gaussians=", progress.gaussian_count,
                    " tile_instances=", progress.rendered_instances,
                    " grown=", progress.grown_count,
                    " pruned=", progress.pruned_count,
                    " loss=", progress.loss,
                    " rgb=", progress.rgb_loss,
                    " alpha=", progress.alpha_loss,
                    " depth=", progress.depth_loss,
                    " normal=", progress.normal_loss,
                    " mv_geo=", progress.multi_view_geometry_loss,
                    " mv_ncc=", progress.multi_view_ncc_loss,
                    " mv_geo_pixels=", progress.multi_view_geometry_pixels,
                    " mv_ncc_pixels=", progress.multi_view_ncc_pixels,
                    " opacity_grad_mean=", progress.opacity_gradient_mean,
                    " opacity_grad_positive=",
                    progress.opacity_gradient_positive_fraction,
                    " opacity_mean=", progress.opacity_mean,
                    " resolution_scale=", progress.resolution_scale,
                    " image=", progress.image_width, 'x',
                    progress.image_height,
                    " sh_degree=", progress.active_sh_degree,
                    " step_ms=", progress.milliseconds);
                return true;
            },
            evaluate);
    }
    const auto ply = cli.gggs_model.empty()
        ? out_dir / (cli.output.stem().string() + "_gggs.ply")
        : cli.gggs_model;
    if (cli.gggs_model.empty())
        aetherscan::splat::save_gaussians_ply(gaussians, ply);
    double final_psnr_sum = 0.0;
    double final_masked_psnr_sum = 0.0;
    for (const std::size_t view_index : evaluation_views) {
        const auto render_path = out_dir /
            (cli.output.stem().string() + "_gggs_view_" +
             std::to_string(view_index) + ".png");
        const auto metrics = aetherscan::splat::render_evaluation_png(
            gaussians, scene.views[view_index], render_path, options);
        aetherscan::core::Logger::instance().info(
            "gggs_render=", render_path,
            " view=", view_index,
            " foreground_psnr=", metrics.psnr,
            " masked_psnr=", metrics.masked_psnr,
            " mae=", metrics.mae,
            " alpha_bce=", metrics.alpha_bce,
            " alpha_coverage=", metrics.alpha_coverage);
        final_psnr_sum += metrics.psnr;
        final_masked_psnr_sum += metrics.masked_psnr;
    }
    aetherscan::core::Logger::instance().info(
        "gggs_final_evaluation_views=", evaluation_views.size(),
        " average_psnr=", final_psnr_sum / evaluation_views.size(),
        " average_masked_psnr=",
        final_masked_psnr_sum / evaluation_views.size());
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    aetherscan::core::Logger::instance().info(
        "gggs_ply=", ply,
        " gaussians=", gaussians.size(),
        cli.gggs_model.empty() ? " training_s=" : " model_load_s=",
        elapsed);
    if (!cli.mesh) return std::nullopt;
    if (mesh_options == nullptr)
        throw std::invalid_argument(
            "GGGS mesh extraction requires configured MVS mesh options");

    aetherscan::splat::GggsMeshOptions extraction_options;
    extraction_options.fusion = *mesh_options;
    extraction_options.fusion.build_mesh = true;
    extraction_options.diagnostics_dir = out_dir;
    const auto mesh_started = std::chrono::steady_clock::now();
    auto extraction = aetherscan::splat::extract_gggs_mesh(
        gaussians, scene, options, extraction_options);
    const auto surface_ply = out_dir /
        (cli.output.stem().string() + "_gggs_surface.ply");
    if (!extraction.surface_cloud.points.empty())
        aetherscan::mvs::save_dense_ply(
            extraction.surface_cloud, surface_ply);
    const double mesh_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - mesh_started).count();
    aetherscan::core::Logger::instance().info(
        "gggs_surface_ply=",
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
    try {
        Utf8Argv utf8_argv(argc, argv);
        const ReconstructCli cli = parse_cli(utf8_argv.argc(), utf8_argv.argv());

        const char* configured_level = std::getenv("AETHERSCAN_LOG_LEVEL");
        const auto console_level = configured_level
            ? aetherscan::core::parse_log_level(
                  configured_level, aetherscan::core::LogLevel::info)
            : aetherscan::core::LogLevel::info;
        std::filesystem::path log_directory = cli.output.parent_path();
        if (log_directory.empty()) log_directory = std::filesystem::current_path();
        const std::filesystem::path log_path =
            aetherscan::core::Logger::instance().configure(
                log_directory, "aetherscan", console_level,
                aetherscan::core::LogLevel::trace);
        aetherscan::core::Logger::instance().info(
            "AetherScan started: mode=", cli.mode, " images_dir=", cli.images_dir,
            " output=", cli.output, " log=", log_path);

#if defined(AETHERSCAN_HAS_GGGS)
        if (!cli.splat_dataset.empty()) {
            aetherscan::splat::DatasetLoadRequest request;
            request.source = cli.splat_dataset;
            request.image_directory = cli.images_dir;
            request.initial_point_cloud = cli.dense_ply;
            request.format =
                aetherscan::splat::parse_dataset_format(cli.splat_format);
            auto loaded =
                aetherscan::splat::load_splat_dataset(request);
            for (const std::string& warning : loaded.warnings)
                aetherscan::core::Logger::instance().warning(
                    "splat dataset: ", warning);
            aetherscan::core::Logger::instance().info(
                "splat_dataset=", loaded.resolved_source,
                " format=",
                aetherscan::splat::dataset_format_name(loaded.format),
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
            aetherscan::mvs::DensifyOptions mesh_options;
            aetherscan::mvs::apply_quality_preset(
                mesh_options, cli.dense_quality);
            mesh_options.mask_dir = cli.masks_dir;
            mesh_options.auto_roi = cli.roi == "auto";
            if (cli.roi != "none" && cli.roi != "auto" && cli.roi != "-")
                mesh_options.roi_path = utf8_to_path(cli.roi);
            mesh_options.roi_margin_fraction = cli.roi_margin;
            mesh_options.auto_roi_mask_dilate_px = cli.roi_mask_dilate;
            mesh_options.auto_roi_mask_close_px = cli.roi_mask_close;
            mesh_options.auto_roi_mask_feather_px = cli.roi_mask_feather;
            mesh_options.mesh_method = aetherscan::mvs::MeshMethod::tsdf;
            mesh_options.mesh_tsdf_voxel_scale =
                cli.mesh_tsdf_voxel_scale > 0.F
                ? cli.mesh_tsdf_voxel_scale
                : 1.F;
            mesh_options.mesh_tsdf_bounds_padding =
                cli.mesh_tsdf_bounds_padding;
            mesh_options.mesh_tsdf_support_closing_axes =
                cli.mesh_tsdf_support_closing_axes;
            mesh_options.mesh_tsdf_frame_export_dir =
                cli.mesh_tsdf_frame_export_dir;
            mesh_options.mesh_tsdf_smooth_iters =
                cli.mesh_tsdf_smooth_iters;
            mesh_options.mesh_tsdf_smooth_lambda =
                cli.mesh_tsdf_smooth_lambda;
            mesh_options.mesh_tsdf_smooth_mu = cli.mesh_tsdf_smooth_mu;
            if (mesh_options.auto_roi ||
                !mesh_options.roi_path.empty()) {
                prepare_loaded_point_cloud_roi(loaded.scene, mesh_options);
            }
            if (cli.mvs_mesh_only) {
                if (!cli.mask_mesh.empty()) {
                    loaded.scene.mesh =
                        aetherscan::mvs::load_mesh_ply(cli.mask_mesh);
                    if (loaded.scene.roi.valid) {
                        const std::size_t input_vertices =
                            loaded.scene.mesh.vertices.size();
                        const std::size_t input_faces =
                            loaded.scene.mesh.faces.size();
                        aetherscan::mvs::DensifyOptions crop_options;
                        crop_options.mesh_clean = false;
                        aetherscan::mvs::detail::clean_mesh(
                            loaded.scene.mesh, crop_options,
                            &loaded.scene.roi);
                        aetherscan::core::Logger::instance().info(
                            "external mask mesh cropped to ROI: vertices=",
                            input_vertices, " -> ",
                            loaded.scene.mesh.vertices.size(), " faces=",
                            input_faces, " -> ",
                            loaded.scene.mesh.faces.size());
                    }
                } else {
                    mesh_options.mesh_method =
                        aetherscan::mvs::MeshMethod::delaunay_cut;
                    mesh_options.build_mesh = true;
                    mesh_options.mesh_max_points = cli.mesh_max_points;
                    if (cli.mesh_dist_insert_px >= 0.F)
                        mesh_options.mesh_dist_insert_px =
                            cli.mesh_dist_insert_px;
                    mesh_options.mesh_use_free_space_support =
                        cli.mesh_free_space_support;
                    mesh_options.mesh_k_free_space_calibration_quantile =
                        std::clamp(
                            cli.mesh_free_space_quantile, 0.F, 0.999F);
                    aetherscan::mvs::reconstruct_mesh(
                        loaded.scene, mesh_options);
                }
                aetherscan::mvs::save_mesh_ply(
                    loaded.scene.mesh, cli.output);
#if defined(AETHERSCAN_HAS_TEXTURE)
                const auto mask_directory =
                    cli.output.parent_path() /
                    (cli.output.stem().string() + "_masks");
                aetherscan::texture::MeshMaskOptions mask_options;
                mask_options.preview_directory =
                    cli.output.parent_path() /
                    (cli.output.stem().string() + "_mesh_previews");
                const auto mask_summary =
                    aetherscan::texture::render_mesh_foreground_masks(
                        loaded.scene, mask_directory, mask_options);
                aetherscan::core::Logger::instance().info(
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
            auto mesh = run_gggs_training(
                loaded.scene, cli, loaded.initial_points_dense,
                cli.mesh ? &mesh_options : nullptr);
            if (mesh) {
#if defined(AETHERSCAN_HAS_ASDIFF_MESH)
                if (cli.mesh_target_faces > 0)
                    repair_and_decimate_mesh(
                        *mesh, cli.mesh_target_faces, cli.mesh_remesh);
#endif
                const auto mesh_path = cli.output.parent_path() /
                    (cli.output.stem().string() + "_gggs_mesh.ply");
                aetherscan::mvs::save_mesh_ply(*mesh, mesh_path);
                if (cli.mesh_obj)
                    aetherscan::mvs::save_mesh_obj(
                        *mesh, mesh_path.parent_path() /
                            (mesh_path.stem().string() + ".obj"));
                aetherscan::core::Logger::instance().info(
                    "mesh_ply=", mesh_path, " faces=", mesh->faces.size());
            }
            return 0;
        }
#endif

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

        aetherscan::sfm::ReconstructionConfig config;
        if (cli.mode == "global")
            config.mode = aetherscan::sfm::ReconstructionMode::global;
        else if (cli.mode == "hierarchical")
            config.mode = aetherscan::sfm::ReconstructionMode::hierarchical;
        else
            config.mode = aetherscan::sfm::ReconstructionMode::incremental;
        config.frontend.focal_pixels = cli.focal_pixels;
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

        const auto started = std::chrono::steady_clock::now();
        aetherscan::sfm::Scene scene;
        const auto summary = aetherscan::sfm::reconstruct(scene, files, config);
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                .count();

        if (!summary.valid) {
            aetherscan::core::Logger::instance().error(
                "reconstruction failed: registered=", summary.registered_views,
                '/', scene.images.size(), " landmarks=", summary.landmarks);
            return 2;
        }

        const auto diagnostics_path = write_sfm_diagnostics(scene, cli.output);
        aetherscan::core::Logger::instance().info(
            "sfm_diagnostics=", diagnostics_path);

        if (lower_extension(cli.output) == ".mvs") {
            aetherscan::sfm::export_openmvs_interface(scene, cli.output);
        } else if (lower_extension(cli.output) == ".ply") {
            save_ply(scene, cli.output);
            auto mvs_path = cli.output.parent_path() / cli.output.stem();
            mvs_path += ".mvs";
            aetherscan::sfm::export_openmvs_interface(scene, mvs_path);
            aetherscan::core::Logger::instance().info("mvs=", mvs_path);
        } else {
            throw std::invalid_argument("Output must end with .mvs or .ply");
        }

        if (cli.dense) {
            aetherscan::mvs::DensifyOptions densify_opts;
            aetherscan::mvs::apply_quality_preset(
                densify_opts, cli.dense_quality);
            if (cli.dense_resolution_overridden)
                densify_opts.resolution_level = cli.dense_resolution_level;
            densify_opts.mask_dir = cli.masks_dir;
            const bool generated_depth_masks =
                (cli.gggs || cli.foreground_mask_only) &&
                cli.gggs_use_mask &&
                cli.masks_dir.empty() &&
                cli.foreground_mask_source == "depth";
            if (generated_depth_masks &&
                !cli.dense_resolution_overridden)
                densify_opts.resolution_level = 0;
            densify_opts.auto_roi =
                cli.roi == "auto" || generated_depth_masks;
            densify_opts.build_depth_roi_masks = generated_depth_masks;
            if (cli.roi != "none" && cli.roi != "auto" && cli.roi != "-")
                densify_opts.roi_path = utf8_to_path(cli.roi);
            densify_opts.roi_margin_fraction = cli.roi_margin;
            densify_opts.auto_roi_ground_margin_fraction =
                generated_depth_masks
                ? (cli.roi_margin_overridden ? cli.roi_margin : 0.25F)
                : cli.roi_margin;
            densify_opts.auto_roi_mask_dilate_px = cli.roi_mask_dilate;
            densify_opts.auto_roi_mask_close_px = cli.roi_mask_close;
            densify_opts.auto_roi_mask_feather_px =
                generated_depth_masks &&
                    !cli.roi_mask_feather_overridden
                ? 1U
                : cli.roi_mask_feather;
            densify_opts.coarse_preview_only = cli.coarse_preview_only;
            {
                const std::filesystem::path diagnostic_dir =
                    cli.output.parent_path().empty()
                    ? std::filesystem::current_path()
                    : cli.output.parent_path();
                densify_opts.coarse_mesh_output_path =
                    diagnostic_dir /
                    (cli.output.stem().string() + "_coarse_mesh.ply");
            }
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
            densify_opts.mesh_k_free_space_calibration_quantile =
                std::clamp(cli.mesh_free_space_quantile, 0.F, 0.999F);
            densify_opts.patchmatch_tile_rows = cli.patchmatch_tile_rows;
            densify_opts.patchmatch_concurrent_views =
                cli.patchmatch_concurrent_views;
            // The default GGGS mask path comes from ROI-filtered MVS depth,
            // so a Delaunay surface is no longer a prerequisite. Build the
            // MVS mesh only when explicitly needed by a non-GGGS product,
            // texture baking, or the legacy mesh-mask diagnostic path.
            densify_opts.build_mesh =
                cli.texture || (!cli.gggs && cli.mesh) ||
                (cli.gggs && cli.gggs_use_mask &&
                 cli.masks_dir.empty() &&
                 cli.foreground_mask_source == "mesh");
            if (!densify_opts.build_mesh) {
                densify_opts.mesh_method = aetherscan::mvs::MeshMethod::none;
            } else if (cli.mesh_method == "tsdf") {
                densify_opts.mesh_method =
                    aetherscan::mvs::MeshMethod::tsdf;
            } else {
                densify_opts.mesh_method =
                    aetherscan::mvs::MeshMethod::delaunay_cut;
            }
            densify_opts.geometric_consistency = true;
            densify_opts.thread_count = scene.thread_count;
            aetherscan::core::Logger::instance().info(
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
                        aetherscan::mvs::MeshMethod::delaunay_cut
                    ? "delaunay"
                    : densify_opts.mesh_method ==
                              aetherscan::mvs::MeshMethod::tsdf
                          ? "tsdf"
                          : "none",
                " mask_border_px=", densify_opts.mask_border_px,
                " foreground_mask_source=",
                densify_opts.build_depth_roi_masks
                    ? "depth"
                    : densify_opts.build_mesh &&
                              cli.foreground_mask_source == "mesh"
                          ? "mesh"
                          : "external_or_none",
                " roi=", densify_opts.auto_roi
                    ? "auto"
                    : densify_opts.roi_path.empty() ? "none" : "manual",
                " grazing_weight_floor=",
                densify_opts.grazing_weight_floor);
            if (!densify_opts.mask_dir.empty())
                aetherscan::core::Logger::instance().info(
                    "mvs masks=", densify_opts.mask_dir);

            const auto dense_started = std::chrono::steady_clock::now();
            aetherscan::mvs::MvsScene mvs_scene;
            if (!cli.dense_ply.empty()) {
                // No external camera metadata: keep the built-in SfM poses,
                // but use the caller's dense initialization directly.
                mvs_scene =
                    aetherscan::mvs::build_mvs_scene(scene, densify_opts);
                mvs_scene.dense_cloud =
                    aetherscan::mvs::load_dense_ply(cli.dense_ply);
#if defined(AETHERSCAN_HAS_GGGS)
                if (densify_opts.auto_roi ||
                    !densify_opts.roi_path.empty())
                    prepare_loaded_point_cloud_roi(
                        mvs_scene, densify_opts);
#endif
                aetherscan::core::Logger::instance().info(
                    "splat cameras=internal_sfm dense_ply=", cli.dense_ply,
                    " points=", mvs_scene.dense_cloud.points.size());
            } else {
                mvs_scene =
                    aetherscan::mvs::densify_from_sfm(scene, densify_opts);
            }
            const double dense_elapsed = std::chrono::duration<double>(
                                             std::chrono::steady_clock::now() -
                                             dense_started)
                                             .count();

            const std::filesystem::path out_dir =
                cli.output.parent_path().empty()
                    ? std::filesystem::current_path()
                    : cli.output.parent_path();
            const auto dense_ply = out_dir / (cli.output.stem().string() + "_dense.ply");
            aetherscan::mvs::save_dense_ply(mvs_scene.dense_cloud, dense_ply);
            if (mvs_scene.roi.valid) {
                const auto roi_path =
                    out_dir / (cli.output.stem().string() + "_roi.txt");
                aetherscan::mvs::save_roi(mvs_scene.roi, roi_path);
                aetherscan::core::Logger::instance().info(
                    "mvs roi=", roi_path,
                    " automatic=", mvs_scene.roi_automatic);
            }
            aetherscan::core::Logger::instance().info(
                "dense_ply=", dense_ply,
                " points=", mvs_scene.dense_cloud.points.size(),
                " densify_s=", dense_elapsed);

            std::filesystem::path effective_mask_dir = cli.masks_dir;
            if (!mvs_scene.mesh.faces.empty()) {
                const auto mvs_mesh_path =
                    out_dir /
                    (cli.output.stem().string() + "_mvs_mesh.ply");
                aetherscan::mvs::save_mesh_ply(
                    mvs_scene.mesh, mvs_mesh_path);
                aetherscan::core::Logger::instance().info(
                    "mvs_mesh_ply=", mvs_mesh_path,
                    " vertices=", mvs_scene.mesh.vertices.size(),
                    " faces=", mvs_scene.mesh.faces.size());
            }

#if defined(AETHERSCAN_HAS_TEXTURE)
            if ((cli.gggs || cli.foreground_mask_only) &&
                cli.gggs_use_mask &&
                effective_mask_dir.empty()) {
                const bool depth_masks_ready = std::all_of(
                    mvs_scene.views.begin(), mvs_scene.views.end(),
                    [](const aetherscan::mvs::MvsView& view) {
                        return view.foreground_mask.size() ==
                            static_cast<std::size_t>(view.width) *
                                view.height;
                    });
                if (depth_masks_ready) {
                    const auto preview_directory =
                        out_dir /
                        (cli.output.stem().string() +
                         "_depth_roi_masks");
                    const auto mask_summary =
                        aetherscan::texture::export_view_foreground_masks(
                            mvs_scene, preview_directory);
                    aetherscan::core::Logger::instance().info(
                        "gggs_mask_source=mvs_depth_roi",
                        " views=", mask_summary.image_count,
                        " pixels=", mask_summary.pixel_count,
                        " preview_directory=", preview_directory);
                } else {
                    if (mvs_scene.mesh.faces.empty())
                        throw std::runtime_error(
                            "GGGS foreground masks require MVS depth ROI, "
                            "an MVS mesh, or --masks");
                    effective_mask_dir =
                        out_dir /
                        (cli.output.stem().string() +
                         "_mvs_mesh_masks");
                    aetherscan::texture::MeshMaskOptions mask_options;
                    mask_options.supersample = 2;
                    const auto mask_summary =
                        aetherscan::texture::render_mesh_foreground_masks(
                            mvs_scene, effective_mask_dir, mask_options);
                    aetherscan::core::Logger::instance().info(
                        "gggs_mask_source=mvs_mesh_asdiff",
                        " views=", mask_summary.image_count,
                        " pixels=", mask_summary.pixel_count,
                        " soft_edge_pixels=",
                        mask_summary.soft_edge_pixels);
                }
            }
#else
            if (cli.gggs && cli.gggs_use_mask &&
                effective_mask_dir.empty())
                throw std::runtime_error(
                    "GGGS mesh-mask training requires asdiff_render; "
                    "enable AETHERSCAN_ENABLE_TEXTURE");
#endif

            if (cli.foreground_mask_only) {
                aetherscan::core::Logger::instance().info(
                    "foreground mask quality gate complete; GGGS skipped");
                return 0;
            }

#if defined(AETHERSCAN_HAS_GGGS)
            if (cli.gggs) {
                auto gggs_mesh_options = densify_opts;
                gggs_mesh_options.mesh_method =
                    aetherscan::mvs::MeshMethod::tsdf;
                auto gggs_mesh = run_gggs_training(
                    mvs_scene, cli, true, &gggs_mesh_options,
                    effective_mask_dir);
                if (gggs_mesh) mvs_scene.mesh = std::move(*gggs_mesh);
            }
#endif

            if (cli.mesh && !mvs_scene.mesh.faces.empty()) {
#if defined(AETHERSCAN_HAS_ASDIFF_MESH)
                if (cli.gggs && cli.mesh_target_faces > 0) {
                    try {
                        repair_and_decimate_mesh(
                            mvs_scene.mesh, cli.mesh_target_faces,
                            cli.mesh_remesh);
                    } catch (const std::exception& error) {
                        aetherscan::core::Logger::instance().warning(
                            "asdiff mesh postprocess failed; retaining cleaned "
                            "mesh: ", error.what());
                    }
                }
#else
                if (cli.gggs && cli.mesh_target_faces > 0)
                    aetherscan::core::Logger::instance().warning(
                        "asdiff mesh postprocess unavailable (CGAL mesh tools "
                        "were not built); retaining cleaned mesh");
#endif
                const std::string mesh_tag = cli.gggs
                    ? "_gggs_mesh"
                    : "_mesh";
                const auto mesh_ply =
                    out_dir / (cli.output.stem().string() + mesh_tag + ".ply");
                const auto mesh_obj =
                    out_dir / (cli.output.stem().string() + mesh_tag + ".obj");
                aetherscan::mvs::save_mesh_ply(mvs_scene.mesh, mesh_ply);
                if (cli.mesh_obj)
                    aetherscan::mvs::save_mesh_obj(mvs_scene.mesh, mesh_obj);
                aetherscan::core::Logger::instance().info(
                    "mesh_ply=", mesh_ply,
                    cli.mesh_obj ? " mesh_obj=" : "",
                    cli.mesh_obj ? mesh_obj.string() : std::string{},
                    " active_mesh=", cli.gggs ? "gggs" : "mvs",
                    " faces=", mvs_scene.mesh.faces.size());

#if defined(AETHERSCAN_HAS_TEXTURE)
                if (cli.texture) {
                    aetherscan::texture::TextureOptions tex_opts;
                    tex_opts.atlas_resolution = cli.atlas_resolution;
                    tex_opts.uv_parallel_partitions =
                        cli.uv_parallel_partitions;
                    tex_opts.delight = cli.delight;
                    tex_opts.mask_dir = effective_mask_dir;
                    if (cli.dense_quality ==
                        aetherscan::mvs::DensifyQuality::high) {
                        tex_opts.blend_mode =
                            aetherscan::texture::BlendMode::weighted_average;
                        tex_opts.visibility_mode = aetherscan::texture::
                            VisibilityMode::hybrid_ray_query;
                    } else if (
                        cli.dense_quality ==
                        aetherscan::mvs::DensifyQuality::preview) {
                        if (!cli.atlas_resolution_overridden)
                            tex_opts.atlas_resolution = 1024U;
                        tex_opts.visibility_mode =
                            aetherscan::texture::VisibilityMode::shadow_map;
                    }
                    const auto textured_stem =
                        out_dir / (cli.output.stem().string() + "_textured");
                    const auto tex_started =
                        std::chrono::steady_clock::now();
                    aetherscan::texture::bake_and_export(
                        mvs_scene, textured_stem, tex_opts);
                    const double tex_elapsed =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - tex_started)
                            .count();
                    aetherscan::core::Logger::instance().info(
                        "textured_obj=", textured_stem.string() + ".obj",
                        " delight=", cli.delight,
                        " atlas=", tex_opts.atlas_resolution,
                        " texture_s=", tex_elapsed);
                }
#endif
            }
        }

        aetherscan::core::Logger::instance().info(
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
        aetherscan::core::Logger::instance().error("error: ", error.what());
        return 1;
    }
}
