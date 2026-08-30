#pragma once

#include "mvs/types.hpp"
#include "mvs/options.hpp"
#include "splat/options.hpp"
#include "splat/rasterizer.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <vector>

namespace aetherscan::splat {

struct TrainingProgress {
    unsigned iteration{};
    unsigned total_iterations{};
    std::size_t gaussian_count{};
    std::size_t rendered_instances{};
    std::size_t view_index{};
    std::size_t grown_count{};
    std::size_t pruned_count{};
    float loss{};
    float rgb_loss{};
    float alpha_loss{};
    float depth_loss{};
    float normal_loss{};
    float opacity_gradient_mean{};
    float opacity_gradient_positive_fraction{};
    float opacity_mean{};
    double milliseconds{};
    float multi_view_geometry_loss{};
    float multi_view_ncc_loss{};
    std::size_t multi_view_geometry_pixels{};
    std::size_t multi_view_ncc_pixels{};
    float resolution_scale{1.F};
    std::uint32_t image_width{};
    std::uint32_t image_height{};
    unsigned active_sh_degree{};
    unsigned multi_view_interval{1};
    float multi_view_depth_consistency{};
};

struct RenderMetrics {
    float mae{};
    // Error over foreground samples only. This is stricter than pygsplat's
    // validation metric because it does not average masked-out zeros.
    float psnr{};
    // pygsplat-compatible PSNR: prediction and target are masked, then MSE is
    // averaged over every image pixel.
    float masked_psnr{};
    float alpha_bce{};
    float alpha_coverage{};
};

struct TrainingPreview {
    unsigned iteration{};
    std::size_t view_index{};
    std::uint32_t width{};
    std::uint32_t height{};
    // Top-left-origin, interleaved 8-bit RGB. Ownership is transferred to the
    // callback so an editor can enqueue the frame without retaining tensors.
    std::vector<std::uint8_t> rgb;
};

struct SplatMeshOptions {
    // GS-2M/pygsplat fallback threshold when the dataset has no input mask.
    // With a mask, the mask alone defines valid extraction pixels.
    float alpha_threshold{0.5F};
    // Zero selects gs2mesh.py's automatic cutoff (2 * camera scene extent).
    float max_depth{0.F};
    // Optional directory for representative median-depth, normal and alpha
    // PNGs used to audit geometry before TSDF fusion.
    std::filesystem::path diagnostics_dir;
    // Optional MVS-inspired extraction guard. Values below -1 disable it,
    // matching pygsplat/GS-2M whose abs-dot 100-degree test rejects no sample.
    // Set 0.5 explicitly to request the former strict 60-degree filter.
    float min_depth_normal_cosine{-2.F};
    mvs::DensifyOptions fusion;
};

struct SplatMeshResult {
    mvs::DenseCloud surface_cloud;
    mvs::Mesh mesh;
    std::size_t valid_depth_pixels{};
};

struct PamMeshOptions {
    // Candidate points are sampled from a coarse seed mesh with probability
    // proportional to face area / nearest-visible-camera-distance^2.
    std::size_t max_points{1'000'000};
    // Maximum number of Gaussian pivot vertices used by the initial
    // tetra_triangulation stage. Two-pivot extraction uses a Gaussian center
    // and one learned-normal offset per selected Gaussian.
    std::size_t pivot_max_points{1'000'000};
    // GaussianWrapping's default learned-normal pivot displacement.
    float pivot_std_factor{3.F};
    // Reserve part of the candidate budget for opacity/anisotropy-weighted
    // Gaussian means offset along their learned normal.  Unlike surface-area
    // sampling, this gives sub-pixel wires and other small disconnected
    // structures a chance to enter the Delaunay complex.
    float gaussian_seed_fraction{0.F};
    unsigned oversampling_factor{2};
    unsigned max_resample_rounds{4};
    unsigned refinement_steps{10};
    unsigned vector_field_neighbors{32};
    unsigned points_per_tetrahedron{10};
    float occupancy_iso_value{0.5F};
    float vacancy_threshold{0.1F};
    float minimum_gradient_norm_squared{0.5F};
    float refinement_step{0.5F};
    float mask_background_threshold{0.01F};
    std::size_t occupancy_chunk_size{250'000};
    unsigned seed{0};
};

struct PamMeshResult {
    mvs::Mesh seed_mesh;
    mvs::Mesh mesh;
    mvs::DenseCloud candidate_cloud;
    std::size_t tetrahedron_count{};
    std::size_t occupied_tetrahedron_count{};
};

using ProgressCallback = std::function<bool(const TrainingProgress&)>;
using EvaluationCallback =
    std::function<void(unsigned iteration, const GaussianModel& model)>;
using PreviewCallback = std::function<void(TrainingPreview)>;
// Called while the raster color tensor is still resident on CUDA. The tensor
// is planar float RGB [3,H,W] and is valid only for the duration of the call.
// This is the zero-copy hook used by CUDA/Vulkan external-memory interop.
using DevicePreviewCallback = std::function<void(
    unsigned iteration, std::size_t view_index, const Camera& camera,
    const tinytensor::Tensor& color)>;

Camera camera_from_mvs_view(const mvs::MvsView& view);

GaussianModel initialize_from_dense_cloud(
    const mvs::MvsScene& scene, const TrainingOptions& options = {});

TrainingView make_training_view(
    const mvs::MvsView& view, const TrainingOptions& options = {});

class Trainer {
public:
    explicit Trainer(TrainingOptions options = {});

    GaussianModel train(
        const mvs::MvsScene& scene, ProgressCallback progress = {},
        EvaluationCallback evaluate = {}, PreviewCallback preview = {},
        DevicePreviewCallback device_preview = {}) const;

private:
    TrainingOptions options_;
};

void save_gaussians_ply(
    const GaussianModel& model, const std::filesystem::path& path);
GaussianModel load_gaussians_ply(const std::filesystem::path& path);

inline constexpr std::uint32_t k_gaussian_chunk_version = 1;
std::vector<std::uint8_t> encode_gaussians(const GaussianModel& model);
GaussianModel decode_gaussians(std::span<const std::uint8_t> bytes);

// Blocks until the process is killed. Renders whenever the orbit-camera
// sidecar revision changes, and submits the CUDA color tensor to the
// editor preview callback.
void run_orbit_preview(
    const GaussianModel& model,
    const std::filesystem::path& camera_file,
    DevicePreviewCallback device_preview,
    float kernel_size = 0.F,
    const std::filesystem::path& vis_file = {});

RenderMetrics render_evaluation_png(
    const GaussianModel& model, const mvs::MvsView& view,
    const std::filesystem::path& path,
    const TrainingOptions& options = {});

// Render learned median-depth/normal maps, fuse them across the registered
// cameras, then run the selected surface backend and topology cleanup. The
// geometry losses follow GGGS ideas, but this is AetherScan's splat API.
SplatMeshResult extract_splat_mesh(
    const GaussianModel& model, const mvs::MvsScene& scene,
    const TrainingOptions& training_options = {},
    const SplatMeshOptions& mesh_options = {});

// GaussianWrapping Primal Adaptive Meshing (PAM). An empty seed_mesh first
// builds the native learned-normal Gaussian pivot mesh using Delaunay
// tetra_triangulation and marching tetrahedra. PAM then samples that coarse
// mesh, projects candidates onto the occupancy isosurface, tetrahedralizes
// again, and returns the occupied/free-space boundary.
PamMeshResult extract_pam_mesh(
    const GaussianModel& model, const mvs::MvsScene& scene,
    const mvs::Mesh& seed_mesh,
    const TrainingOptions& training_options = {},
    const PamMeshOptions& pam_options = {});

}  // namespace aetherscan::splat
