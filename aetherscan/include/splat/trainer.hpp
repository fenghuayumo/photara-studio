#pragma once

#include "mvs/types.hpp"
#include "mvs/options.hpp"
#include "splat/options.hpp"
#include "splat/rasterizer.hpp"

#include <filesystem>
#include <functional>
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

struct GggsMeshOptions {
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

struct GggsMeshResult {
    mvs::DenseCloud surface_cloud;
    mvs::Mesh mesh;
    std::size_t valid_depth_pixels{};
};

using ProgressCallback = std::function<bool(const TrainingProgress&)>;
using EvaluationCallback =
    std::function<void(unsigned iteration, const GaussianModel& model)>;

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
        EvaluationCallback evaluate = {}) const;

private:
    TrainingOptions options_;
};

void save_gaussians_ply(
    const GaussianModel& model, const std::filesystem::path& path);

RenderMetrics render_evaluation_png(
    const GaussianModel& model, const mvs::MvsView& view,
    const std::filesystem::path& path,
    const TrainingOptions& options = {});

// Render GGGS median-depth/normal maps, fuse them across the registered
// cameras, then run the selected MVS surface backend and topology cleanup.
// This path is independent of PatchMatch photometric NCC.
GggsMeshResult extract_gggs_mesh(
    const GaussianModel& model, const mvs::MvsScene& scene,
    const TrainingOptions& training_options = {},
    const GggsMeshOptions& mesh_options = {});

}  // namespace aetherscan::splat
