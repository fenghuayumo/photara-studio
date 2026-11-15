#pragma once
#include "splat/rasterizer.hpp"
namespace photara::splat {
class CudaRasterizer {
  public:
    // Sum over pixels of (opacity * d(sum RGB)/d alpha)^2, as in
    // Faster-GS / Speedy-Splat. Evaluates canonical colours, independent of
    // training residuals, and does not update parameters or Adam moments.
    // GaussianPOP sums the squared per-channel counterfactual deletion change.
    tinytensor::Tensor pruning_scores(
        const GaussianModel& model,
        const Camera& camera,
        RasterizeOptions options = {}, bool gaussian_pop = false) const;
    RenderResult forward(
        const GaussianModel& model,
        const Camera& camera,
        const RasterizeOptions& options = {}) const;

    // CUDA 3DGS auxiliary colors, after source.backward() and before updating
    // model geometry. Consumes source's exclusive context and overwrites its
    // pixel snapshots, retaining its owned geometry/visibility attachments.
    // Camera and raster settings are inherited from source. In-place model
    // geometry changes between the calls are forbidden.
    RenderResult forward_colors(
        const GaussianModel& model,
        RenderResult& source,
        const tinytensor::Tensor& colors_precomp) const;

    ModelGradients backward(
        const GaussianModel& model,
        const RenderResult& rendered,
        const tinytensor::Tensor& grad_color,
        const tinytensor::Tensor& grad_alpha,
        const tinytensor::Tensor& grad_depth,
        const tinytensor::Tensor& grad_normal,
        const tinytensor::Tensor& densify_map = {},
        const SHAdamUpdate* sh_adam = nullptr,
        const StructureAdamUpdate* structure_adam = nullptr) const;

    DepthSampleResult sample_depth(
        const GaussianModel& model,
        const tinytensor::Tensor& world_points,
        const Camera& camera,
        const RasterizeOptions& options = {}) const;

    DepthSampleGradients sample_depth_backward(
        const GaussianModel& model,
        const DepthSampleResult& sampled,
        const tinytensor::Tensor& grad_camera_points) const;

    OccupancyResult evaluate_occupancy(
        const GaussianModel& model,
        const tinytensor::Tensor& world_points,
        const Camera& camera,
        const RasterizeOptions& options = {}) const;

    // Budgets bound resident draw lists on both devices. Views exceeding both
    // budgets are prepared on demand, with identical query results. One active
    // view can temporarily reside beyond the permanent budget until a camera
    // switch, so consecutive point chunks share preparation and upload.
    // Zero selects a budget from currently available VRAM; SIZE_MAX forces host
    // storage (useful for verifying the spill path).
    OccupancyCache prepare_occupancy_cache(
        const GaussianModel& model,
        const std::vector<Camera>& cameras,
        const RasterizeOptions& options = {},
        std::size_t device_budget = 0,
        std::size_t host_budget = std::size_t{4} << 30) const;
    OccupancyResult evaluate_occupancy(
        OccupancyCache& cache,
        std::size_t camera_index,
        const tinytensor::Tensor& world_points) const;
    OccupancyCacheStats occupancy_cache_stats(const OccupancyCache& cache) const;

    // Packed [N,2] xy pixel centres from the last forward, or null.
    const float* projected_mean2d(const RenderResult& rendered) const;

};
}
