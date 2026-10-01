#pragma once

#include "splat/types.hpp"

#include <limits>

namespace photara::splat {

struct RasterizeOptions {
    unsigned active_sh_degree{0};
    std::array<float, 3> background{0.F, 0.F, 0.F};
    float kernel_size{0.F};
    float scale_modifier{1.F};
    bool require_depth{true};
    // Vulkan keeps its live render frame in the backend context. Training can
    // skip color/alpha materialization unless a host preview needs the copies.
    bool copy_attachments{true};
    // Vulkan colour correction only needs an owned RGB tensor. Skipping the
    // alpha copy also avoids allocating a matching zero-gradient tensor.
    bool copy_color_only{false};
    // Vulkan training consumes contribution visibility only after backward.
    // Deferring its uint32-to-float conversion lets that dispatch join the
    // densification/optimizer batch instead of forcing a queue hand-off before
    // the photometric loss. Regular render callers keep the eager default.
    bool defer_visibility{false};
    bool debug{false};
    // Multi-view point queries (sample_depth): median-depth seed window in
    // world units, and the bracket width at which refinement stops. Zero keeps
    // the wide default search (+/-200 with 8 refinements), which is far more
    // precision than a depth-consistency loss can use.
    float point_depth_bracket{0.F};
    float point_depth_tolerance{0.F};
    // Optional [N,3] world-space feature colors. When present, SH evaluation
    // is bypassed and backward returns ModelGradients::colors_precomp.
    tinytensor::Tensor colors_precomp;
};

struct SHAdamUpdate {
    tinytensor::Tensor first;
    float lr{}, rest_lr{}, beta1{.9F}, beta2{.999F};
    float correction1{1.F}, correction2{1.F}, epsilon{1e-8F};
    float regularization_weight{};
    // `first` stores FP16 normalized update u; packed/bounds store the
    // uint8 log-second state.
    tinytensor::Tensor packed;
    tinytensor::Tensor bounds;
};

struct StructureAdamUpdate {
    tinytensor::Tensor means_first, means_second;
    tinytensor::Tensor scales_first, scales_second;
    tinytensor::Tensor rotations_first, rotations_second;
    tinytensor::Tensor opacity_first, opacity_second;
    float means_lr{}, scales_lr{}, quaternions_lr{}, opacities_lr{};
    float beta1{.9F}, beta2{.999F};
    float correction1{1.F}, correction2{1.F}, epsilon{1e-15F};
    float minimum_log_scale{-std::numeric_limits<float>::infinity()};
    float maximum_log_scale{std::numeric_limits<float>::infinity()};
    float max_log_scale_ratio{};
    float opacity_reg{};
    float log_scale_reg{};
    float shape_scale_reg{};
    float shape_erank_reg{};
    float shape_erank_s3_reg{};
    float shape_quat_norm_reg{};
    // Logging-only: retain the fused opacity-logit gradient without bringing
    // back the other structure-gradient tensors.
    bool capture_opacity_gradient{};
};

class Rasterizer {
public:
    RenderResult forward(
        const GaussianModel& model, const Camera& camera,
        const RasterizeOptions& options = {}) const;

    // Fused training objective primitive. The Vulkan implementation keeps the
    // image gradient in its render context for backward(); CUDA training keeps
    // using the existing fused loss path.
    float photometric_loss(
        const RenderResult& rendered,
        const tinytensor::Tensor& target,
        const tinytensor::Tensor& mask,
        bool mask_enabled, float ssim_weight,
        float photometric_weight, bool read_loss_value = true,
        tinytensor::Tensor* color_gradient = nullptr) const;

    ModelGradients backward(
        const GaussianModel& model, const RenderResult& rendered,
        const tinytensor::Tensor& grad_color,
        const tinytensor::Tensor& grad_alpha,
        const tinytensor::Tensor& grad_depth,
        const tinytensor::Tensor& grad_normal,
        const tinytensor::Tensor& densify_map = {},
        const SHAdamUpdate* sh_adam = nullptr,
        const StructureAdamUpdate* structure_adam = nullptr) const;

    // Materialize a deferred Vulkan contribution-visibility tensor. This is a
    // no-op when forward() already returned the regular Float32 visibility.
    void materialize_visibility(RenderResult& rendered) const;

    DepthSampleResult sample_depth(
        const GaussianModel& model, const tinytensor::Tensor& world_points,
        const Camera& camera,
        const RasterizeOptions& options = {}) const;

    DepthSampleGradients sample_depth_backward(
        const GaussianModel& model, const DepthSampleResult& sampled,
        const tinytensor::Tensor& grad_camera_points) const;

    OccupancyResult evaluate_occupancy(
        const GaussianModel& model, const tinytensor::Tensor& world_points,
        const Camera& camera,
        const RasterizeOptions& options = {}) const;

    // Packed [N,2] xy pixel centres from the last forward, or null.
    const float* projected_mean2d(const RenderResult& rendered) const;

private:
    mutable std::shared_ptr<void> backend_;
};

}  // namespace photara::splat
