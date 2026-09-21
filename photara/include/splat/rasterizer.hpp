#pragma once

#include "splat/types.hpp"

namespace photara::splat {

struct RasterizeOptions {
    unsigned active_sh_degree{0};
    std::array<float, 3> background{0.F, 0.F, 0.F};
    float kernel_size{0.F};
    float scale_modifier{1.F};
    bool require_depth{true};
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
    tinytensor::Tensor first, second;
    float lr{}, rest_lr{}, beta1{.9F}, beta2{.999F};
    float correction1{1.F}, correction2{1.F}, epsilon{1e-8F};
    float regularization_weight{};
};

class Rasterizer {
public:
    RenderResult forward(
        const GaussianModel& model, const Camera& camera,
        const RasterizeOptions& options = {}) const;

    ModelGradients backward(
        const GaussianModel& model, const RenderResult& rendered,
        const tinytensor::Tensor& grad_color,
        const tinytensor::Tensor& grad_alpha,
        const tinytensor::Tensor& grad_depth,
        const tinytensor::Tensor& grad_normal,
        const tinytensor::Tensor& densify_map = {},
        const SHAdamUpdate* sh_adam = nullptr) const;

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
};

}  // namespace photara::splat
