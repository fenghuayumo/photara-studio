#pragma once

#include "splat/types.hpp"

namespace aetherscan::splat {

struct RasterizeOptions {
    unsigned active_sh_degree{0};
    std::array<float, 3> background{0.F, 0.F, 0.F};
    float kernel_size{0.F};
    float scale_modifier{1.F};
    bool require_depth{true};
    bool debug{false};
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
        const tinytensor::Tensor& grad_normal) const;

    DepthSampleResult sample_depth(
        const GaussianModel& model, const tinytensor::Tensor& world_points,
        const Camera& camera,
        const RasterizeOptions& options = {}) const;

    DepthSampleGradients sample_depth_backward(
        const GaussianModel& model, const DepthSampleResult& sampled,
        const tinytensor::Tensor& grad_camera_points) const;
};

}  // namespace aetherscan::splat
