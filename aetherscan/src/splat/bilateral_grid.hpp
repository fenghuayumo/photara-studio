#pragma once

#include "cuda_ops.hpp"
#include "splat/options.hpp"

#include "internal/tensor_impl.hpp"

#include <cstddef>

namespace aetherscan::splat::detail {

// Per-view affine bilateral grid. Each cell stores a 3x4 matrix that maps
// rendered RGB onto the training image, interpolated in (x, y, luma). The
// trained Gaussians keep canonical colour: the grid is never applied to
// evaluation, preview, or exported renders.
struct BilateralGridState {
    tinytensor::Tensor grids;     // [views, L, H, W, 12]
    tinytensor::Tensor gradient;  // [views, L, H, W, 12]
    AdamState adam;
    tinytensor::Tensor output;      // [3, H, W]
    tinytensor::Tensor input_grad;  // [3, H, W]
    int luma{8};
    int grid_height{16};
    int grid_width{16};

    [[nodiscard]] bool is_valid() const {
        return grids.is_valid() && grids.numel() != 0;
    }
};

BilateralGridState make_bilateral_grid_state(
    std::size_t views, const TrainingOptions& options);

void apply_bilateral_grid(
    const tinytensor::Tensor& color, BilateralGridState& state,
    std::size_t view);

void backward_bilateral_grid(
    BilateralGridState& state, const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient, std::size_t view);

void step_bilateral_grid(
    BilateralGridState& state, const TrainingOptions& options,
    unsigned iteration);

}  // namespace aetherscan::splat::detail
