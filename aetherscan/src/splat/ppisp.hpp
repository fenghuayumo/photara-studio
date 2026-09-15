#pragma once

#include "cuda_ops.hpp"
#include "splat/options.hpp"
#include "splat/types.hpp"

#include "internal/tensor_impl.hpp"

#include <array>
#include <cstddef>

namespace aetherscan::splat::detail {

// Per-view PPISP (per-pixel image signal processing) table. Each camera owns
// an exposure, an optional vignetting model, a colour-homography white-balance,
// and an optional CRF. The transform is training-only: evaluation, preview and
// exported renders keep the canonical Gaussian colour.
struct PpispState {
    tinytensor::Tensor parameters;  // [views, P]
    tinytensor::Tensor gradient;    // [views, P]
    AdamState adam;
    tinytensor::Tensor output;      // [3, H, W]
    tinytensor::Tensor input_grad;  // [3, H, W]
    tinytensor::Tensor raw_sums;    // regularization scratch
    PpispParamType type{PpispParamType::no_crf_no_vig};
    int num_params{9};
    bool clamp_output{false};

    [[nodiscard]] bool is_valid() const {
        return parameters.is_valid() && parameters.numel() != 0;
    }
};

PpispState make_ppisp_state(
    std::size_t views, const TrainingOptions& options);

void apply_ppisp(
    const tinytensor::Tensor& color, PpispState& state, const Camera& camera,
    std::size_t view);

void backward_ppisp(
    PpispState& state, const tinytensor::Tensor& color,
    const tinytensor::Tensor& output_gradient, const Camera& camera,
    std::size_t view);

void step_ppisp(
    PpispState& state, const TrainingOptions& options, unsigned iteration);

// Diagnostics for the training log: mean and maximum |2^exposure - 1|
// over the per-view PPISP table.
[[nodiscard]] std::array<float, 2> ppisp_identity_deviation(
    const PpispState& state);

}  // namespace aetherscan::splat::detail
