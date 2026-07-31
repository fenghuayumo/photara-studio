#pragma once

#include "splat/options.hpp"
#include "splat/types.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace aetherscan::splat::detail {

struct ActivatedParameters {
    tinytensor::Tensor scales;
    tinytensor::Tensor quaternions;
    tinytensor::Tensor opacities;
};

struct LossGradients {
    tinytensor::Tensor color;
    tinytensor::Tensor alpha;
    tinytensor::Tensor depth;
    tinytensor::Tensor normal;
    float total{};
    float rgb{};
    float alpha_value{};
    float depth_value{};
    float normal_value{};
};

struct DecodedTrainingPixels {
    tinytensor::Tensor rgb;
    tinytensor::Tensor gray;
    tinytensor::Tensor mask;
};

struct AdamState {
    tinytensor::Tensor first;
    tinytensor::Tensor second;
};

struct DensificationStats {
    tinytensor::Tensor gradient;
    tinytensor::Tensor count;
    tinytensor::Tensor max_screen_radius;
    tinytensor::Tensor priority;
};

struct AdcPlusPruneResult {
    tinytensor::Tensor keep_indices;
    tinytensor::Tensor opacities;
    std::size_t pruned{};
};

struct MultiViewLoss {
    float geometry{};
    float ncc{};
    std::size_t geometry_pixels{};
    std::size_t ncc_pixels{};
};

ActivatedParameters activate_parameters(const GaussianModel& model);

// Upload one packed RGBA8 word per pixel and expand it on CUDA. Packing the
// host cache cuts its footprint and per-step PCIe traffic by 3-4x compared
// with cached planar float RGB plus a separate float mask.
DecodedTrainingPixels upload_packed_training_pixels(
    const std::vector<int>& rgba,
    std::uint32_t width,
    std::uint32_t height,
    bool decode_mask,
    bool decode_gray);

// Fold the Mip-Splatting 3D-filter floor into the canonical scale/opacity
// parameters and clear model.filter_3d. This is available for explicit model
// conversion; ADC+ training keeps the filter separate from canonical values.
void bake_3d_filter(GaussianModel& model);

// Build ADC+ opacity/non-finite/bounds pruning and compacted keep indices on
// CUDA. Only scalar counts cross back to the host.
AdcPlusPruneResult adc_plus_prune(
    const GaussianModel& model,
    float minimum_opacity,
    float maximum_bounds,
    const std::array<float, 3>& scene_center,
    std::size_t maximum_count);

tinytensor::Tensor compute_3d_filter(
    const tinytensor::Tensor& means,
    const std::vector<Camera>& cameras,
    float minimum_scale_factor = 0.2F,
    bool all_camera_euclidean = false);

MultiViewLoss add_multi_view_loss(
    const tinytensor::Tensor& sampled_neighbour_points,
    const tinytensor::Tensor& sampled_inside,
    const RenderResult& reference_render,
    const TrainingView& reference,
    const TrainingView& neighbour,
    const TrainingOptions& options,
    LossGradients& gradients,
    tinytensor::Tensor& grad_sampled_points,
    bool collect_scalar_terms);

tinytensor::Tensor unproject_depth_to_world(
    const tinytensor::Tensor& depth, const Camera& camera);

void add_sample_depth_point_gradients(
    const Camera& reference_camera,
    const tinytensor::Tensor& grad_world_points,
    LossGradients& image_gradients);

void add_sample_depth_model_gradients(
    const DepthSampleGradients& sample_gradients,
    ModelGradients& model_gradients);

void chain_parameter_gradients(
    const GaussianModel& model, const ActivatedParameters& activated,
    const tinytensor::Tensor& grad_scales,
    const tinytensor::Tensor& grad_quaternions,
    const tinytensor::Tensor& grad_opacities,
    ModelGradients& gradients);

LossGradients compute_training_loss(
    const RenderResult& rendered, const TrainingView& target,
    const TrainingOptions& options, bool collect_scalar_terms,
    bool depth_normal_active = false);

AdamState make_adam_state(const tinytensor::Tensor& parameter);

// Keep a full first moment but only one second-moment scalar per row. This is
// the storage layout used by Brush's AdamScaled for SH coefficients.
AdamState make_reduced_second_adam_state(
    const tinytensor::Tensor& parameter);

void adam_step(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, float learning_rate, unsigned step,
    const TrainingOptions& options, std::size_t group_stride = 0,
    float secondary_learning_rate = 0.F,
    float clamp_min = -std::numeric_limits<float>::infinity(),
    float clamp_max = std::numeric_limits<float>::infinity());

// brush AdamScaled shares the second moment across every trailing SH
// coefficient of a Gaussian while retaining a per-coefficient first moment.
void adam_step_reduced_second(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, float learning_rate, unsigned step,
    const TrainingOptions& options, std::size_t row_stride,
    float secondary_learning_rate);

// Update only the leading active_row_stride values of each logical row.
// This avoids reading and writing inactive higher-order SH coefficients.
void adam_step_active_prefix(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    AdamState& state, float learning_rate, unsigned step,
    const TrainingOptions& options, std::size_t full_row_stride,
    std::size_t active_row_stride, float secondary_learning_rate);

void constrain_scale_ratio(
    tinytensor::Tensor& log_scales, float maximum_ratio);

DensificationStats make_densification_stats(std::size_t count);

void accumulate_densification_stats(
    const tinytensor::Tensor& refine_weight,
    const tinytensor::Tensor& visibility,
    const tinytensor::Tensor& radii,
    DensificationStats& stats,
    std::uint32_t width,
    std::uint32_t height,
    bool use_maximum,
    bool require_contribution_visibility);

// Mutate selected parents and their already-cloned children in place.
// mode: 1=default split, 2=ADC+ split, 3=ADC-IGS split,
//       4=dense-MVS tangent-plane split.
void split_gaussians(
    GaussianModel& parents,
    GaussianModel& children,
    const tinytensor::Tensor& parent_indices,
    const tinytensor::Tensor& random_samples,
    const tinytensor::Tensor& screen_sizes,
    int mode,
    float minimum_opacity,
    float split_at_screen_size);

void apply_adc_decay(
    GaussianModel& model, float opacity_decay, float scale_decay);

void inject_adc_noise(
    GaussianModel& model,
    const tinytensor::Tensor& visibility,
    float standard_deviation,
    float maximum_noise,
    unsigned seed);

void reset_opacity(GaussianModel& model, float maximum_opacity);

}  // namespace aetherscan::splat::detail
