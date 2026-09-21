#pragma once

#include "splat/options.hpp"
#include "splat/types.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace photara::splat::detail {

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
    // Dense/ADC+: weighted priority. IGS: accumulated log2 screen oversize.
    tinytensor::Tensor priority;
    tinytensor::Tensor geometry_gradient;
    // Distinct contributing camera support, saturated at two per window.
    tinytensor::Tensor first_view;
    tinytensor::Tensor view_support;
    // Sum of contribution-normalized image error, independent of gradients.
    tinytensor::Tensor image_error;
};

// Clear selected parent moments without temporary zero tensors or per-state scatters.
void zero_adam_rows(const tinytensor::Tensor& indices,
    const std::array<AdamState*, 6>& states);

struct AdcPlusPruneResult {
    tinytensor::Tensor keep_indices;
    tinytensor::Tensor opacities;
    std::size_t pruned{};
};

struct MultiViewLoss {
    float geometry{};
    float ncc{};
    std::size_t geometry_pixels{};
    std::size_t geometry_candidates{};
    std::size_t ncc_pixels{};
};

struct GeometryDistributionSummary {
    float opacity_mean{};
    float opacity_stddev{};
    float log_scale_mean{};
    float log_scale_stddev{};
    float log_anisotropy_mean{};
    float log_anisotropy_stddev{};
};

// Logging-only opacity stats. The three scalars are reduced on CUDA so a
// progress callback does not download every Gaussian.
struct OpacityProgressStats {
    float gradient_mean{};
    float positive_gradient_fraction{};
    float opacity_mean{};
};

ActivatedParameters activate_parameters(const GaussianModel& model);

// Exact per-axis finite quantiles, returned as min.xyz/max.xyz. Only six
// floats cross to the CPU; the full position array stays on CUDA.
std::array<float, 6> percentile_bounds(
    const tinytensor::Tensor& means, float percentile);

// Convert GaussianWrapping's [direction.xyz, orientation_logit] features to
// normalize(direction) * tanh(logit), and chain gradients back to features.
tinytensor::Tensor normal_features_to_normals(
    const tinytensor::Tensor& normal_features);
tinytensor::Tensor normal_features_backward(
    const tinytensor::Tensor& normal_features,
    const tinytensor::Tensor& grad_normals);

// Upload one packed RGBA8 word per pixel and expand it on CUDA. Packing the
// host cache cuts its footprint and per-step PCIe traffic by 3-4x compared
// with cached planar float RGB plus a separate float mask.
DecodedTrainingPixels upload_packed_training_pixels(
    const std::vector<int>& rgba,
    std::uint32_t width,
    std::uint32_t height,
    bool decode_mask,
    bool decode_gray);

// Expand a resident RGBA8 image without another host upload.
DecodedTrainingPixels decode_packed_training_pixels(
    const tinytensor::Tensor& packed,
    std::uint32_t width, std::uint32_t height,
    bool decode_mask, bool decode_gray);

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

// The same opacity/non-finite/bounds sweep as adc_plus_prune, returned as raw
// device masks so a caller can keep its own recycle/cap policy on the GPU
// instead of paying a host round trip. Nothing crosses back to the host.
struct PruneMasks {
    tinytensor::Tensor keep;       // Bool {N}: not hard and opacity >= floor
    tinytensor::Tensor hard;       // Bool {N}: non-finite / out of bounds
    tinytensor::Tensor opacities;  // Float32 {N}: sigmoid(opacity_logits)
};

PruneMasks prune_masks(
    const GaussianModel& model,
    float minimum_opacity,
    float maximum_bounds,
    const std::array<float, 3>& scene_center);

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
    bool collect_scalar_terms,
    tinytensor::Tensor* stability_accumulator = nullptr);

GeometryDistributionSummary summarize_geometry_distribution(
    const GaussianModel& model);

OpacityProgressStats summarize_opacity_progress(
    const tinytensor::Tensor& opacity_logits,
    const tinytensor::Tensor& opacity_gradients);

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
    bool depth_normal_active = false,
    // False when the render carried no depth/normal channels
    // (RasterizeOptions::require_depth off): the two full-image gradient
    // tensors are then neither allocated nor written.
    bool need_geometry_gradients = true);

LossGradients compute_normal_field_loss(
    const RenderResult& rendered, const Camera& camera, float weight,
    bool collect_scalar_terms = false);

void add_model_gradients(
    const ModelGradients& source, ModelGradients& destination,
    bool include_refine_weight = false);

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

// Update independent structure groups and project the scale ratio in one launch.
void adam_step_structure(GaussianModel& model, const ModelGradients& gradient,
    AdamState& means, AdamState& scales, AdamState& rotations, AdamState& opacity,
    float means_lr, unsigned step, const TrainingOptions& options,
    float minimum_log_scale, float maximum_log_scale);

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

void add_sh_regularization(const tinytensor::Tensor& sh,
    tinytensor::Tensor& gradient, float weight);

void add_geometry_regularization(const GaussianModel& model,
    ModelGradients& gradients, float opacity_weight, float log_scale_weight);

void accumulate_densification_stats(
    const tinytensor::Tensor& refine_weight,
    const tinytensor::Tensor& visibility,
    const tinytensor::Tensor& radii,
    DensificationStats& stats,
    std::uint32_t width,
    std::uint32_t height,
    bool use_maximum,
    bool require_contribution_visibility,
    float step_score_power = 1.F,
    float oversize_screen_threshold = 0.F,
    const tinytensor::Tensor& geometry_gradient = {},
    int view_index = -1,
    const tinytensor::Tensor& image_error = {});

// Per-pixel nonnegative (1 - SSIM contrast-structure)^power, shape [H, W].
tinytensor::Tensor compute_ssim_cs_error_map(
    const tinytensor::Tensor& prediction,
    const tinytensor::Tensor& target,
    const tinytensor::Tensor& mask,
    bool mask_enabled,
    float power = 1.F);

// Per-view Avg reduction: sum(error*alpha*T) / sum(alpha*T).
tinytensor::Tensor densify_avg_scores(
    const tinytensor::Tensor& numerator,
    const tinytensor::Tensor& denominator);

// Sample the error map at each Gaussian's projected mean (3x3 max).
tinytensor::Tensor scatter_error_map_to_gaussians(
    const tinytensor::Tensor& error_hw,
    const float* mean2d,
    const tinytensor::Tensor& radii,
    const tinytensor::Tensor& visibility);

tinytensor::Tensor densify_oversize_weights(
    const tinytensor::Tensor& scores,
    const tinytensor::Tensor& screens,
    float screen_threshold,
    float blend);

void clip_log_scale_by_screen(
    tinytensor::Tensor& log_scales,
    const tinytensor::Tensor& screens,
    float screen_threshold,
    float hardness);

// Split operators with a live caller, one per densification strategy.
enum class SplitMode {
    // ADC+ (Brush) split along the major variance axis only: k = 1/sqrt(2)
    // keeps the rank-one mixture covariance, and a parent whose projected
    // radius exceeds split_at_screen_size tightens k so the children land on
    // that size. Offsets stay on the major axis, so thin surfaces keep their
    // normal thickness, and the alpha maps (1-a) -> (1-a')^sqrt(2).
    adc_covariance,
    // Dense MVS tangent-plane split: local Z keeps the fused normal thickness.
    dense_tangent,
    // ADC-IGS split: one shared normal scalar xi pushes both children apart
    // along every local axis by its own scale, so the direction is random in
    // the local frame and the mixture variance grows instead of being
    // preserved. Only the largest axis halves, there is no screen-size trim,
    // and the alpha maps (1-a) -> (1-a')^2 so the two children composite back
    // to the parent exactly. ADC-IGS must keep this operator: routing its
    // growth through adc_covariance instead cost 1.33 dB and 1.50 dB on the
    // ori protocol at seeds 42 and 7 (SSIM -0.009 / -0.014). The covariance
    // operator shrinks and displaces along the major axis only, so its
    // children stay wide across the other two while its alpha mapping is the
    // milder (1-a)^(1/sqrt2) one; ADC-IGS then keeps half the surface width
    // repaired and the mesh turns soft (Marching Cubes 12.1M faces against
    // 7.1M for the same seed).
    igs_random,
    // EMC long-axis split, arXiv:2508.12313: the largest axis halves, the
    // other two shrink to 0.85, and both children move apart along the major
    // axis by half its scale. Opacity maps through the scheduled factor k:
    // a' = k / (1 + exp(-l) - k) so the two children composite back to
    // the parent.
    long_axis,
};

// Mutate selected parents and their already-cloned children in place.
void split_gaussians(
    GaussianModel& parents,
    GaussianModel& children,
    const tinytensor::Tensor& parent_indices,
    const tinytensor::Tensor& random_samples,
    const tinytensor::Tensor& screen_sizes,
    SplitMode mode,
    float minimum_opacity,
    float split_at_screen_size,
    float split_opacity_k = 0.5F);

// EMC relocation: overwrite each destination row with the +delta child of a
// long-axis split while the parent takes the -delta side in place. Only
// means/scales/opacity/quaternions are touched; the caller copies appearance
// rows and clears the destination optimizer moments.
void relocate_long_axis(
    GaussianModel& model,
    const tinytensor::Tensor& parent_indices,
    const tinytensor::Tensor& destination_indices,
    float split_opacity_k);

// EMC revised mean noise: per-step, opacity-gated exploration whose
// reach scales with each local sqrt(scale) axis. `scaler` follows the
// noise_lr -> noise_lr_final exponential schedule.
void inject_emc_noise(
    GaussianModel& model, const tinytensor::Tensor& radii,
    float scaler, unsigned seed);

// EMC per-splat regularizer gradients: linear pressure on the floored log
// scale, the effective-rank (entropy) penalty on the relative squared scale
// shares, and the quaternion norm pull toward unit length. Weights arrive
// already decay-scheduled and divided by the splat count.
void apply_shape_regularizers(
    GaussianModel& model, ModelGradients& gradients,
    float scale_weight, float erank_weight, float erank_s3_weight,
    float quat_weight);

// EMC soft on-screen limit: over the limit, each step spends
// penalty * log2(screen / limit) shares of one scale learning-rate step,
// distributed across the local axes by exp(2 * (log_scale - log_scale_max))
// so a flat splat stays flat while it shrinks.
void apply_oversize_penalty(
    GaussianModel& model, const tinytensor::Tensor& radii,
    float inverse_resolution, float screen_limit, float penalty,
    float scales_lr);

void apply_adc_decay(
    GaussianModel& model, float opacity_decay, float scale_decay);

tinytensor::Tensor adc_plus_footprint_weights(
    const tinytensor::Tensor& gradients,
    const tinytensor::Tensor& screens);

void inject_adc_noise(
    GaussianModel& model,
    const tinytensor::Tensor& visibility,
    float standard_deviation,
    float maximum_noise,
    unsigned seed,
    const tinytensor::Tensor& radii = {},
    bool revised = false);

void reset_opacity(GaussianModel& model, float maximum_opacity);

}  // namespace photara::splat::detail
