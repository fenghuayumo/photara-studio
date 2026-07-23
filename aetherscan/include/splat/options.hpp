#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace aetherscan::splat {

enum class AlphaMode {
    masked,
    transparent,
};

enum class DensificationStrategy {
    default_strategy,
    adc_plus,
    adc_igs,
    dense_adaptive,
};

struct TrainingOptions {
    unsigned iterations{10'000};
    unsigned sh_degree{3};
    unsigned sh_degree_interval{1'000};
    unsigned seed{42};
    unsigned log_interval{100};
    std::size_t max_gaussians{500'000};
    // Sparse COLMAP initialization enables dynamic Gaussian management. Dense
    // MVS initialization only enables it for the explicit dense_adaptive mode.
    bool input_is_dense{true};
    bool enable_densification{true};
    DensificationStrategy densification_strategy{
        DensificationStrategy::default_strategy};
    std::size_t densification_cap{10'000'000};
    unsigned refine_start_iter{0};  // 0 selects the strategy preset
    unsigned refine_stop_iter{0};   // 0 selects the strategy preset
    unsigned grow_stop_iter{15'000};
    unsigned refine_every{0};       // 0 selects the strategy preset
    unsigned opacity_reset_every{3'000};
    // pygsplat DefaultStrategy threshold. With the backward accumulation
    // buffer correctly cleared, the native refine statistic has parity scale.
    float densify_gradient_threshold{0.003F};
    float densify_select_fraction{0.4F};
    float densify_scale_threshold{0.01F};
    float densify_screen_threshold{0.25F};
    // brush-train ADC+ defaults. These are separate from the classic 3DGS
    // controls above so selecting ADC+ does not silently change the default
    // strategy's pygsplat-compatible thresholds.
    float adc_plus_growth_gradient_threshold{0.0025F};
    float adc_plus_growth_select_fraction{0.25F};
    float adc_plus_split_at_screen_size{0.5F};
    // Dense MVS points already cover the surface. Recycle only a small part of
    // the budget per refinement and grow more conservatively than sparse ADC.
    float dense_recycle_fraction{0.01F};
    float dense_growth_fraction{0.005F};
    float prune_opacity{1.F / 255.F};
    float opacity_decay{0.004F};
    // ADC-IGS scale decay. brush ADC+ only decays opacity.
    float scale_decay{0.002F};
    float mean_noise_weight{50.F};
    float initial_opacity{0.1F};
    float initial_scale{1.F};
    // Match pygsplat: initialize isotropic scales from the RMS distance to the
    // three nearest neighbours in the selected input cloud.
    bool initialize_scale_from_knn{false};
    // Freeze dense MVS structure after its warm-up and keep optimizing the
    // progressively enabled SH bands. dense_adaptive may still recycle/split
    // rows after this point, without applying further structure Adam steps.
    unsigned dense_structure_freeze_iter{1'000};  // 0 disables the freeze
    // Optional all-input diagnostic: stop means/scale/quaternion/opacity Adam
    // after this iteration while continuing SH optimization.
    unsigned structure_freeze_iter{0};  // 0 disables the freeze
    float means_lr{1.6e-4F};
    float scales_lr{5e-3F};
    float opacities_lr{5e-2F};
    float quaternions_lr{1e-3F};
    float sh0_lr{2.5e-3F};
    float sh_rest_lr{1.25e-4F};
    float beta1{0.9F};
    float beta2{0.999F};
    // Match pygsplat/FusedAdam. The raster gradients are averaged over every
    // image pixel, so 1e-8 suppresses useful geometry updates.
    float adam_epsilon{1e-15F};
    float photometric_weight{1.F};
    float ssim_weight{0.2F};
    float depth_weight{0.05F};
    float normal_weight{0.01F};
    // GGGS meshing loss from the Python reference: cosine consistency between
    // the rasterized Gaussian normal and the normal differentiated from the
    // rendered median-depth map.
    bool use_depth_normal_loss{false};
    float depth_normal_weight{0.05F};
    unsigned depth_normal_from_iter{7'000};
    // Reference GGGS/Mip-Splatting low-pass filter. It expands sub-pixel
    // Gaussians and compensates opacity so the integrated density is stable.
    bool use_3d_filter{true};
    unsigned filter_3d_update_interval{100};
    // GGGS multi-view PatchMatch supervision. Geometry uses a differentiable
    // depth round trip through a nearby camera; NCC uses the reference
    // plane-induced homography and the original image pair.
    float multi_view_geo_weight{0.F};
    float multi_view_ncc_weight{0.F};
    unsigned multi_view_num{8};
    float multi_view_max_angle{30.F};
    float multi_view_min_distance{0.01F};
    float multi_view_max_distance{1.5F};
    float multi_view_pixel_noise_threshold{1.F};
    bool multi_view_robust_ncc{true};
    float multi_view_ncc_lambda_reference{0.45F};
    float multi_view_ncc_sharpness{12.F};
    float multi_view_ncc_min_weight{0.F};
    // Match pygsplat's mask-training controls. In masked mode RGB is supervised
    // only in the foreground and background alpha leakage is penalized. In
    // transparent mode the same foreground RGB loss is combined with full-image
    // BCE(predicted alpha, mask) * match_alpha_weight.
    bool use_mask{false};
    AlphaMode alpha_mode{AlphaMode::transparent};
    float match_alpha_weight{0.25F};
    std::filesystem::path mask_dir;
    float minimum_scale_fraction{1e-4F};
    float maximum_scale_fraction{0.002F};
    // Disabled for pygsplat parity. Its default strategy does not hard-clamp
    // anisotropy; it only prunes Gaussians whose largest axis exceeds 10% of
    // the scene scale during refinement.
    float max_scale_ratio{0.F};
    bool constrain_scale_range{true};
    float geometry_epsilon{1e-3F};
    // Match pygsplat's default classic rasterization. A positive kernel enables
    // antialiasing and must be selected explicitly.
    float kernel_size{0.F};
    float scale_modifier{1.F};
    // Photometric parity baseline. Geometry supervision is opt-in and can be
    // reintroduced after fixed-model L1+SSIM convergence is verified.
    bool use_mvs_depth{false};
    bool use_mvs_normals{false};
    // Train against source-resolution undistorted images rather than the MVS
    // working resolution.
    bool use_source_resolution{false};
    // brush-style dataset controls. Images are resized so their largest
    // dimension does not exceed this value (0 keeps the source resolution),
    // and decoded float training views are retained in a bounded host LRU.
    // Only the current reference/neighbour views are uploaded to CUDA.
    unsigned max_image_dimension{1'920};
    std::size_t training_view_cache_bytes{
        std::size_t{6} * 1024 * 1024 * 1024};
    // Iterations at which the caller may render fixed-view parity snapshots.
    std::vector<unsigned> evaluation_iterations;
};

}  // namespace aetherscan::splat
