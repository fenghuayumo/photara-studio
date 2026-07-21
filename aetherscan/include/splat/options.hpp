#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace aetherscan::splat {

enum class AlphaMode {
    masked,
    transparent,
};

enum class DensificationStrategy {
    default_strategy,
    adc_plus,
    adc_igs,
};

struct TrainingOptions {
    unsigned iterations{30'000};
    unsigned sh_degree{3};
    unsigned sh_degree_interval{1'000};
    unsigned seed{42};
    unsigned log_interval{100};
    std::size_t max_gaussians{500'000};
    // Sparse COLMAP initialization enables dynamic Gaussian management.
    // Dense MVS initialization always disables it to preserve the fused cloud.
    bool input_is_dense{true};
    DensificationStrategy densification_strategy{
        DensificationStrategy::default_strategy};
    std::size_t densification_cap{4'000'000};
    unsigned refine_start_iter{0};  // 0 selects the strategy preset
    unsigned refine_stop_iter{0};   // 0 selects 15k (default/ADC+) or 25k (ADC-IGS)
    unsigned grow_stop_iter{15'000};
    unsigned refine_every{0};       // 0 selects 100 (default) or 200 (ADC)
    unsigned opacity_reset_every{3'000};
    float densify_gradient_threshold{0.003F};
    float densify_select_fraction{0.4F};
    float densify_scale_threshold{0.01F};
    float densify_screen_threshold{0.25F};
    float prune_opacity{1.F / 255.F};
    float opacity_decay{0.004F};
    float scale_decay{0.002F};
    float mean_noise_weight{50.F};
    float initial_opacity{0.1F};
    float initial_scale{1.F};
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
    // Match pygsplat's mask-training controls. In masked mode RGB is supervised
    // only in the foreground and background alpha leakage is penalized. In
    // transparent mode the same foreground RGB loss is combined with full-image
    // BCE(predicted alpha, mask) * match_alpha_weight.
    bool use_mask{false};
    AlphaMode alpha_mode{AlphaMode::transparent};
    float match_alpha_weight{0.25F};
    std::filesystem::path mask_dir;
    float minimum_scale_fraction{1e-4F};
    float maximum_scale_fraction{0.02F};
    float max_scale_ratio{10.F};
    float geometry_epsilon{1e-3F};
    float kernel_size{0.3F};
    float scale_modifier{1.F};
    bool use_mvs_depth{true};
    bool use_mvs_normals{true};
};

}  // namespace aetherscan::splat
