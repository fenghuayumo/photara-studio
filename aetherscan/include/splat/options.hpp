#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace aetherscan::splat {

enum class AlphaMode {
    masked,
    transparent,
};

struct TrainingOptions {
    unsigned iterations{30'000};
    unsigned sh_degree{3};
    unsigned sh_degree_interval{1'000};
    unsigned seed{42};
    unsigned log_interval{100};
    std::size_t max_gaussians{500'000};
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
    float adam_epsilon{1e-8F};
    float photometric_weight{1.F};
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
    float charbonnier_epsilon{1e-3F};
    float kernel_size{0.3F};
    float scale_modifier{1.F};
    bool use_mvs_depth{true};
    bool use_mvs_normals{true};
};

}  // namespace aetherscan::splat
