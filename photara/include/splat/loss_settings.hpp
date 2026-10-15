#pragma once

#include "splat/options.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace photara::splat {

enum class LossGroup { image, gaussian, geometry, colour };

struct TrainingLossSettings {
    float photometric_weight{1.F};
    float ssim_weight{0.2F};
    float match_alpha_weight{0.25F};
    float opacity_regularization_weight{0.F};
    float scale_regularization_weight{0.F};
    float shape_erank_reg{0.F};
    float shape_axis_ratio_reg{0.F};
    float shape_max_axis_ratio{100.F};
    float sh_regularization_weight{0.001F};
    float shape_erank_s3_reg{0.F};
    float shape_quat_norm_reg{0.F};
    float oversize_screen_limit{0.F};
    float oversize_penalty{0.F};
    float depth_normal_weight{0.05F};
    float multi_view_geo_weight{0.02F};
    float multi_view_ncc_weight{0.6F};
    float normal_field_weight{0.05F};
    float normal_field_depth_ratio{0.6F};
    float bilateral_grid_tv_weight{10.F};
    float bilateral_grid_deviation_limit{0.5F};
    float ppisp_reg_exposure_mean{1.F};
    float ppisp_reg_color_mean{1.F};
    float ppisp_reg_vig_center{0.02F};
    float ppisp_reg_vig_non_pos{0.01F};
    float ppisp_reg_vig_channel_var{0.1F};
    float ppisp_reg_crf_channel_var{0.1F};
    float ppisp_exposure_limit{1.F};
    float ppisp_color_limit{2.F};
    bool geometry_training{false};
    int normal_field_from_iter{20'001};
    int ppisp_model{0};
    void sanitize();
    void validate() const;
};

struct TrainingLossField {
    float TrainingLossSettings::* value;
    float TrainingOptions::* training_value;
    float default_value, minimum, maximum;
    LossGroup group;
    bool advanced;
    const char* cli;
    const char* label;
    const char* help;
};

inline constexpr auto k_training_loss_fields = std::to_array<TrainingLossField>({
    {&TrainingLossSettings::photometric_weight, &TrainingOptions::photometric_weight, 1.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::image, false, "splat-photometric-weight", "Image weight",
     "Overall strength of image matching."},
    {&TrainingLossSettings::ssim_weight, &TrainingOptions::ssim_weight, 0.2F, 0.F, 1.F,
     LossGroup::image, false, "splat-ssim-weight", "Structure balance",
     "Blends pixel accuracy with structural similarity. The pixel share is 1 minus this value."},
    {&TrainingLossSettings::match_alpha_weight, &TrainingOptions::match_alpha_weight, 0.25F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::image, false, "splat-match-alpha-weight", "Mask opacity",
     "Matches foreground opacity in Transparent mask mode."},
    {&TrainingLossSettings::opacity_regularization_weight, &TrainingOptions::opacity_regularization_weight, 0.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::gaussian, false, "splat-opacity-reg", "Opacity sparsity",
     "Reduces unnecessary opaque Gaussians. Zero disables this constraint."},
    {&TrainingLossSettings::scale_regularization_weight, &TrainingOptions::scale_regularization_weight, 0.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::gaussian, false, "splat-scale-reg", "Scale compactness",
     "Shrinks large Gaussians more strongly. Strength decreases during training."},
    {&TrainingLossSettings::shape_erank_reg, &TrainingOptions::shape_erank_reg, 0.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::gaussian, false, "splat-shape-erank-reg", "Needle suppression",
     "Discourages needle-like Gaussians while allowing thin surfaces."},
    {&TrainingLossSettings::shape_axis_ratio_reg, &TrainingOptions::shape_axis_ratio_reg, 0.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::gaussian, false, "splat-shape-axis-ratio-reg", "Axis-ratio strength",
     "Penalizes elongated Gaussians only above the axis-ratio limit."},
    {&TrainingLossSettings::shape_max_axis_ratio, &TrainingOptions::shape_max_axis_ratio, 100.F, 1.F, std::numeric_limits<float>::max(),
     LossGroup::gaussian, false, "splat-shape-max-axis-ratio", "Axis-ratio limit",
     "Maximum unpenalized ratio of the longest to shortest axis."},
    {&TrainingLossSettings::sh_regularization_weight, &TrainingOptions::sh_regularization_weight, 0.001F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::gaussian, false, "splat-sh-reg", "View-dependent colour",
     "Limits excessive view-dependent colour variation."},
    {&TrainingLossSettings::shape_erank_s3_reg, &TrainingOptions::shape_erank_s3_reg, 0.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::gaussian, true, "splat-shape-flatness-reg", "Surface flatness",
     "Penalizes the smallest eigenvalue share to encourage flat Gaussians."},
    {&TrainingLossSettings::shape_quat_norm_reg, &TrainingOptions::shape_quat_norm_reg, 0.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::gaussian, true, "splat-shape-rotation-reg", "Rotation normalization",
     "Keeps rotation quaternions close to unit length."},
    {&TrainingLossSettings::oversize_screen_limit, &TrainingOptions::oversize_screen_limit, 0.F, 0.F, 1.F,
     LossGroup::gaussian, true, "splat-screen-size-limit", "Screen size limit",
     "Fraction of the shorter image dimension. Requires a positive screen-size strength."},
    {&TrainingLossSettings::oversize_penalty, &TrainingOptions::oversize_penalty, 0.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::gaussian, true, "splat-screen-size-penalty", "Screen size strength",
     "Shrinks oversized projected Gaussians. Requires a positive screen size limit."},
    {&TrainingLossSettings::depth_normal_weight, &TrainingOptions::depth_normal_weight, 0.05F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::geometry, false, "splat-depth-normal-weight", "Surface consistency",
     "Aligns rendered normals with normals from median depth."},
    {&TrainingLossSettings::multi_view_geo_weight, &TrainingOptions::multi_view_geo_weight, 0.02F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::geometry, false, "splat-mv-geo-weight", "Multi-view depth",
     "Encourages consistent depths between neighbouring views."},
    {&TrainingLossSettings::multi_view_ncc_weight, &TrainingOptions::multi_view_ncc_weight, 0.6F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::geometry, false, "splat-mv-ncc-weight", "Multi-view appearance",
     "Matches image patches across neighbouring views."},
    {&TrainingLossSettings::normal_field_weight, &TrainingOptions::normal_field_weight, 0.05F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::geometry, true, "splat-normal-field-weight", "Normal-field alignment",
     "Aligns the learned normal field with median-depth normals."},
    {&TrainingLossSettings::normal_field_depth_ratio, &TrainingOptions::normal_field_depth_ratio, 0.6F, 0.F, 1.F,
     LossGroup::geometry, true, "splat-normal-field-depth-ratio", "Depth gradient share",
     "Controls how normal-field alignment updates the depth geometry."},
    {&TrainingLossSettings::bilateral_grid_tv_weight, &TrainingOptions::bilateral_grid_tv_weight, 10.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::colour, false, "splat-bilateral-grid-tv", "Spatial smoothness",
     "Smooths neighbouring colour-grid cells to reduce local overfitting."},
    {&TrainingLossSettings::bilateral_grid_deviation_limit, &TrainingOptions::bilateral_grid_deviation_limit, 0.5F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::colour, true, "splat-bilateral-grid-deviation-limit", "Grid deviation limit",
     "Bounds colour-grid coefficients around identity. Zero removes the bound."},
    {&TrainingLossSettings::ppisp_reg_exposure_mean, &TrainingOptions::ppisp_reg_exposure_mean, 1.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::colour, true, "splat-ppisp-reg-exposure-mean", "Exposure balance",
     "Constrains the mean exposure correction across views."},
    {&TrainingLossSettings::ppisp_reg_color_mean, &TrainingOptions::ppisp_reg_color_mean, 1.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::colour, true, "splat-ppisp-reg-color-mean", "White-balance neutrality",
     "Constrains the mean colour transform across views."},
    {&TrainingLossSettings::ppisp_reg_vig_center, &TrainingOptions::ppisp_reg_vig_center, 0.02F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::colour, true, "splat-ppisp-reg-vig-center", "Vignette centre",
     "Constrains the vignetting centre."},
    {&TrainingLossSettings::ppisp_reg_vig_non_pos, &TrainingOptions::ppisp_reg_vig_non_pos, 0.01F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::colour, true, "splat-ppisp-reg-vig-non-pos", "Vignette attenuation",
     "Discourages vignetting from brightening the image."},
    {&TrainingLossSettings::ppisp_reg_vig_channel_var, &TrainingOptions::ppisp_reg_vig_channel_var, 0.1F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::colour, true, "splat-ppisp-reg-vig-channel-var", "Vignette colour consistency",
     "Limits differences in vignetting across colour channels."},
    {&TrainingLossSettings::ppisp_reg_crf_channel_var, &TrainingOptions::ppisp_reg_crf_channel_var, 0.1F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::colour, true, "splat-ppisp-reg-crf-channel-var", "Response colour consistency",
     "Limits differences in camera response across colour channels."},
    {&TrainingLossSettings::ppisp_exposure_limit, &TrainingOptions::ppisp_exposure_limit, 1.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::colour, true, "splat-ppisp-exposure-limit", "Exposure range",
     "Bounds per-view exposure correction in stops. Zero removes the bound."},
    {&TrainingLossSettings::ppisp_color_limit, &TrainingOptions::ppisp_color_limit, 2.F, 0.F, std::numeric_limits<float>::max(),
     LossGroup::colour, true, "splat-ppisp-color-limit", "White-balance range",
     "Bounds per-view white-balance parameters. Zero removes the bound."},
});

inline constexpr std::size_t k_training_loss_payload_bytes =
    k_training_loss_fields.size() * sizeof(float) + 2 * sizeof(std::int32_t) + sizeof(std::uint8_t);

inline void TrainingLossSettings::sanitize() {
    for (const auto& field : k_training_loss_fields) {
        float& current = this->*field.value;
        current = std::isfinite(current)
            ? std::clamp(current, field.minimum, field.maximum) : field.default_value;
    }
    normal_field_from_iter = std::max(normal_field_from_iter, 0);
    ppisp_model = std::clamp(ppisp_model, 0, 2);
}

inline void TrainingLossSettings::validate() const {
    for (const auto& field : k_training_loss_fields) {
        const float current = this->*field.value;
        if (!std::isfinite(current) || current < field.minimum || current > field.maximum)
            throw std::invalid_argument(std::string("Invalid training parameter: --") + field.cli);
    }
    if (normal_field_from_iter < 0 || ppisp_model < 0 || ppisp_model > 2)
        throw std::invalid_argument("Invalid training objective mode or start iteration");
}

}  // namespace photara::splat
