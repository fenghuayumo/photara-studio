#pragma once

#include <cstdint>
#include <filesystem>

namespace photara::texture {

enum class BlendMode : std::uint8_t {
    best_view = 0,
    weighted_average = 1,
    // Pixel-footprint softmax: every sample contributes
    // exp(softmax_scale * |n.v| / d^2), so the view that resolves a texel best
    // dominates and grazing or distant samples cannot blur or alias it.
    softmax = 2,
};

enum class VisibilityMode : std::uint8_t {
    shadow_map = 0,
    hybrid_ray_query = 1,
};

// sRGB matches photographs and the Gaussian renderer. Linear blends in
// scene-referred light and encodes sRGB only when the atlas PNG is written.
enum class BlendColorSpace : std::uint8_t {
    srgb = 0,
    linear = 1,
};

struct TextureOptions {
    // Atlas resolution (square).
    std::uint32_t atlas_resolution{2048};
    // Guard space reserved around every chart in texels. The bake extends valid
    // texels into it, so filtered sampling never blends in undefined black.
    float uv_gutter{4.F};
    float uv_max_stretch{1.F / 3.F};
    // UVAtlas charting is otherwise largely serial on large meshes. The
    // photara_drender wrapper partitions faces spatially, unwraps groups concurrently,
    // then packs the combined charts into one atlas.
    std::uint32_t uv_parallel_partitions{8};

    BlendMode blend_mode{BlendMode::softmax};
    // Exponential sharpness of BlendMode::softmax (<= 0 selects a
    // scene-relative scale).
    float softmax_scale{0.F};
    // Foreground masks hide the volume behind the subject. Keeping masked
    // pixels as weak samples (instead of hard vetoes) gives geometry that is
    // visible in the photographs but outside every mask - the surface a scanned
    // object rests on, for example - its real texture instead of black. 0 keeps
    // strict mask semantics, 1 ignores masks entirely during the bake.
    float texture_mask_floor{0.05F};
    BlendColorSpace color_space{BlendColorSpace::srgb};
    VisibilityMode visibility_mode{VisibilityMode::hybrid_ray_query};
    std::uint32_t pcf_radius{1};
    bool allow_visibility_fallback{true};

    // Atlas guard/dilation pass. Guard texels and rasterized chart texels that
    // received no projection sample (masked out, rejected by the facing test,
    // or occluded in every view) copy the nearest valid texel instead of
    // staying black.
    bool atlas_padding{true};
    std::uint32_t atlas_padding_margin{4};
    // Off by default: samples that survive the mask floor and the facing test
    // already cover everything the photographs show. Only enable this when a
    // smooth colour extrapolation is preferred over leaving unseen texels black,
    // and expect flat per-chart fills rather than measured detail.
    bool atlas_fill_unobserved{false};
    // Optional cap for the in-chart fill distance in texels (0 = unlimited).
    std::uint32_t atlas_fill_maximum_distance{0};

    // The projective bake is only an initialization. By default, refine the
    // atlas directly against all calibrated views with photara_drender's native
    // Vulkan/Adam optimizer, followed by seam-only polish.
    bool optimize{true};
    std::uint32_t optimize_steps{1000};
    std::uint32_t optimize_batch_size{4};
    float optimize_learning_rate{5e-3F};
    float optimize_min_learning_rate{2.5e-4F};
    std::uint32_t seam_samples_per_edge{4};
    std::uint32_t seam_polish_steps{30};
    float seam_learning_rate{1e-3F};

    // Optional foreground masks (same basename as source images).
    std::filesystem::path mask_dir;

    // Run image-domain Intrinsic delighter before projection (ONNX Runtime).
    bool delight{false};
    // Directory containing stage_0.onnx .. stage_3.onnx. Empty = CMake default.
    std::filesystem::path delight_model_dir;
    bool delight_use_cuda{true};

    std::uint32_t vulkan_device_index{0};
};

}  // namespace photara::texture
