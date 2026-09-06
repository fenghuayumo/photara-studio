#pragma once

#include <cstdint>
#include <filesystem>

namespace aetherscan::texture {

enum class BlendMode : std::uint8_t {
    best_view = 0,
    weighted_average = 1,
};

enum class VisibilityMode : std::uint8_t {
    shadow_map = 0,
    hybrid_ray_query = 1,
};

struct TextureOptions {
    // Atlas resolution (square).
    std::uint32_t atlas_resolution{2048};
    float uv_gutter{1.F};
    float uv_max_stretch{1.F / 3.F};
    // UVAtlas charting is otherwise largely serial on large meshes. The
    // aether_drender wrapper partitions faces spatially, unwraps groups concurrently,
    // then packs the combined charts into one atlas.
    std::uint32_t uv_parallel_partitions{8};

    BlendMode blend_mode{BlendMode::weighted_average};
    VisibilityMode visibility_mode{VisibilityMode::hybrid_ray_query};
    std::uint32_t pcf_radius{1};
    bool allow_visibility_fallback{true};

    // The projective bake is only an initialization. By default, refine the
    // atlas directly against all calibrated views with aether_drender's native
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

}  // namespace aetherscan::texture
