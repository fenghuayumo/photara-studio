#pragma once
#include "inference/model_info.hpp"
#include "inference/geometry_prior.hpp"
#include "inference/geometry_predictor.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace photara::inference {
// Metadata for the supported MoGe-2 ViT-B normal checkpoint.
struct MogeModelInfo {
    ModelInfo model;
    std::string source_sha256;
    std::string model_config; // Original checkpoint configuration as JSON.
    std::string weight_precision;
    std::uint32_t format_version = 0;
    std::uint32_t embedding_length = 0;
    std::uint32_t block_count = 0;
    std::uint32_t patch_size = 0;
    std::size_t tensor_count = 0;
    std::size_t tensor_bytes = 0; // Excludes backend allocation alignment.
};

// Validate the GGUF envelope, required metadata and key encoder/head shapes.
// Does not allocate device memory or execute a model.
[[nodiscard]] MogeModelInfo inspect_moge_model(const std::filesystem::path& path);

struct MogeOptions {
    RuntimeOptions runtime{Backend::vulkan, 0, false, 0};
    // Official token-grid rounding is applied to the input aspect ratio.
    // 1200 is the low end of the checkpoint's intended 1200..3600 range.
    std::uint32_t num_tokens = 1200;
    float mask_threshold = 0.5F;
    std::size_t maximum_tensor_bytes = 0;
};
// Vulkan only, with no CPU fallback. Loads weights once. Inputs are undistorted
// pinhole RGB; known intrinsics describe those exact input pixels. Output maps
// retain the input resolution.
// Sessions are serial, as specified by GeometryPredictor. Cancellation throws
// std::runtime_error at stage boundaries (cannot interrupt a GPU dispatch).
[[nodiscard]] std::unique_ptr<GeometryPredictor> create_moge_predictor(
    const std::filesystem::path& path, const MogeOptions& options = {});

enum class MogeRecoveryStatus { recovered, no_support, degenerate };
struct MogeRawGeometry {
    std::uint32_t width = 0, height = 0;
    // Raw, row-major HWC head outputs before resize/remap/normalization.
    std::vector<float> points;
    std::vector<float> normals;
    std::vector<float> mask_logits;
    float log_metric_scale = 0;
};
struct MogeGeometryOptions {
    std::uint32_t width = 0, height = 0; // Original pinhole image dimensions.
    std::optional<PinholeIntrinsics> intrinsics; // Pixel-center convention.
    float mask_threshold = 0.5F;
};
struct MogeRecoveryInfo {
    MogeRecoveryStatus status = MogeRecoveryStatus::no_support;
    bool used_known_intrinsics = false;
    double z_shift = 0; // Before applying metric scale.
    double normalized_focal = 0; // Unknown case: focal / half image diagonal.
    double reprojection_rms = 0; // Normalized view-plane coordinate units.
    std::size_t sample_count = 0;
};
struct MogeGeometryResult {
    GeometryPrior prior;
    MogeRecoveryInfo recovery;
};
// Produces metric camera-Z geometry in original-image coordinates. Failed
// recovery returns an all-invalid prior with an explicit status; unknown K is
// never guessed. Model/device metadata is supplied by the inference adapter.
[[nodiscard]] MogeGeometryResult recover_moge_geometry(const MogeRawGeometry& raw,
    const MogeGeometryOptions& options, const ModelInfo& model, const DeviceInfo& device);
} // namespace photara::inference
