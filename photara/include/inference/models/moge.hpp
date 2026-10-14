#pragma once
#include "inference/model_info.hpp"
#include "inference/geometry_prior.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace photara::inference {
// Model metadata plus geometry-recovery options. Full raw-image prediction is
// still separate from the implemented encoder/decoder and postprocess stages.
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
