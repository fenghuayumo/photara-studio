#pragma once
#include "inference/model_info.hpp"
#include <cstddef>
#include <cstdint>
#include <string>

namespace photara::inference {
// Inspection and weight loading are implemented. Forward prediction is not.
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
} // namespace photara::inference
