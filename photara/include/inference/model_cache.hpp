#pragma once
#include <cstdint>
#include <filesystem>
#include <string>

namespace photara::inference {
// Existing Photara/models directory. No creation, download or licence changes.
[[nodiscard]] std::filesystem::path model_cache_root();
[[nodiscard]] std::filesystem::path cached_model_path(const std::string& filename);
// Size check only, not a checksum or a model-format validation.
[[nodiscard]] bool model_file_ready(
    const std::filesystem::path& path, std::uintmax_t minimum_bytes = 1);
} // namespace photara::inference
