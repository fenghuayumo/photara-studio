#pragma once

// SAM 3 checkpoint location and licence acceptance.
//
// The weights are Meta's. They are never bundled. A download starts only after
// the user has accepted the SAM 3 licence in the editor; this file only
// records that fact and finds a checkpoint that is already on disk.

#include <cstdint>
#include <filesystem>

namespace photara::sam {

inline constexpr const char* k_model_file = "sam3-q4_0.ggml";
inline constexpr const char* k_model_url =
    "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam3-q4_0.ggml";
// A finished q4_0 checkpoint is about 707 MiB. A truncated download is smaller.
inline constexpr std::uint64_t k_model_bytes = 707ull << 20;
inline constexpr const char* k_license_url =
    "https://github.com/facebookresearch/sam3/blob/main/LICENSE";

// Where a download from this app is stored.
[[nodiscard]] std::filesystem::path user_model_path();

// True when the file looks like a finished checkpoint, not a partial download.
[[nodiscard]] bool model_file_ready(const std::filesystem::path& path);

// User cache, then PHOTARA_SAM_MODEL. Empty when nothing usable is on disk.
[[nodiscard]] std::filesystem::path locate_model();

[[nodiscard]] bool license_accepted();
void accept_license();

}  // namespace photara::sam
