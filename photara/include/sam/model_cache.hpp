#pragma once

// Mask-model discovery and download metadata.
//
// Third-party weights and user agreement receipts are never bundled.
// Every model family requires explicit user consent before download or use.

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace photara::sam {

enum class MaskModelKind { sam3, sam2, edgetam, text_detector };

struct MaskModelEntry {
    const char* id;
    const char* label;
    const char* filename;
    const char* url;
    std::uint64_t bytes;
    MaskModelKind kind;
    bool licence_required; // Always true: product policy applies to every family.
};

// Stable ids are used by projects and the CLI; labels are for the editor.
[[nodiscard]] const std::vector<MaskModelEntry>& mask_model_catalog();
[[nodiscard]] const MaskModelEntry* find_mask_model(std::string_view id);

// Where a download from this app is stored.
[[nodiscard]] std::filesystem::path user_model_path(std::string_view filename);

// True when the file looks like a finished download, not a partial file.
[[nodiscard]] bool model_file_ready(
    const std::filesystem::path& path, std::uint64_t expected_bytes);

// User cache, then PHOTARA_SAM_MODEL. Empty when nothing usable is on disk.
[[nodiscard]] std::filesystem::path locate_model(std::string_view id = "sam3-q4_0");
// Resolve a catalog id, cached filename, or explicit path without downloading.
[[nodiscard]] std::filesystem::path resolve_model(const std::filesystem::path& value);

// User cache, then PHOTARA_TEXT_DETECTOR_MODEL / _VOCAB.
[[nodiscard]] std::filesystem::path locate_text_detector_model();
[[nodiscard]] std::filesystem::path locate_text_detector_vocab();
[[nodiscard]] bool text_detector_ready();

struct ModelLicense {
    const char* id;
    const char* label;
    const char* url;
    const char* revision;
};
[[nodiscard]] const ModelLicense& model_license(MaskModelKind kind);
[[nodiscard]] const ModelLicense& moge_model_license();
// Reads only the checkpoint header, including renamed custom checkpoints.
[[nodiscard]] const ModelLicense& segmentation_license(const std::filesystem::path& model);
[[nodiscard]] const ModelLicense* find_model_license(std::string_view id);
[[nodiscard]] std::filesystem::path model_agreements_directory();
[[nodiscard]] bool license_accepted(const ModelLicense& license,
    const std::filesystem::path& directory = model_agreements_directory());
// Called only from an explicit UI or CLI acceptance action.
void accept_model_license(const ModelLicense& license,
    const std::filesystem::path& directory = model_agreements_directory());
void require_model_license(const ModelLicense& license);
[[nodiscard]] bool license_accepted();
void accept_license();

// Legacy SAM 3 metadata retained by the existing licence dialog.
inline constexpr const char* k_model_file = "sam3-q4_0.ggml";
inline constexpr const char* k_model_url =
    "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam3-q4_0.ggml";
inline constexpr std::uint64_t k_model_bytes = 706'606'590ULL;
inline constexpr const char* k_license_url =
    "https://github.com/facebookresearch/sam3/blob/main/LICENSE";

inline constexpr const char* k_default_sam2_model = "sam2.1-base-plus";
inline constexpr const char* k_default_text_detector_model =
    "grounding-dino-tiny.safetensors";
inline constexpr const char* k_default_text_detector_vocab =
    "grounding-dino-vocab.txt";

}  // namespace photara::sam
