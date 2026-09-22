#pragma once

// In-process SAM 3 mask generation. The editor downloads the checkpoint after
// the user accepts Meta's licence; this call only runs a model that is already
// on disk and writes keep-masks (255 = keep) for SfM and 3DGS.

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace photara::sam {

struct GenerateOptions {
    std::filesystem::path model;
    std::filesystem::path output_dir;
    std::vector<std::filesystem::path> images;
    std::string text;
    std::string negative_text;
    // The prompt names the subject to keep. Off means the prompt names
    // distractors to remove.
    bool keep_prompted = true;
    bool video = true;
    int max_size = 1600;
    float threshold = 0.5F;
    float nms = 0.1F;
};

struct GenerateResult {
    std::size_t written{};
    std::size_t skipped{};
};

[[nodiscard]] bool built_with_sam();

// Writes one PNG per image, named `<stem>.png`, in sorted call order so video
// tracking sees the capture in sequence. Throws on a model or frame failure.
// The Vulkan device is released before the function returns.
GenerateResult generate_masks(const GenerateOptions& options);

}  // namespace photara::sam
