#pragma once

// In-process SAM 3 mask generation. The editor downloads the checkpoint after
// the user accepts Meta's licence; this call only runs a model that is already
// on disk and writes keep-masks (255 = keep) for SfM and 3DGS.

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace photara::sfm {
struct Scene;
}

namespace photara::sam {

struct GenerateOptions {
    std::filesystem::path model;
    std::filesystem::path output_dir;
    std::vector<std::filesystem::path> images;
    std::string text;
    std::string negative_text;
    // auto, cpu, cuda, vulkan, or metal.
    std::string backend = "auto";
    // The prompt names the subject to keep. Off means the prompt names
    // distractors to remove.
    bool keep_prompted = true;
    bool video = true;
    // 0 uses the checkpoint's native resolution. Non-native graph shapes are
    // accepted only when the selected SAM model supports them.
    int max_size = 0;
    float threshold = 0.5F;
    float nms = 0.1F;
    // Optional sparse reconstruction used to keep the prompted instance
    // consistent across an orbit. The closest central_fraction of
    // triangulated points to the registered-camera centroid are projected
    // into every image and used only as SAM location/selection guidance.
    // Their pixels are never copied into the output mask.
    const sfm::Scene* sparse_scene = nullptr;
    float central_fraction = 0.01F;
    float min_area_fraction = 0.001F;
    float max_area_fraction = 0.65F;
    int close_kernel = 5;
    bool fill_holes = true;
    // Interactive single-image generation must not replace a preview with an
    // empty result when the prompt did not identify a target.
    bool require_target = false;
    // Optional editor hooks. Cancellation is checked between images because
    // the SAM decoder itself has no interrupt path.
    const std::atomic<bool>* cancel = nullptr;
    std::function<void(std::size_t completed, std::size_t total)> progress;
};

struct GenerateResult {
    std::size_t written{};
    std::size_t skipped{};
    std::size_t geometry_guided{};
    std::size_t empty{};
};

[[nodiscard]] bool built_with_sam();

// Writes one PNG per image, named `<stem>.png`, in sorted call order so video
// tracking sees the capture in sequence. Throws on a model or frame failure.
// Prefetches one input and saves one output in the background. Completion
// callbacks run on the calling thread, in input order, after successful saves.
// All pending I/O finishes before return; save failures propagate to the caller.
// The selected inference backend is released before the function returns.
GenerateResult generate_masks(const GenerateOptions& options);

}  // namespace photara::sam
