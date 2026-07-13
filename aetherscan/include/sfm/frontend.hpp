#pragma once

#include "sfm/checkpoint.hpp"
#include "sfm/geometry.hpp"
#include "sfm/pair_weighting.hpp"
#include "sfm/retrieval.hpp"
#include "sfm/scene.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace aetherscan::sfm {

struct FrontEndOptions {
    double focal_pixels{0.0};  // 0 => 1.2 * max(w,h)
    // A supplied focal is normally an initialization, not a calibration.
    // Set true only for externally calibrated/locked intrinsics.
    bool trust_focal_pixels{false};
    std::size_t neighbor_window{0};  // 0 => exhaustive for small sets, else sequential window
    unsigned thread_count{0};
    // Composable path: extractor × matcher (default siftgpu × gpu_mutual_ratio).
    std::string extractor{"siftgpu"};
    double sift_contrast_threshold{0.005};
    // "gpu_mutual_ratio" (default) | "mutual_ratio" | ...
    // Legacy alias: "siftgpu" normalizes to "gpu_mutual_ratio".
    std::string matcher{"gpu_mutual_ratio"};
    float match_ratio{0.85F};
    bool mutual_check{true};
    // Empty / "none" => compose extractor×matcher.
    // "lightglue_end2end" => fused PairFeaturePipeline (ignores extractor).
    std::string pipeline;
    // Fused LightGluePipeline (--pipeline lightglue_end2end). Requires ONNX.
    std::filesystem::path lightglue_model_path;
    std::string lightglue_extractor{"disk"};  // disk | superpoint
    std::uint32_t lightglue_input_width{1024};
    std::uint32_t lightglue_input_height{1024};
    float lightglue_min_score{0.0F};
    bool lightglue_use_cuda{true};
    RelativePoseOptions relative{};
    float min_pair_weight{0.F};
    PairWeightingOptions pair_weighting{};
    unsigned max_features{27000};
    std::size_t retrieval_min_images{50};
    // When true, sequential window is augmented with BoW pairs (SiftGPU
    // descriptors). LightGlue matching still uses the fused ONNX model.
    bool augment_sequential_with_retrieval{false};
    bool compress_descriptors_u8{true};
    RetrievalOptions retrieval{};
    CheckpointOptions checkpoint{};
};

struct FrontEndTiming {
    unsigned threads_used{0};
    double extract_seconds{0};
    double match_verify_seconds{0};
    double tracks_seconds{0};
};

struct FrontEndResult {
    Scene scene;
    FrontEndTiming timing;
    std::uint64_t tracks_checkpoint_key{0};
};

// Extract features, match pairs, geometric verification, build tracks.
FrontEndResult run_frontend(
    const std::vector<std::filesystem::path>& image_paths,
    const FrontEndOptions& options = {});

}  // namespace aetherscan::sfm
