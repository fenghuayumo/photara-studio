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
    CameraModel camera_model{CameraModel::pinhole};
    double focal_pixels{0.0};  // 0 => 1.2 * max(w,h), fisheye: 0.5 * max(w,h)
    // A supplied focal is normally an initialization, not a calibration.
    // Set true only for externally calibrated/locked intrinsics.
    bool trust_focal_pixels{false};
    std::size_t neighbor_window{0};  // 0 => exhaustive for small sets, else sequential window
    unsigned thread_count{0};
    // Composable path: extractor × matcher (default siftgpu × gpu_mutual_ratio).
    std::string extractor{"siftgpu"};
    double sift_contrast_threshold{0.005};
    // "gpu_mutual_ratio" (default) | "mutual_ratio" | "lightglue" |
    // "hybrid_lightglue" | ...
    // Legacy alias: "siftgpu" normalizes to "gpu_mutual_ratio".
    std::string matcher{"gpu_mutual_ratio"};
    float match_ratio{0.8F};
    bool mutual_check{true};
    // Empty / "none" => compose extractor×matcher.
    // "lightglue_end2end" => fused PairFeaturePipeline (ignores extractor/matcher).
    std::string pipeline;
    // Learned extractor ONNX (--extractor superpoint|disk|aliked).
    std::filesystem::path extractor_model_path;
    std::uint32_t extractor_input_width{1024};
    std::uint32_t extractor_input_height{1024};
    // Negative selects the backend default (ALIKED=0.2, SuperPoint/DISK=0).
    float extractor_min_score{-1.F};
    bool extractor_use_cuda{true};
    // Descriptor LightGlue matcher ONNX (--matcher lightglue or
    // hybrid_lightglue), or fused end2end model for lightglue_end2end.
    std::filesystem::path lightglue_model_path;
    std::string lightglue_extractor{"disk"};  // end2end head only: disk|superpoint
    std::uint32_t lightglue_input_width{1024};  // end2end network size
    std::uint32_t lightglue_input_height{1024};
    float lightglue_min_score{0.0F};
    // Hybrid rescue attention budget; zero disables the cap.
    unsigned hybrid_lightglue_max_features{2048};
    bool lightglue_use_cuda{true};
    RelativePoseOptions relative{};
    float min_pair_weight{0.F};
    PairWeightingOptions pair_weighting{};
    unsigned max_features{27000};
    std::size_t retrieval_min_images{50};
    // When true, sequential window is augmented with BoW pairs (requires
    // descriptors; temporary SiftGPU extract may be used for retrieval only).
    bool augment_sequential_with_retrieval{false};
    // After the primary GPU pass, expand only weak verified views against the
    // remaining images with an independently configurable ratio test. This
    // preserves the fast path for well-connected views while recovering
    // difficult viewpoints without paying for an unconditional all-pairs run.
    bool progressive_pair_expansion{true};
    // Experimental: extra graph connectivity has not yet demonstrated correct
    // branch coordinates on Alameda. Keep disabled in the default pipeline.
    bool structural_pair_expansion{false};
    unsigned progressive_min_verified_degree{4};
    float progressive_rescue_match_ratio{0.85F};
    unsigned progressive_rescue_min_inliers{20};
    // Small scenes can afford exhaustive weak-view expansion. Larger scenes
    // use a bounded union of wider sequential and deeper retrieval proposals.
    std::size_t progressive_max_images{500};
    std::size_t progressive_rescue_neighbor_window{16};
    std::size_t progressive_rescue_retrieval_top_k{64};
    std::size_t progressive_rescue_max_pairs_per_image{96};
    // Re-detect only weak views at this larger budget. Existing descriptors
    // remain first so cached primary-match feature indices stay valid.
    unsigned progressive_rescue_max_features{27000};
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
