#pragma once

#include "sfm/checkpoint.hpp"
#include "sfm/geometry.hpp"
#include "sfm/pair_weighting.hpp"
#include "sfm/retrieval.hpp"
#include "sfm/scene.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace photara::sfm {

class AlignLivePreview;
struct MissingViewRecoveryCache;
std::shared_ptr<MissingViewRecoveryCache> make_missing_view_recovery_cache();

struct FrontEndOptions {
    CameraModel camera_model{CameraModel::automatic};
    double focal_pixels{0.0};  // 0 => 1.2 * max(w,h), fisheye: 0.5 * max(w,h)
    // A supplied focal is normally an initialization, not a calibration.
    // Set true only for externally calibrated/locked intrinsics.
    bool trust_focal_pixels{false};
    std::size_t neighbor_window{3};  // 0 => exhaustive for small sets, else sequential window
    unsigned thread_count{0};
    // Optional valid-region masks matched by image filename/stem. Zero pixels
    // are excluded before retrieval and matching; non-zero pixels are kept.
    std::filesystem::path mask_dir;
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
    bool augment_sequential_with_retrieval{true};
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
    // Supplement existing proposals with a scene-wide vocabulary. London
    // benefits, but Alameda loses a camera; require explicit selection.
    bool whole_scene_rescue{false};
    // Supplemental histogram sampling across detector levels; under evaluation.
    bool balanced_rescue{false};
    std::size_t progressive_rescue_max_pairs_per_image{96};
    // Re-detect only weak views at this larger budget. Existing descriptors
    // remain first so cached primary-match feature indices stay valid.
    unsigned progressive_rescue_max_features{54000};
    // Post-refinement resection may add overlap neighbors of its original anchors.
    // Zero disables the retry; earlier product recovery keeps its 16-view budget.
    unsigned missing_view_retry_max_neighbors{24};
    // Copies share bounded recovery work across targets and rounds.
    // This internal context never stores final pose acceptance.
    std::shared_ptr<MissingViewRecoveryCache> missing_view_cache{make_missing_view_recovery_cache()};
    // Additional consensus floor, used by conservative post-refinement recovery.
    double missing_view_minimum_consensus_ratio{0};
    bool compress_descriptors_u8{true};
    RetrievalOptions retrieval{};
    CheckpointOptions checkpoint{};
    // Optional stage-level feature/pair CSVs for diagnosing cold frontend runs.
    // Does not affect feature, correspondence or geometry selection.
    std::filesystem::path diagnostics_directory;
    // Editor sidecar. Null when running from the CLI without --working-sfm.
    AlignLivePreview* live_preview{};
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

// Late descriptor-to-map recovery. Existing cameras and intrinsics are frozen;
// new poses must pass the stable resection's independent held-out checks.
// Mix verified visual neighbors with temporal neighbors at a fixed total budget.
std::vector<Index> select_missing_view_anchors(
    const Scene& scene, Index target, unsigned maximum_neighbors = 16);
std::vector<Index> select_missing_view_support_anchors(
    const Scene& scene, Index target, unsigned maximum_neighbors = 24);
// A non-empty target list limits both normal and low-contrast attempts.
unsigned recover_missing_views(Scene& scene, const FrontEndOptions& options = {},
    const std::vector<Index>& target_ids = {}, bool require_pose_improvement = false);
// Revisit missing views after pose refinement without moving existing geometry.
// Requires at least half-consensus for both fitting and held-out observations.
unsigned recover_post_refinement_views(Scene& scene, const FrontEndOptions& options = {});
// Conservative registered-camera repair. Failure leaves the input scene intact.
std::vector<Index> select_weak_alignment_views(const Scene& scene);
unsigned refine_weak_alignment_views(Scene& scene, const FrontEndOptions& options = {});
// Bounded, independently validated re-localization of weak scale-breaking leaves
// together with missing views. Other registered poses and intrinsics stay fixed.
std::vector<Index> select_weak_position_repairs(const Scene& scene);
unsigned recover_alignment_views(Scene& scene, const FrontEndOptions& options = {});

}  // namespace photara::sfm
