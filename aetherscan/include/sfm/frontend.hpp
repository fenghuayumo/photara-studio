#pragma once

#include "sfm/checkpoint.hpp"
#include "sfm/geometry.hpp"
#include "sfm/retrieval.hpp"
#include "sfm/scene.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace aetherscan::sfm {

struct FrontEndOptions {
    double focal_pixels{0.0};  // 0 => 1.2 * max(w,h)
    std::size_t neighbor_window{0};  // 0 => exhaustive for small sets, else sequential window
    unsigned thread_count{0};
    std::string extractor{"sift"};
    double sift_contrast_threshold{0.005};
    std::string matcher{"mutual_ratio"};
    float match_ratio{0.85F};
    bool mutual_check{true};
    RelativePoseOptions relative{};
    float min_pair_weight{0.F};
    unsigned max_features{27000};
    std::size_t retrieval_min_images{50};
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
