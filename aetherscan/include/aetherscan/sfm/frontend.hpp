#pragma once

#include "aetherscan/sfm/geometry.hpp"
#include "aetherscan/sfm/scene.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace aetherscan::sfm {

struct FrontEndOptions {
    double focal_pixels{0.0};  // 0 => 1.2 * max(w,h)
    std::size_t neighbor_window{0};  // 0 => exhaustive for small sets, else sequential window
    unsigned thread_count{0};
    std::string extractor{"sift"};
    std::string matcher{"mutual_ratio"};
    RelativePoseOptions relative{};
    float min_pair_weight{0.F};
    unsigned max_features{8000};
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
};

// Extract features, match pairs, geometric verification, build tracks.
FrontEndResult run_frontend(
    const std::vector<std::filesystem::path>& image_paths,
    const FrontEndOptions& options = {});

}  // namespace aetherscan::sfm
