#pragma once

#include "aetherscan/features/features.hpp"
#include "aetherscan/sfm/mapping.hpp"
#include "aetherscan/sfm/two_view.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace aetherscan::sfm {

struct FrontEndOptions {
    // Backend names resolved through features::create_extractor / create_matcher.
    // Defaults: VLFeat SIFT + mutual-ratio. Swap to "superpoint" once implemented.
    std::string extractor_name{"sift"};
    std::string matcher_name{"mutual_ratio"};

    // Optional prototypes. When set, they override extractor_name / matcher_name and
    // are clone()'d per worker thread (required when !info().thread_safe).
    std::shared_ptr<features::FeatureExtractor> extractor;
    std::shared_ptr<features::FeatureMatcher> matcher;

    features::SiftOptions sift;  // used when constructing default "sift" extractor
    features::DescriptorMatcherOptions matcher_options;
    TwoViewOptions two_view;
    double focal_pixels{900.0};
    std::size_t neighbor_window{3};
    unsigned thread_count{0};  // 0 = hardware concurrency
};

struct FrontEndTiming {
    double extract_seconds{};
    double match_verify_seconds{};
    double tracks_seconds{};
    unsigned threads_used{1};
};

struct FrontEndResult {
    Scene scene;
    std::vector<RelativePoseEdge> edges;
    std::size_t seed_edge_index{0};
    FrontEndTiming timing;
};

// Parallel extract + neighbor-window match/E-RANSAC + track building.
FrontEndResult run_frontend(
    const std::vector<std::filesystem::path>& image_files,
    const FrontEndOptions& options = {});

}  // namespace aetherscan::sfm
