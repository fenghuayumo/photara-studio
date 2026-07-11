#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aetherscan::features {

using FeatureIndex = std::uint32_t;

enum class DescriptorMetric {
    l2,          // plain L2 (SIFT before RootSIFT)
    l2_root,     // RootSIFT / Hellinger (default SIFT path)
    cosine,      // L2-normalized vectors, score = 1 - cos
    inner_product  // learned descriptors often use IP / cosine
};

struct Keypoint {
    float x{};
    float y{};
    float scale{1.0F};
    float orientation{};
    float response{};
};

// Descriptors are row-major [keypoint_count, descriptor_dimension].
struct FeatureSet {
    std::uint32_t image_width{};
    std::uint32_t image_height{};
    std::size_t descriptor_dimension{};
    std::vector<Keypoint> keypoints;
    std::vector<float> descriptors;
    DescriptorMetric metric{DescriptorMetric::l2_root};
    std::string extractor_name;

    void validate() const;
};

struct FeatureMatch {
    FeatureIndex query{};
    FeatureIndex train{};
    float score{};
};

struct MatchSet {
    std::vector<FeatureMatch> matches;
};

struct ImagePairFeatures {
    FeatureSet first;
    FeatureSet second;
    MatchSet matches;
};

struct ExtractorInfo {
    bool thread_safe{false};   // false → clone one instance per worker thread
    bool accepts_rgb{false};   // true if extract_rgb is meaningful
    bool accepts_gray{true};
    DescriptorMetric metric{DescriptorMetric::l2_root};
    std::size_t typical_descriptor_dimension{128};
};

}  // namespace aetherscan::features
