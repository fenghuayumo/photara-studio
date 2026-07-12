#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace aetherscan::features {

using FeatureIndex = std::uint32_t;

inline std::uint64_t next_descriptor_identity() noexcept {
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

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
    FeatureSet() = default;
    FeatureSet(const FeatureSet& other)
        : image_width(other.image_width),
          image_height(other.image_height),
          descriptor_dimension(other.descriptor_dimension),
          keypoints(other.keypoints),
          descriptors(other.descriptors),
          metric(other.metric),
          extractor_name(other.extractor_name) {}
    FeatureSet(FeatureSet&& other) noexcept
        : image_width(other.image_width),
          image_height(other.image_height),
          descriptor_dimension(other.descriptor_dimension),
          keypoints(std::move(other.keypoints)),
          descriptors(std::move(other.descriptors)),
          descriptor_generation(other.descriptor_generation),
          metric(other.metric),
          extractor_name(std::move(other.extractor_name)) {}
    FeatureSet& operator=(const FeatureSet& other) {
        if (this == &other) return *this;
        image_width = other.image_width;
        image_height = other.image_height;
        descriptor_dimension = other.descriptor_dimension;
        keypoints = other.keypoints;
        descriptors = other.descriptors;
        descriptor_generation = 0;
        descriptor_identity = next_descriptor_identity();
        metric = other.metric;
        extractor_name = other.extractor_name;
        return *this;
    }
    FeatureSet& operator=(FeatureSet&& other) noexcept {
        if (this == &other) return *this;
        image_width = other.image_width;
        image_height = other.image_height;
        descriptor_dimension = other.descriptor_dimension;
        keypoints = std::move(other.keypoints);
        descriptors = std::move(other.descriptors);
        descriptor_generation = other.descriptor_generation;
        descriptor_identity = next_descriptor_identity();
        metric = other.metric;
        extractor_name = std::move(other.extractor_name);
        return *this;
    }

    std::uint32_t image_width{};
    std::uint32_t image_height{};
    std::size_t descriptor_dimension{};
    std::vector<Keypoint> keypoints;
    std::vector<float> descriptors;
    // Increment after modifying descriptor values or layout once a matcher may
    // have prepared an index for this FeatureSet.
    std::uint64_t descriptor_generation{};
    std::uint64_t descriptor_identity{next_descriptor_identity()};
    DescriptorMetric metric{DescriptorMetric::l2_root};
    std::string extractor_name;

    void mark_descriptors_modified() noexcept { ++descriptor_generation; }
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
