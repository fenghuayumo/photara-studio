#pragma once

#include "features/types.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace aetherscan::sfm {

struct VocabularyConfig {
    std::uint32_t branching{8};
    std::uint32_t depth{5};
    std::uint32_t max_iterations{10};
    std::uint32_t seed{42};
    std::size_t max_descriptors_per_image{2000};
    std::size_t max_training_descriptors{200000};
    unsigned sample_grid{3};
};

// Hierarchical k-means visual vocabulary (openMVS-style), native float L2.
class VocabularyTree {
public:
    VocabularyTree() = default;

    [[nodiscard]] bool empty() const noexcept { return nodes_.empty(); }
    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] std::size_t word_count() const noexcept { return word_count_; }
    [[nodiscard]] const VocabularyConfig& config() const noexcept { return config_; }
    [[nodiscard]] features::DescriptorMetric metric() const noexcept { return metric_; }

    void train(
        std::span<const features::FeatureSet> images,
        const VocabularyConfig& config,
        features::DescriptorMetric metric = features::DescriptorMetric::l2_root);

    [[nodiscard]] std::uint32_t quantize(std::span<const float> descriptor) const;

    void save(const std::filesystem::path& path) const;
    void load(const std::filesystem::path& path);

private:
    struct Node {
        std::uint32_t first_child{0};
        std::uint16_t child_count{0};
        std::int32_t word_id{-1};
    };

    // Fill nodes_[node_index]. Direct children are reserved contiguously at
    // first_child .. first_child+child_count-1 before their subtrees grow.
    void build_into(
        std::size_t node_index,
        std::vector<std::size_t>& subset,
        unsigned depth,
        std::span<const float> descriptors);

    VocabularyConfig config_{};
    features::DescriptorMetric metric_{features::DescriptorMetric::l2_root};
    std::size_t dimension_{0};
    std::size_t word_count_{0};
    std::vector<Node> nodes_;
    std::vector<float> centroids_;  // node * dimension
};

// Spatially balanced keypoint sampling (3x3 round-robin by response/scale).
std::vector<features::FeatureIndex> sample_descriptors_spatially(
    const features::FeatureSet& features,
    std::size_t maximum,
    unsigned grid_size = 3);

}  // namespace aetherscan::sfm
