#include "sfm/vocabulary.hpp"
#include "features/types.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace aetherscan;

int main() {
    features::FeatureSet features;
    features.image_width = 90;
    features.image_height = 90;
    features.descriptor_dimension = 4;
    features.metric = features::DescriptorMetric::l2_root;
    // 9 cells, one strong feature per cell with distinct descriptors.
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            features::Keypoint keypoint;
            keypoint.x = static_cast<float>(x * 30 + 15);
            keypoint.y = static_cast<float>(y * 30 + 15);
            keypoint.scale = 4.F;
            keypoint.response = 1.F + static_cast<float>(x + 3 * y);
            features.keypoints.push_back(keypoint);
            const float base = static_cast<float>(x + 3 * y);
            features.descriptors.insert(
                features.descriptors.end(),
                {base, base * 0.5F, base * 0.25F, 1.F});
        }
    }

    const auto sampled = sfm::sample_descriptors_spatially(features, 4, 3);
    if (sampled.size() != 4) {
        std::cerr << "spatial sampler size mismatch\n";
        return 1;
    }

    std::vector<features::FeatureSet> images(8, features);
    for (std::size_t i = 0; i < images.size(); ++i) {
        for (std::size_t d = 0; d < images[i].descriptors.size(); ++d)
            images[i].descriptors[d] += 0.01F * static_cast<float>(i);
    }

    sfm::VocabularyConfig config;
    config.branching = 2;
    config.depth = 3;
    config.max_iterations = 5;
    config.max_descriptors_per_image = 9;
    config.max_training_descriptors = 1000;
    config.sample_grid = 3;
    sfm::VocabularyTree vocabulary;
    vocabulary.train(images, config, features::DescriptorMetric::l2_root);
    if (vocabulary.empty() || vocabulary.word_count() == 0) {
        std::cerr << "vocabulary train failed\n";
        return 2;
    }
    const auto word = vocabulary.quantize(
        std::span<const float>(features.descriptors.data(), 4));
    if (word >= vocabulary.word_count()) {
        std::cerr << "quantize out of range\n";
        return 3;
    }

    features.compress_descriptors_u8();
    if (features.storage != features::DescriptorStorage::uint8 ||
        features.descriptors_u8.size() != 9 * 4 ||
        !features.descriptors.empty()) {
        std::cerr << "u8 compress failed\n";
        return 4;
    }
    features::FeatureSet quantization_probe;
    quantization_probe.descriptor_dimension = 4;
    quantization_probe.keypoints.resize(1);
    quantization_probe.descriptors = {0.F, 0.25F, 0.5F, 1.F};
    quantization_probe.compress_descriptors_u8();
    if (quantization_probe.descriptors_u8 !=
        std::vector<std::uint8_t>{0, 128, 255, 255}) {
        std::cerr << "RootSIFT byte scale is not SiftGPU-compatible\n";
        return 5;
    }
    const auto rows = features.descriptor_rows_float();
    if (rows.size() != 9 * 4) {
        std::cerr << "u8 expand failed\n";
        return 6;
    }
    features.release_descriptors();
    if (!features.descriptors.empty() || !features.descriptors_u8.empty()) {
        std::cerr << "release failed\n";
        return 7;
    }
    std::cout << "vocabulary/memory tests passed words=" << vocabulary.word_count()
              << '\n';
    return 0;
}
