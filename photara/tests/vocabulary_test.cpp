#include "sfm/vocabulary.hpp"
#include "features/types.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace photara;

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
    // Missing detector responses must not force supplemental retrieval to use
    // only the first detector levels. Preserve sparse-cell quotas and indices.
    features::FeatureSet ordered;
    ordered.image_width = ordered.image_height = 90;
    ordered.keypoints.resize(10);
    for (unsigned i = 0; i < 10; ++i) {
        ordered.keypoints[i].x = i == 9 ? 75.F : 15.F;
        ordered.keypoints[i].y = 15.F;
        ordered.keypoints[i].response = 0.F;
    }
    const auto prefix = sfm::sample_descriptors_spatially(ordered, 4, 3);
    const auto balanced = sfm::sample_descriptors_spatially(ordered, 4, 3, true);
    if (prefix != std::vector<features::FeatureIndex>{0, 9, 1, 2} ||
        balanced != std::vector<features::FeatureIndex>{0, 9, 3, 6} ||
        sfm::sample_descriptors_spatially(ordered, 0, 3, true).size() != 0 ||
        sfm::sample_descriptors_spatially(ordered, 20, 3, true).size() != 10) {
        std::cerr << "detector-level sampling loses coverage or sparse cells\n";
        return 9;
    }

    // A capped vocabulary must see later rooms, even when the sequence prefix
    // alone contains enough descriptors to fill the entire training budget.
    std::vector<features::FeatureSet> rooms(8);
    for (std::size_t i = 0; i < rooms.size(); ++i) {
        auto& room = rooms[i];
        room.image_width = room.image_height = 90;
        room.descriptor_dimension = 2;
        room.keypoints.resize(9);
        for (unsigned j = 0; j < 9; ++j) {
            room.keypoints[j].x = room.keypoints[j].y = 45.F;
            room.descriptors.insert(room.descriptors.end(),
                i < 4 ? std::initializer_list<float>{1.F, 0.F} : std::initializer_list<float>{0.F, 1.F});
        }
    }
    auto capped = config;
    capped.depth = 1;
    capped.max_training_descriptors = 8;
    capped.sample_all_images = true;
    sfm::VocabularyTree whole_scene;
    whole_scene.train(rooms, capped, features::DescriptorMetric::l2_root);
    if (whole_scene.quantize(std::span<const float>(rooms.front().descriptors.data(), 2)) ==
        whole_scene.quantize(std::span<const float>(rooms.back().descriptors.data(), 2))) {
        std::cerr << "capped vocabulary ignores later images\n";
        return 8;
    }
    std::cout << "vocabulary/memory tests passed words=" << vocabulary.word_count()
              << '\n';
    return 0;
}
