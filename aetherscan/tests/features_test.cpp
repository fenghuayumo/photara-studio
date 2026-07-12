#include "features/features.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

int main() {
    constexpr std::uint32_t width = 512, height = 384;
    std::vector<std::uint8_t> first(static_cast<std::size_t>(width) * height);
    std::vector<std::uint8_t> second(first.size(), 0);
    std::mt19937 random(1234);
    for (auto& pixel : first) pixel = static_cast<std::uint8_t>(random() & 0xffU);
    constexpr std::uint32_t shift_x = 7, shift_y = 5;
    for (std::uint32_t y = shift_y; y < height; ++y)
        for (std::uint32_t x = shift_x; x < width; ++x)
            second[static_cast<std::size_t>(y) * width + x] =
                first[static_cast<std::size_t>(y - shift_y) * width + x - shift_x];
    aetherscan::features::SiftOptions options;
    options.maximum_features = 3000;
    aetherscan::features::SiftExtractor extractor(options);
    const auto features0 = extractor.extract_gray(first, width, height);
    const auto features1 = extractor.extract_gray(second, width, height);
    const auto matches = aetherscan::features::match_descriptors(features0, features1);
    std::cout << "features=" << features0.keypoints.size() << ","
              << features1.keypoints.size() << " matches=" << matches.matches.size() << '\n';
    if (features0.keypoints.size() < 150 || features1.keypoints.size() < 150 ||
        matches.matches.size() < 80) return 1;
    for (const auto& match : matches.matches) {
        if (match.query >= features0.keypoints.size() || match.train >= features1.keypoints.size())
            return 2;
    }
    aetherscan::features::DescriptorMatcherOptions ann_options;
    ann_options.ann_min_features = 1;
    aetherscan::features::MutualRatioMatcher prepared_matcher(ann_options);
    prepared_matcher.prepare(features0);
    aetherscan::features::FeatureSet resized_train = features1;
    prepared_matcher.prepare(resized_train);
    resized_train.keypoints.resize(32);
    resized_train.descriptors.resize(
        resized_train.keypoints.size() * resized_train.descriptor_dimension);
    resized_train.mark_descriptors_modified();
    const auto resized_matches =
        prepared_matcher.match(features0, resized_train);
    for (const auto& match : resized_matches.matches) {
        if (match.query >= features0.keypoints.size() ||
            match.train >= resized_train.keypoints.size())
            return 3;
    }
    if (aetherscan::features::SiftGpuExtractor::is_built()) {
        aetherscan::features::SiftGpuOptions gpu_options;
        gpu_options.maximum_features = 3000;
        aetherscan::features::SiftGpuExtractor gpu_extractor(gpu_options);
        aetherscan::features::SiftGpuMatcher gpu_matcher;
        if (gpu_extractor.is_available() && gpu_matcher.is_available()) {
            const auto gpu_features0 =
                gpu_extractor.extract_gray(first, width, height);
            const auto gpu_features1 =
                gpu_extractor.extract_gray(second, width, height);
            const auto gpu_matches =
                gpu_matcher.match(gpu_features0, gpu_features1);
            std::cout << " gpu_features=" << gpu_features0.keypoints.size()
                      << "," << gpu_features1.keypoints.size()
                      << " gpu_matches=" << gpu_matches.matches.size()
                      << '\n';
            if (gpu_features0.keypoints.size() < 150 ||
                gpu_features1.keypoints.size() < 150 ||
                gpu_matches.matches.size() < 80)
                return 4;
        }
    }
    return 0;
}
