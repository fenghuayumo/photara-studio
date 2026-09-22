#include "features/extractor.hpp"

#include "io/image.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace photara::features {

FeatureSet FeatureExtractor::extract_file(const std::filesystem::path& path) const {
    if (info().accepts_gray) {
        const io::GrayImage image = io::load_gray(path);
        return extract_gray(image.pixels, image.width, image.height);
    }
    if (info().accepts_rgb) {
        const io::RgbImage image = io::load_rgb(path);
        return extract_rgb(image.pixels, image.width, image.height);
    }
    throw std::runtime_error(
        std::string(name()) + " does not accept gray or RGB image input");
}

FeatureSet FeatureExtractor::extract_rgb(
    std::span<const std::uint8_t> /*pixels*/, std::uint32_t /*width*/,
    std::uint32_t /*height*/, std::size_t /*row_stride*/) const {
    throw std::runtime_error(std::string(name()) + " does not implement extract_rgb");
}

FeatureSet filter_features_by_mask(
    FeatureSet features, const std::span<const std::uint8_t> mask,
    const std::uint32_t mask_width, const std::uint32_t mask_height) {
    if (mask_width == 0 || mask_height == 0 ||
        mask.size() != static_cast<std::size_t>(mask_width) * mask_height)
        throw std::invalid_argument("Feature mask dimensions do not match its pixels");
    if (features.image_width == 0 || features.image_height == 0)
        throw std::invalid_argument("Cannot apply a mask to unsized features");

    const bool has_float =
        features.storage == DescriptorStorage::float32 &&
        !features.descriptors.empty();
    const bool has_u8 =
        features.storage == DescriptorStorage::uint8 &&
        !features.descriptors_u8.empty();
    const std::size_t expected_descriptors =
        features.keypoints.size() * features.descriptor_dimension;
    if ((has_float && features.descriptors.size() != expected_descriptors) ||
        (has_u8 && features.descriptors_u8.size() != expected_descriptors))
        throw std::invalid_argument(
            "Cannot apply a mask to misaligned feature descriptors");

    FeatureSet filtered = features;
    filtered.keypoints.clear();
    filtered.descriptors.clear();
    filtered.descriptors_u8.clear();
    filtered.keypoints.reserve(features.keypoints.size());
    if (has_float) filtered.descriptors.reserve(features.descriptors.size());
    if (has_u8) filtered.descriptors_u8.reserve(features.descriptors_u8.size());

    for (std::size_t index = 0; index < features.keypoints.size(); ++index) {
        const Keypoint& keypoint = features.keypoints[index];
        if (!std::isfinite(keypoint.x) || !std::isfinite(keypoint.y) ||
            keypoint.x < 0.F || keypoint.y < 0.F ||
            keypoint.x >= static_cast<float>(features.image_width) ||
            keypoint.y >= static_cast<float>(features.image_height))
            continue;
        const auto mask_x = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(
                (keypoint.x + 0.5F) * mask_width / features.image_width),
            mask_width - 1);
        const auto mask_y = std::min<std::uint32_t>(
            static_cast<std::uint32_t>(
                (keypoint.y + 0.5F) * mask_height / features.image_height),
            mask_height - 1);
        if (mask[static_cast<std::size_t>(mask_y) * mask_width + mask_x] == 0)
            continue;

        filtered.keypoints.push_back(keypoint);
        const std::size_t begin = index * features.descriptor_dimension;
        const std::size_t end = begin + features.descriptor_dimension;
        if (has_float)
            filtered.descriptors.insert(
                filtered.descriptors.end(),
                features.descriptors.begin() + begin,
                features.descriptors.begin() + end);
        if (has_u8)
            filtered.descriptors_u8.insert(
                filtered.descriptors_u8.end(),
                features.descriptors_u8.begin() + begin,
                features.descriptors_u8.begin() + end);
    }
    filtered.mark_descriptors_modified();
    return filtered;
}

}  // namespace photara::features
