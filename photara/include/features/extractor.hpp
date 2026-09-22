#pragma once

#include "features/types.hpp"

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace photara::features {

// Backend-agnostic feature detector + descriptor.
// Implementations: VLFeat SIFT, SiftGPU, SuperPoint, DISK, ALIKED, etc.
class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;

    [[nodiscard]] virtual std::string_view name() const = 0;
    [[nodiscard]] virtual ExtractorInfo info() const = 0;

    // Per-worker copies for parallel front-ends when !info().thread_safe.
    [[nodiscard]] virtual std::unique_ptr<FeatureExtractor> clone() const = 0;

    [[nodiscard]] virtual FeatureSet extract_gray(
        std::span<const std::uint8_t> pixels,
        std::uint32_t width,
        std::uint32_t height,
        std::size_t row_stride = 0) const = 0;

    // Default loads grayscale via FreeImage and calls extract_gray.
    [[nodiscard]] virtual FeatureSet extract_file(const std::filesystem::path& path) const;

    // Optional RGB path for learned extractors; default throws.
    [[nodiscard]] virtual FeatureSet extract_rgb(
        std::span<const std::uint8_t> pixels,
        std::uint32_t width,
        std::uint32_t height,
        std::size_t row_stride = 0) const;
};

// Keep only features whose keypoint lands on a non-zero mask pixel. The mask
// may have a different resolution from the source image; keypoint centers are
// mapped proportionally. Descriptor rows remain aligned with keypoints.
[[nodiscard]] FeatureSet filter_features_by_mask(
    FeatureSet features,
    std::span<const std::uint8_t> mask,
    std::uint32_t mask_width,
    std::uint32_t mask_height);

}  // namespace photara::features
