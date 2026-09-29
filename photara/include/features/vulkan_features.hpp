#pragma once

#include "features/extractor.hpp"
#include "features/matcher.hpp"
#include "features/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace photara::features {

// Vulkan compute translation of the default SfM feature path: the SiftGPU
// CUDA SIFT pipeline (Gaussian pyramid -> DoG -> keypoints -> orientation ->
// descriptors) and the native CUDA mutual-ratio matcher. Parameter names and
// semantics mirror SiftGpuOptions / SiftGpuMatcherOptions one-to-one so the
// frontend can swap backends without changing tuning.
struct SiftVulkanOptions {
    std::size_t maximum_features{8192};
    std::uint32_t maximum_image_dimension{5120};
    float peak_threshold{0.005F};
    float edge_threshold{20.0F};
    std::uint32_t maximum_orientations{2};
    int first_octave{-1};
    std::uint32_t octave_layers{3};
    int device_index{0};
    bool root_sift{true};
};

class SiftVulkanExtractor final : public FeatureExtractor {
public:
    explicit SiftVulkanExtractor(SiftVulkanOptions options = {});
    ~SiftVulkanExtractor() override;
    SiftVulkanExtractor(SiftVulkanExtractor&&) noexcept;
    SiftVulkanExtractor& operator=(SiftVulkanExtractor&&) noexcept;
    SiftVulkanExtractor(const SiftVulkanExtractor&) = delete;
    SiftVulkanExtractor& operator=(const SiftVulkanExtractor&) = delete;

    [[nodiscard]] static bool is_built() noexcept;
    [[nodiscard]] bool is_available() const noexcept;

    [[nodiscard]] std::string_view name() const override { return "vulkan_sift"; }
    [[nodiscard]] ExtractorInfo info() const override;
    [[nodiscard]] std::unique_ptr<FeatureExtractor> clone() const override;

    [[nodiscard]] FeatureSet extract_gray(
        std::span<const std::uint8_t> pixels,
        std::uint32_t width,
        std::uint32_t height,
        std::size_t row_stride = 0) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

struct VulkanMutualRatioMatcherOptions {
    float ratio_threshold{0.8F};
    bool mutual_check{true};
    std::size_t maximum_features{32768};
    int device_index{0};
};

// Vulkan translation of photara's native CUDA matcher (matcher_cuda.cu):
// 32x32 descriptor dot-product tiles keep per-row/per-column top-two
// reductions on the GPU, then a finish kernel applies SiftGPU's angular
// distance threshold (acos(dot / 262144) < 0.7), the ratio test, and the
// mutual check.
class VulkanMutualRatioMatcher final : public FeatureMatcher {
public:
    explicit VulkanMutualRatioMatcher(VulkanMutualRatioMatcherOptions options = {});
    ~VulkanMutualRatioMatcher() override;
    VulkanMutualRatioMatcher(VulkanMutualRatioMatcher&&) noexcept;
    VulkanMutualRatioMatcher& operator=(VulkanMutualRatioMatcher&&) noexcept;
    VulkanMutualRatioMatcher(const VulkanMutualRatioMatcher&) = delete;
    VulkanMutualRatioMatcher& operator=(const VulkanMutualRatioMatcher&) = delete;

    [[nodiscard]] static bool is_built() noexcept;
    [[nodiscard]] bool is_available() const noexcept;

    [[nodiscard]] std::string_view name() const override {
        return "vulkan_mutual_ratio";
    }
    [[nodiscard]] std::unique_ptr<FeatureMatcher> clone() const override;
    void clear_prepared() override;

    [[nodiscard]] MatchSet match(
        const FeatureSet& query, const FeatureSet& train) const override;
    [[nodiscard]] std::vector<MatchSet> match_batch(
        std::span<const Pair> pairs) const override;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

// Registers "vulkan_sift" / "vulkan_mutual_ratio" when the Vulkan feature
// backend is compiled in; otherwise a no-op.
void register_vulkan_feature_backends();

// Backend-agnostic construction used by the SfM frontend so that builds without
// PHOTARA_HAS_VULKAN_FEATURES do not reference the class symbols. Both return
// nullptr (and vulkan_feature_backend_available() is false) in such builds.
[[nodiscard]] std::unique_ptr<FeatureExtractor> make_vulkan_sift_extractor(
    const SiftVulkanOptions& options);
[[nodiscard]] std::unique_ptr<FeatureMatcher> make_vulkan_mutual_ratio_matcher(
    const VulkanMutualRatioMatcherOptions& options);

// True when the Vulkan feature compute device initialized successfully.
// Always false in builds without PHOTARA_HAS_VULKAN_FEATURES.
[[nodiscard]] bool vulkan_feature_backend_available();

}  // namespace photara::features
