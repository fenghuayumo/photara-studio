#pragma once

#include "features/extractor.hpp"
#include "features/matcher.hpp"
#include "features/registry.hpp"
#include "features/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace aetherscan::features {

// ---- VLFeat CPU SIFT ------------------------------------------------------------

struct SiftOptions {
    std::size_t maximum_features{8192};
    std::size_t octave_layers{3};
    // VLFeat peak = 255 * contrast / octave_layers (openMVG mapping).
    double contrast_threshold{0.01};
    double edge_threshold{10.0};
    double sigma{1.6};
    bool root_sift{true};
};

class SiftExtractor final : public FeatureExtractor {
public:
    explicit SiftExtractor(SiftOptions options = {});
    ~SiftExtractor() override;
    SiftExtractor(SiftExtractor&&) noexcept;
    SiftExtractor& operator=(SiftExtractor&&) noexcept;
    SiftExtractor(const SiftExtractor&) = delete;
    SiftExtractor& operator=(const SiftExtractor&) = delete;

    [[nodiscard]] std::string_view name() const override { return "sift"; }
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

// ---- Optional SiftGPU -----------------------------------------------------------

struct SiftGpuOptions {
    std::size_t maximum_features{8192};
    std::uint32_t maximum_image_dimension{8192};
    float peak_threshold{0.005F};
    float edge_threshold{20.0F};
    std::uint32_t maximum_orientations{2};
    int device_index{0};
    bool root_sift{true};
};

// Optional adapter around Changchang Wu's SiftGPU. Upstream license is
// non-commercial; never enabled by default. Context/thread-affine.
class SiftGpuExtractor final : public FeatureExtractor {
public:
    explicit SiftGpuExtractor(SiftGpuOptions options = {});
    ~SiftGpuExtractor() override;
    SiftGpuExtractor(SiftGpuExtractor&&) noexcept;
    SiftGpuExtractor& operator=(SiftGpuExtractor&&) noexcept;
    SiftGpuExtractor(const SiftGpuExtractor&) = delete;
    SiftGpuExtractor& operator=(const SiftGpuExtractor&) = delete;

    [[nodiscard]] static bool is_built() noexcept;
    [[nodiscard]] bool is_available() const noexcept;

    [[nodiscard]] std::string_view name() const override { return "siftgpu"; }
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

// ---- SuperPoint (extension point; ONNX wiring comes later) ----------------------

struct SuperPointOptions {
    std::filesystem::path model_path;
    std::size_t maximum_features{2048};
    float keypoint_threshold{0.005F};
    bool remove_borders{true};
    std::uint32_t input_width{1024};
    std::uint32_t input_height{1024};
    bool cuda{true};
    bool allow_cpu_fallback{true};
};

// Registered as "superpoint". Currently a stub so product code can already
// select the backend by name; throws until an ONNX/TensorRT implementation lands.
class SuperPointExtractor final : public FeatureExtractor {
public:
    explicit SuperPointExtractor(SuperPointOptions options = {});
    ~SuperPointExtractor() override;
    SuperPointExtractor(SuperPointExtractor&&) noexcept;
    SuperPointExtractor& operator=(SuperPointExtractor&&) noexcept;
    SuperPointExtractor(const SuperPointExtractor&) = delete;
    SuperPointExtractor& operator=(const SuperPointExtractor&) = delete;

    [[nodiscard]] static bool is_implemented() noexcept;

    [[nodiscard]] std::string_view name() const override { return "superpoint"; }
    [[nodiscard]] ExtractorInfo info() const override;
    [[nodiscard]] std::unique_ptr<FeatureExtractor> clone() const override;

    [[nodiscard]] FeatureSet extract_gray(
        std::span<const std::uint8_t> pixels,
        std::uint32_t width,
        std::uint32_t height,
        std::size_t row_stride = 0) const override;

private:
    SuperPointOptions options_;
};

// ---- Descriptor matchers --------------------------------------------------------

struct DescriptorMatcherOptions {
    float ratio_threshold{0.8F};
    bool mutual_check{true};
    // When true, large single-pair matches may use OpenMP. Automatically disabled
    // while an outer AetherScan task pool is already saturating the machine.
    bool parallel{true};
};

class MutualRatioMatcher final : public FeatureMatcher {
public:
    explicit MutualRatioMatcher(DescriptorMatcherOptions options = {});

    [[nodiscard]] std::string_view name() const override { return "mutual_ratio"; }
    [[nodiscard]] std::unique_ptr<FeatureMatcher> clone() const override;
    [[nodiscard]] MatchSet match(
        const FeatureSet& query, const FeatureSet& train) const override;

    [[nodiscard]] const DescriptorMatcherOptions& options() const { return options_; }

private:
    DescriptorMatcherOptions options_;
};

// Backward-compatible free function → MutualRatioMatcher.
MatchSet match_descriptors(
    const FeatureSet& query,
    const FeatureSet& train,
    const DescriptorMatcherOptions& options = {});

// ---- Fused pair pipelines -------------------------------------------------------

enum class LightGlueExtractor { disk, superpoint };
enum class InferenceDevice { cpu, cuda };

struct LightGlueOptions {
    std::filesystem::path model_path;
    LightGlueExtractor extractor{LightGlueExtractor::disk};
    InferenceDevice device{InferenceDevice::cuda};
    bool allow_cpu_fallback{true};
    std::uint32_t input_width{1024};
    std::uint32_t input_height{1024};
};

class LightGluePipeline final : public PairFeaturePipeline {
public:
    explicit LightGluePipeline(LightGlueOptions options);
    ~LightGluePipeline() override;
    LightGluePipeline(LightGluePipeline&&) noexcept;
    LightGluePipeline& operator=(LightGluePipeline&&) noexcept;
    LightGluePipeline(const LightGluePipeline&) = delete;
    LightGluePipeline& operator=(const LightGluePipeline&) = delete;

    [[nodiscard]] std::string_view name() const override { return "lightglue"; }
    ImagePairFeatures match_files(
        const std::filesystem::path& first,
        const std::filesystem::path& second) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aetherscan::features
