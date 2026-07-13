#pragma once

#include "features/extractor.hpp"
#include "features/matcher.hpp"
#include "features/registry.hpp"
#include "features/types.hpp"
#include "io/image.hpp"

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
    std::size_t maximum_features{27000};
    std::size_t octave_layers{3};
    int first_octave{-1};  // Match OpenCV SIFT's doubled-image first octave.
    // VLFeat peak = 255 * contrast / octave_layers (openMVG mapping).
    double contrast_threshold{0.005};
    double edge_threshold{10.0};
    double sigma{1.6};
    std::size_t grid_size{3};
    std::size_t min_features_per_cell{500};
    std::size_t max_features_per_cell{3000};
    std::size_t cell_border{64};
    unsigned adaptive_retries{5};
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
    std::uint32_t maximum_image_dimension{5120};
    float peak_threshold{0.005F};
    float edge_threshold{20.0F};
    std::uint32_t maximum_orientations{2};
    int first_octave{-1};
    std::uint32_t octave_layers{3};
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

struct SiftGpuMatcherOptions {
    float ratio_threshold{0.8F};
    bool mutual_check{true};
    std::size_t maximum_features{32768};
    int device_index{0};
};

// SiftMatchGPU CUDA matcher. The underlying context is thread-affine, so
// callers must submit pairs from the thread that constructs this matcher.
class SiftGpuMatcher final : public FeatureMatcher {
public:
    explicit SiftGpuMatcher(SiftGpuMatcherOptions options = {});
    ~SiftGpuMatcher() override;
    SiftGpuMatcher(SiftGpuMatcher&&) noexcept;
    SiftGpuMatcher& operator=(SiftGpuMatcher&&) noexcept;
    SiftGpuMatcher(const SiftGpuMatcher&) = delete;
    SiftGpuMatcher& operator=(const SiftGpuMatcher&) = delete;

    [[nodiscard]] static bool is_built() noexcept;
    [[nodiscard]] bool is_available() const noexcept;
    [[nodiscard]] std::string_view name() const override {
        return "gpu_mutual_ratio";
    }
    [[nodiscard]] bool requires_owner_thread() const override { return true; }
    [[nodiscard]] std::unique_ptr<FeatureMatcher> clone() const override;
    [[nodiscard]] MatchSet match(
        const FeatureSet& query, const FeatureSet& train) const override;

private:
    class Impl;
    explicit SiftGpuMatcher(std::shared_ptr<Impl> impl);
    std::shared_ptr<Impl> impl_;
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
    bool approximate{true};
    std::size_t ann_min_features{512};
    std::size_t ann_m{16};
    std::size_t ann_ef_construction{100};
    std::size_t ann_ef_search{64};
    // Soft cap on retained HNSW graphs; pinned images are never evicted.
    std::size_t max_cached_indices{64};
    // When true, large single-pair matches may use OpenMP. Automatically disabled
    // while an outer AetherScan task pool is already saturating the machine.
    bool parallel{true};
};

class MutualRatioMatcher final : public FeatureMatcher {
public:
    explicit MutualRatioMatcher(DescriptorMatcherOptions options = {});

    [[nodiscard]] std::string_view name() const override { return "mutual_ratio"; }
    [[nodiscard]] std::unique_ptr<FeatureMatcher> clone() const override;
    void prepare(const FeatureSet& features) override;
    void clear_prepared() override;
    void pin(const FeatureSet& features);
    void unpin(const FeatureSet& features);
    [[nodiscard]] MatchSet match(
        const FeatureSet& query, const FeatureSet& train) const override;

    [[nodiscard]] const DescriptorMatcherOptions& options() const { return options_; }

private:
    class SharedState;
    MutualRatioMatcher(
        DescriptorMatcherOptions options,
        std::shared_ptr<SharedState> shared);

    DescriptorMatcherOptions options_;
    std::shared_ptr<SharedState> shared_;
};

// Backward-compatible free function → MutualRatioMatcher.
MatchSet match_descriptors(
    const FeatureSet& query,
    const FeatureSet& train,
    const DescriptorMatcherOptions& options = {});

// ---- Fused pair pipelines -------------------------------------------------------

enum class LightGlueExtractor { disk, superpoint };
enum class InferenceDevice { cpu, cuda };

// End-to-end image-pair LightGlue (DISK or SuperPoint extractor head + matcher).
// ONNX signature: images[2,C,H,W] → keypoints[2,N,2], matches[M,3], mscores[M].
struct LightGlueOptions {
    std::filesystem::path model_path;
    LightGlueExtractor extractor{LightGlueExtractor::disk};
    InferenceDevice device{InferenceDevice::cuda};
    bool allow_cpu_fallback{true};
    std::uint32_t input_width{1024};
    std::uint32_t input_height{1024};
    // Drop matches with score below this threshold (post-model filter).
    float min_score{0.0F};
};

class LightGluePipeline final : public PairFeaturePipeline {
public:
    explicit LightGluePipeline(LightGlueOptions options);
    ~LightGluePipeline() override;
    LightGluePipeline(LightGluePipeline&&) noexcept;
    LightGluePipeline& operator=(LightGluePipeline&&) noexcept;
    LightGluePipeline(const LightGluePipeline&) = delete;
    LightGluePipeline& operator=(const LightGluePipeline&) = delete;

    [[nodiscard]] static bool is_built() noexcept;
    [[nodiscard]] bool is_available() const noexcept;
    [[nodiscard]] const LightGlueOptions& options() const;

    [[nodiscard]] std::string_view name() const override {
        return "lightglue_end2end";
    }
    // Ort::Session::Run is not safe for concurrent use of one session.
    [[nodiscard]] bool requires_owner_thread() const { return true; }

    ImagePairFeatures match_files(
        const std::filesystem::path& first,
        const std::filesystem::path& second) override;

    // Match already-loaded RGB images (avoids a second disk decode when the
    // caller already probed dimensions).
    [[nodiscard]] ImagePairFeatures match_rgb(
        const io::RgbImage& first, const io::RgbImage& second);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aetherscan::features
