#pragma once

#include "features/types.hpp"

#include <filesystem>
#include <memory>
#include <string_view>

namespace aetherscan::features {

// Matches two independently extracted FeatureSets (SIFT mutual-ratio, future
// SuperGlue-style descriptor matchers, brute-force IP for SuperPoint, etc.).
class FeatureMatcher {
public:
    virtual ~FeatureMatcher() = default;

    [[nodiscard]] virtual std::string_view name() const = 0;
    [[nodiscard]] virtual std::unique_ptr<FeatureMatcher> clone() const = 0;
    // Build reusable per-image search data before pair tasks start.
    virtual void prepare(const FeatureSet&) {}
    // Drop prepared ANN / index state after matching to release peak memory.
    virtual void clear_prepared() {}
    [[nodiscard]] virtual MatchSet match(
        const FeatureSet& query, const FeatureSet& train) const = 0;
};

// Fused image-pair pipelines that own both detection and matching
// (LightGlue, LoFTR, future ROMA, etc.).
class PairFeaturePipeline {
public:
    virtual ~PairFeaturePipeline() = default;

    [[nodiscard]] virtual std::string_view name() const = 0;
    [[nodiscard]] virtual ImagePairFeatures match_files(
        const std::filesystem::path& first,
        const std::filesystem::path& second) = 0;
};

}  // namespace aetherscan::features
