#pragma once

#include "features/types.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace aetherscan::features {

// Canonical fused image-pair recipe (not an extract×match composition).
inline constexpr std::string_view kLightGlueEnd2EndPipeline = "lightglue_end2end";

[[nodiscard]] inline bool is_pair_pipeline_name(std::string_view name) noexcept {
    return name == kLightGlueEnd2EndPipeline;
}

// Descriptor-space compatibility for the composable extract×match path.
// Fused pair pipelines are excluded from this table.
[[nodiscard]] inline bool matcher_accepts_metric(
    std::string_view matcher, DescriptorMetric metric) noexcept {
    if (matcher == "gpu_mutual_ratio" || matcher == "siftgpu" ||
        matcher == "mutual_ratio")
        return metric == DescriptorMetric::l2 ||
               metric == DescriptorMetric::l2_root;
    // Future descriptor LightGlue / SuperGlue matchers go here.
    (void)metric;
    return false;
}

[[nodiscard]] inline bool extractor_matcher_compatible(
    std::string_view extractor, std::string_view matcher) {
    if (matcher == "gpu_mutual_ratio" || matcher == "siftgpu" ||
        matcher == "mutual_ratio")
        return extractor == "siftgpu" || extractor == "sift";
    // Unknown matcher: allow registry create to decide later.
    (void)extractor;
    return true;
}

inline void validate_extractor_matcher_combo(
    std::string_view extractor, std::string_view matcher) {
    if (matcher == "lightglue")
        throw std::invalid_argument(
            "matcher 'lightglue' is a fused pair recipe; use "
            "--pipeline lightglue_end2end (or the legacy --matcher lightglue "
            "alias)");
    if (!extractor_matcher_compatible(extractor, matcher))
        throw std::invalid_argument(
            "Incompatible feature combo: extractor='" + std::string(extractor) +
            "' matcher='" + std::string(matcher) +
            "'. Use matching descriptor families, or --pipeline "
            "lightglue_end2end for fused LightGlue.");
}

}  // namespace aetherscan::features
