#pragma once

#include "features/extractor.hpp"
#include "features/matcher.hpp"

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace aetherscan::features {

using ExtractorFactory = std::function<std::unique_ptr<FeatureExtractor>()>;
using MatcherFactory = std::function<std::unique_ptr<FeatureMatcher>()>;
using PairPipelineFactory = std::function<std::unique_ptr<PairFeaturePipeline>()>;

void register_extractor(std::string name, ExtractorFactory factory);
void register_matcher(std::string name, MatcherFactory factory);
void register_pair_pipeline(std::string name, PairPipelineFactory factory);

[[nodiscard]] std::unique_ptr<FeatureExtractor> create_extractor(std::string_view name);
[[nodiscard]] std::unique_ptr<FeatureMatcher> create_matcher(std::string_view name);
[[nodiscard]] std::unique_ptr<PairFeaturePipeline> create_pair_pipeline(std::string_view name);

[[nodiscard]] std::vector<std::string> list_extractors();
[[nodiscard]] std::vector<std::string> list_matchers();
[[nodiscard]] std::vector<std::string> list_pair_pipelines();

[[nodiscard]] bool has_extractor(std::string_view name);
[[nodiscard]] bool has_matcher(std::string_view name);
[[nodiscard]] bool has_pair_pipeline(std::string_view name);

// Ensures built-in backends are registered (idempotent). Call from tools/tests
// before create_* if you do not link a TU that already triggers registration.
void ensure_builtin_feature_backends();

// Called by ensure_builtin_feature_backends(); also safe to invoke directly.
void register_sift_feature_backends();
void register_siftgpu_feature_backends();
void register_superpoint_feature_backends();
void register_lightglue_feature_backends();

}  // namespace aetherscan::features
