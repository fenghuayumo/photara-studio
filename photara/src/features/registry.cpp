#include "features/registry.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace photara::features {
namespace {

struct Registries {
    std::mutex mutex;
    std::unordered_map<std::string, ExtractorFactory> extractors;
    std::unordered_map<std::string, MatcherFactory> matchers;
    std::unordered_map<std::string, PairPipelineFactory> pair_pipelines;
};

Registries& registries() {
    static Registries instance;
    return instance;
}

template <class Map>
std::vector<std::string> keys_of(const Map& map) {
    std::vector<std::string> names;
    names.reserve(map.size());
    for (const auto& [name, factory] : map) {
        (void)factory;
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

}  // namespace

void register_extractor(std::string name, ExtractorFactory factory) {
    if (name.empty() || !factory)
        throw std::invalid_argument("Extractor registration requires a non-empty name and factory");
    auto& reg = registries();
    std::lock_guard lock(reg.mutex);
    reg.extractors.insert_or_assign(std::move(name), std::move(factory));
}

void register_matcher(std::string name, MatcherFactory factory) {
    if (name.empty() || !factory)
        throw std::invalid_argument("Matcher registration requires a non-empty name and factory");
    auto& reg = registries();
    std::lock_guard lock(reg.mutex);
    reg.matchers.insert_or_assign(std::move(name), std::move(factory));
}

void register_pair_pipeline(std::string name, PairPipelineFactory factory) {
    if (name.empty() || !factory)
        throw std::invalid_argument(
            "Pair pipeline registration requires a non-empty name and factory");
    auto& reg = registries();
    std::lock_guard lock(reg.mutex);
    reg.pair_pipelines.insert_or_assign(std::move(name), std::move(factory));
}

std::unique_ptr<FeatureExtractor> create_extractor(const std::string_view name) {
    ensure_builtin_feature_backends();
    auto& reg = registries();
    std::lock_guard lock(reg.mutex);
    const auto found = reg.extractors.find(std::string(name));
    if (found == reg.extractors.end())
        throw std::invalid_argument("Unknown feature extractor: " + std::string(name));
    return found->second();
}

std::unique_ptr<FeatureMatcher> create_matcher(const std::string_view name) {
    ensure_builtin_feature_backends();
    auto& reg = registries();
    std::lock_guard lock(reg.mutex);
    const auto found = reg.matchers.find(std::string(name));
    if (found == reg.matchers.end())
        throw std::invalid_argument("Unknown feature matcher: " + std::string(name));
    return found->second();
}

std::unique_ptr<PairFeaturePipeline> create_pair_pipeline(const std::string_view name) {
    ensure_builtin_feature_backends();
    auto& reg = registries();
    std::lock_guard lock(reg.mutex);
    const auto found = reg.pair_pipelines.find(std::string(name));
    if (found == reg.pair_pipelines.end())
        throw std::invalid_argument("Unknown pair feature pipeline: " + std::string(name));
    return found->second();
}

std::vector<std::string> list_extractors() {
    ensure_builtin_feature_backends();
    auto& reg = registries();
    std::lock_guard lock(reg.mutex);
    return keys_of(reg.extractors);
}

std::vector<std::string> list_matchers() {
    ensure_builtin_feature_backends();
    auto& reg = registries();
    std::lock_guard lock(reg.mutex);
    return keys_of(reg.matchers);
}

std::vector<std::string> list_pair_pipelines() {
    ensure_builtin_feature_backends();
    auto& reg = registries();
    std::lock_guard lock(reg.mutex);
    return keys_of(reg.pair_pipelines);
}

bool has_extractor(const std::string_view name) {
    ensure_builtin_feature_backends();
    auto& reg = registries();
    std::lock_guard lock(reg.mutex);
    return reg.extractors.contains(std::string(name));
}

bool has_matcher(const std::string_view name) {
    ensure_builtin_feature_backends();
    auto& reg = registries();
    std::lock_guard lock(reg.mutex);
    return reg.matchers.contains(std::string(name));
}

bool has_pair_pipeline(const std::string_view name) {
    ensure_builtin_feature_backends();
    auto& reg = registries();
    std::lock_guard lock(reg.mutex);
    return reg.pair_pipelines.contains(std::string(name));
}

void ensure_builtin_feature_backends() {
    static std::once_flag once;
    std::call_once(once, [] {
        register_sift_feature_backends();
        register_siftgpu_feature_backends();
        register_superpoint_feature_backends();
        register_disk_feature_backends();
        register_aliked_feature_backends();
        register_lightglue_feature_backends();
        register_lightglue_matcher_backend();
    });
}

}  // namespace photara::features
