#pragma once

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace photara::splat {

enum class TrainingMode { fast, standard, high_quality };

struct TrainingPreset {
    unsigned iterations;
    std::size_t gaussian_cap;
    bool speedy_pruning;
    unsigned multi_view_interval;
};

[[nodiscard]] constexpr TrainingPreset training_preset(TrainingMode mode) {
    switch (mode) {
    case TrainingMode::fast: return {30'000, 1'000'000, true, 2};
    case TrainingMode::standard: return {30'000, 3'000'000, false, 1};
    case TrainingMode::high_quality: return {50'000, 5'000'000, false, 1};
    }
    return {30'000, 3'000'000, false, 1};
}

[[nodiscard]] inline TrainingMode parse_training_mode(std::string_view name) {
    if (name == "fast") return TrainingMode::fast;
    if (name == "standard") return TrainingMode::standard;
    if (name == "high-quality") return TrainingMode::high_quality;
    throw std::invalid_argument("Splat mode must be fast, standard, or high-quality");
}

}  // namespace photara::splat
