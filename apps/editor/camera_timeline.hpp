#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace editor::camera_timeline {

inline float interpolation_weight(float u, int method) {
    u = std::clamp(u, 0.F, 1.F);
    if (method == 2) return u < 1.F ? 0.F : 1.F;
    return method == 1 ? u : u * u * (3.F - 2.F * u);
}

inline float span(float last_time) {
    return std::max(6.F, std::ceil(std::max(0.F, last_time) * 1.15F));
}

// A 1/2/5 progression keeps labels comfortably spaced at any panel width.
inline float tick_step(float seconds_per_tick) {
    const float magnitude = std::pow(10.F, std::floor(std::log10(std::max(0.001F, seconds_per_tick))));
    const float scaled = seconds_per_tick / magnitude;
    return (scaled <= 1.F ? 1.F : scaled <= 2.F ? 2.F : scaled <= 5.F ? 5.F : 10.F) * magnitude;
}

template <typename Key>
float bounded_time(const std::vector<Key>& keys, std::size_t index, float proposed) {
    if (index >= keys.size() || !std::isfinite(proposed))
        return index < keys.size() ? keys[index].time : 0.F;
    const float lower = index ? keys[index - 1].time + 0.01F : 0.F;
    const float upper = index + 1 < keys.size() ? keys[index + 1].time - 0.01F : 36000.F;
    return lower <= upper ? std::clamp(proposed, lower, upper) : keys[index].time;
}

template <typename Key>
std::size_t previous(const std::vector<Key>& keys, float time) {
    const auto it = std::lower_bound(keys.begin(), keys.end(), time - 0.001F,
        [](const Key& key, float t) { return key.time < t; });
    return it == keys.begin() ? 0 : static_cast<std::size_t>((it - keys.begin()) - 1);
}

template <typename Key>
std::size_t next(const std::vector<Key>& keys, float time) {
    if (keys.empty()) return 0;
    const auto it = std::upper_bound(keys.begin(), keys.end(), time + 0.001F,
        [](float t, const Key& key) { return t < key.time; });
    return it == keys.end() ? keys.size() - 1 : static_cast<std::size_t>(it - keys.begin());
}

} // namespace editor::camera_timeline
