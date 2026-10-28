#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <utility>

namespace editor::camera_timeline {

inline float interpolation_weight(float u, int method) {
    u = std::clamp(u, 0.F, 1.F);
    if (method == 2) return u < 1.F ? 0.F : 1.F;
    return method == 1 ? u : u * u * (3.F - 2.F * u);
}

// Append one revolution about a fixed target. Small yaw increments preserve
// direction when playback takes the shortest angular route between keys.
template <typename Key>
std::size_t append_orbit(std::vector<Key>& keys, const Key& pose, float duration, bool clockwise) {
    const float start = keys.empty() ? 0.F : keys.back().time + 2.F;
    if (!std::isfinite(duration) || duration < .1F || start + duration > 36000.F) return keys.size();
    const std::size_t first = keys.size();
    for (int i = 0; i <= 8; ++i) {
        Key key = pose;
        key.time = start + duration * static_cast<float>(i) / 8.F;
        key.yaw = pose.yaw + (clockwise ? -1.F : 1.F) * 6.28318530718F * static_cast<float>(i) / 8.F;
        key.interpolation = 1;
        key.thumbnail_rgb.clear(); key.thumbnail_width = key.thumbnail_height = 0;
        keys.push_back(std::move(key));
    }
    return first;
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
