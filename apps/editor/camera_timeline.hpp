#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <utility>
#include "camera_path_recording.hpp"

namespace editor::camera_timeline {

inline float interpolation_weight(float u, int method) {
    u = std::clamp(u, 0.F, 1.F);
    if (method == 2) return u < 1.F ? 0.F : 1.F;
    return method == 1 ? u : u * u * (3.F - 2.F * u);
}

template <typename Keys> auto evaluate(const Keys& keys, float time) -> typename Keys::value_type {
    if (keys.empty()) return {};
    if (time<=keys.front().time) return camera_recording::interpolate(keys.front(),keys.front(),0.F);
    if (time>=keys.back().time) return camera_recording::interpolate(keys.back(),keys.back(),0.F);
    const auto right=std::upper_bound(keys.begin(),keys.end(),time,[](float t,const auto& key){return t<key.time;});
    const auto& b=*right; const auto& a=*(right-1);
    return camera_recording::interpolate(a,b,interpolation_weight((time-a.time)/(b.time-a.time),a.interpolation));
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

struct View {
    float start{},duration{6.F};
    bool fit{true};
    void constrain(float total) {
        total=std::max(.1F,total);
        if (fit) {start=0.F;duration=total;return;}
        duration=std::clamp(duration,.1F,total);
        start=std::clamp(start,0.F,total-duration);
    }
    void zoom(float total,float factor,float anchor) {
        constrain(total);
        if (!std::isfinite(factor) || factor<=0.F) return;
        anchor=std::clamp(anchor,0.F,1.F);
        const float time=start+duration*anchor;
        duration=std::clamp(duration/factor,.1F,total);
        start=time-duration*anchor; fit=duration>=total;
        constrain(total);
    }
    void pan(float total,float seconds) {
        constrain(total);start+=seconds;constrain(total);
    }
    void reveal(float total,float time) {
        constrain(total);
        if (time<start) start=time-duration*.2F;
        else if (time>start+duration) start=time-duration*.8F;
        constrain(total);
    }
    float fraction(float time) const {return (time-start)/duration;}
    float time_at(float fraction) const {return start+std::clamp(fraction,0.F,1.F)*duration;}
};
inline float jump_time(float current,float delta,float first,float last) {
    return std::clamp(current+delta,first,last);
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
