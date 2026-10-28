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
    const float u=(time-a.time)/(b.time-a.time);
    if (a.interpolation != 3)
        return camera_recording::interpolate(a,b,interpolation_weight(u,a.interpolation));
    auto pose=camera_recording::interpolate(a,b,u);
    const auto index=std::size_t(right-keys.begin());
    const float h=b.time-a.time;
    const auto tangent=[](float left,float right,float hl,float hr) {
        if (left*right<=0.F) return 0.F;
        const float w1=2.F*hr+hl,w2=hr+2.F*hl;
        return (w1+w2)/(w1/left+w2/right);
    };
    // Monotone Hermite tangents share the same derivative on both sides of
    // each cinematic key. Shot boundaries have zero velocity; no overshoot.
    const auto curve=[&](auto value,bool angle=false) {
        const auto delta=[&](float x,float y) {return angle ? std::remainder(y-x,6.28318530718F) : y-x;};
        const float av=value(a),d=delta(av,value(b)),s=d/h;
        float m0=0.F,m1=0.F;
        if (index>=2 && keys[index-2].interpolation==3) {
            const auto& p=keys[index-2]; const float hp=a.time-p.time;
            m0=tangent(delta(value(p),av)/hp,s,hp,h);
        }
        if (index+1<keys.size() && b.interpolation==3) {
            const auto& n=keys[index+1]; const float hn=n.time-b.time;
            m1=tangent(s,delta(value(b),value(n))/hn,h,hn);
        }
        return av+(3.F*u*u-2.F*u*u*u)*d+(u*u*u-2.F*u*u+u)*h*m0+(u*u*u-u*u)*h*m1;
    };
    for (int c=0;c<3;++c) pose.target[c]=curve([c](const auto& k){return k.target[c];});
    pose.yaw=curve([](const auto& k){return k.yaw;},true);
    pose.roll=curve([](const auto& k){return k.roll;},true);
    pose.pitch=curve([](const auto& k){return k.pitch;});
    pose.distance=curve([](const auto& k){return k.distance;});
    pose.fov_degrees=curve([](const auto& k){return k.fov_degrees;});
    pose.ortho_height=curve([](const auto& k){return k.ortho_height;});
    pose.fisheye_k1=curve([](const auto& k){return k.fisheye_k1;});
    return pose;
}

// Opt existing paths into continuous motion without changing their cuts.
template <typename Keys> void make_cinematic(Keys& keys) {
    for (std::size_t i=0;i<keys.size();++i)
        if (keys[i].interpolation!=2) keys[i].interpolation=i+1<keys.size() ? 3 : 0;
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

// Demo shots retain the current target and use small angular steps so the
// shared shortest-angle interpolator preserves the requested orbit direction.
template <typename Key>
std::size_t append_demo_shot(std::vector<Key>& keys, const Key& pose, int kind,
    float duration, bool clockwise, float distance_scale) {
    const float start = keys.empty() ? 0.F : keys.back().time + 2.F;
    const auto first = keys.size();
    if (kind < 0 || kind > 2 || !std::isfinite(duration) || duration < 1.F || duration > 600.F ||
        start + duration > 36000.F || !std::isfinite(distance_scale) || distance_scale < .1F || distance_scale > 5.F)
        return first;
    for (int i = 0; i <= 16; ++i) {
        const float u = float(i) / 16.F;
        const float eased = u*u*(3.F-2.F*u);
        Key key = pose;
        key.time = start + duration*u;
        key.projection = 0; // These cinematic shots require perspective depth.
        key.roll = 0.F;
        const float turn = kind == 2 ? 6.28318530718F : kind == 0 ? 1.57079632679F : .5235987756F;
        key.yaw = pose.yaw + (clockwise ? -1.F : 1.F)*turn*eased;
        key.pitch = std::clamp(pose.pitch + (kind == 2 ? .65F*eased : 0.F), -1.35F, 1.35F);
        const float pullback = kind == 1 ? .12F + .88F*eased : kind == 2 ? 1.F + .25F*eased : 1.F - .12F*eased;
        key.distance = std::max(.001F, pose.distance*distance_scale*pullback);
        key.interpolation = i==16 ? 0 : 3; // Continuous velocity inside this shot.
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
