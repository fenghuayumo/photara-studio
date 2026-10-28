#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace editor::camera_recording {

struct Quality {
    float position{.001F};
    float angle{.00610865F}; // 0.35 degrees
    float lens{.15F};
};
inline Quality quality_for(int preset, float distance) {
    const float scale = std::max(.01F, distance);
    if (preset == 0) return {scale * .00035F, .00261799F, .05F};
    if (preset == 2) return {scale * .003F, .01396263F, .3F};
    return {scale * .001F, .00610865F, .15F};
}

using V3 = std::array<float, 3>;
inline float dot(V3 a, V3 b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
inline V3 cross(V3 a, V3 b) { return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; }
template <typename Key> V3 forward(const Key& p) {
    return {-std::cos(p.pitch)*std::sin(p.yaw), std::sin(p.pitch), -std::cos(p.pitch)*std::cos(p.yaw)};
}
template <typename Key> V3 right(const Key& p) {
    const V3 level{-std::cos(p.yaw), 0.F, std::sin(p.yaw)};
    const auto up = cross(level, forward(p));
    return {level[0]*std::cos(p.roll)+up[0]*std::sin(p.roll),
            level[1]*std::cos(p.roll)+up[1]*std::sin(p.roll),
            level[2]*std::cos(p.roll)+up[2]*std::sin(p.roll)};
}
inline float angular_distance(V3 a, V3 b) {
    // Chord length avoids acos precision loss for tiny motions.
    float d = 0.F;
    for (int j = 0; j < 3; ++j) d += (a[j]-b[j])*(a[j]-b[j]);
    return 2.F * std::asin(std::min(1.F, std::sqrt(d) * .5F));
}
template <typename Key> Key interpolate(const Key& a, const Key& b, float u) {
    // Interpolation is a pose operation. Never copy thumbnails or other payloads.
    Key p{};
    p.time=a.time+(b.time-a.time)*u;
    p.projection=u>=1.F ? b.projection : a.projection;
    p.interpolation=a.interpolation;
    const auto mix = [u](float x, float y) { return x + (y-x)*u; };
    constexpr float tau = 6.28318530718F;
    for (int j = 0; j < 3; ++j) p.target[j] = mix(a.target[j], b.target[j]);
    p.yaw = a.yaw + std::remainder(b.yaw-a.yaw, tau)*u;
    p.roll = a.roll + std::remainder(b.roll-a.roll, tau)*u;
    p.pitch = mix(a.pitch, b.pitch); p.distance = mix(a.distance, b.distance);
    p.fov_degrees = mix(a.fov_degrees, b.fov_degrees);
    p.ortho_height = mix(a.ortho_height, b.ortho_height);
    p.fisheye_k1 = mix(a.fisheye_k1, b.fisheye_k1);
    return p;
}
template <typename Key> float error(const Key& a, const Key& b, Quality q) {
    if (a.projection != b.projection) return 1e6F;
    const auto af = forward(a), bf = forward(b);
    float position = 0.F;
    for (int j = 0; j < 3; ++j) {
        const float delta = (a.target[j]-af[j]*a.distance) - (b.target[j]-bf[j]*b.distance);
        position += delta*delta;
    }
    float e = std::sqrt(position)/q.position;
    e = std::max(e, angular_distance(af, bf)/q.angle);
    e = std::max(e, angular_distance(right(a), right(b))/q.angle);
    e = std::max(e, std::abs(a.fov_degrees-b.fov_degrees)/q.lens);
    e = std::max(e, std::abs(a.ortho_height-b.ortho_height)/q.position);
    return std::max(e, std::abs(a.fisheye_k1-b.fisheye_k1)/.002F);
}

template <typename Key> std::vector<Key> reduce(const std::vector<Key>& samples, Quality q) {
    if (samples.size() < 2) return {};
    std::vector<bool> keep(samples.size(), false);
    keep.front() = keep.back() = true;
    for (std::size_t i = 1; i < samples.size(); ++i)
        if (samples[i].projection != samples[i-1].projection) keep[i-1] = keep[i] = true;
    std::vector<std::pair<std::size_t, std::size_t>> pending;
    std::size_t first = 0;
    for (std::size_t i = 1; i < samples.size(); ++i) if (keep[i]) {
        pending.emplace_back(first, i); first = i;
    }
    while (!pending.empty()) {
        const auto [a,b] = pending.back(); pending.pop_back();
        float worst = 1.F; std::size_t split = 0;
        for (std::size_t i = a+1; i < b; ++i) {
            const float u = (samples[i].time-samples[a].time)/(samples[b].time-samples[a].time);
            const float e = error(samples[i], interpolate(samples[a], samples[b], u), q);
            if (e > worst) { worst = e; split = i; }
        }
        if (split) {
            keep[split] = true;
            pending.emplace_back(a, split); pending.emplace_back(split, b);
        }
    }
    std::vector<Key> keys;
    for (std::size_t i = 0; i < samples.size(); ++i) if (keep[i]) {
        keys.push_back(samples[i]); keys.back().interpolation = 1;
    }
    return keys;
}

// Observations run each UI frame. Only meaningful camera changes allocate
// samples; an idle span is represented by one boundary when motion resumes.
template <typename Key> struct Recorder {
    Quality quality;
    bool preserve_pauses{};
    float last_observed{}, last_motion{}, path_time{};
    Key anchor;
    std::vector<Key> samples;
    void begin(Key pose, Quality q, bool preserve) {
        *this = {}; quality = q; preserve_pauses = preserve; anchor = pose; anchor.time = 0.F;
    }
    void observe(Key pose, float elapsed, bool final = false) {
        if (!std::isfinite(elapsed) || elapsed < last_observed) return;
        const float frame_dt = std::clamp(elapsed-last_observed, .01F, .1F);
        last_observed = elapsed;
        if (error(anchor, pose, quality) < .05F || elapsed-last_motion < (final ? .01F : 1.F/30.F)) return;
        const float gap = elapsed-last_motion;
        if (samples.empty()) {
            samples.push_back(anchor); // Trim the preparation before the first move.
            path_time = gap > .25F ? frame_dt : gap;
        } else if (gap > .25F) {
            if (preserve_pauses) {
                Key hold = anchor; hold.time = path_time + gap-frame_dt;
                samples.push_back(hold); path_time += gap;
            } else path_time += frame_dt;
        } else path_time += gap;
        pose.time = path_time; pose.interpolation = 1;
        samples.push_back(pose); anchor = pose; last_motion = elapsed;
    }
    std::vector<Key> finish(Key pose, float elapsed) {
        observe(pose, elapsed, true);
        if (samples.empty()) return {};
        if (preserve_pauses && elapsed-last_motion > .25F) {
            Key hold = anchor; hold.time = path_time + elapsed-last_motion;
            samples.push_back(hold);
        }
        return reduce(samples, quality);
    }
};
} // namespace editor::camera_recording
