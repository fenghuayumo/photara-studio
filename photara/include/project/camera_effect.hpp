#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace photara::project {

// One reveal clip on the camera path's absolute time axis.
struct CameraEffectTrack {
    bool enabled{};
    float start_time{2.F}, rings_time{4.F}, end_time{6.F};
    float point_size{2.F}, ring_scale{2.828427F};
    int easing{1}; // 0 linear, 1 smoothstep.
    int kind{}; // 0: points/rings/splat crossfade, 1: view-triggered reconstruction.
    int scan_axis{3}; // Legacy plane fields retained for older project readers.
    bool scan_reverse{};
    float scan_softness{.025F}, scan_glow{.7F};
    float scan_direction[3]{1.F, 0.F, 0.F};
    float scan_hold{1.F}, scan_growth{2.5F};
    float scan_range{.32F}, scan_density{.22F};

    void validate() const {
        if (!std::isfinite(scan_hold) || scan_hold < 0.F || scan_hold > 10.F ||
            !std::isfinite(scan_growth) || scan_growth < .1F || scan_growth > 10.F ||
            !std::isfinite(scan_range) || scan_range < .02F || scan_range > 2.F ||
            !std::isfinite(scan_density) || scan_density < .01F || scan_density > 1.F ||
            !std::isfinite(start_time) || !std::isfinite(rings_time) || !std::isfinite(end_time) ||
            start_time < 0.F || rings_time - start_time < .01F ||
            end_time - rings_time < .01F || end_time > 36000.F ||
            !std::isfinite(point_size) || point_size < 1.F || point_size > 6.F ||
            !std::isfinite(ring_scale) || ring_scale < .5F || ring_scale > 8.F ||
            easing < 0 || easing > 1 || kind < 0 || kind > 1 || scan_axis < 0 || scan_axis > 3 ||
            !std::isfinite(scan_softness) || scan_softness < .001F || scan_softness > .2F ||
            !std::isfinite(scan_glow) || scan_glow < 0.F || scan_glow > 1.F ||
            !std::isfinite(scan_direction[0]) || !std::isfinite(scan_direction[1]) || !std::isfinite(scan_direction[2]) ||
            !std::isfinite(scan_direction[0]*scan_direction[0] + scan_direction[1]*scan_direction[1] + scan_direction[2]*scan_direction[2]) ||
            scan_direction[0]*scan_direction[0] + scan_direction[1]*scan_direction[1] + scan_direction[2]*scan_direction[2] < 1e-8F)
            throw std::invalid_argument("Invalid camera effect track");
    }
};

struct CameraEffectWeights { float points{}, rings{}, splat{1.F}; };

inline CameraEffectWeights camera_effect_weights(const CameraEffectTrack& effect, float time) {
    if (!effect.enabled) return {};
    effect.validate();
    const auto weight = [&](float value) {
        value = std::clamp(value, 0.F, 1.F);
        return effect.easing ? value * value * (3.F - 2.F * value) : value;
    };
    if (time <= effect.start_time) return {1.F, 0.F, 0.F};
    if (time >= effect.end_time) return {0.F, 0.F, 1.F};
    if (time < effect.rings_time) {
        const float u = weight((time - effect.start_time) / (effect.rings_time - effect.start_time));
        return {1.F - u, u, 0.F};
    }
    const float u = weight((time - effect.rings_time) / (effect.end_time - effect.rings_time));
    return {0.F, 1.F - u, u};
}

inline CameraEffectTrack camera_effect_demo(float first, float last) {
    CameraEffectTrack effect;
    effect.enabled = true;
    const float duration = std::max(.08F, last - first);
    effect.start_time = first + duration * .25F;
    effect.rings_time = first + duration * .5F;
    effect.end_time = first + duration * .75F;
    return effect;
}

inline CameraEffectTrack camera_scan_demo(float first,float last) {
    auto effect=camera_effect_demo(first,last);
    effect.kind=1;
    const float duration=std::max(.08F,last-first);
    effect.start_time=first+std::min(.7F,duration*.1F);
    effect.end_time=first+duration*.8F;
    effect.rings_time=(effect.start_time+effect.end_time)*.5F;
    return effect;
}

} // namespace photara::project
