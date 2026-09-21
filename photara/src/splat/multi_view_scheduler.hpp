#pragma once

#include "cuda_ops.hpp"
#include "splat/options.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace photara::splat::detail {

struct GeometryStabilitySample {
    std::size_t gaussian_count{};
    std::size_t grown{};
    std::size_t pruned{};
    float depth_consistency{};
    GeometryDistributionSummary distribution;
};

struct GeometryStabilityDecision {
    bool reference_ready{};
    bool stable{};
    bool reduced{};
    bool recovered{};
    unsigned interval{1};
    unsigned stable_refinements{};
    float count_delta{};
    float churn{};
    float depth_delta{};
    float distribution_delta{};
};

class MultiViewStabilityScheduler {
public:
    explicit MultiViewStabilityScheduler(const TrainingOptions& options)
        : maximum_interval_(std::max(
              1U, options.multi_view_adaptive_max_interval)),
          required_stable_refinements_(std::max(
              1U, options.multi_view_adaptive_stable_refinements)),
          count_threshold_(options.multi_view_adaptive_count_threshold),
          churn_threshold_(options.multi_view_adaptive_churn_threshold),
          depth_threshold_(options.multi_view_adaptive_depth_threshold),
          minimum_depth_consistency_(
              options.multi_view_adaptive_min_depth_consistency),
          distribution_threshold_(
              options.multi_view_adaptive_distribution_threshold) {}

    [[nodiscard]] unsigned interval() const noexcept { return interval_; }

    GeometryStabilityDecision invalidate() noexcept {
        GeometryStabilityDecision decision;
        decision.recovered = interval_ != 1U;
        interval_ = 1U;
        stable_refinements_ = 0;
        has_previous_ = false;
        decision.interval = interval_;
        return decision;
    }

    GeometryStabilityDecision update(const GeometryStabilitySample& sample) {
        GeometryStabilityDecision decision;
        decision.interval = interval_;
        if (!has_previous_) {
            previous_ = sample;
            has_previous_ = true;
            return decision;
        }

        decision.reference_ready = true;
        const float previous_count = static_cast<float>(
            std::max<std::size_t>(previous_.gaussian_count, 1));
        decision.count_delta = std::abs(
            static_cast<float>(sample.gaussian_count) - previous_count) /
            previous_count;
        decision.churn = static_cast<float>(sample.grown + sample.pruned) /
            previous_count;
        decision.depth_delta = std::abs(
            sample.depth_consistency - previous_.depth_consistency);
        decision.distribution_delta = distribution_delta(
            sample.distribution, previous_.distribution);
        decision.stable =
            std::isfinite(sample.depth_consistency) &&
            sample.depth_consistency >= minimum_depth_consistency_ &&
            decision.count_delta <= count_threshold_ &&
            decision.churn <= churn_threshold_ &&
            decision.depth_delta <= depth_threshold_ &&
            decision.distribution_delta <= distribution_threshold_;

        if (decision.stable) {
            ++stable_refinements_;
            if (stable_refinements_ >= required_stable_refinements_ &&
                interval_ < maximum_interval_) {
                interval_ = std::min(maximum_interval_, interval_ * 2U);
                stable_refinements_ = 0;
                decision.reduced = true;
            }
        } else {
            stable_refinements_ = 0;
            if (interval_ != 1U) {
                interval_ = 1U;
                decision.recovered = true;
            }
        }
        previous_ = sample;
        decision.interval = interval_;
        decision.stable_refinements = stable_refinements_;
        return decision;
    }

private:
    [[nodiscard]] static float relative_delta(
        const float current, const float previous,
        const float floor) noexcept {
        return std::abs(current - previous) /
            std::max(std::abs(previous), floor);
    }

    [[nodiscard]] static float distribution_delta(
        const GeometryDistributionSummary& current,
        const GeometryDistributionSummary& previous) noexcept {
        return std::max({
            relative_delta(
                current.opacity_mean, previous.opacity_mean, 0.1F),
            relative_delta(
                current.opacity_stddev, previous.opacity_stddev, 0.05F),
            std::abs(current.log_scale_mean - previous.log_scale_mean),
            relative_delta(
                current.log_scale_stddev,
                previous.log_scale_stddev, 0.1F),
            relative_delta(
                current.log_anisotropy_mean,
                previous.log_anisotropy_mean, 0.1F),
            relative_delta(
                current.log_anisotropy_stddev,
                previous.log_anisotropy_stddev, 0.1F)});
    }

    unsigned maximum_interval_{1};
    unsigned required_stable_refinements_{1};
    float count_threshold_{};
    float churn_threshold_{};
    float depth_threshold_{};
    float minimum_depth_consistency_{};
    float distribution_threshold_{};
    unsigned interval_{1};
    unsigned stable_refinements_{};
    bool has_previous_{};
    GeometryStabilitySample previous_;
};

}  // namespace photara::splat::detail
