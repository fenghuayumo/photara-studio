#include "densification_igs.hpp"

#include "core/logging.hpp"
#include "densification_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace photara::splat::densification {
namespace {

// Upper median of the entries selected by `positive`, computed on the device:
// the host rule was nth_element at position size/2 over the positive subset,
// which is the same order statistic. Unselected entries are pushed to +inf so
// they cannot enter the prefix, and NaN inputs (never positive, never finite)
// are replaced on the way in.
float positive_upper_median(
    const tinytensor::Tensor& values, const tinytensor::Tensor& positive,
    const std::size_t positive_count) {
    if (positive_count == 0) return 0.F;
    auto ranked = values.masked_fill(
        positive.logical_not(), std::numeric_limits<float>::infinity());
    auto sorted = ranked.sort(0, false);
    const std::size_t middle = positive_count / 2;
    const auto value = sorted.first
        .slice(0, middle, middle + 1)
        .to(tinytensor::Device::CPU)
        .to_vector();
    return value.empty() ? 0.F : value.front();
}

}  // namespace

IgsSelection select_igs_parents(
    const tinytensor::Tensor& replacement_weights,
    const tinytensor::Tensor& oversize_scores,
    const tinytensor::Tensor& growth_weights,
    const std::size_t replacement_slots, const std::size_t desired_growth,
    const std::size_t capacity) {
    IgsSelection result;
    auto chosen = tinytensor::Tensor::zeros_bool(
        {replacement_weights.numel()}, tinytensor::Device::CUDA);
    const auto replacement = gpu_detail::weighted_sample_without_replacement(
        replacement_weights, std::min(replacement_slots, capacity));
    result.replacement = replacement.numel();
    if (result.replacement) chosen.index_fill_(0, replacement, 1.F);
    std::size_t remaining = capacity - result.replacement;
    if (remaining && oversize_scores.is_valid()) {
        const auto eligible = oversize_scores.isfinite()
            .logical_and(oversize_scores.gt(0.F)).logical_and(chosen.logical_not());
        result.oversized = std::min(remaining, eligible.count_nonzero());
        if (result.oversized) {
            // Rank actual severity, never the storage position. Equal scores
            // are equivalent; no later truncation can evict replacements.
            const auto ranked = oversize_scores.masked_fill(
                eligible.logical_not(), -std::numeric_limits<float>::infinity())
                .sort(0, true);
            chosen.index_fill_(0, ranked.second.slice(0, 0, result.oversized)
                .to(tinytensor::DataType::Int32), 1.F);
            remaining -= result.oversized;
        }
    }
    if (remaining && desired_growth && growth_weights.is_valid()) {
        const auto growth = gpu_detail::weighted_sample_without_replacement(
            growth_weights.masked_fill(chosen, 0.F),
            std::min(remaining, desired_growth));
        result.growth = growth.numel();
        if (result.growth) chosen.index_fill_(0, growth, 1.F);
    }
    result.parents = chosen.nonzero().squeeze(1).to(tinytensor::DataType::Int32);
    return result;
}

RefinementCounts IgsStrategy::refine(
    GaussianModel& model, detail::DensificationStats& stats,
    const unsigned iteration, const float scene_extent,
    const mvs::Vec3f& scene_center, const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states) const {
    const std::size_t old_count = model.size();
    if (!is_refinement_iteration(iteration, options) || old_count == 0)
        return {};

    // The host path clips oversize splats before it reads any statistics, so
    // the mask below already sees the clipped scales.
    if (oversize_penalty_at(options, iteration) > 0.F)
        detail::clip_log_scale_by_screen(
            model.log_scales, stats.max_screen_radius,
            options.oversize_screen_limit, options.oversize_clip_hardness);

    auto masks = detail::prune_masks(
        model, options.prune_opacity, 100.F * scene_extent,
        {scene_center.x(), scene_center.y(), scene_center.z()});
    // Rows below the opacity floor are not pruned outright: the host path
    // keeps every non-hard row and spends at most `dense_recycle_fraction` of
    // the model on the least opaque of them.
    const auto recycled_rows = masks.keep.logical_not().logical_and(
        masks.hard.logical_not());
    auto keep = masks.hard.logical_not();

    // Low-opacity recycling: at most `dense_recycle_fraction` of the model,
    // hard-pruned rows included, spent on the least opaque rows.
    const std::size_t hard_count = masks.hard.count_nonzero();
    const std::size_t recycle_limit = static_cast<std::size_t>(std::ceil(
        static_cast<double>(old_count) *
        static_cast<double>(
            std::clamp(options.dense_recycle_fraction, 0.F, 1.F))));
    const std::size_t opacity_budget =
        recycle_limit > hard_count ? recycle_limit - hard_count : 0;
    if (opacity_budget != 0) {
        const std::size_t low_count = recycled_rows.count_nonzero();
        const std::size_t drop = std::min(opacity_budget, low_count);
        if (drop != 0) {
            auto ranked = masks.opacities.masked_fill(
                recycled_rows.logical_not(),
                std::numeric_limits<float>::infinity());
            auto sorted = ranked.sort(0, false);
            auto recycled = tinytensor::Tensor::zeros_bool(
                {old_count}, tinytensor::Device::CUDA);
            recycled.index_fill_(
                0,
                sorted.second.slice(0, 0, drop).to(
                    tinytensor::DataType::Int32),
                1.F);
            keep = keep.logical_and(recycled.logical_not());
        }
    }

    // The strongest row survives unless it is itself hard-pruned. Ties keep
    // the lowest row index, like the host scan; the maximum crosses to the
    // host because the device argmax is not usable here.
    const auto maximum_values = masks.opacities
        .max().to(tinytensor::Device::CPU).to_vector();
    const float maximum_opacity =
        maximum_values.empty() ? -1.F : maximum_values.front();
    const auto best = masks.opacities.ge(maximum_opacity)
        .nonzero().squeeze(1).slice(0, 0, 1)
        .to(tinytensor::DataType::Int32);
    auto best_mask = tinytensor::Tensor::zeros_bool(
        {old_count}, tinytensor::Device::CUDA);
    if (best.numel() != 0) best_mask.index_fill_(0, best, 1.F);
    keep = keep.logical_or(best_mask.logical_and(masks.hard.logical_not()));

    // Cap: drop the least opaque survivors, never the rescued row.
    std::size_t retained = keep.count_nonzero();
    if (retained > options.densification_cap) {
        const bool best_kept =
            best_mask.logical_and(keep).count_nonzero() != 0;
        const std::size_t removable = retained - (best_kept ? 1 : 0);
        const std::size_t remove = std::min(
            retained - options.densification_cap, removable);
        if (remove != 0) {
            auto ranked = masks.opacities.masked_fill(
                keep.logical_not().logical_or(best_mask),
                std::numeric_limits<float>::infinity());
            auto sorted = ranked.sort(0, false);
            auto clamped = tinytensor::Tensor::zeros_bool(
                {old_count}, tinytensor::Device::CUDA);
            clamped.index_fill_(
                0,
                sorted.second.slice(0, 0, remove).to(
                    tinytensor::DataType::Int32),
                1.F);
            keep = keep.logical_and(clamped.logical_not());
        }
    }

    auto keep_indices = keep.nonzero().squeeze(1).to(
        tinytensor::DataType::Int32);
    retained = keep_indices.numel();
    const std::size_t pruned = old_count - retained;
    const auto retained_screen =
        stats.max_screen_radius.index_select(0, keep_indices);
    gpu_detail::select_training_rows_gpu(model, keep_indices, states);

    const std::size_t growth_cap = stats.growth_cap == 0
        ? options.densification_cap
        : std::min(options.densification_cap, std::max(old_count, stats.growth_cap));
    const std::size_t capacity = growth_cap > model.size()
        ? growth_cap - model.size()
        : 0;
    const float remaining_progress = 1.F -
        static_cast<float>(iteration) /
            std::max(1.F, static_cast<float>(options.iterations));
    const auto decay_adc = [&]() {
        detail::apply_adc_decay(
            model,
            options.opacity_decay * std::max(remaining_progress, 0.F),
            options.scale_decay * std::max(remaining_progress, 0.F));
    };
    if (capacity == 0) {
        decay_adc();
        stats = detail::make_densification_stats(model.size());
        return {0, pruned};
    }

    const auto retained_count = stats.count.index_select(0, keep_indices);
    const auto retained_gradient = stats.gradient.index_select(0, keep_indices);
    const auto retained_priority = stats.priority.index_select(0, keep_indices);
    const auto retained_opacity = masks.opacities.index_select(0, keep_indices);
    const auto candidate = retained_count.gt(0.F);
    // Persistent oversize evidence per contributing observation.
    const auto priority = retained_priority.div(retained_count.clamp_min(1.F));
    const auto positive = candidate.logical_and(priority.gt(0.F));
    const std::size_t positive_count = positive.count_nonzero();
    float priority_median = 1.F;
    if (positive_count != 0)
        priority_median = std::max(
            positive_upper_median(priority, positive, positive_count), 1e-9F);
    const auto edge_factor = priority
        .masked_fill(priority.gt(0.F).logical_not(), 0.F)
        .div(priority_median)
        .clamp_max(10.F)
        .mul(0.25F)
        .add(1.F);

    const bool allow_growth = iteration < grow_stop_iteration(options);
    const auto oversized = candidate.logical_and(
        retained_screen.gt(options.densify_screen_threshold));
    // Replacement parents: opacity x edge evidence over every candidate.
    const auto replacement_weights = retained_opacity.mul(edge_factor)
        .masked_fill(candidate.logical_not(), 0.F);

    tinytensor::Tensor growth_weights;
    std::size_t desired_growth = 0;
    if (allow_growth) {
        const auto growth_mask = candidate.logical_and(
            retained_gradient.gt(options.densify_gradient_threshold));
        // On-screen oversize rows are sampled twice as often as the rest.
        const auto screen_factor = tinytensor::Tensor::full(
            {retained}, 1.F, tinytensor::Device::CUDA)
            .masked_fill(oversized, 2.F);
        growth_weights = retained_gradient.mul(edge_factor)
            .mul(screen_factor)
            .masked_fill(growth_mask.logical_not(), 0.F);
        desired_growth = static_cast<std::size_t>(std::llround(
            static_cast<double>(growth_mask.count_nonzero()) *
            static_cast<double>(options.densify_select_fraction)));
    }

    const std::size_t oversized_count =
        allow_growth ? oversized.count_nonzero() : 0;
    const auto oversize_scores = allow_growth
        ? retained_screen.mul(edge_factor).masked_fill(oversized.logical_not(), 0.F)
        : tinytensor::Tensor{};
    const auto selection = select_igs_parents(replacement_weights,
        oversize_scores, growth_weights, pruned, desired_growth, capacity);
    const auto& split_parents = selection.parents;
    const std::size_t grown = split_parents.numel();
    gpu_detail::grow_igs_random_gpu(
        model, split_parents, retained_screen, options, random, states);
    decay_adc();
    stats = detail::make_densification_stats(model.size());

    core::Logger::instance().info(
        "igs_refine iteration=", iteration,
        " gaussians=", model.size(),
        " pruned=", pruned,
        " retained=", retained,
        " replacement_selected=", selection.replacement,
        " oversized_candidates=", oversized_count,
        " oversized_selected=", selection.oversized,
        " growth_selected=", selection.growth,
        " grown=", grown,
        " capacity=", capacity,
        " growth_cap=", growth_cap,
        " growing=", allow_growth);
    return {grown, pruned};
}

}  // namespace photara::splat::densification
