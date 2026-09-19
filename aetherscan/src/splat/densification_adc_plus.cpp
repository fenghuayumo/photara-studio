#include "densification_adc_plus.hpp"

#include "core/logging.hpp"
#include "densification_internal.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace aetherscan::splat::densification {
namespace gpu_detail {

GaussianModel select_model_rows(
    const GaussianModel& model, const tinytensor::Tensor& indices) {
    GaussianModel selected;
    selected.means = model.means.index_select(0, indices);
    selected.log_scales = model.log_scales.index_select(0, indices);
    selected.quaternions = model.quaternions.index_select(0, indices);
    selected.opacity_logits = model.opacity_logits.index_select(0, indices);
    selected.sh = model.sh.index_select(0, indices);
    if (model.normal_features.is_valid())
        selected.normal_features =
            model.normal_features.index_select(0, indices);
    selected.sh_degree = model.sh_degree;
    return selected;
}

void append_model(GaussianModel& model, const GaussianModel& added) {
    if (added.size() == 0) return;
    model.means = tinytensor::Tensor::cat({model.means, added.means}, 0);
    model.log_scales = tinytensor::Tensor::cat(
        {model.log_scales, added.log_scales}, 0);
    model.quaternions = tinytensor::Tensor::cat(
        {model.quaternions, added.quaternions}, 0);
    model.opacity_logits = tinytensor::Tensor::cat(
        {model.opacity_logits, added.opacity_logits}, 0);
    model.sh = tinytensor::Tensor::cat({model.sh, added.sh}, 0);
    if (model.normal_features.is_valid())
        model.normal_features = tinytensor::Tensor::cat(
            {model.normal_features, added.normal_features}, 0);
}

void select_adam_rows(
    detail::AdamState& state, const tinytensor::Tensor& indices) {
    state.first = state.first.index_select(0, indices);
    state.second = state.second.index_select(0, indices);
}

void append_zero_adam(detail::AdamState& state, const std::size_t count) {
    if (count == 0) return;
    const auto append_zeros = [count](tinytensor::Tensor& tensor) {
        std::vector<std::size_t> dimensions = tensor.shape().dims();
        dimensions[0] = count;
        tensor = tinytensor::Tensor::cat(
            {tensor, tinytensor::Tensor::zeros(
                         tinytensor::TensorShape(dimensions),
                         tinytensor::Device::CUDA)},
            0);
    };
    append_zeros(state.first);
    append_zeros(state.second);
}

void select_training_rows_gpu(
    GaussianModel& model, const tinytensor::Tensor& indices,
    const AdamStates& states) {
    model = select_model_rows(model, indices);
    for (detail::AdamState* state : states)
        select_adam_rows(*state, indices);
}

tinytensor::Tensor weighted_sample_without_replacement(
    const tinytensor::Tensor& weights, const std::size_t count) {
    if (count == 0 || weights.numel() == 0) return {};
    // The host rule kept finite, strictly positive weights only; multi-
    // nomial-without-replacement then realises the same ordered draw.
    const auto valid = weights.isfinite().logical_and(weights.gt(0.F));
    const auto indices = valid.nonzero().squeeze(1).to(
        tinytensor::DataType::Int32);
    const std::size_t available = indices.numel();
    if (available == 0) return {};
    const std::size_t requested = std::min(count, available);
    const auto slots = tinytensor::Tensor::multinomial(
        weights.index_select(0, indices).clamp_min(1e-12F),
        static_cast<int>(requested), false);
    return indices.index_select(0, slots).to(tinytensor::DataType::Int32);
}

void grow_igs_random_gpu(
    GaussianModel& model, const tinytensor::Tensor& parents,
    const tinytensor::Tensor& screen_sizes, const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states) {
    const std::size_t count = parents.numel();
    if (count == 0) return;
    // One normal per local axis: the random-axis split direction, sampled on
    // the host exactly like the legacy grow path it replaces.
    std::vector<float> samples(3 * count);
    std::normal_distribution<float> normal(0.F, 1.F);
    for (float& value : samples) value = normal(random);
    const auto random_tensor = tinytensor::Tensor::from_vector(
        samples, {count, std::size_t{3}}, tinytensor::Device::CUDA);
    grow_parents_gpu(
        model, parents, screen_sizes, detail::SplitMode::igs_random,
        random_tensor, options.prune_opacity, 0.F, false, states);
}

void grow_parents_gpu(
    GaussianModel& model, const tinytensor::Tensor& parents,
    const tinytensor::Tensor& screen_sizes, const detail::SplitMode mode,
    const tinytensor::Tensor& samples, const float minimum_opacity,
    const float split_at_screen_size, const bool keep_parent_adam,
    const AdamStates& states) {
    const std::size_t count = parents.numel();
    if (count == 0) return;
    GaussianModel children = select_model_rows(model, parents);
    const auto split_samples = samples.is_valid()
        ? samples
        : tinytensor::Tensor::zeros(
              {count, std::size_t{3}}, tinytensor::Device::CUDA);
    const auto selected_screen = screen_sizes.index_select(0, parents);
    detail::split_gaussians(
        model, children, parents, split_samples, selected_screen, mode,
        minimum_opacity, split_at_screen_size);
    // Splitting mutates the retained parent as well as creating a child, so
    // both rows start with clean optimizer moments.
    if (!keep_parent_adam) detail::zero_adam_rows(parents, states);
    append_model(model, children);
    for (detail::AdamState* state : states)
        append_zero_adam(*state, count);
}

}  // namespace gpu_detail

tinytensor::Tensor AdcPlusStrategy::growth_candidates(
    const tinytensor::Tensor& eligible, const tinytensor::Tensor& selected) const {
    return eligible.logical_and(!selected).nonzero().squeeze(1).to(
        tinytensor::DataType::Int32);
}

RefinementCounts AdcPlusStrategy::refine(
    GaussianModel& model, detail::DensificationStats& stats,
    const unsigned iteration, const float scene_extent,
    const mvs::Vec3f& scene_center, const TrainingOptions& options,
    const AdamStates& states) const {
    const std::size_t old_count = model.size();
    if (options.densify_clip_screen_size &&
        stats.max_screen_radius.is_valid() &&
        stats.max_screen_radius.numel() == old_count) {
        detail::clip_log_scale_by_screen(
            model.log_scales, stats.max_screen_radius,
            options.densify_screen_threshold,
            options.densify_screen_clip_hardness);
    }
    const std::size_t count_cap = options.densification_cap;
    auto pruning = detail::adc_plus_prune(
        model, options.prune_opacity, 100.F * scene_extent,
        {scene_center.x(), scene_center.y(), scene_center.z()},
        count_cap);
    // Evidence prune: a row that was actually contributing in fewer than two
    // steps of the observation window while carrying soft-floor opacity is a
    // floater in waiting. The opacity floor reuses prune_opacity (12x) so no
    // new strategy parameter is introduced.
    constexpr float k_evidence_visible_steps = 2.F;
    constexpr float k_low_visibility_opacity_factor = 12.F;
    const float low_visibility_opacity = std::max(
        k_low_visibility_opacity_factor * options.prune_opacity, 1e-4F);
    auto keep_indices = pruning.keep_indices;
    std::size_t evidence_pruned = 0;
    if (keep_indices.numel() != 0) {
        const auto kept_counts =
            stats.count.index_select(0, keep_indices);
        const auto kept_opacities =
            pruning.opacities.index_select(0, keep_indices);
        const auto evidence = kept_counts.ge(k_evidence_visible_steps);
        const auto supported = evidence
            .logical_or(kept_opacities.ge(low_visibility_opacity));
        const auto supported_positions =
            supported.nonzero().squeeze(1).to(
                tinytensor::DataType::Int32);
        if (supported_positions.numel() != 0) {
            evidence_pruned =
                keep_indices.numel() - supported_positions.numel();
            keep_indices =
                keep_indices.index_select(0, supported_positions);
        }
    }
    const std::size_t retained = keep_indices.numel();
    auto retained_gradient =
        stats.gradient.index_select(0, keep_indices);
    auto retained_count =
        stats.count.index_select(0, keep_indices);
    auto retained_screen =
        stats.max_screen_radius.index_select(0, keep_indices);
    auto retained_opacity =
        pruning.opacities.index_select(0, keep_indices);
    gpu_detail::select_training_rows_gpu(model, keep_indices, states);
    const std::size_t pruned = old_count - retained;


    const std::size_t capacity =
        count_cap > retained ? count_cap - retained : 0;
    auto selected = tinytensor::Tensor::zeros_bool(
        {retained}, tinytensor::Device::CUDA);
    std::size_t selected_count = 0;
    std::size_t replacement_selected = 0;
    std::size_t oversized_selected_count = 0;
    std::size_t growth_selected_count = 0;
    if (capacity != 0 && retained != 0) {
        const auto growth_eligible = retained_count.gt(0.F);
        std::size_t extra_budget = 0;
        if (iteration < options.grow_stop_iter &&
            selected_count < capacity) {
            auto growth_mask = growth_eligible.logical_and(
                retained_gradient.gt(options.densify_gradient_threshold));
            const auto eligible_count = growth_mask.nonzero().numel();
            const std::size_t threshold_growth =
                static_cast<std::size_t>(std::llround(
                    eligible_count * options.densify_select_fraction));
            extra_budget = threshold_growth > pruning.pruned
                ? threshold_growth - pruning.pruned
                : 0;
            extra_budget = std::min(extra_budget, capacity - selected_count);
        }

        const bool sampled_oversize =
            options.densify_oversize_split_fraction > 0.F;
        if (sampled_oversize) {
            // Recycled slots remain usable at the cap. Previously generic
            // replacements consumed every freed slot before size repair.
            const std::size_t repair_budget = std::min(
                capacity, extra_budget + std::min(pruning.pruned, capacity));
            std::size_t n_oversize = repair_budget == 0
                ? 0
                : static_cast<std::size_t>(std::llround(
                      static_cast<double>(repair_budget) *
                      options.densify_oversize_split_fraction));
            n_oversize = std::min(n_oversize, repair_budget);
            auto oversize_weights = detail::densify_oversize_weights(
                retained_gradient, retained_screen,
                options.densify_screen_threshold,
                options.densify_oversize_score_blend);
            oversize_weights = oversize_weights.masked_fill(selected, 0.F);
            oversize_weights =
                oversize_weights.masked_fill(growth_eligible.logical_not(), 0.F);
            auto oversize_indices = oversize_weights.gt(0.F)
                .nonzero().squeeze(1).to(tinytensor::DataType::Int32);
            n_oversize = std::min(n_oversize, oversize_indices.numel());
            if (n_oversize != 0) {
                auto sampled =
                    gpu_detail::weighted_sample_without_replacement(
                        oversize_weights, n_oversize);
                selected.index_fill_(0, sampled, 1.F);
                selected_count += n_oversize;
                oversized_selected_count = n_oversize;
            }
        } else {
            auto oversized = growth_eligible.logical_and(
                retained_screen.gt(
                    options.densify_screen_threshold));
            oversized = oversized.logical_and(!selected);
            auto oversized_indices = oversized.nonzero().squeeze(1).to(
                tinytensor::DataType::Int32);
            const std::size_t oversized_count = std::min(
                oversized_indices.numel(), capacity - selected_count);
            if (oversized_count != 0) {
                if (oversized_count != oversized_indices.numel())
                    oversized_indices = oversized_indices.slice(
                        0, 0, oversized_count);
                selected.index_fill_(0, oversized_indices, 1.F);
                selected_count += oversized_count;
                oversized_selected_count = oversized_count;
            }
        }

        // Repair large splats first, then spend the remaining recycled slots
        // on ordinary replacements. All paths select disjoint parent rows.
        const std::size_t repair_growth = selected_count > pruning.pruned
            ? selected_count - pruning.pruned : 0;
        if (sampled_oversize)
            extra_budget -= std::min(extra_budget, repair_growth);
        auto visible_indices = growth_eligible.logical_and(!selected)
            .nonzero().squeeze(1).to(tinytensor::DataType::Int32);
        const std::size_t replacement_count = std::min(
            {pruning.pruned > selected_count ? pruning.pruned - selected_count : 0,
             capacity - selected_count, visible_indices.numel()});
        if (replacement_count != 0) {
            auto weights = options.densify_relocate
                ? retained_gradient.index_select(0, visible_indices)
                : retained_opacity.index_select(0, visible_indices);
            auto sampled_slots = tinytensor::Tensor::multinomial(
                weights.clamp_min(1e-12F), static_cast<int>(replacement_count), false);
            auto sampled = visible_indices.index_select(0, sampled_slots)
                .to(tinytensor::DataType::Int32);
            selected.index_fill_(0, sampled, 1.F);
            selected_count += replacement_count;
            replacement_selected = replacement_count;
        }

        if (extra_budget > 0 && selected_count < capacity) {
            auto growth_mask = growth_eligible.logical_and(
                retained_gradient.gt(
                    options.densify_gradient_threshold));
            auto growth_indices = growth_candidates(growth_mask, selected);
            const std::size_t growth_count = std::min(
                {extra_budget, capacity - selected_count,
                 growth_indices.numel()});
            if (growth_count != 0) {
                const auto raw_weights =
                    retained_gradient.index_select(0, growth_indices);
                const auto weights = detail::adc_plus_footprint_weights(
                    raw_weights,
                    retained_screen.index_select(0, growth_indices))
                    .clamp_min(1e-12F);
                auto sampled_slots = tinytensor::Tensor::multinomial(
                    weights, static_cast<int>(growth_count), false);
                auto sampled = growth_indices.index_select(
                    0, sampled_slots).to(tinytensor::DataType::Int32);
                selected.index_fill_(0, sampled, 1.F);
                growth_selected_count = growth_count;
            }
        }
    }

    auto split_parents = selected.nonzero().squeeze(1).to(
        tinytensor::DataType::Int32);
    gpu_detail::grow_parents_gpu(
        model, split_parents, retained_screen,
        detail::SplitMode::adc_covariance, {}, options.prune_opacity,
        options.densify_screen_threshold, options.densify_keep_parent_adam,
        states);
    const float remaining_progress = 1.F -
        static_cast<float>(iteration) /
            std::max(1.F, static_cast<float>(options.iterations));
    detail::apply_adc_decay(
        model,
        // Calibrated for the shared 200-step refinement interval; the
        // optional scale factor defaults to zero (matching Brush).
        options.opacity_decay * std::max(remaining_progress, 0.F),
        options.scale_decay * std::max(remaining_progress, 0.F));
    stats = detail::make_densification_stats(model.size());
    aetherscan::core::Logger::instance().info(
        name(), "_refine iteration=", iteration,
        " gaussians=", model.size(),
        " pruned_total=", pruned,
        " pruned_low_visibility=", evidence_pruned,
        " replacement_selected=", replacement_selected,
        " oversized_selected=", oversized_selected_count,
        " growth_selected=", growth_selected_count,
        " capacity=", capacity,
        " count_cap=", count_cap,
        " growing=", iteration < options.grow_stop_iter);
    return {split_parents.numel(), pruned};
}

}  // namespace aetherscan::splat::densification
