#include "densification_adc_plus.hpp"

#include "core/logging.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace aetherscan::splat::densification {
namespace {

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

void grow_adc_plus_gpu(
    GaussianModel& model, const tinytensor::Tensor& parents,
    const TrainingOptions& options, const AdamStates& states,
    const tinytensor::Tensor& screen_sizes, const int split_mode,
    const float split_opacity_k) {
    const std::size_t count = parents.numel();
    if (count == 0) return;
    GaussianModel children = select_model_rows(model, parents);
    auto samples = tinytensor::Tensor::zeros(
        {count, std::size_t{3}}, tinytensor::Device::CUDA);
    auto selected_screen = screen_sizes.index_select(0, parents);
    detail::split_gaussians(
        model, children, parents, samples, selected_screen, split_mode,
        options.prune_opacity,
        options.densify_screen_threshold,
        split_opacity_k);
    if (!options.densify_keep_parent_adam)
        detail::zero_adam_rows(parents, states);
    append_model(model, children);
    for (detail::AdamState* state : states)
        append_zero_adam(*state, count);
}

float scheduled_las_opacity_k(
    const TrainingOptions& options, const unsigned iteration) {
    float k = options.densify_las_opacity_k_final;
    if (options.densify_las_opacity_k_warmup > 0) {
        const float t = std::clamp(
            static_cast<float>(iteration) /
                static_cast<float>(options.densify_las_opacity_k_warmup),
            0.F, 1.F);
        k = options.densify_las_opacity_k_init +
            t * (options.densify_las_opacity_k_final -
                 options.densify_las_opacity_k_init);
    }
    return k;
}

}  // namespace

tinytensor::Tensor AdcPlusStrategy::growth_candidates(
    const tinytensor::Tensor& eligible, const tinytensor::Tensor&) const {
    return eligible.nonzero().squeeze(1).to(tinytensor::DataType::Int32);
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
        const auto supported = kept_counts.ge(k_evidence_visible_steps)
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
    if (options.densify_use_error_map)
        retained_gradient = detail::densify_mean_scores(
            retained_gradient, retained_count, 1.F);
    if (options.densification_strategy == DensificationStrategy::adc_igs &&
        options.densify_use_error_map)
        // Retain ADC+'s near-camera footprint correction when replacing its
        // gradient score with IGS image/world evidence. Use the corrected
        // score for relocation and oversize ranking as well as net growth.
        retained_gradient = detail::adc_plus_footprint_weights(
            retained_gradient, retained_screen);
    select_training_rows_gpu(model, keep_indices, states);
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
        // A single contributing observation is insufficient evidence to
        // replicate geometry. Keep its parent, but spend IGS growth/recycle
        // budget on rows observed repeatedly within this refinement window.
        const auto visible = options.densification_strategy == DensificationStrategy::adc_igs
            ? retained_count.ge(2.F) : retained_count.gt(0.F);
        auto growth_eligible = visible;
        const bool geometry_gated_growth =
            options.densification_strategy == DensificationStrategy::adc_igs &&
            options.densify_geometry_gradient_threshold > 0.F;
        if (geometry_gated_growth)
            growth_eligible = visible.logical_and(
                stats.geometry_gradient.index_select(0, keep_indices).gt(
                    options.densify_geometry_gradient_threshold));
        auto visible_indices = visible.nonzero().squeeze(1).to(
            tinytensor::DataType::Int32);
        const std::size_t replacement_count = std::min(
            {pruning.pruned, capacity, visible_indices.numel()});
        if (replacement_count != 0) {
            auto weights = options.densify_relocate
                ? retained_gradient.index_select(0, visible_indices)
                : retained_opacity.index_select(0, visible_indices);
            weights = weights.clamp_min(1e-12F);
            auto sampled_slots = tinytensor::Tensor::multinomial(
                weights, static_cast<int>(replacement_count), false);
            auto sampled = visible_indices.index_select(
                0, sampled_slots).to(tinytensor::DataType::Int32);
            selected.index_fill_(0, sampled, 1.F);
            selected_count = replacement_count;
            replacement_selected = replacement_count;
        }

        std::size_t extra_budget = 0;
        if (iteration < options.grow_stop_iter &&
            selected_count < capacity) {
            if (options.densify_growth_factor > 1.F) {
                const std::size_t n_target = std::min(
                    count_cap,
                    static_cast<std::size_t>(
                        options.densify_growth_factor *
                        static_cast<double>(retained)));
                extra_budget = n_target > retained ? n_target - retained : 0;
                if (geometry_gated_growth) {
                    const auto unresolved = growth_eligible.nonzero().numel();
                    const auto requested = static_cast<std::size_t>(std::llround(
                        unresolved * options.densify_select_fraction *
                        static_cast<double>(strategy_schedule(options).every) / 200.0));
                    extra_budget = std::min(extra_budget,
                        requested > pruning.pruned ? requested - pruning.pruned : 0);
                }
            } else {
                auto growth_mask = visible.logical_and(
                    retained_gradient.gt(
                        options.densify_gradient_threshold));
                const auto eligible_count = growth_mask.nonzero().numel();
                const std::size_t threshold_growth =
                    static_cast<std::size_t>(std::llround(
                        eligible_count *
                        options.densify_select_fraction));
                extra_budget = threshold_growth > pruning.pruned
                    ? threshold_growth - pruning.pruned
                    : 0;
            }
            extra_budget = std::min(extra_budget, capacity - selected_count);
        }

        const bool sampled_oversize =
            options.densify_oversize_split_fraction > 0.F;
        if (sampled_oversize) {
            std::size_t n_oversize = extra_budget == 0
                ? 0
                : static_cast<std::size_t>(std::llround(
                      static_cast<double>(extra_budget) *
                      options.densify_oversize_split_fraction));
            n_oversize = std::min(n_oversize, extra_budget);
            tinytensor::Tensor oversize_weights;
            if (options.densification_strategy == DensificationStrategy::adc_igs) {
                // max_screen_radius remains the hard-clip backstop; budget
                // sampling uses the sum of per-observation log2 oversize.
                auto evidence = stats.priority.index_select(0, keep_indices);
                auto score = retained_gradient.clamp_min(0.F);
                if (options.densify_oversize_score_blend == 0.F)
                    oversize_weights = evidence;
                else if (options.densify_oversize_score_blend == 1.F)
                    oversize_weights = evidence * score;
                else
                    oversize_weights = evidence * score.pow(options.densify_oversize_score_blend);
            } else {
                oversize_weights = detail::densify_oversize_weights(
                    retained_gradient, retained_screen,
                    options.densify_screen_threshold,
                    options.densify_oversize_score_blend);
            }
            oversize_weights = oversize_weights.masked_fill(selected, 0.F);
            oversize_weights =
                oversize_weights.masked_fill(visible.logical_not(), 0.F);
            auto oversize_indices = oversize_weights.gt(0.F)
                .nonzero().squeeze(1).to(tinytensor::DataType::Int32);
            n_oversize = std::min(n_oversize, oversize_indices.numel());
            if (n_oversize != 0) {
                auto sampled_slots = tinytensor::Tensor::multinomial(
                    oversize_weights.index_select(0, oversize_indices)
                        .clamp_min(1e-12F),
                    static_cast<int>(n_oversize), false);
                auto sampled = oversize_indices.index_select(
                    0, sampled_slots).to(tinytensor::DataType::Int32);
                selected.index_fill_(0, sampled, 1.F);
                selected_count += n_oversize;
                oversized_selected_count = n_oversize;
                extra_budget -= n_oversize;
            }
        } else {
            auto oversized = visible.logical_and(
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

        if (extra_budget > 0 && selected_count < capacity) {
            auto growth_mask = growth_eligible.logical_and(
                retained_gradient.gt(
                    options.densify_gradient_threshold));
            auto growth_indices = growth_candidates(growth_mask, selected);
            const std::size_t growth_count = std::min(
                {extra_budget, capacity - selected_count,
                 growth_indices.numel()});
            if (growth_count != 0) {
                auto raw_weights =
                    retained_gradient.index_select(0, growth_indices);
                auto weights = options.densify_use_error_map
                    ? raw_weights
                    : detail::adc_plus_footprint_weights(
                          raw_weights,
                          retained_screen.index_select(0, growth_indices));
                weights = weights.clamp_min(1e-12F);
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
    grow_adc_plus_gpu(
        model, split_parents, options, states, retained_screen, split_mode(),
        scheduled_las_opacity_k(options, iteration));
    const float remaining_progress = 1.F -
        static_cast<float>(iteration) /
            std::max(1.F, static_cast<float>(options.iterations));
    detail::apply_adc_decay(
        model,
        // opacity_decay was calibrated for ADC+'s 200-step interval.
        // IGS refining every 100 steps must not double that regularizer.
        options.opacity_decay * std::max(remaining_progress, 0.F) *
            (options.densification_strategy == DensificationStrategy::adc_igs
                ? static_cast<float>(strategy_schedule(options).every) / 200.F
                : 1.F),
        0.F);
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
