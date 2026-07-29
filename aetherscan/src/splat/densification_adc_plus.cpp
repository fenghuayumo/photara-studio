#include "densification_internal.hpp"

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

void zero_adam_rows_gpu(
    const tinytensor::Tensor& indices, const AdamStates& states) {
    if (indices.numel() == 0) return;
    for (detail::AdamState* state : states) {
        const auto zero_rows = [&indices](tinytensor::Tensor& tensor) {
            std::vector<std::size_t> dimensions = tensor.shape().dims();
            dimensions[0] = indices.numel();
            const auto zeros = tinytensor::Tensor::zeros(
                tinytensor::TensorShape(dimensions),
                tinytensor::Device::CUDA);
            tensor.index_copy_(0, indices, zeros);
        };
        zero_rows(state->first);
        zero_rows(state->second);
    }
}

void grow_adc_plus_gpu(
    GaussianModel& model, const tinytensor::Tensor& parents,
    const TrainingOptions& options, const AdamStates& states,
    const tinytensor::Tensor& screen_sizes) {
    const std::size_t count = parents.numel();
    if (count == 0) return;
    GaussianModel children = select_model_rows(model, parents);
    auto samples = tinytensor::Tensor::zeros(
        {count, std::size_t{3}}, tinytensor::Device::CUDA);
    auto selected_screen = screen_sizes.index_select(0, parents);
    detail::split_gaussians(
        model, children, parents, samples, selected_screen, 2,
        options.prune_opacity,
        options.densify_screen_threshold);
    zero_adam_rows_gpu(parents, states);
    append_model(model, children);
    for (detail::AdamState* state : states)
        append_zero_adam(*state, count);
}

}  // namespace

RefinementCounts internal::refine_adc_plus_gpu(
    GaussianModel& model, detail::DensificationStats& stats,
    const unsigned iteration, const float scene_extent,
    const mvs::Vec3f& scene_center, const TrainingOptions& options,
    const AdamStates& states) {
    const std::size_t old_count = model.size();
    auto pruning = detail::adc_plus_prune(
        model, options.prune_opacity, 100.F * scene_extent,
        {scene_center.x(), scene_center.y(), scene_center.z()},
        options.densification_cap);
    const std::size_t retained = pruning.keep_indices.numel();
    auto retained_gradient =
        stats.gradient.index_select(0, pruning.keep_indices);
    auto retained_count =
        stats.count.index_select(0, pruning.keep_indices);
    auto retained_screen =
        stats.max_screen_radius.index_select(0, pruning.keep_indices);
    auto retained_opacity =
        pruning.opacities.index_select(0, pruning.keep_indices);
    select_training_rows_gpu(model, pruning.keep_indices, states);

    const std::size_t capacity =
        options.densification_cap > retained
            ? options.densification_cap - retained
            : 0;
    auto selected = tinytensor::Tensor::zeros_bool(
        {retained}, tinytensor::Device::CUDA);
    std::size_t selected_count = 0;
    if (capacity != 0 && retained != 0) {
        const auto visible = retained_count.gt(0.F);
        auto visible_indices = visible.nonzero().squeeze(1).to(
            tinytensor::DataType::Int32);
        const std::size_t replacement_count = std::min(
            {pruning.pruned, capacity, visible_indices.numel()});
        if (replacement_count != 0) {
            auto weights = retained_opacity.index_select(
                0, visible_indices);
            auto sampled_slots = tinytensor::Tensor::multinomial(
                weights, static_cast<int>(replacement_count), false);
            auto sampled = visible_indices.index_select(
                0, sampled_slots).to(tinytensor::DataType::Int32);
            selected.index_fill_(0, sampled, 1.F);
            selected_count = replacement_count;
        }

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
        }

        if (iteration < options.grow_stop_iter &&
            selected_count < capacity) {
            auto growth_mask = visible.logical_and(
                retained_gradient.gt(
                    options.densify_gradient_threshold));
            auto growth_indices = growth_mask.nonzero().squeeze(1).to(
                tinytensor::DataType::Int32);
            const std::size_t threshold_growth =
                static_cast<std::size_t>(std::llround(
                    growth_indices.numel() *
                    options.densify_select_fraction));
            const std::size_t requested =
                threshold_growth > pruning.pruned
                    ? threshold_growth - pruning.pruned
                    : 0;
            const std::size_t growth_count = std::min(
                {requested, capacity - selected_count,
                 growth_indices.numel()});
            if (growth_count != 0) {
                auto weights = retained_gradient.index_select(
                    0, growth_indices);
                auto sampled_slots = tinytensor::Tensor::multinomial(
                    weights, static_cast<int>(growth_count), false);
                auto sampled = growth_indices.index_select(
                    0, sampled_slots).to(tinytensor::DataType::Int32);
                selected.index_fill_(0, sampled, 1.F);
            }
        }
    }

    auto split_parents = selected.nonzero().squeeze(1).to(
        tinytensor::DataType::Int32);
    grow_adc_plus_gpu(
        model, split_parents, options, states, retained_screen);
    const float remaining_progress = 1.F -
        static_cast<float>(iteration) /
            std::max(1.F, static_cast<float>(options.iterations));
    detail::apply_adc_decay(
        model,
        options.opacity_decay * std::max(remaining_progress, 0.F),
        0.F);
    stats = detail::make_densification_stats(model.size());
    return {split_parents.numel(), pruning.pruned};
}

}  // namespace aetherscan::splat::densification
