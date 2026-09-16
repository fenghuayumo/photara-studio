#include "densification.hpp"

#include "densification_adc_plus.hpp"
#include "densification_igs.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace aetherscan::splat::densification {
namespace {

template <typename T>
std::vector<T> download(const tinytensor::Tensor& tensor) {
    std::vector<T> values(tensor.numel());
    if (!values.empty()) {
        const cudaError_t error = cudaMemcpy(
            values.data(), tensor.data_ptr(), values.size() * sizeof(T),
            cudaMemcpyDeviceToHost);
        if (error != cudaSuccess)
            throw std::runtime_error(
                std::string("Failed to download tensor: ") +
                cudaGetErrorString(error));
    }
    return values;
}

tinytensor::Tensor index_tensor(const std::vector<int>& indices) {
    return tinytensor::Tensor::from_vector(
        indices, {indices.size()}, tinytensor::Device::CUDA);
}

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

void zero_adam_rows(
    const std::vector<int>& rows, const AdamStates& states) {
    if (rows.empty()) return;
    const auto indices = index_tensor(rows);
    detail::zero_adam_rows(indices, states);
}

void select_training_rows(
    GaussianModel& model, const std::vector<int>& keep,
    const AdamStates& states) {
    const auto indices = index_tensor(keep);
    model = select_model_rows(model, indices);
    for (detail::AdamState* state : states) select_adam_rows(*state, indices);
}

void grow_training_model(
    GaussianModel& model, const std::vector<int>& parents,
    const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states,
    const std::vector<float>& screen_sizes) {
    if (parents.empty()) return;
    const auto indices = index_tensor(parents);
    GaussianModel children = select_model_rows(model, indices);
    std::vector<float> samples(3 * parents.size());
    std::normal_distribution<float> normal(0.F, 1.F);
    for (float& value : samples) value = normal(random);
    std::vector<float> selected_screen_sizes(parents.size(), 0.F);
    for (std::size_t index = 0; index < parents.size(); ++index)
        if (static_cast<std::size_t>(parents[index]) < screen_sizes.size())
            selected_screen_sizes[index] =
                screen_sizes[static_cast<std::size_t>(parents[index])];
    auto random_tensor = tinytensor::Tensor::from_vector(
        samples, {parents.size(), 3}, tinytensor::Device::CUDA);
    auto screen_tensor = tinytensor::Tensor::from_vector(
        selected_screen_sizes, {parents.size()}, tinytensor::Device::CUDA);
    detail::split_gaussians(
        model, children, indices, random_tensor, screen_tensor,
        detail::SplitMode::dense_tangent,
        options.prune_opacity, 0.F);
    // Splitting mutates the retained parent as well as creating a child.
    // Both are new primitives and must start with clean optimizer moments,
    // matching pygsplat's replacement-based split.
    zero_adam_rows(parents, states);
    append_model(model, children);
    for (detail::AdamState* state : states)
        append_zero_adam(*state, parents.size());
}

std::vector<std::size_t> weighted_unique_sample(
    const std::vector<std::pair<std::size_t, float>>& candidates,
    const std::size_t requested, std::mt19937& random) {
    if (requested == 0 || candidates.empty()) return {};
    std::vector<std::pair<std::size_t, float>> valid;
    valid.reserve(candidates.size());
    for (const auto& candidate : candidates)
        if (std::isfinite(candidate.second) && candidate.second > 0.F)
            valid.push_back(candidate);
    if (valid.empty()) return {};
    const std::size_t count = std::min(requested, valid.size());
    std::vector<double> weights;
    weights.reserve(valid.size());
    for (const auto& candidate : valid)
        weights.push_back(candidate.second);
    std::discrete_distribution<std::size_t> distribution(
        weights.begin(), weights.end());
    std::unordered_set<std::size_t> selected;
    const std::size_t attempts_limit = valid.size() * 8 + requested * 4;
    for (std::size_t attempt = 0;
         attempt < attempts_limit && selected.size() < count; ++attempt)
        selected.insert(valid[distribution(random)].first);
    if (selected.size() < count) {
        std::vector<std::pair<float, std::size_t>> sorted;
        for (const auto& [index, weight] : valid)
            if (!selected.contains(index)) sorted.emplace_back(weight, index);
        std::sort(sorted.begin(), sorted.end(), std::greater<>());
        for (const auto& entry : sorted) {
            if (selected.size() >= count) break;
            selected.insert(entry.second);
        }
    }
    return {selected.begin(), selected.end()};
}

}  // namespace

RefinementCounts refine_gaussians(
    GaussianModel& model, detail::DensificationStats& stats,
    const unsigned iteration, const float scene_extent,
    const mvs::Vec3f& scene_center, const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states) {
    if (!is_refinement_iteration(iteration, options) || model.size() == 0)
        return {};

    if (options.densification_strategy == DensificationStrategy::adc_igs)
        return IgsStrategy{}.refine(
            model, stats, iteration, scene_extent, scene_center, options, states);
    if (options.densification_strategy == DensificationStrategy::adc_plus)
        return AdcPlusStrategy{}.refine(
            model, stats, iteration, scene_extent, scene_center,
            options, states);

    const std::size_t old_count = model.size();
    const auto gradients = download<float>(stats.gradient);
    const auto counts = download<float>(stats.count);
    const auto screen = download<float>(stats.max_screen_radius);
    const auto opacities = download<float>(model.opacity_logits);
    const auto log_scales = download<float>(model.log_scales);
    // dense_adaptive performs a non-finite and bounds sweep over the complete
    // model before selecting replacement rows.
    const std::vector<float> means = download<float>(model.means);
    const std::vector<float> quaternions =
        download<float>(model.quaternions);
    const std::vector<float> sh = download<float>(model.sh);
    std::vector<bool> prune(old_count, false);
    std::vector<bool> hard_prune(old_count, false);
    std::vector<float> opacity_values(old_count);
    std::size_t best = 0;
    float best_opacity = -1.F;
    for (std::size_t index = 0; index < old_count; ++index) {
        const float opacity = 1.F / (1.F + std::exp(-opacities[index]));
        opacity_values[index] = opacity;
        if (opacity > best_opacity) {
            best_opacity = opacity;
            best = index;
        }
        float max_scale = 0.F;
        for (int axis = 0; axis < 3; ++axis) {
            const float scale = std::exp(log_scales[3 * index + axis]);
            max_scale = std::max(max_scale, scale);
        }
        bool non_finite = !std::isfinite(opacities[index]);
        for (int axis = 0; axis < 3; ++axis)
            non_finite = non_finite ||
                !std::isfinite(means[3 * index + axis]) ||
                !std::isfinite(log_scales[3 * index + axis]);
        for (int component = 0; component < 4; ++component)
            non_finite = non_finite ||
                !std::isfinite(quaternions[4 * index + component]);
        const std::size_t sh_stride = model.sh.numel() / old_count;
        for (std::size_t component = 0; component < sh_stride; ++component)
            non_finite = non_finite ||
                !std::isfinite(sh[index * sh_stride + component]);
        const mvs::Vec3f position(
            means[3 * index], means[3 * index + 1], means[3 * index + 2]);
        hard_prune[index] = non_finite ||
            max_scale > 100.F * scene_extent ||
            (position - scene_center).cwiseAbs().maxCoeff() >
                100.F * scene_extent;
        prune[index] = opacity < options.prune_opacity || hard_prune[index];
    }
    std::vector<std::pair<float, std::size_t>> low_opacity;
    std::size_t hard_count = 0;
    for (std::size_t index = 0; index < old_count; ++index) {
        if (hard_prune[index]) {
            ++hard_count;
        } else if (prune[index]) {
            low_opacity.emplace_back(opacity_values[index], index);
            prune[index] = false;
        }
    }
    std::sort(low_opacity.begin(), low_opacity.end());
    const std::size_t recycle_limit = static_cast<std::size_t>(std::ceil(
        old_count * std::clamp(options.dense_recycle_fraction, 0.F, 1.F)));
    const std::size_t opacity_budget = recycle_limit > hard_count
        ? recycle_limit - hard_count
        : 0;
    for (std::size_t index = 0;
         index < std::min(opacity_budget, low_opacity.size()); ++index)
        prune[low_opacity[index].second] = true;
    // Keep training alive for pathological all-low-opacity inputs, but never
    // rescue a non-finite/out-of-bounds row.
    if (!hard_prune[best]) prune[best] = false;
    std::size_t retained = static_cast<std::size_t>(
        std::count(prune.begin(), prune.end(), false));
    if (retained > options.densification_cap) {
        std::vector<std::pair<float, std::size_t>> by_opacity;
        by_opacity.reserve(retained);
        for (std::size_t index = 0; index < old_count; ++index)
            if (!prune[index] && index != best)
                by_opacity.emplace_back(opacity_values[index], index);
        std::sort(by_opacity.begin(), by_opacity.end());
        const std::size_t remove_count = retained - options.densification_cap;
        for (std::size_t index = 0; index < remove_count; ++index)
            prune[by_opacity[index].second] = true;
    }

    std::vector<int> keep;
    keep.reserve(old_count);
    std::vector<int> remap(old_count, -1);
    for (std::size_t index = 0; index < old_count; ++index) {
        if (!prune[index]) {
            remap[index] = static_cast<int>(keep.size());
            keep.push_back(static_cast<int>(index));
        }
    }
    const std::size_t pruned = old_count - keep.size();

    struct Candidate {
        std::size_t old_index{};
        std::size_t new_index{};
        float score{};
        bool oversized{};
    };
    std::vector<Candidate> candidates;
    for (std::size_t index = 0; index < old_count; ++index) {
        if (remap[index] < 0 || counts[index] <= 0.F) continue;
        candidates.push_back({
            index, static_cast<std::size_t>(remap[index]), gradients[index],
            screen[index] > options.densify_screen_threshold});
    }
    select_training_rows(model, keep, states);

    const std::size_t capacity = options.densification_cap > model.size()
        ? options.densification_cap - model.size()
        : 0;
    if (capacity == 0) {
        stats = detail::make_densification_stats(model.size());
        return {0, pruned};
    }

    std::vector<int> split_parents;
    std::vector<std::pair<std::size_t, float>> replacement_weights;
    std::vector<std::pair<std::size_t, float>> growth_weights;
    std::unordered_set<std::size_t> forced;
    for (const Candidate& candidate : candidates) {
        const float screen_factor = candidate.oversized ? 2.F : 1.F;
        replacement_weights.emplace_back(
            candidate.new_index,
            std::max(candidate.score, 1e-12F) * screen_factor);
        if (candidate.score > options.densify_gradient_threshold)
            growth_weights.emplace_back(
                candidate.new_index, candidate.score * screen_factor);
    }
    auto selected = weighted_unique_sample(
        replacement_weights, std::min(pruned, capacity), random);
    forced.insert(selected.begin(), selected.end());
    std::size_t desired_growth = static_cast<std::size_t>(std::llround(
        growth_weights.size() * options.densify_select_fraction));
    desired_growth = std::min(
        desired_growth,
        static_cast<std::size_t>(std::ceil(
            model.size() * std::clamp(
                options.dense_growth_fraction, 0.F, 1.F))));
    growth_weights.erase(
        std::remove_if(
            growth_weights.begin(), growth_weights.end(),
            [&](const auto& candidate) {
                return forced.contains(candidate.first);
            }),
        growth_weights.end());
    const std::size_t remaining = capacity > forced.size()
        ? capacity - forced.size()
        : 0;
    selected = weighted_unique_sample(
        growth_weights, std::min(remaining, desired_growth), random);
    forced.insert(selected.begin(), selected.end());
    split_parents.reserve(std::min(capacity, forced.size()));
    for (const std::size_t parent : forced) {
        if (split_parents.size() >= capacity) break;
        split_parents.push_back(static_cast<int>(parent));
    }

    std::vector<float> retained_screen;
    retained_screen.reserve(keep.size());
    for (const int old_index : keep)
        retained_screen.push_back(
            screen[static_cast<std::size_t>(old_index)]);
    grow_training_model(
        model, split_parents, options, random, states,
        retained_screen);
    stats = detail::make_densification_stats(model.size());
    return {split_parents.size(), pruned};
}

}  // namespace aetherscan::splat::densification
