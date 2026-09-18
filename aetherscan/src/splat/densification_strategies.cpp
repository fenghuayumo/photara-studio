#include "densification.hpp"

#include "densification_adc_plus.hpp"
#include "densification_igs.hpp"
#include "core/logging.hpp"

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
    const detail::SplitMode split_mode, const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states,
    const std::vector<float>& screen_sizes,
    const float split_opacity_k = 0.5F) {
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
        split_mode,
        options.prune_opacity,
        split_mode == detail::SplitMode::adc_covariance
            ? options.densify_screen_threshold
            : 0.F,
        split_opacity_k);
    // Splitting mutates the retained parent as well as creating a child.
    // Both are new primitives and must start with clean optimizer moments.
    zero_adam_rows(parents, states);
    append_model(model, children);
    for (detail::AdamState* state : states)
        append_zero_adam(*state, parents.size());
}

std::vector<std::size_t> weighted_unique_sample(
    const std::vector<std::pair<std::size_t, float>>& candidates,
    const std::size_t requested, const bool gumbel, std::mt19937& random) {
    if (requested == 0 || candidates.empty()) return {};
    std::vector<std::pair<std::size_t, float>> valid;
    valid.reserve(candidates.size());
    for (const auto& candidate : candidates)
        if (std::isfinite(candidate.second) && candidate.second > 0.F)
            valid.push_back(candidate);
    if (valid.empty()) return {};
    const std::size_t count = std::min(requested, valid.size());
    if (gumbel) {
        // Gumbel top-k sampling: log-weight plus standard Gumbel noise,
        // then take the requested largest perturbed scores. This is the
        // ADC-IGS selection rule.
        std::uniform_real_distribution<float> uniform(1e-7F, 1.F - 1e-7F);
        std::vector<std::pair<float, std::size_t>> scores;
        scores.reserve(valid.size());
        for (const auto& [index, weight] : valid) {
            const float u = uniform(random);
            scores.emplace_back(
                std::log(weight) - std::log(-std::log(u)), index);
        }
        const auto middle = scores.begin() +
            static_cast<std::ptrdiff_t>(count);
        std::partial_sort(scores.begin(), middle, scores.end(),
                          std::greater<>());
        std::vector<std::size_t> result;
        result.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
            result.push_back(scores[i].second);
        return result;
    }
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

RefinementCounts refine_emc(
    GaussianModel& model, detail::DensificationStats& stats,
    const unsigned iteration, const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states) {
    const std::size_t old_count = model.size();
    const auto errors = download<float>(stats.image_error);
    const auto counts = download<float>(stats.count);
    const auto oversize = download<float>(stats.priority);
    const auto opacity_logits = download<float>(model.opacity_logits);
    const auto log_scales = download<float>(model.log_scales);

    // The long-axis opacity split factor is scheduled from 0.5 to 0.6
    // over the first 15k iterations.
    const float split_k = 0.5F + 0.1F * std::min(
        1.F, static_cast<float>(iteration) / 15'000.F);

    std::vector<float> score(old_count, 0.F);
    std::vector<char> dead(old_count, 0);
    std::size_t dead_count = 0;
    constexpr float k_dead_log_scale = -40.F;
    for (std::size_t index = 0; index < old_count; ++index) {
        const float opacity =
            1.F / (1.F + std::exp(-opacity_logits[index]));
        float maximum_log_scale = -std::numeric_limits<float>::infinity();
        bool finite = std::isfinite(opacity_logits[index]);
        for (int axis = 0; axis < 3; ++axis) {
            const float log_scale = log_scales[3 * index + axis];
            finite = finite && std::isfinite(log_scale);
            maximum_log_scale = std::max(maximum_log_scale, log_scale);
        }
        dead[index] = !finite ||
            opacity < options.prune_opacity ||
            maximum_log_scale <= k_dead_log_scale;
        if (dead[index]) {
            ++dead_count;
            continue;
        }
        if (counts[index] <= 0.F) continue;
        // EMC score shape: contribution-weighted mean image error,
        // opacity-gated so invisible rows can never draw budget, then
        // compressed by the score power. The window mean is already the
        // accumulated error divided by the observation count.
        const float mean_error =
            errors[index] / std::max(counts[index], 1.F);
        if (!(mean_error > 0.F)) continue;
        score[index] = std::pow(
            mean_error * opacity,
            std::max(options.densify_score_power, 1e-3F));
    }

    // Recycle dead rows in place: each dead row becomes the +delta child of
    // an error-sampled parent. The total count is preserved, which is the
    // MCMC property that keeps error-primary selection from bloating the
    // model when the error map drifts toward background rays.
    std::size_t relocated = 0;
    if (dead_count != 0) {
        std::vector<std::pair<std::size_t, float>> parents;
        parents.reserve(old_count);
        for (std::size_t index = 0; index < old_count; ++index)
            if (!dead[index] && score[index] > 0.F)
                parents.emplace_back(index, score[index]);
        const auto selected = weighted_unique_sample(
            parents, dead_count, true, random);
        if (!selected.empty()) {
            relocated = selected.size();
            std::vector<int> parent_rows;
            std::vector<int> destination_rows;
            parent_rows.reserve(relocated);
            destination_rows.reserve(relocated);
            auto dead_row = dead.begin();
            for (const std::size_t parent : selected) {
                while (dead_row != dead.end() && !*dead_row) ++dead_row;
                if (dead_row == dead.end()) break;
                const std::size_t destination =
                    static_cast<std::size_t>(
                        dead_row - dead.begin());
                ++dead_row;
                parent_rows.push_back(static_cast<int>(parent));
                destination_rows.push_back(static_cast<int>(destination));
            }
            if (!parent_rows.empty()) {
                const auto parent_tensor = index_tensor(parent_rows);
                const auto destination_tensor =
                    index_tensor(destination_rows);
                model.sh.index_copy_(
                    0, destination_tensor,
                    model.sh.index_select(0, parent_tensor));
                if (model.normal_features.is_valid())
                    model.normal_features.index_copy_(
                        0, destination_tensor,
                        model.normal_features.index_select(
                            0, parent_tensor));
                detail::relocate_long_axis(
                    model, parent_tensor, destination_tensor, split_k);
                zero_adam_rows(destination_rows, states);
            }
        }
    }

    // Fixed-budget growth: a plain multiplier of the live count, all draws
    // error-sampled, with a reserved share for accumulated oversize rows.
    std::size_t grown = 0;
    std::size_t oversize_grown = 0;
    std::size_t desired = old_count;
    if (options.densify_growth_factor > 1.F) {
        const auto target = static_cast<std::size_t>(std::floor(
            options.densify_growth_factor *
            static_cast<float>(old_count)));
        desired = std::min(options.densification_cap, target);
    }
    std::size_t additions = desired > old_count ? desired - old_count : 0;
    if (additions != 0) {
        std::vector<std::size_t> split_parents;
        split_parents.reserve(additions);
        const float oversize_share = std::clamp(
            options.densify_oversize_split_fraction, 0.F, 1.F);
        if (oversize_share > 0.F) {
            std::vector<std::pair<std::size_t, float>> weights;
            weights.reserve(old_count);
            for (std::size_t index = 0; index < old_count; ++index) {
                if (dead[index] || score[index] <= 0.F ||
                    !(oversize[index] > 0.F))
                    continue;
                weights.emplace_back(
                    index,
                    (oversize[index] + 1e-3F) *
                        std::pow(
                            score[index],
                            std::clamp(
                                options.densify_oversize_score_blend,
                                0.F, 1.F)));
            }
            const std::size_t oversize_budget = std::min(
                additions,
                static_cast<std::size_t>(std::floor(
                    oversize_share * static_cast<float>(additions))));
            for (const auto index : weighted_unique_sample(
                     weights, oversize_budget, true, random)) {
                split_parents.push_back(index);
                ++oversize_grown;
            }
        }
        if (split_parents.size() < additions) {
            std::vector<std::pair<std::size_t, float>> weights;
            weights.reserve(old_count);
            for (std::size_t index = 0; index < old_count; ++index)
                if (!dead[index] && score[index] > 0.F)
                    weights.emplace_back(index, score[index]);
            for (const auto index : weighted_unique_sample(
                     weights, additions - split_parents.size(), true,
                     random))
                split_parents.push_back(index);
        }
        if (!split_parents.empty()) {
            std::vector<int> rows;
            rows.reserve(split_parents.size());
            for (const std::size_t parent : split_parents)
                rows.push_back(static_cast<int>(parent));
            grow_training_model(
                model, rows, detail::SplitMode::long_axis, options,
                random, states, {}, split_k);
            grown = rows.size();
        }
    }

    stats = detail::make_densification_stats(model.size());
    core::Logger::instance().info(
        "emc_refine iteration=", iteration,
        " split_k=", split_k,
        " dead=", dead_count,
        " relocated=", relocated,
        " oversize_grown=", oversize_grown,
        " grown=", grown,
        " gaussians=", model.size());
    return {grown, 0};
}

RefinementCounts refine_gaussians(
    GaussianModel& model, detail::DensificationStats& stats,
    const unsigned iteration, const float scene_extent,
    const mvs::Vec3f& scene_center, const TrainingOptions& options,
    std::mt19937& random, const AdamStates& states) {
    if (!is_refinement_iteration(iteration, options) || model.size() == 0)
        return {};

    if (options.densification_strategy == DensificationStrategy::adc_plus)
        return AdcPlusStrategy{}.refine(
            model, stats, iteration, scene_extent, scene_center,
            options, states);
    if (options.densification_strategy == DensificationStrategy::emc)
        return refine_emc(model, stats, iteration, options, random, states);

    const std::size_t old_count = model.size();
    const auto gradients = download<float>(stats.gradient);
    const auto counts = download<float>(stats.count);
    const auto screen = download<float>(stats.max_screen_radius);
    const auto priorities = download<float>(stats.priority);
    const bool error_guided = options.densification_strategy == DensificationStrategy::adc_igs &&
        options.densify_use_error_map && options.densify_error_map_weight != 0.F;
    const auto image_error = error_guided ? download<float>(stats.image_error) : std::vector<float>{};
    const auto view_support = error_guided ? download<float>(stats.view_support) : std::vector<float>{};
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
    std::vector<float> positive_priorities;
    for (std::size_t index = 0; index < old_count; ++index) {
        if (remap[index] < 0 || counts[index] <= 0.F) continue;
        candidates.push_back({
            index, static_cast<std::size_t>(remap[index]), gradients[index],
            screen[index] > options.densify_screen_threshold});
        const float priority = priorities[index] /
                               std::max(counts[index], 1.F);
        if (priority > 0.F) positive_priorities.push_back(priority);
    }
    select_training_rows(model, keep, states);

    const std::size_t capacity = options.densification_cap > model.size()
        ? options.densification_cap - model.size()
        : 0;
    if (capacity == 0) {
        if (options.densification_strategy == DensificationStrategy::adc_igs) {
            const float remaining_progress = 1.F -
                static_cast<float>(iteration) /
                    std::max(1.F, static_cast<float>(options.iterations));
            detail::apply_adc_decay(
                model,
                options.opacity_decay *
                    std::max(remaining_progress, 0.F),
                // Optional scale decay; zero by default (matching Brush).
                options.scale_decay *
                    std::max(remaining_progress, 0.F));
        }
        stats = detail::make_densification_stats(model.size());
        return {0, pruned};
    }

    std::vector<int> split_parents;
    std::vector<std::pair<std::size_t, float>> replacement_weights;
    std::vector<std::pair<std::size_t, float>> growth_weights;
    std::unordered_set<std::size_t> forced;
    const bool adc = options.densification_strategy ==
                     DensificationStrategy::adc_igs;
    float priority_median = 1.F;
    if (!positive_priorities.empty()) {
        const auto middle = positive_priorities.begin() +
            static_cast<std::ptrdiff_t>(positive_priorities.size() / 2);
        std::nth_element(
            positive_priorities.begin(), middle,
            positive_priorities.end());
        priority_median = std::max(*middle, 1e-9F);
    }
    const bool allow_growth = !adc || iteration < options.grow_stop_iter;
    // One candidate's mean error over the observations that saw it. The
    // accumulated value is a contribution-weighted average per view, so the
    // division is the temporal average over those observations.
    const auto sample_error = [&](const std::size_t index) {
        return image_error[index] / std::max(counts[index], 1.F);
    };
    std::vector<float> candidate_errors;
    if (error_guided && allow_growth)
        for (const auto& candidate : candidates) {
            const auto i = candidate.old_index;
            if (candidate.score <= options.densify_gradient_threshold || view_support[i] < 2.F)
                continue;
            const float error = sample_error(i);
            if (std::isfinite(error) && error > 0.F) candidate_errors.push_back(error);
        }
    float error_median = 0.F;
    if (!candidate_errors.empty()) {
        auto middle = candidate_errors.begin() + candidate_errors.size() / 2;
        std::nth_element(candidate_errors.begin(), middle, candidate_errors.end());
        error_median = *middle;
    }
    // Selected-error diagnostics: the ratio between the median error of the
    // parents that actually spent the growth budget and the candidate median.
    // One means the error evidence did not move the choice at all.
    std::vector<float> error_by_row(error_guided ? model.size() : 0, 0.F);
    std::size_t error_reweighted = 0;
    for (const Candidate& candidate : candidates) {
        const float opacity = 1.F /
            (1.F + std::exp(-opacities[candidate.old_index]));
        float edge_factor = 1.F;
        if (adc) {
            const float priority = priorities[candidate.old_index] /
                std::max(counts[candidate.old_index], 1.F);
            if (priority > 0.F)
                edge_factor += 0.25F * std::min(
                    priority / priority_median, 10.F);
        }
        const float screen_factor = candidate.oversized ? 2.F : 1.F;
        replacement_weights.emplace_back(
            candidate.new_index,
            adc ? opacity * edge_factor
                : std::max(candidate.score, 1e-12F) * screen_factor);
        if (allow_growth &&
            candidate.score > options.densify_gradient_threshold) {
            float error_factor = 1.F;
            if (error_guided && view_support[candidate.old_index] >= 2.F &&
                error_median > 0.F) {
                const float error = sample_error(candidate.old_index);
                error_by_row[candidate.new_index] = error;
                error_factor = error_map_sampling_factor(
                    error, error_median, options.densify_error_map_weight);
                ++error_reweighted;
            }
            growth_weights.emplace_back(
                candidate.new_index,
                candidate.score * edge_factor * screen_factor * error_factor);
        }
        if (adc && allow_growth && candidate.oversized)
            forced.insert(candidate.new_index);
    }
    const bool gumbel = adc;
    auto selected = weighted_unique_sample(
        replacement_weights, std::min(pruned, capacity), gumbel, random);
    forced.insert(selected.begin(), selected.end());
    std::size_t desired_growth = static_cast<std::size_t>(std::llround(
        growth_weights.size() * options.densify_select_fraction));
    if (!adc) {
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
    }
    const std::size_t remaining = capacity > forced.size()
        ? capacity - forced.size()
        : 0;
    selected = weighted_unique_sample(
        growth_weights, std::min(remaining, desired_growth), gumbel, random);
    forced.insert(selected.begin(), selected.end());
    float selected_error_ratio = 0.F;
    if (error_guided && !selected.empty()) {
        std::vector<float> chosen;
        chosen.reserve(selected.size());
        for (const std::size_t index : selected)
            if (error_by_row[index] > 0.F)
                chosen.push_back(error_by_row[index]);
        if (!chosen.empty()) {
            auto middle = chosen.begin() + chosen.size() / 2;
            std::nth_element(chosen.begin(), middle, chosen.end());
            selected_error_ratio = *middle / error_median;
        }
    }
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
        model, split_parents,
        adc ? detail::SplitMode::igs_random
            : detail::SplitMode::dense_tangent,
        options, random, states,
        retained_screen);
    if (adc) {
        const float remaining_progress = 1.F -
            static_cast<float>(iteration) /
                std::max(1.F, static_cast<float>(options.iterations));
        detail::apply_adc_decay(
            model,
            options.opacity_decay *
                std::max(remaining_progress, 0.F),
            // Optional scale decay; zero by default (matching Brush).
            options.scale_decay *
                std::max(remaining_progress, 0.F));
    }
    stats = detail::make_densification_stats(model.size());
    if (error_guided)
        core::Logger::instance().info("igs_error_refine iteration=", iteration,
            " weight=", options.densify_error_map_weight,
            " candidates=", growth_weights.size(),
            " reweighted=", error_reweighted,
            " error_median=", error_median,
            " selected_error_ratio=", selected_error_ratio,
            " grown=", split_parents.size(),
            " pruned=", pruned, " gaussians=", model.size());
    return {split_parents.size(), pruned};
}

}  // namespace aetherscan::splat::densification
