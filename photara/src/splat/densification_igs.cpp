#include "densification_igs.hpp"

#include "core/logging.hpp"
#include "densification_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#ifdef TINYTENSOR_HAS_VULKAN
#include "vulkan/backend.hpp"
#include "vulkan/ops.hpp"
#endif

namespace photara::splat::densification {
namespace {

// Wall-clock stopwatch for the optional refinement profile. Each selection
// stage ends with a host readback, so the interesting cost is the total
// (recording + queue drain + wait), not the dispatch time alone.
class PhaseClock {
public:
    void start() { started_ = std::chrono::steady_clock::now(); }
    double split() {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double, std::milli>(
            now - started_).count();
        started_ = now;
        return elapsed;
    }

private:
    std::chrono::steady_clock::time_point started_{
        std::chrono::steady_clock::now()};
};

// Device counters sampled alongside the wall clock. A phase that reports more
// wait than wall time is impossible; a phase whose wait_ms covers most of its
// wall time was stalled draining an earlier phase's recorded work, and a phase
// with many dispatches but little wait is CPU-bound in command recording.
struct DeviceCounters {
    double wait_ms{};
    std::uint64_t dispatches{};
};

DeviceCounters device_counters() {
#ifdef TINYTENSOR_HAS_VULKAN
    if (tinytensor::vulkan::available())
        return {tinytensor::vulkan::device_wait_ms(),
                tinytensor::vulkan::dispatch_count()};
#endif
    return {};
}

// Device busy milliseconds, from the timestamp profiler. A phase's wall time
// splits into this, the fence wait reported by device_counters(), and the host
// remainder - command recording, tensor plumbing, driver allocation - which is
// the number that says whether the next fix belongs in a shader or in the
// runtime.
double device_busy_ms() {
#ifdef TINYTENSOR_HAS_VULKAN
    if (tinytensor::vulkan::available()) return tinytensor::vulkan::device_busy_ms();
#endif
    return 0.0;
}

// "Sort `values` in the requested order and keep the front `count` of the rows
// `mask` selects." Vulkan resolves it with radix selection - three dispatches
// and no sort - which returns the same set the bitonic sort produced and caps
// itself at the number of qualifying rows, so the caller never needs a
// separate count. Every other backend keeps the host sort it has always used,
// because that is the rule the CUDA path selects its parents with.
tinytensor::Tensor select_front(
    const tinytensor::Tensor& values, const tinytensor::Tensor& mask,
    const std::size_t count, const bool descending,
    std::size_t* selected = nullptr) {
    if (selected != nullptr) *selected = 0;
    if (count == 0 || values.numel() == 0) return {};
#ifdef TINYTENSOR_HAS_VULKAN
    // SPLAT_IGS_LEGACY_SELECT=1 forces the sort path the radix selection
    // replaced, so an A/B on one binary can prove the replacement exact.
    if (values.device() == tinytensor::Device::Vulkan &&
        std::getenv("SPLAT_IGS_LEGACY_SELECT") == nullptr) {
        std::uint32_t emitted = 0;
        auto picked = tinytensor::vulkan::select_topk_indices(
            values, mask, static_cast<std::uint32_t>(count), descending, &emitted);
        if (selected != nullptr) *selected = emitted;
        return picked;
    }
#endif
    const float sentinel = descending
        ? -std::numeric_limits<float>::infinity()
        : std::numeric_limits<float>::infinity();
    const auto ranked = values.masked_fill(mask.logical_not(), sentinel)
                            .sort(0, descending);
    const std::size_t found = std::min(count, mask.count_nonzero());
    if (selected != nullptr) *selected = found;
    if (found == 0) return {};
    return ranked.second.slice(0, 0, found).to(tinytensor::DataType::Int32);
}

// Device form of the same selection restricted to one order statistic. The
// caller only needs the value, never the indices.
bool device_select_nth(
    const tinytensor::Tensor& values, const tinytensor::Tensor& mask,
    const std::size_t rank, const bool descending, float* value) {
    if (values.device() != tinytensor::Device::Vulkan || rank == 0) return false;
    if (std::getenv("SPLAT_IGS_LEGACY_SELECT") != nullptr) return false;
#ifdef TINYTENSOR_HAS_VULKAN
    return tinytensor::vulkan::select_nth_value(
        values, mask, static_cast<std::uint32_t>(rank), descending, value);
#else
    (void)value;
    return false;
#endif
}

// Upper median of the entries selected by `positive`, computed on the device:
// the host rule was nth_element at position size/2 over the positive subset,
// which is the same order statistic. Unselected entries are pushed to +inf so
// they cannot enter the prefix, and NaN inputs (never positive, never finite)
// are replaced on the way in.
float positive_upper_median(
    const tinytensor::Tensor& values, const tinytensor::Tensor& positive,
    const std::size_t positive_count) {
    if (positive_count == 0) return 0.F;
    // Order statistic, not a sort: the device radix select resolves the exact
    // (middle + 1)-th smallest positive value in three dispatches where the
    // bitonic sort took two hundred.
    float selected = 0.F;
    if (device_select_nth(values, positive, positive_count / 2 + 1, false,
                          &selected)) {
        return selected;
    }
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

tinytensor::Tensor weighted_gumbel_sample(
    const tinytensor::Tensor& weights, const std::size_t requested,
    std::mt19937& random, double* count_ms = nullptr,
    double* sort_ms = nullptr) {
    if (requested == 0 || weights.numel() == 0) return {};
    if (weights.device() == tinytensor::Device::Vulkan) {
        // Keep the million-row selection on the GPU. Downloading all weights
        // for a host partial_sort left the queue idle at every IGS refinement.
        const auto eligible = weights.isfinite().logical_and(weights.gt(0.F));
        PhaseClock clock;
        clock.start();
        const auto uniform = tinytensor::Tensor::rand(
            weights.shape(), weights.device()).clamp_min(1e-7F).clamp_max(1.F - 1e-7F);
        auto scores = weights.log().sub(uniform.log().mul(-1.F).log());
        if (count_ms) *count_ms += clock.split();
        // The mask replaces the -inf fill: only the eligible scores can come
        // back, and the op already caps the selection at the eligible count.
        auto picked = select_front(scores, eligible, requested, true);
        if (sort_ms) *sort_ms += clock.split();
        return picked;
    }
    const auto values = weights.to_vector();
    std::uniform_real_distribution<float> uniform(1e-7F, 1.F - 1e-7F);
    std::vector<std::pair<float, int>> scores;
    scores.reserve(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        const float weight = values[index];
        if (!std::isfinite(weight) || !(weight > 0.F)) continue;
        const float u = uniform(random);
        scores.emplace_back(
            std::log(weight) - std::log(-std::log(u)),
            static_cast<int>(index));
    }
    const std::size_t count = std::min(requested, scores.size());
    if (count == 0) return {};
    std::partial_sort(scores.begin(), scores.begin() +
        static_cast<std::ptrdiff_t>(count), scores.end(), std::greater<>());
    std::vector<int> indices(count);
    for (std::size_t i = 0; i < count; ++i) indices[i] = scores[i].second;
    return tinytensor::Tensor::from_vector(
        indices, {count}, weights.device());
}

}  // namespace

IgsSelection select_igs_parents(
    const tinytensor::Tensor& replacement_weights,
    const tinytensor::Tensor& oversize_scores,
    const tinytensor::Tensor& growth_weights,
    const std::size_t replacement_slots, const std::size_t desired_growth,
    const std::size_t capacity, std::mt19937* random,
    IgsSelectionProfile* profile) {
    IgsSelection result;
    const auto device = replacement_weights.device();
    std::mt19937 fallback_random(0);
    std::mt19937& selection_random = random ? *random : fallback_random;
    auto chosen = tinytensor::Tensor::zeros_bool(
        {replacement_weights.numel()}, device);
    PhaseClock clock;
    clock.start();
    const auto replacement = weighted_gumbel_sample(
        replacement_weights, std::min(replacement_slots, capacity),
        selection_random, profile ? &profile->replacement_count_ms : nullptr,
        profile ? &profile->replacement_sort_ms : nullptr);
    if (profile) profile->replacement_gumbel_ms = clock.split();
    result.replacement = replacement.numel();
    if (result.replacement) chosen.index_fill_(0, replacement, 1.F);
    clock.start();
    std::size_t remaining = capacity - result.replacement;
    if (remaining && oversize_scores.is_valid()) {
        const auto eligible = oversize_scores.isfinite()
            .logical_and(oversize_scores.gt(0.F)).logical_and(chosen.logical_not());
        // Rank actual severity, never the storage position. Equal scores are
        // equivalent; no later truncation can evict replacements.
        auto ranked = select_front(oversize_scores, eligible, remaining, true,
                                   &result.oversized);
        if (result.oversized) {
            chosen.index_fill_(0, ranked, 1.F);
            remaining -= result.oversized;
        }
    }
    if (profile) profile->oversized_rank_ms = clock.split();
    clock.start();
    if (remaining && desired_growth && growth_weights.is_valid()) {
        const auto growth = weighted_gumbel_sample(
            growth_weights.masked_fill(chosen, 0.F),
            std::min(remaining, desired_growth), selection_random,
            profile ? &profile->growth_count_ms : nullptr,
            profile ? &profile->growth_sort_ms : nullptr);
        result.growth = growth.numel();
        if (result.growth) chosen.index_fill_(0, growth, 1.F);
    }
    if (profile) profile->growth_gumbel_ms = clock.split();
    clock.start();
    result.parents = chosen.nonzero().squeeze(1).to(tinytensor::DataType::Int32);
    if (profile) profile->compact_ms = clock.split();
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
    const bool profile_refine = std::getenv("SPLAT_IGS_PROFILE") != nullptr;
    const DeviceCounters refine_start_counters =
        profile_refine ? device_counters() : DeviceCounters{};
    auto phase_started = std::chrono::steady_clock::now();
    const auto phase_ms = [&phase_started]() {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double, std::milli>(
            now - phase_started).count();
        phase_started = now;
        return elapsed;
    };
    double phase_busy = profile_refine ? device_busy_ms() : 0.0;
    const auto phase_gpu = [&phase_busy, profile_refine]() {
        if (!profile_refine) return 0.0;
        const double now = device_busy_ms();
        const double delta = now - phase_busy;
        phase_busy = now;
        return delta;
    };

    // The host path clips oversize splats before it reads any statistics, so
    // the mask below already sees the clipped scales.
    if (oversize_penalty_at(options, iteration) > 0.F)
        detail::clip_log_scale_by_screen(
            model.log_scales, stats.max_screen_radius,
            options.oversize_screen_limit, options.oversize_clip_hardness);

    auto masks = detail::prune_masks(
        model, options.prune_opacity, 100.F * scene_extent,
        {scene_center.x(), scene_center.y(), scene_center.z()});
    const double masks_ms = profile_refine ? phase_ms() : 0.0;
    const double masks_gpu = phase_gpu();
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
        // Recycling is the `opacity_budget` least opaque rows below the floor;
        // the op caps itself at the rows that exist, which is the count the
        // old `min(opacity_budget, low_count)` read back for.
        std::size_t drop = 0;
        auto recycled = select_front(masks.opacities, recycled_rows,
                                     opacity_budget, false, &drop);
        if (drop != 0) {
            // The released rows never re-enter `keep`; clearing them here
            // replaces the mask-and-negate round trip the host rule needed.
            keep.index_fill_(0, recycled, 0.F);
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
        {old_count}, model.means.device());
    if (best.numel() != 0) best_mask.index_fill_(0, best, 1.F);
    keep = keep.logical_or(best_mask.logical_and(masks.hard.logical_not()));
    const double recycle_ms = profile_refine ? phase_ms() : 0.0;
    const double recycle_gpu = phase_gpu();

    std::size_t retained = keep.count_nonzero();
    if (retained > options.densification_cap) {
        // Cap: drop the least opaque survivors, never the rescued row.
        // Keeping `best_mask` out of the candidate set is the same protection
        // the removed `best_kept` readback provided, without the round trip.
        const auto removable = keep.logical_and(best_mask.logical_not());
        std::size_t remove = 0;
        auto clamped = select_front(masks.opacities, removable,
                                    retained - options.densification_cap, false,
                                    &remove);
        if (remove != 0) keep.index_fill_(0, clamped, 0.F);
    }

    auto keep_indices = keep.nonzero().squeeze(1).to(
        tinytensor::DataType::Int32);
    retained = keep_indices.numel();
    const std::size_t pruned = old_count - retained;
    const auto retained_screen =
        stats.max_screen_radius.index_select(0, keep_indices);
    const double prune_ms = profile_refine ? phase_ms() : 0.0;
    const double prune_gpu = phase_gpu();
    gpu_detail::select_training_rows_gpu(model, keep_indices, states);
    const double remap_ms = profile_refine ? phase_ms() : 0.0;
    const double remap_gpu = phase_gpu();

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
        stats = detail::make_densification_stats(
            model.size(), model.means.device());
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
    const double stats_ms = profile_refine ? phase_ms() : 0.0;
    const double stats_gpu = phase_gpu();
    float priority_median = 1.F;
    if (positive_count != 0)
        priority_median = std::max(
            positive_upper_median(priority, positive, positive_count), 1e-9F);
    const double median_ms = profile_refine ? phase_ms() : 0.0;
    const double median_gpu = phase_gpu();
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
    std::size_t growth_candidates = 0;
    if (allow_growth) {
        const auto growth_mask = candidate.logical_and(
            retained_gradient.gt(options.densify_gradient_threshold));
        growth_candidates = growth_mask.count_nonzero();
        // On-screen oversize rows are sampled twice as often as the rest.
        const auto screen_factor = tinytensor::Tensor::full(
            {retained}, 1.F, model.means.device())
            .masked_fill(oversized, 2.F);
        growth_weights = retained_gradient.mul(edge_factor)
            .mul(screen_factor)
            .masked_fill(growth_mask.logical_not(), 0.F);
        desired_growth = static_cast<std::size_t>(std::llround(
            static_cast<double>(growth_candidates) *
            static_cast<double>(options.densify_select_fraction)));
    }

    const std::size_t oversized_count =
        allow_growth ? oversized.count_nonzero() : 0;
    const auto oversize_scores = allow_growth
        ? retained_screen.mul(edge_factor).masked_fill(oversized.logical_not(), 0.F)
        : tinytensor::Tensor{};
    const double edge_ms = profile_refine ? phase_ms() : 0.0;
    const double edge_gpu = phase_gpu();
    IgsSelectionProfile selection_profile;
    const auto selection = select_igs_parents(replacement_weights,
        oversize_scores, growth_weights, pruned, desired_growth, capacity,
        &random, profile_refine ? &selection_profile : nullptr);
    const auto& split_parents = selection.parents;
    const std::size_t grown = split_parents.numel();
    const double selection_ms = profile_refine ? phase_ms() : 0.0;
    const double selection_gpu = phase_gpu();
    gpu_detail::grow_igs_random_gpu(
        model, split_parents, retained_screen, options, random, states);
    decay_adc();
    stats = detail::make_densification_stats(
        model.size(), model.means.device());
    const double growth_ms = profile_refine ? phase_ms() : 0.0;
    const double growth_gpu = phase_gpu();

    if (profile_refine) {
        const DeviceCounters end_counters = device_counters();
        core::Logger::instance().info(
            "igs_profile iteration=", iteration,
            " wait_ms=", end_counters.wait_ms - refine_start_counters.wait_ms,
            " dispatches=", end_counters.dispatches - refine_start_counters.dispatches,
            " masks_ms=", masks_ms,
            " recycle_ms=", recycle_ms,
            " prune_ms=", prune_ms,
            " remap_ms=", remap_ms,
            " selection_ms=", selection_ms,
            " growth_ms=", growth_ms,
            " gpu_mask_recycle_prune=", masks_gpu, "/", recycle_gpu, "/", prune_gpu,
            " gpu_remap_stats=", remap_gpu, "/", stats_gpu,
            " gpu_median_edge_sel_growth=", median_gpu, "/", edge_gpu, "/",
            selection_gpu, "/", growth_gpu,
            " | stats_ms=", stats_ms,
            " median_ms=", median_ms,
            " edge_ms=", edge_ms,
            " repl_ms=", selection_profile.replacement_gumbel_ms,
            " repl_count_ms=", selection_profile.replacement_count_ms,
            " repl_sort_ms=", selection_profile.replacement_sort_ms,
            " over_ms=", selection_profile.oversized_rank_ms,
            " grow_ms=", selection_profile.growth_gumbel_ms,
            " grow_count_ms=", selection_profile.growth_count_ms,
            " grow_sort_ms=", selection_profile.growth_sort_ms,
            " compact_ms=", selection_profile.compact_ms);
    }

    core::Logger::instance().info(
        "igs_refine iteration=", iteration,
        " gaussians=", model.size(),
        " pruned=", pruned,
        " retained=", retained,
        " replacement_selected=", selection.replacement,
        " oversized_candidates=", oversized_count,
        " oversized_selected=", selection.oversized,
        " growth_candidates=", growth_candidates,
        " gradient_threshold=", options.densify_gradient_threshold,
        " growth_selected=", selection.growth,
        " grown=", grown,
        " capacity=", capacity,
        " growth_cap=", growth_cap,
        " growing=", allow_growth);
    return {grown, pruned};
}

}  // namespace photara::splat::densification
