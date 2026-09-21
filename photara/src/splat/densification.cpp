#include "densification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace photara::splat::densification {

SceneGeometry training_scene_geometry(
    const mvs::MvsScene& scene, const bool dense_input) {
    mvs::Vec3f minimum = scene.dense_cloud.points.front().position;
    mvs::Vec3f maximum = minimum;
    for (const auto& point : scene.dense_cloud.points) {
        minimum = minimum.cwiseMin(point.position);
        maximum = maximum.cwiseMax(point.position);
    }
    SceneGeometry geometry{
        std::max((maximum - minimum).norm(), 1e-6F),
        0.5F * (minimum + maximum),
        0.5F * (maximum - minimum).maxCoeff()};
    if (scene.views.empty()) return geometry;

    // Optimization scale is measured from camera centers (COLMAP convention),
    // not the untrimmed sparse-point AABB. A handful of bad triangulations
    // can make that AABB tens of times too large, which in turn inflates both
    // the means LR and the minimum Gaussian scale.
    mvs::Vec3f camera_center = mvs::Vec3f::Zero();
    std::size_t finite_cameras = 0;
    for (const auto& view : scene.views) {
        const mvs::Vec3f position = view.pose.C.cast<float>();
        if (!position.allFinite()) continue;
        camera_center += position;
        ++finite_cameras;
    }
    if (finite_cameras == 0) {
        if (dense_input) return geometry;
        throw std::invalid_argument(
            "Sparse GGGS training requires at least one finite camera center");
    }
    camera_center /= static_cast<float>(finite_cameras);
    float camera_radius = 0.F;
    for (const auto& view : scene.views) {
        const mvs::Vec3f position = view.pose.C.cast<float>();
        if (position.allFinite())
            camera_radius = std::max(
                camera_radius, (position - camera_center).norm());
    }
    if (!(camera_radius > 1e-6F) || !std::isfinite(camera_radius)) {
        if (dense_input) return geometry;
        throw std::invalid_argument(
            "Sparse GGGS training requires non-degenerate camera coverage");
    }
    return {1.1F * camera_radius, camera_center, 1.1F * camera_radius};
}

SceneGeometry brush_scene_geometry(
    const std::vector<float>& xyz, const float percentile) {
    std::array<std::vector<float>, 3> axes;
    const std::size_t count = xyz.size() / 3;
    for (auto& values : axes) values.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
        for (std::size_t axis = 0; axis < axes.size(); ++axis) {
            const float value = xyz[3 * index + axis];
            if (std::isfinite(value)) axes[axis].push_back(value);
        }

    if (std::any_of(
            axes.begin(), axes.end(),
            [](const auto& values) { return values.empty(); }))
        return {2.F, mvs::Vec3f::Zero(), 1.F};

    const float p = std::clamp(percentile, 0.F, 1.F);
    mvs::Vec3f minimum;
    mvs::Vec3f maximum;
    for (std::size_t axis = 0; axis < axes.size(); ++axis) {
        auto& values = axes[axis];
        const std::size_t size = values.size();
        const std::size_t low = static_cast<std::size_t>(
            (1.F - p) * 0.5F * static_cast<float>(size));
        const std::size_t high = std::min(
            size - 1,
            static_cast<std::size_t>(
                (1.F + p) * 0.5F * static_cast<float>(size)));
        // Only two order statistics are needed. Sorting all N coordinates
        // on the training thread stalls CUDA at every ADC refinement.
        std::nth_element(values.begin(), values.begin() + high, values.end());
        maximum(static_cast<Eigen::Index>(axis)) = values[high];
        if (low < high)
            std::nth_element(
                values.begin(), values.begin() + low, values.begin() + high);
        minimum(static_cast<Eigen::Index>(axis)) = values[low];
    }

    const mvs::Vec3f half_extent = 0.5F * (maximum - minimum);
    std::array<float, 3> extents{
        half_extent.x(), half_extent.y(), half_extent.z()};
    std::sort(extents.begin(), extents.end());
    return {
        2.F * extents[1],
        0.5F * (minimum + maximum),
        extents[2]};
}

SceneGeometry brush_scene_geometry_cuda(
    const tinytensor::Tensor& means, const float percentile) {
    const auto bounds = detail::percentile_bounds(means, percentile);
    const mvs::Vec3f minimum(bounds[0], bounds[1], bounds[2]);
    const mvs::Vec3f maximum(bounds[3], bounds[4], bounds[5]);
    const mvs::Vec3f half_extent = 0.5F * (maximum - minimum);
    std::array<float, 3> extents{
        half_extent.x(), half_extent.y(), half_extent.z()};
    std::sort(extents.begin(), extents.end());
    return {2.F * extents[1], 0.5F * (minimum + maximum), extents[2]};
}

StrategySchedule strategy_schedule(const TrainingOptions& options) {
    const bool igs =
        options.densification_strategy == DensificationStrategy::adc_igs;
    const bool adc = is_adc_strategy(options.densification_strategy);
    const bool emc =
        options.densification_strategy == DensificationStrategy::emc;
    const unsigned stop_iter = options.refine_stop_iter != 0
        ? options.refine_stop_iter
        : (igs || emc) ? 14'000U : adc ? options.iterations : 5'000U;
    const unsigned stop_from_end =
        options.iterations > options.refine_stop_num_iter
            ? options.iterations - options.refine_stop_num_iter
            : 0U;
    const unsigned stop = (igs || emc)
        ? std::min(std::max(stop_iter, stop_from_end), options.iterations)
        : std::min(stop_iter, options.iterations);
    return {
        options.refine_start_iter != 0
            ? options.refine_start_iter
            : emc ? 500U : adc ? 0U : 750U,
        stop,
        options.refine_every != 0
            ? options.refine_every
            : emc ? 100U : adc ? 200U : 500U};
}

GaussianModel clone_model(const GaussianModel& model) {
    GaussianModel cloned;
    cloned.means = model.means.clone();
    cloned.log_scales = model.log_scales.clone();
    cloned.quaternions = model.quaternions.clone();
    cloned.opacity_logits = model.opacity_logits.clone();
    cloned.sh = model.sh.clone();
    if (model.normal_features.is_valid())
        cloned.normal_features = model.normal_features.clone();
    if (model.filter_3d.is_valid())
        cloned.filter_3d = model.filter_3d.clone();
    cloned.sh_degree = model.sh_degree;
    return cloned;
}

bool is_enabled(const TrainingOptions& options) {
    const bool dense_adaptive =
        options.densification_strategy ==
        DensificationStrategy::dense_adaptive;
    return options.enable_densification &&
        ((!options.input_is_dense && !dense_adaptive) ||
         (options.input_is_dense && dense_adaptive)) &&
        strategy_schedule(options).stop > 0;
}

bool is_refinement_iteration(
    const unsigned iteration, const TrainingOptions& options) {
    const StrategySchedule schedule = strategy_schedule(options);
    if (iteration <= schedule.start || iteration >= schedule.stop ||
        schedule.every == 0 || iteration % schedule.every != 0)
        return false;
    if (options.densification_strategy == DensificationStrategy::adc_igs ||
        options.densification_strategy == DensificationStrategy::emc)
        return true;
    return static_cast<float>(iteration) /
           std::max(1.F, static_cast<float>(options.iterations)) <=
        0.95F;
}

}  // namespace photara::splat::densification

namespace photara::splat {

void apply_strategy_defaults(TrainingOptions& options) {
    const bool emc = options.densification_strategy ==
        DensificationStrategy::emc;
    // Brush stops growth at half of total_train_iters (its 15000 of 30000) and
    // spends the rest on prune/replace/oversize split/decay. Measured on the
    // ori statue set at 10k iterations (seed 42), stopping at 50% moved
    // held-out PSNR by -0.2 dB and training-view PSNR by -3.0 dB
    // (artifacts/ori_igs_device_20260918/growstop): that is the trade the
    // Brush cadence makes on short runs.
    options.grow_stop_iter = grow_stop_iteration(options);
    if (is_adc_strategy(options.densification_strategy) || emc) {
        // Shared training parameters. Apply explicit caller overrides
        // after this preset; dense input keeps its separate optimizer setup.
        options.densify_gradient_threshold = 0.0025F;
        options.densify_select_fraction = 0.25F;
        options.densify_screen_threshold = 0.5F;
        if (!options.input_is_dense) {
            options.means_lr = 2e-5F;
            options.opacities_lr = 0.012F;
            options.initial_opacity = 0.5F;
        }
    }
    switch (options.densification_strategy) {
    case DensificationStrategy::adc_plus:
        break;
    case DensificationStrategy::adc_igs:
        // ADC-IGS decides growth with the projected-position gradient gate,
        // opacity/edge evidence sampling, gumbel selection, the random-axis
        // exact-alpha split and the default cadence. Neither ADC-IGS nor ADC+
        // scores by image error: the 2026-09-17 sweep measured the re-order
        // inside the ori seed noise on every dataset
        // (docs/ADC_IGS_ERROR_MAP_20260917), so the error map stays EMC's
        // strategy.
        break;
    case DensificationStrategy::emc:
        // Fixed-budget growth: net growth is a fixed multiplier of the
        // live count per refinement, and a fixed share of that budget is
        // reserved for accumulated on-screen oversize repair. The error map
        // is the strategy itself, so no flag is required.
        options.densify_growth_factor = 1.05F;
        options.densify_oversize_split_fraction = 0.15F;
        options.densify_oversize_score_blend = 1.F;
        // The log-scale shrink pressure from the reference MCMC family is
        // disabled: with our per-pixel loss normalization and Adam it
        // systematically under-sizes splats and cost 0.57 dB on the office
        // protocol (artifacts/office_emc_v3*_20260918). The erank and
        // quaternion pulls measured neutral-to-positive and stay on.
        options.shape_scale_reg = 0.F;
        options.shape_erank_reg = 0.001F;
        options.shape_quat_norm_reg = 0.01F;
        options.oversize_screen_limit = 0.3F;
        options.oversize_penalty = 1.F;
        break;
    case DensificationStrategy::dense_adaptive:
        break;
    }
}

}  // namespace photara::splat
