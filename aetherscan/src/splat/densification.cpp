#include "densification.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace aetherscan::splat::densification {

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
        0.5F * (minimum + maximum)};
    if (scene.views.empty()) return geometry;

    // Match pygsplat's COLMAP convention: optimization scale is measured from
    // camera centers, not the untrimmed sparse-point AABB. A handful of bad
    // triangulations can make that AABB tens of times too large, which in turn
    // inflates both the means LR and the minimum Gaussian scale.
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
    return {1.1F * camera_radius, camera_center};
}

StrategySchedule strategy_schedule(const TrainingOptions& options) {
    const bool dense_adaptive = options.densification_strategy ==
                                DensificationStrategy::dense_adaptive;
    const bool adc = options.densification_strategy ==
                         DensificationStrategy::adc_plus ||
                     options.densification_strategy ==
                         DensificationStrategy::adc_igs;
    const unsigned preset_stop = dense_adaptive
        ? 5'000U
        : options.densification_strategy == DensificationStrategy::adc_plus
            ? options.iterations
        : options.densification_strategy == DensificationStrategy::adc_igs
            ? 25'000U
            : 15'000U;
    return {
        options.refine_start_iter != 0
            ? options.refine_start_iter
            : dense_adaptive ? 750U
                             : options.densification_strategy ==
                                       DensificationStrategy::adc_plus
                                 ? 0U
                                 : adc ? 600U : 500U,
        std::min(
            options.refine_stop_iter != 0
                ? options.refine_stop_iter
                : preset_stop,
            options.iterations),
        options.refine_every != 0
            ? options.refine_every
            : dense_adaptive ? 500U : adc ? 200U : 100U};
}

GaussianModel clone_model(const GaussianModel& model) {
    GaussianModel cloned;
    cloned.means = model.means.clone();
    cloned.log_scales = model.log_scales.clone();
    cloned.quaternions = model.quaternions.clone();
    cloned.opacity_logits = model.opacity_logits.clone();
    cloned.sh = model.sh.clone();
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
    return iteration > schedule.start && iteration < schedule.stop &&
        schedule.every != 0 && iteration % schedule.every == 0;
}

}  // namespace aetherscan::splat::densification
