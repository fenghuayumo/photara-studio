#include "aetherscan/ba/linearizer.hpp"

#include "reprojection_detail.cuh"

#include <chrono>
#include <cstdint>

namespace aetherscan::ba {

EvaluationStats linearize_cpu(
    const Problem& problem,
    LinearizationOutput& output,
    const LinearizerOptions& options) {
    problem.validate();
    const auto count = problem.observations.size();
    output.resize(count);

    const auto started = std::chrono::steady_clock::now();
#if defined(AETHERSCAN_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) {
        const auto observation = static_cast<std::size_t>(i);
        const auto camera_index = problem.observations.camera[observation];
        const auto point_index = problem.observations.point[observation];
        detail::linearize_observation(
            problem.poses[camera_index],
            problem.intrinsics[camera_index],
            problem.points[point_index],
            problem.observations.x[observation],
            problem.observations.y[observation],
            problem.observations.weight[observation],
            options,
            output.observations[observation]);
    }
    const auto stopped = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double, std::milli>(stopped - started);
    return EvaluationStats{elapsed.count(), count};
}

}  // namespace aetherscan::ba
