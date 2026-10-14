#pragma once

#include <ceres/solver.h>
#include <algorithm>
#include <cmath>

namespace photara::sfm::detail {

struct PositioningPassProgress {
    unsigned accepted_updates{};
    bool converged{};
    bool time_limited{};
    double minimum_next_budget{};
};

inline PositioningPassProgress positioning_pass_progress(
    const ceres::Solver::Summary& summary, double budget) {
    PositioningPassProgress progress;
    for (const auto& iteration : summary.iterations)
        if (iteration.iteration > 0 && iteration.step_is_successful)
            ++progress.accepted_updates;
    progress.converged = summary.termination_type == ceres::CONVERGENCE;
    progress.time_limited = summary.termination_type == ceres::NO_CONVERGENCE &&
        std::isfinite(budget) && budget > 0 && summary.total_time_in_seconds >= budget;
    const auto attempts = std::max<std::size_t>(1, summary.iterations.size() > 1
        ? summary.iterations.size() - 1 : 0);
    const double step = (std::max(0.0, summary.linear_solver_time_in_seconds) +
                         std::max(0.0, summary.jacobian_evaluation_time_in_seconds)) /
                        static_cast<double>(attempts);
    progress.minimum_next_budget = 1.2 *
        (std::max(0.0, summary.preprocessor_time_in_seconds) + step) + 0.01;
    return progress;
}

// Reserve measured preprocessing plus at least one attempted update instead
// of repeatedly assigning slices that can only evaluate the starting point.
inline double positioning_pass_budget(double remaining, unsigned remaining_passes,
                                      bool first, double minimum_useful_budget) {
    const double share = first ? 0.6 * remaining
        : remaining / static_cast<double>(std::max(1U, remaining_passes));
    return std::min(remaining, std::max({0.05, share, minimum_useful_budget}));
}

inline bool can_retry_positioning_pass(double remaining,
                                      const PositioningPassProgress& progress) {
    return progress.time_limited && !progress.converged &&
        progress.accepted_updates == 0 && std::isfinite(remaining) &&
        remaining >= progress.minimum_next_budget && remaining > 0.05;
}

} // namespace photara::sfm::detail
