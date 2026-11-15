#pragma once
#include "sfm/bundle.hpp"
#include "sfm/global_positioning.hpp"
#include "ba/solver_backend.hpp"
#include <functional>
namespace photara::sfm::detail {
void solve_bundle(
    ba::Problem& problem,
    const ba::OptimizerOptions& opt,
    const BundleOptions& options,
    BundleBackendPreference backend,
    bool supported_parameterization,
    bool has_partial_pose_locks,
    BundleSummary& summary);
struct PositioningSolver {
    std::unique_ptr<ba::BearingSolver> solver;
    const char* name{"cpu"};
};
PositioningSolver make_positioning_solver(
    const GlobalPositioningOptions& options,
    unsigned observations,
    unsigned cameras,
    const std::function<const ba::BearingProblem&()>& prepare);
bool solve_positioning_pass(
    PositioningSolver& selected,
    ba::BearingProblem& problem,
    const std::vector<double>& weights,
    unsigned iterations,
    double tolerance,
    double seconds,
    unsigned pass,
    ba::BearingSolveSummary& step);
} // namespace photara::sfm::detail
