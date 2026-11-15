#include "solver_dispatch.hpp"
#include "core/logging.hpp"
#include <stdexcept>
namespace photara::sfm::detail {
void solve_bundle(
    ba::Problem& problem,
    const ba::OptimizerOptions& opt,
    const BundleOptions& options,
    BundleBackendPreference backend,
    bool supported_parameterization,
    bool has_partial_pose_locks,
    BundleSummary& summary) {
    const auto& cuda = ba::solver_backend(ba::SolverDevice::cuda);
    const auto& vulkan = ba::solver_backend(ba::SolverDevice::vulkan);

    const bool automatic = backend == BundleBackendPreference::automatic;
    const bool want_cuda =
        (automatic && options.prefer_cuda) || backend == BundleBackendPreference::cuda;
    const bool want_vulkan = automatic || backend == BundleBackendPreference::vulkan;
    const bool use_cuda = want_cuda && supported_parameterization &&
                          problem.observations.size() >= options.cuda_min_observations &&
                          cuda.available();

    const bool use_vulkan = want_vulkan && !use_cuda && supported_parameterization &&
                            problem.observations.size() >= options.cuda_min_observations &&
                            vulkan.available();

    const char* requested = backend == BundleBackendPreference::cuda     ? "cuda"
                            : backend == BundleBackendPreference::vulkan ? "vulkan"
                            : backend == BundleBackendPreference::cpu    ? "cpu"
                                                                         : "automatic";
    const char* reason = backend == BundleBackendPreference::cpu ? "requested_cpu"
                         : has_partial_pose_locks                ? "partial_pose_locks"
                         : !supported_parameterization           ? "unsupported_parameterization"
                         : problem.observations.size() < options.cuda_min_observations
                             ? "below_observation_threshold"
                             : "gpu_unavailable_or_not_built";
    core::Logger::instance().info(
        "sfm backend selection: stage=bundle requested=", requested, " selected=",
        use_cuda     ? "cuda"
        : use_vulkan ? "vulkan"
                     : "cpu",
        " reason=", (use_cuda || use_vulkan) ? "gpu_supported" : reason,
        " observations=", problem.observations.size(), " minimum=", options.cuda_min_observations,
        " intrinsic_groups=", problem.intrinsics.size());
    if (!use_cuda && !use_vulkan &&
        (backend == BundleBackendPreference::cuda || backend == BundleBackendPreference::vulkan))
        core::Logger::instance().warning(
            "requested GPU bundle solver stays on CPU: reason=", reason);

    bool solved = false;
    const auto* selected = use_cuda ? &cuda : use_vulkan ? &vulkan : nullptr;
    if (selected) {
        try {
            core::Logger::instance().info(
                "bundle backend=", selected->name, " device=", selected->device_name(),
                " observations=", problem.observations.size());
            core::StageScope stage("sfm.bundle_solve", core::LogLevel::info, selected->name);
            summary.optimizer = selected->optimize(problem, opt);
            summary.backend = use_cuda ? BundleBackend::cuda : BundleBackend::vulkan;
            solved = summary.optimizer.usable();
            if (!solved)
                core::Logger::instance().warning(
                    selected->name, " bundle adjustment produced an unusable step; continuing with "
                                    "CPU from the last accepted state");
        } catch (const std::exception& error) {
            core::Logger::instance().warning(
                selected->name,
                " bundle adjustment unavailable at runtime; falling back to CPU: ", error.what());
        }
    }
    if (!solved) {
        core::StageScope solve_stage("sfm.bundle_solve", core::LogLevel::info, "cpu");
        summary.optimizer = ba::solver_backend(ba::SolverDevice::cpu).optimize(problem, opt);
        summary.backend = BundleBackend::cpu;
    }
}
PositioningSolver make_positioning_solver(
    const GlobalPositioningOptions& options,
    unsigned observations,
    unsigned cameras,
    const std::function<const ba::BearingProblem&()>& prepare) {
    PositioningSolver result;
    const bool automatic = options.backend == PositioningBackend::automatic;
    const bool size_ok = observations >= 50000 && cameras <= 2000;
    for (auto device : {ba::SolverDevice::cuda, ba::SolverDevice::vulkan}) {
        const bool is_cuda = device == ba::SolverDevice::cuda;
        const bool explicitly_requested =
            options.backend == (is_cuda ? PositioningBackend::cuda : PositioningBackend::vulkan);
        if (!explicitly_requested && !(automatic && (!is_cuda || options.prefer_cuda)))
            continue;
        const auto& backend = ba::solver_backend(device);
        if (!size_ok || !backend.available()) {
            if (explicitly_requested)
                core::Logger::instance().warning(
                    "positioning-backend=", backend.name,
                    " requested but this solve stays on the CPU: built=", backend.built,
                    " observations=", observations, " minimum=50000 cameras=", cameras,
                    " maximum=2000");
            continue;
        }
        try {
            const auto& problem = prepare();
            result.solver = backend.make_bearing(problem);
            result.name = backend.name;
            core::Logger::instance().info(
                "global positioning backend=", backend.name,
                " bearing_schur cameras=", problem.cameras.size(),
                " points=", problem.points.size(), " observations=", observations);
            break;
        } catch (const std::exception& error) {
            core::Logger::instance().warning(
                backend.name, " bearing initialization failed; using CPU: ", error.what());
        }
    }
    core::Logger::instance().info(
        "sfm backend selection: stage=positioning requested=",
        automatic                                       ? "automatic"
        : options.backend == PositioningBackend::cuda   ? "cuda"
        : options.backend == PositioningBackend::vulkan ? "vulkan"
                                                        : "cpu",
        " selected=", result.name, " reason=",
        result.solver                                ? "gpu_supported"
        : options.backend == PositioningBackend::cpu ? "requested_cpu"
        : observations < 50000                       ? "below_observation_threshold"
        : cameras > 2000                             ? "above_camera_limit"
                                                     : "gpu_unavailable_or_initialization_failed",
        " observations=", observations, " cameras=", cameras);
    return result;
}
bool solve_positioning_pass(
    PositioningSolver& selected,
    ba::BearingProblem& problem,
    const std::vector<double>& weights,
    unsigned iterations,
    double tolerance,
    double seconds,
    unsigned pass,
    ba::BearingSolveSummary& step) {
    if (!selected.solver)
        return false;
    try {
        core::StageScope stage("sfm.positioning_solve", core::LogLevel::info, selected.name);
        step = selected.solver->solve(weights, iterations, tolerance, seconds);
        if (!step.usable)
            throw std::runtime_error("unusable bearing solve");
        selected.solver->download(problem);
        core::Logger::instance().info(
            selected.name, " bearing pass=", pass, " iterations=", step.iterations,
            " cost=", step.initial_cost, " -> ", step.final_cost, " seconds=", step.seconds);
        return true;
    } catch (const std::exception& error) {
        core::Logger::instance().warning(
            selected.name, " bearing solve failed; using CPU: ", error.what());
        selected.solver.reset();
        return false;
    }
}
} // namespace photara::sfm::detail
