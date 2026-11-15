#include "ba/solver_backend.hpp"
#include <stdexcept>
namespace photara::ba {
const SolverBackend& cuda_solver_backend();
const SolverBackend& vulkan_solver_backend();
namespace {
bool available() noexcept {
    return true;
}
bool unavailable() noexcept {
    return false;
}
std::string cpu_name() {
    return "CPU";
}
std::string disabled_name() {
    return "unavailable";
}
OptimizerSummary disabled_solve(Problem&, const OptimizerOptions&) {
    throw std::runtime_error("Requested BA backend was not built");
}
std::unique_ptr<BearingSolver> disabled_bearing(const BearingProblem&) {
    throw std::runtime_error("Requested bearing backend was not built");
}
} // namespace
const SolverBackend& solver_backend(SolverDevice device) {
    static const SolverBackend cpu{"cpu",    true,         available,
                                   cpu_name, optimize_cpu, disabled_bearing};
    static const SolverBackend no_cuda{"cuda",        false,          unavailable,
                                       disabled_name, disabled_solve, disabled_bearing};
    static const SolverBackend no_vulkan{"vulkan",      false,          unavailable,
                                         disabled_name, disabled_solve, disabled_bearing};
    switch (device) {
    case SolverDevice::cpu:
        return cpu;
    case SolverDevice::cuda:
#if defined(PHOTARA_HAS_CUDA)
        return cuda_solver_backend();
#else
        return no_cuda;
#endif
    case SolverDevice::vulkan:
#if defined(PHOTARA_HAS_VULKAN_BA)
        return vulkan_solver_backend();
#else
        return no_vulkan;
#endif
    }
    throw std::invalid_argument("Unknown BA solver device");
}
} // namespace photara::ba
