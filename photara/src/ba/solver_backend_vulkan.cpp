#include "ba/solver_backend.hpp"
#include "ba/bearing_vulkan.hpp"
namespace photara::ba {
namespace {
class BearingAdapter final : public BearingSolver {
    VulkanBearingOptimizer optimizer_;

  public:
    explicit BearingAdapter(const BearingProblem& problem) : optimizer_(problem) {
    }
    BearingSolveSummary solve(
        const std::vector<double>& weights,
        unsigned iterations,
        double tolerance,
        double max_seconds) override {
        return optimizer_.solve(weights, iterations, tolerance, max_seconds);
    }
    void download(BearingProblem& problem) const override {
        optimizer_.download(problem);
    }
};
std::unique_ptr<BearingSolver> make_bearing(const BearingProblem& problem) {
    return std::make_unique<BearingAdapter>(problem);
}
} // namespace
const SolverBackend& vulkan_solver_backend() {
    static const SolverBackend backend{
        "vulkan",        true,        VulkanOptimizer::is_available, VulkanOptimizer::device_name,
        optimize_vulkan, make_bearing};
    return backend;
}
} // namespace photara::ba
