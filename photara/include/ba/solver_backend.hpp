#pragma once
#include "ba/bearing_problem.hpp"
#include "ba/optimizer.hpp"

namespace photara::ba {
enum class SolverDevice { cpu, cuda, vulkan };
class BearingSolver {
  public:
    virtual ~BearingSolver() = default;
    virtual BearingSolveSummary solve(
        const std::vector<double>& weights,
        unsigned iterations,
        double tolerance,
        double max_seconds) = 0;
    virtual void download(BearingProblem& problem) const = 0;
};
// Backend implementations supply these operations. Selection and fallback
// belong to shared C++; numerical kernels remain in their respective backends.
struct SolverBackend {
    const char* name;
    bool built;
    bool (*available)() noexcept;
    std::string (*device_name)();
    OptimizerSummary (*optimize)(Problem&, const OptimizerOptions&);
    std::unique_ptr<BearingSolver> (*make_bearing)(const BearingProblem&);
};
const SolverBackend& solver_backend(SolverDevice device);
} // namespace photara::ba
