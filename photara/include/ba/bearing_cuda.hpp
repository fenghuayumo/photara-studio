#pragma once
#include "ba/bearing_problem.hpp"
#include <memory>

namespace photara::ba {
class CudaBearingOptimizer {
public:
    explicit CudaBearingOptimizer(const BearingProblem& problem);
    ~CudaBearingOptimizer();
    CudaBearingOptimizer(const CudaBearingOptimizer&)=delete;
    CudaBearingOptimizer& operator=(const CudaBearingOptimizer&)=delete;
    BearingSolveSummary solve(const std::vector<double>& weights, unsigned iterations,
                              double tolerance, double max_seconds);
    void download(BearingProblem& problem) const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}
