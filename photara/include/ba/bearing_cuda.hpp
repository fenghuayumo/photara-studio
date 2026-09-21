#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace photara::ba {
// Fixed-rotation positioning: d - normalize(point - camera), with one fixed
// center and one scalar baseline. No focal approximation or extra pose locks.
struct BearingObservation {
    std::uint32_t camera{}, point{};
    std::array<double,3> direction{};
};
struct BearingProblem {
    std::vector<std::array<double,3>> cameras, points;
    std::vector<BearingObservation> observations;
    std::uint32_t anchor{}, baseline_first{}, baseline_second{};
    double baseline{}, huber{0.03};
};
struct BearingSolveSummary {
    bool usable{};
    unsigned iterations{};
    double initial_cost{}, final_cost{}, seconds{};
};
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
