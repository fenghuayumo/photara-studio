#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace photara::ba {
// Fixed-rotation positioning: d - normalize(point - camera), with one fixed
// center and one scalar baseline. No focal approximation or extra pose locks.
struct BearingObservation {
    std::uint32_t camera{}, point{};
    std::array<double, 3> direction{};
};
struct BearingProblem {
    std::vector<std::array<double, 3>> cameras, points;
    std::vector<BearingObservation> observations;
    std::uint32_t anchor{}, baseline_first{}, baseline_second{};
    double baseline{}, huber{0.03};
};
struct BearingSolveSummary {
    bool usable{};
    unsigned iterations{};
    double initial_cost{}, final_cost{}, seconds{};
};
} // namespace photara::ba
