#pragma once

#include "ba/problem.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace aetherscan::ba {

enum class TerminationReason {
    converged,
    maximum_iterations,
    numerical_failure
};

struct OptimizerOptions {
    std::size_t maximum_iterations{20};
    std::size_t maximum_pcg_iterations{100};
    double huber_delta{2.0};
    double minimum_depth{1e-8};
    double initial_damping{1e-3};
    double minimum_damping{1e-12};
    double maximum_damping{1e12};
    double function_tolerance{1e-7};
    double step_tolerance{1e-9};
    double pcg_tolerance{1e-4};
    bool fix_first_pose{true};
    bool fix_first_point{true};
    bool optimize_rotations{true};
    bool optimize_points{true};
    // Optimized independently for every intrinsic group referenced by poses.
    bool optimize_focal{false};            // tied fx = fy
    bool optimize_principal_point{false};  // cx, cy
    bool optimize_distortion{false};       // k1, k2, p1, p2
};

struct IterationSummary {
    std::size_t iteration{0};
    double cost{0.0};
    double damping{0.0};
    double step_norm{0.0};
    std::size_t pcg_iterations{0};
    bool accepted{false};
};

struct OptimizerSummary {
    TerminationReason termination{TerminationReason::maximum_iterations};
    double initial_cost{0.0};
    double final_cost{0.0};
    double total_time_ms{0.0};
    double linearization_time_ms{0.0};
    double assembly_time_ms{0.0};
    double solve_time_ms{0.0};
    double update_and_cost_time_ms{0.0};
    std::size_t successful_steps{0};
    std::size_t unsuccessful_steps{0};
    std::vector<IterationSummary> iterations;

    [[nodiscard]] bool usable() const noexcept;
    [[nodiscard]] std::string brief_report() const;
};

// Optimizes camera poses, 3D points, and optionally grouped intrinsics.
OptimizerSummary optimize_cpu(
    Problem& problem,
    const OptimizerOptions& options = {});

#if defined(AETHERSCAN_HAS_CUDA)
// Persistent GPU optimizer. upload() and download() are explicit so callers
// can exclude PCIe transfers from iterative BA timing and later keep the SfM
// scene resident across local-BA invocations.
class CudaOptimizer {
public:
    explicit CudaOptimizer(OptimizerOptions options = {});
    ~CudaOptimizer();

    CudaOptimizer(CudaOptimizer&&) noexcept;
    CudaOptimizer& operator=(CudaOptimizer&&) noexcept;
    CudaOptimizer(const CudaOptimizer&) = delete;
    CudaOptimizer& operator=(const CudaOptimizer&) = delete;

    [[nodiscard]] static bool is_available() noexcept;
    [[nodiscard]] static std::string device_name();
    void upload(const Problem& problem);
    OptimizerSummary optimize();
    void download(Problem& problem) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

OptimizerSummary optimize_cuda(
    Problem& problem,
    const OptimizerOptions& options = {});
#endif

[[nodiscard]] double evaluate_cost(
    const Problem& problem,
    double huber_delta = 2.0,
    double minimum_depth = 1e-8);

}  // namespace aetherscan::ba
