#pragma once

#include "ba/bearing_cuda.hpp"

#include <memory>
#include <string>

namespace photara::ba {

// Fixed-rotation bearing Schur solver on Vulkan. The step matches
// CudaBearingOptimizer, including the frozen Jacobi column scale.
class VulkanBearingOptimizer {
  public:
    explicit VulkanBearingOptimizer(const BearingProblem &problem);
    ~VulkanBearingOptimizer();
    VulkanBearingOptimizer(const VulkanBearingOptimizer &) = delete;
    VulkanBearingOptimizer &operator=(const VulkanBearingOptimizer &) = delete;

    [[nodiscard]] static bool is_available() noexcept;
    [[nodiscard]] static std::string device_name();

    BearingSolveSummary solve(const std::vector<double> &weights, unsigned iterations,
                              double tolerance, double max_seconds);
    void download(BearingProblem &problem) const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace photara::ba
