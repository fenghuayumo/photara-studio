#pragma once

#include "aetherscan/ba/problem.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace aetherscan::ba {

// Jacobians use a six-dimensional local pose increment:
// left-multiplicative rotation followed by an additive camera-center update.
struct LinearizedObservation {
    double residual[2]{};          // NOLINT(modernize-avoid-c-arrays)
    double pose_jacobian[12]{};    // NOLINT(modernize-avoid-c-arrays), row-major 2 x 6
    double point_jacobian[6]{};    // NOLINT(modernize-avoid-c-arrays), row-major 2 x 3
    double robust_weight{1.0};
    bool valid{false};
};

struct LinearizationOutput {
    std::vector<LinearizedObservation> observations;

    void resize(const std::size_t count) { observations.resize(count); }
};

struct EvaluationStats {
    double elapsed_ms{0.0};
    std::size_t observation_count{0};

    [[nodiscard]] double million_observations_per_second() const noexcept {
        return elapsed_ms > 0.0
            ? static_cast<double>(observation_count) / (elapsed_ms * 1000.0)
            : 0.0;
    }
};

struct LinearizerOptions {
    double huber_delta{2.0};
    double minimum_depth{1e-8};
};

EvaluationStats linearize_cpu(
    const Problem& problem,
    LinearizationOutput& output,
    const LinearizerOptions& options = {});

#if defined(AETHERSCAN_HAS_CUDA)
class CudaLinearizer {
public:
    explicit CudaLinearizer(LinearizerOptions options = {});
    ~CudaLinearizer();

    CudaLinearizer(CudaLinearizer&&) noexcept;
    CudaLinearizer& operator=(CudaLinearizer&&) noexcept;
    CudaLinearizer(const CudaLinearizer&) = delete;
    CudaLinearizer& operator=(const CudaLinearizer&) = delete;

    [[nodiscard]] static bool is_available() noexcept;
    [[nodiscard]] static std::string device_name();

    // Upload once, then evaluate repeatedly while parameters remain on-device.
    void upload(const Problem& problem);
    EvaluationStats evaluate();
    void download(LinearizationOutput& output) const;
    [[nodiscard]] std::size_t observation_count() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace aetherscan::ba
