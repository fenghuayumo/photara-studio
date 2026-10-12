#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace photara::core {

// Stream intervals, including submission gaps, rather than summed kernel time.
// Read only after the caller's existing synchronization; never synchronize here.
template <std::size_t Categories>
class CudaTimeline {
    struct Mark { cudaEvent_t event; std::size_t category; };
    std::vector<Mark> marks_;
    static void check(cudaError_t status) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("CUDA timing: ") + cudaGetErrorString(status));
    }
public:
    CudaTimeline() = default;
    CudaTimeline(const CudaTimeline&) = delete;
    CudaTimeline& operator=(const CudaTimeline&) = delete;
    ~CudaTimeline() { for (const auto& mark : marks_) cudaEventDestroy(mark.event); }

    void mark(std::size_t category, cudaStream_t stream = nullptr) {
        cudaEvent_t event{};
        check(cudaEventCreate(&event));
        try {
            marks_.push_back({event, category});
        } catch (...) { cudaEventDestroy(event); throw; }
        check(cudaEventRecord(event, stream));
    }
    void finish(cudaStream_t stream = nullptr) { mark(Categories, stream); }
    std::array<double, Categories> elapsed_ms() const {
        std::array<double, Categories> result{};
        for (std::size_t i = 1; i < marks_.size(); ++i) {
            if (marks_[i-1].category >= Categories) continue;
            float elapsed{};
            check(cudaEventElapsedTime(&elapsed, marks_[i-1].event, marks_[i].event));
            result[marks_[i-1].category] += elapsed;
        }
        return result;
    }
};

} // namespace photara::core
