#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "core/export.hpp"

namespace tinytensor {

    // Thread-local current CUDA stream (PyTorch-style).
    // Exported from lfs_core so the singleton is shared across DSO boundaries.
    LFS_CORE_API cudaStream_t getCurrentCUDAStream();
    LFS_CORE_API void setCurrentCUDAStream(cudaStream_t stream);

    inline void waitForCUDAStream(cudaStream_t execution_stream, cudaStream_t dependency_stream) {
        if (dependency_stream == nullptr || dependency_stream == execution_stream) {
            return;
        }

        cudaEvent_t ready = nullptr;
        cudaError_t status = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
        if (status == cudaSuccess) {
            status = cudaEventRecord(ready, dependency_stream);
            if (status == cudaSuccess) {
                status = cudaStreamWaitEvent(execution_stream, ready, 0);
            }
        }

        if (ready != nullptr) {
            const cudaError_t destroy_status = cudaEventDestroy(ready);
            if (status == cudaSuccess && destroy_status != cudaSuccess) {
                status = destroy_status;
            }
        }

        if (status != cudaSuccess) {
            const cudaError_t sync_status = cudaStreamSynchronize(dependency_stream);
            if (sync_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Failed to synchronize CUDA streams: ") +
                    cudaGetErrorString(sync_status));
            }
        }
    }

    /**
     * RAII guard for temporarily setting the current CUDA stream
     * (PyTorch's CUDAStreamGuard pattern)
     */
    class CUDAStreamGuard {
    public:
        explicit CUDAStreamGuard(cudaStream_t stream)
            : prev_stream_(getCurrentCUDAStream()) {
            setCurrentCUDAStream(stream);
        }

        ~CUDAStreamGuard() {
            setCurrentCUDAStream(prev_stream_);
        }

        CUDAStreamGuard(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard(CUDAStreamGuard&&) = delete;
        CUDAStreamGuard& operator=(CUDAStreamGuard&&) = delete;

    private:
        cudaStream_t prev_stream_;
    };

} // namespace tinytensor
