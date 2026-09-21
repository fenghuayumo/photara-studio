#include "internal/cuda_stream_context.hpp"

namespace tinytensor {

    static thread_local cudaStream_t tl_current_stream = nullptr;

    cudaStream_t getCurrentCUDAStream() {
        return tl_current_stream;
    }

    void setCurrentCUDAStream(cudaStream_t stream) {
        tl_current_stream = stream;
    }

} // namespace tinytensor
