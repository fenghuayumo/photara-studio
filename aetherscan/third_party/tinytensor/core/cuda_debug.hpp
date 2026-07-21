#pragma once

#include <iostream>
#include <cuda_runtime.h>

// TinyTensor CUDA Debug - Minimal CUDA debugging utilities

namespace tinytensor {

inline void check_cuda_impl(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "[CUDA Error] " << file << ":" << line << " - "
                  << cudaGetErrorName(error) << ": "
                  << cudaGetErrorString(error) << std::endl;
    }
}

} // namespace tinytensor

#define CHECK_CUDA(call) tinytensor::check_cuda_impl(call, __FILE__, __LINE__)
