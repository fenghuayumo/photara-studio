#pragma once

#include "internal/tensor_impl.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace photara::splat::detail {

inline constexpr unsigned k_cuda_threads = 256;

inline void check_cuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
}

inline void ensure_same_shape(
    tinytensor::Tensor& tensor, const tinytensor::Tensor& like) {
    if (!tensor.is_valid() || tensor.shape() != like.shape() ||
        tensor.device() != like.device())
        tensor = tinytensor::Tensor::empty(like.shape(), like.device());
}

}  // namespace photara::splat::detail
