#include <stdexcept>
#pragma once

#include <cstddef>
#include "core/cuda_runtime_api.hpp"

namespace tinytensor::tensor_ops {

// MaxPool2d: window-based max pooling [N,C,H,W] -> [N,C,H_out,W_out]
// kernel_size: pooling window size
// stride: step between windows (default = kernel_size)
// padding: zero-padding added to input
#if defined(TINYTENSOR_NO_CUDA)
inline void launch_max_pool2d(
    const float* input,
    float* output,
    int N,
    int C,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream = nullptr) {
    throw std::runtime_error("CUDA tensor backend is disabled; select Vulkan");
}
#else
void launch_max_pool2d(
    const float* input,
    float* output,
    int N,
    int C,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream = nullptr);
#endif

// AdaptiveAvgPool2d: pools to fixed output size regardless of input size
// Adaptive window sizes computed per output position
#if defined(TINYTENSOR_NO_CUDA)
inline void launch_adaptive_avg_pool2d(
    const float* input,
    float* output,
    int N,
    int C,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    cudaStream_t stream = nullptr) {
    throw std::runtime_error("CUDA tensor backend is disabled; select Vulkan");
}
#else
void launch_adaptive_avg_pool2d(
    const float* input,
    float* output,
    int N,
    int C,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    cudaStream_t stream = nullptr);
#endif

// Fused bias + relu: output = relu(input + bias)
// input: [N, C] or [N, C, H, W], bias: [C], output: same as input
#if defined(TINYTENSOR_NO_CUDA)
inline void launch_bias_relu(
    const float* input,
    const float* bias,
    float* output,
    int total_elements,
    int channels,
    int spatial_size,
    cudaStream_t stream = nullptr) {
    throw std::runtime_error("CUDA tensor backend is disabled; select Vulkan");
}
#else
void launch_bias_relu(
    const float* input,
    const float* bias,
    float* output,
    int total_elements,
    int channels,
    int spatial_size,
    cudaStream_t stream = nullptr);
#endif

// Fused bias only: output = input + bias
#if defined(TINYTENSOR_NO_CUDA)
inline void launch_bias_add(
    const float* input,
    const float* bias,
    float* output,
    int total_elements,
    int channels,
    int spatial_size,
    cudaStream_t stream = nullptr) {
    throw std::runtime_error("CUDA tensor backend is disabled; select Vulkan");
}
#else
void launch_bias_add(
    const float* input,
    const float* bias,
    float* output,
    int total_elements,
    int channels,
    int spatial_size,
    cudaStream_t stream = nullptr);
#endif

// ReLU: output = max(0, input)
#if defined(TINYTENSOR_NO_CUDA)
inline void launch_relu(const float* input, float* output, int n, cudaStream_t stream = nullptr) {
    throw std::runtime_error("CUDA tensor backend is disabled; select Vulkan");
}
#else
void launch_relu(const float* input, float* output, int n, cudaStream_t stream = nullptr);
#endif

} // namespace tinytensor::tensor_ops
