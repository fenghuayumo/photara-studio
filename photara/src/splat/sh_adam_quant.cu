#include "sh_adam_quant.hpp"

#include "cuda_common.hpp"
#include "core/vram_profiler.hpp"
#include "splat_drender/sh_adam_quant.cuh"

#include <stdexcept>

namespace photara::splat::detail {
namespace {

__global__ void sh_adam_quant_rows_kernel(
    int rows, int stride, int active, float* parameter, const float* gradient,
    __half* first, std::uint8_t* packed, float* bounds, float dc_lr,
    float rest_lr, float beta1, float beta2, float correction1,
    float correction2, float adam_epsilon, float regularization) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows) return;
    splat_drender::sh_adam_quant_warp_row(
        stride, active, parameter + static_cast<std::size_t>(row) * stride,
        gradient + static_cast<std::size_t>(row) * stride,
        first + static_cast<std::size_t>(row) * stride,
        packed + static_cast<std::size_t>(row) * stride,
        bounds + static_cast<std::size_t>(row) * 4, dc_lr, rest_lr, beta1,
        beta2, correction1, correction2, adam_epsilon, regularization);
}

__global__ void sh_adam_quant_zero_rows_kernel(
    const int* indices, int count, int stride, __half* first,
    std::uint8_t* packed, float* bounds) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count) return;
    const int row = indices[slot];
    __half* first_row = first + static_cast<std::size_t>(row) * stride;
    std::uint8_t* packed_row = packed + static_cast<std::size_t>(row) * stride;
    for (int col = 0; col < stride; ++col) {
        first_row[col] = __float2half(0.f);
        packed_row[col] = 0;
    }
    float* bound_row = bounds + static_cast<std::size_t>(row) * 4;
    for (int value = 0; value < 4; ++value) bound_row[value] = 0.f;
}

__global__ void sh_adam_quant_select_first_kernel(
    const __half* source, const int* indices, int rows, int stride,
    __half* destination) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * stride;
    if (slot >= count) return;
    const int row = slot / stride;
    const int col = slot - row * stride;
    destination[slot] =
        source[static_cast<std::size_t>(indices[row]) * stride + col];
}

}  // namespace

ShAdamQuant make_sh_adam_quant(
    const std::size_t rows, const int stride, const tinytensor::Device device) {
    if (stride <= 0 || stride > 64)
        throw std::invalid_argument(
            "quantized SH Adam stride must be in [1, 64]");
    tinytensor::VramScope scope("optimizer.state");
    ShAdamQuant state;
    state.stride = stride;
    state.first = tinytensor::Tensor::zeros(
        {rows, static_cast<std::size_t>(stride)}, device,
        tinytensor::DataType::Float16);
    state.packed = tinytensor::Tensor::zeros(
        {rows, static_cast<std::size_t>(stride)}, device,
        tinytensor::DataType::UInt8);
    state.bounds = tinytensor::Tensor::zeros(
        {rows, std::size_t{4}}, device);
    return state;
}

void sh_adam_quant_select_rows(
    ShAdamQuant& state, const tinytensor::Tensor& indices) {
    if (!state.active()) return;
    const auto device = state.packed.device();
    if (!indices.is_valid() || indices.numel() == 0) {
        state.first = tinytensor::Tensor::zeros(
            {std::size_t{0}, static_cast<std::size_t>(state.stride)}, device,
            tinytensor::DataType::Float16);
        state.packed = tinytensor::Tensor::zeros(
            {std::size_t{0}, static_cast<std::size_t>(state.stride)},
            device, tinytensor::DataType::UInt8);
        state.bounds = tinytensor::Tensor::zeros(
            {std::size_t{0}, std::size_t{4}}, device);
        return;
    }
    tinytensor::Tensor index = indices.dtype() == tinytensor::DataType::Int32
        ? indices
        : indices.to(tinytensor::DataType::Int32);
    if (index.device() != device) index = index.to(device);
    const int rows = static_cast<int>(index.numel());
    tinytensor::Tensor selected_first = tinytensor::Tensor::empty(
        {static_cast<std::size_t>(rows),
         static_cast<std::size_t>(state.stride)},
        device, tinytensor::DataType::Float16);
    const int count = rows * state.stride;
    sh_adam_quant_select_first_kernel<<<(count + 255) / 256, 256>>>(
        state.first.ptr<__half>(), index.ptr<int>(), rows, state.stride,
        selected_first.ptr<__half>());
    check_cuda(cudaGetLastError(), "select quantized SH Adam first rows");
    state.first = std::move(selected_first);
    state.packed = state.packed.index_select(0, index);
    state.bounds = state.bounds.index_select(0, index);
}

void sh_adam_quant_zero_rows(
    ShAdamQuant& state, const tinytensor::Tensor& indices) {
    if (!state.active() || !indices.is_valid() || indices.numel() == 0) return;
    tinytensor::Tensor index = indices.dtype() == tinytensor::DataType::Int32
        ? indices
        : indices.to(tinytensor::DataType::Int32);
    if (index.device() != state.packed.device())
        index = index.to(state.packed.device());
    const int count = static_cast<int>(index.numel());
    sh_adam_quant_zero_rows_kernel<<<(count + 255) / 256, 256>>>(
        index.ptr<int>(), count, state.stride, state.first.ptr<__half>(),
        state.packed.ptr<std::uint8_t>(), state.bounds.ptr<float>());
    check_cuda(cudaGetLastError(), "zero quantized SH Adam rows");
}

void sh_adam_quant_append_zeros(ShAdamQuant& state, const std::size_t count) {
    if (!state.active() || count == 0) return;
    const auto device = state.packed.device();
    state.first = tinytensor::Tensor::cat(
        {state.first, tinytensor::Tensor::zeros(
             {count, static_cast<std::size_t>(state.stride)}, device,
             tinytensor::DataType::Float16)},
        0);
    state.packed = tinytensor::Tensor::cat(
        {state.packed, tinytensor::Tensor::zeros(
             {count, static_cast<std::size_t>(state.stride)}, device,
             tinytensor::DataType::UInt8)},
        0);
    state.bounds = tinytensor::Tensor::cat(
        {state.bounds, tinytensor::Tensor::zeros({count, std::size_t{4}}, device)},
        0);
}

void sh_adam_quant_step(
    tinytensor::Tensor& parameter, const tinytensor::Tensor& gradient,
    ShAdamQuant& state, const int active_stride, const float learning_rate,
    const float rest_learning_rate, const float beta1, const float beta2,
    const float correction1, const float correction2, const float adam_epsilon,
    const float regularization_factor) {
    if (!state.active())
        throw std::invalid_argument("quantized SH Adam state is empty");
    if (active_stride <= 0 || active_stride > state.stride)
        throw std::invalid_argument(
            "quantized SH Adam active stride is outside the row");
    const std::size_t count = parameter.numel();
    if (count == 0) return;
    if (state.stride <= 0 ||
        count % static_cast<std::size_t>(state.stride) != 0)
        throw std::invalid_argument(
            "quantized SH Adam parameter is not a whole number of rows");
    const int rows = static_cast<int>(
        count / static_cast<std::size_t>(state.stride));
    if (gradient.numel() != count ||
        state.first.numel() != static_cast<std::size_t>(rows) * state.stride ||
        state.first.dtype() != tinytensor::DataType::Float16 ||
        state.packed.numel() != static_cast<std::size_t>(rows) * state.stride ||
        state.bounds.numel() != static_cast<std::size_t>(rows) * 4)
        throw std::invalid_argument(
            "quantized SH Adam storage does not match the parameter");
    if (!parameter.is_contiguous() || !state.first.is_contiguous() ||
        !state.packed.is_contiguous() ||
        !state.bounds.is_contiguous())
        throw std::invalid_argument(
            "quantized SH Adam tensors must be contiguous");
    if (parameter.device() != tinytensor::Device::CUDA ||
        gradient.device() != tinytensor::Device::CUDA ||
        state.first.device() != tinytensor::Device::CUDA ||
        state.packed.device() != tinytensor::Device::CUDA)
        throw std::invalid_argument("quantized SH Adam step is CUDA-only");
    const tinytensor::Tensor gradient_storage = gradient.is_contiguous()
        ? gradient
        : gradient.contiguous();
    constexpr int k_threads = 128;
    const int blocks = (rows + k_threads / 32 - 1) / (k_threads / 32);
    sh_adam_quant_rows_kernel<<<blocks, k_threads>>>(
        rows, state.stride, active_stride, parameter.ptr<float>(),
        gradient_storage.ptr<float>(), state.first.ptr<__half>(),
        state.packed.ptr<std::uint8_t>(), state.bounds.ptr<float>(),
        learning_rate, rest_learning_rate, beta1, beta2, correction1,
        correction2, adam_epsilon, regularization_factor);
    check_cuda(cudaGetLastError(), "quantized SH Adam step");
}

}  // namespace photara::splat::detail
