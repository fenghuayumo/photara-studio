#include "igs_selection_cuda.hpp"
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include <chrono>
#include <climits>
#include <stdexcept>
#include <string>

namespace photara::splat::densification::gpu_detail {
namespace {
constexpr unsigned threads = 256;
void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}
void order_input(const tinytensor::Tensor& weights, cudaStream_t stream) {
    if (weights.stream() == stream)
        return;
    cudaEvent_t ready{};
    check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "create IGS input event");
    const auto recorded = cudaEventRecord(ready, weights.stream());
    const auto waited = recorded == cudaSuccess ? cudaStreamWaitEvent(stream, ready, 0) : recorded;
    const auto destroyed = cudaEventDestroy(ready);
    check(waited, "order IGS input stream");
    check(destroyed, "release IGS input event");
}
__device__ std::uint64_t mix(std::uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
__global__ void make_keys(
    const float* weights,
    std::uint64_t* keys,
    unsigned* eligible,
    unsigned n,
    std::uint64_t seed) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    using Reduce = cub::BlockReduce<unsigned, threads>;
    __shared__ typename Reduce::TempStorage reduction;
    unsigned valid = 0;
    if (i < n) {
        const unsigned bits = __float_as_uint(weights[i]);
        // Classify by bits so positive subnormals survive fast-math FTZ.
        valid = bits != 0 && bits < 0x7f800000U;
        std::uint64_t key = 0;
        if (valid) {
            const auto random = mix(seed + 0x9e3779b97f4a7c15ULL * (std::uint64_t(i) + 1));
            const float u =
                fminf(1.F - 1e-7F, fmaxf(1e-7F, (float(random >> 41) + .5F) * 0x1p-23F));
            const float log_weight = (bits < 0x00800000U)
                                         ? logf(float(bits)) - 149.F * 0.6931471805599453F
                                         : logf(weights[i]);
            const float score = log_weight - logf(-logf(u));
            const unsigned score_bits = __float_as_uint(score);
            const unsigned ordered =
                score_bits ^ ((score_bits & 0x80000000U) ? 0xffffffffU : 0x80000000U);
            key = (std::uint64_t(ordered) << 32) | i;
        }
        keys[i] = key;
    }
    const unsigned count = Reduce(reduction).Sum(valid);
    if (threadIdx.x == 0)
        atomicAdd(eligible, count);
}
__global__ void unpack_indices(const std::uint64_t* keys, int* indices, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        indices[i] = int(unsigned(keys[i]));
}
} // namespace

tinytensor::Tensor igs_weighted_sample_cuda(
    const tinytensor::Tensor& weights,
    const std::size_t requested,
    const std::uint64_t seed,
    double* prepare_ms,
    double* sort_ms) {
    using tinytensor::DataType;
    using tinytensor::Tensor;
    constexpr auto cuda = tinytensor::Device::CUDA;
    if (!requested || !weights.numel())
        return {};
    if (!weights.is_valid() || weights.device() != cuda || weights.dtype() != DataType::Float32 ||
        !weights.is_contiguous() || weights.numel() > std::size_t(INT_MAX))
        throw std::invalid_argument(
            "IGS CUDA sampling requires contiguous float32 CUDA weights, at most INT_MAX rows");
    const auto started = std::chrono::steady_clock::now();
    const auto stream = tinytensor::getCurrentCUDAStream();
    order_input(weights, stream);
    const auto n = unsigned(weights.numel());
    auto keys = Tensor::empty({n}, cuda, DataType::Int64);
    auto count = Tensor::empty({1}, cuda, DataType::Int32);
    auto* in = reinterpret_cast<std::uint64_t*>(keys.data_ptr());
    auto* counter = reinterpret_cast<unsigned*>(count.data_ptr());
    check(cudaMemsetAsync(counter, 0, sizeof(unsigned), stream), "clear IGS candidate count");
    make_keys<<<(n + threads - 1) / threads, threads, 0, stream>>>(
        weights.ptr<float>(), in, counter, n, seed);
    check(cudaGetLastError(), "generate IGS Gumbel keys");
    unsigned eligible = 0;
    check(
        cudaMemcpyAsync(&eligible, counter, sizeof(eligible), cudaMemcpyDeviceToHost, stream),
        "read IGS candidate count");
    check(cudaStreamSynchronize(stream), "wait IGS candidate count");
    const auto prepared = std::chrono::steady_clock::now();
    if (prepare_ms)
        *prepare_ms += std::chrono::duration<double, std::milli>(prepared - started).count();
    const auto k = unsigned(std::min(requested, std::size_t(eligible)));
    if (!k)
        return {};
    auto sorted = Tensor::empty({n}, cuda, DataType::Int64);
    auto* out = reinterpret_cast<std::uint64_t*>(sorted.data_ptr());
    std::size_t scratch_bytes = 0;
    check(
        cub::DeviceRadixSort::SortKeysDescending(nullptr, scratch_bytes, in, out, n, 0, 64, stream),
        "size IGS key-sort workspace");
    auto scratch = Tensor::empty({scratch_bytes}, cuda, DataType::UInt8);
    check(
        cub::DeviceRadixSort::SortKeysDescending(
            scratch.data_ptr(), scratch_bytes, in, out, n, 0, 64, stream),
        "sort IGS Gumbel keys");
    auto result = Tensor::empty({k}, cuda, DataType::Int32);
    unpack_indices<<<(k + threads - 1) / threads, threads, 0, stream>>>(out, result.ptr<int>(), k);
    check(cudaGetLastError(), "unpack IGS parent indices");
    if (sort_ms) {
        // Profiling is opt-in: normal refinements do not wait for sort completion.
        check(cudaStreamSynchronize(stream), "profile IGS key sort");
        *sort_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prepared)
                .count();
    }
    return result;
}
} // namespace photara::splat::densification::gpu_detail
