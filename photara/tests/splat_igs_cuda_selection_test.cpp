#include "../src/splat/igs_selection_cuda.hpp"
#include "../src/splat/densification_igs.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

using photara::splat::densification::gpu_detail::igs_weighted_sample_cuda;
using tinytensor::Tensor;
constexpr auto cuda = tinytensor::Device::CUDA;
void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}
void sync() {
    require(cudaDeviceSynchronize() == cudaSuccess, "CUDA synchronization failed");
}
std::uint64_t mix(std::uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
// Independent host ranking with the same counter sequence, using double log
// so the reference does not simply call the production score kernel.
std::vector<int> oracle(const std::vector<float>& weights, std::size_t k, std::uint64_t seed) {
    std::vector<std::pair<double, int>> scores;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (!std::isfinite(weights[i]) || !(weights[i] > 0))
            continue;
        const auto random = mix(seed + 0x9e3779b97f4a7c15ULL * (i + 1));
        const float u = std::clamp((float(random >> 41) + .5F) * 0x1p-23F, 1e-7F, 1.F - 1e-7F);
        scores.emplace_back(std::log(double(weights[i])) - std::log(-std::log(double(u))), int(i));
    }
    std::sort(scores.begin(), scores.end(), std::greater<>());
    std::vector<int> indices(std::min(k, scores.size()));
    for (std::size_t i = 0; i < indices.size(); ++i)
        indices[i] = scores[i].second;
    return indices;
}
std::vector<int> sample(const std::vector<float>& w, std::size_t k, std::uint64_t seed) {
    auto t = Tensor::from_vector(w, {w.size()}, cuda);
    const auto result = igs_weighted_sample_cuda(t, k, seed);
    if (!result.is_valid())
        return {};
    require(
        result.device() == cuda && result.dtype() == tinytensor::DataType::Int32,
        "Result must be CUDA Int32");
    return result.to_vector_int();
}
void correctness() {
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const auto inf = std::numeric_limits<float>::infinity();
    for (std::size_t n : {1U, 31U, 255U, 256U, 257U, 2049U}) {
        std::vector<float> w(n);
        for (std::size_t i = 0; i < n; ++i)
            w[i] = std::exp(float(int(i % 41) - 20) * .4F);
        for (std::size_t k : {std::size_t(0), std::size_t(1), n / 4, n, n + 1}) {
            const auto a = sample(w, k, 42), b = sample(w, k, 42), expected = oracle(w, k, 42);
            require(a == b, "Same CUDA seed must reproduce the same order");
            require(a == expected, "GPU top-k differs from independent Gumbel ranking");
            require(
                std::set<int>(a.begin(), a.end()).size() == a.size(),
                "Sample contains duplicate parents");
        }
    }
    const std::vector<float> edge{
        0,
        -0.F,
        -1,
        nan,
        inf,
        -inf,
        1e-38F,
        std::numeric_limits<float>::denorm_min(),
        1,
        9,
        std::numeric_limits<float>::max()};
    const auto selected = sample(edge, 99, 123);
    require(
        std::set<int>(selected.begin(), selected.end()) == std::set<int>({6, 7, 8, 9, 10}),
        "Invalid weight filtering or subnormal handling changed");
    require(sample(std::vector<float>(257, 0), 10, 42).empty(), "Zero weights must emit no parent");
    require(
        sample({nan, inf, -1}, 100, 42).empty(), "Nonfinite/negative weights must emit no parent");
    require(sample({}, 1, 42).empty(), "Empty weights must emit no parent");
    // A single huge row must not occupy all k slots; selection is without replacement.
    std::vector<float> dominant(257, 1.F);
    dominant[17] = 1e30F;
    const auto heavy = sample(dominant, 64, 123);
    require(heavy.size() == 64 && heavy.front() == 17, "Heavy parent priority changed");
    require(std::set<int>(heavy.begin(), heavy.end()).size() == 64, "Heavy parent repeated");
    // Repeated seeds sample the intended weighted distribution, not uniform IDs.
    const auto pair = Tensor::from_vector({1.F, 9.F}, {2}, cuda);
    unsigned high = 0;
    for (unsigned seed = 0; seed < 512; ++seed)
        high += igs_weighted_sample_cuda(pair, 1, seed).to_vector_int()[0] == 1;
    require(high > 420 && high < 500, "Weighted sampling distribution is biased");
    auto w = Tensor::from_vector(std::vector<float>(2049, 1.F), {2049}, cuda);
    const auto a = igs_weighted_sample_cuda(w, 128, 42).to_vector_int();
    const auto b = igs_weighted_sample_cuda(w, 128, 43).to_vector_int();
    require(a != b, "Changing the CUDA seed must change selection");
    // Async callers can consume default-stream weights on a non-blocking
    // compute stream; scratch must outlive the queued sort/unpack kernels.
    cudaStream_t stream{};
    require(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess,
        "Create test stream");
    {
        tinytensor::CUDAStreamGuard guard(stream);
        for (unsigned repeat = 0; repeat < 8; ++repeat) {
            const auto queued = igs_weighted_sample_cuda(w, 128, 42);
            // Tensor::to_vector_int uses the default stream, so explicitly
            // order this host read after work on the non-blocking stream.
            std::vector<int> observed(queued.numel());
            require(
                cudaMemcpyAsync(
                    observed.data(), queued.data_ptr(), queued.bytes(), cudaMemcpyDeviceToHost,
                    stream) == cudaSuccess,
                "Read stream result");
            require(cudaStreamSynchronize(stream) == cudaSuccess, "Wait stream result");
            require(observed == a, "Non-default stream selection changed");
        }
    }
    require(cudaStreamDestroy(stream) == cudaSuccess, "Destroy test stream");
    // Integration: replacements have priority, oversized rows cannot evict them,
    // growth cannot repeat either set, and the final union respects capacity.
    using namespace photara::splat::densification;
    auto replacement = Tensor::from_vector({1.F, 0.F, 0.F, 0.F, 0.F, 0.F}, {6}, cuda);
    auto oversize = Tensor::from_vector({100.F, 10.F, 9.F, 0.F, 0.F, 0.F}, {6}, cuda);
    auto growth = Tensor::from_vector({10.F, 10.F, 10.F, 1.F, 2.F, 3.F}, {6}, cuda);
    std::mt19937 rng(42), repeat(42);
    auto picked = select_igs_parents(replacement, oversize, growth, 1, 3, 4, &rng);
    auto again = select_igs_parents(replacement, oversize, growth, 1, 3, 4, &repeat);
    const auto ids = picked.parents.to_vector_int();
    require(
        ids == again.parents.to_vector_int(), "Seeded integrated selection is not reproducible");
    require(
        picked.replacement == 1 && picked.oversized == 2 && picked.growth == 1 && ids.size() == 4,
        "Selection budgets changed");
    require(
        std::set<int>(ids.begin(), ids.end()).size() == 4 &&
            std::count(ids.begin(), ids.end(), 0) == 1,
        "Parent union is not disjoint");
    sync();
    std::cout << "CUDA IGS selection correctness passed; heavy frequency=" << high << "/512\n";
}
// Exact previous CUDA boundary, retained only as a benchmark reference:
// full download, mt19937 draws + host logs/partial_sort, then index upload.
Tensor old_host_sample(const Tensor& weights, std::size_t k) {
    const auto w = weights.to_vector();
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uniform(1e-7F, 1.F - 1e-7F);
    std::vector<std::pair<float, int>> scores;
    scores.reserve(w.size());
    for (std::size_t i = 0; i < w.size(); ++i)
        if (std::isfinite(w[i]) && w[i] > 0) {
            const float u = uniform(rng);
            scores.emplace_back(std::log(w[i]) - std::log(-std::log(u)), int(i));
        }
    k = std::min(k, scores.size());
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(), std::greater<>());
    std::vector<int> ids(k);
    for (std::size_t i = 0; i < k; ++i)
        ids[i] = scores[i].second;
    return Tensor::from_vector(ids, {k}, cuda);
}
void benchmark(std::size_t n) {
    std::vector<float> w(n);
    for (std::size_t i = 0; i < n; ++i)
        w[i] = .001F + float((i * 17) % 1000) * .001F;
    const auto weights = Tensor::from_vector(w, {n}, cuda);
    for (const std::size_t k : {std::max<std::size_t>(1, n / 100), n / 4}) {
        for (unsigned warm = 0; warm < 2; ++warm)
            (void)igs_weighted_sample_cuda(weights, k, 42);
        sync();
        for (unsigned round = 0; round < 3; ++round)
            for (unsigned arm = 0; arm < 2; ++arm) {
                const bool gpu = (round % 2 ? 1 - arm : arm) != 0;
                sync();
                cudaMemPool_t pool{};
                if (gpu) {
                    require(
                        cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess, "Default CUDA pool");
                    std::uint64_t zero = 0;
                    require(
                        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero) ==
                            cudaSuccess,
                        "Reset selection memory peak");
                }
                const auto start = std::chrono::steady_clock::now();
                const auto result =
                    gpu ? igs_weighted_sample_cuda(weights, k, 42) : old_host_sample(weights, k);
                sync();
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
                require(result.numel() == k, "Benchmark selection lost parents");
                std::uint64_t peak = 0;
                if (gpu) {
                    require(
                        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &peak) ==
                            cudaSuccess,
                        "Read selection memory peak");
                    // Outside the timing: verify every returned index at scale.
                    const auto ids = result.to_vector_int();
                    std::vector<bool> seen(n, false);
                    for (const int id : ids) {
                        require(
                            id >= 0 && std::size_t(id) < n, "Large selection index out of range");
                        require(!seen[id], "Large selection contains duplicates");
                        seen[id] = true;
                    }
                }
                std::cout << "igs_selection_bench n=" << n << " k=" << k << " round=" << round
                          << " backend=" << (gpu ? "cuda" : "old_cpu") << " ms=" << ms
                          << " peak_bytes=" << peak << std::endl;
            }
    }
}
int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--bench")
            benchmark(argc > 2 ? std::stoul(argv[2]) : 1000000);
        else
            correctness();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
