#pragma once
#include "splat/types.hpp"
#include <cstdint>

namespace photara::splat::densification::gpu_detail {
// Gumbel top-k without replacement. Scores/sort/indices stay on CUDA; only
// the eligible count (one uint32) is read back to size the returned Int32 tensor.
// Equal float scores prefer the larger row index, matching the CPU pair order.
// Seeded determinism is local to CUDA, not the former mt19937 per-row sequence.
tinytensor::Tensor igs_weighted_sample_cuda(
    const tinytensor::Tensor& weights,
    std::size_t requested,
    std::uint64_t seed,
    double* prepare_ms = nullptr,
    double* sort_ms = nullptr);
} // namespace photara::splat::densification::gpu_detail
