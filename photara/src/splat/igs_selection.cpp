#include "cuda_backend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace photara::splat::densification::gpu_detail {
tinytensor::Tensor igs_weighted_sample_cuda(
    const tinytensor::Tensor& weights,
    const std::size_t requested,
    const std::uint64_t seed,
    double* prepare_ms,
    double* sort_ms) {

#if defined(PHOTARA_HAS_CUDA_SPLAT)
    if (weights.device() == tinytensor::Device::CUDA)
        return cuda_backend::igs_weighted_sample_cuda(
            weights, requested, seed, prepare_ms, sort_ms);
#endif
    throw std::runtime_error("igs_weighted_sample_cuda: this operation requires the CUDA backend");
}

} // namespace photara::splat::densification::gpu_detail
