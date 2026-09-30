#pragma once

#include "features/features.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace photara::features {

[[nodiscard]] bool sift_cuda_device_available(int device_index);

// CUDA SIFT extractor. The numeric steps follow the Vulkan shaders in
// src/features/vulkan/shaders/sift_*.hlsl (the same pyramid, DoG extrema,
// orientation, and UBC descriptor as that translation of the SiftGPU pipeline).
class SiftCudaEngine {
public:
    explicit SiftCudaEngine(const SiftGpuOptions& options);
    ~SiftCudaEngine();
    SiftCudaEngine(const SiftCudaEngine&) = delete;
    SiftCudaEngine& operator=(const SiftCudaEngine&) = delete;

    [[nodiscard]] bool available() const noexcept;
    [[nodiscard]] FeatureSet extract(
        std::span<const std::uint8_t> pixels, std::uint32_t width,
        std::uint32_t height, std::size_t row_stride);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace photara::features
