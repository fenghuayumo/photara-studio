#pragma once

#include "internal/tensor_impl.hpp"

#include <cstdint>
#include <memory>

namespace photara::splat {

struct CudaVulkanPreviewOptions {
    std::uint64_t memory_handle{};
    std::uint64_t semaphore_handle{};
    std::uint64_t allocation_size{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint64_t device_luid{};
    std::uint32_t device_node_mask{};
};

class CudaVulkanPreview {
public:
    explicit CudaVulkanPreview(const CudaVulkanPreviewOptions& options);
    ~CudaVulkanPreview();
    CudaVulkanPreview(CudaVulkanPreview&&) noexcept;
    CudaVulkanPreview& operator=(CudaVulkanPreview&&) noexcept;
    CudaVulkanPreview(const CudaVulkanPreview&) = delete;
    CudaVulkanPreview& operator=(const CudaVulkanPreview&) = delete;

    // Scales the CUDA planar RGB tensor into the shared RGBA8 Vulkan image,
    // then signals timeline value 2*N-1. Before frame N it waits for Vulkan
    // to signal 2*(N-1), so the image is never overwritten while sampled.
    void submit(const tinytensor::Tensor& planar_rgb,
                std::uint32_t source_width,
                std::uint32_t source_height);
    [[nodiscard]] std::uint64_t frame_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace photara::splat
