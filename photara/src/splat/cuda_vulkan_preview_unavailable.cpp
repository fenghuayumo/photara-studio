#include "splat/cuda_vulkan_preview.hpp"
#include <stdexcept>
namespace photara::splat {
struct CudaVulkanPreview::Impl {};
CudaVulkanPreview::CudaVulkanPreview(const CudaVulkanPreviewOptions&) {
    throw std::runtime_error("CUDA/Vulkan interop is unavailable in this build; use Vulkan preview");
}
CudaVulkanPreview::~CudaVulkanPreview() = default;
CudaVulkanPreview::CudaVulkanPreview(CudaVulkanPreview&&) noexcept = default;
CudaVulkanPreview& CudaVulkanPreview::operator=(CudaVulkanPreview&&) noexcept = default;
void CudaVulkanPreview::submit(const tinytensor::Tensor&, std::uint32_t, std::uint32_t) {
    throw std::runtime_error("CUDA/Vulkan interop is unavailable in this build");
}
std::uint64_t CudaVulkanPreview::frame_count() const noexcept { return 0; }
}
