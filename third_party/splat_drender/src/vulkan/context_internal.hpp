#pragma once

// Vulkan plumbing shared by the splat_drender Vulkan passes: a mapped buffer, a
// compute pipeline built from embedded SPIR-V, and the device/queue/pool owner.
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <string>

#include <vulkan/vulkan.h>

#include "splat_drender/vulkan_api.h"

namespace splat_drender::vulkan {

// Where a buffer lives. The rasterizer keeps its per-frame working set in
// device memory, which is what makes the compute passes run at VRAM bandwidth,
// and only crosses to the host through the context's staging buffer.
enum class BufferMemory : std::uint32_t {
    // Mapped host memory. Cached when the device offers it (uncached reads are
    // ~20x slower), coherent when we can get it, otherwise invalidated/flushed.
    host_visible,
    // Host-visible and device-local (resizable BAR). Mappable, but the GPU
    // reaches it over PCIe.
    host_visible_device_local,
    // VRAM. Not mappable: every host access goes through staging.
    device_local,
};

struct Buffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
    bool coherent = true;
    VkDeviceSize non_coherent_atom_size = 1;

    Buffer() = default;
    Buffer(
        VkPhysicalDevice physical_device,
        VkDevice logical_device,
        VkDeviceSize byte_size,
        VkBufferUsageFlags usage,
        BufferMemory memory_kind = BufferMemory::host_visible);
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    [[nodiscard]] bool host_visible() const noexcept { return mapped != nullptr; }
    void upload(const void* source, std::size_t byte_size, std::size_t offset = 0) const;
    void download(void* destination, std::size_t byte_size, std::size_t offset = 0) const;
};

struct ComputePipeline {
    VkDevice device = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;
    std::uint32_t binding_count = 0;

    ComputePipeline() = default;
    ComputePipeline(
        VkDevice logical_device,
        std::span<const std::byte> spir_v_bytes,
        std::uint32_t storage_buffer_count,
        std::uint32_t push_constant_size);
    ~ComputePipeline();
    ComputePipeline(ComputePipeline&& other) noexcept;
    ComputePipeline& operator=(ComputePipeline&& other) noexcept;
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
};

class Context::Impl {
public:
    explicit Impl(const ContextOptions& options);
    ~Impl();

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    [[nodiscard]] Buffer create_buffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        BufferMemory memory = BufferMemory::host_visible) const;
    [[nodiscard]] ComputePipeline create_pipeline(
        const std::string& shader_name,
        std::uint32_t storage_buffer_count,
        std::uint32_t push_constant_size) const;

    // Host transfers for buffers that are not mapped. Both go through one
    // staging buffer and one submit; callers must hold dispatch_mutex.
    void write_buffer(
        const Buffer& destination, const void* source, std::size_t byte_size,
        std::size_t offset = 0);
    void read_buffer(
        const Buffer& source, void* destination, std::size_t byte_size,
        std::size_t offset = 0);
    // Device-side memset, for the zero fills the host used to do on a mapping.
    void fill_buffer(const Buffer& destination, std::uint32_t value, std::size_t byte_size);
    // Device-to-device copy, so a reordering stays on the GPU.
    void copy_buffer(const Buffer& source, const Buffer& destination, std::size_t byte_size) {
        copy_between(source, 0, destination, 0, byte_size);
    }

    DeviceInfo device_info;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family_index = 0;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
    // An adopted instance/device belongs to the caller and must survive us.
    bool owns_instance = false;
    bool owns_device = false;
    mutable std::mutex dispatch_mutex;

private:
    void create_instance_and_device(const ContextOptions& options);
    void adopt_external_device(const ExternalDevice& external);
    void create_pools();
    void copy_between(
        const Buffer& source, std::size_t source_offset, const Buffer& destination,
        std::size_t destination_offset, std::size_t byte_size);
    void ensure_staging(VkDeviceSize bytes);

    Buffer staging_;
};

}  // namespace splat_drender::vulkan
