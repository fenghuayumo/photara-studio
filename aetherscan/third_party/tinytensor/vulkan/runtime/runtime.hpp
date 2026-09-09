#pragma once

#include "vulkan/backend.hpp"
#include "vulkan/runtime/check.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>

namespace tinytensor::vulkan::runtime {

enum class ShaderId : std::uint32_t {
    Elementwise = 0,
    StridedCopy,
    IndexSelect,
    IndexFill,
    MaskFlags,
    ScanBlock,
    ScanAdd,
    Compact,
    Multinomial,
    Count
};

struct BufferBinding {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize range = VK_WHOLE_SIZE;
};

class Buffer {
public:
    Buffer() = default;
    Buffer(VkPhysicalDevice physical, VkDevice logical, VkDeviceSize bytes, bool host_visible);
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    [[nodiscard]] VkBuffer handle() const { return handle_; }
    [[nodiscard]] VkDeviceSize size() const { return size_; }
    [[nodiscard]] bool host_visible() const { return mapped_ != nullptr; }
    [[nodiscard]] void* mapped() const { return mapped_; }
    [[nodiscard]] BufferBinding binding(VkDeviceSize offset = 0,
                                        VkDeviceSize range = VK_WHOLE_SIZE) const;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer handle_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    void* mapped_ = nullptr;
};

class Context {
public:
    static Context& get();
    static bool available();
    static void shutdown();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context();

    [[nodiscard]] const DeviceInfo& info() const { return info_; }
    [[nodiscard]] VkDevice device() const { return device_; }

    [[nodiscard]] std::shared_ptr<Buffer> alloc(std::size_t bytes);
    void fill_zero(Buffer& buffer, std::size_t offset, std::size_t bytes);
    void copy(Buffer& dst, std::size_t dst_offset, const Buffer& src, std::size_t src_offset,
              std::size_t bytes);
    void upload(Buffer& dst, std::size_t dst_offset, const void* data, std::size_t bytes);
    void download(const Buffer& src, std::size_t src_offset, void* data, std::size_t bytes);

    void dispatch(ShaderId shader,
                  std::span<const BufferBinding> bindings,
                  const void* push_constants,
                  std::uint32_t push_bytes,
                  std::uint32_t groups_x,
                  std::uint32_t groups_y = 1,
                  std::uint32_t groups_z = 1);

    [[nodiscard]] Buffer& dummy();

private:
    Context();

    void create_instance();
    void pick_device();
    void create_device();
    void create_pools();
    void create_pipelines();
    void destroy();
    void submit(VkCommandBuffer cmd);
    [[nodiscard]] VkCommandBuffer begin_commands();
    void ensure_staging(std::size_t bytes);
    void barrier_buffer(VkCommandBuffer cmd, VkBuffer buffer,
                        VkAccessFlags src_access, VkAccessFlags dst_access,
                        VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage);

    DeviceInfo info_{};
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    bool validation_ = false;

    struct Pipeline {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline handle = VK_NULL_HANDLE;
        std::uint32_t binding_count = 0;
        std::uint32_t push_size = 0;
    };
    Pipeline pipelines_[static_cast<std::size_t>(ShaderId::Count)]{};

    std::unique_ptr<Buffer> staging_;
    std::unique_ptr<Buffer> dummy_;
    std::mutex mutex_;
};

} // namespace tinytensor::vulkan::runtime
