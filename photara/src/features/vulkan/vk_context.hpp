#pragma once

// Minimal Vulkan compute plumbing owned by the feature backends. One process
// wide device is shared by every extractor/matcher clone, and recording is per
// thread: backend clones driven from a worker pool each own a command buffer,
// fence, descriptor set batch and download list, so parallel matching never
// interleaves on shared state. Pipeline creation, descriptor allocation and
// queue submission serialize on the context mutex. Host-visible scratch is
// written directly through mapped memory; device-local work buffers are
// recycled through a size-bucketed pool so repeated work does not keep asking
// the driver for memory.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include <vulkan/vulkan.h>

namespace photara::features::vulkan_backend {

enum class Shader : std::uint32_t {
    Convert = 0,
    Upsample,
    Downsample,
    Blur,
    Dog,
    Key,
    HistInit,
    HistReduce,
    ListGen,
    Orientation,
    Descriptor,
    DescriptorNorm,
    MatchTiles,
    MatchFinish,
    Count
};

struct Buffer {
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer handle_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    void* mapped_ = nullptr;
    bool coherent_ = true;
    VkDeviceSize non_coherent_atom_size_ = 1;

    Buffer() = default;
    Buffer(VkPhysicalDevice physical, VkDevice logical, VkDeviceSize bytes,
           bool host_visible, bool host_cached);
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    [[nodiscard]] void* map();
    [[nodiscard]] VkBuffer handle() const noexcept { return handle_; }
    [[nodiscard]] void* mapped_host() const noexcept { return mapped_; }
    [[nodiscard]] VkDeviceSize size_bytes() const noexcept { return size_; }
    void unmap() noexcept;
    // Invalidate non-coherent mapped memory after the device wrote to it.
    void invalidate() const;
    // Flush non-coherent mapped memory after the host wrote to it.
    void flush(VkDeviceSize offset, VkDeviceSize bytes) const;
};

// RAII view of a pooled buffer. Returning it to the pool keeps the allocation
// alive for the next extraction/match with a similar shape.
class BufferRef {
public:
    BufferRef() = default;
    BufferRef(class Context& context, std::unique_ptr<Buffer> buffer,
              bool host_visible, std::size_t bucket);
    ~BufferRef();
    BufferRef(BufferRef&& other) noexcept;
    BufferRef& operator=(BufferRef&& other) noexcept;
    BufferRef(const BufferRef&) = delete;
    BufferRef& operator=(const BufferRef&) = delete;

    [[nodiscard]] Buffer& operator*() const { return *buffer_; }
    [[nodiscard]] Buffer* operator->() const { return buffer_.get(); }
    [[nodiscard]] bool valid() const noexcept { return buffer_ != nullptr; }

private:
    Context* context_ = nullptr;
    std::unique_ptr<Buffer> buffer_;
    bool host_visible_ = false;
    std::size_t bucket_ = 0;
};

struct BufferBinding {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize range = VK_WHOLE_SIZE;
};

class Context {
public:
    static Context* try_get();
    static Context& get();      // throws when Vulkan is unavailable
    static bool available();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    ~Context();

    [[nodiscard]] std::uint32_t device_index() const noexcept { return device_index_; }
    [[nodiscard]] VkPhysicalDeviceMemoryProperties memory_properties() const;

    // Pooled allocation. host_visible buffers are directly mappable;
    // device-local buffers are recycled the same way.
    [[nodiscard]] BufferRef allocate(VkDeviceSize bytes, bool host_visible,
                                     bool host_cached = false);

    // ---- recording session -------------------------------------------------
    // begin() starts the calling thread's command buffer; dispatch() appends a
    // compute dispatch (a full compute-stage memory barrier separates every
    // dispatch, which is what the multi-pass SIFT pipeline relies on);
    // submit_and_wait() hands the batch to the queue and waits for completion
    // before returning so the caller can read results and continue recording
    // later phases on the same session.
    void begin();
    void dispatch(Shader shader, std::span<const BufferBinding> bindings,
                  const void* push_constants, std::uint32_t push_bytes,
                  std::uint32_t groups_x, std::uint32_t groups_y = 1,
                  std::uint32_t groups_z = 1);
    // Records a vkCmdFillBuffer (byte offsets/sizes are rounded down to 4).
    void fill_u32(const Buffer& destination, VkDeviceSize offset, VkDeviceSize bytes,
                  std::uint32_t value);
    // Records a device-to-device copy into the host visible `destination`
    // buffer; the bytes become readable from destination.map() after the next
    // submit_and_wait().
    void queue_download(const Buffer& source, VkDeviceSize source_offset,
                        VkDeviceSize bytes, const Buffer& destination,
                        VkDeviceSize destination_offset);
    void submit_and_wait();

private:
    Context();

    struct Pipeline {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline handle = VK_NULL_HANDLE;
        std::uint32_t binding_count = 0;
        std::uint32_t push_bytes = 0;
    };

    [[nodiscard]] const Pipeline& pipeline(Shader shader);
    void destroy_pipeline(Pipeline& pipeline) noexcept;

    [[nodiscard]] std::uint32_t find_memory_type(
        std::uint32_t type_bits,
        VkMemoryPropertyFlags required) const;

    [[nodiscard]] VkDescriptorSet acquire_set(const Pipeline& pipeline);
    void write_set(VkDescriptorSet set, const Pipeline& pipeline,
                   std::span<const BufferBinding> bindings);

    struct Download {
        const Buffer* source;
        VkDeviceSize source_offset;
        VkDeviceSize bytes;
        const Buffer* destination;
        VkDeviceSize destination_offset;
    };

    // One recording session per thread. Sessions are owned by the context so
    // their command buffers and fences outlive the threads that used them.
    struct Session {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool recording = false;
        std::vector<VkDescriptorSet> sets;
        std::vector<Download> downloads;
    };

    [[nodiscard]] Session& session();
    void destroy_session(Session& session) noexcept;

    static VkDeviceSize pooled_size(VkDeviceSize bytes);
    void recycle(std::unique_ptr<Buffer> buffer, bool host_visible,
                 std::size_t bucket);
    friend class BufferRef;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    std::uint32_t device_index_ = 0;
    VkPhysicalDeviceMemoryProperties memory_properties_{};
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;

    Pipeline pipelines_[static_cast<std::size_t>(Shader::Count)]{};

    std::vector<std::unique_ptr<Session>> sessions_;

    struct PoolBucket {
        std::vector<std::unique_ptr<Buffer>> free_buffers;
    };
    std::mutex mutex_;
    std::vector<PoolBucket> device_pools_;
    std::vector<PoolBucket> host_pools_;

    bool alive_ = true;
};

}  // namespace photara::features::vulkan_backend

