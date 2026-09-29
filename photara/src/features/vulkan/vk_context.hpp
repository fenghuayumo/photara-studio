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
#include <condition_variable>
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
    // Descriptor-tile matcher variants: the host picks the tallest tile the
    // device's shared-memory limit allows, using the dp4a shaders when the
    // device reports shaderIntegerDotProduct.
    MatchTiles,       // 32 query rows, scalar byte dot product
    MatchTiles64,     // 64 query rows, scalar
    MatchTiles160,    // 160 query rows, scalar (48 KiB shared-memory devices)
    MatchTilesDp4a64,  // 64 query rows, integer dot product
    MatchTilesDp4a128, // 128 query rows, integer dot product
    MatchTilesDp4a160, // 160 query rows, integer dot product
    MatchWaveDp4a,     // 512 query rows, 32-lane wave reductions
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
    // True when the device enabled shaderIntegerDotProduct and can load the
    // SPIR-V 1.5 variant of the descriptor matcher.
    [[nodiscard]] bool integer_dot_product() const noexcept {
        return integer_dot_product_;
    }
    // Tile height and shader to use for the descriptor matcher, chosen from the
    // device's shared-memory limit and dot-product support.
    struct TilePlan {
        Shader shader = Shader::MatchTiles;
        std::uint32_t rows = 32;
    };
    [[nodiscard]] TilePlan descriptor_tile_plan() const noexcept;

    // Pooled allocation. host_visible buffers are directly mappable;
    // device-local buffers are recycled the same way.
    [[nodiscard]] BufferRef allocate(VkDeviceSize bytes, bool host_visible,
                                     bool host_cached = false);

    // Soft device-memory budget for concurrent feature work. One SIFT pyramid
    // keeps a few hundred megabytes alive in the buffer pool, so the SfM
    // frontend's per-worker extractor clones would otherwise exhaust the device
    // (vkAllocateMemory returns VK_ERROR_OUT_OF_DEVICE_MEMORY). Callers reserve
    // an estimate for the duration of one extraction.
    void acquire_device_budget(std::uint64_t bytes);
    void release_device_budget(std::uint64_t bytes);

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
    // Records a host-visible staging to device-local copy. Later compute
    // dispatches in the same command buffer see the copied bytes.
    void queue_upload(const Buffer& source, VkDeviceSize source_offset,
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
        // Descriptor sets come from pools owned by the session: extraction
        // records hundreds of dispatches per image, and one pool per worker
        // thread keeps parallel extractions independent. Pools are reset
        // between submissions, so nothing is freed while a batch is in flight.
        std::vector<VkDescriptorPool> pools;
        std::vector<std::uint32_t> pool_capacities;
        std::size_t sets_in_active_pool = 0;
        std::vector<VkDescriptorSet> sets;
        std::vector<Download> downloads;
    };

    [[nodiscard]] Session& session();
    [[nodiscard]] VkDescriptorSet acquire_set(const Pipeline& pipeline,
                                              Session& session);
    void reset_session_pools(Session& session);
    void grow_session_pool(Session& session);
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

    Pipeline pipelines_[static_cast<std::size_t>(Shader::Count)]{};

    std::vector<std::unique_ptr<Session>> sessions_;

    struct PoolBucket {
        std::vector<std::unique_ptr<Buffer>> free_buffers;
    };
    std::mutex mutex_;
    std::vector<PoolBucket> device_pools_;
    std::vector<PoolBucket> host_pools_;

    bool alive_ = true;
    bool integer_dot_product_ = false;
    std::uint32_t max_shared_bytes_ = 0;
    std::uint32_t subgroup_size_ = 0;

    std::mutex budget_mutex_;
    std::condition_variable budget_condition_;
    std::uint64_t budget_bytes_{0};
    std::uint64_t budget_in_use_{0};
};

}  // namespace photara::features::vulkan_backend

