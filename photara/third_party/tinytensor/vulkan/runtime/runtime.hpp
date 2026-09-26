#pragma once

#include "vulkan/backend.hpp"
#include "vulkan/runtime/check.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

namespace tinytensor::vulkan::runtime {

enum class ShaderId : std::uint32_t {
    AdamF32 = 0,
    Elementwise,
    FusedPointwise,
    StridedCopy,
    IndexSelect,
    IndexFill,
    MaskFlags,
    ScanBlock,
    ScanAdd,
    Compact,
    Multinomial,
    Reduce,
    ReduceAllF32,
    Matmul,
    Random,
    Cumsum,
    Pool,
    Scatter,
    SelectCompact,
    Cat,
    UnpackRgba,
    Count
};

struct BufferBinding {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize range = VK_WHOLE_SIZE;
};

// Where a buffer lives. The upload staging buffer is written by the CPU and
// read by the GPU, so plain host-visible (write-combined) memory is right for
// it; a readback buffer is read back by the CPU, where cached host memory is
// an order of magnitude faster to read.
enum class MemoryKind {
    device_local,
    host_visible,
    host_cached,
};

class Buffer {
public:
    Buffer() = default;
    Buffer(VkPhysicalDevice physical, VkDevice logical, VkDeviceSize bytes,
           MemoryKind kind = MemoryKind::device_local);
    ~Buffer();
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    [[nodiscard]] VkBuffer handle() const { return handle_; }
    [[nodiscard]] VkDeviceMemory memory() const { return memory_; }
    [[nodiscard]] VkDeviceSize size() const { return size_; }
    [[nodiscard]] bool host_visible() const { return mapped_ != nullptr; }
    [[nodiscard]] void* mapped() const { return mapped_; }
    [[nodiscard]] bool coherent() const { return coherent_; }
    // Non-coherent mapped memory has to be invalidated in whole atoms before
    // the host may read what the device wrote.
    [[nodiscard]] VkDeviceSize non_coherent_atom_size() const { return non_coherent_atom_size_; }
    [[nodiscard]] BufferBinding binding(VkDeviceSize offset = 0,
                                        VkDeviceSize range = VK_WHOLE_SIZE) const;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer handle_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    void* mapped_ = nullptr;
    bool coherent_ = true;
    VkDeviceSize non_coherent_atom_size_ = 1;
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
    [[nodiscard]] VkInstance instance() const { return instance_; }
    [[nodiscard]] VkPhysicalDevice physical_device() const { return physical_; }
    [[nodiscard]] VkDevice device() const { return device_; }
    [[nodiscard]] VkQueue queue() const { return queue_; }
    [[nodiscard]] std::uint32_t queue_family() const { return queue_family_; }

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

    // Submits every command recorded so far and waits for the queue to drain.
    // Ops record into one command buffer and submit once per flush instead of
    // once per op; a host readback (download) and shutdown flush implicitly.
    // Call it explicitly to bound how much work one submission carries.
    void flush();

    // Per-op GPU timing. Enabled with TINYTENSOR_VULKAN_PROFILE_OPS=1: every
    // recorded dispatch gets a timestamp pair, the deltas are accumulated per
    // shader and per element-count bucket (a shader run over 630k parameters
    // and one over 10M spherical-harmonic values are different costs), and a
    // table is printed to stderr every
    // TINYTENSOR_VULKAN_PROFILE_OPS_INTERVAL flushes (default 200). A flush
    // window is what a caller's synchronize() sees, so one window is normally
    // one training stage. Disabled by default: two timestamp writes per op are
    // not free.
    struct OpProfileEntry {
        const char* name = "";
        std::uint64_t element_bucket = 0;
        std::uint64_t calls = 0;
        double total_ms = 0.0;
    };
    [[nodiscard]] std::vector<OpProfileEntry> op_profile() const;
    [[nodiscard]] std::uint32_t op_profile_flushes() const { return op_profile_.flushes; }

    [[nodiscard]] Buffer& dummy();

private:
    Context();

    void create_instance();
    void pick_device();
    void create_device();
    void create_pools();
    void create_pipelines();
    void destroy();

    // Batch recording. flush_locked() is the only place that submits.
    void begin_batch_locked();
    void flush_locked();
    void record_barrier_locked();
    [[nodiscard]] VkDescriptorSet acquire_descriptor_locked(VkDescriptorSetLayout layout);

    // Buffer reuse: a released Buffer goes back to the pool instead of being
    // destroyed, so a repeated shape stops paying vkCreateBuffer and
    // vkAllocateMemory. Buffers are only destroyed while no batch is recording,
    // because a recorded command still holds their handle.
    [[nodiscard]] std::shared_ptr<Buffer> acquire_buffer_locked(std::size_t bytes);
    void recycle_locked(Buffer* buffer);
    void trim_pool_locked();

    void ensure_staging(std::size_t bytes);
    void ensure_readback_staging(std::size_t bytes);
    void barrier_buffer(VkCommandBuffer cmd, VkBuffer buffer,
                        VkAccessFlags src_access, VkAccessFlags dst_access,
                        VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage);
    void create_op_profile_locked();
    void collect_op_profile_locked();
    void report_op_profile_locked();

    DeviceInfo info_{};
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    std::uint32_t queue_family_ = 0;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkFence completion_fence_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    PFN_vkCmdPushDescriptorSetKHR cmd_push_descriptor_ = nullptr;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    bool validation_ = false;
    bool push_descriptors_ = false;

    struct Pipeline {
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline handle = VK_NULL_HANDLE;
        std::uint32_t binding_count = 0;
        std::uint32_t push_size = 0;
    };
    Pipeline pipelines_[static_cast<std::size_t>(ShaderId::Count)]{};

    std::unordered_map<std::size_t, std::vector<Buffer*>> free_buffers_;
    std::size_t pooled_bytes_ = 0;
    std::size_t pool_budget_bytes_ = 0;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    bool recording_ = false;
    std::uint32_t batch_commands_ = 0;
    std::size_t staging_cursor_ = 0;
    std::vector<VkDescriptorBufferInfo> descriptor_infos_;
    std::vector<VkWriteDescriptorSet> descriptor_writes_;
    // Cleared by destroy(). A tensor that outlives the context then leaks its
    // buffer instead of touching a dead device.
    bool alive_ = true;
    std::unique_ptr<Buffer> staging_;
    std::unique_ptr<Buffer> readback_staging_;
    std::unique_ptr<Buffer> dummy_;

    struct OpProfile {
        // Element-count buckets are powers of two: one entry per shader and
        // size class keeps the table readable.
        static constexpr std::uint32_t kElementBuckets = 26;
        bool enabled = false;
        VkQueryPool pool = VK_NULL_HANDLE;
        std::uint32_t interval = 200;
        std::uint32_t flushes = 0;
        // Timestamp pairs recorded into the batch that is currently open.
        std::uint32_t recorded = 0;
        std::vector<std::uint32_t> order;
        std::vector<std::uint64_t> calls;
        std::vector<double> total_ms;
        float period = 0.0F;
        std::uint32_t valid_bits = 0;
    };
    OpProfile op_profile_;
    std::mutex mutex_;
};

} // namespace tinytensor::vulkan::runtime
