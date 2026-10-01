#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include <vulkan/vulkan.h>

namespace tinytensor { class Tensor; }

namespace tinytensor {
namespace vulkan {

// Device-level capabilities for the independent compute VkDevice.
struct DeviceInfo {
    std::string name;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t api_version = 0;
    std::uint32_t subgroup_size = 0;
    bool shader_float64 = false;
    bool buffer_atomic_f32 = false;
    bool buffer_atomic_f64 = false;
    bool push_descriptors = false;
};

// Zero-copy bridge for compute libraries that can adopt an existing Vulkan
// device (splat_drender does).  The handles remain owned by TinyTensor and are
// valid until shutdown().  Call synchronize() before an external submit if
// TinyTensor may have an open command batch.
struct DeviceHandles {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = VK_QUEUE_FAMILY_IGNORED;
};

struct BufferView {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize bytes = 0;
};

struct AdamStepOptions {
    float learning_rate = 1e-3F;
    float secondary_learning_rate = 0.0F;
    std::uint32_t group_stride = 0;
    // When non-zero, indices whose column within group_stride is outside this
    // prefix are left completely untouched, including both moments.
    std::uint32_t active_row_stride = 0;
    float beta1 = 0.9F;
    float beta2 = 0.999F;
    float correction1 = 1.0F;
    float correction2 = 1.0F;
    float epsilon = 1e-8F;
    float clamp_min = -std::numeric_limits<float>::infinity();
    float clamp_max = std::numeric_limits<float>::infinity();
    // Optional L2 gradient applied to non-DC columns (column >= 3) of each
    // group. Used by shared 3DGS SH regularization without a temporary tensor.
    float grouped_rest_regularization = 0.0F;
    // For [N,3] log scales, update one row and project it in the same dispatch.
    float max_scale_ratio = 0.0F;
};

// True when the library was built with TINYTENSOR_HAS_VULKAN and a compute
// device can be created.
bool available();
const DeviceInfo& device_info();
DeviceHandles device_handles();
BufferView buffer_view(const Tensor& tensor);
void adam_step(Tensor& parameter, const Tensor& gradient, Tensor& first, Tensor& second,
               const AdamStepOptions& options);
// SH Adam codec shared with splat_drender CUDA: FP16 normalized update,
// uint8 log-second moment, and FP32 [N,4] DC/rest bounds. Arithmetic is FP32.
void sh_adam_quant_step(Tensor& parameter, const Tensor& gradient, Tensor& first,
                        Tensor& packed, Tensor& bounds,
                        const AdamStepOptions& options);
// Projects each Gaussian's [N,3] log scales so the longest/shortest axis
// ratio stays within maximum_ratio (> 1). No-op for compliant rows.
void constrain_scale_ratio(Tensor& log_scales, float maximum_ratio);
void synchronize();
// Submit pending work for a subsequent compute stage on the same Vulkan queue.
// Unlike synchronize(), this does not wait for GPU completion.
void submit_async();
// Cumulative host-side counters, independent of the timestamp profiler:
// wall milliseconds spent blocked in vkWaitForFences, and the number of
// dispatches recorded. Use the difference between two samples to attribute a
// phase: a phase whose wall time is mostly wait_ms is stalled on earlier work.
double device_wait_ms();
std::uint64_t dispatch_count();
// Device busy milliseconds accumulated by the timestamp profiler. Only moves
// when TINYTENSOR_VULKAN_PROFILE_OPS is on; a phase whose wall time is not
// covered by wait_ms or busy_ms is spending it on the host, in allocation or
// in command recording rather than in the GPU or the fence.
double device_busy_ms();
// Buffer pool traffic since startup: reuse, fresh allocation (with the bytes
// it asked the driver for) and teardown. A phase whose wall time is host-heavy
// while its miss bytes grow by the size of its working set is churning
// allocations rather than reusing them.
struct BufferPoolStats {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t miss_bytes = 0;
    std::uint64_t drops = 0;
    std::uint64_t drop_bytes = 0;
    std::uint64_t reserved_bytes = 0;
    std::uint64_t free_bytes = 0;
    std::uint64_t live_bytes = 0;
    std::uint64_t budget_bytes = 0;
};
BufferPoolStats buffer_pool_stats();
void shutdown();

} // namespace vulkan
} // namespace tinytensor
