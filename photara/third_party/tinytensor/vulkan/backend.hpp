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
};

// True when the library was built with TINYTENSOR_HAS_VULKAN and a compute
// device can be created.
bool available();
const DeviceInfo& device_info();
DeviceHandles device_handles();
BufferView buffer_view(const Tensor& tensor);
void adam_step(Tensor& parameter, const Tensor& gradient, Tensor& first, Tensor& second,
               const AdamStepOptions& options);
void synchronize();
void shutdown();

} // namespace vulkan
} // namespace tinytensor
