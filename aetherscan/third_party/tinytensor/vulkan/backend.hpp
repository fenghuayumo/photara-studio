#pragma once

#include <cstdint>
#include <string>

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
};

// True when the library was built with TINYTENSOR_HAS_VULKAN and a compute
// device can be created.
bool available();
const DeviceInfo& device_info();
void shutdown();

} // namespace vulkan
} // namespace tinytensor
