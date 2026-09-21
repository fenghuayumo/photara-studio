#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <vulkan/vulkan.h>

namespace tinytensor::vulkan::runtime {

inline void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

constexpr std::uint32_t kGroupSize = 256;

inline std::uint32_t ceil_div(std::uint32_t value, std::uint32_t divisor) {
    return divisor == 0 ? 0 : (value + divisor - 1U) / divisor;
}

inline std::uint32_t align_up(std::uint32_t value, std::uint32_t alignment) {
    return (value + alignment - 1U) & ~(alignment - 1U);
}

inline std::size_t align_up_size(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1U) & ~(alignment - 1U);
}

} // namespace tinytensor::vulkan::runtime
