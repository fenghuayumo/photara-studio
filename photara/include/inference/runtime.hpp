#pragma once
#include "inference/options.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace photara::inference {
struct DeviceInfo {
    Backend backend = Backend::cpu;
    std::uint32_t index = 0;
    std::string name;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0; // 0 means unknown; memory is a snapshot.
};
[[nodiscard]] std::vector<DeviceInfo> available_devices();
[[nodiscard]] const char* backend_name(Backend backend) noexcept;
} // namespace photara::inference
