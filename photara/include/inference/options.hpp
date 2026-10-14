#pragma once
#include <cstdint>
#include <functional>
#include <stop_token>

namespace photara::inference {
enum class Backend { vulkan, cpu };
struct RuntimeOptions {
    Backend backend = Backend::vulkan;
    std::uint32_t device_index = 0; // Index within the selected backend.
    bool allow_cpu_fallback = true; // Initialization only, never mid-graph.
    int cpu_threads = 0; // 0 selects the hardware thread count.
};
struct PredictOptions {
    std::stop_token stop;
    // Optional callback on the calling thread, with a value in [0, 1].
    std::function<void(float)> progress;
};
} // namespace photara::inference
