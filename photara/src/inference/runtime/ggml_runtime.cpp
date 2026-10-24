#include "ggml_runtime.hpp"
#include "ggml-cpu.h"
#if defined(PHOTARA_INFERENCE_HAS_VULKAN)
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace photara::inference {
const char* backend_name(Backend backend) noexcept {
    switch (backend) {
    case Backend::vulkan: return "vulkan";
    case Backend::cpu: return "cpu";
    }
    return "unknown";
}

std::vector<DeviceInfo> available_devices() {
    std::vector<DeviceInfo> devices;
#if defined(PHOTARA_INFERENCE_HAS_VULKAN)
    const int count = ggml_backend_vk_get_device_count();
    for (int i = 0; i < count; ++i) {
        DeviceInfo device;
        device.backend = Backend::vulkan;
        device.index = static_cast<std::uint32_t>(i);
        char name[256]{};
        ggml_backend_vk_get_device_description(i, name, sizeof(name));
        device.name = name;
        ggml_backend_vk_get_device_memory(i, &device.free_bytes, &device.total_bytes);
        devices.push_back(std::move(device));
    }
#endif
    devices.push_back({Backend::cpu, 0, "CPU", 0, 0});
    return devices;
}

namespace detail {
GgmlRuntime::GgmlRuntime(const RuntimeOptions& options) {
    if (options.cpu_threads < 0)
        throw std::invalid_argument("Inference CPU thread count must be nonnegative");
    if (options.backend != Backend::vulkan && options.backend != Backend::cpu)
        throw std::invalid_argument("Unknown inference backend");
    if (options.backend == Backend::cpu && options.device_index != 0)
        throw std::invalid_argument("CPU inference device index must be zero");
    if (options.backend == Backend::vulkan) {
#if defined(PHOTARA_INFERENCE_HAS_VULKAN)
        const auto count = ggml_backend_vk_get_device_count();
        if (options.device_index < static_cast<std::uint32_t>(count)) {
            backend_ = ggml_backend_vk_init_for_image_model(options.device_index);
            if (backend_) {
                device_.backend = Backend::vulkan;
                device_.index = options.device_index;
                char name[256]{};
                ggml_backend_vk_get_device_description(
                    static_cast<int>(options.device_index), name, sizeof(name));
                device_.name = name;
                ggml_backend_vk_get_device_memory(static_cast<int>(options.device_index),
                    &device_.free_bytes, &device_.total_bytes);
            }
        }
#endif
        if (!backend_ && !options.allow_cpu_fallback)
            throw std::runtime_error("Requested Vulkan inference device is unavailable");
    }
    if (!backend_) {
        backend_ = ggml_backend_cpu_init();
        if (!backend_) throw std::runtime_error("Could not initialize CPU inference");
        device_ = {Backend::cpu, 0, "CPU", 0, 0};
        const int threads = options.cpu_threads == 0
            ? static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))
            : options.cpu_threads;
        ggml_backend_cpu_set_n_threads(backend_, threads);
    }
    allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!allocator_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
        throw std::runtime_error("Could not initialize inference graph allocator");
    }
}

GgmlRuntime::~GgmlRuntime() {
    if (allocator_) ggml_gallocr_free(allocator_);
    if (backend_) ggml_backend_free(backend_);
}

void GgmlRuntime::allocate(ggml_cgraph* graph) {
    if (!graph) throw std::invalid_argument("Inference graph is null");
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto* node = ggml_graph_node(graph, i);
        if (!ggml_backend_supports_op(backend_, node))
            throw std::runtime_error(std::string("Inference backend does not support ") +
                ggml_op_name(node->op) + " " + ggml_get_name(node) +
                " shape=" + std::to_string(node->ne[0]) + "," + std::to_string(node->ne[1]) + "," + std::to_string(node->ne[2]) + "," + std::to_string(node->ne[3]));
    }
    if (!ggml_gallocr_alloc_graph(allocator_, graph))
        throw std::runtime_error("Could not allocate inference graph buffers");
}

void GgmlRuntime::compute(ggml_cgraph* graph) {
    if (!graph) throw std::invalid_argument("Inference graph is null");
    if (ggml_backend_graph_compute(backend_, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("Inference graph execution failed");
}

std::size_t GgmlRuntime::compute_buffer_bytes() const {
    return ggml_gallocr_get_buffer_size(allocator_, 0);
}
} // namespace detail
} // namespace photara::inference
