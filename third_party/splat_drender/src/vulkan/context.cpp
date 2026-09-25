#include "context_internal.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "splat_blend.hlsl.embedded.hpp"
#include "splat_emit.hlsl.embedded.hpp"
#include "splat_pack_rgba.hlsl.embedded.hpp"
#include "splat_preprocess.hlsl.embedded.hpp"
#include "splat_radix_hist.hlsl.embedded.hpp"
#include "splat_radix_scatter.hlsl.embedded.hpp"
#include "splat_ranges.hlsl.embedded.hpp"
#include "splat_scan.hlsl.embedded.hpp"

namespace splat_drender::vulkan {
namespace {

void check_vk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " + std::to_string(result));
    }
}

std::uint32_t find_memory_type(
    VkPhysicalDevice physical_device,
    std::uint32_t type_bits,
    VkMemoryPropertyFlags required_flags) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((type_bits & (1U << i)) != 0 &&
            (properties.memoryTypes[i].propertyFlags & required_flags) == required_flags) {
            return i;
        }
    }
    throw std::runtime_error("No compatible Vulkan memory type was found");
}

std::optional<std::uint32_t> try_find_memory_type(
    VkPhysicalDevice physical_device,
    std::uint32_t type_bits,
    VkMemoryPropertyFlags required_flags) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((type_bits & (1U << i)) != 0 &&
            (properties.memoryTypes[i].propertyFlags & required_flags) == required_flags) {
            return i;
        }
    }
    return std::nullopt;
}

// VRAM that the host cannot map. Host memory carries DEVICE_LOCAL on a
// resizable-BAR system, and the GPU reaches that over PCIe.
std::optional<std::uint32_t> try_find_pure_device_local(
    VkPhysicalDevice physical_device,
    const std::uint32_t type_bits) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((type_bits & (1U << i)) != 0 &&
            (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 &&
            (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
            return i;
        }
    }
    return std::nullopt;
}

std::vector<std::byte> read_spir_v(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Unable to open shader: " + path.string());
    }
    const auto byte_size = input.tellg();
    if (byte_size <= 0 || (byte_size % 4) != 0) {
        throw std::runtime_error("Invalid SPIR-V file: " + path.string());
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(byte_size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), byte_size);
    return bytes;
}

// SPLAT_DRENDER_SHADER_DIR replaces the embedded SPIR-V, so a pass can be
// recompiled and swapped in without rebuilding the library.
std::optional<std::filesystem::path> shader_directory_override() {
#ifdef _WIN32
    char* raw_value = nullptr;
    std::size_t value_size = 0;
    if (_dupenv_s(&raw_value, &value_size, "SPLAT_DRENDER_SHADER_DIR") != 0 || raw_value == nullptr) {
        return std::nullopt;
    }
    std::filesystem::path value(raw_value);
    std::free(raw_value);
    return value;
#else
    const char* raw_value = std::getenv("SPLAT_DRENDER_SHADER_DIR");
    return raw_value == nullptr ? std::nullopt : std::optional<std::filesystem::path>(raw_value);
#endif
}

std::span<const std::byte> embedded_shader(const std::string& shader_name) {
    if (shader_name == "splat_preprocess.hlsl.spv") {
        return std::as_bytes(std::span{splat_preprocess_hlsl_spv});
    }
    if (shader_name == "splat_scan.hlsl.spv") {
        return std::as_bytes(std::span{splat_scan_hlsl_spv});
    }
    if (shader_name == "splat_emit.hlsl.spv") {
        return std::as_bytes(std::span{splat_emit_hlsl_spv});
    }
    if (shader_name == "splat_radix_hist.hlsl.spv") {
        return std::as_bytes(std::span{splat_radix_hist_hlsl_spv});
    }
    if (shader_name == "splat_radix_scatter.hlsl.spv") {
        return std::as_bytes(std::span{splat_radix_scatter_hlsl_spv});
    }
    if (shader_name == "splat_ranges.hlsl.spv") {
        return std::as_bytes(std::span{splat_ranges_hlsl_spv});
    }
    if (shader_name == "splat_blend.hlsl.spv") {
        return std::as_bytes(std::span{splat_blend_hlsl_spv});
    }
    if (shader_name == "splat_pack_rgba.hlsl.spv") {
        return std::as_bytes(std::span{splat_pack_rgba_hlsl_spv});
    }
    throw std::invalid_argument("Unknown embedded shader: " + shader_name);
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT && callback_data != nullptr) {
        // A callback must not throw. The application can attach a debugger here.
    }
    return VK_FALSE;
}

DeviceInfo make_device_info(VkPhysicalDevice physical_device) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);
    return {
        properties.deviceName,
        properties.vendorID,
        properties.deviceID,
        properties.apiVersion,
    };
}

std::uint32_t find_compute_queue_family(VkPhysicalDevice physical_device) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> properties(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, properties.data());
    for (std::uint32_t i = 0; i < count; ++i) {
        if ((properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            return i;
        }
    }
    throw std::runtime_error("The selected Vulkan device has no compute queue");
}

// The adopted queue family must be the one `queue` came from: vkGetDeviceQueue
// is undefined for a family the device was not created with, so the handle
// alone cannot tell us the index and the caller has to state it.
bool queue_family_has_compute(VkPhysicalDevice physical_device, const std::uint32_t family) {
    if (physical_device == VK_NULL_HANDLE || family == VK_QUEUE_FAMILY_IGNORED) return false;
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
    if (family >= count) return false;
    std::vector<VkQueueFamilyProperties> properties(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, properties.data());
    return (properties[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
}

bool has_instance_extension(const char* extension_name) {
    std::uint32_t count = 0;
    check_vk(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
             "vkEnumerateInstanceExtensionProperties");
    std::vector<VkExtensionProperties> properties(count);
    check_vk(vkEnumerateInstanceExtensionProperties(nullptr, &count, properties.data()),
             "vkEnumerateInstanceExtensionProperties");
    return std::ranges::any_of(properties, [extension_name](const VkExtensionProperties& property) {
        return std::strcmp(property.extensionName, extension_name) == 0;
    });
}

bool has_instance_layer(const char* layer_name) {
    std::uint32_t count = 0;
    check_vk(vkEnumerateInstanceLayerProperties(&count, nullptr), "vkEnumerateInstanceLayerProperties");
    std::vector<VkLayerProperties> properties(count);
    check_vk(vkEnumerateInstanceLayerProperties(&count, properties.data()), "vkEnumerateInstanceLayerProperties");
    return std::ranges::any_of(properties, [layer_name](const VkLayerProperties& property) {
        return std::strcmp(property.layerName, layer_name) == 0;
    });
}

bool has_device_extension(VkPhysicalDevice physical_device, const char* extension_name) {
    std::uint32_t count = 0;
    check_vk(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr),
             "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> properties(count);
    check_vk(vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, properties.data()),
             "vkEnumerateDeviceExtensionProperties");
    return std::ranges::any_of(properties, [extension_name](const VkExtensionProperties& property) {
        return std::strcmp(property.extensionName, extension_name) == 0;
    });
}

}  // namespace

Buffer::Buffer(
    VkPhysicalDevice physical_device,
    VkDevice logical_device,
    VkDeviceSize byte_size,
    VkBufferUsageFlags usage,
    const BufferMemory memory_kind)
    : device(logical_device), size(std::max<VkDeviceSize>(byte_size, 4)) {
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check_vk(vkCreateBuffer(device, &buffer_info, nullptr, &handle), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, handle, &requirements);
    VkMemoryAllocateInfo allocation_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation_info.allocationSize = requirements.size;
    constexpr auto host_visible_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    switch (memory_kind) {
    case BufferMemory::host_visible_device_local:
        allocation_info.memoryTypeIndex = try_find_memory_type(
                                              physical_device, requirements.memoryTypeBits,
                                              host_visible_flags | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                                              .value_or(find_memory_type(
                                                  physical_device, requirements.memoryTypeBits, host_visible_flags));
        break;
    case BufferMemory::device_local:
        // Prefer VRAM without the host bit: a resizable-BAR heap is host
        // visible but the GPU still crosses PCIe to reach it.
        allocation_info.memoryTypeIndex = try_find_pure_device_local(
                                              physical_device, requirements.memoryTypeBits)
                                              .value_or(find_memory_type(
                                                  physical_device, requirements.memoryTypeBits,
                                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
        break;
    case BufferMemory::host_visible:
    default:
        // Cached host memory when the device has it: the CPU readback of a
        // render target is memory-bound, and write-combined memory is ~20x
        // slower to read than cached memory.
        allocation_info.memoryTypeIndex =
            try_find_memory_type(
                physical_device, requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                .value_or(find_memory_type(physical_device, requirements.memoryTypeBits, host_visible_flags));
        break;
    }
    check_vk(vkAllocateMemory(device, &allocation_info, nullptr, &memory), "vkAllocateMemory");
    check_vk(vkBindBufferMemory(device, handle, memory, 0), "vkBindBufferMemory");
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
    const auto& type = properties.memoryTypes[allocation_info.memoryTypeIndex];
    coherent = (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(physical_device, &device_properties);
    non_coherent_atom_size = device_properties.limits.nonCoherentAtomSize;
    if ((type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        check_vk(vkMapMemory(device, memory, 0, size, 0, &mapped), "vkMapMemory");
    }
}

Buffer::~Buffer() {
    if (device != VK_NULL_HANDLE && memory != VK_NULL_HANDLE) {
        if (mapped != nullptr) {
            vkUnmapMemory(device, memory);
        }
        vkDestroyBuffer(device, handle, nullptr);
        vkFreeMemory(device, memory, nullptr);
    }
}

Buffer::Buffer(Buffer&& other) noexcept {
    *this = std::move(other);
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        std::swap(device, other.device);
        std::swap(handle, other.handle);
        std::swap(memory, other.memory);
        std::swap(size, other.size);
        std::swap(mapped, other.mapped);
        std::swap(coherent, other.coherent);
        std::swap(non_coherent_atom_size, other.non_coherent_atom_size);
    }
    return *this;
}

// Flush/invalidate the atom-aligned range that covers the transfer: mapped
// memory that is not coherent only guarantees visibility at those boundaries.
void flush_range(VkDevice device, VkDeviceMemory memory, VkDeviceSize atom, const std::size_t offset,
                 const std::size_t bytes, const std::size_t size, const bool invalidate) {
    if (bytes == 0) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = memory;
    range.offset = (static_cast<VkDeviceSize>(offset) / atom) * atom;
    const VkDeviceSize end = static_cast<VkDeviceSize>(offset) + bytes;
    range.size = std::min<VkDeviceSize>(((end + atom - 1) / atom) * atom, size) - range.offset;
    if (invalidate) {
        check_vk(vkInvalidateMappedMemoryRanges(device, 1, &range), "vkInvalidateMappedMemoryRanges");
    } else {
        check_vk(vkFlushMappedMemoryRanges(device, 1, &range), "vkFlushMappedMemoryRanges");
    }
}

void Buffer::upload(const void* source, std::size_t byte_size, std::size_t offset) const {
    if (mapped == nullptr) {
        throw std::logic_error("Buffer is device-local; upload through the Context staging path");
    }
    if (offset + byte_size > size) {
        throw std::out_of_range("Buffer upload exceeds allocation");
    }
    if (byte_size != 0) {
        std::memcpy(static_cast<std::byte*>(mapped) + offset, source, byte_size);
        if (!coherent) {
            flush_range(device, memory, non_coherent_atom_size, offset, byte_size, size, false);
        }
    }
}

void Buffer::download(void* destination, std::size_t byte_size, std::size_t offset) const {
    if (mapped == nullptr) {
        throw std::logic_error("Buffer is device-local; download through the Context staging path");
    }
    if (offset + byte_size > size) {
        throw std::out_of_range("Buffer download exceeds allocation");
    }
    if (byte_size != 0) {
        if (!coherent) {
            flush_range(device, memory, non_coherent_atom_size, offset, byte_size, size, true);
        }
        std::memcpy(destination, static_cast<const std::byte*>(mapped) + offset, byte_size);
    }
}

ComputePipeline::ComputePipeline(
    VkDevice logical_device,
    std::span<const std::byte> spir_v_bytes,
    std::uint32_t storage_buffer_count,
    std::uint32_t push_constant_size)
    : device(logical_device), binding_count(storage_buffer_count) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(storage_buffer_count);
    for (std::uint32_t i = 0; i < storage_buffer_count; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo descriptor_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor_layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    descriptor_layout_info.pBindings = bindings.data();
    check_vk(
        vkCreateDescriptorSetLayout(device, &descriptor_layout_info, nullptr, &descriptor_set_layout),
        "vkCreateDescriptorSetLayout");

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = push_constant_size;
    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = push_constant_size == 0 ? 0U : 1U;
    pipeline_layout_info.pPushConstantRanges = push_constant_size == 0 ? nullptr : &push_range;
    check_vk(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout), "vkCreatePipelineLayout");

    if (spir_v_bytes.empty() || spir_v_bytes.size() % sizeof(std::uint32_t) != 0) {
        throw std::runtime_error("Invalid embedded SPIR-V bytecode");
    }
    std::vector<std::uint32_t> spir_v(spir_v_bytes.size() / sizeof(std::uint32_t));
    std::memcpy(spir_v.data(), spir_v_bytes.data(), spir_v_bytes.size());
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = spir_v.size() * sizeof(std::uint32_t);
    shader_info.pCode = spir_v.data();
    VkShaderModule shader_module = VK_NULL_HANDLE;
    check_vk(vkCreateShaderModule(device, &shader_info, nullptr, &shader_module), "vkCreateShaderModule");

    VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_module;
    stage_info.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = stage_info;
    pipeline_info.layout = pipeline_layout;
    const VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &handle);
    vkDestroyShaderModule(device, shader_module, nullptr);
    check_vk(result, "vkCreateComputePipelines");
}

ComputePipeline::~ComputePipeline() {
    if (device != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, handle, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    }
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept {
    *this = std::move(other);
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
    if (this != &other) {
        std::swap(device, other.device);
        std::swap(descriptor_set_layout, other.descriptor_set_layout);
        std::swap(pipeline_layout, other.pipeline_layout);
        std::swap(handle, other.handle);
        std::swap(binding_count, other.binding_count);
    }
    return *this;
}

Context::Impl::Impl(const ContextOptions& options) {
    if (options.external_device.valid()) {
        adopt_external_device(options.external_device);
    } else {
        create_instance_and_device(options);
    }
    create_pools();
}

// Runs on a device the caller created and keeps alive. Only the command pool
// and the descriptor pool belong to us, and every pipeline, buffer and submit
// goes to the caller's device and queue.
void Context::Impl::adopt_external_device(const ExternalDevice& external) {
    if (external.queue == VK_NULL_HANDLE || external.physical_device == VK_NULL_HANDLE) {
        throw std::invalid_argument(
            "Adopting a Vulkan device requires external_device.physical_device and external_device.queue");
    }
    if (!queue_family_has_compute(external.physical_device, external.queue_family)) {
        throw std::invalid_argument(
            "ContextOptions::external_device.queue_family " + std::to_string(external.queue_family) +
            " is out of range or has no compute queue");
    }
    instance = external.instance;
    physical_device = external.physical_device;
    device = external.device;
    queue = external.queue;
    queue_family_index = external.queue_family;
    device_info = make_device_info(physical_device);
}

void Context::Impl::create_instance_and_device(const ContextOptions& options) {
    const bool validation_enabled = options.enable_validation || SPLAT_DRENDER_ENABLE_VALIDATION;
    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    if (validation_enabled) {
        if (!has_instance_layer("VK_LAYER_KHRONOS_validation")) {
            throw std::runtime_error("Vulkan validation was requested but VK_LAYER_KHRONOS_validation is unavailable");
        }
        layers.push_back("VK_LAYER_KHRONOS_validation");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    const bool portability_enumeration_available =
        has_instance_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    if (portability_enumeration_available) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
#endif

    VkApplicationInfo application_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application_info.pApplicationName = "splat_drender";
    application_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application_info.pEngineName = "splat_drender";
    application_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    application_info.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
#ifdef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
    if (portability_enumeration_available) {
        instance_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif
    instance_info.pApplicationInfo = &application_info;
    instance_info.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    instance_info.ppEnabledLayerNames = layers.data();
    instance_info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    instance_info.ppEnabledExtensionNames = extensions.data();
    check_vk(vkCreateInstance(&instance_info, nullptr, &instance), "vkCreateInstance");

    if (validation_enabled) {
        const auto create_debug_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if (create_debug_messenger != nullptr) {
            VkDebugUtilsMessengerCreateInfoEXT debug_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            debug_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debug_info.pfnUserCallback = debug_callback;
            check_vk(create_debug_messenger(instance, &debug_info, nullptr, &debug_messenger), "debug messenger creation");
        }
    }

    std::uint32_t device_count = 0;
    check_vk(vkEnumeratePhysicalDevices(instance, &device_count, nullptr), "vkEnumeratePhysicalDevices");
    if (device_count == 0) {
        throw std::runtime_error("No Vulkan devices were found");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    check_vk(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()), "vkEnumeratePhysicalDevices");
    if (options.device_index >= device_count) {
        throw std::out_of_range("Vulkan device_index is out of range");
    }
    physical_device = devices[options.device_index];
    device_info = make_device_info(physical_device);
    queue_family_index = find_compute_queue_family(physical_device);

    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_create_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_info;
    std::vector<const char*> device_extensions;
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
    if (has_device_extension(physical_device, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
#endif
    device_create_info.enabledExtensionCount = static_cast<std::uint32_t>(device_extensions.size());
    device_create_info.ppEnabledExtensionNames = device_extensions.data();
    check_vk(vkCreateDevice(physical_device, &device_create_info, nullptr, &device), "vkCreateDevice");
    vkGetDeviceQueue(device, queue_family_index, 0, &queue);
    owns_instance = true;
    owns_device = true;
}

void Context::Impl::create_pools() {
    VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = queue_family_index;
    check_vk(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool");

    const std::vector<VkDescriptorPoolSize> pool_sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
    };
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 512;
    pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    check_vk(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool), "vkCreateDescriptorPool");
}

Context::Impl::~Impl() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        // Members outlive this body: release the staging buffer while the
        // device it was allocated from still exists.
        staging_ = Buffer{};
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyCommandPool(device, command_pool, nullptr);
        if (owns_device) {
            vkDestroyDevice(device, nullptr);
        }
    }
    if (debug_messenger != VK_NULL_HANDLE) {
        const auto destroy_debug_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy_debug_messenger != nullptr) {
            destroy_debug_messenger(instance, debug_messenger, nullptr);
        }
    }
    if (owns_instance && instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }
}

Buffer Context::Impl::create_buffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    const BufferMemory memory) const {
    return Buffer(physical_device, device, size, usage, memory);
}

void Context::Impl::ensure_staging(const VkDeviceSize bytes) {
    if (staging_.size >= bytes) return;
    staging_ = create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             BufferMemory::host_visible);
}

// One-shot copy on the compute queue. Callers hold dispatch_mutex, so the
// queue is ours for the length of the submit.
void Context::Impl::copy_between(
    const Buffer& source, const std::size_t source_offset, const Buffer& destination,
    const std::size_t destination_offset, const std::size_t byte_size) {
    if (byte_size == 0) return;
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = command_pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    check_vk(vkAllocateCommandBuffers(device, &allocate, &command), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
    VkBufferCopy region{};
    region.srcOffset = source_offset;
    region.dstOffset = destination_offset;
    region.size = byte_size;
    vkCmdCopyBuffer(command, source.handle, destination.handle, 1, &region);
    check_vk(vkEndCommandBuffer(command), "vkEndCommandBuffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check_vk(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
    check_vk(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
    vkFreeCommandBuffers(device, command_pool, 1, &command);
}

void Context::Impl::write_buffer(
    const Buffer& destination, const void* source, const std::size_t byte_size,
    const std::size_t offset) {
    if (destination.host_visible()) {
        destination.upload(source, byte_size, offset);
        return;
    }
    ensure_staging(byte_size);
    staging_.upload(source, byte_size);
    copy_between(staging_, 0, destination, offset, byte_size);
}

void Context::Impl::read_buffer(
    const Buffer& source, void* destination, const std::size_t byte_size,
    const std::size_t offset) {
    if (source.host_visible()) {
        source.download(destination, byte_size, offset);
        return;
    }
    ensure_staging(byte_size);
    copy_between(source, offset, staging_, 0, byte_size);
    staging_.download(destination, byte_size);
}

void Context::Impl::fill_buffer(const Buffer& destination, const std::uint32_t value, const std::size_t byte_size) {
    if (byte_size == 0) return;
    if (destination.host_visible()) {
        std::memset(destination.mapped, value & 0xFFU, byte_size);
        return;
    }
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = command_pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    check_vk(vkAllocateCommandBuffers(device, &allocate, &command), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_vk(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
    vkCmdFillBuffer(command, destination.handle, 0, static_cast<VkDeviceSize>(byte_size), value);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
        command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
        0, nullptr, 0, nullptr);
    check_vk(vkEndCommandBuffer(command), "vkEndCommandBuffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check_vk(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
    check_vk(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
    vkFreeCommandBuffers(device, command_pool, 1, &command);
}

ComputePipeline Context::Impl::create_pipeline(
    const std::string& shader_name,
    std::uint32_t storage_buffer_count,
    std::uint32_t push_constant_size) const {
    if (const auto override_directory = shader_directory_override()) {
        const auto bytes = read_spir_v(*override_directory / shader_name);
        return ComputePipeline(device, bytes, storage_buffer_count, push_constant_size);
    }
    return ComputePipeline(device, embedded_shader(shader_name), storage_buffer_count, push_constant_size);
}

Context::Context(const ContextOptions& options) : impl_(std::make_unique<Impl>(options)) {}
Context::~Context() = default;
Context::Context(Context&&) noexcept = default;
Context& Context::operator=(Context&&) noexcept = default;

const DeviceInfo& Context::device_info() const noexcept {
    return impl_->device_info;
}

VkInstance Context::instance() const noexcept {
    return impl_->instance;
}

VkPhysicalDevice Context::physical_device() const noexcept {
    return impl_->physical_device;
}

VkDevice Context::device() const noexcept {
    return impl_->device;
}

VkQueue Context::queue() const noexcept {
    return impl_->queue;
}

std::uint32_t Context::queue_family() const noexcept {
    return impl_->queue_family_index;
}

std::vector<DeviceInfo> Context::enumerate_devices() {
    VkApplicationInfo application_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application_info.pApplicationName = "splat_drender_device_enumeration";
    application_info.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo create_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create_info.pApplicationInfo = &application_info;
    std::vector<const char*> extensions;
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    if (has_instance_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#ifdef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
        create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    }
#endif
    create_info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    VkInstance temporary_instance = VK_NULL_HANDLE;
    check_vk(vkCreateInstance(&create_info, nullptr, &temporary_instance), "vkCreateInstance");
    std::uint32_t count = 0;
    check_vk(vkEnumeratePhysicalDevices(temporary_instance, &count, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> physical_devices(count);
    check_vk(vkEnumeratePhysicalDevices(temporary_instance, &count, physical_devices.data()), "vkEnumeratePhysicalDevices");
    std::vector<DeviceInfo> result;
    result.reserve(count);
    for (const auto physical_device : physical_devices) {
        result.push_back(make_device_info(physical_device));
    }
    vkDestroyInstance(temporary_instance, nullptr);
    return result;
}

}  // namespace splat_drender::vulkan
