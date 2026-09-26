#include "vulkan/runtime/runtime.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#include <stdlib.h>
#endif

#include "adam_f32.hlsl.embedded.hpp"
#include "cat.hlsl.embedded.hpp"
#include "compact.hlsl.embedded.hpp"
#include "cumsum.hlsl.embedded.hpp"
#include "elementwise.hlsl.embedded.hpp"
#include "fused_pointwise.hlsl.embedded.hpp"
#include "index_fill.hlsl.embedded.hpp"
#include "index_select.hlsl.embedded.hpp"
#include "mask_flags.hlsl.embedded.hpp"
#include "matmul.hlsl.embedded.hpp"
#include "multinomial.hlsl.embedded.hpp"
#include "pool.hlsl.embedded.hpp"
#include "random.hlsl.embedded.hpp"
#include "reduce.hlsl.embedded.hpp"
#include "reduce_all_f32.hlsl.embedded.hpp"
#include "scan_add.hlsl.embedded.hpp"
#include "scan_block.hlsl.embedded.hpp"
#include "scatter.hlsl.embedded.hpp"
#include "select_compact.hlsl.embedded.hpp"
#include "strided_copy.hlsl.embedded.hpp"
#include "unpack_rgba.hlsl.embedded.hpp"

namespace tinytensor::vulkan::runtime {
namespace {

constexpr std::size_t kMinBufferBytes = 16;

// A batch submits once it has this many recorded commands, so one submission
// stays small and a caller that never reads back still makes progress. It sits
// far above the op count of a real step.
constexpr std::uint32_t kMaxCommandsPerBatch = 4096;

// Recycled buffers are kept up to this budget, because a workload with moving
// tensor shapes must not grow VRAM without bound.
constexpr std::size_t kDefaultPoolBudgetBytes = 1ULL << 30;

// Matches the rounding Buffer's constructor applies, so the pool is keyed by
// the size an allocation actually has.
VkDeviceSize pooled_size_of(VkDeviceSize bytes) {
    return (std::max<VkDeviceSize>(bytes, kMinBufferBytes) + 3U) & ~VkDeviceSize{3};
}

std::optional<std::uint32_t> env_u32(const char* name) {
#ifdef _WIN32
    char* raw = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&raw, &size, name) != 0 || raw == nullptr) {
        return std::nullopt;
    }
    const std::uint32_t value = static_cast<std::uint32_t>(std::strtoul(raw, nullptr, 10));
    std::free(raw);
    return value;
#else
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(std::strtoul(raw, nullptr, 10));
#endif
}

bool env_flag(const char* name) {
    const auto value = env_u32(name);
    return value.has_value() && *value != 0;
}

std::uint32_t env_u32_or(const char* name, const std::uint32_t fallback) {
    const auto value = env_u32(name);
    return value.has_value() ? *value : fallback;
}

const char* shader_name(const ShaderId shader) {
    static constexpr std::array<const char*, static_cast<std::size_t>(ShaderId::Count)> names{
        "adam_f32",    "elementwise", "fused_pointwise", "strided_copy", "index_select",
        "index_fill",  "mask_flags",  "scan_block",      "scan_add",     "compact",
        "multinomial", "reduce",      "reduce_all_f32",  "matmul",       "random",
        "cumsum",      "pool",        "scatter",         "select_compact",
        "cat",         "unpack_rgba"};
    const std::size_t index = static_cast<std::size_t>(shader);
    return index < names.size() ? names[index] : "unknown";
}

// Timestamps are a fixed-width counter; a queue may expose fewer than 64 valid
// bits, and the delta has to wrap at that width.
std::uint64_t timestamp_delta(
    const std::uint64_t begin, const std::uint64_t end,
    const std::uint32_t valid_bits) {
    if (valid_bits >= 64) return end - begin;
    const std::uint64_t mask = (std::uint64_t{1} << valid_bits) - 1;
    const std::uint64_t masked_begin = begin & mask;
    const std::uint64_t masked_end = end & mask;
    return masked_end >= masked_begin
        ? masked_end - masked_begin
        : ((mask - masked_begin + 1) + masked_end) & mask;
}

bool has_instance_extension(const char* name) {
    std::uint32_t count = 0;
    check(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
          "vkEnumerateInstanceExtensionProperties");
    std::vector<VkExtensionProperties> properties(count);
    check(vkEnumerateInstanceExtensionProperties(nullptr, &count, properties.data()),
          "vkEnumerateInstanceExtensionProperties");
    return std::ranges::any_of(properties, [name](const VkExtensionProperties& property) {
        return std::strcmp(property.extensionName, name) == 0;
    });
}

bool has_instance_layer(const char* name) {
    std::uint32_t count = 0;
    check(vkEnumerateInstanceLayerProperties(&count, nullptr), "vkEnumerateInstanceLayerProperties");
    std::vector<VkLayerProperties> properties(count);
    check(vkEnumerateInstanceLayerProperties(&count, properties.data()),
          "vkEnumerateInstanceLayerProperties");
    return std::ranges::any_of(properties, [name](const VkLayerProperties& property) {
        return std::strcmp(property.layerName, name) == 0;
    });
}

bool has_device_extension(VkPhysicalDevice physical, const char* name) {
    std::uint32_t count = 0;
    check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr),
          "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> properties(count);
    check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, properties.data()),
          "vkEnumerateDeviceExtensionProperties");
    return std::ranges::any_of(properties, [name](const VkExtensionProperties& property) {
        return std::strcmp(property.extensionName, name) == 0;
    });
}

std::uint32_t find_memory_type(VkPhysicalDevice physical, std::uint32_t type_bits,
                               VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((type_bits & (1U << i)) != 0 &&
            (properties.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    throw std::runtime_error("No compatible Vulkan memory type was found");
}

std::optional<std::uint32_t> try_find_memory_type(VkPhysicalDevice physical, std::uint32_t type_bits,
                                                  VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((type_bits & (1U << i)) != 0 &&
            (properties.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    return std::nullopt;
}

std::uint32_t find_compute_queue_family(VkPhysicalDevice physical) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    std::vector<VkQueueFamilyProperties> properties(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, properties.data());
    for (std::uint32_t i = 0; i < count; ++i) {
        if ((properties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            return i;
        }
    }
    throw std::runtime_error("The selected Vulkan device has no compute queue");
}

int device_score(VkPhysicalDevice physical) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical, &properties);
    int score = 0;
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 300;
    } else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        score += 100;
    } else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        return -1;
    }
    try {
        (void)find_compute_queue_family(physical);
    } catch (...) {
        return -1;
    }
    return score;
}

struct ShaderBlob {
    ShaderId id;
    std::span<const std::byte> spirv;
    std::uint32_t bindings;
    std::uint32_t push_bytes;
};

std::array<ShaderBlob, static_cast<std::size_t>(ShaderId::Count)> shader_blobs() {
    using std::as_bytes;
    using std::span;
    return {{
        {ShaderId::AdamF32, as_bytes(span{adam_f32_hlsl_spv}), 4, 64},
        {ShaderId::Elementwise, as_bytes(span{elementwise_hlsl_spv}), 4, 72},
        {ShaderId::FusedPointwise, as_bytes(span{fused_pointwise_hlsl_spv}), 3, 16},
        {ShaderId::StridedCopy, as_bytes(span{strided_copy_hlsl_spv}), 2, 96},
        {ShaderId::IndexSelect, as_bytes(span{index_select_hlsl_spv}), 3, 40},
        {ShaderId::IndexFill, as_bytes(span{index_fill_hlsl_spv}), 2, 40},
        {ShaderId::MaskFlags, as_bytes(span{mask_flags_hlsl_spv}), 2, 16},
        {ShaderId::ScanBlock, as_bytes(span{scan_block_hlsl_spv}), 3, 24},
        {ShaderId::ScanAdd, as_bytes(span{scan_add_hlsl_spv}), 2, 16},
        {ShaderId::Compact, as_bytes(span{compact_hlsl_spv}), 3, 52},
        {ShaderId::Multinomial, as_bytes(span{multinomial_hlsl_spv}), 3, 32},
        {ShaderId::Reduce, as_bytes(span{reduce_hlsl_spv}), 3, 40},
        {ShaderId::ReduceAllF32, as_bytes(span{reduce_all_f32_hlsl_spv}), 2, 24},
        {ShaderId::Matmul, as_bytes(span{matmul_hlsl_spv}), 3, 56},
        {ShaderId::Random, as_bytes(span{random_hlsl_spv}), 2, 44},
        {ShaderId::Cumsum, as_bytes(span{cumsum_hlsl_spv}), 2, 28},
        {ShaderId::Pool, as_bytes(span{pool_hlsl_spv}), 2, 48},
        {ShaderId::Scatter, as_bytes(span{scatter_hlsl_spv}), 3, 44},
        {ShaderId::SelectCompact, as_bytes(span{select_compact_hlsl_spv}), 4, 28},
        {ShaderId::Cat, as_bytes(span{cat_hlsl_spv}), 3, 24},
        {ShaderId::UnpackRgba, as_bytes(span{unpack_rgba_hlsl_spv}), 4, 24},
    }};
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT*,
    void*) {
    return VK_FALSE;
}

std::unique_ptr<Context>& singleton() {
    static std::unique_ptr<Context> instance;
    return instance;
}

std::mutex& singleton_mutex() {
    static std::mutex mutex;
    return mutex;
}

} // namespace

Buffer::Buffer(VkPhysicalDevice physical, VkDevice logical, VkDeviceSize bytes,
               const MemoryKind kind)
    : device_(logical),
      size_((std::max<VkDeviceSize>(bytes, kMinBufferBytes) + 3u) & ~VkDeviceSize{3}) {
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size_;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(device_, &buffer_info, nullptr, &handle_), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, handle_, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    const VkMemoryPropertyFlags host_flags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    switch (kind) {
    case MemoryKind::host_cached:
        // A readback is bounded by how fast the CPU can read the staging
        // memory, and write-combined memory is an order of magnitude slower to
        // read than cached memory.
        allocation.memoryTypeIndex =
            try_find_memory_type(physical, requirements.memoryTypeBits,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                .value_or(try_find_memory_type(physical, requirements.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                   VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                              .value_or(find_memory_type(physical, requirements.memoryTypeBits,
                                                         host_flags)));
        break;
    case MemoryKind::host_visible:
        allocation.memoryTypeIndex =
            find_memory_type(physical, requirements.memoryTypeBits, host_flags);
        break;
    case MemoryKind::device_local:
    default:
        const auto device_local = try_find_memory_type(
            physical, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        allocation.memoryTypeIndex = device_local.value_or(
            find_memory_type(physical, requirements.memoryTypeBits, host_flags));
        break;
    }
    check(vkAllocateMemory(device_, &allocation, nullptr, &memory_), "vkAllocateMemory");
    check(vkBindBufferMemory(device_, handle_, memory_, 0), "vkBindBufferMemory");
    if (kind != MemoryKind::device_local) {
        check(vkMapMemory(device_, memory_, 0, size_, 0, &mapped_), "vkMapMemory");
        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(physical, &properties);
        coherent_ = (properties.memoryTypes[allocation.memoryTypeIndex].propertyFlags &
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        VkPhysicalDeviceProperties limits{};
        vkGetPhysicalDeviceProperties(physical, &limits);
        non_coherent_atom_size_ =
            std::max<VkDeviceSize>(limits.limits.nonCoherentAtomSize, 1);
    }
}

Buffer::~Buffer() {
    if (device_ == VK_NULL_HANDLE) {
        return;
    }
    if (mapped_ != nullptr) {
        vkUnmapMemory(device_, memory_);
    }
    if (handle_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, handle_, nullptr);
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, memory_, nullptr);
    }
}

Buffer::Buffer(Buffer&& other) noexcept {
    *this = std::move(other);
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        std::swap(device_, other.device_);
        std::swap(handle_, other.handle_);
        std::swap(memory_, other.memory_);
        std::swap(size_, other.size_);
        std::swap(mapped_, other.mapped_);
        std::swap(coherent_, other.coherent_);
        std::swap(non_coherent_atom_size_, other.non_coherent_atom_size_);
    }
    return *this;
}

BufferBinding Buffer::binding(VkDeviceSize offset, VkDeviceSize range) const {
    BufferBinding result;
    result.buffer = handle_;
    result.offset = offset;
    result.range = range;
    return result;
}

Context& Context::get() {
    std::scoped_lock lock(singleton_mutex());
    auto& instance = singleton();
    if (!instance) {
        instance.reset(new Context());
    }
    return *instance;
}

bool Context::available() {
    try {
        (void)get();
        return true;
    } catch (...) {
        std::scoped_lock lock(singleton_mutex());
        singleton().reset();
        return false;
    }
}

void Context::shutdown() {
    std::scoped_lock lock(singleton_mutex());
    singleton().reset();
}

Context::Context() {
    create_instance();
    pick_device();
    create_device();
    create_pools();
    // Recycled buffers are held up to this budget; override with
    // TINYTENSOR_VULKAN_POOL_BYTES for a workload with much larger tensors.
    const std::uint32_t pool_budget_override = env_u32("TINYTENSOR_VULKAN_POOL_BYTES").value_or(0);
    pool_budget_bytes_ =
        pool_budget_override != 0 ? static_cast<std::size_t>(pool_budget_override)
                                  : kDefaultPoolBudgetBytes;
    dummy_ = std::make_unique<Buffer>(physical_, device_, kMinBufferBytes, MemoryKind::device_local);
    create_pipelines();
    op_profile_.enabled = env_flag("TINYTENSOR_VULKAN_PROFILE_OPS");
    if (op_profile_.enabled) {
        op_profile_.interval =
            std::max(1U, env_u32_or("TINYTENSOR_VULKAN_PROFILE_OPS_INTERVAL", 200));
        create_op_profile_locked();
    }
}

Context::~Context() {
    destroy();
}

void Context::create_instance() {
    validation_ = env_flag("TINYTENSOR_VULKAN_VALIDATION");
    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    if (validation_) {
        if (!has_instance_layer("VK_LAYER_KHRONOS_validation")) {
            throw std::runtime_error("TINYTENSOR_VULKAN_VALIDATION is set but the validation layer is missing");
        }
        layers.push_back("VK_LAYER_KHRONOS_validation");
        if (has_instance_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    const bool portability = has_instance_extension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    if (portability) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
#endif

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "tinytensor";
    app.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    app.pEngineName = "tinytensor";
    app.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
#ifdef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
    if (portability) {
        create.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif
    create.pApplicationInfo = &app;
    create.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    create.ppEnabledLayerNames = layers.data();
    create.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    check(vkCreateInstance(&create, nullptr, &instance_), "vkCreateInstance");

    if (validation_ && has_instance_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        const auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (create_messenger != nullptr) {
            VkDebugUtilsMessengerCreateInfoEXT debug{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            debug.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debug.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debug.pfnUserCallback = debug_callback;
            check(create_messenger(instance_, &debug, nullptr, &debug_messenger_),
                  "vkCreateDebugUtilsMessengerEXT");
        }
    }
}

void Context::pick_device() {
    std::uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
    if (count == 0) {
        throw std::runtime_error("No Vulkan devices were found");
    }
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices");

    if (const auto index = env_u32("TINYTENSOR_VULKAN_DEVICE")) {
        if (*index >= count) {
            throw std::out_of_range("TINYTENSOR_VULKAN_DEVICE is out of range");
        }
        physical_ = devices[*index];
    } else {
        int best = -1;
        for (VkPhysicalDevice candidate : devices) {
            const int score = device_score(candidate);
            if (score > best) {
                best = score;
                physical_ = candidate;
            }
        }
        if (physical_ == VK_NULL_HANDLE) {
            throw std::runtime_error("No Vulkan compute device was found");
        }
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_, &properties);
    info_.name = properties.deviceName;
    info_.vendor_id = properties.vendorID;
    info_.device_id = properties.deviceID;
    info_.api_version = properties.apiVersion;
    if (properties.limits.maxComputeWorkGroupInvocations < 256 ||
        properties.limits.maxComputeWorkGroupSize[0] < 256 ||
        properties.limits.maxPushConstantsSize < 128) {
        throw std::runtime_error(
            "The selected Vulkan device cannot run TinyTensor compute shaders");
    }

    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(physical_, &properties2);
    info_.subgroup_size = subgroup.subgroupSize;

    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    vkGetPhysicalDeviceFeatures2(physical_, &features);
    info_.shader_float64 = features.features.shaderFloat64 == VK_TRUE;

#ifdef VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME
    if (has_device_extension(physical_, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 atomic_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        atomic_query.pNext = &atomic_float;
        vkGetPhysicalDeviceFeatures2(physical_, &atomic_query);
        info_.buffer_atomic_f32 = atomic_float.shaderBufferFloat32AtomicAdd == VK_TRUE;
        info_.buffer_atomic_f64 = atomic_float.shaderBufferFloat64AtomicAdd == VK_TRUE;
    }
#endif

    queue_family_ = find_compute_queue_family(physical_);
}

void Context::create_device() {
    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    std::vector<const char*> extensions;
#ifdef VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME
    // Push descriptors remove one descriptor-set allocation and one set update
    // from every tensor op.  This matters much more than shader time for the
    // small, numerous elementwise ops in an optimizer/backward pass.  Keep the
    // descriptor-pool path below as a portability fallback and as an A/B knob.
    push_descriptors_ =
        !env_flag("TINYTENSOR_VULKAN_DISABLE_PUSH_DESCRIPTORS") &&
        has_device_extension(physical_, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    if (push_descriptors_) {
        extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    }
#endif
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
    if (has_device_extension(physical_, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
#endif

    VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    create.queueCreateInfoCount = 1;
    create.pQueueCreateInfos = &queue_info;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_enable{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
#ifdef VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME
    // Buffer float32 atomic adds let adopted compute libraries (the splat
    // rasterizer's gradient commit) replace CAS retry loops with one hardware
    // atomic. info_.buffer_atomic_f32 records support; enabling it here is
    // what makes the feature legal on the shared device.
    if (info_.buffer_atomic_f32 &&
        has_device_extension(physical_, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
        extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        atomic_float_enable.shaderBufferFloat32AtomicAdd = VK_TRUE;
        create.pNext = &atomic_float_enable;
    }
#endif
    create.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    check(vkCreateDevice(physical_, &create, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
#ifdef VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME
    if (push_descriptors_) {
        cmd_push_descriptor_ = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
            vkGetDeviceProcAddr(device_, "vkCmdPushDescriptorSetKHR"));
        if (cmd_push_descriptor_ == nullptr) {
            throw std::runtime_error(
                "VK_KHR_push_descriptor was enabled but vkCmdPushDescriptorSetKHR is missing");
        }
    }
#endif
    info_.push_descriptors = push_descriptors_;
}

void Context::create_pools() {
    VkCommandPoolCreateInfo command_pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool.queueFamilyIndex = queue_family_;
    check(vkCreateCommandPool(device_, &command_pool, nullptr, &command_pool_), "vkCreateCommandPool");

    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    check(vkCreateFence(device_, &fence, nullptr, &completion_fence_), "vkCreateFence");

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool.maxSets = 1024;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(device_, &pool, nullptr, &descriptor_pool_), "vkCreateDescriptorPool");
}

void Context::create_pipelines() {
    for (const ShaderBlob& blob : shader_blobs()) {
        Pipeline& pipeline = pipelines_[static_cast<std::size_t>(blob.id)];
        pipeline.binding_count = blob.bindings;
        pipeline.push_size = blob.push_bytes;

        std::vector<VkDescriptorSetLayoutBinding> bindings(blob.bindings);
        for (std::uint32_t i = 0; i < blob.bindings; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        if (push_descriptors_) {
            layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        }
        layout_info.bindingCount = blob.bindings;
        layout_info.pBindings = bindings.data();
        check(vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &pipeline.set_layout),
              "vkCreateDescriptorSetLayout");

        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = blob.push_bytes;
        VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout.setLayoutCount = 1;
        pipeline_layout.pSetLayouts = &pipeline.set_layout;
        pipeline_layout.pushConstantRangeCount = blob.push_bytes == 0 ? 0U : 1U;
        pipeline_layout.pPushConstantRanges = blob.push_bytes == 0 ? nullptr : &push;
        check(vkCreatePipelineLayout(device_, &pipeline_layout, nullptr, &pipeline.layout),
              "vkCreatePipelineLayout");

        if (blob.spirv.empty() || blob.spirv.size() % sizeof(std::uint32_t) != 0) {
            throw std::runtime_error("Invalid embedded SPIR-V");
        }
        std::vector<std::uint32_t> words(blob.spirv.size() / sizeof(std::uint32_t));
        std::memcpy(words.data(), blob.spirv.data(), blob.spirv.size());
        VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader_info.codeSize = blob.spirv.size();
        shader_info.pCode = words.data();
        VkShaderModule module = VK_NULL_HANDLE;
        check(vkCreateShaderModule(device_, &shader_info, nullptr, &module), "vkCreateShaderModule");

        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = stage;
        pipeline_info.layout = pipeline.layout;
        const VkResult result =
            vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline.handle);
        vkDestroyShaderModule(device_, module, nullptr);
        check(result, "vkCreateComputePipelines");
    }
}

void Context::destroy() {
    if (device_ != VK_NULL_HANDLE) {
        // Retire the open batch before anything it references is destroyed.
        // Teardown runs from a destructor, so a device that is already lost
        // must not turn into a thrown exception here.
        if (recording_) {
            vkEndCommandBuffer(command_);
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command_;
            if (vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS) {
                vkQueueWaitIdle(queue_);
            }
        }
        recording_ = false;
        vkDeviceWaitIdle(device_);
    }
    if (op_profile_.enabled) {
        // Report the run total before the pool goes away: a training run ends
        // long before the next aggregation window would have fired.
        if (op_profile_.flushes != 0) report_op_profile_locked();
        if (op_profile_.pool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device_, op_profile_.pool, nullptr);
            op_profile_.pool = VK_NULL_HANDLE;
        }
        op_profile_.enabled = false;
    }
    alive_ = false;
    for (auto& entry : free_buffers_) {
        for (Buffer* buffer : entry.second) {
            delete buffer;
        }
    }
    free_buffers_.clear();
    pooled_bytes_ = 0;
    dummy_.reset();
    staging_.reset();
    readback_staging_.reset();
    for (Pipeline& pipeline : pipelines_) {
        if (pipeline.handle != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline.handle, nullptr);
        }
        if (pipeline.layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, pipeline.layout, nullptr);
        }
        if (pipeline.set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, pipeline.set_layout, nullptr);
        }
        pipeline = {};
    }
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    }
    if (command_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device_, command_pool_, 1, &command_);
        command_ = VK_NULL_HANDLE;
    }
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
    }
    if (completion_fence_ != VK_NULL_HANDLE) {
        vkDestroyFence(device_, completion_fence_, nullptr);
        completion_fence_ = VK_NULL_HANDLE;
    }
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
    if (debug_messenger_ != VK_NULL_HANDLE) {
        const auto destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy_messenger != nullptr) {
            destroy_messenger(instance_, debug_messenger_, nullptr);
        }
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
    device_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
}

std::shared_ptr<Buffer> Context::alloc(std::size_t bytes) {
    std::scoped_lock lock(mutex_);
    return acquire_buffer_locked(bytes);
}

std::shared_ptr<Buffer> Context::acquire_buffer_locked(const std::size_t bytes) {
    // A released Buffer comes back through this deleter instead of being
    // destroyed, which is what stops a repeated shape from paying
    // vkCreateBuffer/vkAllocateMemory/vkBindBufferMemory on every step.
    const auto release = [this](Buffer* buffer) {
        std::scoped_lock lock(mutex_);
        if (!alive_) {
            // The device is already gone; touching the handle would be worse
            // than leaking it at process exit.
            return;
        }
        recycle_locked(buffer);
    };
    const VkDeviceSize size = pooled_size_of(static_cast<VkDeviceSize>(bytes));
    auto& free_list = free_buffers_[static_cast<std::size_t>(size)];
    if (!free_list.empty()) {
        Buffer* buffer = free_list.back();
        free_list.pop_back();
        pooled_bytes_ -= static_cast<std::size_t>(buffer->size());
        return std::shared_ptr<Buffer>(buffer, release);
    }
    return std::shared_ptr<Buffer>(new Buffer(physical_, device_, size, MemoryKind::device_local),
                                   release);
}

void Context::recycle_locked(Buffer* buffer) {
    pooled_bytes_ += static_cast<std::size_t>(buffer->size());
    free_buffers_[static_cast<std::size_t>(buffer->size())].push_back(buffer);
    if (!recording_) {
        // A buffer can only be destroyed while no batch is recording: a
        // recorded command still holds the handle even though no tensor does.
        trim_pool_locked();
    }
}

void Context::trim_pool_locked() {
    if (recording_ || pooled_bytes_ <= pool_budget_bytes_) {
        return;
    }
    for (auto& entry : free_buffers_) {
        std::vector<Buffer*>& list = entry.second;
        while (pooled_bytes_ > pool_budget_bytes_ && !list.empty()) {
            Buffer* buffer = list.back();
            list.pop_back();
            pooled_bytes_ -= static_cast<std::size_t>(buffer->size());
            delete buffer;
        }
        if (pooled_bytes_ <= pool_budget_bytes_) {
            break;
        }
    }
}

Buffer& Context::dummy() {
    return *dummy_;
}

void Context::ensure_staging(std::size_t bytes) {
    if (staging_ && staging_->size() >= bytes) {
        return;
    }
    staging_ = std::make_unique<Buffer>(physical_, device_, static_cast<VkDeviceSize>(bytes),
                                        MemoryKind::host_visible);
    staging_cursor_ = 0;
}

// Cached host memory, because pulling a render target back to the host is
// bound by the CPU read of the staging buffer.
void Context::ensure_readback_staging(std::size_t bytes) {
    if (readback_staging_ && readback_staging_->size() >= bytes) {
        return;
    }
    readback_staging_ = std::make_unique<Buffer>(
        physical_, device_, static_cast<VkDeviceSize>(bytes), MemoryKind::host_cached);
}

// One command buffer is recorded into and submitted once per flush, instead of
// a fresh command buffer and a queue drain per op. The command buffer is kept
// alive across flushes; only its contents are reset.
void Context::begin_batch_locked() {
    if (recording_) {
        return;
    }
    if (command_ == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool = command_pool_;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device_, &alloc, &command_), "vkAllocateCommandBuffers");
    }
    check(vkResetCommandBuffer(command_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command_, &begin), "vkBeginCommandBuffer");
    recording_ = true;
    batch_commands_ = 0;
    staging_cursor_ = 0;
    if (op_profile_.enabled && !op_profile_.order.empty()) {
        // Retire the previous batch's timestamp range before reusing indices.
        vkCmdResetQueryPool(
            command_, op_profile_.pool, 0,
            static_cast<std::uint32_t>(op_profile_.order.size()) * 2);
        op_profile_.order.clear();
    }
}

void Context::flush_locked() {
    if (!recording_) {
        // Nothing is pending, so nothing references the staging buffer and the
        // cursor starts clean for the next batch.
        staging_cursor_ = 0;
        return;
    }
    check(vkEndCommandBuffer(command_), "vkEndCommandBuffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    check(vkQueueSubmit(queue_, 1, &submit, completion_fence_), "vkQueueSubmit");
    check(vkWaitForFences(device_, 1, &completion_fence_, VK_TRUE, UINT64_MAX),
          "vkWaitForFences");
    check(vkResetFences(device_, 1, &completion_fence_), "vkResetFences");
    // The queue is idle, so the batch's timestamp pairs are readable.
    if (op_profile_.enabled) collect_op_profile_locked();
    recording_ = false;
    batch_commands_ = 0;
    // Every descriptor set the batch allocated is dead once the queue is idle,
    // so the arena is recycled with one reset instead of a free per op.
    staging_cursor_ = 0;
    check(vkResetDescriptorPool(device_, descriptor_pool_, 0), "vkResetDescriptorPool");
    trim_pool_locked();
}

// One timestamp pair per recorded dispatch, accumulated per shader. The pool is
// created once and re-reset inside each batch, which is where the reset has to
// be recorded for the next batch's timestamps to land in order.
void Context::create_op_profile_locked() {
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, families.data());
    op_profile_.valid_bits = queue_family_ < families.size()
        ? families[queue_family_].timestampValidBits
        : 0;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_, &properties);
    op_profile_.period = properties.limits.timestampPeriod;
    if (op_profile_.valid_bits == 0) {
        std::fprintf(
            stderr,
            "tinytensor_vulkan_op_profile disabled: queue has timestampValidBits=0\n");
        op_profile_.enabled = false;
        return;
    }
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = 2 * kMaxCommandsPerBatch;
    check(vkCreateQueryPool(device_, &info, nullptr, &op_profile_.pool),
          "vkCreateQueryPool");
    const std::size_t slots = static_cast<std::size_t>(ShaderId::Count) *
        OpProfile::kElementBuckets;
    op_profile_.calls.assign(slots, 0);
    op_profile_.total_ms.assign(slots, 0.0);
    op_profile_.order.reserve(kMaxCommandsPerBatch);
}

void Context::collect_op_profile_locked() {
    const std::uint32_t pairs = op_profile_.recorded;
    op_profile_.recorded = 0;
    if (pairs == 0) return;
    std::vector<std::uint64_t> stamps(static_cast<std::size_t>(pairs) * 2);
    const VkResult result = vkGetQueryPoolResults(
        device_, op_profile_.pool, 0, pairs * 2, stamps.size() * sizeof(std::uint64_t),
        stamps.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
    if (result != VK_SUCCESS) {
        op_profile_.order.clear();
        return;
    }
    const double ticks_to_ms = static_cast<double>(op_profile_.period) / 1.0e6;
    for (std::uint32_t pair = 0; pair < pairs; ++pair) {
        const std::size_t slot = op_profile_.order[pair];
        const std::uint64_t ticks = timestamp_delta(
            stamps[2 * pair], stamps[2 * pair + 1], op_profile_.valid_bits);
        op_profile_.total_ms[slot] += static_cast<double>(ticks) * ticks_to_ms;
        ++op_profile_.calls[slot];
    }
    op_profile_.order.clear();
    ++op_profile_.flushes;
    if (op_profile_.flushes % op_profile_.interval == 0) report_op_profile_locked();
}

void Context::report_op_profile_locked() {
    const std::uint32_t flushes = op_profile_.flushes;
    std::fprintf(
        stderr, "tinytensor_vulkan_op_profile flushes=%u cumulative_from_start=1\n",
        flushes);
    const std::size_t slots = op_profile_.calls.size();
    double total_ms = 0.0;
    for (std::size_t slot = 0; slot < slots; ++slot)
        total_ms += op_profile_.total_ms[slot];
    for (std::size_t slot = 0; slot < slots; ++slot) {
        if (op_profile_.calls[slot] == 0) continue;
        const std::size_t shader = slot / OpProfile::kElementBuckets;
        const std::uint32_t bucket =
            static_cast<std::uint32_t>(slot % OpProfile::kElementBuckets);
        std::fprintf(
            stderr,
            "  %-16s elems<=%-10llu calls=%-7llu total_ms=%-9.4f "
            "per_call_ms=%-8.4f per_flush_ms=%-7.4f share_pct=%.2f\n",
            shader_name(static_cast<ShaderId>(shader)),
            // The bucket is the last power of two below the covered count, so
            // the printed bound is the one above it.
            static_cast<unsigned long long>(std::uint64_t{1} << (bucket + 1)),
            static_cast<unsigned long long>(op_profile_.calls[slot]),
            op_profile_.total_ms[slot],
            op_profile_.total_ms[slot] / static_cast<double>(op_profile_.calls[slot]),
            op_profile_.total_ms[slot] / static_cast<double>(flushes),
            total_ms > 0.0 ? 100.0 * op_profile_.total_ms[slot] / total_ms : 0.0);
    }
    std::fprintf(stderr, "  %-16s calls=%-8s total_ms=%-9.4f\n", "TOTAL", "-", total_ms);
}

std::vector<Context::OpProfileEntry> Context::op_profile() const {
    std::vector<OpProfileEntry> entries;
    const std::size_t slots = op_profile_.calls.size();
    for (std::size_t slot = 0; slot < slots; ++slot) {
        if (op_profile_.calls[slot] == 0) continue;
        const std::size_t shader = slot / OpProfile::kElementBuckets;
        entries.push_back(
            {shader_name(static_cast<ShaderId>(shader)),
             std::uint64_t{1} << (slot % OpProfile::kElementBuckets),
             op_profile_.calls[slot], op_profile_.total_ms[slot]});
    }
    return entries;
}

void Context::flush() {
    std::scoped_lock lock(mutex_);
    flush_locked();
}

// Dispatches used to be separated by a queue submission, and that is what made
// one op's writes visible to the next. In a single command buffer the
// dependency has to be explicit.
void Context::record_barrier_locked() {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    // Tensor temporaries may return to the buffer pool immediately after their
    // dispatch is recorded.  A later op can therefore reuse a previously read
    // input as its output before this batch is submitted.  Include shader reads
    // in the source scope so that reuse has a real read-to-write (WAR) ordering
    // dependency instead of relying on the driver's incidental execution order.
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(command_,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                         &barrier, 0, nullptr, 0, nullptr);
}

VkDescriptorSet Context::acquire_descriptor_locked(const VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = descriptor_pool_;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult result = vkAllocateDescriptorSets(device_, &set_info, &set);
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
        // The arena is exhausted: retire the batch (which is what makes the
        // pool reset legal) and start a new one.
        flush_locked();
        begin_batch_locked();
        result = vkAllocateDescriptorSets(device_, &set_info, &set);
    }
    check(result, "vkAllocateDescriptorSets");
    return set;
}

void Context::barrier_buffer(VkCommandBuffer cmd, VkBuffer buffer, VkAccessFlags src_access,
                             VkAccessFlags dst_access, VkPipelineStageFlags src_stage,
                             VkPipelineStageFlags dst_stage) {
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void Context::fill_zero(Buffer& buffer, std::size_t offset, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    std::scoped_lock lock(mutex_);
    const VkDeviceSize aligned_offset = offset & ~std::size_t{3};
    VkDeviceSize aligned_size = align_up_size(offset + bytes, 4) - aligned_offset;
    if (aligned_offset >= buffer.size()) {
        return;
    }
    if (aligned_offset + aligned_size > buffer.size()) {
        aligned_size = buffer.size() - aligned_offset;
    }
    if (aligned_size == 0) {
        return;
    }
    begin_batch_locked();
    vkCmdFillBuffer(command_, buffer.handle(), aligned_offset, aligned_size, 0);
    barrier_buffer(command_, buffer.handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
    if (++batch_commands_ >= kMaxCommandsPerBatch) {
        flush_locked();
    }
}

void Context::copy(Buffer& dst, std::size_t dst_offset, const Buffer& src, std::size_t src_offset,
                   std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    std::scoped_lock lock(mutex_);
    begin_batch_locked();
    VkBufferCopy region{};
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = bytes;
    vkCmdCopyBuffer(command_, src.handle(), dst.handle(), 1, &region);
    barrier_buffer(command_, dst.handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
    if (++batch_commands_ >= kMaxCommandsPerBatch) {
        flush_locked();
    }
}

void Context::upload(Buffer& dst, std::size_t dst_offset, const void* data, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    std::scoped_lock lock(mutex_);
    // The staging buffer is shared by every recorded copy, so writing the same
    // bytes again has to wait for the batch that already read them, and growing
    // the staging buffer may not free memory a recorded copy still reads.
    const std::size_t aligned = (bytes + 3U) & ~std::size_t{3};
    if (staging_ == nullptr || staging_cursor_ + aligned > staging_->size()) {
        flush_locked();
        if (staging_ == nullptr || staging_->size() < aligned) {
            ensure_staging(aligned);
        }
    }
    begin_batch_locked();
    std::memcpy(static_cast<char*>(staging_->mapped()) + staging_cursor_, data, bytes);
    VkBufferCopy region{};
    region.srcOffset = staging_cursor_;
    region.dstOffset = dst_offset;
    region.size = bytes;
    vkCmdCopyBuffer(command_, staging_->handle(), dst.handle(), 1, &region);
    barrier_buffer(command_, dst.handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    staging_cursor_ += aligned;
    if (++batch_commands_ >= kMaxCommandsPerBatch) {
        flush_locked();
    }
}

void Context::download(const Buffer& src, std::size_t src_offset, void* data, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    std::scoped_lock lock(mutex_);
    // The copy joins the open batch instead of forcing it to retire first, so a
    // readback costs one submission and one wait instead of two. The readback
    // buffer is a different buffer from the upload staging, and every readback
    // ends by retiring the batch, so the queue is idle again by the time the
    // host reads and offset zero is free for the next one.
    const std::size_t aligned = (bytes + 3U) & ~std::size_t{3};
    if (readback_staging_ == nullptr || readback_staging_->size() < aligned) {
        flush_locked();
        ensure_readback_staging(aligned);
    }
    begin_batch_locked();
    barrier_buffer(command_, src.handle(), VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferCopy region{};
    region.srcOffset = src_offset;
    region.dstOffset = 0;
    region.size = bytes;
    vkCmdCopyBuffer(command_, src.handle(), readback_staging_->handle(), 1, &region);
    barrier_buffer(command_, readback_staging_->handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_HOST_BIT);
    flush_locked();
    char* source = static_cast<char*>(readback_staging_->mapped());
    if (!readback_staging_->coherent()) {
        // The device wrote the staging buffer, so the host has to invalidate the
        // range (rounded out to whole atoms) before reading it.
        const VkDeviceSize atom = readback_staging_->non_coherent_atom_size();
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = readback_staging_->memory();
        range.offset = 0;
        const VkDeviceSize end = aligned;
        range.size =
            std::min<VkDeviceSize>((end + atom - 1) & ~(atom - 1), readback_staging_->size());
        check(vkInvalidateMappedMemoryRanges(device_, 1, &range),
              "vkInvalidateMappedMemoryRanges");
    }
    std::memcpy(data, source, bytes);
}

void Context::dispatch(ShaderId shader, std::span<const BufferBinding> bindings,
                       const void* push_constants, std::uint32_t push_bytes, std::uint32_t groups_x,
                       std::uint32_t groups_y, std::uint32_t groups_z) {
    if (groups_x == 0 || groups_y == 0 || groups_z == 0) {
        return;
    }
    std::scoped_lock lock(mutex_);
    Pipeline& pipeline = pipelines_[static_cast<std::size_t>(shader)];
    if (bindings.size() != pipeline.binding_count) {
        throw std::invalid_argument("Vulkan dispatch binding count mismatch");
    }

    begin_batch_locked();
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!push_descriptors_) {
        set = acquire_descriptor_locked(pipeline.set_layout);
    }

    // Per-dispatch scratch kept off the heap: this runs once per op, so two
    // vector allocations per op add up over a step.
    descriptor_infos_.resize(bindings.size());
    descriptor_writes_.resize(bindings.size());
    std::vector<VkDescriptorBufferInfo>& infos = descriptor_infos_;
    std::vector<VkWriteDescriptorSet>& writes = descriptor_writes_;
    for (std::uint32_t i = 0; i < bindings.size(); ++i) {
        infos[i].buffer = bindings[i].buffer;
        infos[i].offset = bindings[i].offset;
        infos[i].range = bindings[i].range;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        // dstSet is ignored by vkCmdPushDescriptorSetKHR.  Keeping it null in
        // the push path also prevents accidentally retaining a stale set.
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    if (push_descriptors_) {
        cmd_push_descriptor_(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0,
                             static_cast<std::uint32_t>(writes.size()), writes.data());
    } else {
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0,
                               nullptr);
    }

    vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
    if (!push_descriptors_) {
        vkCmdBindDescriptorSets(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1,
                                &set, 0, nullptr);
    }
    if (push_bytes != 0 && push_constants != nullptr) {
        vkCmdPushConstants(command_, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes,
                           push_constants);
    }
    const bool profile = op_profile_.enabled &&
        op_profile_.recorded < kMaxCommandsPerBatch;
    if (profile) {
        vkCmdWriteTimestamp(
            command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, op_profile_.pool,
            op_profile_.recorded * 2);
    }
    vkCmdDispatch(command_, groups_x, groups_y, groups_z);
    if (profile) {
        vkCmdWriteTimestamp(
            command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, op_profile_.pool,
            op_profile_.recorded * 2 + 1);
        // The dispatch grid says how much work the op covers; bucketing it to
        // the next power of two separates a 630k-parameter update from a 10M
        // spherical-harmonic one without threading a label through every op.
        const std::uint64_t covered = std::uint64_t{groups_x} * 256u * groups_y * groups_z;
        std::uint32_t bucket = 0;
        while (bucket + 1 < OpProfile::kElementBuckets &&
               (std::uint64_t{1} << (bucket + 1)) < covered)
            ++bucket;
        op_profile_.order.push_back(
            static_cast<std::uint32_t>(shader) * OpProfile::kElementBuckets + bucket);
        ++op_profile_.recorded;
    }
    // The next dispatch in this batch reads what this one wrote.
    record_barrier_locked();
    if (++batch_commands_ >= kMaxCommandsPerBatch) {
        flush_locked();
    }
}

} // namespace tinytensor::vulkan::runtime

namespace tinytensor::vulkan {

bool available() {
    return runtime::Context::available();
}

const DeviceInfo& device_info() {
    return runtime::Context::get().info();
}

DeviceHandles device_handles() {
    const auto& context = runtime::Context::get();
    return {context.instance(), context.physical_device(), context.device(), context.queue(),
            context.queue_family()};
}

void synchronize() {
    runtime::Context::get().flush();
}

void shutdown() {
    runtime::Context::shutdown();
}

} // namespace tinytensor::vulkan
