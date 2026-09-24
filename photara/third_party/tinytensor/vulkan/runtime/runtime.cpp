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

#include "cat.hlsl.embedded.hpp"
#include "compact.hlsl.embedded.hpp"
#include "cumsum.hlsl.embedded.hpp"
#include "elementwise.hlsl.embedded.hpp"
#include "index_fill.hlsl.embedded.hpp"
#include "index_select.hlsl.embedded.hpp"
#include "mask_flags.hlsl.embedded.hpp"
#include "matmul.hlsl.embedded.hpp"
#include "multinomial.hlsl.embedded.hpp"
#include "pool.hlsl.embedded.hpp"
#include "random.hlsl.embedded.hpp"
#include "reduce.hlsl.embedded.hpp"
#include "scan_add.hlsl.embedded.hpp"
#include "scan_block.hlsl.embedded.hpp"
#include "scatter.hlsl.embedded.hpp"
#include "select_compact.hlsl.embedded.hpp"
#include "strided_copy.hlsl.embedded.hpp"

namespace tinytensor::vulkan::runtime {
namespace {

constexpr std::size_t kMinBufferBytes = 16;

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
        {ShaderId::Elementwise, as_bytes(span{elementwise_hlsl_spv}), 4, 72},
        {ShaderId::StridedCopy, as_bytes(span{strided_copy_hlsl_spv}), 2, 96},
        {ShaderId::IndexSelect, as_bytes(span{index_select_hlsl_spv}), 3, 40},
        {ShaderId::IndexFill, as_bytes(span{index_fill_hlsl_spv}), 2, 40},
        {ShaderId::MaskFlags, as_bytes(span{mask_flags_hlsl_spv}), 2, 16},
        {ShaderId::ScanBlock, as_bytes(span{scan_block_hlsl_spv}), 3, 24},
        {ShaderId::ScanAdd, as_bytes(span{scan_add_hlsl_spv}), 2, 16},
        {ShaderId::Compact, as_bytes(span{compact_hlsl_spv}), 3, 52},
        {ShaderId::Multinomial, as_bytes(span{multinomial_hlsl_spv}), 3, 32},
        {ShaderId::Reduce, as_bytes(span{reduce_hlsl_spv}), 3, 40},
        {ShaderId::Matmul, as_bytes(span{matmul_hlsl_spv}), 3, 56},
        {ShaderId::Random, as_bytes(span{random_hlsl_spv}), 2, 44},
        {ShaderId::Cumsum, as_bytes(span{cumsum_hlsl_spv}), 2, 28},
        {ShaderId::Pool, as_bytes(span{pool_hlsl_spv}), 2, 48},
        {ShaderId::Scatter, as_bytes(span{scatter_hlsl_spv}), 3, 44},
        {ShaderId::SelectCompact, as_bytes(span{select_compact_hlsl_spv}), 4, 28},
        {ShaderId::Cat, as_bytes(span{cat_hlsl_spv}), 3, 24},
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

Buffer::Buffer(VkPhysicalDevice physical, VkDevice logical, VkDeviceSize bytes, bool host_visible)
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
    if (host_visible) {
        allocation.memoryTypeIndex = find_memory_type(physical, requirements.memoryTypeBits, host_flags);
    } else {
        const auto device_local = try_find_memory_type(
            physical, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        allocation.memoryTypeIndex = device_local.value_or(
            find_memory_type(physical, requirements.memoryTypeBits, host_flags));
    }
    check(vkAllocateMemory(device_, &allocation, nullptr, &memory_), "vkAllocateMemory");
    check(vkBindBufferMemory(device_, handle_, memory_, 0), "vkBindBufferMemory");
    if (host_visible) {
        check(vkMapMemory(device_, memory_, 0, size_, 0, &mapped_), "vkMapMemory");
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
    dummy_ = std::make_unique<Buffer>(physical_, device_, kMinBufferBytes, false);
    create_pipelines();
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
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
    if (has_device_extension(physical_, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    }
#endif

    VkDeviceCreateInfo create{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    create.queueCreateInfoCount = 1;
    create.pQueueCreateInfos = &queue_info;
    create.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    check(vkCreateDevice(physical_, &create, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
}

void Context::create_pools() {
    VkCommandPoolCreateInfo command_pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool.queueFamilyIndex = queue_family_;
    check(vkCreateCommandPool(device_, &command_pool, nullptr, &command_pool_), "vkCreateCommandPool");

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
        vkDeviceWaitIdle(device_);
    }
    dummy_.reset();
    staging_.reset();
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
    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
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
    return std::make_shared<Buffer>(physical_, device_, static_cast<VkDeviceSize>(bytes), false);
}

Buffer& Context::dummy() {
    return *dummy_;
}

void Context::ensure_staging(std::size_t bytes) {
    if (staging_ && staging_->size() >= bytes) {
        return;
    }
    staging_ = std::make_unique<Buffer>(physical_, device_, static_cast<VkDeviceSize>(bytes), true);
}

VkCommandBuffer Context::begin_commands() {
    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = command_pool_;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    check(vkAllocateCommandBuffers(device_, &alloc, &cmd), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
    return cmd;
}

void Context::submit(VkCommandBuffer cmd) {
    check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    check(vkQueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
    check(vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
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
    VkCommandBuffer cmd = begin_commands();
    vkCmdFillBuffer(cmd, buffer.handle(), aligned_offset, aligned_size, 0);
    barrier_buffer(cmd, buffer.handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
    submit(cmd);
}

void Context::copy(Buffer& dst, std::size_t dst_offset, const Buffer& src, std::size_t src_offset,
                   std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    std::scoped_lock lock(mutex_);
    VkCommandBuffer cmd = begin_commands();
    VkBufferCopy region{};
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = bytes;
    vkCmdCopyBuffer(cmd, src.handle(), dst.handle(), 1, &region);
    barrier_buffer(cmd, dst.handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
    submit(cmd);
}

void Context::upload(Buffer& dst, std::size_t dst_offset, const void* data, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    std::scoped_lock lock(mutex_);
    ensure_staging(bytes);
    std::memcpy(staging_->mapped(), data, bytes);
    VkCommandBuffer cmd = begin_commands();
    VkBufferCopy region{};
    region.srcOffset = 0;
    region.dstOffset = dst_offset;
    region.size = bytes;
    vkCmdCopyBuffer(cmd, staging_->handle(), dst.handle(), 1, &region);
    barrier_buffer(cmd, dst.handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    submit(cmd);
}

void Context::download(const Buffer& src, std::size_t src_offset, void* data, std::size_t bytes) {
    if (bytes == 0) {
        return;
    }
    std::scoped_lock lock(mutex_);
    ensure_staging(bytes);
    VkCommandBuffer cmd = begin_commands();
    barrier_buffer(cmd, src.handle(), VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferCopy region{};
    region.srcOffset = src_offset;
    region.dstOffset = 0;
    region.size = bytes;
    vkCmdCopyBuffer(cmd, src.handle(), staging_->handle(), 1, &region);
    submit(cmd);
    std::memcpy(data, staging_->mapped(), bytes);
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

    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = descriptor_pool_;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &pipeline.set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult alloc_result = vkAllocateDescriptorSets(device_, &set_info, &set);
    if (alloc_result == VK_ERROR_OUT_OF_POOL_MEMORY || alloc_result == VK_ERROR_FRAGMENTED_POOL) {
        check(vkResetDescriptorPool(device_, descriptor_pool_, 0), "vkResetDescriptorPool");
        alloc_result = vkAllocateDescriptorSets(device_, &set_info, &set);
    }
    check(alloc_result, "vkAllocateDescriptorSets");

    std::vector<VkDescriptorBufferInfo> infos(bindings.size());
    std::vector<VkWriteDescriptorSet> writes(bindings.size());
    for (std::uint32_t i = 0; i < bindings.size(); ++i) {
        infos[i].buffer = bindings[i].buffer;
        infos[i].offset = bindings[i].offset;
        infos[i].range = bindings[i].range;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkCommandBuffer cmd = begin_commands();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout, 0, 1, &set, 0, nullptr);
    if (push_bytes != 0 && push_constants != nullptr) {
        vkCmdPushConstants(cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push_constants);
    }
    vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
    submit(cmd);
    vkFreeDescriptorSets(device_, descriptor_pool_, 1, &set);
}

} // namespace tinytensor::vulkan::runtime

namespace tinytensor::vulkan {

bool available() {
    return runtime::Context::available();
}

const DeviceInfo& device_info() {
    return runtime::Context::get().info();
}

void shutdown() {
    runtime::Context::shutdown();
}

} // namespace tinytensor::vulkan
