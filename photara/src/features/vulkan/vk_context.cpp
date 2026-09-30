#include "vk_context.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" {
#include "sift_convert.hlsl.embedded.hpp"
#include "sift_upsample.hlsl.embedded.hpp"
#include "sift_downsample.hlsl.embedded.hpp"
#include "sift_blur.hlsl.embedded.hpp"
#include "sift_dog.hlsl.embedded.hpp"
#include "sift_key.hlsl.embedded.hpp"
#include "sift_hist_init.hlsl.embedded.hpp"
#include "sift_hist_reduce.hlsl.embedded.hpp"
#include "sift_list_gen.hlsl.embedded.hpp"
#include "sift_orientation.hlsl.embedded.hpp"
#include "sift_descriptor.hlsl.embedded.hpp"
#include "sift_descriptor_norm.hlsl.embedded.hpp"
#include "sift_match_tiles.hlsl.embedded.hpp"
#include "sift_match_tiles_64.hlsl.embedded.hpp"
#include "sift_match_tiles_160.hlsl.embedded.hpp"
#include "sift_match_tiles_dp4a_64.hlsl.embedded.hpp"
#include "sift_match_tiles_dp4a_128.hlsl.embedded.hpp"
#include "sift_match_tiles_dp4a_160.hlsl.embedded.hpp"
#include "sift_match_wave_dp4a.hlsl.embedded.hpp"
#include "sift_match_finish.hlsl.embedded.hpp"
}

namespace photara::features::vulkan_backend {
namespace {

using SpirVBytes = std::span<const unsigned char>;

void check(VkResult result, const char* what) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string("Vulkan feature backend: ") + what +
                                 " failed (" + std::to_string(result) + ")");
}

constexpr std::uint32_t kMaxShaderBindings = 8;

struct ShaderInfo {
    SpirVBytes spirv;
    std::uint32_t bindings;
    std::uint32_t push_bytes;
};

ShaderInfo shader_info(Shader shader) {
    switch (shader) {
        case Shader::Convert:
            return {SpirVBytes{sift_convert_hlsl_spv, sizeof(sift_convert_hlsl_spv)}, 2, 16};
        case Shader::Upsample:
            return {SpirVBytes{sift_upsample_hlsl_spv, sizeof(sift_upsample_hlsl_spv)}, 2, 16};
        case Shader::Downsample:
            return {SpirVBytes{sift_downsample_hlsl_spv, sizeof(sift_downsample_hlsl_spv)}, 2, 16};
        case Shader::Blur:
            return {SpirVBytes{sift_blur_hlsl_spv, sizeof(sift_blur_hlsl_spv)}, 3, 16};
        case Shader::Dog:
            return {SpirVBytes{sift_dog_hlsl_spv, sizeof(sift_dog_hlsl_spv)}, 4, 16};
        case Shader::Key:
            return {SpirVBytes{sift_key_hlsl_spv, sizeof(sift_key_hlsl_spv)}, 4, 24};
        case Shader::HistInit:
            return {SpirVBytes{sift_hist_init_hlsl_spv, sizeof(sift_hist_init_hlsl_spv)}, 2, 16};
        case Shader::HistReduce:
            return {SpirVBytes{sift_hist_reduce_hlsl_spv, sizeof(sift_hist_reduce_hlsl_spv)}, 2, 16};
        case Shader::ListGen:
            return {SpirVBytes{sift_list_gen_hlsl_spv, sizeof(sift_list_gen_hlsl_spv)}, 2, 8};
        case Shader::Orientation:
            return {SpirVBytes{sift_orientation_hlsl_spv, sizeof(sift_orientation_hlsl_spv)}, 3, 32};
        case Shader::Descriptor:
            return {SpirVBytes{sift_descriptor_hlsl_spv, sizeof(sift_descriptor_hlsl_spv)}, 3, 16};
        case Shader::DescriptorNorm:
            return {SpirVBytes{sift_descriptor_norm_hlsl_spv, sizeof(sift_descriptor_norm_hlsl_spv)}, 1, 8};
        case Shader::MatchTiles:
            return {SpirVBytes{sift_match_tiles_hlsl_spv, sizeof(sift_match_tiles_hlsl_spv)}, 4, 16};
        case Shader::MatchTiles64:
            return {SpirVBytes{sift_match_tiles_64_hlsl_spv,
                               sizeof(sift_match_tiles_64_hlsl_spv)},
                    4, 16};
        case Shader::MatchTiles160:
            return {SpirVBytes{sift_match_tiles_160_hlsl_spv,
                               sizeof(sift_match_tiles_160_hlsl_spv)},
                    4, 16};
        case Shader::MatchTilesDp4a64:
            return {SpirVBytes{sift_match_tiles_dp4a_64_hlsl_spv,
                               sizeof(sift_match_tiles_dp4a_64_hlsl_spv)},
                    4, 16};
        case Shader::MatchTilesDp4a128:
            return {SpirVBytes{sift_match_tiles_dp4a_128_hlsl_spv,
                               sizeof(sift_match_tiles_dp4a_128_hlsl_spv)},
                    4, 16};
        case Shader::MatchTilesDp4a160:
            return {SpirVBytes{sift_match_tiles_dp4a_160_hlsl_spv,
                               sizeof(sift_match_tiles_dp4a_160_hlsl_spv)},
                    4, 16};
        case Shader::MatchWaveDp4a:
            return {SpirVBytes{sift_match_wave_dp4a_hlsl_spv,
                               sizeof(sift_match_wave_dp4a_hlsl_spv)},
                    4, 16};
        case Shader::MatchFinish:
            return {SpirVBytes{sift_match_finish_hlsl_spv, sizeof(sift_match_finish_hlsl_spv)}, 3, 32};
        default: break;
    }
    throw std::logic_error("Unknown feature shader");
}

std::int64_t device_score(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device, &properties);
    if (properties.apiVersion < VK_API_VERSION_1_1) return -1;
    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
    bool has_compute = false;
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());
    for (const auto& family : families) {
        if ((family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) has_compute = true;
    }
    if (!has_compute) return -1;
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        return 300;
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
        return 100;
    return -1;
}

}  // namespace

// ---- Buffer -----------------------------------------------------------------

Buffer::Buffer(VkPhysicalDevice physical, VkDevice logical, VkDeviceSize bytes,
               bool host_visible, bool host_cached)
    : device_(logical), size_(std::max<VkDeviceSize>(bytes, 4)) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size_;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    check(vkCreateBuffer(device_, &buffer_info, nullptr, &handle_), "vkCreateBuffer");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, handle_, &requirements);

    VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags preferred = 0;
    if (host_visible) {
        required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        preferred = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if (host_cached) preferred |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }

    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    std::optional<std::pair<std::uint32_t, std::uint32_t>> best;
    for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if ((requirements.memoryTypeBits & (1U << index)) == 0) continue;
        const auto flags = properties.memoryTypes[index].propertyFlags;
        if ((flags & required) != required) continue;
        const auto bonus = flags & preferred;
        const auto bonus_count = static_cast<std::uint32_t>(
            std::popcount(static_cast<std::uint32_t>(bonus)));
        if (!best || bonus_count > best->second) best = std::make_pair(index, bonus_count);
    }
    if (!best)
        throw std::runtime_error("Vulkan feature backend: no compatible memory type");

    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = best->first;
    check(vkAllocateMemory(device_, &allocation, nullptr, &memory_), "vkAllocateMemory");
    check(vkBindBufferMemory(device_, handle_, memory_, 0), "vkBindBufferMemory");

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(physical, &device_properties);
    non_coherent_atom_size_ =
        device_properties.limits.nonCoherentAtomSize > 1
            ? device_properties.limits.nonCoherentAtomSize
            : 1;
    if ((properties.memoryTypes[best->first].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
        coherent_ = true;
    } else {
        coherent_ = false;
    }
    if (host_visible) (void)map();
}

Buffer::~Buffer() {
    if (device_ == VK_NULL_HANDLE) return;
    unmap();
    if (handle_ != VK_NULL_HANDLE) vkDestroyBuffer(device_, handle_, nullptr);
    if (memory_ != VK_NULL_HANDLE) vkFreeMemory(device_, memory_, nullptr);
}

Buffer::Buffer(Buffer&& other) noexcept
    : device_(other.device_),
      handle_(other.handle_),
      memory_(other.memory_),
      size_(other.size_),
      mapped_(other.mapped_),
      coherent_(other.coherent_),
      non_coherent_atom_size_(other.non_coherent_atom_size_) {
    other.device_ = VK_NULL_HANDLE;
    other.handle_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.size_ = 0;
    other.mapped_ = nullptr;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this == &other) return *this;
    Buffer doomed(std::move(*this));
    device_ = other.device_;
    handle_ = other.handle_;
    memory_ = other.memory_;
    size_ = other.size_;
    mapped_ = other.mapped_;
    coherent_ = other.coherent_;
    non_coherent_atom_size_ = other.non_coherent_atom_size_;
    other.device_ = VK_NULL_HANDLE;
    other.handle_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.size_ = 0;
    other.mapped_ = nullptr;
    return *this;
}

void* Buffer::map() {
    if (mapped_ != nullptr) return mapped_;
    if (memory_ == VK_NULL_HANDLE)
        throw std::runtime_error("Vulkan feature backend: cannot map device memory");
    check(vkMapMemory(device_, memory_, 0, size_, 0, &mapped_), "vkMapMemory");
    return mapped_;
}

void Buffer::unmap() noexcept {
    if (mapped_ == nullptr) return;
    vkUnmapMemory(device_, memory_);
    mapped_ = nullptr;
}

void Buffer::invalidate() const {
    if (coherent_ || memory_ == VK_NULL_HANDLE) return;
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory_;
    range.offset = 0;
    range.size = size_;
    check(vkInvalidateMappedMemoryRanges(device_, 1, &range),
          "vkInvalidateMappedMemoryRanges");
}

void Buffer::flush(VkDeviceSize offset, VkDeviceSize bytes) const {
    if (coherent_ || memory_ == VK_NULL_HANDLE) return;
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory_;
    range.offset = (offset / non_coherent_atom_size_) * non_coherent_atom_size_;
    range.size = std::min(size_ - range.offset,
                          ((bytes + non_coherent_atom_size_ - 1) /
                           non_coherent_atom_size_) *
                              non_coherent_atom_size_);
    check(vkFlushMappedMemoryRanges(device_, 1, &range), "vkFlushMappedMemoryRanges");
}

// ---- BufferRef ----------------------------------------------------------------

BufferRef::BufferRef(Context& context, std::unique_ptr<Buffer> buffer,
                     bool host_visible, std::size_t bucket)
    : context_(&context),
      buffer_(std::move(buffer)),
      host_visible_(host_visible),
      bucket_(bucket) {}

BufferRef::~BufferRef() {
    if (context_ != nullptr && buffer_ != nullptr)
        context_->recycle(std::move(buffer_), host_visible_, bucket_);
}

BufferRef::BufferRef(BufferRef&& other) noexcept
    : context_(other.context_),
      buffer_(std::move(other.buffer_)),
      host_visible_(other.host_visible_),
      bucket_(other.bucket_) {
    other.context_ = nullptr;
}

BufferRef& BufferRef::operator=(BufferRef&& other) noexcept {
    if (this == &other) return *this;
    BufferRef doomed(std::move(*this));
    context_ = other.context_;
    buffer_ = std::move(other.buffer_);
    host_visible_ = other.host_visible_;
    bucket_ = other.bucket_;
    other.context_ = nullptr;
    return *this;
}

// ---- Context -----------------------------------------------------------------

Context::Context() {
    VkApplicationInfo application{};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "Photara Features";
    application.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    application.pEngineName = "Photara";
    application.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    application.apiVersion = VK_API_VERSION_1_1;
    // Promoting to 1.3 when the loader allows it keeps the (promoted) feature
    // structures used for device creation legal.
    {
        std::uint32_t loader_version = 0;
        if (vkEnumerateInstanceVersion(&loader_version) == VK_SUCCESS &&
            loader_version > application.apiVersion)
            application.apiVersion = std::min(loader_version, VK_API_VERSION_1_3);
    }

    VkInstanceCreateInfo create{};
    create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create.pApplicationInfo = &application;
    check(vkCreateInstance(&create, nullptr, &instance_), "vkCreateInstance");

    std::uint32_t device_count = 0;
    check(vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
          "vkEnumeratePhysicalDevices");
    if (device_count == 0)
        throw std::runtime_error("Vulkan feature backend: no devices");
    std::vector<VkPhysicalDevice> devices(device_count);
    check(vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
          "vkEnumeratePhysicalDevices");

    std::int64_t best_score = -1;
    for (std::uint32_t index = 0; index < device_count; ++index) {
        const auto score = device_score(devices[index]);
        if (score > best_score) {
            best_score = score;
            physical_ = devices[index];
            device_index_ = index;
        }
    }
    if (best_score < 0)
        throw std::runtime_error("Vulkan feature backend: no compute device");

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, families.data());
    bool queue_found = false;
    for (std::uint32_t family = 0; family < family_count; ++family) {
        const auto flags = families[family].queueFlags;
        if ((flags & VK_QUEUE_COMPUTE_BIT) != 0) {
            queue_family_ = family;
            queue_found = true;
            break;
        }
    }
    if (!queue_found)
        throw std::runtime_error("Vulkan feature backend: no compute queue");

    // Integer dot product (dp4a) lets the descriptor matcher consume four
    // descriptor bytes per instruction. The SPIR-V variant is 1.5, so it is only
    // taken on devices that report the feature and support Vulkan 1.2 or newer.
    std::vector<VkExtensionProperties> device_extensions;
    {
        std::uint32_t count = 0;
        if (vkEnumerateDeviceExtensionProperties(physical_, nullptr, &count, nullptr) ==
                VK_SUCCESS &&
            count > 0) {
            device_extensions.resize(count);
            if (vkEnumerateDeviceExtensionProperties(physical_, nullptr, &count,
                                                     device_extensions.data()) !=
                VK_SUCCESS)
                device_extensions.clear();
        }
    }
    const bool has_dot_extension = std::any_of(
        device_extensions.begin(), device_extensions.end(),
        [](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName,
                               VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) == 0;
        });
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot_features{};
    dot_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    std::uint32_t device_api_version = VK_API_VERSION_1_0;
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical_, &properties);
        device_api_version = properties.apiVersion;
        max_shared_bytes_ = properties.limits.maxComputeSharedMemorySize;
        storage_buffer_offset_alignment_ =
            properties.limits.minStorageBufferOffsetAlignment;
        VkPhysicalDeviceSubgroupProperties subgroup{};
        subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &subgroup;
        vkGetPhysicalDeviceProperties2(physical_, &properties2);
        subgroup_size_ = subgroup.subgroupSize;
    }
    if (has_dot_extension || device_api_version >= VK_API_VERSION_1_3) {
        VkPhysicalDeviceFeatures2 query{};
        query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        query.pNext = &dot_features;
        vkGetPhysicalDeviceFeatures2(physical_, &query);
    }
    const bool use_dot_product = dot_features.shaderIntegerDotProduct != 0 &&
                                 device_api_version >= VK_API_VERSION_1_2;

    const float priority = 1.0F;
    VkDeviceQueueCreateInfo queue_create{};
    queue_create.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create.queueFamilyIndex = queue_family_;
    queue_create.queueCount = 1;
    queue_create.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_create{};
    device_create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create.queueCreateInfoCount = 1;
    device_create.pQueueCreateInfos = &queue_create;
    std::array<const char*, 1> enabled_extensions{};
    if (use_dot_product) {
        dot_features.shaderIntegerDotProduct = VK_TRUE;
        device_create.pNext = &dot_features;
        if (has_dot_extension) {
            enabled_extensions[0] =
                VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME;
            device_create.enabledExtensionCount =
                static_cast<std::uint32_t>(enabled_extensions.size());
            device_create.ppEnabledExtensionNames = enabled_extensions.data();
        }
    }
    check(vkCreateDevice(physical_, &device_create, nullptr, &device_),
          "vkCreateDevice");
    integer_dot_product_ = use_dot_product;
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    vkGetPhysicalDeviceMemoryProperties(physical_, &memory_properties_);
    {
        std::uint64_t device_local = 0;
        for (std::uint32_t heap = 0; heap < memory_properties_.memoryHeapCount; ++heap)
            if ((memory_properties_.memoryHeaps[heap].flags &
                 VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
                device_local += memory_properties_.memoryHeaps[heap].size;
        budget_bytes_ = std::max<std::uint64_t>(256ULL << 20, device_local / 3);
    }

    photara::vk::ExternalDevice external;
    external.instance = instance_;
    external.physical = physical_;
    external.device = device_;
    external.queue = queue_;
    external.queue_family = queue_family_;
    external.enabled.push_descriptors = false;
    external.enabled.integer_dot_product = use_dot_product;
    runtime_ = photara::vk::Device::adopt(external);

    device_pools_.resize(48);
    host_pools_.resize(48);
}

Context::~Context() {
    std::vector<std::unique_ptr<Session>> sessions;
    {
        std::lock_guard lock(mutex_);
        alive_ = false;
        sessions = std::move(sessions_);
    }
    if (device_ == VK_NULL_HANDLE) {
        if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
        return;
    }
    if (runtime_.valid()) {
        photara::vk::QueueLock lock(runtime_);
        vkDeviceWaitIdle(device_);
    } else if (queue_ != VK_NULL_HANDLE) {
        photara::vk::QueueLock lock(queue_);
        vkDeviceWaitIdle(device_);
    } else {
        vkDeviceWaitIdle(device_);
    }
    sessions.clear();
    for (photara::vk::ComputePipeline& pipeline : pipelines_) pipeline = {};
    runtime_ = {};
    vkDestroyDevice(device_, nullptr);
    if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
}

Context* Context::try_get() {
    static Context* instance = []() -> Context* {
        try {
            return new Context();
        } catch (const std::exception&) {
            return nullptr;
        }
    }();
    return instance;
}

Context& Context::get() {
    auto* instance = try_get();
    if (instance == nullptr)
        throw std::runtime_error(
            "Vulkan feature backend is unavailable on this machine");
    return *instance;
}

bool Context::available() { return try_get() != nullptr; }

Context::Session& Context::session() {
    // Each thread records into its own command buffer; the context owns the
    // sessions so their handles are released with the device.
    thread_local Session* cached = nullptr;
    if (cached != nullptr) return *cached;
    std::lock_guard lock(mutex_);
    sessions_.push_back(std::make_unique<Session>());
    cached = sessions_.back().get();
    return *cached;
}

VkPhysicalDeviceMemoryProperties Context::memory_properties() const {
    return memory_properties_;
}

Context::TilePlan Context::descriptor_tile_plan() const noexcept {
    // Shared memory per block is the query tile (rows * 128 B) plus the train
    // tile (4 KB) and the dot matrix (rows * 128 B). Taller tiles amortize the
    // fixed per-block work, which is what the matcher is bound by, so the host
    // picks the tallest tile the device's limit allows.
    const auto fits = [&](std::uint32_t rows) {
        return static_cast<std::uint64_t>(rows) * 256 + 4096 <= max_shared_bytes_;
    };
    if (integer_dot_product_) {
        if (subgroup_size_ == 32)
            return TilePlan{Shader::MatchWaveDp4a, 512};
        if (fits(160)) return TilePlan{Shader::MatchTilesDp4a160, 160};
        if (fits(128)) return TilePlan{Shader::MatchTilesDp4a128, 128};
        if (fits(64)) return TilePlan{Shader::MatchTilesDp4a64, 64};
    }
    if (fits(160)) return TilePlan{Shader::MatchTiles160, 160};
    if (fits(64)) return TilePlan{Shader::MatchTiles64, 64};
    return TilePlan{Shader::MatchTiles, 32};
}

void Context::acquire_device_budget(std::uint64_t bytes) {
    std::unique_lock lock(budget_mutex_);
    budget_condition_.wait(lock, [&] {
        return budget_in_use_ == 0 || budget_in_use_ + bytes <= budget_bytes_;
    });
    budget_in_use_ += bytes;
}

void Context::release_device_budget(std::uint64_t bytes) {
    std::lock_guard lock(budget_mutex_);
    budget_in_use_ = budget_in_use_ > bytes ? budget_in_use_ - bytes : 0;
    budget_condition_.notify_all();
}

VkDeviceSize Context::pooled_size(VkDeviceSize bytes) {
    if (bytes <= 16) return 16;
    const std::size_t bucket = 64 - static_cast<std::size_t>(
                                       std::countl_zero(static_cast<std::uint64_t>(bytes - 1)));
    return static_cast<VkDeviceSize>(1) << bucket;
}

BufferRef Context::allocate(VkDeviceSize bytes, bool host_visible, bool host_cached) {
    const VkDeviceSize rounded = pooled_size(bytes);
    const std::size_t bucket = static_cast<std::size_t>(
        std::bit_width(static_cast<std::uint64_t>(rounded - 1)));
    auto& pools = host_visible ? host_pools_ : device_pools_;
    if (pools.size() <= bucket) pools.resize(bucket + 1);
    std::unique_ptr<Buffer> buffer;
    {
        std::lock_guard lock(mutex_);
        auto& free_list = pools[bucket].free_buffers;
        if (!free_list.empty()) {
            buffer = std::move(free_list.back());
            free_list.pop_back();
        }
    }
    if (buffer == nullptr)
        buffer = std::make_unique<Buffer>(physical_, device_, rounded, host_visible,
                                          host_cached);
    return BufferRef(*this, std::move(buffer), host_visible, bucket);
}

void Context::recycle(std::unique_ptr<Buffer> buffer, bool host_visible,
                      std::size_t bucket) {
    std::lock_guard lock(mutex_);
    if (!alive_ || buffer == nullptr) return;
    auto& pools = host_visible ? host_pools_ : device_pools_;
    if (pools.size() <= bucket) pools.resize(bucket + 1);
    pools[bucket].free_buffers.push_back(std::move(buffer));
}

const photara::vk::ComputePipeline& Context::pipeline(Shader shader) {
    auto& entry = pipelines_[static_cast<std::size_t>(shader)];
    if (entry.valid()) return entry;

    const auto info = shader_info(shader);
    if (info.bindings > kMaxShaderBindings)
        throw std::logic_error("Too many shader bindings");
    const auto* code = reinterpret_cast<const std::byte*>(info.spirv.data());
    entry = runtime_.create_compute(
        std::span<const std::byte>{code, info.spirv.size()}, info.bindings,
        info.push_bytes);
    return entry;
}

void Context::begin() {
    Session& session = this->session();
    std::lock_guard lock(mutex_);
    if (!session.has_encoder) {
        session.encoder = runtime_.encoder(photara::vk::BarrierPolicy::after_compute);
        session.has_encoder = true;
    }
    if (!session.encoder.recording()) {
        (void)session.encoder.native();
        session.downloads.clear();
    }
}

void Context::dispatch(Shader shader, std::span<const BufferBinding> bindings,
                       const void* push_constants, std::uint32_t push_bytes,
                       std::uint32_t groups_x, std::uint32_t groups_y,
                       std::uint32_t groups_z) {
    Session& session = this->session();
    std::lock_guard lock(mutex_);
    if (!session.has_encoder || !session.encoder.recording())
        throw std::logic_error("dispatch() called without begin()");

    const photara::vk::ComputePipeline& entry = pipeline(shader);
    if (push_bytes != entry.push_bytes())
        throw std::logic_error("Push constant size mismatch for shader " +
                               std::to_string(static_cast<std::uint32_t>(shader)));
    if (bindings.size() != entry.binding_count())
        throw std::logic_error("Descriptor binding count mismatch");
    std::array<photara::vk::BufferBinding, kMaxShaderBindings> converted{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        converted[index].buffer = bindings[index].buffer;
        converted[index].offset = bindings[index].offset;
        converted[index].range = bindings[index].range;
    }
    // after_compute inserts the same compute-to-compute/transfer barrier the
    // SIFT passes used to record after every dispatch.
    session.encoder.dispatch(
        entry,
        std::span<const photara::vk::BufferBinding>{converted.data(), bindings.size()},
        push_constants, push_bytes, groups_x, groups_y, groups_z);
}

void Context::queue_download(const Buffer& source, VkDeviceSize source_offset,
                             VkDeviceSize bytes, const Buffer& destination,
                             VkDeviceSize destination_offset) {
    Session& session = this->session();
    std::lock_guard lock(mutex_);
    if (!session.has_encoder || !session.encoder.recording())
        throw std::logic_error("queue_download() called without begin()");
    if (bytes == 0) return;
    const VkCommandBuffer command = session.encoder.native();
    const VkBufferCopy region{source_offset, destination_offset, bytes};
    vkCmdCopyBuffer(command, source.handle(), destination.handle(), 1, &region);
    // The post-dispatch barrier covers compute->transfer, but a queued download
    // can also directly follow another download.
    session.encoder.barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    session.downloads.push_back({&source, source_offset, bytes, &destination,
                                 destination_offset});
}

void Context::queue_upload(const Buffer& source, VkDeviceSize source_offset,
                           VkDeviceSize bytes, const Buffer& destination,
                           VkDeviceSize destination_offset) {
    Session& session = this->session();
    std::lock_guard lock(mutex_);
    if (!session.has_encoder || !session.encoder.recording())
        throw std::logic_error("queue_upload() called without begin()");
    if (bytes == 0) return;
    const VkCommandBuffer command = session.encoder.native();
    const VkBufferCopy region{source_offset, destination_offset, bytes};
    vkCmdCopyBuffer(command, source.handle(), destination.handle(), 1, &region);
    session.encoder.barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
}

void Context::fill_u32(const Buffer& destination, VkDeviceSize offset,
                       VkDeviceSize bytes, std::uint32_t value) {
    Session& session = this->session();
    std::lock_guard lock(mutex_);
    if (!session.has_encoder || !session.encoder.recording())
        throw std::logic_error("fill_u32() called without begin()");
    if (bytes == 0) return;
    const VkCommandBuffer command = session.encoder.native();
    vkCmdFillBuffer(command, destination.handle(), offset, bytes, value);
    session.encoder.barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT);
}

void Context::submit_and_wait() {
    Session& session = this->session();
    {
        std::lock_guard lock(mutex_);
        if (!session.has_encoder || !session.encoder.recording())
            throw std::logic_error("submit_and_wait() called without begin()");
        session.encoder.submit();
    }
    // Wait outside the context mutex so another worker can record and submit.
    session.encoder.wait();
    for (const auto& download : session.downloads)
        download.destination->invalidate();
    session.downloads.clear();
}

}  // namespace photara::features::vulkan_backend
