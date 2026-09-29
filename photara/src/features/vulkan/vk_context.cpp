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
        const auto bonus_count = std::popcount(static_cast<std::uint32_t>(bonus));
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
    check(vkCreateDevice(physical_, &device_create, nullptr, &device_),
          "vkCreateDevice");
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

    VkCommandPoolCreateInfo pool_create{};
    pool_create.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_create.queueFamilyIndex = queue_family_;
    pool_create.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(device_, &pool_create, nullptr, &command_pool_),
          "vkCreateCommandPool");

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
    vkDeviceWaitIdle(device_);
    for (auto& session : sessions)
        if (session != nullptr) destroy_session(*session);
    for (auto& pipeline : pipelines_) destroy_pipeline(pipeline);
    if (command_pool_ != VK_NULL_HANDLE)
        vkDestroyCommandPool(device_, command_pool_, nullptr);
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

void Context::destroy_session(Session& session) noexcept {
    if (device_ == VK_NULL_HANDLE) return;
    for (const auto pool : session.pools)
        if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, pool, nullptr);
    session.pools.clear();
    session.pool_capacities.clear();
    if (command_pool_ != VK_NULL_HANDLE)
        vkFreeCommandBuffers(device_, command_pool_, 1, &session.command);
    if (session.command != VK_NULL_HANDLE)
        vkFreeCommandBuffers(device_, command_pool_, 1, &session.command);
    if (session.fence != VK_NULL_HANDLE)
        vkDestroyFence(device_, session.fence, nullptr);
    session.command = VK_NULL_HANDLE;
    session.fence = VK_NULL_HANDLE;
}

VkPhysicalDeviceMemoryProperties Context::memory_properties() const {
    return memory_properties_;
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

const Context::Pipeline& Context::pipeline(Shader shader) {
    auto& entry = pipelines_[static_cast<std::size_t>(shader)];
    if (entry.handle != VK_NULL_HANDLE) return entry;

    const auto info = shader_info(shader);
    if (info.bindings > kMaxShaderBindings)
        throw std::logic_error("Too many shader bindings");

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(info.bindings);
    for (std::uint32_t index = 0; index < info.bindings; ++index) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = index;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(binding);
    }
    VkDescriptorSetLayoutCreateInfo layout_create{};
    layout_create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_create.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout_create.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(device_, &layout_create, nullptr,
                                      &entry.set_layout),
          "vkCreateDescriptorSetLayout");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = info.push_bytes;
    VkPipelineLayoutCreateInfo pipeline_layout{};
    pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout.setLayoutCount = 1;
    pipeline_layout.pSetLayouts = &entry.set_layout;
    pipeline_layout.pushConstantRangeCount = 1;
    pipeline_layout.pPushConstantRanges = &push;
    check(vkCreatePipelineLayout(device_, &pipeline_layout, nullptr, &entry.layout),
          "vkCreatePipelineLayout");

    VkShaderModuleCreateInfo module_create{};
    module_create.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_create.codeSize = info.spirv.size();
    module_create.pCode = reinterpret_cast<const std::uint32_t*>(info.spirv.data());
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device_, &module_create, nullptr, &module),
          "vkCreateShaderModule");

    VkComputePipelineCreateInfo compute{};
    compute.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compute.stage.module = module;
    compute.stage.pName = "main";
    compute.layout = entry.layout;
    check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &compute, nullptr,
                                   &entry.handle),
          "vkCreateComputePipelines");
    vkDestroyShaderModule(device_, module, nullptr);
    entry.binding_count = info.bindings;
    entry.push_bytes = info.push_bytes;
    return entry;
}

void Context::destroy_pipeline(Pipeline& entry) noexcept {
    if (entry.handle != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, entry.handle, nullptr);
    if (entry.layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, entry.layout, nullptr);
    if (entry.set_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, entry.set_layout, nullptr);
    entry = {};
}

VkDescriptorSet Context::acquire_set(const Pipeline& entry, Session& session) {
    if (session.pools.empty() || session.sets_in_active_pool >= session.pool_capacities.back())
        grow_session_pool(session);
    VkDescriptorSetAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocation.descriptorPool = session.pools.back();
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &entry.set_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    check(vkAllocateDescriptorSets(device_, &allocation, &set),
          "vkAllocateDescriptorSets");
    ++session.sets_in_active_pool;
    return set;
}

// A session allocates descriptor sets between submissions only: one image of
// the SIFT pipeline records a few hundred dispatches, and parallel extractions
// (one session per worker thread) must not share a pool or run out of it.
void Context::grow_session_pool(Session& session) {
    const std::uint32_t capacity =
        std::max<std::uint32_t>(512U, session.pool_capacities.empty()
                                          ? 0U
                                          : session.pool_capacities.back() * 2U);
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = capacity * kMaxShaderBindings;
    VkDescriptorPoolCreateInfo create{};
    create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    create.maxSets = capacity;
    create.poolSizeCount = 1;
    create.pPoolSizes = &pool_size;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    check(vkCreateDescriptorPool(device_, &create, nullptr, &pool),
          "vkCreateDescriptorPool");
    session.pools.push_back(pool);
    session.pool_capacities.push_back(capacity);
    session.sets_in_active_pool = 0;
}

// Called between submissions, when no recorded command can still reference a
// descriptor set: every pool is reset and the smaller ones are retired.
void Context::reset_session_pools(Session& session) {
    if (session.pools.empty()) return;
    const std::size_t keep = session.pool_capacities.size() - 1;
    for (std::size_t index = 0; index < session.pools.size(); ++index) {
        if (index == keep) continue;
        vkDestroyDescriptorPool(device_, session.pools[index], nullptr);
    }
    const VkDescriptorPool active = session.pools[keep];
    const std::uint32_t capacity = session.pool_capacities[keep];
    session.pools.assign(1, active);
    session.pool_capacities.assign(1, capacity);
    check(vkResetDescriptorPool(device_, active, 0), "vkResetDescriptorPool");
    session.sets_in_active_pool = 0;
}

void Context::write_set(VkDescriptorSet set, const Pipeline& entry,
                        std::span<const BufferBinding> bindings) {
    if (bindings.size() != entry.binding_count)
        throw std::logic_error("Descriptor binding count mismatch");
    std::array<VkDescriptorBufferInfo, kMaxShaderBindings> infos{};
    std::array<VkWriteDescriptorSet, kMaxShaderBindings> writes{};
    for (std::uint32_t index = 0; index < entry.binding_count; ++index) {
        infos[index].buffer = bindings[index].buffer;
        infos[index].offset = bindings[index].offset;
        infos[index].range = bindings[index].range;
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = set;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = &infos[index];
    }
    vkUpdateDescriptorSets(device_, entry.binding_count, writes.data(), 0, nullptr);
}

void Context::begin() {
    Session& session = this->session();
    std::lock_guard lock(mutex_);
    if (!session.recording) {
        if (session.command == VK_NULL_HANDLE) {
            VkCommandBufferAllocateInfo allocation{};
            allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocation.commandPool = command_pool_;
            allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocation.commandBufferCount = 1;
            check(vkAllocateCommandBuffers(device_, &allocation, &session.command),
                  "vkAllocateCommandBuffers");
        }
        if (session.fence == VK_NULL_HANDLE) {
            VkFenceCreateInfo fence_create{};
            fence_create.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            check(vkCreateFence(device_, &fence_create, nullptr, &session.fence),
                  "vkCreateFence");
        }
        check(vkResetCommandBuffer(session.command, 0), "vkResetCommandBuffer");
        reset_session_pools(session);
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(session.command, &begin_info),
              "vkBeginCommandBuffer");
        session.recording = true;
        session.sets.clear();
        session.downloads.clear();
    }
}

void Context::dispatch(Shader shader, std::span<const BufferBinding> bindings,
                       const void* push_constants, std::uint32_t push_bytes,
                       std::uint32_t groups_x, std::uint32_t groups_y,
                       std::uint32_t groups_z) {
    Session& session = this->session();
    std::lock_guard lock(mutex_);
    if (!session.recording)
        throw std::logic_error("dispatch() called without begin()");

    const Pipeline& entry = pipeline(shader);
    if (push_bytes != entry.push_bytes)
        throw std::logic_error("Push constant size mismatch for shader " +
                               std::to_string(static_cast<std::uint32_t>(shader)));
    VkDescriptorSet set = acquire_set(entry, session);
    session.sets.push_back(set);
    write_set(set, entry, bindings);

    vkCmdBindPipeline(session.command, VK_PIPELINE_BIND_POINT_COMPUTE, entry.handle);
    vkCmdBindDescriptorSets(session.command, VK_PIPELINE_BIND_POINT_COMPUTE, entry.layout, 0,
                            1, &set, 0, nullptr);
    if (push_bytes > 0)
        vkCmdPushConstants(session.command, entry.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           push_bytes, push_constants);
    vkCmdDispatch(session.command, groups_x, groups_y, groups_z);

    // One conservative barrier per dispatch keeps every pass ordered without
    // per-buffer dependency tracking; the SIFT pipeline is tens of dispatches
    // per image, so the cost is negligible next to the kernels themselves.
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(session.command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void Context::queue_download(const Buffer& source, VkDeviceSize source_offset,
                             VkDeviceSize bytes, const Buffer& destination,
                             VkDeviceSize destination_offset) {
    Session& session = this->session();
    std::lock_guard lock(mutex_);
    if (!session.recording)
        throw std::logic_error("queue_download() called without begin()");
    if (bytes == 0) return;
    const VkBufferCopy region{source_offset, destination_offset, bytes};
    vkCmdCopyBuffer(session.command, source.handle(), destination.handle(), 1, &region);
    // The conservative post-dispatch barrier covers compute->transfer, but a
    // queued download can also directly follow another download.
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(session.command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT |
                             VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
    session.downloads.push_back({&source, source_offset, bytes, &destination,
                                 destination_offset});
}

void Context::fill_u32(const Buffer& destination, VkDeviceSize offset,
                       VkDeviceSize bytes, std::uint32_t value) {
    Session& session = this->session();
    std::lock_guard lock(mutex_);
    if (!session.recording)
        throw std::logic_error("fill_u32() called without begin()");
    if (bytes == 0) return;
    vkCmdFillBuffer(session.command, destination.handle(), offset, bytes, value);
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(session.command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void Context::submit_and_wait() {
    Session& session = this->session();
    std::vector<VkDescriptorSet> sets;
    {
        std::lock_guard lock(mutex_);
        if (!session.recording)
            throw std::logic_error("submit_and_wait() called without begin()");
        check(vkEndCommandBuffer(session.command), "vkEndCommandBuffer");
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &session.command;
        check(vkResetFences(device_, 1, &session.fence), "vkResetFences");
        check(vkQueueSubmit(queue_, 1, &submit, session.fence), "vkQueueSubmit");
        session.recording = false;
        session.sets.clear();
    }

    check(vkWaitForFences(device_, 1, &session.fence, VK_TRUE, UINT64_MAX),
          "vkWaitForFences");

    // The descriptor sets stay allocated from the session's pool until the next
    // begin() resets it, which happens after this wait.
    for (const auto& download : session.downloads)
        download.destination->invalidate();
    session.downloads.clear();
}

}  // namespace photara::features::vulkan_backend
