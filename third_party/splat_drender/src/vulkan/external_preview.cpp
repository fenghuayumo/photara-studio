#if defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "splat_drender/vulkan_api.h"

#include "photara_vk/device.hpp"

#include "splat_preview_pack.hlsl.embedded.hpp"

#include <array>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace splat_drender::vulkan {
namespace {

void check(const VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            std::string("Vulkan preview ") + operation + " failed (" +
            std::to_string(static_cast<int>(result)) + ")");
    }
}

std::uint32_t memory_type(
    const VkPhysicalDevice physical, const std::uint32_t bits) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical, &properties);
    for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if ((bits & (1U << index)) != 0 &&
            (properties.memoryTypes[index].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
            return index;
        }
    }
    throw std::runtime_error("Vulkan preview image has no device-local memory type");
}

}  // namespace

struct ExternalImagePreview::Impl {
    PreviewDevice device;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    VkBuffer packed = VK_NULL_HANDLE;
    VkDeviceMemory packed_memory = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t frames = 0;

    ~Impl() {
        if (device.device == VK_NULL_HANDLE) return;
        std::unique_ptr<photara::vk::QueueLock> queue_lock;
        if (device.queue != VK_NULL_HANDLE)
            queue_lock = std::make_unique<photara::vk::QueueLock>(device.queue);
        vkDeviceWaitIdle(device.device);
        if (command_pool != VK_NULL_HANDLE)
            vkDestroyCommandPool(device.device, command_pool, nullptr);
        if (descriptor_pool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(device.device, descriptor_pool, nullptr);
        if (pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(device.device, pipeline, nullptr);
        if (pipeline_layout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device.device, pipeline_layout, nullptr);
        if (set_layout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device.device, set_layout, nullptr);
        if (packed != VK_NULL_HANDLE) vkDestroyBuffer(device.device, packed, nullptr);
        if (packed_memory != VK_NULL_HANDLE)
            vkFreeMemory(device.device, packed_memory, nullptr);
        if (image != VK_NULL_HANDLE) vkDestroyImage(device.device, image, nullptr);
        if (memory != VK_NULL_HANDLE) vkFreeMemory(device.device, memory, nullptr);
        if (timeline != VK_NULL_HANDLE)
            vkDestroySemaphore(device.device, timeline, nullptr);
    }
};

ExternalImagePreview::ExternalImagePreview(
    const PreviewDevice& device, const ExternalPreviewOptions& options)
    : impl_(std::make_unique<Impl>()) {
#if !defined(_WIN32)
    (void)device;
    (void)options;
    throw std::runtime_error(
        "Vulkan external preview currently requires Win32 handles");
#else
    if (device.device == VK_NULL_HANDLE || device.queue == VK_NULL_HANDLE ||
        !options.memory_handle || !options.semaphore_handle ||
        !options.allocation_size || !options.width || !options.height) {
        throw std::invalid_argument("Incomplete Vulkan preview options");
    }
    impl_->device = device;
    impl_->width = options.width;
    impl_->height = options.height;

    if (options.device_luid != 0) {
        VkPhysicalDeviceIDProperties identity{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &identity;
        vkGetPhysicalDeviceProperties2(device.physical_device, &properties);
        std::uint64_t luid = 0;
        static_assert(sizeof(luid) == sizeof(identity.deviceLUID));
        std::memcpy(&luid, identity.deviceLUID, sizeof(luid));
        if (!identity.deviceLUIDValid || luid != options.device_luid ||
            identity.deviceNodeMask != options.device_node_mask) {
            throw std::runtime_error(
                "The Vulkan training device does not match the editor preview image");
        }
    }

    auto close_imports = [&options]() {
        CloseHandle(reinterpret_cast<HANDLE>(
            static_cast<std::uintptr_t>(options.memory_handle)));
        CloseHandle(reinterpret_cast<HANDLE>(
            static_cast<std::uintptr_t>(options.semaphore_handle)));
    };

    VkExternalMemoryImageCreateInfo external_image{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
    external_image.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.pNext = &external_image;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_info.extent = {options.width, options.height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    try {
        check(vkCreateImage(device.device, &image_info, nullptr, &impl_->image),
              "create image");
        VkImportMemoryWin32HandleInfoKHR import_memory{
            VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
        import_memory.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        import_memory.handle = reinterpret_cast<HANDLE>(
            static_cast<std::uintptr_t>(options.memory_handle));
        VkMemoryDedicatedAllocateInfo dedicated{
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        dedicated.pNext = &import_memory;
        dedicated.image = impl_->image;
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device.device, impl_->image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.pNext = &dedicated;
        allocation.allocationSize = options.allocation_size;
        allocation.memoryTypeIndex =
            memory_type(device.physical_device, requirements.memoryTypeBits);
        check(vkAllocateMemory(device.device, &allocation, nullptr, &impl_->memory),
              "import preview image");
        check(vkBindImageMemory(device.device, impl_->image, impl_->memory, 0),
              "bind preview image");

        VkImportSemaphoreWin32HandleInfoKHR import_semaphore{
            VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
        import_semaphore.handleType =
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        import_semaphore.handle = reinterpret_cast<HANDLE>(
            static_cast<std::uintptr_t>(options.semaphore_handle));
        VkSemaphoreTypeCreateInfo timeline_type{
            VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timeline_type.initialValue = 0;
        VkSemaphoreCreateInfo semaphore_info{
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        semaphore_info.pNext = &timeline_type;
        check(vkCreateSemaphore(
                  device.device, &semaphore_info, nullptr, &impl_->timeline),
              "create preview semaphore");
        import_semaphore.semaphore = impl_->timeline;
        const auto import_semaphore_handle =
            reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(
                vkGetDeviceProcAddr(
                    device.device, "vkImportSemaphoreWin32HandleKHR"));
        if (!import_semaphore_handle)
            throw std::runtime_error(
                "Vulkan Win32 semaphore import function is unavailable");
        check(import_semaphore_handle(device.device, &import_semaphore),
              "import preview semaphore");
    } catch (...) {
        close_imports();
        throw;
    }
    close_imports();

    const VkDeviceSize packed_bytes =
        static_cast<VkDeviceSize>(options.width) * options.height * 4U;
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = packed_bytes;
    buffer_info.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    check(vkCreateBuffer(device.device, &buffer_info, nullptr, &impl_->packed),
          "create packed preview buffer");
    VkMemoryRequirements buffer_requirements{};
    vkGetBufferMemoryRequirements(device.device, impl_->packed, &buffer_requirements);
    VkMemoryAllocateInfo buffer_allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    buffer_allocation.allocationSize = buffer_requirements.size;
    buffer_allocation.memoryTypeIndex =
        memory_type(device.physical_device, buffer_requirements.memoryTypeBits);
    check(vkAllocateMemory(
              device.device, &buffer_allocation, nullptr, &impl_->packed_memory),
          "allocate packed preview buffer");
    check(vkBindBufferMemory(
              device.device, impl_->packed, impl_->packed_memory, 0),
          "bind packed preview buffer");

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    for (std::uint32_t index = 0; index < bindings.size(); ++index) {
        bindings[index].binding = index;
        bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[index].descriptorCount = 1;
        bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    set_info.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(
              device.device, &set_info, nullptr, &impl_->set_layout),
          "create preview descriptor layout");
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = sizeof(std::uint32_t) * 4;
    VkPipelineLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &impl_->set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    check(vkCreatePipelineLayout(
              device.device, &layout_info, nullptr, &impl_->pipeline_layout),
          "create preview pipeline layout");

    const std::span<const unsigned char> spir_v{splat_preview_pack_hlsl_spv};
    VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader_info.codeSize = spir_v.size();
    shader_info.pCode = reinterpret_cast<const std::uint32_t*>(spir_v.data());
    VkShaderModule shader = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device.device, &shader_info, nullptr, &shader),
          "create preview shader");
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = stage;
    pipeline_info.layout = impl_->pipeline_layout;
    const VkResult pipeline_result = vkCreateComputePipelines(
        device.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &impl_->pipeline);
    vkDestroyShaderModule(device.device, shader, nullptr);
    check(pipeline_result, "create preview pipeline");

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    check(vkCreateDescriptorPool(
              device.device, &pool_info, nullptr, &impl_->descriptor_pool),
          "create preview descriptor pool");
    VkDescriptorSetAllocateInfo allocate_set{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate_set.descriptorPool = impl_->descriptor_pool;
    allocate_set.descriptorSetCount = 1;
    allocate_set.pSetLayouts = &impl_->set_layout;
    check(vkAllocateDescriptorSets(
              device.device, &allocate_set, &impl_->descriptor),
          "allocate preview descriptor");

    VkCommandPoolCreateInfo command_pool{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool.queueFamilyIndex = device.queue_family;
    check(vkCreateCommandPool(
              device.device, &command_pool, nullptr, &impl_->command_pool),
          "create preview command pool");
    VkCommandBufferAllocateInfo allocate_command{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate_command.commandPool = impl_->command_pool;
    allocate_command.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_command.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(
              device.device, &allocate_command, &impl_->command),
          "allocate preview command");
#endif
}

ExternalImagePreview::~ExternalImagePreview() = default;

void ExternalImagePreview::submit(
    const SplatBufferView& color, const std::uint32_t source_width,
    const std::uint32_t source_height) {
    if (!impl_ || color.buffer == VK_NULL_HANDLE || source_width == 0 ||
        source_height == 0) {
        throw std::invalid_argument("Vulkan preview requires a device RGB buffer");
    }
    // The rasterizer submitted the color write without waiting. Hold the queue
    // gate from that drain through the copy so another submit cannot land
    // between them. The caller's tensor is released only after the second wait.
    photara::vk::QueueLock lock(impl_->device.queue);
    check(vkQueueWaitIdle(impl_->device.queue), "wait for the preview source");

    VkDescriptorBufferInfo source{};
    source.buffer = color.buffer;
    source.offset = color.offset;
    source.range = color.bytes == 0 ? VK_WHOLE_SIZE : color.bytes;
    VkDescriptorBufferInfo destination{};
    destination.buffer = impl_->packed;
    destination.offset = 0;
    destination.range = static_cast<VkDeviceSize>(impl_->width) * impl_->height * 4U;
    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = impl_->descriptor;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &source;
    writes[1] = writes[0];
    writes[1].dstBinding = 1;
    writes[1].pBufferInfo = &destination;
    vkUpdateDescriptorSets(
        impl_->device.device, static_cast<std::uint32_t>(writes.size()),
        writes.data(), 0, nullptr);

    check(vkResetCommandBuffer(impl_->command, 0), "reset preview command");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(impl_->command, &begin), "begin preview command");

    vkCmdBindPipeline(
        impl_->command, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipeline);
    vkCmdBindDescriptorSets(
        impl_->command, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipeline_layout,
        0, 1, &impl_->descriptor, 0, nullptr);
    const std::uint32_t push_values[4] = {
        source_width, source_height, impl_->width, impl_->height};
    vkCmdPushConstants(
        impl_->command, impl_->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(push_values), push_values);
    vkCmdDispatch(
        impl_->command, (impl_->width + 15U) / 16U, (impl_->height + 15U) / 16U, 1);

    VkBufferMemoryBarrier buffer_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    buffer_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    buffer_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    buffer_barrier.buffer = impl_->packed;
    buffer_barrier.size = destination.range;
    VkImageMemoryBarrier acquire{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    acquire.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    acquire.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    acquire.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    acquire.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    acquire.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    acquire.dstQueueFamilyIndex = impl_->device.queue_family;
    acquire.image = impl_->image;
    acquire.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(
        impl_->command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &buffer_barrier, 1,
        &acquire);

    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {impl_->width, impl_->height, 1};
    vkCmdCopyBufferToImage(
        impl_->command, impl_->packed, impl_->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    VkImageMemoryBarrier release = acquire;
    release.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    release.dstAccessMask = 0;
    release.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    release.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    release.srcQueueFamilyIndex = impl_->device.queue_family;
    release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    vkCmdPipelineBarrier(
        impl_->command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1,
        &release);
    check(vkEndCommandBuffer(impl_->command), "end preview command");

    const std::uint64_t frame = ++impl_->frames;
    const std::uint64_t wait_value = 2 * (frame - 1);
    const std::uint64_t signal_value = 2 * frame - 1;
    VkTimelineSemaphoreSubmitInfo timeline{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    // Gate the whole command buffer. A transfer-only wait would let the pack
    // dispatch start before the editor has finished sampling the image.
    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (frame > 1) {
        timeline.waitSemaphoreValueCount = 1;
        timeline.pWaitSemaphoreValues = &wait_value;
    }
    timeline.signalSemaphoreValueCount = 1;
    timeline.pSignalSemaphoreValues = &signal_value;
    submit.pNext = &timeline;
    if (frame > 1) {
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &impl_->timeline;
        submit.pWaitDstStageMask = &wait_stage;
    }
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &impl_->command;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &impl_->timeline;
    check(vkQueueSubmit(impl_->device.queue, 1, &submit, VK_NULL_HANDLE),
          "submit preview");
    check(vkQueueWaitIdle(impl_->device.queue), "finish preview");
}

std::uint64_t ExternalImagePreview::frame_count() const noexcept {
    return impl_ ? impl_->frames : 0;
}

}  // namespace splat_drender::vulkan
