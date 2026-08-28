#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif

#include "io/image.hpp"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifndef AETHERSCAN_CLI_PATH
#define AETHERSCAN_CLI_PATH "aetherscan"
#endif

namespace {
VkInstance instance{};
VkPhysicalDevice physical_device{};
VkDevice device{};
std::uint32_t queue_family = UINT32_MAX;
VkQueue queue{};
VkDescriptorPool descriptor_pool{};
ImGui_ImplVulkanH_Window window_data{};
bool rebuild_swapchain{};
constexpr std::uint32_t min_images = 2;
VkImage external_preview_image{};
VkImage display_preview_image{};
VkSemaphore external_preview_timeline{};
std::uint32_t external_preview_width{};
std::uint32_t external_preview_height{};
std::uint64_t external_ready_value{};
std::uint64_t external_consumed_value{};
std::uint64_t physical_device_luid{};
std::uint32_t physical_device_node_mask{};

void vk_check(VkResult result) {
    if (result < 0) {
        std::fprintf(stderr, "Vulkan error: %d\n", result);
        std::abort();
    }
}

bool has_extension(
    const ImVector<VkExtensionProperties>& properties, const char* name) {
    for (const auto& property : properties)
        if (std::strcmp(property.extensionName, name) == 0) return true;
    return false;
}

void setup_vulkan(ImVector<const char*> extensions) {
    std::uint32_t count{};
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    ImVector<VkExtensionProperties> properties;
    properties.resize(count);
    vk_check(vkEnumerateInstanceExtensionProperties(
        nullptr, &count, properties.Data));
    if (has_extension(
            properties,
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
        extensions.push_back(
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "AetherScan Editor";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &app;
    create.enabledExtensionCount = extensions.Size;
    create.ppEnabledExtensionNames = extensions.Data;
    vk_check(vkCreateInstance(&create, nullptr, &instance));

    physical_device = ImGui_ImplVulkanH_SelectPhysicalDevice(instance);
    queue_family = ImGui_ImplVulkanH_SelectQueueFamilyIndex(physical_device);
    VkPhysicalDeviceIDProperties id{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 device_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    device_properties.pNext = &id;
    vkGetPhysicalDeviceProperties2(physical_device, &device_properties);
    if (id.deviceLUIDValid) {
        static_assert(sizeof(physical_device_luid) == VK_LUID_SIZE);
        std::memcpy(
            &physical_device_luid, id.deviceLUID,
            sizeof(physical_device_luid));
        physical_device_node_mask = id.deviceNodeMask;
    }
    const float priority = 1.F;
    VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    const char* required_device_extensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME};
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    timeline.timelineSemaphore = VK_TRUE;
    device_info.pNext = &timeline;
    device_info.enabledExtensionCount = static_cast<std::uint32_t>(
        std::size(required_device_extensions));
    device_info.ppEnabledExtensionNames = required_device_extensions;
    vk_check(vkCreateDevice(physical_device, &device_info, nullptr, &device));
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    std::array<VkDescriptorPoolSize, 2> sizes{{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 128},
    }};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool.maxSets = 1024;
    pool.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    pool.pPoolSizes = sizes.data();
    vk_check(vkCreateDescriptorPool(device, &pool, nullptr, &descriptor_pool));
}

void setup_window(VkSurfaceKHR surface, int width, int height) {
    window_data.Surface = surface;
    VkBool32 supported{};
    vkGetPhysicalDeviceSurfaceSupportKHR(
        physical_device, queue_family, surface, &supported);
    if (!supported) throw std::runtime_error("Vulkan queue has no WSI support");
    const VkFormat formats[] = {
        VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    window_data.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        physical_device, surface, formats, 2,
        VK_COLORSPACE_SRGB_NONLINEAR_KHR);
    const VkPresentModeKHR present = VK_PRESENT_MODE_FIFO_KHR;
    window_data.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        physical_device, surface, &present, 1);
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        instance, physical_device, device, &window_data, queue_family,
        nullptr, width, height, min_images);
}

void render_frame(ImDrawData* draw) {
    auto& wd = window_data;
    VkSemaphore acquired =
        wd.FrameSemaphores[wd.SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore complete =
        wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore;
    VkResult result = vkAcquireNextImageKHR(
        device, wd.Swapchain, UINT64_MAX, acquired, {}, &wd.FrameIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        rebuild_swapchain = true;
    if (result == VK_ERROR_OUT_OF_DATE_KHR) return;
    if (result != VK_SUBOPTIMAL_KHR) vk_check(result);
    auto& frame = wd.Frames[wd.FrameIndex];
    vk_check(vkWaitForFences(device, 1, &frame.Fence, VK_TRUE, UINT64_MAX));
    vk_check(vkResetFences(device, 1, &frame.Fence));
    vk_check(vkResetCommandPool(device, frame.CommandPool, 0));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk_check(vkBeginCommandBuffer(frame.CommandBuffer, &begin));
    const bool copy_external_preview =
        external_preview_image && display_preview_image &&
        external_preview_timeline &&
        external_ready_value > external_consumed_value;
    if (copy_external_preview) {
        std::array<VkImageMemoryBarrier, 2> acquire{};
        acquire[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        acquire[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        acquire[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        acquire[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        acquire[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        acquire[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        acquire[0].dstQueueFamilyIndex = queue_family;
        acquire[0].image = external_preview_image;
        acquire[0].subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        acquire[1] = acquire[0];
        acquire[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        acquire[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        acquire[1].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        acquire[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        acquire[1].srcQueueFamilyIndex = queue_family;
        acquire[1].image = display_preview_image;
        vkCmdPipelineBarrier(
            frame.CommandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
            static_cast<std::uint32_t>(acquire.size()), acquire.data());
        VkImageCopy copy{};
        copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.extent = {
            external_preview_width, external_preview_height, 1};
        vkCmdCopyImage(
            frame.CommandBuffer, external_preview_image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, display_preview_image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        std::array<VkImageMemoryBarrier, 2> release{};
        release[0] = acquire[0];
        release[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        release[0].dstAccessMask = 0;
        release[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        release[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        release[0].srcQueueFamilyIndex = queue_family;
        release[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        release[1] = acquire[1];
        release[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        release[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        release[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        release[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(
            frame.CommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
            nullptr, static_cast<std::uint32_t>(release.size()),
            release.data());
    }
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = wd.RenderPass;
    pass.framebuffer = frame.Framebuffer;
    pass.renderArea.extent = {static_cast<std::uint32_t>(wd.Width),
                              static_cast<std::uint32_t>(wd.Height)};
    pass.clearValueCount = 1;
    pass.pClearValues = &wd.ClearValue;
    vkCmdBeginRenderPass(
        frame.CommandBuffer, &pass, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(draw, frame.CommandBuffer);
    vkCmdEndRenderPass(frame.CommandBuffer);
    vk_check(vkEndCommandBuffer(frame.CommandBuffer));
    const VkPipelineStageFlags wait =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    const std::array<VkSemaphore, 2> waits{{
        acquired, external_preview_timeline}};
    const std::array<VkPipelineStageFlags, 2> wait_stages{{
        wait, VK_PIPELINE_STAGE_TRANSFER_BIT}};
    const std::array<VkSemaphore, 2> signals{{
        complete, external_preview_timeline}};
    const std::array<std::uint64_t, 2> wait_values{{
        0, external_ready_value}};
    const std::array<std::uint64_t, 2> signal_values{{
        0, external_ready_value + 1}};
    VkTimelineSemaphoreSubmitInfo timeline{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    if (copy_external_preview) {
        timeline.waitSemaphoreValueCount = 2;
        timeline.pWaitSemaphoreValues = wait_values.data();
        timeline.signalSemaphoreValueCount = 2;
        timeline.pSignalSemaphoreValues = signal_values.data();
        submit.pNext = &timeline;
    }
    submit.waitSemaphoreCount = copy_external_preview ? 2U : 1U;
    submit.pWaitSemaphores = waits.data();
    submit.pWaitDstStageMask = wait_stages.data();
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &frame.CommandBuffer;
    submit.signalSemaphoreCount = copy_external_preview ? 2U : 1U;
    submit.pSignalSemaphores = signals.data();
    vk_check(vkQueueSubmit(queue, 1, &submit, frame.Fence));
    if (copy_external_preview)
        external_consumed_value = external_ready_value;

    VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &complete;
    info.swapchainCount = 1;
    info.pSwapchains = &wd.Swapchain;
    info.pImageIndices = &wd.FrameIndex;
    result = vkQueuePresentKHR(queue, &info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        rebuild_swapchain = true;
    else
        vk_check(result);
    wd.SemaphoreIndex = (wd.SemaphoreIndex + 1) % wd.SemaphoreCount;
}

std::uint32_t memory_type(std::uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i)
        if ((bits & (1U << i)) &&
            (properties.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    throw std::runtime_error("No Vulkan memory type");
}

struct PreviewTexture {
    VkImage image{};
    VkDeviceMemory memory{};
    VkImageView view{};
    VkSampler sampler{};
    VkDescriptorSet descriptor{};
    std::uint32_t width{};
    std::uint32_t height{};

    void reset() {
        if (!device) return;
        vkDeviceWaitIdle(device);
        if (descriptor) ImGui_ImplVulkan_RemoveTexture(descriptor);
        if (sampler) vkDestroySampler(device, sampler, nullptr);
        if (view) vkDestroyImageView(device, view, nullptr);
        if (image) vkDestroyImage(device, image, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
        *this = {};
    }

    void upload(const aetherscan::io::RgbImage& source) {
        reset();
        width = source.width;
        height = source.height;
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(width) * height * 4;
        std::vector<std::uint8_t> rgba(static_cast<std::size_t>(bytes));
        for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
            rgba[4 * i] = source.pixels[3 * i];
            rgba[4 * i + 1] = source.pixels[3 * i + 1];
            rgba[4 * i + 2] = source.pixels[3 * i + 2];
            rgba[4 * i + 3] = 255;
        }
        VkBuffer staging{};
        VkDeviceMemory staging_memory{};
        VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer.size = bytes;
        buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        vk_check(vkCreateBuffer(device, &buffer, nullptr, &staging));
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, staging, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memory_type(
            requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vk_check(vkAllocateMemory(device, &allocation, nullptr, &staging_memory));
        vk_check(vkBindBufferMemory(device, staging, staging_memory, 0));
        void* mapped{};
        vk_check(vkMapMemory(device, staging_memory, 0, bytes, 0, &mapped));
        std::memcpy(mapped, rgba.data(), static_cast<std::size_t>(bytes));
        vkUnmapMemory(device, staging_memory);

        VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        image_info.extent = {width, height, 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                           VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vk_check(vkCreateImage(device, &image_info, nullptr, &image));
        vkGetImageMemoryRequirements(device, image, &requirements);
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memory_type(
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vk_check(vkAllocateMemory(device, &allocation, nullptr, &memory));
        vk_check(vkBindImageMemory(device, image, memory, 0));

        auto& frame = window_data.Frames[window_data.FrameIndex];
        vk_check(vkResetCommandPool(device, frame.CommandPool, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(frame.CommandBuffer, &begin));
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(
            frame.CommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
            &barrier);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(
            frame.CommandBuffer, staging, image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(
            frame.CommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
            1, &barrier);
        vk_check(vkEndCommandBuffer(frame.CommandBuffer));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frame.CommandBuffer;
        vk_check(vkQueueSubmit(queue, 1, &submit, {}));
        vk_check(vkQueueWaitIdle(queue));
        vkDestroyBuffer(device, staging, nullptr);
        vkFreeMemory(device, staging_memory, nullptr);

        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vk_check(vkCreateImageView(device, &view_info, nullptr, &view));
        VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler_info.magFilter = VK_FILTER_LINEAR;
        sampler_info.minFilter = VK_FILTER_LINEAR;
        sampler_info.addressModeU = sampler_info.addressModeV =
            sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.maxLod = 1.F;
        vk_check(vkCreateSampler(device, &sampler_info, nullptr, &sampler));
        descriptor = ImGui_ImplVulkan_AddTexture(
            sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
};

struct ExternalPreview {
    PreviewTexture display;
    VkImage image{};
    VkDeviceMemory memory{};
    VkSemaphore timeline{};
    HANDLE memory_handle{};
    HANDLE semaphore_handle{};
    VkDeviceSize allocation_size{};
    std::uint32_t width{};
    std::uint32_t height{};

    void create(const std::uint32_t image_width,
                const std::uint32_t image_height) {
        reset();
        width = image_width;
        height = image_height;
        aetherscan::io::RgbImage black;
        black.width = width;
        black.height = height;
        black.pixels.assign(
            static_cast<std::size_t>(width) * height * 3, 10);
        display.upload(black);
        display_preview_image = display.image;

        VkExternalMemoryImageCreateInfo external_image{
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
        external_image.handleTypes =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image_info.pNext = &external_image;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        image_info.extent = {width, height, 1};
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                           VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        vk_check(vkCreateImage(device, &image_info, nullptr, &image));
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, image, &requirements);
        allocation_size = requirements.size;

        SECURITY_ATTRIBUTES security{
            sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        VkExportMemoryWin32HandleInfoKHR win32_export{
            VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
        win32_export.pAttributes = &security;
        win32_export.dwAccess = GENERIC_ALL;
        VkExportMemoryAllocateInfo export_info{
            VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        export_info.pNext = &win32_export;
        export_info.handleTypes =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        VkMemoryDedicatedAllocateInfo dedicated{
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        dedicated.pNext = &export_info;
        dedicated.image = image;
        VkMemoryAllocateInfo allocation{
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.pNext = &dedicated;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memory_type(
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vk_check(vkAllocateMemory(device, &allocation, nullptr, &memory));
        vk_check(vkBindImageMemory(device, image, memory, 0));

        VkExportSemaphoreWin32HandleInfoKHR semaphore_win32{
            VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
        semaphore_win32.pAttributes = &security;
        semaphore_win32.dwAccess = GENERIC_ALL;
        VkExportSemaphoreCreateInfo semaphore_export{
            VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        semaphore_export.pNext = &semaphore_win32;
        semaphore_export.handleTypes =
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        VkSemaphoreTypeCreateInfo timeline_type{
            VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        timeline_type.pNext = &semaphore_export;
        timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timeline_type.initialValue = 0;
        VkSemaphoreCreateInfo semaphore_info{
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        semaphore_info.pNext = &timeline_type;
        vk_check(vkCreateSemaphore(
            device, &semaphore_info, nullptr, &timeline));

        renew_export_handles();

        auto& frame = window_data.Frames[window_data.FrameIndex];
        vk_check(vkResetCommandPool(device, frame.CommandPool, 0));
        VkCommandBufferBeginInfo begin{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(frame.CommandBuffer, &begin));
        VkImageMemoryBarrier release{
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        release.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        release.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        release.srcQueueFamilyIndex = queue_family;
        release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        release.image = image;
        release.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(
            frame.CommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr,
            1, &release);
        vk_check(vkEndCommandBuffer(frame.CommandBuffer));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frame.CommandBuffer;
        vk_check(vkQueueSubmit(queue, 1, &submit, {}));
        vk_check(vkQueueWaitIdle(queue));

        external_preview_image = image;
        external_preview_timeline = timeline;
        external_preview_width = width;
        external_preview_height = height;
        external_ready_value = 0;
        external_consumed_value = 0;
    }

    void renew_export_handles() {
        close_export_handles();
        const auto get_memory = reinterpret_cast<
            PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(
                device, "vkGetMemoryWin32HandleKHR"));
        const auto get_semaphore = reinterpret_cast<
            PFN_vkGetSemaphoreWin32HandleKHR>(vkGetDeviceProcAddr(
                device, "vkGetSemaphoreWin32HandleKHR"));
        if (!get_memory || !get_semaphore)
            throw std::runtime_error("Vulkan Win32 export functions missing");
        VkMemoryGetWin32HandleInfoKHR memory_info{
            VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
        memory_info.memory = memory;
        memory_info.handleType =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        vk_check(get_memory(device, &memory_info, &memory_handle));
        VkSemaphoreGetWin32HandleInfoKHR timeline_info{
            VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
        timeline_info.semaphore = timeline;
        timeline_info.handleType =
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        vk_check(get_semaphore(
            device, &timeline_info, &semaphore_handle));
    }

    void poll() {
        if (!timeline) return;
        std::uint64_t value{};
        if (vkGetSemaphoreCounterValue(device, timeline, &value) == VK_SUCCESS &&
            (value & 1U) != 0 && value > external_consumed_value)
            external_ready_value = value;
    }

    // Keep training non-blocking while the swapchain is minimized. CUDA's
    // next preview waits for the even release value, so consume/copy without
    // presenting when no WSI frame can be acquired.
    void consume_without_present() {
        if (external_ready_value <= external_consumed_value) return;
        vk_check(vkQueueWaitIdle(queue));
        auto& frame = window_data.Frames[window_data.FrameIndex];
        vk_check(vkResetCommandPool(device, frame.CommandPool, 0));
        VkCommandBufferBeginInfo begin{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vk_check(vkBeginCommandBuffer(frame.CommandBuffer, &begin));
        std::array<VkImageMemoryBarrier, 2> acquire{};
        acquire[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        acquire[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        acquire[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        acquire[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        acquire[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        acquire[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        acquire[0].dstQueueFamilyIndex = queue_family;
        acquire[0].image = image;
        acquire[0].subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        acquire[1] = acquire[0];
        acquire[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        acquire[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        acquire[1].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        acquire[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        acquire[1].srcQueueFamilyIndex = queue_family;
        acquire[1].image = display.image;
        vkCmdPipelineBarrier(
            frame.CommandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
            static_cast<std::uint32_t>(acquire.size()), acquire.data());
        VkImageCopy copy{};
        copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.extent = {width, height, 1};
        vkCmdCopyImage(
            frame.CommandBuffer, image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, display.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        std::array<VkImageMemoryBarrier, 2> release{};
        release[0] = acquire[0];
        release[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        release[0].dstAccessMask = 0;
        release[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        release[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        release[0].srcQueueFamilyIndex = queue_family;
        release[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        release[1] = acquire[1];
        release[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        release[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        release[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        release[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(
            frame.CommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
            nullptr, static_cast<std::uint32_t>(release.size()),
            release.data());
        vk_check(vkEndCommandBuffer(frame.CommandBuffer));
        const VkPipelineStageFlags wait_stage =
            VK_PIPELINE_STAGE_TRANSFER_BIT;
        const std::uint64_t signal_value = external_ready_value + 1;
        VkTimelineSemaphoreSubmitInfo timeline_info{
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        timeline_info.waitSemaphoreValueCount = 1;
        timeline_info.pWaitSemaphoreValues = &external_ready_value;
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &signal_value;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.pNext = &timeline_info;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &timeline;
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frame.CommandBuffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &timeline;
        vk_check(vkQueueSubmit(queue, 1, &submit, {}));
        vk_check(vkQueueWaitIdle(queue));
        external_consumed_value = external_ready_value;
    }

    void close_export_handles() {
        if (memory_handle) CloseHandle(std::exchange(memory_handle, nullptr));
        if (semaphore_handle)
            CloseHandle(std::exchange(semaphore_handle, nullptr));
    }

    void reset() {
        if (!device) return;
        vkDeviceWaitIdle(device);
        external_preview_image = {};
        display_preview_image = {};
        external_preview_timeline = {};
        external_ready_value = external_consumed_value = 0;
        close_export_handles();
        if (timeline) vkDestroySemaphore(device, timeline, nullptr);
        if (image) vkDestroyImage(device, image, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
        timeline = {};
        image = {};
        memory = {};
        allocation_size = 0;
        width = height = 0;
        display.reset();
    }
};

std::string quote(const std::filesystem::path& path) {
    return '"' + path.string() + '"';
}

struct Job {
    std::atomic_bool running{};
    std::atomic_int exit_code{-1};
    std::filesystem::path log;
#if defined(_WIN32)
    HANDLE process{};
#endif

    void start(const std::string& command, const std::filesystem::path& log_path) {
        if (running.exchange(true)) return;
        exit_code = -1;
        log = log_path;
#if defined(_WIN32)
        SECURITY_ATTRIBUTES security{
            sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        HANDLE log_handle = CreateFileW(
            log.c_str(), GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE, &security, CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL, nullptr);
        if (log_handle == INVALID_HANDLE_VALUE) {
            running = false;
            throw std::runtime_error("Cannot create reconstruction log");
        }
        const int count = MultiByteToWideChar(
            CP_UTF8, 0, command.c_str(), -1, nullptr, 0);
        std::vector<wchar_t> mutable_command(
            static_cast<std::size_t>(count));
        MultiByteToWideChar(
            CP_UTF8, 0, command.c_str(), -1, mutable_command.data(), count);
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdOutput = log_handle;
        startup.hStdError = log_handle;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        PROCESS_INFORMATION info{};
        const BOOL created = CreateProcessW(
            nullptr, mutable_command.data(), nullptr, nullptr, TRUE,
            CREATE_NO_WINDOW, nullptr, nullptr, &startup, &info);
        CloseHandle(log_handle);
        if (!created) {
            running = false;
            throw std::runtime_error(
                "CreateProcess failed: " + std::to_string(GetLastError()));
        }
        process = info.hProcess;
        CloseHandle(info.hThread);
#else
        running = false;
        throw std::runtime_error(
            "External-memory editor launch is currently Win32-only");
#endif
    }

    void poll() {
#if defined(_WIN32)
        if (!running || !process) return;
        DWORD code = STILL_ACTIVE;
        if (GetExitCodeProcess(process, &code) && code != STILL_ACTIVE) {
            exit_code = static_cast<int>(code);
            running = false;
            CloseHandle(std::exchange(process, nullptr));
        }
#endif
    }

    void stop() {
#if defined(_WIN32)
        if (process) {
            TerminateProcess(process, 2);
            WaitForSingleObject(process, INFINITE);
            CloseHandle(std::exchange(process, nullptr));
        }
#endif
        running = false;
    }
};

std::filesystem::path newest_preview(const std::filesystem::path& directory) {
    std::filesystem::path newest;
    std::filesystem::file_time_type time{};
    std::error_code error;
    if (!std::filesystem::exists(directory, error)) return {};
    for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".png") continue;
        const auto candidate = entry.last_write_time(error);
        if (newest.empty() || candidate > time) {
            newest = entry.path();
            time = candidate;
        }
    }
    return newest;
}

std::string tail(const std::filesystem::path& path, std::size_t max_bytes) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return "Waiting for reconstruction output...";
    const auto end = input.tellg();
    const auto begin = std::max<std::streamoff>(
        0, static_cast<std::streamoff>(end) -
               static_cast<std::streamoff>(max_bytes));
    input.seekg(begin);
    return {std::istreambuf_iterator<char>(input), {}};
}

void apply_editor_style() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding = {0, 0};
    style.FramePadding = {9, 6};
    style.CellPadding = {8, 5};
    style.ItemSpacing = {8, 7};
    style.ItemInnerSpacing = {6, 5};
    style.ScrollbarSize = 11.F;
    style.GrabMinSize = 8.F;
    style.WindowBorderSize = 0.F;
    style.ChildBorderSize = 1.F;
    style.PopupBorderSize = 1.F;
    style.FrameBorderSize = 1.F;
    style.WindowRounding = 0.F;
    style.ChildRounding = 0.F;
    style.FrameRounding = 3.F;
    style.PopupRounding = 4.F;
    style.ScrollbarRounding = 8.F;
    style.GrabRounding = 3.F;
    style.TabRounding = 3.F;
    auto& c = style.Colors;
    c[ImGuiCol_Text] = ImVec4(0.88F, 0.88F, 0.89F, 1.F);
    c[ImGuiCol_TextDisabled] = ImVec4(0.43F, 0.43F, 0.47F, 1.F);
    c[ImGuiCol_WindowBg] = ImVec4(0.055F, 0.055F, 0.06F, 1.F);
    c[ImGuiCol_ChildBg] = ImVec4(0.075F, 0.075F, 0.08F, 1.F);
    c[ImGuiCol_PopupBg] = ImVec4(0.12F, 0.12F, 0.13F, 0.98F);
    c[ImGuiCol_Border] = ImVec4(0.20F, 0.20F, 0.22F, 1.F);
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = ImVec4(0.105F, 0.105F, 0.115F, 1.F);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.16F, 0.16F, 0.18F, 1.F);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.19F, 0.19F, 0.21F, 1.F);
    c[ImGuiCol_TitleBg] = ImVec4(0.10F, 0.10F, 0.11F, 1.F);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.12F, 0.12F, 0.13F, 1.F);
    c[ImGuiCol_MenuBarBg] = ImVec4(0.13F, 0.13F, 0.14F, 1.F);
    c[ImGuiCol_ScrollbarBg] = ImVec4(0.07F, 0.07F, 0.075F, 1.F);
    c[ImGuiCol_ScrollbarGrab] = ImVec4(0.25F, 0.25F, 0.27F, 1.F);
    c[ImGuiCol_CheckMark] = ImVec4(0.31F, 0.76F, 0.97F, 1.F);
    c[ImGuiCol_SliderGrab] = ImVec4(0.18F, 0.55F, 0.84F, 1.F);
    c[ImGuiCol_Button] = ImVec4(0.13F, 0.13F, 0.145F, 1.F);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.20F, 0.20F, 0.22F, 1.F);
    c[ImGuiCol_ButtonActive] = ImVec4(0.035F, 0.28F, 0.44F, 1.F);
    c[ImGuiCol_Header] = ImVec4(0.035F, 0.28F, 0.44F, 1.F);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.12F, 0.30F, 0.43F, 1.F);
    c[ImGuiCol_HeaderActive] = ImVec4(0.05F, 0.37F, 0.58F, 1.F);
    c[ImGuiCol_Separator] = ImVec4(0.20F, 0.20F, 0.22F, 1.F);
    c[ImGuiCol_SeparatorHovered] = ImVec4(0.0F, 0.47F, 0.83F, 1.F);
    c[ImGuiCol_ResizeGrip] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ResizeGripHovered] = ImVec4(0.0F, 0.47F, 0.83F, 0.7F);
    c[ImGuiCol_Tab] = ImVec4(0.11F, 0.11F, 0.12F, 1.F);
    c[ImGuiCol_TabHovered] = ImVec4(0.17F, 0.17F, 0.19F, 1.F);
    c[ImGuiCol_TabSelected] = ImVec4(0.15F, 0.15F, 0.165F, 1.F);
    c[ImGuiCol_TabDimmed] = ImVec4(0.08F, 0.08F, 0.09F, 1.F);
}

void panel_title(const char* title, const char* suffix = nullptr) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.15F, 0.15F, 0.165F, 1.F));
    ImGui::BeginChild(
        (std::string("##title_") + title).c_str(), {0, 34.F}, false,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::SetCursorPos({12.F, 9.F});
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.66F, 0.67F, 0.71F, 1.F));
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    if (suffix) {
        const float width = ImGui::CalcTextSize(suffix).x;
        ImGui::SameLine(ImGui::GetWindowWidth() - width - 12.F);
        ImGui::TextDisabled("%s", suffix);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void status_dot(bool active) {
    const ImVec2 p = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddCircleFilled(
        {p.x + 5.F, p.y + 8.F}, 4.F,
        ImGui::ColorConvertFloat4ToU32(
            active ? ImVec4(0.28F, 0.80F, 0.47F, 1.F)
                   : ImVec4(0.38F, 0.39F, 0.42F, 1.F)));
    ImGui::Dummy({14.F, 16.F});
}

std::pair<int, int> parse_iteration(const std::string& log) {
    const std::string marker = "splat iteration=";
    const std::size_t found = log.rfind(marker);
    if (found == std::string::npos) return {0, 0};
    const char* begin = log.c_str() + found + marker.size();
    char* end{};
    const long current = std::strtol(begin, &end, 10);
    if (!end || *end != '/') return {0, 0};
    const long total = std::strtol(end + 1, nullptr, 10);
    return {static_cast<int>(current), static_cast<int>(total)};
}
}  // namespace

int main(const int argc, char** argv) {
    const bool interop_smoke =
        argc > 1 && std::string_view(argv[1]) == "--interop-smoke";
    glfwSetErrorCallback([](int code, const char* text) {
        std::fprintf(stderr, "GLFW %d: %s\n", code, text);
    });
    if (!glfwInit() || !glfwVulkanSupported()) return 1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(
        1500, 900, "AetherScan Reconstruction Editor", nullptr, nullptr);
    ImVector<const char*> extensions;
    std::uint32_t extension_count{};
    const char** required =
        glfwGetRequiredInstanceExtensions(&extension_count);
    for (std::uint32_t i = 0; i < extension_count; ++i)
        extensions.push_back(required[i]);
    setup_vulkan(extensions);
    VkSurfaceKHR surface{};
    vk_check(glfwCreateWindowSurface(instance, window, nullptr, &surface));
    int width{}, height{};
    glfwGetFramebufferSize(window, &width, &height);
    setup_window(surface, width, height);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    const char* preferred_font = "C:\\Windows\\Fonts\\msyh.ttc";
    if (!std::filesystem::exists(preferred_font))
        preferred_font = "C:\\Windows\\Fonts\\segoeui.ttf";
    io.Fonts->AddFontFromFileTTF(
        preferred_font, 15.F, nullptr, io.Fonts->GetGlyphRangesChineseFull());
    apply_editor_style();
    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init{};
    init.Instance = instance;
    init.PhysicalDevice = physical_device;
    init.Device = device;
    init.QueueFamily = queue_family;
    init.Queue = queue;
    init.DescriptorPool = descriptor_pool;
    init.RenderPass = window_data.RenderPass;
    init.MinImageCount = min_images;
    init.ImageCount = window_data.ImageCount;
    init.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init.CheckVkResultFn = vk_check;
    ImGui_ImplVulkan_Init(&init);

    std::array<char, 1024> images{"D:\\ScanVideo\\ori_img\\images"};
    std::array<char, 1024> output{"D:\\ScanVideo\\ori_img\\aetherscan_gui\\object.ply"};
    if (interop_smoke)
        std::snprintf(
            output.data(), output.size(),
            "D:\\ProgramCode\\C++\\3dgs\\AetherScan\\artifacts\\cuda_vulkan_smoke\\scene.ply");
    int iterations = interop_smoke ? 100 : 30'000;
    int preview_interval = 50;
    bool scene_mode = false;
    Job job;
    ExternalPreview preview;
    preview.create(1920, 1920);
    bool smoke_started = false;
    bool smoke_success = false;
    const auto smoke_begin = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        job.poll();
        preview.poll();
        if (interop_smoke && smoke_started && !job.running &&
            external_consumed_value >= 3) {
            smoke_success = job.exit_code == 0;
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        } else if (interop_smoke &&
                   std::chrono::steady_clock::now() - smoke_begin >
                       std::chrono::seconds(60)) {
            job.stop();
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        glfwGetFramebufferSize(window, &width, &height);
        if (width > 0 && height > 0 &&
            (rebuild_swapchain || window_data.Width != width ||
             window_data.Height != height)) {
            ImGui_ImplVulkan_SetMinImageCount(min_images);
            ImGui_ImplVulkanH_CreateOrResizeWindow(
                instance, physical_device, device, &window_data, queue_family,
                nullptr, width, height, min_images);
            window_data.FrameIndex = 0;
            rebuild_swapchain = false;
        }
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
            preview.consume_without_present();
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin(
            "AetherScan", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse);

        iterations = std::max(iterations, 1);
        preview_interval = std::max(preview_interval, 1);
        const std::filesystem::path output_path(output.data());
        const auto job_dir = output_path.parent_path();
        const auto log_path = job_dir / "editor_reconstruction.log";
        const std::string log = tail(log_path, 32 * 1024);
        const auto [current_iteration, total_iterations] = parse_iteration(log);
        const float progress = total_iterations > 0
            ? std::clamp(
                  static_cast<float>(current_iteration) / total_iterations,
                  0.F, 1.F)
            : 0.F;
        bool start_requested = interop_smoke && !smoke_started;

        // Menubar: deliberately restrained, matching the reference editor.
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.095F, 0.095F, 0.105F, 1.F));
        ImGui::BeginChild("##menubar", {0, 31.F}, false,
                          ImGuiWindowFlags_NoScrollbar);
        ImGui::SetCursorPos({12.F, 6.F});
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.31F, 0.76F, 0.97F, 1.F));
        ImGui::TextUnformatted("AETHER");
        ImGui::PopStyleColor();
        ImGui::SameLine(67.F);
        ImGui::TextUnformatted("SCAN");
        ImGui::SameLine(116.F);
        ImGui::TextDisabled("File");
        ImGui::SameLine(); ImGui::TextDisabled("Edit");
        ImGui::SameLine(); ImGui::TextDisabled("View");
        ImGui::SameLine(); ImGui::TextDisabled("Reconstruction");
        ImGui::SameLine(); ImGui::TextDisabled("Help");
        const std::string project_name = output_path.stem().empty()
            ? "Untitled Project" : output_path.stem().string();
        const float project_width = ImGui::CalcTextSize(project_name.c_str()).x;
        ImGui::SameLine(ImGui::GetWindowWidth() - project_width - 15.F);
        ImGui::TextDisabled("%s", project_name.c_str());
        ImGui::EndChild();
        ImGui::PopStyleColor();

        // Main toolbar.
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.145F, 0.145F, 0.155F, 1.F));
        ImGui::BeginChild("##toolbar", {0, 47.F}, false,
                          ImGuiWindowFlags_NoScrollbar);
        ImGui::SetCursorPos({10.F, 8.F});
        ImGui::Button("+  Add Images", {112.F, 31.F});
        ImGui::SameLine();
        ImGui::Button("Align Photos", {112.F, 31.F});
        ImGui::SameLine();
        ImGui::PushStyleColor(
            ImGuiCol_Button,
            job.running ? ImVec4(0.20F, 0.20F, 0.22F, 1.F)
                        : ImVec4(0.02F, 0.38F, 0.62F, 1.F));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              ImVec4(0.03F, 0.47F, 0.75F, 1.F));
        if (ImGui::Button(
                job.running ? "Training..." : "Train 3DGS",
                {112.F, 31.F}) && !job.running && !interop_smoke)
            start_requested = true;
        ImGui::PopStyleColor(2);
        ImGui::SameLine();
        ImGui::Button("Build Mesh", {104.F, 31.F});
        ImGui::SameLine();
        ImGui::Button("Export", {82.F, 31.F});
        if (job.running) {
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.96F, 0.38F, 0.35F, 1.F));
            if (ImGui::Button("Stop", {70.F, 31.F})) job.stop();
            ImGui::PopStyleColor();
        }
        const char* gpu_text = "CUDA / Vulkan  |  External Memory";
        const float gpu_width = ImGui::CalcTextSize(gpu_text).x;
        ImGui::SameLine(ImGui::GetWindowWidth() - gpu_width - 16.F);
        ImGui::SetCursorPosY(15.F);
        ImGui::TextDisabled("%s", gpu_text);
        ImGui::EndChild();
        ImGui::PopStyleColor();

        const float status_height = 25.F;
        const float content_height = ImGui::GetContentRegionAvail().y - status_height;
        const float left_width = 255.F;
        const float right_width = 310.F;
        const float console_height = 190.F;

        ImGui::BeginChild("##workspace", {0, content_height}, false,
                          ImGuiWindowFlags_NoScrollbar);

        // Left: scene hierarchy and assets.
        ImGui::BeginChild("ScenePanel", {left_width, 0}, true);
        panel_title("SCENE");
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {7.F, 8.F});
        ImGui::SetCursorPosX(11.F);
        if (ImGui::TreeNodeEx(
                "Capture 01", ImGuiTreeNodeFlags_DefaultOpen |
                                  ImGuiTreeNodeFlags_SpanAvailWidth)) {
            ImGui::TreeNodeEx(
                "Input Images", ImGuiTreeNodeFlags_Leaf |
                                    ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                    ImGuiTreeNodeFlags_SpanAvailWidth);
            ImGui::TreeNodeEx(
                "Camera Poses", ImGuiTreeNodeFlags_Leaf |
                                    ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                    ImGuiTreeNodeFlags_SpanAvailWidth);
            ImGui::TreeNodeEx(
                "Sparse Point Cloud", ImGuiTreeNodeFlags_Leaf |
                                         ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                         ImGuiTreeNodeFlags_SpanAvailWidth);
            ImGui::TreeNodeEx(
                "Gaussian Model", ImGuiTreeNodeFlags_Leaf |
                                      ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                      ImGuiTreeNodeFlags_SpanAvailWidth |
                                      ImGuiTreeNodeFlags_Selected);
            ImGui::TreeNodeEx(
                "Reconstructed Mesh", ImGuiTreeNodeFlags_Leaf |
                                         ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                         ImGuiTreeNodeFlags_SpanAvailWidth);
            ImGui::TreePop();
        }
        ImGui::PopStyleVar();
        ImGui::Dummy({0, 6.F});
        panel_title("PIPELINE");
        const std::array<std::pair<const char*, const char*>, 4> stages{{
            {"01", "Align cameras"}, {"02", "Sparse reconstruction"},
            {"03", "ADCPlus training"}, {"04", "Mesh extraction"}}};
        for (std::size_t i = 0; i < stages.size(); ++i) {
            const bool active = job.running &&
                ((i < 2 && log.find("sfm") != std::string::npos) ||
                 (i == 2 && log.find("splat training") != std::string::npos) ||
                 (i == 3 && log.find("mesh") != std::string::npos));
            ImGui::SetCursorPosX(13.F);
            status_dot(active);
            ImGui::SameLine();
            ImGui::TextDisabled("%s", stages[i].first);
            ImGui::SameLine();
            ImGui::TextUnformatted(stages[i].second);
        }
        ImGui::Dummy({0, 8.F});
        panel_title("ASSETS");
        ImGui::SetCursorPosX(13.F);
        ImGui::TextDisabled("IMAGES");
        ImGui::Spacing();
        ImGui::SetCursorPosX(13.F);
        ImGui::TextWrapped("%s", images.data());
        ImGui::EndChild();

        ImGui::SameLine(0, 3.F);

        // Centre: viewport plus console.
        const float center_width = ImGui::GetContentRegionAvail().x - right_width - 3.F;
        ImGui::BeginChild("CenterStack", {center_width, 0}, false,
                          ImGuiWindowFlags_NoScrollbar);
        const float viewport_height = ImGui::GetContentRegionAvail().y - console_height - 3.F;
        ImGui::BeginChild("Viewport", {0, viewport_height}, true,
                          ImGuiWindowFlags_NoScrollbar);
        panel_title("SCENE VIEW", job.running ? "LIVE" : "READY");
        const ImVec2 view_origin = ImGui::GetCursorScreenPos();
        const ImVec2 available = ImGui::GetContentRegionAvail();
        ImGui::GetWindowDrawList()->AddRectFilled(
            view_origin, {view_origin.x + available.x, view_origin.y + available.y},
            IM_COL32(8, 9, 12, 255));
        const bool has_rendered_preview =
            preview.display.descriptor && (job.running || current_iteration > 0);
        if (!has_rendered_preview) {
            ImDrawList* draw = ImGui::GetWindowDrawList();
            const ImVec2 view_max{view_origin.x + available.x,
                                  view_origin.y + available.y};
            const float horizon_y = view_origin.y + available.y * 0.43F;
            const ImVec2 vanishing{view_origin.x + available.x * 0.5F,
                                   horizon_y};
            draw->PushClipRect(view_origin, view_max, true);

            // Subtle perspective construction grid for the empty scene.
            draw->AddLine({view_origin.x, horizon_y},
                          {view_max.x, horizon_y},
                          IM_COL32(32, 36, 43, 255));
            constexpr int ray_count = 18;
            for (int i = -ray_count; i <= ray_count; ++i) {
                const float x = vanishing.x +
                                i * available.x / static_cast<float>(ray_count);
                const ImU32 colour = i == 0 ? IM_COL32(58, 67, 78, 210)
                                            : IM_COL32(34, 38, 45, 190);
                draw->AddLine(vanishing, {x, view_max.y}, colour);
            }
            for (int i = 0; i < 18; ++i) {
                const float t = static_cast<float>(i) / 17.F;
                const float eased = t * t;
                const float y = horizon_y + eased * (view_max.y - horizon_y);
                draw->AddLine({view_origin.x, y}, {view_max.x, y},
                              IM_COL32(34, 38, 45, 190));
            }

            // World origin and compact orientation gizmo.
            draw->AddLine({vanishing.x, view_max.y}, vanishing,
                          IM_COL32(55, 126, 184, 230), 1.5F);
            draw->AddCircleFilled(vanishing, 3.F,
                                  IM_COL32(78, 170, 227, 255));
            const ImVec2 gizmo{view_max.x - 54.F, view_origin.y + 50.F};
            draw->AddCircleFilled(gizmo, 23.F, IM_COL32(18, 20, 25, 220));
            draw->AddCircle(gizmo, 23.F, IM_COL32(55, 59, 68, 255));
            draw->AddLine(gizmo, {gizmo.x + 16.F, gizmo.y},
                          IM_COL32(226, 82, 82, 255), 2.F);
            draw->AddLine(gizmo, {gizmo.x, gizmo.y - 16.F},
                          IM_COL32(86, 202, 121, 255), 2.F);
            draw->AddLine(gizmo, {gizmo.x - 10.F, gizmo.y + 11.F},
                          IM_COL32(79, 154, 235, 255), 2.F);
            draw->AddText({gizmo.x + 18.F, gizmo.y - 7.F},
                          IM_COL32(226, 82, 82, 255), "X");
            draw->AddText({gizmo.x - 4.F, gizmo.y - 31.F},
                          IM_COL32(86, 202, 121, 255), "Y");
            draw->PopClipRect();
        } else {
            const float scale = std::min(
                available.x / preview.width, available.y / preview.height);
            const ImVec2 size{preview.width * scale, preview.height * scale};
            ImGui::SetCursorPosX(
                ImGui::GetCursorPosX() + (available.x - size.x) * 0.5F);
            ImGui::SetCursorPosY(
                ImGui::GetCursorPosY() + (available.y - size.y) * 0.5F);
            ImGui::Image(
                reinterpret_cast<ImTextureID>(preview.display.descriptor), size);
        }
        const ImVec2 overlay_min{view_origin.x + 14.F, view_origin.y + 14.F};
        const ImVec2 overlay_max{view_origin.x + 225.F, view_origin.y + 47.F};
        ImGui::GetWindowDrawList()->AddRectFilled(
            overlay_min, overlay_max, IM_COL32(20, 20, 23, 220), 4.F);
        ImGui::GetWindowDrawList()->AddCircleFilled(
            {overlay_min.x + 16.F, overlay_min.y + 16.F}, 4.F,
            job.running ? IM_COL32(69, 204, 120, 255)
                        : IM_COL32(96, 98, 104, 255));
        ImGui::GetWindowDrawList()->AddText(
            {overlay_min.x + 29.F, overlay_min.y + 8.F},
            IM_COL32(205, 207, 212, 255),
            job.running ? "LIVE TRAINING PREVIEW" : "WAITING FOR RECONSTRUCTION");
        ImGui::EndChild();

        ImGui::BeginChild("ConsolePanel", {0, 0}, true);
        panel_title("CONSOLE", job.running ? "FOLLOWING OUTPUT" : nullptr);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.045F, 0.045F, 0.05F, 1.F));
        ImGui::BeginChild("Log", {0, 0}, false,
                          ImGuiWindowFlags_HorizontalScrollbar);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.68F, 0.70F, 0.73F, 1.F));
        ImGui::TextUnformatted(log.c_str());
        ImGui::PopStyleColor();
        if (job.running) ImGui::SetScrollHereY(1.F);
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::EndChild();
        ImGui::EndChild();

        ImGui::SameLine(0, 3.F);

        // Right: inspector/settings.
        ImGui::BeginChild("Inspector", {right_width, 0}, true);
        panel_title("INSPECTOR");
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.16F, 0.16F, 0.175F, 1.F));
        if (ImGui::CollapsingHeader(
                "Project", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Spacing();
            ImGui::TextDisabled("Image source");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputText("##images", images.data(), images.size());
            ImGui::TextDisabled("Output model");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputText("##output", output.data(), output.size());
            ImGui::Spacing();
        }
        if (ImGui::CollapsingHeader(
                "Reconstruction", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Spacing();
            ImGui::TextDisabled("Capture mode");
            ImGui::SetNextItemWidth(-1.F);
            const char* modes[] = {"Object", "Scene"};
            int mode = scene_mode ? 1 : 0;
            if (ImGui::Combo("##mode", &mode, modes, 2)) scene_mode = mode == 1;
            ImGui::TextDisabled("ADCPlus iterations");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputInt("##iterations", &iterations, 1000, 5000);
            ImGui::TextDisabled("Live preview cadence");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputInt("##cadence", &preview_interval, 10, 50);
            ImGui::Spacing();
        }
        if (ImGui::CollapsingHeader(
                "Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Spacing();
            ImGui::TextDisabled("Training backend");
            ImGui::SameLine(ImGui::GetWindowWidth() - 76.F);
            ImGui::TextColored(
                ImVec4(0.31F, 0.76F, 0.97F, 1.F), "CUDA");
            ImGui::TextDisabled("Preview transport");
            ImGui::TextWrapped("Vulkan external memory / timeline semaphore");
            ImGui::TextDisabled("Shared texture");
            ImGui::SameLine(ImGui::GetWindowWidth() - 104.F);
            ImGui::Text("%u x %u", preview.width, preview.height);
            ImGui::Spacing();
        }
        if (job.running || current_iteration > 0) {
            if (ImGui::CollapsingHeader(
                    "Training status", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Spacing();
                ImGui::Text("Iteration  %d / %d", current_iteration,
                            total_iterations);
                ImGui::PushStyleColor(
                    ImGuiCol_PlotHistogram,
                    ImVec4(0.0F, 0.47F, 0.83F, 1.F));
                ImGui::ProgressBar(progress, {-1.F, 7.F}, "");
                ImGui::PopStyleColor();
                ImGui::TextDisabled(
                    "Timeline value  %llu",
                    static_cast<unsigned long long>(external_consumed_value));
            }
        }
        ImGui::Dummy({0, 12.F});
        if (!job.running) {
            ImGui::PushStyleColor(ImGuiCol_Button,
                                  ImVec4(0.02F, 0.38F, 0.62F, 1.F));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                  ImVec4(0.03F, 0.48F, 0.76F, 1.F));
            if (!interop_smoke && ImGui::Button(
                    "Start Reconstruction", {-1.F, 39.F}))
                start_requested = true;
            ImGui::PopStyleColor(2);
        } else if (ImGui::Button("Stop Reconstruction", {-1.F, 39.F})) {
            job.stop();
        }
        ImGui::PopStyleColor();
        ImGui::EndChild();
        ImGui::EndChild();

        // Status bar.
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.035F, 0.28F, 0.44F, 1.F));
        ImGui::BeginChild("##status", {0, status_height}, false,
                          ImGuiWindowFlags_NoScrollbar);
        ImGui::SetCursorPos({10.F, 5.F});
        ImGui::Text("%s", job.running ? "Reconstruction running" : "Ready");
        ImGui::SameLine();
        ImGui::TextDisabled("  |  ");
        ImGui::SameLine();
        ImGui::Text("%s", scene_mode ? "Scene capture" : "Object capture");
        const char* status_right = "AetherScan 0.2  |  GPU accelerated";
        ImGui::SameLine(ImGui::GetWindowWidth() -
                        ImGui::CalcTextSize(status_right).x - 12.F);
        ImGui::TextUnformatted(status_right);
        ImGui::EndChild();
        ImGui::PopStyleColor();

        if (!job.running && start_requested) {
            smoke_started = true;
            std::filesystem::create_directories(job_dir);
            if (!preview.memory_handle || !preview.semaphore_handle)
                preview.create(1920, 1920);
            std::ostringstream command;
            command << quote(AETHERSCAN_CLI_PATH) << " --images "
                    << quote(images.data()) << " --output "
                    << quote(output_path);
            if (interop_smoke)
                command
                    << " --splat-dataset "
                    << quote("D:\\ScanVideo\\ori_img")
                    << " --splat-format colmap --splat-use-mask false";
            else
                command << " --capture-mode "
                        << (scene_mode ? "scene" : "object");
            command << " --splat --splat-strategy adc_plus"
                    << " --splat-iterations " << iterations
                    << " --splat-preview-interval " << preview_interval
                    << " --splat-preview-vk-memory-handle "
                    << reinterpret_cast<std::uintptr_t>(
                           preview.memory_handle)
                    << " --splat-preview-vk-semaphore-handle "
                    << reinterpret_cast<std::uintptr_t>(
                           preview.semaphore_handle)
                    << " --splat-preview-vk-allocation-size "
                    << preview.allocation_size
                    << " --splat-preview-vk-width " << preview.width
                    << " --splat-preview-vk-height " << preview.height;
            command << " --splat-preview-vk-device-luid "
                    << physical_device_luid
                    << " --splat-preview-vk-device-node-mask "
                    << physical_device_node_mask;
            job.start(command.str(), log_path);
            preview.close_export_handles();
            if (interop_smoke) glfwIconifyWindow(window);
        }
        ImGui::End();

        ImGui::Render();
        if (ImGui::GetDrawData()->DisplaySize.x > 0 &&
            ImGui::GetDrawData()->DisplaySize.y > 0) {
            window_data.ClearValue.color = {{0.025F, 0.03F, 0.04F, 1.F}};
            render_frame(ImGui::GetDrawData());
        }
    }

    if (job.running) job.stop();
    vkDeviceWaitIdle(device);
    preview.reset();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    ImGui_ImplVulkanH_DestroyWindow(
        instance, device, &window_data, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
    return interop_smoke && !smoke_success ? 4 : 0;
}
