#include "vulkan_backend.hpp"
#include "mesh_shaders.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace editor::gpu {
namespace {

VkInstance g_instance{};
VkPhysicalDevice g_physical_device{};
VkDevice g_device{};
std::uint32_t g_queue_family = UINT32_MAX;
VkQueue g_queue{};
VkDescriptorPool g_descriptor_pool{};
ImGui_ImplVulkanH_Window g_window{};
bool g_rebuild_swapchain{};
std::uint64_t g_device_luid{};
std::uint32_t g_device_node_mask{};

// The active shared-image state consulted by present().
VkImage g_external_image{};
VkImage g_display_image{};
VkSemaphore g_external_timeline{};
std::uint32_t g_external_width{};
std::uint32_t g_external_height{};
std::uint64_t g_ready_value{};
std::uint64_t g_consumed_value{};

bool has_extension(
    const ImVector<VkExtensionProperties>& properties, const char* name) {
    for (const auto& property : properties)
        if (std::strcmp(property.extensionName, name) == 0) return true;
    return false;
}

// Hands the shared image to the external queue family and takes the display
// image for transfer. Shared by the present and headless copy paths.
std::array<VkImageMemoryBarrier, 2> acquire_barriers() {
    std::array<VkImageMemoryBarrier, 2> acquire{};
    acquire[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    acquire[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    acquire[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    acquire[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    acquire[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    acquire[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    acquire[0].dstQueueFamilyIndex = g_queue_family;
    acquire[0].image = g_external_image;
    acquire[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    acquire[1] = acquire[0];
    acquire[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    acquire[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    acquire[1].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    acquire[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    acquire[1].srcQueueFamilyIndex = g_queue_family;
    acquire[1].image = g_display_image;
    return acquire;
}

std::array<VkImageMemoryBarrier, 2> release_barriers(
    const std::array<VkImageMemoryBarrier, 2>& acquire) {
    std::array<VkImageMemoryBarrier, 2> release{};
    release[0] = acquire[0];
    release[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    release[0].dstAccessMask = 0;
    release[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    release[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    release[0].srcQueueFamilyIndex = g_queue_family;
    release[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    release[1] = acquire[1];
    release[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    release[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    release[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    release[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return release;
}

void record_shared_copy(VkCommandBuffer command) {
    const auto acquire = acquire_barriers();
    vkCmdPipelineBarrier(
        command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
        static_cast<std::uint32_t>(acquire.size()), acquire.data());
    VkImageCopy copy{};
    copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {g_external_width, g_external_height, 1};
    vkCmdCopyImage(
        command, g_external_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        g_display_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    const auto release = release_barriers(acquire);
    vkCmdPipelineBarrier(
        command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
        static_cast<std::uint32_t>(release.size()), release.data());
}

}  // namespace

void check(const VkResult result) {
    if (result < 0) {
        std::fprintf(stderr, "Vulkan error: %d\n", result);
        std::abort();
    }
}

VkInstance instance() { return g_instance; }
VkPhysicalDevice physical_device() { return g_physical_device; }
VkDevice device() { return g_device; }
std::uint32_t queue_family() { return g_queue_family; }
VkQueue queue() { return g_queue; }
VkDescriptorPool descriptor_pool() { return g_descriptor_pool; }
ImGui_ImplVulkanH_Window& window() { return g_window; }
std::uint64_t device_luid() { return g_device_luid; }
std::uint32_t device_node_mask() { return g_device_node_mask; }
bool swapchain_needs_rebuild() { return g_rebuild_swapchain; }
void clear_swapchain_rebuild() { g_rebuild_swapchain = false; }
std::uint64_t consumed_timeline_value() { return g_consumed_value; }
std::uint64_t ready_timeline_value() { return g_ready_value; }

void create_context(ImVector<const char*> extensions) {
    std::uint32_t count{};
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    ImVector<VkExtensionProperties> properties;
    properties.resize(static_cast<int>(count));
    check(vkEnumerateInstanceExtensionProperties(
        nullptr, &count, properties.Data));
    if (has_extension(
            properties, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
        extensions.push_back(
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "AetherScan Editor";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &app;
    create.enabledExtensionCount = static_cast<std::uint32_t>(extensions.Size);
    create.ppEnabledExtensionNames = extensions.Data;
    check(vkCreateInstance(&create, nullptr, &g_instance));

    g_physical_device = ImGui_ImplVulkanH_SelectPhysicalDevice(g_instance);
    g_queue_family =
        ImGui_ImplVulkanH_SelectQueueFamilyIndex(g_physical_device);

    VkPhysicalDeviceIDProperties id{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 device_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    device_properties.pNext = &id;
    vkGetPhysicalDeviceProperties2(g_physical_device, &device_properties);
    if (id.deviceLUIDValid) {
        static_assert(sizeof(g_device_luid) == VK_LUID_SIZE);
        std::memcpy(&g_device_luid, id.deviceLUID, sizeof(g_device_luid));
        g_device_node_mask = id.deviceNodeMask;
    }

    const float priority = 1.F;
    VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = g_queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    const char* required_device_extensions[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#if defined(_WIN32)
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#endif
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#if defined(_WIN32)
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#endif
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME};
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    timeline.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &timeline;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount =
        static_cast<std::uint32_t>(std::size(required_device_extensions));
    device_info.ppEnabledExtensionNames = required_device_extensions;
    check(vkCreateDevice(
        g_physical_device, &device_info, nullptr, &g_device));
    vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);

    std::array<VkDescriptorPoolSize, 2> sizes{{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024},
        {VK_DESCRIPTOR_TYPE_SAMPLER, 128},
    }};
    VkDescriptorPoolCreateInfo pool{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool.maxSets = 1024;
    pool.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    pool.pPoolSizes = sizes.data();
    check(vkCreateDescriptorPool(
        g_device, &pool, nullptr, &g_descriptor_pool));
}

void create_window(VkSurfaceKHR surface, const int width, const int height) {
    g_window.Surface = surface;
    VkBool32 supported{};
    vkGetPhysicalDeviceSurfaceSupportKHR(
        g_physical_device, g_queue_family, surface, &supported);
    if (!supported)
        throw std::runtime_error("Vulkan queue has no WSI support");
    const VkFormat formats[] = {
        VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    g_window.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        g_physical_device, surface, formats, 2,
        VK_COLORSPACE_SRGB_NONLINEAR_KHR);
    const VkPresentModeKHR present = VK_PRESENT_MODE_FIFO_KHR;
    g_window.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        g_physical_device, surface, &present, 1);
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        g_instance, g_physical_device, g_device, &g_window, g_queue_family,
        nullptr, width, height, k_min_images);
}

void resize_window(const int width, const int height) {
    ImGui_ImplVulkan_SetMinImageCount(k_min_images);
    ImGui_ImplVulkanH_CreateOrResizeWindow(
        g_instance, g_physical_device, g_device, &g_window, g_queue_family,
        nullptr, width, height, k_min_images);
    g_window.FrameIndex = 0;
    g_rebuild_swapchain = false;
}

void destroy_context() {
    ImGui_ImplVulkanH_DestroyWindow(g_instance, g_device, &g_window, nullptr);
    vkDestroyDescriptorPool(g_device, g_descriptor_pool, nullptr);
    vkDestroyDevice(g_device, nullptr);
    vkDestroyInstance(g_instance, nullptr);
    g_device = {};
    g_instance = {};
}

void present(ImDrawData* draw, const ImVec4& clear_colour) {
    auto& wd = g_window;
    wd.ClearValue.color = {
        {clear_colour.x, clear_colour.y, clear_colour.z, clear_colour.w}};
    VkSemaphore acquired =
        wd.FrameSemaphores[wd.SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore complete =
        wd.FrameSemaphores[wd.SemaphoreIndex].RenderCompleteSemaphore;
    VkResult result = vkAcquireNextImageKHR(
        g_device, wd.Swapchain, UINT64_MAX, acquired, {}, &wd.FrameIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        g_rebuild_swapchain = true;
    if (result == VK_ERROR_OUT_OF_DATE_KHR) return;
    if (result != VK_SUBOPTIMAL_KHR) check(result);

    auto& frame = wd.Frames[wd.FrameIndex];
    check(vkWaitForFences(g_device, 1, &frame.Fence, VK_TRUE, UINT64_MAX));
    check(vkResetFences(g_device, 1, &frame.Fence));
    check(vkResetCommandPool(g_device, frame.CommandPool, 0));
    VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(frame.CommandBuffer, &begin));

    const bool copy_shared = g_external_image && g_display_image &&
                             g_external_timeline &&
                             g_ready_value > g_consumed_value;
    if (copy_shared) record_shared_copy(frame.CommandBuffer);

    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = wd.RenderPass;
    pass.framebuffer = frame.Framebuffer;
    pass.renderArea.extent = {
        static_cast<std::uint32_t>(wd.Width),
        static_cast<std::uint32_t>(wd.Height)};
    pass.clearValueCount = 1;
    pass.pClearValues = &wd.ClearValue;
    vkCmdBeginRenderPass(
        frame.CommandBuffer, &pass, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(draw, frame.CommandBuffer);
    vkCmdEndRenderPass(frame.CommandBuffer);
    check(vkEndCommandBuffer(frame.CommandBuffer));

    const std::array<VkSemaphore, 2> waits{{acquired, g_external_timeline}};
    const std::array<VkPipelineStageFlags, 2> wait_stages{
        {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT}};
    const std::array<VkSemaphore, 2> signals{{complete, g_external_timeline}};
    const std::array<std::uint64_t, 2> wait_values{{0, g_ready_value}};
    const std::array<std::uint64_t, 2> signal_values{{0, g_ready_value + 1}};
    VkTimelineSemaphoreSubmitInfo timeline{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    if (copy_shared) {
        timeline.waitSemaphoreValueCount = 2;
        timeline.pWaitSemaphoreValues = wait_values.data();
        timeline.signalSemaphoreValueCount = 2;
        timeline.pSignalSemaphoreValues = signal_values.data();
        submit.pNext = &timeline;
    }
    submit.waitSemaphoreCount = copy_shared ? 2U : 1U;
    submit.pWaitSemaphores = waits.data();
    submit.pWaitDstStageMask = wait_stages.data();
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &frame.CommandBuffer;
    submit.signalSemaphoreCount = copy_shared ? 2U : 1U;
    submit.pSignalSemaphores = signals.data();
    check(vkQueueSubmit(g_queue, 1, &submit, frame.Fence));
    if (copy_shared) g_consumed_value = g_ready_value;

    VkPresentInfoKHR info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &complete;
    info.swapchainCount = 1;
    info.pSwapchains = &wd.Swapchain;
    info.pImageIndices = &wd.FrameIndex;
    result = vkQueuePresentKHR(g_queue, &info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        g_rebuild_swapchain = true;
    else
        check(result);
    wd.SemaphoreIndex = (wd.SemaphoreIndex + 1) % wd.SemaphoreCount;
}

std::uint32_t memory_type(
    const std::uint32_t bits, const VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(g_physical_device, &properties);
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i)
        if ((bits & (1U << i)) &&
            (properties.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    throw std::runtime_error("No Vulkan memory type");
}

void PreviewTexture::reset() {
    if (!g_device) return;
    vkDeviceWaitIdle(g_device);
    if (descriptor) ImGui_ImplVulkan_RemoveTexture(descriptor);
    if (sampler) vkDestroySampler(g_device, sampler, nullptr);
    if (view) vkDestroyImageView(g_device, view, nullptr);
    if (image) vkDestroyImage(g_device, image, nullptr);
    if (memory) vkFreeMemory(g_device, memory, nullptr);
    *this = {};
}

void PreviewTexture::upload(const aetherscan::io::RgbImage& source) {
    if (source.width == 0 || source.height == 0) {
        reset();
        return;
    }
    const bool recreate =
        image == nullptr || width != source.width || height != source.height;
    if (recreate) {
        reset();
        width = source.width;
        height = source.height;
    }
    const VkDeviceSize bytes =
        static_cast<VkDeviceSize>(width) * height * 4;
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(bytes));
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    for (std::size_t i = 0; i < pixels; ++i) {
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
    check(vkCreateBuffer(g_device, &buffer, nullptr, &staging));
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(g_device, staging, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type(
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkAllocateMemory(g_device, &allocation, nullptr, &staging_memory));
    check(vkBindBufferMemory(g_device, staging, staging_memory, 0));
    void* mapped{};
    check(vkMapMemory(g_device, staging_memory, 0, bytes, 0, &mapped));
    std::memcpy(mapped, rgba.data(), static_cast<std::size_t>(bytes));
    vkUnmapMemory(g_device, staging_memory);

    if (!recreate) {
        auto& frame = g_window.Frames[g_window.FrameIndex];
        check(vkResetCommandPool(g_device, frame.CommandPool, 0));
        VkCommandBufferBeginInfo begin{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(frame.CommandBuffer, &begin));
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(
            frame.CommandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
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
        check(vkEndCommandBuffer(frame.CommandBuffer));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &frame.CommandBuffer;
        check(vkQueueSubmit(g_queue, 1, &submit, {}));
        check(vkQueueWaitIdle(g_queue));
        vkDestroyBuffer(g_device, staging, nullptr);
        vkFreeMemory(g_device, staging_memory, nullptr);
        return;
    }

    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
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
    check(vkCreateImage(g_device, &image_info, nullptr, &image));
    vkGetImageMemoryRequirements(g_device, image, &requirements);
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(g_device, &allocation, nullptr, &memory));
    check(vkBindImageMemory(g_device, image, memory, 0));

    auto& frame = g_window.Frames[g_window.FrameIndex];
    check(vkResetCommandPool(g_device, frame.CommandPool, 0));
    VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(frame.CommandBuffer, &begin));
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
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
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
        &barrier);
    check(vkEndCommandBuffer(frame.CommandBuffer));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &frame.CommandBuffer;
    check(vkQueueSubmit(g_queue, 1, &submit, {}));
    check(vkQueueWaitIdle(g_queue));
    vkDestroyBuffer(g_device, staging, nullptr);
    vkFreeMemory(g_device, staging_memory, nullptr);

    VkImageViewCreateInfo view_info{
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(g_device, &view_info, nullptr, &view));
    VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = sampler_info.addressModeV =
        sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.maxLod = 1.F;
    check(vkCreateSampler(g_device, &sampler_info, nullptr, &sampler));
    descriptor = ImGui_ImplVulkan_AddTexture(
        sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

bool PreviewTexture::download_rgb(aetherscan::io::RgbImage& destination) const {
    destination = {};
    if (!g_device || !image || width == 0 || height == 0) return false;

    const VkDeviceSize bytes =
        static_cast<VkDeviceSize>(width) * height * 4;
    VkBuffer staging{};
    VkDeviceMemory staging_memory{};
    VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer.size = bytes;
    buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    check(vkCreateBuffer(g_device, &buffer, nullptr, &staging));
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(g_device, staging, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type(
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkAllocateMemory(g_device, &allocation, nullptr, &staging_memory));
    check(vkBindBufferMemory(g_device, staging, staging_memory, 0));

    check(vkQueueWaitIdle(g_queue));
    auto& frame = g_window.Frames[g_window.FrameIndex];
    check(vkResetCommandPool(g_device, frame.CommandPool, 0));
    VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(frame.CommandBuffer, &begin));
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(
        frame.CommandBuffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
        &barrier);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(
        frame.CommandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        staging, 1, &copy);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        frame.CommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
        &barrier);
    check(vkEndCommandBuffer(frame.CommandBuffer));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &frame.CommandBuffer;
    check(vkQueueSubmit(g_queue, 1, &submit, {}));
    check(vkQueueWaitIdle(g_queue));

    void* mapped{};
    check(vkMapMemory(g_device, staging_memory, 0, bytes, 0, &mapped));
    const auto* rgba = static_cast<const std::uint8_t*>(mapped);
    destination.width = width;
    destination.height = height;
    destination.pixels.resize(static_cast<std::size_t>(width) * height * 3);
    const std::size_t pixels = static_cast<std::size_t>(width) * height;
    for (std::size_t i = 0; i < pixels; ++i) {
        destination.pixels[3 * i] = rgba[4 * i];
        destination.pixels[3 * i + 1] = rgba[4 * i + 1];
        destination.pixels[3 * i + 2] = rgba[4 * i + 2];
    }
    vkUnmapMemory(g_device, staging_memory);
    vkDestroyBuffer(g_device, staging, nullptr);
    vkFreeMemory(g_device, staging_memory, nullptr);
    return !destination.pixels.empty();
}

void ExternalPreview::create(
    const std::uint32_t image_width, const std::uint32_t image_height) {
    reset();
    width = image_width;
    height = image_height;

    aetherscan::io::RgbImage placeholder;
    placeholder.width = width;
    placeholder.height = height;
    placeholder.pixels.assign(
        static_cast<std::size_t>(width) * height * 3, 10);
    display.upload(placeholder);
    g_display_image = display.image;

#if defined(_WIN32)
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
    check(vkCreateImage(g_device, &image_info, nullptr, &image));
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(g_device, image, &requirements);
    allocation_size = requirements.size;

    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
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
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.pNext = &dedicated;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(g_device, &allocation, nullptr, &memory));
    check(vkBindImageMemory(g_device, image, memory, 0));

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
    check(vkCreateSemaphore(
        g_device, &semaphore_info, nullptr, &timeline));

    renew_export_handles();

    auto& frame = g_window.Frames[g_window.FrameIndex];
    check(vkResetCommandPool(g_device, frame.CommandPool, 0));
    VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(frame.CommandBuffer, &begin));
    VkImageMemoryBarrier release{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    release.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    release.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    release.srcQueueFamilyIndex = g_queue_family;
    release.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    release.image = image;
    release.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(
        frame.CommandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1,
        &release);
    check(vkEndCommandBuffer(frame.CommandBuffer));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &frame.CommandBuffer;
    check(vkQueueSubmit(g_queue, 1, &submit, {}));
    check(vkQueueWaitIdle(g_queue));

    g_external_image = image;
    g_external_timeline = timeline;
    g_external_width = width;
    g_external_height = height;
    g_ready_value = 0;
    g_consumed_value = 0;
#endif
}

void ExternalPreview::renew_export_handles() {
#if defined(_WIN32)
    close_export_handles();
    const auto get_memory = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(g_device, "vkGetMemoryWin32HandleKHR"));
    const auto get_semaphore =
        reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(g_device, "vkGetSemaphoreWin32HandleKHR"));
    if (!get_memory || !get_semaphore)
        throw std::runtime_error("Vulkan Win32 export functions missing");
    VkMemoryGetWin32HandleInfoKHR memory_info{
        VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    memory_info.memory = memory;
    memory_info.handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    check(get_memory(g_device, &memory_info, &memory_handle));
    VkSemaphoreGetWin32HandleInfoKHR timeline_info{
        VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    timeline_info.semaphore = timeline;
    timeline_info.handleType =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    check(get_semaphore(g_device, &timeline_info, &semaphore_handle));
#endif
}

void ExternalPreview::close_export_handles() {
#if defined(_WIN32)
    if (memory_handle) CloseHandle(std::exchange(memory_handle, nullptr));
    if (semaphore_handle)
        CloseHandle(std::exchange(semaphore_handle, nullptr));
#endif
}

void ExternalPreview::poll() {
    if (!timeline) return;
    std::uint64_t value{};
    if (vkGetSemaphoreCounterValue(g_device, timeline, &value) == VK_SUCCESS &&
        (value & 1U) != 0 && value > g_consumed_value)
        g_ready_value = value;
}

void ExternalPreview::consume_without_present() {
    if (g_ready_value <= g_consumed_value) return;
    check(vkQueueWaitIdle(g_queue));
    auto& frame = g_window.Frames[g_window.FrameIndex];
    check(vkResetCommandPool(g_device, frame.CommandPool, 0));
    VkCommandBufferBeginInfo begin{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(frame.CommandBuffer, &begin));
    record_shared_copy(frame.CommandBuffer);
    check(vkEndCommandBuffer(frame.CommandBuffer));

    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const std::uint64_t signal_value = g_ready_value + 1;
    VkTimelineSemaphoreSubmitInfo timeline_info{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline_info.waitSemaphoreValueCount = 1;
    timeline_info.pWaitSemaphoreValues = &g_ready_value;
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
    check(vkQueueSubmit(g_queue, 1, &submit, {}));
    check(vkQueueWaitIdle(g_queue));
    g_consumed_value = g_ready_value;
}

void ExternalPreview::reset() {
    if (!g_device) return;
    vkDeviceWaitIdle(g_device);
    g_external_image = {};
    g_display_image = {};
    g_external_timeline = {};
    g_ready_value = g_consumed_value = 0;
    close_export_handles();
    if (timeline) vkDestroySemaphore(g_device, timeline, nullptr);
    if (image) vkDestroyImage(g_device, image, nullptr);
    if (memory) vkFreeMemory(g_device, memory, nullptr);
    timeline = {};
    image = {};
    memory = {};
    allocation_size = 0;
    width = height = 0;
    display.reset();
}

namespace {

constexpr std::uint32_t k_thumb_long_edge = 320;
constexpr std::size_t k_max_inflight = 2;

aetherscan::io::RgbImage load_camera_thumbnail(
    const std::filesystem::path& path) {
    aetherscan::io::RgbImage rgb = aetherscan::io::load_rgb(path);
    const std::uint32_t long_edge = std::max(rgb.width, rgb.height);
    if (long_edge <= k_thumb_long_edge || long_edge == 0) return rgb;
    const float scale =
        static_cast<float>(k_thumb_long_edge) / static_cast<float>(long_edge);
    const std::uint32_t width = std::max(
        1U, static_cast<std::uint32_t>(std::lround(
                static_cast<float>(rgb.width) * scale)));
    const std::uint32_t height = std::max(
        1U, static_cast<std::uint32_t>(std::lround(
                static_cast<float>(rgb.height) * scale)));
    aetherscan::io::RgbImage thumb;
    thumb.width = width;
    thumb.height = height;
    thumb.pixels.resize(static_cast<std::size_t>(width) * height * 3);
    aetherscan::io::resize_bilinear(
        rgb.pixels.data(), rgb.width, rgb.height, 3, thumb.pixels.data(),
        width, height);
    return thumb;
}

}  // namespace

void CameraPhotoCache::clear() {
    for (auto& slot : slots_) {
        if (!slot) continue;
        if (slot->pending.valid()) slot->pending.wait();
        slot->texture.reset();
    }
    slots_.clear();
    ids_.clear();
}

void CameraPhotoCache::resize(const std::size_t view_count) {
    if (view_count == slots_.size()) return;
    if (view_count < slots_.size()) {
        for (std::size_t i = view_count; i < slots_.size(); ++i) {
            if (!slots_[i]) continue;
            if (slots_[i]->pending.valid()) slots_[i]->pending.wait();
            slots_[i]->texture.reset();
        }
    }
    slots_.resize(view_count);
    ids_.resize(view_count);
}

std::size_t CameraPhotoCache::inflight() const {
    std::size_t count = 0;
    for (const auto& slot : slots_)
        if (slot && slot->loading) ++count;
    return count;
}

void CameraPhotoCache::request(
    const std::size_t index, const std::filesystem::path& path) {
    if (index >= slots_.size() || path.empty()) return;
    if (!slots_[index]) slots_[index] = std::make_unique<Slot>();
    Slot& slot = *slots_[index];
    if (slot.path != path) {
        if (slot.pending.valid()) slot.pending.wait();
        slot.texture.reset();
        ids_[index] = {};
        slot.pending = {};
        slot.loading = false;
        slot.failed = false;
        slot.path = path;
    }
    if (slot.failed || slot.loading || ids_[index]) return;
    if (inflight() >= k_max_inflight) return;
    slot.loading = true;
    slot.pending = std::async(std::launch::async, [path] {
        return load_camera_thumbnail(path);
    });
}

void CameraPhotoCache::poll() {
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        Slot* slot = slots_[i].get();
        if (!slot || !slot->loading || !slot->pending.valid()) continue;
        if (slot->pending.wait_for(std::chrono::seconds(0)) !=
            std::future_status::ready)
            continue;
        try {
            const aetherscan::io::RgbImage image = slot->pending.get();
            if (image.width == 0 || image.height == 0 || image.pixels.empty())
                throw std::runtime_error("empty camera thumbnail");
            slot->texture.upload(image);
            ids_[i] = reinterpret_cast<ImTextureID>(slot->texture.descriptor);
        } catch (...) {
            slot->failed = true;
            ids_[i] = {};
        }
        slot->loading = false;
        return;
    }
}

ImTextureID CameraPhotoCache::id(const std::size_t index) const {
    return index < ids_.size() ? ids_[index] : ImTextureID{};
}

namespace {

struct MeshGpuVertex {
    float px, py, pz, pad0;
    float nx, ny, nz, pad1;
    float r, g, b, pad2;
    float u, v, pad3, pad4;
};
static_assert(sizeof(MeshGpuVertex) == 64, "mesh vertex must stay 16-byte aligned");

struct MeshPush {
    float world_to_clip[16];
    float cam0[4];
    float cam1[4];
    float cam2[4];
    float clay_flags[4];
};
static_assert(sizeof(MeshPush) == 128, "mesh push constants must fit in 128 bytes");

VkFormat mesh_depth_format() {
    const VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM};
    for (const VkFormat format : candidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(g_physical_device, format, &properties);
        if (properties.optimalTilingFeatures &
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            return format;
    }
    return VK_FORMAT_D32_SFLOAT;
}

VkShaderModule mesh_shader_module(
    const std::uint32_t* words, const std::size_t word_count) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = word_count * sizeof(std::uint32_t);
    info.pCode = words;
    VkShaderModule module{};
    check(vkCreateShaderModule(g_device, &info, nullptr, &module));
    return module;
}

void destroy_buffer(VkBuffer& buffer, VkDeviceMemory& memory) {
    if (buffer) vkDestroyBuffer(g_device, buffer, nullptr);
    if (memory) vkFreeMemory(g_device, memory, nullptr);
    buffer = {};
    memory = {};
}

bool create_buffer_with_data(
    VkBuffer& buffer, VkDeviceMemory& memory, const VkBufferUsageFlags usage,
    const void* data, const std::size_t bytes) {
    destroy_buffer(buffer, memory);
    if (bytes == 0 || data == nullptr) return true;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = bytes;
    info.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    check(vkCreateBuffer(g_device, &info, nullptr, &buffer));
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(g_device, buffer, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(g_device, &allocation, nullptr, &memory));
    check(vkBindBufferMemory(g_device, buffer, memory, 0));

    VkBuffer staging{};
    VkDeviceMemory staging_memory{};
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    check(vkCreateBuffer(g_device, &info, nullptr, &staging));
    vkGetBufferMemoryRequirements(g_device, staging, &requirements);
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type(
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkAllocateMemory(g_device, &allocation, nullptr, &staging_memory));
    check(vkBindBufferMemory(g_device, staging, staging_memory, 0));
    void* mapped{};
    check(vkMapMemory(g_device, staging_memory, 0, bytes, 0, &mapped));
    std::memcpy(mapped, data, bytes);
    vkUnmapMemory(g_device, staging_memory);

    VkCommandPool pool{};
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = g_queue_family;
    check(vkCreateCommandPool(g_device, &pool_info, nullptr, &pool));
    VkCommandBufferAllocateInfo cmd_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_info.commandPool = pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;
    VkCommandBuffer command{};
    check(vkAllocateCommandBuffers(g_device, &cmd_info, &command));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin));
    VkBufferCopy copy{};
    copy.size = bytes;
    vkCmdCopyBuffer(command, staging, buffer, 1, &copy);
    check(vkEndCommandBuffer(command));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(g_queue, 1, &submit, {}));
    check(vkQueueWaitIdle(g_queue));
    vkDestroyCommandPool(g_device, pool, nullptr);
    vkDestroyBuffer(g_device, staging, nullptr);
    vkFreeMemory(g_device, staging_memory, nullptr);
    return true;
}

void destroy_image(
    VkImage& image, VkDeviceMemory& memory, VkImageView& view) {
    if (view) vkDestroyImageView(g_device, view, nullptr);
    if (image) vkDestroyImage(g_device, image, nullptr);
    if (memory) vkFreeMemory(g_device, memory, nullptr);
    view = {};
    image = {};
    memory = {};
}

void create_sampled_image(
    VkImage& image, VkDeviceMemory& memory, VkImageView& view,
    const std::uint32_t width, const std::uint32_t height) {
    destroy_image(image, memory, view);
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R8G8B8A8_UNORM;
    info.extent = {width, height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    check(vkCreateImage(g_device, &info, nullptr, &image));
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(g_device, image, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type(
        requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkAllocateMemory(g_device, &allocation, nullptr, &memory));
    check(vkBindImageMemory(g_device, image, memory, 0));
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    view_info.image = image;
    check(vkCreateImageView(g_device, &view_info, nullptr, &view));
}

void upload_sampled_image(
    VkImage image, const std::uint32_t width, const std::uint32_t height,
    const std::uint8_t* rgba) {
    const VkDeviceSize bytes =
        static_cast<VkDeviceSize>(width) * height * 4U;
    VkBuffer staging{};
    VkDeviceMemory staging_memory{};
    VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer.size = bytes;
    buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    check(vkCreateBuffer(g_device, &buffer, nullptr, &staging));
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(g_device, staging, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type(
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkAllocateMemory(g_device, &allocation, nullptr, &staging_memory));
    check(vkBindBufferMemory(g_device, staging, staging_memory, 0));
    void* mapped{};
    check(vkMapMemory(g_device, staging_memory, 0, bytes, 0, &mapped));
    std::memcpy(mapped, rgba, static_cast<std::size_t>(bytes));
    vkUnmapMemory(g_device, staging_memory);

    VkCommandPool pool{};
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = g_queue_family;
    check(vkCreateCommandPool(g_device, &pool_info, nullptr, &pool));
    VkCommandBufferAllocateInfo cmd_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_info.commandPool = pool;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;
    VkCommandBuffer command{};
    check(vkAllocateCommandBuffers(g_device, &cmd_info, &command));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin));
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(
        command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
        &barrier);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(
        command, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
        &copy);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
        &barrier);
    check(vkEndCommandBuffer(command));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(g_queue, 1, &submit, {}));
    check(vkQueueWaitIdle(g_queue));
    vkDestroyCommandPool(g_device, pool, nullptr);
    vkDestroyBuffer(g_device, staging, nullptr);
    vkFreeMemory(g_device, staging_memory, nullptr);
}

VkPipeline create_mesh_pipeline(
    VkRenderPass render_pass, VkPipelineLayout layout, VkShaderModule vert,
    VkShaderModule frag, const VkPrimitiveTopology topology,
    const bool depth_bias) {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.stride = sizeof(MeshGpuVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attributes[4]{};
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].location = 1;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset = 16;
    attributes[2].location = 2;
    attributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[2].offset = 32;
    attributes[3].location = 3;
    attributes[3].format = VK_FORMAT_R32G32_SFLOAT;
    attributes[3].offset = 48;
    VkPipelineVertexInputStateCreateInfo vertex{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex.vertexBindingDescriptionCount = 1;
    vertex.pVertexBindingDescriptions = &binding;
    vertex.vertexAttributeDescriptionCount = 4;
    vertex.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = topology;
    VkPipelineViewportStateCreateInfo viewport{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.F;
    raster.depthBiasEnable = depth_bias ? VK_TRUE : VK_FALSE;
    raster.depthBiasConstantFactor = depth_bias ? -1.F : 0.F;
    raster.depthBiasSlopeFactor = depth_bias ? -1.F : 0.F;
    VkPipelineMultisampleStateCreateInfo sample{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    sample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        ? VK_TRUE
        : VK_FALSE;
    depth.depthCompareOp = topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
        ? VK_COMPARE_OP_LESS
        : VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                      VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &blend_attachment;
    const VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;
    VkGraphicsPipelineCreateInfo info{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertex;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &sample;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    info.renderPass = render_pass;
    VkPipeline pipeline{};
    check(vkCreateGraphicsPipelines(
        g_device, {}, 1, &info, nullptr, &pipeline));
    return pipeline;
}

}  // namespace

void MeshPreviewRenderer::destroy_mesh_buffers() {
    if (!g_device) {
        vertex_buffer_ = {};
        index_buffer_ = {};
        edge_buffer_ = {};
        vertex_memory_ = {};
        index_memory_ = {};
        edge_memory_ = {};
        index_count_ = 0;
        edge_count_ = 0;
        return;
    }
    vkDeviceWaitIdle(g_device);
    destroy_buffer(vertex_buffer_, vertex_memory_);
    destroy_buffer(index_buffer_, index_memory_);
    destroy_buffer(edge_buffer_, edge_memory_);
    index_count_ = 0;
    edge_count_ = 0;
}

void MeshPreviewRenderer::destroy_frames() {
    if (!g_device) {
        for (Frame& frame : frames_) frame = {};
        width_ = 0;
        height_ = 0;
        display_ = -1;
        write_ = 0;
        return;
    }
    vkDeviceWaitIdle(g_device);
    for (Frame& frame : frames_) {
        if (frame.descriptor)
            ImGui_ImplVulkan_RemoveTexture(frame.descriptor);
        if (frame.sampler) vkDestroySampler(g_device, frame.sampler, nullptr);
        if (frame.framebuffer)
            vkDestroyFramebuffer(g_device, frame.framebuffer, nullptr);
        if (frame.color_view)
            vkDestroyImageView(g_device, frame.color_view, nullptr);
        if (frame.depth_view)
            vkDestroyImageView(g_device, frame.depth_view, nullptr);
        if (frame.color) vkDestroyImage(g_device, frame.color, nullptr);
        if (frame.depth) vkDestroyImage(g_device, frame.depth, nullptr);
        if (frame.color_memory)
            vkFreeMemory(g_device, frame.color_memory, nullptr);
        if (frame.depth_memory)
            vkFreeMemory(g_device, frame.depth_memory, nullptr);
        if (frame.command && command_pool_)
            vkFreeCommandBuffers(g_device, command_pool_, 1, &frame.command);
        if (frame.fence) vkDestroyFence(g_device, frame.fence, nullptr);
        frame = {};
    }
    width_ = 0;
    height_ = 0;
    display_ = -1;
    write_ = 0;
}

void MeshPreviewRenderer::destroy_albedo() {
    if (!g_device) {
        albedo_image_ = {};
        albedo_memory_ = {};
        albedo_view_ = {};
        albedo_sampler_ = {};
        albedo_set_ = {};
        albedo_pool_ = {};
        albedo_layout_ = {};
        albedo_width_ = 0;
        albedo_height_ = 0;
        return;
    }
    vkDeviceWaitIdle(g_device);
    if (albedo_view_) vkDestroyImageView(g_device, albedo_view_, nullptr);
    if (albedo_image_) vkDestroyImage(g_device, albedo_image_, nullptr);
    if (albedo_memory_) vkFreeMemory(g_device, albedo_memory_, nullptr);
    if (albedo_sampler_) vkDestroySampler(g_device, albedo_sampler_, nullptr);
    if (albedo_pool_) vkDestroyDescriptorPool(g_device, albedo_pool_, nullptr);
    if (albedo_layout_)
        vkDestroyDescriptorSetLayout(g_device, albedo_layout_, nullptr);
    albedo_view_ = {};
    albedo_image_ = {};
    albedo_memory_ = {};
    albedo_sampler_ = {};
    albedo_set_ = {};
    albedo_pool_ = {};
    albedo_layout_ = {};
    albedo_width_ = 0;
    albedo_height_ = 0;
}

void MeshPreviewRenderer::destroy_pipeline() {
    if (!g_device) {
        fill_pipeline_ = {};
        wire_pipeline_ = {};
        pipeline_layout_ = {};
        render_pass_ = {};
        command_pool_ = {};
        return;
    }
    vkDeviceWaitIdle(g_device);
    if (fill_pipeline_)
        vkDestroyPipeline(g_device, fill_pipeline_, nullptr);
    if (wire_pipeline_)
        vkDestroyPipeline(g_device, wire_pipeline_, nullptr);
    if (pipeline_layout_)
        vkDestroyPipelineLayout(g_device, pipeline_layout_, nullptr);
    if (render_pass_) vkDestroyRenderPass(g_device, render_pass_, nullptr);
    if (command_pool_)
        vkDestroyCommandPool(g_device, command_pool_, nullptr);
    fill_pipeline_ = {};
    wire_pipeline_ = {};
    pipeline_layout_ = {};
    render_pass_ = {};
    command_pool_ = {};
}

void MeshPreviewRenderer::reset() {
    destroy_frames();
    destroy_mesh_buffers();
    destroy_pipeline();
    destroy_albedo();
}

bool MeshPreviewRenderer::ensure_pipeline() {
    if (fill_pipeline_ && wire_pipeline_ && render_pass_) return true;
    destroy_pipeline();

    const VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
    const VkFormat depth_format = mesh_depth_format();
    VkAttachmentDescription attachments[2]{};
    attachments[0].format = color_format;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    attachments[1].format = depth_format;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depth_ref{
        1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;
    subpass.pDepthStencilAttachment = &depth_ref;
    VkSubpassDependency dependencies[2]{};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass.attachmentCount = 2;
    pass.pAttachments = attachments;
    pass.subpassCount = 1;
    pass.pSubpasses = &subpass;
    pass.dependencyCount = 2;
    pass.pDependencies = dependencies;
    check(vkCreateRenderPass(g_device, &pass, nullptr, &render_pass_));

    if (!ensure_albedo()) return false;

    VkPushConstantRange push{};
    push.stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push.size = sizeof(MeshPush);
    VkPipelineLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &albedo_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    check(vkCreatePipelineLayout(
        g_device, &layout_info, nullptr, &pipeline_layout_));

    const VkShaderModule vert = mesh_shader_module(
        mesh_shaders::k_vert, std::size(mesh_shaders::k_vert));
    const VkShaderModule frag = mesh_shader_module(
        mesh_shaders::k_frag, std::size(mesh_shaders::k_frag));
    const VkShaderModule wire_frag = mesh_shader_module(
        mesh_shaders::k_wire_frag, std::size(mesh_shaders::k_wire_frag));
    fill_pipeline_ = create_mesh_pipeline(
        render_pass_, pipeline_layout_, vert, frag,
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false);
    wire_pipeline_ = create_mesh_pipeline(
        render_pass_, pipeline_layout_, vert, wire_frag,
        VK_PRIMITIVE_TOPOLOGY_LINE_LIST, true);
    vkDestroyShaderModule(g_device, vert, nullptr);
    vkDestroyShaderModule(g_device, frag, nullptr);
    vkDestroyShaderModule(g_device, wire_frag, nullptr);

    VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool.queueFamilyIndex = g_queue_family;
    check(vkCreateCommandPool(g_device, &pool, nullptr, &command_pool_));
    return fill_pipeline_ && wire_pipeline_;
}

void MeshPreviewRenderer::bind_albedo_view(VkImageView view) {
    if (!albedo_set_ || !albedo_sampler_ || !view) return;
    VkDescriptorImageInfo image{};
    image.sampler = albedo_sampler_;
    image.imageView = view;
    image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = albedo_set_;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image;
    vkUpdateDescriptorSets(g_device, 1, &write, 0, nullptr);
}

bool MeshPreviewRenderer::ensure_albedo() {
    if (albedo_set_ && albedo_view_) return true;
    if (!g_device) return false;

    if (!albedo_layout_) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo layout{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.bindingCount = 1;
        layout.pBindings = &binding;
        check(vkCreateDescriptorSetLayout(
            g_device, &layout, nullptr, &albedo_layout_));
    }
    if (!albedo_pool_) {
        VkDescriptorPoolSize size{};
        size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        size.descriptorCount = 1;
        VkDescriptorPoolCreateInfo pool{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &size;
        check(vkCreateDescriptorPool(g_device, &pool, nullptr, &albedo_pool_));
    }
    if (!albedo_set_) {
        VkDescriptorSetAllocateInfo alloc{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool = albedo_pool_;
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts = &albedo_layout_;
        check(vkAllocateDescriptorSets(g_device, &alloc, &albedo_set_));
    }
    if (!albedo_sampler_) {
        VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.maxLod = 1.F;
        check(vkCreateSampler(g_device, &sampler, nullptr, &albedo_sampler_));
    }
    if (!albedo_view_) {
        const std::uint8_t white[4] = {255, 255, 255, 255};
        create_sampled_image(albedo_image_, albedo_memory_, albedo_view_, 1, 1);
        upload_sampled_image(albedo_image_, 1, 1, white);
        albedo_width_ = 1;
        albedo_height_ = 1;
        bind_albedo_view(albedo_view_);
    }
    return albedo_set_ != VK_NULL_HANDLE;
}

void MeshPreviewRenderer::set_albedo(const aetherscan::io::RgbImage& atlas) {
    if (!g_device || !ensure_albedo()) return;
    if (atlas.width == 0 || atlas.height == 0 || atlas.pixels.size() <
            static_cast<std::size_t>(atlas.width) * atlas.height * 3U) {
        if (albedo_width_ != 1 || albedo_height_ != 1) {
            vkDeviceWaitIdle(g_device);
            const std::uint8_t white[4] = {255, 255, 255, 255};
            create_sampled_image(
                albedo_image_, albedo_memory_, albedo_view_, 1, 1);
            upload_sampled_image(albedo_image_, 1, 1, white);
            albedo_width_ = 1;
            albedo_height_ = 1;
            bind_albedo_view(albedo_view_);
        }
        return;
    }
    vkDeviceWaitIdle(g_device);
    create_sampled_image(
        albedo_image_, albedo_memory_, albedo_view_, atlas.width, atlas.height);
    std::vector<std::uint8_t> rgba(
        static_cast<std::size_t>(atlas.width) * atlas.height * 4U);
    const std::size_t pixels =
        static_cast<std::size_t>(atlas.width) * atlas.height;
    for (std::size_t i = 0; i < pixels; ++i) {
        rgba[4U * i] = atlas.pixels[3U * i];
        rgba[4U * i + 1U] = atlas.pixels[3U * i + 1U];
        rgba[4U * i + 2U] = atlas.pixels[3U * i + 2U];
        rgba[4U * i + 3U] = 255;
    }
    upload_sampled_image(
        albedo_image_, atlas.width, atlas.height, rgba.data());
    albedo_width_ = atlas.width;
    albedo_height_ = atlas.height;
    bind_albedo_view(albedo_view_);
}

bool MeshPreviewRenderer::ensure_frames(
    const std::uint32_t width, const std::uint32_t height) {
    if (width_ == width && height_ == height && frames_[0].framebuffer)
        return true;
    destroy_frames();
    if (!ensure_pipeline()) return false;
    width_ = width;
    height_ = height;
    const VkFormat depth_format = mesh_depth_format();
    VkCommandBufferAllocateInfo cmd_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmd_info.commandPool = command_pool_;
    cmd_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_info.commandBufferCount = 1;
    for (Frame& frame : frames_) {
        VkImageCreateInfo color_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        color_info.imageType = VK_IMAGE_TYPE_2D;
        color_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        color_info.extent = {width, height, 1};
        color_info.mipLevels = 1;
        color_info.arrayLayers = 1;
        color_info.samples = VK_SAMPLE_COUNT_1_BIT;
        color_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        color_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_SAMPLED_BIT;
        color_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        check(vkCreateImage(g_device, &color_info, nullptr, &frame.color));
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(g_device, frame.color, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memory_type(
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(
            g_device, &allocation, nullptr, &frame.color_memory));
        check(vkBindImageMemory(g_device, frame.color, frame.color_memory, 0));

        VkImageCreateInfo depth_info = color_info;
        depth_info.format = depth_format;
        depth_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        check(vkCreateImage(g_device, &depth_info, nullptr, &frame.depth));
        vkGetImageMemoryRequirements(g_device, frame.depth, &requirements);
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memory_type(
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(
            g_device, &allocation, nullptr, &frame.depth_memory));
        check(vkBindImageMemory(g_device, frame.depth, frame.depth_memory, 0));

        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        view.image = frame.color;
        view.format = VK_FORMAT_R8G8B8A8_UNORM;
        check(vkCreateImageView(g_device, &view, nullptr, &frame.color_view));
        view.image = frame.depth;
        view.format = depth_format;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        check(vkCreateImageView(g_device, &view, nullptr, &frame.depth_view));

        const VkImageView fb_views[] = {frame.color_view, frame.depth_view};
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fb.renderPass = render_pass_;
        fb.attachmentCount = 2;
        fb.pAttachments = fb_views;
        fb.width = width;
        fb.height = height;
        fb.layers = 1;
        check(vkCreateFramebuffer(g_device, &fb, nullptr, &frame.framebuffer));

        VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.maxLod = 1.F;
        check(vkCreateSampler(g_device, &sampler, nullptr, &frame.sampler));
        frame.descriptor = ImGui_ImplVulkan_AddTexture(
            frame.sampler, frame.color_view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        check(vkAllocateCommandBuffers(g_device, &cmd_info, &frame.command));
        VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        check(vkCreateFence(g_device, &fence, nullptr, &frame.fence));
    }
    return true;
}

void MeshPreviewRenderer::set_mesh(
    const std::vector<float>& positions, const std::vector<float>& normals,
    const std::vector<float>& colours,
    const std::vector<std::uint32_t>& indices,
    const std::vector<float>& uvs) {
    destroy_mesh_buffers();
    const std::size_t vertex_count = positions.size() / 3U;
    if (vertex_count == 0 || indices.size() < 3) return;
    std::vector<MeshGpuVertex> vertices(vertex_count);
    const bool have_n = normals.size() == positions.size();
    const bool have_c = colours.size() == positions.size();
    const bool have_uv = uvs.size() == vertex_count * 2U;
    for (std::size_t i = 0; i < vertex_count; ++i) {
        MeshGpuVertex& v = vertices[i];
        v.px = positions[3U * i];
        v.py = positions[3U * i + 1U];
        v.pz = positions[3U * i + 2U];
        if (have_n) {
            v.nx = normals[3U * i];
            v.ny = normals[3U * i + 1U];
            v.nz = normals[3U * i + 2U];
        } else {
            v.nz = -1.F;
        }
        if (have_c) {
            v.r = colours[3U * i];
            v.g = colours[3U * i + 1U];
            v.b = colours[3U * i + 2U];
        } else {
            v.r = v.g = v.b = 1.F;
        }
        if (have_uv) {
            v.u = uvs[2U * i];
            v.v = uvs[2U * i + 1U];
        }
    }
    std::vector<std::uint32_t> edges;
    edges.reserve(indices.size() * 2U);
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
        const std::uint32_t a = indices[i];
        const std::uint32_t b = indices[i + 1];
        const std::uint32_t c = indices[i + 2];
        edges.push_back(a);
        edges.push_back(b);
        edges.push_back(b);
        edges.push_back(c);
        edges.push_back(c);
        edges.push_back(a);
    }
    create_buffer_with_data(
        vertex_buffer_, vertex_memory_, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        vertices.data(), vertices.size() * sizeof(MeshGpuVertex));
    create_buffer_with_data(
        index_buffer_, index_memory_, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        indices.data(), indices.size() * sizeof(std::uint32_t));
    create_buffer_with_data(
        edge_buffer_, edge_memory_, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        edges.data(), edges.size() * sizeof(std::uint32_t));
    index_count_ = static_cast<std::uint32_t>(indices.size());
    edge_count_ = static_cast<std::uint32_t>(edges.size());
}

bool MeshPreviewRenderer::draw(
    const std::uint32_t width, const std::uint32_t height,
    const MeshPreviewUniforms& uniforms) {
    if (!g_device || index_count_ == 0 || width == 0 || height == 0)
        return false;
    if (!ensure_frames(width, height)) return false;
    Frame& frame = frames_[write_];
    check(vkWaitForFences(g_device, 1, &frame.fence, VK_TRUE, UINT64_MAX));
    check(vkResetFences(g_device, 1, &frame.fence));
    check(vkResetCommandBuffer(frame.command, 0));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(frame.command, &begin));

    VkClearValue clears[2]{};
    clears[0].color = {{
        uniforms.background[0], uniforms.background[1], uniforms.background[2],
        1.F}};
    clears[1].depthStencil = {1.F, 0};
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = render_pass_;
    pass.framebuffer = frame.framebuffer;
    pass.renderArea.extent = {width, height};
    pass.clearValueCount = 2;
    pass.pClearValues = clears;
    vkCmdBeginRenderPass(frame.command, &pass, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{};
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.maxDepth = 1.F;
    VkRect2D scissor{};
    scissor.extent = {width, height};
    vkCmdSetViewport(frame.command, 0, 1, &viewport);
    vkCmdSetScissor(frame.command, 0, 1, &scissor);

    MeshPush push{};
    std::memcpy(push.world_to_clip, uniforms.world_to_clip.data(), 16 * 4);
    const float* w2c = uniforms.world_to_camera.data();
    push.cam0[0] = w2c[0];
    push.cam0[1] = w2c[4];
    push.cam0[2] = w2c[8];
    push.cam1[0] = w2c[1];
    push.cam1[1] = w2c[5];
    push.cam1[2] = w2c[9];
    push.cam2[0] = w2c[2];
    push.cam2[1] = w2c[6];
    push.cam2[2] = w2c[10];
    push.clay_flags[0] = uniforms.clay[0];
    push.clay_flags[1] = uniforms.clay[1];
    push.clay_flags[2] = uniforms.clay[2];
    push.clay_flags[3] = uniforms.textured
        ? 2.F
        : (uniforms.vertex_colour ? 1.F : 0.F);
    vkCmdPushConstants(
        frame.command, pipeline_layout_,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
        sizeof(push), &push);
    if (albedo_set_)
        vkCmdBindDescriptorSets(
            frame.command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_,
            0, 1, &albedo_set_, 0, nullptr);

    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(frame.command, 0, 1, &vertex_buffer_, &offset);
    vkCmdBindIndexBuffer(
        frame.command, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindPipeline(
        frame.command, VK_PIPELINE_BIND_POINT_GRAPHICS, fill_pipeline_);
    vkCmdDrawIndexed(frame.command, index_count_, 1, 0, 0, 0);
    if (uniforms.wireframe && edge_count_ > 0) {
        vkCmdBindPipeline(
            frame.command, VK_PIPELINE_BIND_POINT_GRAPHICS, wire_pipeline_);
        vkCmdBindIndexBuffer(
            frame.command, edge_buffer_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(frame.command, edge_count_, 1, 0, 0, 0);
    }
    vkCmdEndRenderPass(frame.command);
    check(vkEndCommandBuffer(frame.command));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &frame.command;
    check(vkQueueSubmit(g_queue, 1, &submit, frame.fence));
    display_ = write_;
    write_ = (write_ + 1) % k_frames;
    return frame.descriptor != VK_NULL_HANDLE;
}

}  // namespace editor::gpu
