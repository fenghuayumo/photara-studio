#pragma once

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif

#include "io/image.hpp"

#include "imgui.h"
#include "imgui_impl_vulkan.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <vector>

namespace editor::gpu {

// The swapchain and the ImGui Vulkan backend must agree on this value.
constexpr std::uint32_t k_min_images = 2;

void check(VkResult result);

// Instance, device, queue and descriptor pool. `extensions` carries the
// platform surface extensions requested by GLFW.
void create_context(ImVector<const char*> extensions);
void create_window(VkSurfaceKHR surface, int width, int height);
void resize_window(int width, int height);
void destroy_context();

VkInstance instance();
VkPhysicalDevice physical_device();
VkDevice device();
std::uint32_t queue_family();
VkQueue queue();
VkDescriptorPool descriptor_pool();
ImGui_ImplVulkanH_Window& window();

// Win32 LUID / node mask of the presenting adapter, forwarded to the trainer so
// CUDA opens the shared allocation on the same GPU.
std::uint64_t device_luid();
std::uint32_t device_node_mask();

bool swapchain_needs_rebuild();
void clear_swapchain_rebuild();
void present(ImDrawData* draw, const ImVec4& clear_colour);

std::uint32_t memory_type(std::uint32_t bits, VkMemoryPropertyFlags flags);

// A sampled RGBA8 image plus its ImGui descriptor.
struct PreviewTexture {
    VkImage image{};
    VkDeviceMemory memory{};
    VkImageView view{};
    VkSampler sampler{};
    VkDescriptorSet descriptor{};
    std::uint32_t width{};
    std::uint32_t height{};

    void reset();
    void upload(const aetherscan::io::RgbImage& source);
    // Copies the sampled image back to host RGB. Used by the 2D QA compare path.
    bool download_rgb(aetherscan::io::RgbImage& destination) const;
};

// Lazy GPU thumbnails for sparse-viewport camera frustums. `clear()` must run
// while the ImGui Vulkan backend is still alive.
struct CameraPhotoCache {
    CameraPhotoCache() = default;
    CameraPhotoCache(const CameraPhotoCache&) = delete;
    CameraPhotoCache& operator=(const CameraPhotoCache&) = delete;

    void clear();
    void resize(std::size_t view_count);
    void request(std::size_t index, const std::filesystem::path& path);
    void poll();
    [[nodiscard]] ImTextureID id(std::size_t index) const;
    [[nodiscard]] const ImTextureID* ids() const {
        return ids_.empty() ? nullptr : ids_.data();
    }
    [[nodiscard]] std::size_t size() const { return ids_.size(); }

private:
    struct Slot {
        PreviewTexture texture;
        std::future<aetherscan::io::RgbImage> pending;
        std::filesystem::path path;
        bool loading{};
        bool failed{};
    };

    [[nodiscard]] std::size_t inflight() const;

    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<ImTextureID> ids_;
};

// The CUDA-written image is exported as a Win32 handle and copied into a
// sampled display image each frame, synchronised by a shared timeline
// semaphore. The trainer signals odd values; the editor signals even ones.
struct ExternalPreview {
    PreviewTexture display;
    VkImage image{};
    VkDeviceMemory memory{};
    VkSemaphore timeline{};
#if defined(_WIN32)
    HANDLE memory_handle{};
    HANDLE semaphore_handle{};
#else
    void* memory_handle{};
    void* semaphore_handle{};
#endif
    VkDeviceSize allocation_size{};
    std::uint32_t width{};
    std::uint32_t height{};

    void create(std::uint32_t image_width, std::uint32_t image_height);
    void renew_export_handles();
    void close_export_handles();
    void poll();
    // Keeps training unblocked while the swapchain is minimised: consume and
    // copy the pending frame without acquiring a WSI image.
    void consume_without_present();
    void reset();
};

std::uint64_t consumed_timeline_value();
// Preview frames copied out of the shared CUDA/Vulkan image so far. The editor
// publishes this to the trainer's acknowledgement sidecar.
std::uint64_t copied_preview_frames();
// Timeline value of a trainer frame that is signalled but not yet consumed.
// Zero when nothing is pending.
std::uint64_t ready_timeline_value();

// Offscreen triangle rasterizer on the editor device. Orbit only updates a
// 128-byte push constant; the mesh stays in GPU buffers (no CPU readback).
struct MeshPreviewUniforms {
    std::array<float, 16> world_to_clip{};
    std::array<float, 16> world_to_camera{};
    std::array<float, 3> clay{0.77F, 0.73F, 0.68F};
    std::array<float, 3> background{0.027F, 0.031F, 0.043F};
    bool vertex_colour{};
    bool textured{};
    bool wireframe{};
};

class MeshPreviewRenderer {
public:
    MeshPreviewRenderer() = default;
    MeshPreviewRenderer(const MeshPreviewRenderer&) = delete;
    MeshPreviewRenderer& operator=(const MeshPreviewRenderer&) = delete;
    ~MeshPreviewRenderer() { reset(); }

    void set_mesh(
        const std::vector<float>& positions,
        const std::vector<float>& normals,
        const std::vector<float>& colours,
        const std::vector<std::uint32_t>& indices,
        const std::vector<float>& uvs = {});
    void set_albedo(const aetherscan::io::RgbImage& atlas);
    bool draw(
        std::uint32_t width, std::uint32_t height,
        const MeshPreviewUniforms& uniforms);
    [[nodiscard]] VkDescriptorSet descriptor() const noexcept {
        return display_ >= 0 ? frames_[static_cast<std::size_t>(display_)].descriptor
                             : VK_NULL_HANDLE;
    }
    void reset();

private:
    static constexpr int k_frames = 3;

    struct Frame {
        VkImage color{};
        VkImage depth{};
        VkDeviceMemory color_memory{};
        VkDeviceMemory depth_memory{};
        VkImageView color_view{};
        VkImageView depth_view{};
        VkFramebuffer framebuffer{};
        VkSampler sampler{};
        VkDescriptorSet descriptor{};
        VkCommandBuffer command{};
        VkFence fence{};
    };

    void destroy_mesh_buffers();
    void destroy_frames();
    void destroy_pipeline();
    void destroy_albedo();
    bool ensure_pipeline();
    bool ensure_albedo();
    bool ensure_frames(std::uint32_t width, std::uint32_t height);
    void bind_albedo_view(VkImageView view);

    VkRenderPass render_pass_{};
    VkPipelineLayout pipeline_layout_{};
    VkPipeline fill_pipeline_{};
    VkPipeline wire_pipeline_{};
    VkCommandPool command_pool_{};
    VkDescriptorSetLayout albedo_layout_{};
    VkDescriptorPool albedo_pool_{};
    VkDescriptorSet albedo_set_{};
    VkSampler albedo_sampler_{};
    VkImage albedo_image_{};
    VkDeviceMemory albedo_memory_{};
    VkImageView albedo_view_{};
    std::uint32_t albedo_width_{};
    std::uint32_t albedo_height_{};
    VkBuffer vertex_buffer_{};
    VkBuffer index_buffer_{};
    VkBuffer edge_buffer_{};
    VkDeviceMemory vertex_memory_{};
    VkDeviceMemory index_memory_{};
    VkDeviceMemory edge_memory_{};
    std::uint32_t index_count_{};
    std::uint32_t edge_count_{};
    std::uint32_t width_{};
    std::uint32_t height_{};
    int write_{};
    int display_{-1};
    Frame frames_[k_frames]{};
};

}  // namespace editor::gpu
