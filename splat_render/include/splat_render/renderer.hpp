#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace splat_render {

// OpenCV camera, column-major world-to-camera, matching photara::splat::Camera.
// model matches the inspector combo: Perspective, Orthographic, Fisheye, Panorama.
// Orthographic fx/fy are pixels per world unit, not a focal length.
inline constexpr std::uint32_t k_camera_pinhole = 0;
inline constexpr std::uint32_t k_camera_orthographic = 1;
inline constexpr std::uint32_t k_camera_fisheye = 2;
inline constexpr std::uint32_t k_camera_equirectangular = 3;
struct Camera {
    std::array<float, 16> world_to_camera{};
    std::array<float, 3> position{};
    float fx{1.F};
    float fy{1.F};
    float cx{};
    float cy{};
    float k1{};
    float k2{};
    float k3{};
    float k4{};
    std::uint32_t width{1};
    std::uint32_t height{1};
    std::uint32_t model{};
};

// Host Gaussian attributes. Scales are log-space and opacity is a logit.
// Quaternions are scalar-first. sh is [count, sh_bases, 3] in INRIA order.
struct GaussianCloud {
    std::uint32_t count{};
    std::uint32_t sh_degree{};
    std::uint32_t sh_bases{1};
    const float* means{};
    const float* log_scales{};
    const float* quaternions{};
    const float* opacity_logits{};
    const float* sh{};
};

struct FrameTarget {
    int slot{-1};
    VkImageView view{};
    VkSampler sampler{};
    VkImageLayout layout{VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    std::uint32_t width{};
    std::uint32_t height{};
};

struct Device {
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    std::uint32_t queue_family{VK_QUEUE_FAMILY_IGNORED};
};

// Vulkan 3DGUT forward rasterizer. Attributes stay in storage buffers so a
// later editor can change them and redraw without a CUDA preview process.
class Renderer {
public:
    Renderer() = default;
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    ~Renderer() { reset(); }

    void attach(const Device& device);
    [[nodiscard]] bool attached() const noexcept { return device_ != VK_NULL_HANDLE; }

    bool upload(const GaussianCloud& cloud, std::string_view source_key);
    // Replaces world positions. `centers` is xyzw, and w is the activated opacity.
    bool update_centers(const float* centers, std::uint32_t count);
    void clear_model();

    // Sorts back-to-front and draws. An unchanged camera reuses the last image.
    bool draw(const Camera& camera, FrameTarget& target);
    bool download_rgb(
        std::vector<std::uint8_t>& rgb, std::uint32_t& width,
        std::uint32_t& height, std::uint32_t max_long_edge = 0);

    [[nodiscard]] const std::string& source_key() const noexcept { return source_key_; }
    [[nodiscard]] const std::string& failure() const noexcept { return failure_; }
    [[nodiscard]] std::uint32_t splat_count() const noexcept { return count_; }
    [[nodiscard]] bool supported() const noexcept { return supported_; }
    [[nodiscard]] std::uint64_t frames_epoch() const noexcept { return frames_epoch_; }
    void reset();

private:
    struct Storage {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        VkDeviceSize size{};
    };
    struct Frame {
        VkImage color{};
        VkDeviceMemory memory{};
        VkImageView view{};
        VkFramebuffer framebuffer{};
        VkSampler sampler{};
        std::uint32_t width{};
        std::uint32_t height{};
    };
    struct FrameData {
        float world_to_camera[16]{};
        float eye[4]{};
        float intrinsics[4]{};
        float distortion[4]{};
        float viewport[4]{};
        std::uint32_t sh_degree{};
        std::uint32_t count{};
        std::uint32_t bases{};
        std::uint32_t reserved{};
    };

    bool ensure_device();
    bool ensure_frames(std::uint32_t width, std::uint32_t height);
    void destroy_storage(Storage& storage);
    void destroy_frames();
    void destroy_device_objects();
    void create_storage(
        Storage& storage, VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags memory);
    void upload_storage(Storage& storage, const void* data, VkDeviceSize size);
    void write_descriptors();
    void wait_gpu();
    std::uint32_t memory_type(std::uint32_t bits, VkMemoryPropertyFlags flags) const;

    VkPhysicalDevice physical_{};
    VkDevice device_{};
    VkQueue queue_{};
    std::uint32_t family_{VK_QUEUE_FAMILY_IGNORED};

    std::string source_key_;
    std::string failure_;
    bool supported_{true};
    bool ready_{};
    std::uint32_t count_{};
    std::uint32_t sh_degree_{};
    std::uint32_t sh_bases_{1};
    std::uint64_t generation_{1};
    std::uint64_t frames_epoch_{1};

    VkDescriptorSetLayout set_layout_{};
    VkPipelineLayout pipeline_layout_{};
    VkDescriptorPool pool_{};
    VkDescriptorSet set_{};
    VkRenderPass render_pass_{};
    VkPipeline keys_pipeline_{};
    VkPipeline hist_pipeline_{};
    VkPipeline scan_pipeline_{};
    VkPipeline scatter_pipeline_{};
    VkPipeline prepare_pipeline_{};
    VkPipeline draw_pipeline_{};
    VkCommandPool command_pool_{};
    VkCommandBuffer command_{};
    VkFence fence_{};
    void* frame_mapped_{};
    Storage frame_ubo_{};
    Storage centers_{};
    Storage scales_{};
    Storage rotations_{};
    Storage harmonics_{};
    Storage quads_{};
    Storage keys0_{};
    Storage keys1_{};
    Storage vals0_{};
    Storage vals1_{};
    Storage wg_counts_{};
    Storage wg_offsets_{};
    Storage ranks_{};
    Storage totals_{};
    Storage bin_start_{};
    Storage dummy_{};

    static constexpr int k_frames = 3;
    Frame frames_[k_frames]{};
    int write_{};
    int display_{-1};
    FrameData cached_{};
    std::uint64_t cached_generation_{};
    bool cache_valid_{};
};

}  // namespace splat_render
