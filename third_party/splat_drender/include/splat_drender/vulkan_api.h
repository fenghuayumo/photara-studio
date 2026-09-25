#pragma once

// Vulkan backend of splat_drender: the same EWA forward the CUDA backend
// implements, expressed as Vulkan compute passes. It renders a trained model
// without a CUDA context, which is what the editor preview needs, and it is the
// base for the differentiable Vulkan pass that follows: the per-Gaussian state,
// the sorted instances and the pixel snapshots of the last forward() all stay
// alive on the device.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace splat_drender {
namespace vulkan {

// Column-major world-to-camera, matching splat_drender::CameraView.
// mode: 0 pinhole, 1 OpenCV fisheye (k1..k4), 3 equirectangular,
// 4 orthographic (fx/fy are pixels per world unit).
struct SplatCamera {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    float fx = 1.0F;
    float fy = 1.0F;
    float cx = 0.0F;
    float cy = 0.0F;
    std::uint32_t mode = 0;
    float k1 = 0.0F;
    float k2 = 0.0F;
    float k3 = 0.0F;
    float k4 = 0.0F;
    std::span<const float> world_to_camera;
    std::span<const float> center;
};

// Activated opacity, activated scales, and scalar-first quaternions.
// Exactly one of sh / colors, and exactly one of scales / covariances.
struct SplatGaussians {
    std::span<const float> means;
    std::span<const float> sh;
    std::span<const float> colors;
    std::span<const float> opacities;
    std::span<const float> scales;
    std::span<const float> rotations;
    std::span<const float> covariances;
    std::uint32_t sh_degree = 0;
    std::uint32_t sh_bases = 0;
};

struct SplatSettings {
    float background[3] = {0.0F, 0.0F, 0.0F};
    float scale_modifier = 1.0F;
    float kernel_size = 0.0F;
    bool need_depth = true;
    // Keeps the per-pixel bucket snapshots the backward pass consumes. A
    // forward-only caller (the editor preview) leaves it off: at a million
    // instances the buffer is hundreds of megabytes of unused writes.
    bool pixel_snapshots = false;
};

// Channel-major images: color and normal are [3, H, W], alpha and median depth are [H, W].
// median_depth and normal are empty when settings.need_depth is false.
struct SplatForwardOutput {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    int instance_count = 0;
    int visible_count = 0;
    std::vector<float> color;
    std::vector<float> alpha;
    std::vector<float> median_depth;
    std::vector<float> normal;
    std::vector<float> visibility;
    std::vector<int> radii;
};

// 8-bit RGBA frame that stayed on the device. The buffer is owned by the
// rasterizer and stays valid until the next render call; copy it into an image
// on the same queue, or read it back through the context's staging path.
struct SplatRgbaImage {
    VkBuffer buffer = VK_NULL_HANDLE;
    std::uint64_t bytes = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

// Device a Context runs on. Set this to adopt a device the caller already owns
// (the editor's, for example) instead of letting the Context create a private
// instance and device: the two then share one VkDevice, so a later pass can
// hand its result over without leaving the GPU. `instance` and
// `physical_device` are only used for reporting; `queue_family` must be the
// family `queue` was created from and must support compute. An adopted device
// is never destroyed by the Context.
struct ExternalDevice {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = VK_QUEUE_FAMILY_IGNORED;

    [[nodiscard]] bool valid() const noexcept { return device != VK_NULL_HANDLE; }
};

struct ContextOptions {
    std::uint32_t device_index = 0;
    bool enable_validation = false;
    // When valid(), this replaces instance/device creation entirely.
    ExternalDevice external_device{};
};

struct DeviceInfo {
    std::string name;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t api_version = 0;
};

// Owns the Vulkan instance, device, queue and descriptor pool the Vulkan
// backend runs on. Create one, then build rasterizers against it.
class Context {
public:
    class Impl;

    explicit Context(const ContextOptions& options = {});
    ~Context();

    Context(Context&&) noexcept;
    Context& operator=(Context&&) noexcept;
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    [[nodiscard]] const DeviceInfo& device_info() const noexcept;
    [[nodiscard]] static std::vector<DeviceInfo> enumerate_devices();

    // Handles every rasterizer built on this Context submits to. They are the
    // adopted ones when ContextOptions::external_device was set.
    [[nodiscard]] VkInstance instance() const noexcept;
    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept;
    [[nodiscard]] VkDevice device() const noexcept;
    [[nodiscard]] VkQueue queue() const noexcept;
    [[nodiscard]] std::uint32_t queue_family() const noexcept;

private:
    std::unique_ptr<Impl> impl_;

    friend class SplatRasterizer;
};

// Vulkan EWA forward rasterizer. The per-Gaussian state, sorted instances, and
// pixel snapshots from the latest forward() stay alive for a later backward pass.
class SplatRasterizer {
public:
    explicit SplatRasterizer(Context& context);
    ~SplatRasterizer();

    SplatRasterizer(SplatRasterizer&&) noexcept;
    SplatRasterizer& operator=(SplatRasterizer&&) noexcept;
    SplatRasterizer(const SplatRasterizer&) = delete;
    SplatRasterizer& operator=(const SplatRasterizer&) = delete;

    [[nodiscard]] SplatForwardOutput forward(
        const SplatGaussians& gaussians,
        const SplatCamera& camera,
        const SplatSettings& settings);

    // Keeps the activated Gaussian attributes on the device. render() and
    // render_rgba() reuse them until the next upload or clear.
    void upload_model(const SplatGaussians& gaussians);
    void update_means_and_opacities(
        std::span<const float> means, std::span<const float> opacities);
    void clear_model();
    [[nodiscard]] bool has_model() const noexcept;

    [[nodiscard]] SplatForwardOutput render(
        const SplatCamera& camera, const SplatSettings& settings);
    // Same frame as render() with need_depth off, but the color stays on the
    // device as RGBA8: no float readback and no host-side quantization.
    [[nodiscard]] SplatRgbaImage render_rgba_device(
        const SplatCamera& camera, const SplatSettings& settings);
    void render_rgba(
        const SplatCamera& camera, const SplatSettings& settings,
        std::vector<std::uint8_t>& rgba);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vulkan
}  // namespace splat_drender
