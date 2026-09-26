#pragma once

// Vulkan backend of splat_drender: CUDA-equivalent EWA forward, complete
// differentiable geometry/RGB backward, and multi-view sampling/loss compute
// passes. It also renders a trained model without a CUDA context. Per-Gaussian
// state, sorted instances and pixel snapshots stay alive on the device.

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
    // Optional pre-activation parameters used by the training backward pass.
    // When all three are supplied, gradients are also propagated through
    // exp/filter_3d, quaternion normalization, and sigmoid/determinant scaling.
    std::span<const float> log_scales;
    std::span<const float> raw_rotations;
    std::span<const float> opacity_logits;
    // Per-Gaussian isotropic 3D filter radius. Empty means zero.
    std::span<const float> filter_3d;
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
    float point_depth_bracket = 0.0F;
    float point_depth_tolerance = 0.0F;
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

// Intermediate gradients produced by the alpha-compositing backward pass.
// These are the inputs to the projection/SH parameter backward pass: mean2d
// is [N,3] (xy gradient plus the densification magnitude), conic_opacity is
// [N,4], and colors is [N,3].
struct SplatBlendGradients {
    std::vector<float> mean2d;
    std::vector<float> conic_opacity;
    std::vector<float> colors;
    std::vector<float> ray_plane;
    std::vector<float> normal;
};

// Gradients of every model input consumed by the forward pass. Activated
// gradients match splat_drender::ModelGradients directly. The final three
// arrays are populated when SplatGaussians supplied the corresponding raw
// parameters, and include the complete activation chain used by training.
struct SplatModelGradients {
    std::vector<float> means;
    std::vector<float> sh;
    std::vector<float> colors;
    std::vector<float> opacities;
    std::vector<float> scales;
    std::vector<float> rotations;
    std::vector<float> covariances;
    std::vector<float> log_scales;
    std::vector<float> raw_rotations;
    std::vector<float> opacity_logits;
    // Screen-space projected-mean gradient magnitude used by the shared
    // densification policy.
    std::vector<float> refine_weight;
};

struct SplatDepthSamples {
    std::vector<float> camera_points;  // [P,3]
    std::vector<float> median_depth;   // [P], ray length
    std::vector<std::uint32_t> n_contrib;
    std::vector<std::uint32_t> inside;
};

struct SplatDepthSampleGradients {
    SplatModelGradients model;
    std::vector<float> points;  // [P,3] world-space query-point gradient
};

// Inputs and weights for the GGGS multi-view round-trip and planar-NCC loss.
// Both cameras must be pinhole, matching the CUDA training path. Images are
// channel-major where applicable; sampled_neighbour_points are interleaved
// [P,3] camera-space points returned by sample_depth().
struct SplatMultiViewInput {
    SplatCamera reference_camera;
    SplatCamera neighbour_camera;
    std::span<const float> reference_depth;   // [P]
    std::span<const float> reference_normal;  // [3,P]
    std::span<const float> reference_gray;    // [P]
    std::span<const float> neighbour_gray;    // [Hn,Wn]
    std::span<const float> sampled_neighbour_points;  // [P,3]
    std::span<const std::uint32_t> sampled_inside;    // [P]
    std::span<const float> reference_mask;    // optional [P]
    std::span<const float> neighbour_mask;    // optional [Hn,Wn]
    float geometry_weight = 0.0F;
    float ncc_weight = 0.0F;
    float pixel_noise_threshold = 1.0F;
    bool robust_ncc = true;
    float ncc_lambda_reference = 0.2F;
    float ncc_sharpness = 10.0F;
    float ncc_min_weight = 0.0F;
};

struct SplatMultiViewOutput {
    float geometry = 0.0F;
    float ncc = 0.0F;
    std::uint64_t geometry_pixels = 0;
    std::uint64_t geometry_candidates = 0;
    std::uint64_t ncc_pixels = 0;
    // Already multiplied by geometry_weight / ncc_weight and normalized by
    // their respective accepted-pixel counts, exactly like CUDA.
    std::vector<float> reference_depth_gradient;   // [P]
    std::vector<float> reference_normal_gradient;  // [3,P]
    std::vector<float> sampled_point_gradient;     // [P,3]
};

struct SplatPhotometricOutput {
    float loss = 0.0F;
    std::vector<float> gradient;  // [3,H,W]
};

// Non-owning storage-buffer slice for zero-copy interop (for example with a
// TinyTensor Vulkan tensor). The caller owns the buffer and must synchronize
// any earlier writes before calling the rasterizer.
struct SplatBufferView {
    VkBuffer buffer = VK_NULL_HANDLE;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

// Float32 CHW render attachments that remain in device-local memory. The
// slices are owned by the rasterizer and remain valid until its next render.
// Empty geometry slices mean the render used need_depth=false.
struct SplatDeviceFrame {
    SplatBufferView color;         // [3,H,W]
    SplatBufferView alpha;         // [H,W]
    SplatBufferView normal;        // [3,H,W], optional
    SplatBufferView median_depth;  // [H,W], optional
    SplatBufferView radii;         // [N], int32
    // Raw uint32 contribution flags. A non-zero value means visible.
    SplatBufferView visibility_bits; // [N], uint32
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    int instance_count = 0;
    int visible_count = 0;
};

// Fused L1+SSIM output kept on the Vulkan device. `gradient` is consumed
// directly by backward_device(); `loss_scalar` contains one reduced float and
// is intended for the occasional logging readback only.
struct SplatDevicePhotometricOutput {
    SplatBufferView gradient;     // [3,H,W]
    SplatBufferView loss_map;     // [3,H,W]
    SplatBufferView loss_scalar;  // [1]
};

// Raw trainable Gaussian parameters owned by the caller. This is the Vulkan
// counterpart of photara::splat::GaussianModel: activation (exp scales,
// quaternion normalization and filtered sigmoid opacity) happens in the
// raster shaders, so Adam can update these exact buffers in place.
struct SplatDeviceGaussians {
    SplatBufferView means;           // [N,3]
    SplatBufferView sh;              // [N,B,3]
    SplatBufferView log_scales;      // [N,3]
    SplatBufferView raw_rotations;   // [N,4]
    SplatBufferView opacity_logits;  // [N]
    SplatBufferView filter_3d;       // [N]
    std::uint32_t count = 0;
    std::uint32_t sh_degree = 0;
    std::uint32_t sh_bases = 0;
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
    // Zero-copy training model. The caller retains ownership and must keep all
    // views alive until another model is bound or clear_model() is called.
    void bind_model_device(const SplatDeviceGaussians& gaussians);
    void update_means_and_opacities(
        std::span<const float> means, std::span<const float> opacities);
    void clear_model();
    [[nodiscard]] bool has_model() const noexcept;

    [[nodiscard]] SplatForwardOutput render(
        const SplatCamera& camera, const SplatSettings& settings);
    // Render without downloading color, alpha, depth, normals, visibility or
    // radii. The returned attachments can feed loss/backward passes directly.
    [[nodiscard]] SplatDeviceFrame render_device(
        const SplatCamera& camera, const SplatSettings& settings);
    // Copies selected attachments into caller-owned VkBuffer slices on the
    // adopted device. Empty destination slices are skipped. This is the bridge
    // used by backend-neutral tensor code without staging through the CPU.
    void copy_frame_device(
        const SplatDeviceFrame& source,
        const SplatDeviceFrame& destination);
    // Consumes the most recent render() made with pixel_snapshots=true.
    // Loss images are channel-major: color/normal [3,H,W], alpha/depth [H,W].
    // The geometry losses may be empty only when the matching forward used
    // need_depth=false; otherwise an empty span means a zero upstream gradient.
    [[nodiscard]] SplatBlendGradients backward_blend(
        std::span<const float> dL_color, std::span<const float> dL_alpha,
        std::span<const float> dL_median_depth = {},
        std::span<const float> dL_normal = {});
    // Full RGB/alpha/depth/normal backward through compositing,
    // projection/covariance, SH, and the optional model activation chain.
    [[nodiscard]] SplatModelGradients backward(
        std::span<const float> dL_color, std::span<const float> dL_alpha,
        std::span<const float> dL_median_depth = {},
        std::span<const float> dL_normal = {});
    // Device-resident form of backward_blend. `packed_gradients` contains
    // [mean2d:3*N, conic_opacity:4*N, colors:3*N, ray_plane:4*N, normal:3*N]
    // Float32 values and is zeroed before accumulation. All buffers must belong
    // to this Context's VkDevice.
    void backward_blend_device(
        const SplatBufferView& dL_color, const SplatBufferView& dL_alpha,
        const SplatBufferView& packed_gradients,
        const SplatBufferView& dL_median_depth = {},
        const SplatBufferView& dL_normal = {});
    [[nodiscard]] std::uint64_t blend_gradient_float_count() const noexcept;
    // Fully device-resident color/alpha backward. The output layout is
    // [means, SH-or-colors, opacities, scales, rotations, covariances,
    //  log_scales, raw_rotations, opacity_logits, refine_weight]. Slots that do not apply to
    // the uploaded representation remain zero, preserving one stable layout.
    void backward_device(
        const SplatBufferView& dL_color, const SplatBufferView& dL_alpha,
        const SplatBufferView& packed_model_gradients,
        const SplatBufferView& dL_median_depth = {},
        const SplatBufferView& dL_normal = {});
    [[nodiscard]] std::uint64_t model_gradient_float_count() const noexcept;
    // GGGS multi-view median-depth query and its complete backward pass.
    // sample_depth_backward consumes the most recent sample_depth call.
    [[nodiscard]] SplatDepthSamples sample_depth(
        std::span<const float> world_points, const SplatCamera& camera,
        const SplatSettings& settings = {});
    [[nodiscard]] SplatDepthSampleGradients sample_depth_backward(
        std::span<const float> dL_camera_points);
    // CUDA-equivalent multi-view loss stage. Feed sampled_point_gradient to
    // sample_depth_backward(); add that result's point gradient through the
    // reference unprojection, and feed the reference gradients to backward().
    [[nodiscard]] SplatMultiViewOutput multi_view_loss(
        const SplatMultiViewInput& input);
    // Host parity/debug adapter. It uploads both images and downloads the
    // gradient, so training code must use photara/splat/photometric_loss and
    // fused_l1_ssim_device() instead. Statistics use the CUDA-compatible
    // zero-padded 11x11 Gaussian window and valid 5-pixel reduction crop.
    [[nodiscard]] SplatPhotometricOutput photometric_loss(
        std::span<const float> prediction, std::span<const float> target,
        std::uint32_t width, std::uint32_t height, float ssim_weight = 0.2F,
        float photometric_weight = 1.0F,
        std::span<const float> mask = {});
    // Low-level device primitive used by photara/splat/photometric_loss. All
    // views must belong to this Context's VkDevice. No image or gradient is
    // copied to the host.
    [[nodiscard]] SplatDevicePhotometricOutput fused_l1_ssim_device(
        const SplatBufferView& prediction, const SplatBufferView& target,
        std::uint32_t width, std::uint32_t height, float ssim_weight = 0.2F,
        float photometric_weight = 1.0F,
        const SplatBufferView& mask = {});
    // Reads exactly one float from the most recent device loss result.
    [[nodiscard]] float read_photometric_loss(
        const SplatDevicePhotometricOutput& output);
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
