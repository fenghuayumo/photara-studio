#include "features/vulkan_features.hpp"
#include "features/registry.hpp"
#include "vk_context.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace photara::features {
namespace {

using vulkan_backend::BufferBinding;
using vulkan_backend::BufferRef;
using vulkan_backend::Context;
using vulkan_backend::Shader;

constexpr std::uint32_t kFilterWidthFactor = 4;
constexpr std::uint32_t kKernelMaxWidth = 33;
constexpr std::uint32_t kKernelMinWidth = 5;
constexpr double kTwoPi = 6.283185307179586476925286766559;


BufferBinding binding(const BufferRef& buffer) {
    return BufferBinding{buffer->handle(), 0, VK_WHOLE_SIZE};
}

struct ConvertPush {
    std::uint32_t pixels;
    std::uint32_t source_stride;
    std::uint32_t destination_stride;
    std::uint32_t pad0;
};

struct UpsamplePush {
    std::uint32_t src_width;
    std::uint32_t src_height;
    std::uint32_t dst_width;
    std::uint32_t pad0;
};

struct DownsamplePush {
    std::uint32_t src_width;
    std::uint32_t dst_width;
    std::uint32_t scale;
    std::uint32_t pad0;
};

struct BlurPush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t radius;
    std::uint32_t direction;
};

struct DogPush {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t write_gradient;
    std::uint32_t pad0;
};

struct KeyPush {
    std::uint32_t width;
    std::uint32_t height;
    float dog_threshold0;
    float dog_threshold;
    float edge_threshold;
    std::uint32_t pad0;
};

struct HistInitPush {
    std::uint32_t key_width;
    std::uint32_t key_height;
    std::uint32_t hist_width;
    std::uint32_t pad0;
};

struct HistReducePush {
    std::uint32_t in_width;
    std::uint32_t out_width;
    std::uint32_t out_height;
    std::uint32_t pad0;
};

struct ListGenPush {
    std::uint32_t list_length;
    std::uint32_t hist_width;
};

struct OrientationPush {
    std::uint32_t list_length;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t num_orientation;
    float sigma;
    float sigma_step;
    float gaussian_factor;
    float sample_factor;
};

struct DescriptorPush {
    std::uint32_t count;
    std::uint32_t width;
    std::uint32_t height;
    float window_factor;
};

struct DescriptorNormPush {
    std::uint32_t count;
    std::uint32_t pad0;
};

// SiftGPU's CreateFilterKernel.
std::vector<float> gaussian_kernel(float sigma) {
    const auto radius = static_cast<int>(
        std::ceil(static_cast<float>(kFilterWidthFactor) * sigma - 0.5F));
    int width = 2 * radius + 1;
    if (width > static_cast<int>(kKernelMaxWidth)) width = kKernelMaxWidth;
    if (width < static_cast<int>(kKernelMinWidth)) width = kKernelMinWidth;
    std::vector<float> kernel(static_cast<std::size_t>(width));
    const int half = width / 2;
    float sum = 0.0F;
    for (int i = 0; i <= half; ++i) {
        const float value =
            std::exp(-0.5F * static_cast<float>(i * i) / (sigma * sigma));
        kernel[static_cast<std::size_t>(half - i)] = value;
        kernel[static_cast<std::size_t>(half + i)] = value;
        sum += i == 0 ? value : 2.0F * value;
    }
    const float inverse = 1.0F / sum;
    for (auto& value : kernel) value *= inverse;
    return kernel;
}

std::uint32_t aligned_width(std::uint32_t width) { return (width + 3U) / 4U * 4U; }

}  // namespace

class SiftVulkanExtractor::Impl {
   public:
    struct Workspace {
        std::mutex mutex;
        std::array<BufferRef, 3> gaussian, dog;
        BufferRef input_bytes, input_float, scratch, carry, keys, gradient;
        BufferRef kernel, histogram_readback, list, descriptor_lists;
        BufferRef descriptors, descriptor_readback;
        std::vector<BufferRef> histogram;
    };
    static std::shared_ptr<Workspace> shared_workspace(bool create = true) {
        static std::mutex mutex;
        static std::weak_ptr<Workspace> cached;
        std::lock_guard lock(mutex);
        auto workspace = cached.lock();
        if (!workspace && create) {
            workspace = std::make_shared<Workspace>();
            cached = workspace;
        }
        return workspace;
    }
    explicit Impl(SiftVulkanOptions value) : options(std::move(value)) {
        if (options.maximum_features == 0 || options.maximum_image_dimension == 0 ||
            options.maximum_orientations > 4 || options.octave_layers == 0 ||
            options.octave_layers > 16)
            throw std::invalid_argument("Invalid Vulkan SIFT limits");
        if (options.first_octave < -1 || options.first_octave > 4)
            throw std::invalid_argument("Vulkan SIFT first_octave must be in [-1, 4]");

        dog_level_num = static_cast<int>(options.octave_layers);
        level_min = -1;
        level_max = dog_level_num + 1;
        level_num = level_max - level_min + 1;
        sigma0 = 1.6F * std::pow(2.0F, 1.0F / static_cast<float>(dog_level_num));
        sigman = 0.5F;
        level_ds = std::min(level_min + dog_level_num, level_max);
        sigmak = std::pow(2.0F, 1.0F / static_cast<float>(dog_level_num));
        sigma_step = sigmak;
        dsigma0 = sigma0 * std::sqrt(1.0F - 1.0F / (sigmak * sigmak));

        level_sigma.resize(static_cast<std::size_t>(level_num));
        for (int level = level_min; level <= level_max; ++level)
            level_sigma[static_cast<std::size_t>(level - level_min)] =
                sigma0 *
                std::pow(2.0F, static_cast<float>(level) / static_cast<float>(dog_level_num));
        sigma_inc.resize(static_cast<std::size_t>(level_num - 1));
        for (int level = level_min + 1; level <= level_max; ++level)
            sigma_inc[static_cast<std::size_t>(level - level_min - 1)] =
                dsigma0 * std::pow(sigmak, static_cast<float>(level));
    }

    SiftVulkanOptions options;
    std::shared_ptr<Workspace> workspace{shared_workspace()};
    Context* context{};

    int dog_level_num{};
    int level_min{};
    int level_max{};
    int level_num{};
    int level_ds{};
    float sigma0{};
    float sigman{};
    float sigmak{};
    float sigma_step{};
    float dsigma0{};
    std::vector<float> level_sigma;
    std::vector<float> sigma_inc;

    [[nodiscard]] float initial_smooth_sigma(int octave_min) const {
        const float sa = sigma0 * std::pow(2.0F, static_cast<float>(level_min) /
                                                     static_cast<float>(dog_level_num));
        const float sb = sigman / std::pow(2.0F, static_cast<float>(octave_min));
        return sa > sb + 0.001F ? std::sqrt(sa * sa - sb * sb) : 0.0F;
    }

    [[nodiscard]] float skip_sigma() const {
        const float sa = sigma0 * std::pow(sigmak, static_cast<float>(level_min));
        const float sb = sigma0 * std::pow(sigmak, static_cast<float>(level_ds - dog_level_num));
        return sa > sb + 0.001F ? std::sqrt(sa * sa - sb * sb) : 0.0F;
    }

    struct Plan {
        int octave_min{};
        int octave_num{};
        std::uint32_t first_width{};
        std::uint32_t first_height{};
    };

    [[nodiscard]] Plan plan(std::uint32_t width, std::uint32_t height) const {
        Plan result;
        result.octave_min = options.first_octave;
        std::uint32_t working_width = width;
        std::uint32_t working_height = height;
        if (result.octave_min < 0) {
            working_width <<= static_cast<std::uint32_t>(-result.octave_min);
            working_height <<= static_cast<std::uint32_t>(-result.octave_min);
        } else {
            working_width >>= static_cast<std::uint32_t>(result.octave_min);
            working_height >>= static_cast<std::uint32_t>(result.octave_min);
        }
        const std::uint32_t maximum_dimension =
            options.maximum_image_dimension *
            (1U << static_cast<std::uint32_t>(std::max(0, -options.first_octave)));
        while (working_width > maximum_dimension || working_height > maximum_dimension) {
            ++result.octave_min;
            working_width = std::max(1U, working_width >> 1);
            working_height = std::max(1U, working_height >> 1);
        }
        const double smallest = static_cast<double>(std::min(working_width, working_height));
        result.octave_num = std::max(1, static_cast<int>(std::floor(std::log2(smallest))) - 3);
        result.first_width = working_width;
        result.first_height = working_height;
        return result;
    }
};

SiftVulkanExtractor::SiftVulkanExtractor(SiftVulkanOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {
    impl_->context = Context::available() ? &Context::get() : nullptr;
}

SiftVulkanExtractor::~SiftVulkanExtractor() = default;
SiftVulkanExtractor::SiftVulkanExtractor(SiftVulkanExtractor&&) noexcept = default;
SiftVulkanExtractor& SiftVulkanExtractor::operator=(SiftVulkanExtractor&&) noexcept = default;

bool SiftVulkanExtractor::is_built() noexcept { return Context::available(); }

bool SiftVulkanExtractor::is_available() const noexcept {
    return impl_ != nullptr && impl_->context != nullptr;
}

ExtractorInfo SiftVulkanExtractor::info() const {
    ExtractorInfo value;
    value.thread_safe = false;
    value.thread_affine = false;
    value.accepts_gray = true;
    value.accepts_rgb = false;
    value.metric = impl_->options.root_sift ? DescriptorMetric::l2_root : DescriptorMetric::l2;
    value.typical_descriptor_dimension = 128;
    return value;
}

std::unique_ptr<FeatureExtractor> SiftVulkanExtractor::clone() const {
    return std::make_unique<SiftVulkanExtractor>(impl_->options);
}

void SiftVulkanExtractor::release_working_memory() {
    auto workspace = Impl::shared_workspace(false);
    if (!workspace) return;
    std::lock_guard lock(workspace->mutex);
    workspace->gaussian = {};
    workspace->dog = {};
    workspace->input_bytes = {};
    workspace->input_float = {};
    workspace->scratch = {};
    workspace->carry = {};
    workspace->keys = {};
    workspace->gradient = {};
    workspace->kernel = {};
    workspace->histogram_readback = {};
    workspace->histogram.clear();
    workspace->list = {};
    workspace->descriptor_lists = {};
    workspace->descriptors = {};
    workspace->descriptor_readback = {};
}

FeatureSet SiftVulkanExtractor::extract_gray(const std::span<const std::uint8_t> pixels,
                                             const std::uint32_t width, const std::uint32_t height,
                                             std::size_t row_stride) const {
    if (impl_ == nullptr || impl_->context == nullptr)
        throw std::runtime_error("Vulkan SIFT backend is unavailable");
    if (row_stride == 0) row_stride = width;
    if (width < 8 || height < 8 || row_stride < width ||
        pixels.size() < row_stride * static_cast<std::size_t>(height))
        throw std::invalid_argument("Invalid grayscale image view");

    Impl& impl = *impl_;
    Context& context = *impl.context;

    std::vector<std::uint8_t> contiguous;
    const std::uint8_t* data = pixels.data();
    if (row_stride != width) {
        contiguous.resize(static_cast<std::size_t>(width) * height);
        for (std::uint32_t row = 0; row < height; ++row) {
            std::copy_n(pixels.data() + static_cast<std::size_t>(row) * row_stride, width,
                        contiguous.data() + static_cast<std::size_t>(row) * width);
        }
        data = contiguous.data();
    }

    // SiftGPU truncates the input width to a multiple of four before building
    // the pyramid (TruncateWidthCU) and compacts each row; the trailing columns
    // never contribute. Keeping the same geometry also guarantees that the
    // upsampled octave has exactly twice the source row stride, which the raw
    // linear reads of the upsample pass rely on.
    const std::uint32_t effective_width = std::max(4U, width & ~3U);
    const Impl::Plan plan = impl.plan(effective_width, height);
    const int octave_num = plan.octave_num;
    const std::size_t level_count = static_cast<std::size_t>(impl.level_num);
    const std::size_t dog_level_count = static_cast<std::size_t>(impl.dog_level_num);
    const std::size_t level_total = static_cast<std::size_t>(octave_num) * dog_level_count;

    struct OctaveShape {
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t pixels;
    };
    std::vector<OctaveShape> shapes(static_cast<std::size_t>(octave_num));
    {
        std::uint32_t w = plan.first_width;
        std::uint32_t h = plan.first_height;
        for (int octave = 0; octave < octave_num; ++octave) {
            const std::uint32_t aligned = aligned_width(w);
            shapes[static_cast<std::size_t>(octave)] = OctaveShape{aligned, h, aligned * h};
            w = std::max(1U, w >> 1);
            h = std::max(1U, h >> 1);
        }
    }

    // The queue consumes one bounded workspace shared by extractor clones.
    // Keep only a three-level Gaussian/DoG window, one key image and one
    // gradient image; compact oriented keypoints onto the host per level.
    auto& work = *impl.workspace;
    std::unique_lock workspace_lock(work.mutex);
    const auto& largest = shapes.front();
    const auto rounded = [](std::uint64_t bytes) { return Context::allocation_size(bytes); };
    std::uint64_t estimated_bytes = rounded(std::uint64_t(largest.pixels) * 4) * 8 +
                                    rounded(std::uint64_t(largest.pixels) * 16) +
                                    rounded(std::uint64_t(largest.pixels) * 8) +
                                    rounded(std::uint64_t(effective_width) * height * 4);
    for (std::uint32_t w = (largest.width + 2U) >> 2U;; w = std::max(1U, (w + 3U) >> 2U)) {
        estimated_bytes += rounded(std::uint64_t(w) * largest.height * 16);
        if (w == 1) break;
    }
    context.acquire_device_budget(estimated_bytes);
    struct BudgetRelease {
        Context& context;
        std::uint64_t bytes;
        ~BudgetRelease() { context.release_device_budget(bytes); }
    } budget_release{context, estimated_bytes};
    const auto ensure = [&](BufferRef& buffer, VkDeviceSize bytes, bool host, bool cached = false) {
        if (!buffer.valid() || buffer->size_bytes() < bytes) {
            buffer = {};
            buffer = context.allocate(bytes, host, cached);
        }
    };
    const VkDeviceSize plane_bytes = VkDeviceSize(largest.pixels) * 4;
    for (auto& buffer : work.gaussian) ensure(buffer, plane_bytes, false);
    for (auto& buffer : work.dog) ensure(buffer, plane_bytes, false);
    ensure(work.scratch, plane_bytes, false);
    ensure(work.carry, plane_bytes, false);
    ensure(work.keys, plane_bytes * 4, false);
    ensure(work.gradient, plane_bytes * 2, false);
    ensure(work.input_bytes, VkDeviceSize(width) * height, true);
    ensure(work.input_float, VkDeviceSize(effective_width) * height * 4, false);

    constexpr std::size_t kKernelSlots = 8;
    const VkDeviceSize kernel_alignment =
        std::max<VkDeviceSize>(sizeof(float), context.storage_buffer_offset_alignment());
    const std::size_t kernel_stride =
        static_cast<std::size_t>((kKernelMaxWidth * sizeof(float) + kernel_alignment - 1) /
                                 kernel_alignment * kernel_alignment / sizeof(float));
    ensure(work.kernel, kKernelSlots * kernel_stride * 4, true);
    std::vector<float> kernel_slots(kKernelSlots * kernel_stride, 0.0F);
    std::vector<float> kernel_sigmas(kKernelSlots, -1.0F);
    const auto stage_kernel = [&](float sigma) -> std::size_t {
        for (std::size_t slot = 0; slot < kKernelSlots; ++slot) {
            if (kernel_sigmas[slot] == sigma) return slot;
            if (kernel_sigmas[slot] < 0.0F) {
                const auto weights = gaussian_kernel(sigma);
                std::copy(weights.begin(), weights.end(),
                          kernel_slots.begin() + slot * kernel_stride);
                kernel_sigmas[slot] = sigma;
                return slot;
            }
        }
        throw std::runtime_error("Vulkan SIFT ran out of kernel slots");
    };
    if (impl.initial_smooth_sigma(plan.octave_min) > 0)
        stage_kernel(impl.initial_smooth_sigma(plan.octave_min));
    if (impl.skip_sigma() > 0) stage_kernel(impl.skip_sigma());
    for (const auto sigma : impl.sigma_inc) stage_kernel(sigma);
    std::memcpy(work.kernel->mapped_host(), kernel_slots.data(),
                kernel_slots.size() * sizeof(float));
    work.kernel->flush(0, kernel_slots.size() * sizeof(float));
    std::memcpy(work.input_bytes->mapped_host(), data, std::size_t(width) * height);
    work.input_bytes->flush(0, VkDeviceSize(width) * height);
    context.begin();
    const ConvertPush convert{effective_width * height, width, effective_width, 0};
    const std::array<BufferBinding, 2> convert_bindings{binding(work.input_bytes),
                                                        binding(work.input_float)};
    context.dispatch(Shader::Convert, convert_bindings, &convert, sizeof(convert),
                     (effective_width / 4 + 127) / 128, height);

    const auto blur = [&](const BufferRef& destination, const BufferRef& source,
                          const OctaveShape& shape, float sigma) {
        if (!(sigma > 0)) return;
        const auto slot = stage_kernel(sigma);
        const auto weights = gaussian_kernel(sigma);
        const auto radius = static_cast<std::uint32_t>(weights.size() / 2);
        const auto offset = VkDeviceSize(slot * kernel_stride * 4);
        const BlurPush horizontal{shape.width, shape.height, radius, 0};
        const std::array<BufferBinding, 3> h_bindings{
            binding(source), binding(work.scratch),
            BufferBinding{work.kernel->handle(), offset, VK_WHOLE_SIZE}};
        context.dispatch(Shader::Blur, h_bindings, &horizontal, sizeof(horizontal),
                         (shape.width + 127) / 128, shape.height);
        const BlurPush vertical{shape.width, shape.height, radius, 1};
        const std::array<BufferBinding, 3> v_bindings{
            binding(work.scratch), binding(destination),
            BufferBinding{work.kernel->handle(), offset, VK_WHOLE_SIZE}};
        context.dispatch(Shader::Blur, v_bindings, &vertical, sizeof(vertical),
                         (shape.width + 127) / 128, shape.height);
    };
    const auto initialize_octave = [&](int octave) {
        context.begin();
        const auto& shape = shapes[octave];
        auto& base = work.gaussian[0];
        if (octave == 0 && plan.octave_min < 0) {
            const UpsamplePush push{effective_width, height, shape.width, 0};
            const std::array<BufferBinding, 2> bindings{binding(work.input_float), binding(base)};
            context.dispatch(Shader::Upsample, bindings, &push, sizeof(push),
                             (shape.width + 127) / 128, shape.height);
        } else {
            const bool first = octave == 0;
            const DownsamplePush push{
                first ? effective_width : shapes[octave - 1].width, shape.width,
                first ? (1U << static_cast<std::uint32_t>(plan.octave_min)) : 2U, 0};
            const std::array<BufferBinding, 2> bindings{
                binding(first ? work.input_float : work.carry), binding(base)};
            context.dispatch(Shader::Downsample, bindings, &push, sizeof(push),
                             (shape.width + 255) / 256, shape.height);
        }
        blur(base, base, shape,
             octave == 0 ? impl.initial_smooth_sigma(plan.octave_min) : impl.skip_sigma());
    };
    const auto make_level = [&](std::size_t level, const OctaveShape& shape) {
        context.begin();
        blur(work.gaussian[level % 3], work.gaussian[(level - 1) % 3], shape,
             impl.sigma_inc[level - 1]);
        if (level == static_cast<std::size_t>(impl.level_ds - impl.level_min))
            context.queue_upload(*work.gaussian[level % 3], 0, VkDeviceSize(shape.pixels) * 4,
                                 *work.carry, 0);
    };
    const auto make_gradient = [&](const BufferRef& gaussian, const BufferRef& scratch,
                                   const OctaveShape& shape) {
        // Only the current Gaussian contributes to the gradient. A zero DoG
        // into a retired scratch slot leaves the same shader arithmetic intact.
        const DogPush push{shape.width, shape.height, 1, 0};
        const std::array<BufferBinding, 4> bindings{binding(gaussian), binding(gaussian),
                                                    binding(scratch), binding(work.gradient)};
        context.dispatch(Shader::Dog, bindings, &push, sizeof(push), (shape.width + 15) / 16,
                         (shape.height + 15) / 16);
    };
    struct Level {
        int octave{};
        std::size_t level{};
        int count{};
        std::vector<float> oriented;
    };
    std::vector<Level> levels(level_total);
    const float threshold = impl.options.peak_threshold;
    const float edge = (impl.options.edge_threshold + 1.0F) * (impl.options.edge_threshold + 1.0F) /
                       impl.options.edge_threshold;
    for (int octave = 0; octave < octave_num; ++octave) {
        const auto& shape = shapes[octave];
        std::vector<std::uint32_t> widths;
        for (std::uint32_t w = (shape.width + 2U) >> 2U;; w = std::max(1U, (w + 3U) >> 2U)) {
            widths.push_back(w);
            if (w == 1) break;
        }
        std::reverse(widths.begin(), widths.end());
        if (work.histogram.size() < widths.size()) work.histogram.resize(widths.size());
        for (std::size_t i = 0; i < widths.size(); ++i)
            ensure(work.histogram[i], VkDeviceSize(widths[i]) * shape.height * 16, false);
        ensure(work.histogram_readback, VkDeviceSize(shape.height) * 16, true, true);
        initialize_octave(octave);
        for (std::size_t level = 1; level < level_count; ++level) {
            make_level(level, shape);
            const std::size_t dog_level = level - 1;
            const DogPush dog_push{shape.width, shape.height, 0, 0};
            const std::array<BufferBinding, 4> dog_bindings{
                binding(work.gaussian[level % 3]), binding(work.gaussian[(level - 1) % 3]),
                binding(work.dog[dog_level % 3]), binding(work.dog[dog_level % 3])};
            context.dispatch(Shader::Dog, dog_bindings, &dog_push, sizeof(dog_push),
                             (shape.width + 15) / 16, (shape.height + 15) / 16);
            if (level < 3 || level - 3 >= dog_level_count) continue;
            const std::size_t feature_level = level - 3;
            auto& detected = levels[std::size_t(octave) * dog_level_count + feature_level];
            detected.octave = octave;
            detected.level = feature_level;
            context.fill_u32(*work.keys, 0, VkDeviceSize(shape.pixels) * 16, 0);
            const KeyPush key{shape.width, shape.height, 0.8F * threshold, threshold, edge, 0};
            const std::array<BufferBinding, 4> key_bindings{
                binding(work.dog[feature_level % 3]), binding(work.dog[(feature_level + 1) % 3]),
                binding(work.dog[(feature_level + 2) % 3]), binding(work.keys)};
            context.dispatch(Shader::Key, key_bindings, &key, sizeof(key), (shape.width + 7) / 8,
                             (shape.height + 7) / 8);
            const HistInitPush init{shape.width, shape.height, widths.back(), 0};
            const std::array<BufferBinding, 2> init_bindings{
                binding(work.keys), binding(work.histogram[widths.size() - 1])};
            context.dispatch(Shader::HistInit, init_bindings, &init, sizeof(init),
                             (widths.back() + 127) / 128, shape.height);
            for (std::size_t i = widths.size() - 1; i-- > 0;) {
                const HistReducePush push{widths[i + 1], widths[i], shape.height, 0};
                const std::array<BufferBinding, 2> bindings{binding(work.histogram[i + 1]),
                                                            binding(work.histogram[i])};
                context.dispatch(Shader::HistReduce, bindings, &push, sizeof(push),
                                 (widths[i] + 127) / 128, shape.height);
            }
            context.queue_download(*work.histogram[0], 0, VkDeviceSize(shape.height) * 16,
                                   *work.histogram_readback, 0);
            context.submit_and_wait();
            const auto* histogram =
                static_cast<const std::int32_t*>(work.histogram_readback->mapped_host());
            const std::size_t histogram_values = std::size_t(shape.height) * 4;
            for (std::size_t i = 0; i < histogram_values; ++i)
                if (histogram[i] > 0) detected.count += histogram[i];
            if (detected.count == 0) continue;
            ensure(work.list, VkDeviceSize(detected.count) * 16, true);
            auto* list = static_cast<std::int32_t*>(work.list->mapped_host());
            std::size_t cursor = 0;
            for (std::size_t i = 0; i < histogram_values; ++i) {
                for (std::int32_t j = 0; j < histogram[i]; ++j) {
                    list[cursor * 4] = static_cast<std::int32_t>(i % 4);
                    list[cursor * 4 + 1] = static_cast<std::int32_t>(i / 4);
                    list[cursor * 4 + 2] = j;
                    list[cursor * 4 + 3] = 0;
                    ++cursor;
                }
            }
            work.list->flush(0, VkDeviceSize(detected.count) * 16);
            context.begin();
            for (std::size_t i = 1; i < widths.size(); ++i) {
                const ListGenPush push{static_cast<std::uint32_t>(detected.count), widths[i]};
                const std::array<BufferBinding, 2> bindings{binding(work.list),
                                                            binding(work.histogram[i])};
                context.dispatch(Shader::ListGen, bindings, &push, sizeof(push),
                                 (detected.count + 127) / 128);
            }
            // Gaussian f+1 still occupies the three-level ring when f+3 is
            // detected. DoG f has retired and can serve as gradient scratch.
            make_gradient(work.gaussian[(feature_level + 1) % 3], work.dog[feature_level % 3],
                          shape);
            const OrientationPush orientation{static_cast<std::uint32_t>(detected.count),
                                              shape.width,
                                              shape.height,
                                              impl.options.maximum_orientations,
                                              impl.level_sigma[feature_level + 1],
                                              impl.sigma_step,
                                              1.5F,
                                              3.0F};
            const std::array<BufferBinding, 3> orientation_bindings{
                binding(work.list), binding(work.keys), binding(work.gradient)};
            context.dispatch(Shader::Orientation, orientation_bindings, &orientation,
                             sizeof(orientation), (detected.count + 63) / 64);
            context.submit_and_wait();
            work.list->invalidate();
            detected.oriented.resize(std::size_t(detected.count) * 4);
            std::memcpy(detected.oriented.data(), work.list->mapped_host(),
                        detected.oriented.size() * sizeof(float));
        }
        context.begin();
        context.submit_and_wait();
    }
    std::vector<Level*> kept;
    long long accumulated = 0;
    for (std::size_t index = levels.size(); index-- > 0;) {
        if (impl.options.maximum_features > 0 &&
            accumulated > static_cast<long long>(impl.options.maximum_features))
            continue;
        accumulated += levels[index].count;
        if (levels[index].count > 0) kept.push_back(&levels[index]);
    }
    std::reverse(kept.begin(), kept.end());
    FeatureSet result;
    result.image_width = width;
    result.image_height = height;
    result.descriptor_dimension = 128;
    result.extractor_name = std::string(name());
    result.metric = impl.options.root_sift ? DescriptorMetric::l2_root : DescriptorMetric::l2;
    struct ExpandedLevel {
        int octave;
        std::size_t level;
        std::vector<float> data;
    };
    std::vector<ExpandedLevel> expanded;
    std::size_t total_features = 0;
    const double orientation_factor = kTwoPi / 65535.0;
    for (const auto* level : kept) {
        ExpandedLevel out{level->octave, level->level, {}};
        for (std::size_t f = 0; f < std::size_t(level->count); ++f) {
            const auto* source = level->oriented.data() + f * 4;
            std::uint32_t packed = 0;
            std::memcpy(&packed, source + 3, sizeof(packed));
            const std::uint32_t first = packed & 0xFFFFU, second = packed >> 16;
            if (first == 65535U) continue;
            out.data.insert(out.data.end(), {source[0], source[1], source[2],
                                             static_cast<float>(orientation_factor * first)});
            if (second != 65535U && second != first)
                out.data.insert(out.data.end(), {source[0], source[1], source[2],
                                                 static_cast<float>(orientation_factor * second)});
        }
        total_features += out.data.size() / 4;
        expanded.push_back(std::move(out));
    }
    for (auto& level : expanded) {
        if (total_features <= impl.options.maximum_features) break;
        const auto count = level.data.size() / 4;
        if (count == 0) continue;
        if (total_features - count > impl.options.maximum_features) {
            total_features -= count;
            level.data.clear();
        } else
            break;
    }
    if (total_features == 0) return result;
    const float octave_scale_base = plan.octave_min >= 0
                                        ? static_cast<float>(1 << plan.octave_min)
                                        : 1.0F / static_cast<float>(1 << -plan.octave_min);
    result.keypoints.reserve(total_features);
    ensure(work.descriptor_lists, VkDeviceSize(total_features) * 16, true);
    ensure(work.descriptors, VkDeviceSize(total_features) * 128 * 4, false);
    ensure(work.descriptor_readback, VkDeviceSize(total_features) * 128 * 4, true, true);
    auto* descriptor_lists = static_cast<float*>(work.descriptor_lists->mapped_host());
    std::size_t cursor = 0;
    for (const auto& level : expanded) {
        const float scale = octave_scale_base * static_cast<float>(1 << level.octave);
        for (std::size_t f = 0; f < level.data.size() / 4; ++f) {
            const auto* key = level.data.data() + f * 4;
            double mirrored = std::fmod(kTwoPi - static_cast<double>(key[3]), kTwoPi);
            if (mirrored < 0) mirrored += kTwoPi;
            result.keypoints.push_back({scale * (key[0] - 0.5F) + 0.5F,
                                        scale * (key[1] - 0.5F) + 0.5F, scale * key[2],
                                        static_cast<float>(mirrored), 0.0F});
        }
        if (!level.data.empty())
            std::memcpy(descriptor_lists + cursor * 4, level.data.data(),
                        level.data.size() * sizeof(float));
        cursor += level.data.size() / 4;
    }
    work.descriptor_lists->flush(0, VkDeviceSize(total_features) * 16);
    // Rebuild the same pyramid, evaluating gradients only for retained levels.
    // This exchanges inexpensive blur passes for all-octave gradient storage.
    std::size_t feature_cursor = 0;
    for (int octave = 0; octave < octave_num; ++octave) {
        const auto& shape = shapes[octave];
        initialize_octave(octave);
        for (std::size_t level = 1; level < level_count; ++level) {
            make_level(level, shape);
            const auto found =
                std::find_if(expanded.begin(), expanded.end(), [&](const ExpandedLevel& candidate) {
                    return candidate.octave == octave && candidate.level == level - 1;
                });
            if (found == expanded.end() || found->data.empty()) continue;
            const std::size_t count = found->data.size() / 4;
            make_gradient(work.gaussian[level % 3], work.dog[0], shape);
            const DescriptorPush descriptor{static_cast<std::uint32_t>(count), shape.width,
                                            shape.height, 3.0F};
            const std::array<BufferBinding, 3> bindings{
                BufferBinding{work.descriptor_lists->handle(), VkDeviceSize(feature_cursor) * 16,
                              VK_WHOLE_SIZE},
                binding(work.gradient),
                BufferBinding{work.descriptors->handle(), VkDeviceSize(feature_cursor) * 128 * 4,
                              VK_WHOLE_SIZE}};
            context.dispatch(Shader::Descriptor, bindings, &descriptor, sizeof(descriptor),
                             static_cast<std::uint32_t>((count * 16 + 63) / 64));
            const DescriptorNormPush normalization{static_cast<std::uint32_t>(count), 0};
            const std::array<BufferBinding, 1> normalization_bindings{bindings[2]};
            context.dispatch(Shader::DescriptorNorm, normalization_bindings, &normalization,
                             sizeof(normalization), static_cast<std::uint32_t>((count + 63) / 64));
            feature_cursor += count;
        }
    }
    context.queue_download(*work.descriptors, 0, VkDeviceSize(total_features) * 128 * 4,
                           *work.descriptor_readback, 0);
    context.submit_and_wait();
    result.descriptors.resize(total_features * 128);
    std::memcpy(result.descriptors.data(), work.descriptor_readback->mapped_host(),
                total_features * 128 * sizeof(float));

    if (impl.options.root_sift) {
        for (std::size_t row = 0; row < result.keypoints.size(); ++row) {
            float* descriptor = result.descriptors.data() + row * 128;
            double sum = 0.0;
            for (int column = 0; column < 128; ++column) sum += std::max(0.0F, descriptor[column]);
            const float inverse = static_cast<float>(1.0 / std::max(sum, 1e-12));
            for (int column = 0; column < 128; ++column)
                descriptor[column] = std::sqrt(std::max(0.0F, descriptor[column]) * inverse);
        }
    }
    return result;
}

void register_vulkan_feature_backends() {
    if (!vulkan_backend::Context::available()) return;
    register_extractor("vulkan_sift", [] {
        auto extractor = std::make_unique<SiftVulkanExtractor>();
        if (!extractor->is_available())
            throw std::runtime_error(
                "vulkan_sift backend is not available on this machine");
        return std::unique_ptr<FeatureExtractor>(std::move(extractor));
    });
    register_matcher("vulkan_mutual_ratio", [] {
        auto matcher = std::make_unique<VulkanMutualRatioMatcher>();
        if (!matcher->is_available())
            throw std::runtime_error(
                "vulkan_mutual_ratio matcher is not available on this machine");
        return std::unique_ptr<FeatureMatcher>(std::move(matcher));
    });
}

std::unique_ptr<FeatureExtractor> make_vulkan_sift_extractor(
    const SiftVulkanOptions& options) {
    if (!vulkan_backend::Context::available()) return nullptr;
    return std::make_unique<SiftVulkanExtractor>(options);
}

std::unique_ptr<FeatureMatcher> make_vulkan_mutual_ratio_matcher(
    const VulkanMutualRatioMatcherOptions& options) {
    if (!vulkan_backend::Context::available()) return nullptr;
    return std::make_unique<VulkanMutualRatioMatcher>(options);
}

bool vulkan_feature_backend_available() {
    return vulkan_backend::Context::available();
}

}  // namespace photara::features
