#include "features/vulkan_features.hpp"
#include "features/registry.hpp"
#include "vk_context.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
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
    explicit Impl(SiftVulkanOptions value) : options(std::move(value)) {
        if (options.maximum_features == 0 || options.maximum_image_dimension == 0 ||
            options.maximum_orientations > 4 || options.octave_layers == 0 ||
            options.octave_layers > 16)
            throw std::invalid_argument("Invalid Vulkan SIFT limits");
        if (options.first_octave < -1 || options.first_octave > 4)
            throw std::invalid_argument(
                "Vulkan SIFT first_octave must be in [-1, 4]");

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
                sigma0 * std::pow(2.0F, static_cast<float>(level) /
                                            static_cast<float>(dog_level_num));
        sigma_inc.resize(static_cast<std::size_t>(level_num - 1));
        for (int level = level_min + 1; level <= level_max; ++level)
            sigma_inc[static_cast<std::size_t>(level - level_min - 1)] =
                dsigma0 * std::pow(sigmak, static_cast<float>(level));
    }

    SiftVulkanOptions options;
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
        const float sb = sigma0 *
                         std::pow(sigmak, static_cast<float>(level_ds - dog_level_num));
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
        while (working_width > maximum_dimension ||
               working_height > maximum_dimension) {
            ++result.octave_min;
            working_width = std::max(1U, working_width >> 1);
            working_height = std::max(1U, working_height >> 1);
        }
        const double smallest =
            static_cast<double>(std::min(working_width, working_height));
        result.octave_num =
            std::max(1, static_cast<int>(std::floor(std::log2(smallest))) - 3);
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
SiftVulkanExtractor& SiftVulkanExtractor::operator=(
    SiftVulkanExtractor&&) noexcept = default;

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
    value.metric =
        impl_->options.root_sift ? DescriptorMetric::l2_root : DescriptorMetric::l2;
    value.typical_descriptor_dimension = 128;
    return value;
}

std::unique_ptr<FeatureExtractor> SiftVulkanExtractor::clone() const {
    return std::make_unique<SiftVulkanExtractor>(impl_->options);
}

FeatureSet SiftVulkanExtractor::extract_gray(
    const std::span<const std::uint8_t> pixels, const std::uint32_t width,
    const std::uint32_t height, std::size_t row_stride) const {
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
            std::copy_n(pixels.data() + static_cast<std::size_t>(row) * row_stride,
                        width,
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
    const std::size_t dog_count = static_cast<std::size_t>(impl.level_num - 1);
    const std::size_t dog_level_count = static_cast<std::size_t>(impl.dog_level_num);
    const std::size_t level_total =
        static_cast<std::size_t>(octave_num) * dog_level_count;

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
            shapes[static_cast<std::size_t>(octave)] =
                OctaveShape{aligned, h, aligned * h};
            w = std::max(1U, w >> 1);
            h = std::max(1U, h >> 1);
        }
    }

    BufferRef input_bytes =
        context.allocate(static_cast<VkDeviceSize>(width) * height, true);
    BufferRef input_float = context.allocate(
        static_cast<VkDeviceSize>(effective_width) * height * 4, false);
    std::uint32_t max_pixels = 0;
    for (const auto& shape : shapes) max_pixels = std::max(max_pixels, shape.pixels);
    BufferRef blur_scratch =
        context.allocate(static_cast<VkDeviceSize>(max_pixels) * 4, false);
    // Every blur pass needs its own taps at execution time, so all kernels are
    // staged side by side and selected through the descriptor binding offset.
    constexpr std::size_t kKernelSlots = 8;
    constexpr std::size_t kKernelStride = kKernelMaxWidth; // floats per slot
    BufferRef kernel_weights =
        context.allocate(kKernelSlots * kKernelStride * 4, true);
    std::vector<float> kernel_slots(kKernelSlots * kKernelStride, 0.0F);
    std::vector<float> kernel_sigmas(kKernelSlots, -1.0F);
    auto stage_kernel = [&](float sigma) {
        for (std::size_t slot = 0; slot < kKernelSlots; ++slot) {
            if (kernel_sigmas[slot] == sigma) return slot;
            if (kernel_sigmas[slot] < 0.0F) {
                const std::vector<float> kernel = gaussian_kernel(sigma);
                std::copy(kernel.begin(), kernel.end(),
                          kernel_slots.begin() +
                              static_cast<std::ptrdiff_t>(slot * kKernelStride));
                kernel_sigmas[slot] = sigma;
                return slot;
            }
        }
        throw std::runtime_error("Vulkan SIFT ran out of kernel slots");
    };
    // Stage every sigma the pyramid will need before recording starts; the
    // dispatches then only look up their slot.
    if (impl.initial_smooth_sigma(plan.octave_min) > 0.0F)
        (void)stage_kernel(impl.initial_smooth_sigma(plan.octave_min));
    if (impl.skip_sigma() > 0.0F) (void)stage_kernel(impl.skip_sigma());
    for (const auto sigma : impl.sigma_inc) (void)stage_kernel(sigma);
    std::memcpy(kernel_weights->mapped_host(), kernel_slots.data(),
                kernel_slots.size() * sizeof(float));
    kernel_weights->flush(0, static_cast<VkDeviceSize>(kernel_slots.size() * 4));

    std::vector<BufferRef> gaussian(static_cast<std::size_t>(octave_num) * level_count);
    std::vector<BufferRef> dog(static_cast<std::size_t>(octave_num) * dog_count);
    std::vector<BufferRef> gradient(
        static_cast<std::size_t>(octave_num) * dog_level_count);
    std::vector<BufferRef> keys(static_cast<std::size_t>(octave_num) * dog_level_count);
    for (int octave = 0; octave < octave_num; ++octave) {
        const auto& shape = shapes[static_cast<std::size_t>(octave)];
        const auto base = static_cast<std::size_t>(octave);
        for (std::size_t level = 0; level < level_count; ++level)
            gaussian[base * level_count + level] =
                context.allocate(static_cast<VkDeviceSize>(shape.pixels) * 4, false);
        for (std::size_t level = 0; level < dog_count; ++level)
            dog[base * dog_count + level] =
                context.allocate(static_cast<VkDeviceSize>(shape.pixels) * 4, false);
        for (std::size_t level = 0; level < dog_level_count; ++level) {
            gradient[base * dog_level_count + level] =
                context.allocate(static_cast<VkDeviceSize>(shape.pixels) * 8, false);
            keys[base * dog_level_count + level] =
                context.allocate(static_cast<VkDeviceSize>(shape.pixels) * 16, false);
        }
    }

    struct HistChain {
        std::vector<BufferRef> levels; // index 0 = top (width 1), last = finest
        std::vector<std::uint32_t> widths;
    };
    std::vector<HistChain> hist_chains(static_cast<std::size_t>(octave_num));
    for (int octave = 0; octave < octave_num; ++octave) {
        const auto& shape = shapes[static_cast<std::size_t>(octave)];
        auto& chain = hist_chains[static_cast<std::size_t>(octave)];
        std::vector<std::uint32_t> widths;
        std::uint32_t w = (shape.width + 2U) >> 2U;
        while (true) {
            widths.push_back(w);
            if (w == 1U) break;
            w = std::max(1U, (w + 3U) >> 2U);
            if (widths.size() > 32) break;
        }
        for (std::size_t i = widths.size(); i-- > 0;)
            chain.widths.push_back(widths[i]);
        for (const auto level_width : chain.widths)
            chain.levels.push_back(context.allocate(
                static_cast<VkDeviceSize>(level_width) * shape.height * 16, false));
    }

    std::memcpy(input_bytes->mapped_host(), data,
                static_cast<std::size_t>(width) * height);
    input_bytes->flush(0, static_cast<VkDeviceSize>(width) * height);
    context.begin();
    {
        const ConvertPush push{effective_width * height, width, effective_width, 0};
        const std::array<BufferBinding, 2> bindings{binding(input_bytes),
                                                    binding(input_float)};
        context.dispatch(Shader::Convert, bindings, &push, sizeof(push),
                         (effective_width / 4 + 127) / 128, height);
    }

    auto blur = [&](const BufferRef& destination, const BufferRef& source,
                    std::uint32_t w, std::uint32_t h, float sigma) {
        if (!(sigma > 0.0F)) return;
        const std::size_t slot = stage_kernel(sigma);
        const std::vector<float> kernel = gaussian_kernel(sigma);
        const std::uint32_t radius =
            static_cast<std::uint32_t>(kernel.size() / 2);
        const VkDeviceSize kernel_offset =
            static_cast<VkDeviceSize>(slot * kKernelStride * 4);
        const BlurPush horizontal{w, h, radius, 0};
        const std::array<BufferBinding, 3> horizontal_bindings{
            binding(source), binding(blur_scratch),
            BufferBinding{kernel_weights->handle(), kernel_offset,
                          VK_WHOLE_SIZE}};
        context.dispatch(Shader::Blur, horizontal_bindings, &horizontal,
                         sizeof(horizontal), (w + 127) / 128, h);
        const BlurPush vertical{w, h, radius, 1};
        const std::array<BufferBinding, 3> vertical_bindings{
            binding(blur_scratch), binding(destination),
            BufferBinding{kernel_weights->handle(), kernel_offset,
                          VK_WHOLE_SIZE}};
        context.dispatch(Shader::Blur, vertical_bindings, &vertical,
                         sizeof(vertical), (w + 127) / 128, h);
    };

    for (int octave = 0; octave < octave_num; ++octave) {
        const auto& shape = shapes[static_cast<std::size_t>(octave)];
        const auto base = static_cast<std::size_t>(octave);
        auto& base_level = gaussian[base * level_count];

        if (octave == 0) {
            if (plan.octave_min < 0) {
                const UpsamplePush push{effective_width, height, shape.width, 0};
                const std::array<BufferBinding, 2> bindings{binding(input_float),
                                                            binding(base_level)};
                context.dispatch(Shader::Upsample, bindings, &push, sizeof(push),
                                 (shape.width + 127) / 128, shape.height);
            } else {
                const DownsamplePush push{
                    effective_width, shape.width,
                    1U << static_cast<std::uint32_t>(plan.octave_min), 0};
                const std::array<BufferBinding, 2> bindings{binding(input_float),
                                                            binding(base_level)};
                context.dispatch(Shader::Downsample, bindings, &push, sizeof(push),
                                 (shape.width + 255) / 256, shape.height);
            }
            blur(base_level, base_level, shape.width, shape.height,
                 impl.initial_smooth_sigma(plan.octave_min));
        } else {
            const auto& previous = shapes[base - 1];
            const BufferRef& source =
                gaussian[(base - 1) * level_count +
                         static_cast<std::size_t>(impl.level_ds - impl.level_min)];
            const DownsamplePush push{previous.width, shape.width, 2, 0};
            const std::array<BufferBinding, 2> bindings{binding(source),
                                                        binding(base_level)};
            context.dispatch(Shader::Downsample, bindings, &push, sizeof(push),
                             (shape.width + 255) / 256, shape.height);
            blur(base_level, base_level, shape.width, shape.height, impl.skip_sigma());
        }

        for (std::size_t level = 1; level < level_count; ++level) {
            blur(gaussian[base * level_count + level],
                 gaussian[base * level_count + level - 1], shape.width, shape.height,
                 impl.sigma_inc[level - 1]);
        }

        for (std::size_t level = 0; level < dog_count; ++level) {
            const DogPush push{shape.width, shape.height,
                               level < dog_level_count ? 1U : 0U, 0};
            const std::array<BufferBinding, 4> bindings{
                binding(gaussian[base * level_count + level + 1]),
                binding(gaussian[base * level_count + level]),
                binding(dog[base * dog_count + level]),
                binding(level < dog_level_count
                            ? gradient[base * dog_level_count + level]
                            : dog[base * dog_count + level])};
            context.dispatch(Shader::Dog, bindings, &push, sizeof(push),
                             (shape.width + 15) / 16, (shape.height + 15) / 16);
        }
    }
    const float dog_threshold = impl.options.peak_threshold;
    const float dog_threshold0 = 0.8F * dog_threshold;
    const float edge_threshold = (impl.options.edge_threshold + 1.0F) *
                                 (impl.options.edge_threshold + 1.0F) /
                                 impl.options.edge_threshold;
    // Every (octave, level) needs its own row-sum histogram readback: the
    // per-octave reduction chain is reused by the three DoG levels, so a single
    // download after the loop would only capture the last level.
    std::vector<std::size_t> hist_offsets(level_total, 0);
    std::size_t hist_bytes = 0;
    {
        std::size_t cursor = 0;
        for (int octave = 0; octave < octave_num; ++octave) {
            const std::size_t bytes = static_cast<std::size_t>(
                                          shapes[static_cast<std::size_t>(octave)].height) *
                                      16;
            for (std::size_t level = 0; level < dog_level_count; ++level) {
                hist_offsets[static_cast<std::size_t>(octave) * dog_level_count +
                             level] = cursor;
                cursor += bytes;
            }
        }
        hist_bytes = cursor;
    }
    BufferRef hist_readback = context.allocate(hist_bytes, true, true);
    for (int octave = 0; octave < octave_num; ++octave) {
        const auto& shape = shapes[static_cast<std::size_t>(octave)];
        const auto base = static_cast<std::size_t>(octave);
        for (std::size_t level = 0; level < dog_level_count; ++level) {
            context.fill_u32(*keys[base * dog_level_count + level], 0,
                             static_cast<VkDeviceSize>(shape.pixels) * 16, 0);
            const KeyPush push{shape.width, shape.height, dog_threshold0,
                               dog_threshold, edge_threshold, 0};
            const std::array<BufferBinding, 4> bindings{
                binding(dog[base * dog_count + level]),
                binding(dog[base * dog_count + level + 1]),
                binding(dog[base * dog_count + level + 2]),
                binding(keys[base * dog_level_count + level])};
            context.dispatch(Shader::Key, bindings, &push, sizeof(push),
                             (shape.width + 7) / 8, (shape.height + 7) / 8);

            auto& chain = hist_chains[base];
            const HistInitPush init_push{shape.width, shape.height,
                                         chain.widths.back(), 0};
            const std::array<BufferBinding, 2> init_bindings{
                binding(keys[base * dog_level_count + level]),
                binding(chain.levels.back())};
            context.dispatch(Shader::HistInit, init_bindings, &init_push,
                             sizeof(init_push), (chain.widths.back() + 127) / 128,
                             shape.height);
            for (std::size_t i = chain.levels.size() - 1; i-- > 0;) {
                const HistReducePush reduce_push{chain.widths[i + 1],
                                                 chain.widths[i], shape.height, 0};
                const std::array<BufferBinding, 2> reduce_bindings{
                    binding(chain.levels[i + 1]), binding(chain.levels[i])};
                context.dispatch(Shader::HistReduce, reduce_bindings, &reduce_push,
                                 sizeof(reduce_push),
                                 (chain.widths[i] + 127) / 128, shape.height);
            }
            context.queue_download(
                *chain.levels[0], 0,
                static_cast<VkDeviceSize>(shape.height) * 16, *hist_readback,
                static_cast<VkDeviceSize>(
                    hist_offsets[base * dog_level_count + level]));
        }
    }
    context.submit_and_wait();
    struct LevelId {
        int octave;
        std::size_t level;
    };
    std::vector<int> level_counts(level_total, 0);
    std::vector<LevelId> level_ids(level_total);
    {
        const auto* hist_values =
            static_cast<const std::int32_t*>(hist_readback->mapped_host());
        for (int octave = 0; octave < octave_num; ++octave) {
            const auto& shape = shapes[static_cast<std::size_t>(octave)];
            const std::size_t length = static_cast<std::size_t>(shape.height) * 4;
            for (std::size_t level = 0; level < dog_level_count; ++level) {
                const std::size_t index =
                    static_cast<std::size_t>(octave) * dog_level_count + level;
                level_ids[index] = LevelId{octave, level};
                const std::int32_t* level_hist =
                    hist_values + hist_offsets[index] / sizeof(std::int32_t);
                int count = 0;
                for (std::size_t i = 0; i < length; ++i) {
                    const auto value = level_hist[i];
                    count += value > 0 ? value : 0;
                }
                level_counts[index] = count;
            }
        }
    }
    struct LevelPlan {
        int octave;
        std::size_t level;
        int count;
    };
    std::vector<LevelPlan> kept;
    {
        long long accumulated = 0;
        const long long threshold =
            static_cast<long long>(impl.options.maximum_features);
        for (std::size_t index = level_total; index-- > 0;) {
            if (threshold > 0 && accumulated > threshold) continue;
            accumulated += level_counts[index];
            if (level_counts[index] > 0)
                kept.push_back(
                    LevelPlan{level_ids[index].octave, level_ids[index].level,
                              level_counts[index]});
        }
        std::reverse(kept.begin(), kept.end());
    }

    FeatureSet result;
    result.image_width = width;
    result.image_height = height;
    result.descriptor_dimension = 128;
    result.extractor_name = std::string(name());
    result.metric =
        impl.options.root_sift ? DescriptorMetric::l2_root : DescriptorMetric::l2;
    if (kept.empty()) return result;

    std::vector<BufferRef> lists(kept.size());
    for (std::size_t i = 0; i < kept.size(); ++i)
        lists[i] = context.allocate(static_cast<VkDeviceSize>(kept[i].count) * 16, true);

    {
        const auto* hist_values =
            static_cast<const std::int32_t*>(hist_readback->mapped_host());
        std::size_t next = 0;
        for (int octave = 0; octave < octave_num; ++octave) {
            const auto& shape = shapes[static_cast<std::size_t>(octave)];
            const std::size_t length = static_cast<std::size_t>(shape.height) * 4;
            for (std::size_t level = 0; level < dog_level_count; ++level) {
                const auto found = std::find_if(
                    kept.begin(), kept.end(), [&](const LevelPlan& entry) {
                        return entry.octave == octave && entry.level == level;
                    });
                if (found == kept.end() || found->count == 0) continue;
                const std::int32_t* level_hist =
                    hist_values +
                    hist_offsets[static_cast<std::size_t>(octave) * dog_level_count +
                                 level] /
                        sizeof(std::int32_t);
                auto* destination = static_cast<std::int32_t*>(lists[next]->mapped_host());
                std::size_t written = 0;
                for (std::size_t i = 0; i < length; ++i) {
                    const int count = level_hist[i];
                    if (count <= 0) continue;
                    const std::int32_t x = static_cast<std::int32_t>(i % 4);
                    const std::int32_t y = static_cast<std::int32_t>(i / 4);
                    for (std::int32_t j = 0; j < count; ++j) {
                        destination[written * 4 + 0] = x;
                        destination[written * 4 + 1] = y;
                        destination[written * 4 + 2] = j;
                        destination[written * 4 + 3] = 0;
                        ++written;
                    }
                }
                lists[next]->flush(0, written * 16);
                ++next;
            }
        }
    }
    context.begin();
    for (std::size_t i = 0; i < kept.size(); ++i) {
        const auto& entry = kept[i];
        const auto& chain = hist_chains[static_cast<std::size_t>(entry.octave)];
        const auto& shape = shapes[static_cast<std::size_t>(entry.octave)];
        // The chain is shared by the three levels of the octave, so the
        // histogram pyramid has to be rebuilt for this level before the list
        // refinement reads it (SiftGPU runs histogram -> list -> refine per
        // level).
        const HistInitPush init_push{shape.width, shape.height,
                                     chain.widths.back(), 0};
        const std::array<BufferBinding, 2> init_bindings{
            binding(keys[static_cast<std::size_t>(entry.octave) * dog_level_count +
                         entry.level]),
            binding(chain.levels.back())};
        context.dispatch(Shader::HistInit, init_bindings, &init_push,
                         sizeof(init_push), (chain.widths.back() + 127) / 128,
                         shape.height);
        for (std::size_t level = chain.levels.size() - 1; level-- > 0;) {
            const HistReducePush reduce_push{chain.widths[level + 1],
                                             chain.widths[level], shape.height, 0};
            const std::array<BufferBinding, 2> reduce_bindings{
                binding(chain.levels[level + 1]), binding(chain.levels[level])};
            context.dispatch(Shader::HistReduce, reduce_bindings, &reduce_push,
                             sizeof(reduce_push),
                             (chain.widths[level] + 127) / 128, shape.height);
        }
        for (std::size_t level = 1; level < chain.levels.size(); ++level) {
            const ListGenPush push{static_cast<std::uint32_t>(entry.count),
                                   chain.widths[level]};
            const std::array<BufferBinding, 2> bindings{binding(lists[i]),
                                                        binding(chain.levels[level])};
            context.dispatch(Shader::ListGen, bindings, &push, sizeof(push),
                             (static_cast<std::uint32_t>(entry.count) + 127) / 128);
        }

        const OrientationPush push{
            static_cast<std::uint32_t>(entry.count), shape.width, shape.height,
            impl.options.maximum_orientations, impl.level_sigma[entry.level + 1],
            impl.sigma_step, 1.5F, 3.0F};
        const std::array<BufferBinding, 3> bindings{
            binding(lists[i]),
            binding(keys[static_cast<std::size_t>(entry.octave) * dog_level_count +
                         entry.level]),
            binding(gradient[static_cast<std::size_t>(entry.octave) *
                                 dog_level_count +
                             entry.level])};
        context.dispatch(Shader::Orientation, bindings, &push, sizeof(push),
                         (static_cast<std::uint32_t>(entry.count) + 63) / 64);
    }
    context.submit_and_wait();

    struct ExpandedLevel {
        int octave;
        std::size_t level;
        std::vector<float> data;
    };
    std::vector<ExpandedLevel> expanded(kept.size());
    std::size_t total_features = 0;
    {
        const double orientation_factor = kTwoPi / 65535.0;
        for (std::size_t i = 0; i < kept.size(); ++i) {
            const auto& entry = kept[i];
            auto& out = expanded[i];
            out.octave = entry.octave;
            out.level = entry.level;
            out.data.reserve(static_cast<std::size_t>(entry.count) * 8);
            lists[i]->invalidate();
            const auto* source = static_cast<const float*>(lists[i]->mapped_host());
            for (int f = 0; f < entry.count; ++f) {
                const float x = source[f * 4 + 0];
                const float y = source[f * 4 + 1];
                const float scale = source[f * 4 + 2];
                std::uint32_t packed = 0;
                std::memcpy(&packed, &source[f * 4 + 3], sizeof(packed));
                const std::uint32_t first = packed & 0xFFFFU;
                const std::uint32_t second = packed >> 16;
                if (first == 65535U) continue;
                out.data.push_back(x);
                out.data.push_back(y);
                out.data.push_back(scale);
                out.data.push_back(
                    static_cast<float>(orientation_factor * first));
                if (second != 65535U && second != first) {
                    out.data.push_back(x);
                    out.data.push_back(y);
                    out.data.push_back(scale);
                    out.data.push_back(
                        static_cast<float>(orientation_factor * second));
                }
            }
            total_features += out.data.size() / 4;
        }
    }

    if (impl.options.maximum_features > 0) {
        std::size_t total = total_features;
        std::size_t index = 0;
        while (total > impl.options.maximum_features && index < expanded.size()) {
            const std::size_t count = expanded[index].data.size() / 4;
            if (count > 0 && total - count > impl.options.maximum_features) {
                total -= count;
                expanded[index].data.clear();
            } else {
                break;
            }
            ++index;
        }
        total_features = total;
    }
    if (total_features == 0) return result;

    const float octave_scale_base =
        plan.octave_min >= 0
            ? static_cast<float>(1 << plan.octave_min)
            : 1.0F / static_cast<float>(1 << -plan.octave_min);
    result.keypoints.reserve(total_features);
    for (const auto& level : expanded) {
        const float octave_scale =
            octave_scale_base * static_cast<float>(1 << level.octave);
        for (std::size_t f = 0; f < level.data.size() / 4; ++f) {
            const float x = level.data[f * 4 + 0];
            const float y = level.data[f * 4 + 1];
            const float scale = level.data[f * 4 + 2];
            const float orientation = level.data[f * 4 + 3];
            double mirrored =
                std::fmod(kTwoPi - static_cast<double>(orientation), kTwoPi);
            if (mirrored < 0.0) mirrored += kTwoPi;
            result.keypoints.push_back(
                {octave_scale * (x - 0.5F) + 0.5F,
                 octave_scale * (y - 0.5F) + 0.5F,
                 octave_scale * scale,
                 static_cast<float>(mirrored),
                 0.0F});
        }
    }

    BufferRef descriptor_lists = context.allocate(
        static_cast<VkDeviceSize>(total_features) * 16, true);
    BufferRef descriptors = context.allocate(
        static_cast<VkDeviceSize>(total_features) * 128 * 4, false);
    BufferRef descriptor_readback = context.allocate(
        static_cast<VkDeviceSize>(total_features) * 128 * 4, true, true);
    {
        auto* destination = static_cast<float*>(descriptor_lists->mapped_host());
        std::size_t cursor = 0;
        for (const auto& level : expanded) {
            std::memcpy(destination + cursor * 4, level.data.data(),
                        level.data.size() * sizeof(float));
            cursor += level.data.size() / 4;
        }
        descriptor_lists->flush(0, static_cast<VkDeviceSize>(total_features) * 16);
    }
    context.begin();
    {
        std::size_t feature_cursor = 0;
        for (const auto& level : expanded) {
            const std::size_t count = level.data.size() / 4;
            if (count == 0) continue;
            const auto& shape = shapes[static_cast<std::size_t>(level.octave)];
            const DescriptorPush push{static_cast<std::uint32_t>(count), shape.width,
                                      shape.height, 3.0F};
            const std::array<BufferBinding, 3> bindings{
                BufferBinding{descriptor_lists->handle(),
                              static_cast<VkDeviceSize>(feature_cursor) * 16,
                              VK_WHOLE_SIZE},
                binding(gradient[static_cast<std::size_t>(level.octave) *
                                     dog_level_count +
                                 level.level]),
                BufferBinding{descriptors->handle(),
                              static_cast<VkDeviceSize>(feature_cursor) * 128 * 4,
                              VK_WHOLE_SIZE}};
            context.dispatch(Shader::Descriptor, bindings, &push, sizeof(push),
                             static_cast<std::uint32_t>((count * 16 + 63) / 64));
            const DescriptorNormPush norm_push{static_cast<std::uint32_t>(count), 0};
            const std::array<BufferBinding, 1> norm_bindings{
                BufferBinding{descriptors->handle(),
                              static_cast<VkDeviceSize>(feature_cursor) * 128 * 4,
                              VK_WHOLE_SIZE}};
            context.dispatch(Shader::DescriptorNorm, norm_bindings, &norm_push,
                             sizeof(norm_push),
                             static_cast<std::uint32_t>((count + 63) / 64));
            feature_cursor += count;
        }
        context.queue_download(*descriptors, 0,
                               static_cast<VkDeviceSize>(total_features) * 128 * 4,
                               *descriptor_readback, 0);
    }
    context.submit_and_wait();

    result.descriptors.resize(total_features * 128);
    std::memcpy(result.descriptors.data(), descriptor_readback->mapped_host(),
                total_features * 128 * sizeof(float));

    if (impl.options.root_sift) {
        for (std::size_t row = 0; row < result.keypoints.size(); ++row) {
            float* descriptor = result.descriptors.data() + row * 128;
            double sum = 0.0;
            for (int column = 0; column < 128; ++column)
                sum += std::max(0.0F, descriptor[column]);
            const float inverse = static_cast<float>(1.0 / std::max(sum, 1e-12));
            for (int column = 0; column < 128; ++column)
                descriptor[column] =
                    std::sqrt(std::max(0.0F, descriptor[column]) * inverse);
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
