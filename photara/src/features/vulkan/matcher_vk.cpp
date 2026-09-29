#include "features/vulkan_features.hpp"
#include "vk_context.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace photara::features {
namespace {

using vulkan_backend::BufferBinding;
using vulkan_backend::BufferRef;
using vulkan_backend::Context;
using vulkan_backend::Shader;

std::size_t padded(std::size_t count) { return (count + 31) / 32 * 32; }

struct TilesPush {
    std::uint32_t query_count;
    std::uint32_t train_count;
    std::uint32_t mutual;
    std::uint32_t pad0;
};

struct FinishPush {
    std::uint32_t query_count;
    std::uint32_t train_count;
    std::uint32_t row_chunks;
    std::uint32_t col_chunks;
    float ratio;
    float mutual;
    std::uint32_t pad0;
    std::uint32_t pad1;
};

BufferBinding binding(const BufferRef& buffer, VkDeviceSize offset = 0) {
    return BufferBinding{buffer->handle(), offset, VK_WHOLE_SIZE};
}

}  // namespace

class VulkanMutualRatioMatcher::Impl {
public:
    explicit Impl(VulkanMutualRatioMatcherOptions value) : options(std::move(value)) {
        if (!(options.ratio_threshold > 0.0F && options.ratio_threshold <= 1.0F) ||
            options.maximum_features < 2)
            throw std::invalid_argument("Invalid Vulkan matcher options");
        context = &Context::get();
        const auto properties = context->memory_properties();
        std::uint64_t device_bytes = 0;
        for (std::uint32_t heap = 0; heap < properties.memoryHeapCount; ++heap) {
            if ((properties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
                device_bytes += properties.memoryHeaps[heap].size;
        }
        budget = std::min<std::uint64_t>(768ULL << 20, device_bytes / 8);
        const std::uint64_t minimum = 2ULL * padded(options.maximum_features) * 128;
        if (budget < minimum)
            throw std::runtime_error("Insufficient Vulkan descriptor cache memory");
        initial_capacity = std::min<std::uint64_t>(
            budget, std::max<std::uint64_t>(minimum, 4ULL << 20));
    }

    struct Entry {
        std::size_t offset;
        std::size_t bytes;
        std::uint64_t generation;
    };

    // Grows the host-visible descriptor mirror so that `bytes` more descriptors
    // fit. Growth copies the used prefix, keeping every cached offset valid, and
    // happens before a batch is recorded, where the queue is idle. Allocating
    // the whole budget up front would reserve hundreds of megabytes per matcher
    // clone, and the SfM frontend creates one clone per worker thread.
    void ensure_capacity(std::size_t bytes) {
        if (descriptors.valid() && used + bytes <= capacity) return;
        if (used + bytes > budget) {
            cache.clear();
            used = 0;
        }
        if (used + bytes > budget)
            throw std::invalid_argument(
                "Vulkan matching pair exceeds descriptor budget");
        std::size_t next = capacity != 0 ? capacity : initial_capacity;
        const std::size_t needed = used + bytes;
        while (next < needed) next = std::min<std::size_t>(budget, next * 2);
        BufferRef grown = context->allocate(next, true);
        if (descriptors.valid() && used > 0) {
            std::memcpy(grown->mapped_host(), descriptors->mapped_host(), used);
            grown->flush(0, used);
        }
        descriptors = std::move(grown);
        capacity = next;
    }

    // Reserves (or reuses) a padded device range for one FeatureSet and copies
    // its descriptors into the host-visible mirror buffer.
    std::size_t reserve(const FeatureSet& features) {
        const auto found = cache.find(features.descriptor_identity);
        if (found != cache.end() &&
            found->second.generation == features.descriptor_generation &&
            found->second.bytes == features.descriptors_u8.size())
            return found->second.offset;
        const std::size_t bytes = padded(features.keypoints.size()) * 128;
        if (!descriptors.valid() || used + bytes > capacity)
            throw std::logic_error("Vulkan descriptor mirror is too small");
        const std::size_t offset = used;
        std::uint8_t* destination =
            static_cast<std::uint8_t*>(descriptors->mapped_host()) + offset;
        std::memset(destination, 0, bytes);
        std::memcpy(destination, features.descriptors_u8.data(),
                    features.descriptors_u8.size());
        descriptors->flush(offset, bytes);
        cache[features.descriptor_identity] =
            Entry{offset, features.descriptors_u8.size(),
                  features.descriptor_generation};
        used += bytes;
        return offset;
    }

    VulkanMutualRatioMatcherOptions options;
    Context* context{};
    std::uint64_t budget{};
    std::uint64_t initial_capacity{};
    std::size_t capacity{};
    mutable std::mutex mutex;
    mutable std::unordered_map<std::uint64_t, Entry> cache;
    mutable std::size_t used{};
    mutable BufferRef descriptors;
};

VulkanMutualRatioMatcher::VulkanMutualRatioMatcher(
    VulkanMutualRatioMatcherOptions options)
    : impl_(std::make_shared<Impl>(std::move(options))) {}

VulkanMutualRatioMatcher::~VulkanMutualRatioMatcher() = default;
VulkanMutualRatioMatcher::VulkanMutualRatioMatcher(
    VulkanMutualRatioMatcher&&) noexcept = default;
VulkanMutualRatioMatcher& VulkanMutualRatioMatcher::operator=(
    VulkanMutualRatioMatcher&&) noexcept = default;

bool VulkanMutualRatioMatcher::is_built() noexcept { return Context::available(); }

bool VulkanMutualRatioMatcher::is_available() const noexcept {
    return impl_ != nullptr && impl_->context != nullptr;
}

std::unique_ptr<FeatureMatcher> VulkanMutualRatioMatcher::clone() const {
    return std::make_unique<VulkanMutualRatioMatcher>(impl_->options);
}

void VulkanMutualRatioMatcher::clear_prepared() {
    std::lock_guard lock(impl_->mutex);
    impl_->cache.clear();
    impl_->used = 0;
}

MatchSet VulkanMutualRatioMatcher::match(
    const FeatureSet& query, const FeatureSet& train) const {
    const Pair pair{&query, &train};
    return std::move(match_batch(std::span(&pair, 1))[0]);
}

std::vector<MatchSet> VulkanMutualRatioMatcher::match_batch(
    std::span<const Pair> pairs) const {
    std::vector<MatchSet> result(pairs.size());
    if (pairs.empty()) return result;

    Context& context = *impl_->context;
    for (std::size_t begin = 0; begin < pairs.size();) {
        std::size_t batch_count = 0;
        std::size_t batch_bytes = 0;
        while (batch_count < 8 && begin + batch_count < pairs.size()) {
            const auto& [query, train] = pairs[begin + batch_count];
            const auto bytes =
                (padded(query->keypoints.size()) + padded(train->keypoints.size())) *
                128;
            if (bytes > impl_->budget)
                throw std::invalid_argument(
                    "Vulkan matching pair exceeds descriptor budget");
            if (batch_count != 0 && batch_bytes + bytes > impl_->budget) break;
            batch_bytes += bytes;
            ++batch_count;
        }
        const auto batch = pairs.subspan(begin, batch_count);

        std::size_t outputs = 0;
        std::size_t row_size = 0;
        std::size_t col_size = 0;
        for (const auto& [query, train] : batch) {
            for (const auto* features : {query, train}) {
                features->validate();
                if (features->storage != DescriptorStorage::uint8 ||
                    features->descriptor_dimension != 128 ||
                    features->keypoints.size() > impl_->options.maximum_features)
                    throw std::invalid_argument(
                        "Vulkan matcher expects bounded 128D u8 descriptors");
            }
            const auto query_count = query->keypoints.size();
            const auto train_count = train->keypoints.size();
            outputs += query_count;
            row_size = std::max(row_size, query_count * ((train_count + 31) / 32));
            col_size = std::max(col_size, train_count * ((query_count + 31) / 32));
        }

        std::lock_guard lock(impl_->mutex);
        // Grow before recording: the previous batch was synchronized, so no
        // queued work can still reference the old mirror.
        impl_->ensure_capacity(batch_bytes);

        BufferRef row_partials = context.allocate(row_size * 12, false);
        BufferRef col_partials =
            impl_->options.mutual_check ? context.allocate(col_size * 12, false)
                                        : BufferRef{};
        BufferRef output = context.allocate(outputs * 4 + 4, false);
        BufferRef readback = context.allocate(outputs * 4 + 4, true, true);

        context.begin();
        if (outputs > 0)
            context.fill_u32(*output, 0, outputs * 4, 0xFFFFFFFFU);

        std::size_t output_offset = 0;
        for (const auto& [query, train] : batch) {
            const auto query_count = query->keypoints.size();
            const auto train_count = train->keypoints.size();
            if (query_count > 0 && train_count > 0) {
                const std::size_t query_offset = impl_->reserve(*query);
                const std::size_t train_offset = impl_->reserve(*train);

                const TilesPush tiles_push{
                    static_cast<std::uint32_t>(query_count),
                    static_cast<std::uint32_t>(train_count),
                    impl_->options.mutual_check ? 1U : 0U, 0U};
                const std::array<BufferBinding, 4> tiles_bindings{
                    binding(impl_->descriptors, query_offset),
                    binding(impl_->descriptors, train_offset),
                    binding(row_partials),
                    binding(col_partials.valid() ? col_partials : row_partials)};
                context.dispatch(Shader::MatchTiles, tiles_bindings, &tiles_push,
                                 sizeof(tiles_push),
                                 static_cast<std::uint32_t>((train_count + 31) / 32),
                                 static_cast<std::uint32_t>((query_count + 31) / 32));

                const FinishPush finish_push{
                    static_cast<std::uint32_t>(query_count),
                    static_cast<std::uint32_t>(train_count),
                    static_cast<std::uint32_t>((train_count + 31) / 32),
                    static_cast<std::uint32_t>((query_count + 31) / 32),
                    impl_->options.ratio_threshold,
                    impl_->options.mutual_check ? 1.0F : 0.0F,
                    0U,
                    0U};
                const std::array<BufferBinding, 3> finish_bindings{
                    binding(row_partials),
                    binding(col_partials.valid() ? col_partials : row_partials),
                    binding(output, output_offset * 4)};
                context.dispatch(Shader::MatchFinish, finish_bindings, &finish_push,
                                 sizeof(finish_push),
                                 static_cast<std::uint32_t>((query_count + 255) / 256));
            }
            output_offset += query_count;
        }

        if (outputs > 0)
            context.queue_download(*output, 0, outputs * 4, *readback, 0);
        context.submit_and_wait();

        if (outputs > 0) {
            const auto* matches =
                static_cast<const std::int32_t*>(readback->mapped_host());
            std::size_t offset = 0;
            for (std::size_t pair_index = 0; pair_index < batch.size(); ++pair_index) {
                const auto query_count =
                    batch[pair_index].first->keypoints.size();
                auto& matches_out = result[begin + pair_index].matches;
                matches_out.reserve(query_count / 4 + 1);
                for (std::size_t i = 0; i < query_count; ++i) {
                    const std::int32_t train_index = matches[offset + i];
                    if (train_index >= 0)
                        matches_out.push_back(
                            {static_cast<FeatureIndex>(i),
                             static_cast<FeatureIndex>(train_index), 1.0F});
                }
                offset += query_count;
            }
        }
        begin += batch_count;
    }
    return result;
}

}  // namespace photara::features
