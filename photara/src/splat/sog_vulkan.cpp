#include "sog_vulkan.hpp"
#include "sog_palette.hpp"
#include "sog_cluster.hlsl.embedded.hpp"
#include "photara_vk/photara_vk.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstdlib>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>

namespace photara::splat::detail {
namespace {
struct Push {
    unsigned mode, count, dims, clusters, branches, iteration;
};
static_assert(sizeof(Push) == 24);
struct Backend {
    photara::vk::Device device;
    photara::vk::ComputePipeline pipeline;
    photara::vk::EncoderRing ring;
    Backend() {
        photara::vk::DeviceRequest request;
        if (const char* validation = std::getenv("TINYTENSOR_VULKAN_VALIDATION"))
            request.validation = *validation && std::string_view(validation) != "0";
        std::optional<std::array<std::uint8_t, VK_UUID_SIZE>> wanted;
        const char* uuid = std::getenv("PHOTARA_VULKAN_DEVICE_UUID");
        if (uuid && *uuid) {
            const std::string_view value(uuid);
            if (value.size() != 2 * VK_UUID_SIZE)
                throw std::invalid_argument("PHOTARA_VULKAN_DEVICE_UUID must be 32 hex digits");
            wanted.emplace();
            for (unsigned i = 0; i < VK_UUID_SIZE; ++i) {
                unsigned byte = 0;
                const auto parsed =
                    std::from_chars(value.data() + 2 * i, value.data() + 2 * i + 2, byte, 16);
                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + 2 * i + 2)
                    throw std::invalid_argument("Invalid PHOTARA_VULKAN_DEVICE_UUID");
                (*wanted)[i] = static_cast<std::uint8_t>(byte);
            }
        } else if (const char* index = std::getenv("TINYTENSOR_VULKAN_DEVICE")) {
            const std::string_view value(index);
            const auto parsed =
                std::from_chars(value.data(), value.data() + value.size(), request.device_index);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
                throw std::invalid_argument("Invalid TINYTENSOR_VULKAN_DEVICE index");
        }
        // The export owns its compute device and all its buffers. In Python a
        // process-global adopted runtime can be torn down after the Vulkan
        // loader, so neither a device nor its pipelines may escape this scope.
        device = photara::vk::Device::create(request);
        if (wanted) {
            unsigned count = 0;
            if (vkEnumeratePhysicalDevices(device.instance(), &count, nullptr) != VK_SUCCESS)
                throw std::runtime_error("Cannot enumerate SOG Vulkan devices");
            std::vector<VkPhysicalDevice> candidates(count);
            if (vkEnumeratePhysicalDevices(device.instance(), &count, candidates.data()) !=
                VK_SUCCESS)
                throw std::runtime_error("Cannot enumerate SOG Vulkan devices");
            unsigned selected = count;
            for (unsigned i = 0; i < count; ++i) {
                VkPhysicalDeviceIDProperties identity{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
                VkPhysicalDeviceProperties2 properties{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                properties.pNext = &identity;
                vkGetPhysicalDeviceProperties2(candidates[i], &properties);
                if (std::equal(wanted->begin(), wanted->end(), identity.deviceUUID))
                    selected = i;
            }
            if (selected == count)
                throw std::runtime_error("No SOG Vulkan device matches PHOTARA_VULKAN_DEVICE_UUID");
            if (candidates[selected] != device.physical()) {
                device = {};
                request.device_index = selected;
                device = photara::vk::Device::create(request);
            }
        }
        pipeline =
            device.create_compute(std::as_bytes(std::span{sog_cluster_hlsl_spv}), 10, sizeof(Push));
        ring = photara::vk::EncoderRing(device, 1, photara::vk::BarrierPolicy::after_compute);
    }
};

SogPalette cluster_batch(
    Backend& backend,
    const std::vector<float>& values,
    unsigned first,
    unsigned n,
    unsigned dims,
    unsigned budget) {
    using photara::vk::MemoryKind;
    const unsigned target = std::bit_floor(std::min(n, budget));
    const auto buffer = [&](std::size_t words) {
        return backend.device.create_buffer(words * 4, MemoryKind::device_local);
    };
    auto points = buffer(std::size_t(n) * dims);
    auto centers = buffer(std::size_t(target) * dims);
    auto labels = buffer(n), counts = buffer(target), offsets = buffer(target + 1U);
    auto cursors = buffer(target), indices = buffer(n);
    auto parent_offsets = buffer(target + 1U), parent_indices = buffer(n);
    auto blocks = buffer(2U * ((target + 63U) / 64U));
    auto upload = backend.device.create_buffer(points.size(), MemoryKind::host_visible);
    upload.upload(values.data() + std::size_t(first) * dims, points.size());
    auto bindings = [&] {
        const std::array buffers{&points,  &centers, &labels,         &counts,         &offsets,
                                 &cursors, &indices, &parent_offsets, &parent_indices, &blocks};
        std::array<photara::vk::BufferBinding, 10> result;
        for (unsigned i = 0; i < result.size(); ++i) {
            result[i] = photara::vk::binding(*buffers[i]);
        }
        return result;
    };
    Push push{0, n, dims, 1, 1, 0};
    // One command submission per hierarchy level, not per iteration or kernel.
    auto dispatch = [&](auto& encoder, const auto& buffers, unsigned mode, unsigned threads) {
        push.mode = mode;
        const unsigned groups = (threads + 63U) / 64U;
        encoder.dispatch(
            backend.pipeline, buffers, &push, sizeof(push), std::min(groups, 65535U),
            (groups + 65534U) / 65535U);
    };
    {
        auto buffers = bindings();
        auto& encoder = backend.ring.acquire();
        encoder.copy(points, 0, upload, 0, points.size());
        dispatch(encoder, buffers, 0, n);
        backend.ring.submit();
        encoder.wait();
    }
    upload = {};
    unsigned parents = 1;
    do {
        std::swap(offsets, parent_offsets);
        std::swap(indices, parent_indices);
        push.branches = std::min(16U, target / parents);
        push.clusters = parents * push.branches;
        auto buffers = bindings();
        auto& encoder = backend.ring.acquire();
        dispatch(encoder, buffers, 1, push.clusters * dims);
        for (push.iteration = 0; push.iteration < 8; ++push.iteration) {
            dispatch(encoder, buffers, 2, n);
            dispatch(encoder, buffers, 3, push.clusters);
            dispatch(encoder, buffers, 4, n);
            dispatch(encoder, buffers, 5, push.clusters);
            dispatch(encoder, buffers, 6, (push.clusters + 63U) / 64U);
            dispatch(encoder, buffers, 7, push.clusters);
            dispatch(encoder, buffers, 8, n);
            dispatch(encoder, buffers, 9, push.clusters * 64U);
        }
        backend.ring.submit();
        encoder.wait();
        parents = push.clusters;
    } while (parents < target);
    std::vector<unsigned> host_counts(target), host_labels(n);
    std::vector<float> host_centers(std::size_t(target) * dims);
    auto readback = backend.device.create_buffer(
        counts.size() + labels.size() + centers.size(), MemoryKind::host_cached);
    auto& encoder = backend.ring.acquire();
    encoder.barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    encoder.copy(readback, 0, counts, 0, counts.size());
    encoder.copy(readback, counts.size(), labels, 0, labels.size());
    encoder.copy(readback, counts.size() + labels.size(), centers, 0, centers.size());
    backend.ring.submit();
    encoder.wait();
    readback.download(host_counts.data(), counts.size());
    readback.download(host_labels.data(), labels.size(), counts.size());
    readback.download(host_centers.data(), centers.size(), counts.size() + labels.size());
    SogPalette result;
    std::vector<unsigned> remap(target, ~0U);
    for (unsigned c = 0; c < target; ++c) {
        if (!host_counts[c])
            continue;
        remap[c] = static_cast<unsigned>(result.centroids.size() / dims);
        result.centroids.insert(
            result.centroids.end(), host_centers.begin() + std::size_t(c) * dims,
            host_centers.begin() + std::size_t(c + 1) * dims);
    }
    result.labels.reserve(n);
    for (unsigned c : host_labels) {
        if (c >= target || remap[c] == ~0U)
            throw std::runtime_error("Invalid Vulkan SOG cluster membership");
        result.labels.push_back(remap[c]);
    }
    return result;
}

// Assign one bounded point batch against the entire shared codebook. All
// batches see the same centers; aggregate their sufficient statistics before
// updating centers, so a batch boundary never limits cluster membership.
void global_batch(
    Backend& backend,
    const std::vector<float>& values,
    unsigned first,
    unsigned n,
    unsigned dims,
    SogPalette& palette,
    std::vector<double>& sums,
    std::vector<std::uint64_t>& totals) {
    using photara::vk::MemoryKind;
    const unsigned k = unsigned(palette.centroids.size() / dims);
    const auto buffer = [&](std::size_t words) {
        return backend.device.create_buffer(words * 4, MemoryKind::device_local);
    };
    auto points = buffer(std::size_t(n) * dims), centers = buffer(std::size_t(k) * dims);
    auto labels = buffer(n), counts = buffer(k), offsets = buffer(k + 1), cursors = buffer(k);
    auto indices = buffer(n), blocks = buffer(2 * ((k + 63) / 64));
    auto upload = backend.device.create_buffer(
        points.size() + centers.size() + labels.size(), MemoryKind::host_visible);
    upload.upload(values.data() + std::size_t(first) * dims, points.size());
    upload.upload(palette.centroids.data(), centers.size(), points.size());
    upload.upload(palette.labels.data() + first, labels.size(), points.size() + centers.size());
    const std::array buffers{photara::vk::binding(points),  photara::vk::binding(centers),
                             photara::vk::binding(labels),  photara::vk::binding(counts),
                             photara::vk::binding(offsets), photara::vk::binding(cursors),
                             photara::vk::binding(indices), photara::vk::binding(offsets),
                             photara::vk::binding(indices), photara::vk::binding(blocks)};
    Push push{10, n, dims, k, 1, 0};
    auto& encoder = backend.ring.acquire();
    encoder.copy(points, 0, upload, 0, points.size());
    encoder.copy(centers, 0, upload, points.size(), centers.size());
    encoder.copy(labels, 0, upload, points.size() + centers.size(), labels.size());
    const auto dispatch = [&](unsigned mode, unsigned threads) {
        push.mode = mode;
        const unsigned groups = (threads + 63) / 64;
        encoder.dispatch(
            backend.pipeline, buffers, &push, sizeof(push), std::min(groups, 65535U),
            (groups + 65534) / 65535);
    };
    for (unsigned at = 0; at < n; at += 8192) {
        push.iteration = at;
        dispatch(10, std::min(8192U, n - at));
    }
    push.iteration = 0;
    dispatch(3, k);
    dispatch(4, n);
    dispatch(5, k);
    dispatch(6, (k + 63) / 64);
    dispatch(7, k);
    dispatch(8, n);
    dispatch(11, k * 64);
    auto readback = backend.device.create_buffer(
        counts.size() + centers.size() + labels.size(), MemoryKind::host_cached);
    encoder.barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    encoder.copy(readback, 0, counts, 0, counts.size());
    encoder.copy(readback, counts.size(), centers, 0, centers.size());
    encoder.copy(readback, counts.size() + centers.size(), labels, 0, labels.size());
    backend.ring.submit();
    encoder.wait();
    std::vector<unsigned> host_counts(k);
    std::vector<float> host_centers(std::size_t(k) * dims);
    readback.download(host_counts.data(), counts.size());
    readback.download(host_centers.data(), centers.size(), counts.size());
    readback.download(palette.labels.data() + first, labels.size(), counts.size() + centers.size());
    for (unsigned c = 0; c < k; ++c) {
        totals[c] += host_counts[c];
        for (unsigned d = 0; d < dims; ++d)
            sums[std::size_t(c) * dims + d] +=
                double(host_centers[std::size_t(c) * dims + d]) * host_counts[c];
    }
}
} // namespace

SogPalette cluster_sog_vulkan(
    const std::vector<float>& values,
    unsigned n,
    unsigned dims,
    unsigned budget) {
    if (!n || !dims || dims > 45 || !budget || budget > 65536 ||
        values.size() != std::size_t(n) * dims)
        throw std::invalid_argument("Invalid Vulkan SOG clustering input");
    // At degree 3 a points buffer occupies at most 45 MiB. No N*K distance
    // matrix and no float atomic feature requirements. Each batch gets its
    // proportional share of the 16-bit palette; there is no CPU K-means path.
    constexpr unsigned batch_limit = 262144;
    const unsigned batches = std::bit_ceil((n - 1) / batch_limit + 1);
    const unsigned palette_budget = std::bit_floor(std::min(n, budget));
    if (batches > palette_budget)
        throw std::runtime_error("SOG scene exceeds the bounded Vulkan clustering budget");
    // Reuse the device across batches, and release it before returning from
    // the export (including Python calls that unload the library at exit).
    Backend backend;
    SogPalette result;
    result.labels.reserve(n);
    for (unsigned batch = 0; batch < batches; ++batch) {
        // Balanced batches prevent a one-point tail from consuming half the
        // palette. Power-of-two shares use the entire hierarchical budget.
        const unsigned first = static_cast<unsigned>(std::uint64_t(n) * batch / batches);
        const unsigned end = static_cast<unsigned>(std::uint64_t(n) * (batch + 1) / batches);
        const unsigned count = end - first, share = palette_budget / batches;
        auto chunk = cluster_batch(backend, values, first, count, dims, share);
        const unsigned base = static_cast<unsigned>(result.centroids.size() / dims);
        result.centroids.insert(
            result.centroids.end(), chunk.centroids.begin(), chunk.centroids.end());
        for (auto label : chunk.labels)
            result.labels.push_back(base + label);
    }
    // Hierarchical batches are only an initialization. Spend the exact
    // requested budget, then run global Lloyd updates with no N*K allocation.
    complete_sog_palette(values, n, dims, budget, result);
    const unsigned k = unsigned(result.centroids.size() / dims);
    for (unsigned iteration = 0; iteration < 3; ++iteration) {
        std::vector<double> sums(std::size_t(k) * dims);
        std::vector<std::uint64_t> counts(k);
        for (unsigned first = 0; first < n; first += batch_limit)
            global_batch(
                backend, values, first, std::min(batch_limit, n - first), dims, result, sums,
                counts);
        for (unsigned c = 0; c < k; ++c)
            if (counts[c])
                for (unsigned d = 0; d < dims; ++d)
                    result.centroids[std::size_t(c) * dims + d] =
                        float(sums[std::size_t(c) * dims + d] / counts[c]);
        complete_sog_palette(values, n, dims, budget, result);
    }
    return result;
}
} // namespace photara::splat::detail
