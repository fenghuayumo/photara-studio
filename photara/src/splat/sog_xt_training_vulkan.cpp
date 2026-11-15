#include "formats_sog_xt.hpp"
#include "sog_xt_training.hlsl.embedded.hpp"
#include "photara_vk/photara_vk.hpp"
#include "vulkan/backend.hpp"
#include <array>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace photara::splat::detail {
namespace {
using tinytensor::Tensor;
constexpr auto vk_device = tinytensor::Device::Vulkan;
struct Push {
    unsigned mode{}, n{}, width{}, stride{}, first{}, dims{}, total{};
    float tv{}, qtv{};
};
static_assert(sizeof(Push) == 36);
struct Backend {
    photara::vk::Device device;
    photara::vk::ComputePipeline pipeline;
    photara::vk::EncoderRing ring;
    Backend() {
        auto h = tinytensor::vulkan::device_handles();
        photara::vk::ExternalDevice ext;
        ext.instance = h.instance;
        ext.physical = h.physical_device;
        ext.device = h.device;
        ext.queue = h.queue;
        ext.queue_family = h.queue_family;
        ext.enabled.push_descriptors = tinytensor::vulkan::device_info().push_descriptors;
        device = photara::vk::Device::adopt(ext);
        pipeline = device.create_compute(
            std::as_bytes(std::span{sog_xt_training_hlsl_spv}), 8, sizeof(Push));
        ring = photara::vk::EncoderRing(device, 1, photara::vk::BarrierPolicy::after_compute);
    }
};
void dispatch(Push p, const std::array<Tensor, 8>& tensors) {
    if (!p.total)
        return;
    static thread_local Backend b;
    tinytensor::vulkan::synchronize();
    std::array<photara::vk::BufferBinding, 8> views;
    for (unsigned i = 0; i < 8; ++i) {
        auto v = tinytensor::vulkan::buffer_view(
            tensors[i].is_valid() && tensors[i].numel() ? tensors[i] : tensors[0]);
        views[i] = {v.buffer, v.offset, v.bytes};
    }
    auto& e = b.ring.acquire();
    unsigned groups = (p.total + 255) / 256;
    e.dispatch(
        b.pipeline, views, &p, sizeof(p), std::min(groups, 65535u), (groups + 65534) / 65535);
    b.ring.submit();
    // These borrowed Tensor allocations must survive the external submission.
    e.wait();
}
void topology(SogXtTrainingCodec& c) {
    auto& p = c.latent;
    if (p.means.device() != vk_device || !p.size() || !c.dimensions ||
        p.size() > std::numeric_limits<unsigned>::max() / std::max(14u, c.dimensions))
        throw std::invalid_argument("Invalid Vulkan SOG-XT codec size or device");
    const auto validate = [](const Tensor& t, tinytensor::DataType dtype, std::size_t count) {
        if (!t.is_valid() || t.device() != vk_device || t.dtype() != dtype || !t.is_contiguous() ||
            t.numel() != count)
            throw std::invalid_argument("Invalid Vulkan SOG-XT tensor layout");
    };
    const auto nvalues = p.size();
    if (p.sh.shape().rank() != 3 || p.sh.shape()[2] != 3 ||
        p.sh.shape()[1] * 3 != c.dimensions + 3 ||
        nvalues > std::size_t(std::numeric_limits<int>::max()) / std::max(14u, c.dimensions + 3))
        throw std::invalid_argument("Invalid SOG-XT SH shape or index range");
    for (const auto* t : {&p.means, &p.log_scales, &p.quaternions, &p.opacity_logits, &p.sh})
        validate(
            *t, tinytensor::DataType::Float32,
            t == &p.opacity_logits ? nvalues
            : t == &p.quaternions  ? nvalues * 4
            : t == &p.sh           ? nvalues * (c.dimensions + 3)
                                   : nvalues * 3);
    validate(c.low, tinytensor::DataType::Float32, 14);
    validate(c.high, tinytensor::DataType::Float32, 14);
    validate(c.sh_low, tinytensor::DataType::Float32, c.dimensions);
    validate(c.sh_high, tinytensor::DataType::Float32, c.dimensions);
    validate(c.labels, tinytensor::DataType::Int32, nvalues);
    validate(c.neighbors, tinytensor::DataType::Int32, nvalues * 2);
    validate(c.counts, tinytensor::DataType::Int32, std::size_t(c.center_side) * c.center_side);
    if (c.center_offsets.is_valid())
        return;
    const auto n = p.size(), centers = std::size_t(c.center_side) * c.center_side;
    if (!centers || centers > std::numeric_limits<unsigned>::max() / c.dimensions ||
        c.labels.numel() != n || c.counts.numel() != centers || c.neighbors.numel() != n * 2)
        throw std::invalid_argument("Invalid SOG-XT topology dimensions");
    auto labels = c.labels.to(tinytensor::Device::CPU);
    auto neighbors = c.neighbors.to(tinytensor::Device::CPU);
    auto counts = c.counts.to(tinytensor::Device::CPU);
    std::vector<std::vector<int>> members(centers), incoming(n);
    for (std::size_t i = 0; i < n; ++i) {
        int label = labels.ptr<int>()[i];
        if (label < 0 || std::size_t(label) >= centers)
            throw std::invalid_argument("Invalid SOG-XT label");
        members[label].push_back(int(i));
        for (unsigned axis = 0; axis < 2; ++axis) {
            int j = neighbors.ptr<int>()[i * 2 + axis];
            if (j < -1 || (j >= 0 && std::size_t(j) >= n))
                throw std::invalid_argument("Invalid SOG-XT neighbor");
            if (j >= 0)
                incoming[j].push_back(int(i));
        }
    }
    for (std::size_t i = 0; i < centers; ++i)
        if (counts.ptr<int>()[i] != int(members[i].size()))
            throw std::invalid_argument("Invalid SOG-XT center count");
    auto pack = [](const auto& lists, Tensor& offsets, Tensor& indices) {
        std::vector<int> off{0}, ids;
        for (const auto& list : lists) {
            ids.insert(ids.end(), list.begin(), list.end());
            off.push_back(int(ids.size()));
        }
        offsets = Tensor::from_vector(off, {off.size()}, vk_device);
        if (ids.empty())
            ids.push_back(0);
        indices = Tensor::from_vector(ids, {ids.size()}, vk_device);
    };
    pack(members, c.center_offsets, c.center_indices);
    pack(incoming, c.tv_offsets, c.tv_sources);
}
std::array<Tensor*, 5> fields(GaussianModel& p) {
    return {&p.means, &p.log_scales, &p.quaternions, &p.opacity_logits, &p.sh};
}
std::array<Tensor*, 5> gradients(ModelGradients& g) {
    return {&g.means, &g.log_scales, &g.quaternions, &g.opacity_logits, &g.sh};
}
constexpr unsigned widths[] = {3, 3, 4, 1, 3}, firsts[] = {0, 3, 6, 10, 11};
} // namespace
GaussianModel sog_xt_training_forward_vulkan(SogXtTrainingCodec& c) {
    topology(c);
    auto& p = c.latent;
    GaussianModel out = p;
    auto src = fields(p), dst = fields(out);
    const unsigned n = unsigned(p.size()), stride = unsigned(p.sh.shape()[1] * 3);
    if (c.dimensions + 3 != stride)
        throw std::invalid_argument("Invalid SOG-XT SH dimensions");
    for (unsigned f = 0; f < 5; ++f) {
        *dst[f] = Tensor::empty_like(*src[f]);
        dispatch(
            {0, n, widths[f], f == 4 ? stride : widths[f], firsts[f], c.dimensions, n * widths[f]},
            {*src[f], *dst[f], c.low, c.high, c.labels, c.counts, c.center_offsets,
             c.center_indices});
    }
    const std::size_t centers = std::size_t(c.center_side) * c.center_side;
    c.centers = Tensor::empty({centers, std::size_t(c.dimensions)}, vk_device);
    c.raw_centers = Tensor::empty_like(c.centers);
    dispatch(
        {1, n, 0, stride, 0, c.dimensions, unsigned(c.centers.numel())},
        {p.sh, c.centers, c.sh_low, c.sh_high, c.raw_centers, c.counts, c.center_offsets,
         c.center_indices});
    dispatch(
        {2, n, 0, stride, 0, c.dimensions, n * c.dimensions},
        {c.centers, out.sh, c.sh_low, c.sh_high, c.labels, c.counts, c.center_offsets,
         c.center_indices});
    return out;
}
void sog_xt_training_backward_vulkan(
    SogXtTrainingCodec& c,
    const GaussianModel& decoded,
    ModelGradients& g,
    float tv,
    float qtv) {
    topology(c);
    if (!c.raw_centers.is_valid())
        throw std::invalid_argument("SOG-XT backward requires forward");
    auto src = fields(c.latent), grad = gradients(g);
    GaussianModel decoded_handles = decoded;
    auto dec = fields(decoded_handles);
    unsigned n = unsigned(c.latent.size()), stride = unsigned(c.latent.sh.shape()[1] * 3);
    for (unsigned f = 0; f < 5; ++f) {
        if (!grad[f]->is_valid() || grad[f]->shape() != src[f]->shape() ||
            grad[f]->device() != vk_device || grad[f]->dtype() != tinytensor::DataType::Float32 ||
            !grad[f]->is_contiguous() || !dec[f]->is_valid() ||
            dec[f]->shape() != src[f]->shape() || dec[f]->device() != vk_device ||
            dec[f]->dtype() != tinytensor::DataType::Float32 || !dec[f]->is_contiguous())
            throw std::invalid_argument("Invalid SOG-XT gradient");
        Push p{
            3, n, widths[f], f == 4 ? stride : widths[f], firsts[f], c.dimensions, n * widths[f]};
        dispatch(
            p, {*src[f], *grad[f], c.low, c.high, *dec[f], c.counts, c.center_offsets,
                c.center_indices});
        if (tv > 0 || qtv > 0) {
            p.mode = 4;
            p.tv = tv;
            p.qtv = qtv;
            dispatch(
                p, {*src[f], *grad[f], c.low, c.high, c.neighbors, c.tv_sources, c.tv_offsets,
                    c.center_indices});
        }
    }
    auto sums = Tensor::empty_like(c.centers);
    dispatch(
        {5, n, 0, stride, 0, c.dimensions, unsigned(sums.numel())},
        {g.sh, sums, c.sh_low, c.sh_high, c.raw_centers, c.counts, c.center_offsets,
         c.center_indices});
    dispatch(
        {2, n, 0, stride, 0, c.dimensions, n * c.dimensions},
        {sums, g.sh, c.sh_low, c.sh_high, c.labels, c.counts, c.center_offsets, c.center_indices});
}
} // namespace photara::splat::detail
