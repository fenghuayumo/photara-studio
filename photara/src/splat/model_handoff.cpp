#include "splat/model_handoff.hpp"
#include "splat/trainer.hpp"
#include "io/shared_mapping.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace photara::splat {
namespace {
constexpr std::size_t kHeader = 20;

using io::detail::Mapping;
using io::detail::ack_path;
using io::detail::publish_text;

GaussianModel cpu_snapshot(const GaussianModel& model) {
    GaussianModel copy = model;
    const auto cpu = [](const tinytensor::Tensor& tensor) {
        if (!tensor.is_valid()) return tinytensor::Tensor{};
        return (tensor.device() == tinytensor::Device::CPU
                    ? tensor : tensor.to(tinytensor::Device::CPU)).contiguous();
    };
    copy.means = cpu(model.means);
    copy.log_scales = cpu(model.log_scales);
    copy.quaternions = cpu(model.quaternions);
    copy.opacity_logits = cpu(model.opacity_logits);
    copy.sh = cpu(model.sh);
    copy.normal_features = cpu(model.normal_features);
    copy.filter_3d = cpu(model.filter_3d);
    return copy;
}
}  // namespace

struct ModelHandoff::Impl {
    GaussianModel snapshot;
    Mapping mapping;
    std::filesystem::path ready;
    std::string name;
};

ModelHandoff::ModelHandoff(const GaussianModel& model, const std::filesystem::path& ready)
    : impl_(std::make_unique<Impl>()) {
    const std::size_t count = model.size();
    if (count == 0 || count > 50'000'000 || model.sh_degree > 3)
        throw std::runtime_error("Invalid final-model handoff size");
    impl_->snapshot = cpu_snapshot(model);
    const auto& s = impl_->snapshot;
    const std::size_t bases = (s.sh_degree + 1U) * (s.sh_degree + 1U);
    const bool normals = s.normal_features.is_valid() && s.normal_features.numel() == count * 4U;
    const bool filter = s.filter_3d.is_valid() && s.filter_3d.numel() == count;
    if (s.means.numel() != count * 3U || s.log_scales.numel() != count * 3U ||
        s.quaternions.numel() != count * 4U || s.opacity_logits.numel() != count ||
        s.sh.numel() != count * bases * 3U)
        throw std::runtime_error("Invalid final-model handoff layout");
    const auto bytes = kHeader + count * (11U + bases * 3U + (normals ? 4U : 0U) +
                                        (filter ? 1U : 0U)) * sizeof(float);
    const auto token = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
#ifdef _WIN32
    impl_->name = "Local\\Photara.model." + std::to_string(GetCurrentProcessId()) + "." + token;
#else
    impl_->name = "/photara_model_" + std::to_string(getpid()) + "_" + token;
#endif
    impl_->ready = ready;
    impl_->mapping.open(impl_->name, bytes, true);
    auto* cursor = static_cast<std::uint8_t*>(impl_->mapping.data);
    const auto append = [&](const void* data, std::size_t size) {
        std::memcpy(cursor, data, size);
        cursor += size;
    };
    const std::uint32_t version = k_gaussian_chunk_version;
    const std::uint64_t stored_count = count;
    const std::uint32_t flags = (filter ? 1U : 0U) | (normals ? 2U : 0U);
    append(&version, 4); append(&stored_count, 8); append(&s.sh_degree, 4); append(&flags, 4);
    const auto tensor = [&](const tinytensor::Tensor& t) {
        const auto* data = t.ptr<float>();
        for (std::size_t i = 0; i < t.numel(); ++i)
            if (!std::isfinite(data[i]))
                throw std::runtime_error("Non-finite final-model handoff parameter");
        append(data, t.numel() * sizeof(float));
    };
    tensor(s.means); tensor(s.log_scales); tensor(s.quaternions);
    tensor(s.opacity_logits); tensor(s.sh);
    if (normals) tensor(s.normal_features);
    if (filter) tensor(s.filter_3d);
    if (!ready.parent_path().empty()) std::filesystem::create_directories(ready.parent_path());
    publish_text(ready, "PHOTARA_MODEL_HANDOFF 1\n" + impl_->name + "\n" +
                        std::to_string(bytes) + "\n");
}

ModelHandoff::~ModelHandoff() = default;
const GaussianModel& ModelHandoff::model() const { return impl_->snapshot; }

bool ModelHandoff::wait_for_reader(const std::chrono::milliseconds timeout) const {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        std::ifstream in(ack_path(impl_->ready));
        std::string name;
        if (std::getline(in, name) && name == impl_->name) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

GaussianModel receive_model_handoff(const std::filesystem::path& ready) {
    std::ifstream in(ready);
    std::string magic, name;
    std::size_t bytes{};
    if (!std::getline(in, magic) || magic != "PHOTARA_MODEL_HANDOFF 1" ||
        !std::getline(in, name) || !(in >> bytes) || bytes < kHeader ||
        bytes > 50'000'000ULL * 64U * sizeof(float) + kHeader)
        throw std::runtime_error("Invalid final-model handoff manifest");
    Mapping mapping;
    mapping.open(name, bytes, false);
    const auto* data = static_cast<const std::uint8_t*>(mapping.data);
    std::uint32_t version{}, degree{}, flags{};
    std::uint64_t count{};
    std::memcpy(&version, data, 4); std::memcpy(&count, data + 4, 8);
    std::memcpy(&degree, data + 12, 4); std::memcpy(&flags, data + 16, 4);
    if (version != k_gaussian_chunk_version || count == 0 || count > 50'000'000 ||
        degree > 3 || (flags & ~3U) != 0)
        throw std::runtime_error("Invalid final-model handoff header");
    const auto expected = kHeader + count *
        (11U + (degree + 1U) * (degree + 1U) * 3U +
         ((flags & 2U) ? 4U : 0U) + ((flags & 1U) ? 1U : 0U)) * sizeof(float);
    if (expected != bytes) throw std::runtime_error("Truncated final-model handoff payload");
    // Acknowledging the open mapping lets the publisher finish without waiting
    // for CPU decoding or GPU upload. Our handle keeps its storage alive.
    publish_text(ack_path(ready), name + "\n");
    return decode_gaussians(std::span<const std::uint8_t>(data, bytes), tinytensor::Device::CPU);
}
}  // namespace photara::splat
