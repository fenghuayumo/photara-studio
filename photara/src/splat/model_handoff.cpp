#include "splat/model_handoff.hpp"
#include "splat/trainer.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace photara::splat {
namespace {
constexpr std::size_t kHeader = 20;

std::filesystem::path ack_path(std::filesystem::path path) {
    path += ".ack";
    return path;
}

struct Mapping {
    void* data{};
    std::size_t size{};
#ifdef _WIN32
    HANDLE handle{};
#else
    int handle{-1};
    std::string owned_name;
#endif
    ~Mapping() {
#ifdef _WIN32
        if (data) UnmapViewOfFile(data);
        if (handle) CloseHandle(handle);
#else
        if (data) munmap(data, size);
        if (handle >= 0) close(handle);
        if (!owned_name.empty()) shm_unlink(owned_name.c_str());
#endif
    }
    void open(const std::string& name, const std::size_t bytes, const bool create) {
        size = bytes;
#ifdef _WIN32
        handle = create
            ? CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                static_cast<DWORD>(static_cast<std::uint64_t>(bytes) >> 32U),
                static_cast<DWORD>(bytes), name.c_str())
            : OpenFileMappingA(FILE_MAP_READ, FALSE, name.c_str());
        if (!handle) throw std::runtime_error("Cannot open final-model shared memory");
        data = MapViewOfFile(handle, create ? FILE_MAP_WRITE : FILE_MAP_READ, 0, 0, bytes);
#else
        handle = shm_open(name.c_str(), create ? O_CREAT | O_EXCL | O_RDWR : O_RDONLY, 0600);
        if (handle < 0) throw std::runtime_error("Cannot open final-model shared memory");
        if (create) {
            owned_name = name;
            struct statvfs available{};
            if (fstatvfs(handle, &available) != 0 ||
                bytes > static_cast<std::uint64_t>(available.f_bavail) * available.f_frsize)
                throw std::runtime_error("Insufficient shared-memory capacity for final model");
            if (ftruncate(handle, static_cast<off_t>(bytes)) != 0)
                throw std::runtime_error("Cannot size final-model shared memory");
        } else {
            struct stat info{};
            if (fstat(handle, &info) != 0 || info.st_size < 0 ||
                static_cast<std::uint64_t>(info.st_size) < bytes)
                throw std::runtime_error("Truncated final-model shared memory");
        }
        data = mmap(nullptr, bytes, create ? PROT_READ | PROT_WRITE : PROT_READ,
                    MAP_SHARED, handle, 0);
        if (data == MAP_FAILED) data = nullptr;
#endif
        if (!data) throw std::runtime_error("Cannot map final-model shared memory");
    }
};

void publish_text(const std::filesystem::path& path, const std::string& text) {
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        out << text;
        out.close();
        if (!out) throw std::runtime_error("Cannot publish final-model handoff");
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::rename(temporary, path);
}

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
