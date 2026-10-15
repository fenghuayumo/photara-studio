#include "mvs/mesh_handoff.hpp"
#include "mvs/export.hpp"
#include "io/shared_mapping.hpp"

namespace photara::mvs {
namespace {
constexpr std::size_t kHeader = 24;
constexpr std::uint64_t kMaxVertices = 100'000'000;
constexpr std::uint64_t kMaxFaces = 200'000'000;
constexpr std::uint64_t kMaxBytes = kHeader + kMaxVertices * 36ULL + kMaxFaces * 12ULL;

void validate(const Mesh& mesh) {
    const auto count = mesh.vertices.size();
    if (!count || count > kMaxVertices || mesh.faces.empty() || mesh.faces.size() > kMaxFaces ||
        (!mesh.normals.empty() && mesh.normals.size() != count) ||
        (!mesh.colors.empty() && mesh.colors.size() != count))
        throw std::runtime_error("Invalid mesh handoff dimensions");
    for (const auto* values : {&mesh.vertices, &mesh.normals, &mesh.colors})
        for (const auto& value : *values)
            if (!value.allFinite()) throw std::runtime_error("Non-finite mesh handoff attribute");
    for (const auto& f : mesh.faces)
        for (int k = 0; k < 3; ++k)
            if (f[k] < 0 || static_cast<std::size_t>(f[k]) >= count)
                throw std::runtime_error("Invalid mesh handoff face index");
}
}  // namespace

struct MeshHandoff::Impl {
    io::detail::Mapping mapping;
    std::filesystem::path ready;
    std::string name;
};

MeshHandoff::MeshHandoff(const Mesh& mesh, const std::filesystem::path& ready)
    : impl_(std::make_unique<Impl>()) {
    validate(mesh);
    const std::uint64_t vertices = mesh.vertices.size(), faces = mesh.faces.size();
    const std::uint32_t version = k_mesh_chunk_version;
    const std::uint32_t flags = (!mesh.normals.empty() ? 1U : 0U) | (!mesh.colors.empty() ? 2U : 0U);
    const std::size_t bytes = kHeader + vertices * (12ULL + ((flags & 1U) ? 12ULL : 0ULL) +
        ((flags & 2U) ? 12ULL : 0ULL)) + faces * 12ULL;
    const auto token = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
#ifdef _WIN32
    impl_->name = "Local\\Photara.mesh." + std::to_string(GetCurrentProcessId()) + "." + token;
#else
    impl_->name = "/photara_mesh_" + std::to_string(getpid()) + "_" + token;
#endif
    impl_->ready = ready;
    impl_->mapping.open(impl_->name, bytes, true);
    auto* cursor = static_cast<std::uint8_t*>(impl_->mapping.data);
    const auto append = [&](const void* data, std::size_t size) {
        std::memcpy(cursor, data, size);
        cursor += size;
    };
    append(&version, 4); append(&vertices, 8); append(&faces, 8); append(&flags, 4);
    for (const auto* values : {&mesh.vertices, &mesh.normals, &mesh.colors})
        for (const auto& v : *values) append(v.data(), 12);
    for (const auto& f : mesh.faces) {
        const std::int32_t indices[3] = {f.x(), f.y(), f.z()};
        append(indices, 12);
    }
    if (!ready.parent_path().empty()) std::filesystem::create_directories(ready.parent_path());
    io::detail::publish_text(ready, "PHOTARA_MESH_HANDOFF 1\n" + impl_->name + "\n" +
        std::to_string(bytes) + "\n");
}

MeshHandoff::~MeshHandoff() = default;

bool MeshHandoff::wait_for_reader(std::chrono::milliseconds timeout) const {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        std::ifstream in(io::detail::ack_path(impl_->ready));
        std::string name;
        if (std::getline(in, name) && name == impl_->name) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

Mesh receive_mesh_handoff(const std::filesystem::path& ready) {
    std::ifstream in(ready);
    std::string magic, name;
    std::uint64_t bytes{};
    if (!std::getline(in, magic) || magic != "PHOTARA_MESH_HANDOFF 1" ||
        !std::getline(in, name) || !(in >> bytes) || bytes < kHeader || bytes > kMaxBytes ||
        bytes > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error("Invalid mesh handoff manifest");
    io::detail::Mapping mapping;
    mapping.open(name, static_cast<std::size_t>(bytes), false);
    const auto* data = static_cast<const std::uint8_t*>(mapping.data);
    std::uint32_t version{}, flags{};
    std::uint64_t vertices{}, faces{};
    std::memcpy(&version, data, 4); std::memcpy(&vertices, data + 4, 8);
    std::memcpy(&faces, data + 12, 8); std::memcpy(&flags, data + 20, 4);
    if (version != k_mesh_chunk_version || !vertices || vertices > kMaxVertices ||
        !faces || faces > kMaxFaces || (flags & ~3U))
        throw std::runtime_error("Invalid mesh handoff header");
    const auto expected = kHeader + vertices * (12ULL + ((flags & 1U) ? 12ULL : 0ULL) +
        ((flags & 2U) ? 12ULL : 0ULL)) + faces * 12ULL;
    if (bytes != expected) throw std::runtime_error("Invalid mesh handoff payload length");
    io::detail::publish_text(io::detail::ack_path(ready), name + "\n");
    auto mesh = decode_mesh(std::span<const std::uint8_t>(data, static_cast<std::size_t>(bytes)));
    validate(mesh);
    return mesh;
}
}  // namespace photara::mvs
