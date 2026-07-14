#include "mvs/export.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace aetherscan::mvs {
namespace {

void write_ply_header(
    std::ostream& out, const std::size_t vertices, const std::size_t faces,
    const bool has_color, const bool has_normal) {
    out << "ply\nformat binary_little_endian 1.0\n";
    out << "element vertex " << vertices << '\n';
    out << "property float x\nproperty float y\nproperty float z\n";
    if (has_normal)
        out << "property float nx\nproperty float ny\nproperty float nz\n";
    if (has_color)
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    if (faces > 0) {
        out << "element face " << faces << '\n';
        out << "property list uchar int vertex_indices\n";
    }
    out << "end_header\n";
}

[[nodiscard]] std::uint8_t to_u8(const float v) {
    return static_cast<std::uint8_t>(
        std::clamp(static_cast<int>(std::lround(v * 255.F)), 0, 255));
}

template <class T>
void write_little_endian(std::ostream& out, const T value) {
    std::array<char, sizeof(T)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(T));
    if constexpr (std::endian::native == std::endian::big)
        std::reverse(bytes.begin(), bytes.end());
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

void save_dense_ply(const DenseCloud& cloud, const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to create PLY: " + path.string());
    const bool has_color = !cloud.points.empty();
    const bool has_normal = !cloud.points.empty();
    write_ply_header(out, cloud.points.size(), 0, has_color, has_normal);
    for (const auto& p : cloud.points) {
        for (int k = 0; k < 3; ++k) write_little_endian(out, p.position[k]);
        for (int k = 0; k < 3; ++k) write_little_endian(out, p.normal[k]);
        write_little_endian(out, to_u8(p.color.x()));
        write_little_endian(out, to_u8(p.color.y()));
        write_little_endian(out, to_u8(p.color.z()));
    }
}

void save_mesh_ply(const Mesh& mesh, const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to create PLY: " + path.string());
    const bool has_color = mesh.colors.size() == mesh.vertices.size();
    const bool has_normal = mesh.normals.size() == mesh.vertices.size();
    write_ply_header(
        out, mesh.vertices.size(), mesh.faces.size(), has_color, has_normal);
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        const auto& v = mesh.vertices[i];
        for (int k = 0; k < 3; ++k) write_little_endian(out, v[k]);
        if (has_normal) {
            const auto& n = mesh.normals[i];
            for (int k = 0; k < 3; ++k) write_little_endian(out, n[k]);
        }
        if (has_color) {
            const auto& c = mesh.colors[i];
            write_little_endian(out, to_u8(c.x()));
            write_little_endian(out, to_u8(c.y()));
            write_little_endian(out, to_u8(c.z()));
        }
    }
    for (const auto& f : mesh.faces) {
        write_little_endian(out, static_cast<std::uint8_t>(3));
        for (int k = 0; k < 3; ++k)
            write_little_endian(out, static_cast<std::int32_t>(f[k]));
    }
}

void save_mesh_obj(const Mesh& mesh, const std::filesystem::path& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to create OBJ: " + path.string());
    out << "# AetherScan MVS mesh\n";
    for (const auto& v : mesh.vertices)
        out << "v " << v.x() << ' ' << v.y() << ' ' << v.z() << '\n';
    if (mesh.normals.size() == mesh.vertices.size()) {
        for (const auto& n : mesh.normals)
            out << "vn " << n.x() << ' ' << n.y() << ' ' << n.z() << '\n';
        for (const auto& f : mesh.faces) {
            out << "f " << (f[0] + 1) << "//" << (f[0] + 1) << ' '
                << (f[1] + 1) << "//" << (f[1] + 1) << ' ' << (f[2] + 1)
                << "//" << (f[2] + 1) << '\n';
        }
    } else {
        for (const auto& f : mesh.faces)
            out << "f " << (f[0] + 1) << ' ' << (f[1] + 1) << ' ' << (f[2] + 1)
                << '\n';
    }
}

void save_depth_map(const DepthMap& depth, const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to create depth map: " + path.string());
    const char magic[8] = {'A', 'D', 'M', 'A', 'P', '0', '0', '1'};
    out.write(magic, 8);
    const std::uint32_t meta[5] = {
        depth.view_id, depth.width, depth.height,
        static_cast<std::uint32_t>(depth.depth.size()),
        static_cast<std::uint32_t>(depth.normal.size())};
    out.write(reinterpret_cast<const char*>(meta), sizeof(meta));
    out.write(reinterpret_cast<const char*>(&depth.depth_min), sizeof(float));
    out.write(reinterpret_cast<const char*>(&depth.depth_max), sizeof(float));
    out.write(
        reinterpret_cast<const char*>(depth.depth.data()),
        static_cast<std::streamsize>(depth.depth.size() * sizeof(float)));
    out.write(
        reinterpret_cast<const char*>(depth.confidence.data()),
        static_cast<std::streamsize>(depth.confidence.size() * sizeof(float)));
    out.write(
        reinterpret_cast<const char*>(depth.normal.data()),
        static_cast<std::streamsize>(depth.normal.size() * sizeof(Vec3f)));
}

bool load_depth_map(DepthMap& depth, const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[8]{};
    in.read(magic, 8);
    if (std::string(magic, 8) != "ADMAP001") return false;
    std::uint32_t meta[5]{};
    in.read(reinterpret_cast<char*>(meta), sizeof(meta));
    depth.view_id = meta[0];
    depth.width = meta[1];
    depth.height = meta[2];
    depth.depth.resize(meta[3]);
    depth.normal.resize(meta[4]);
    depth.confidence.resize(meta[3]);
    in.read(reinterpret_cast<char*>(&depth.depth_min), sizeof(float));
    in.read(reinterpret_cast<char*>(&depth.depth_max), sizeof(float));
    in.read(
        reinterpret_cast<char*>(depth.depth.data()),
        static_cast<std::streamsize>(depth.depth.size() * sizeof(float)));
    in.read(
        reinterpret_cast<char*>(depth.confidence.data()),
        static_cast<std::streamsize>(depth.confidence.size() * sizeof(float)));
    in.read(
        reinterpret_cast<char*>(depth.normal.data()),
        static_cast<std::streamsize>(depth.normal.size() * sizeof(Vec3f)));
    return static_cast<bool>(in);
}

}  // namespace aetherscan::mvs
