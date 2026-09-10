#include "io/mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace aetherscan::io {
namespace {

using Bytes = std::vector<std::uint8_t>;
using mvs::Mesh;
using mvs::Vec2f;
using mvs::Vec3f;

constexpr std::uint32_t k_glb_magic = 0x46546c67U;
constexpr std::uint32_t k_json_chunk = 0x4e4f534aU;
constexpr std::uint32_t k_bin_chunk = 0x004e4942U;
constexpr int k_component_float = 5126;
constexpr int k_component_uint = 5125;
constexpr int k_target_array = 34962;
constexpr int k_target_element = 34963;
constexpr int k_triangles = 4;

void append_u32(Bytes& bytes, const std::uint32_t value) {
    for (unsigned byte = 0; byte < 4; ++byte)
        bytes.push_back(static_cast<std::uint8_t>(value >> (8U * byte)));
}

void pad4(Bytes& bytes) {
    while (bytes.size() % 4U != 0U) bytes.push_back(0U);
}

void append_bytes(Bytes& bytes, const void* data, const std::size_t size) {
    const auto* begin = static_cast<const std::uint8_t*>(data);
    bytes.insert(bytes.end(), begin, begin + size);
}

void append_f32(Bytes& bytes, const float value) {
    append_bytes(bytes, &value, sizeof(value));
}

struct BufferView {
    std::size_t offset{};
    std::size_t length{};
    int target{};
};

struct Accessor {
    int view{};
    int component_type{};
    std::size_t count{};
    const char* type{};
    bool has_range{};
    std::array<float, 3> minimum{};
    std::array<float, 3> maximum{};
};

int add_view(
    Bytes& binary, std::vector<BufferView>& views, const std::size_t length,
    const int target) {
    pad4(binary);
    BufferView view;
    view.offset = binary.size();
    view.length = length;
    view.target = target;
    views.push_back(view);
    return static_cast<int>(views.size() - 1U);
}

void write_positions(
    Bytes& binary, std::vector<BufferView>& views,
    std::vector<Accessor>& accessors, const Mesh& mesh) {
    const std::size_t count = mesh.vertices.size();
    const int view = add_view(
        binary, views, count * 3U * sizeof(float), k_target_array);
    Accessor accessor;
    accessor.view = view;
    accessor.component_type = k_component_float;
    accessor.count = count;
    accessor.type = "VEC3";
    accessor.has_range = true;
    accessor.minimum = {
        std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    accessor.maximum = {
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    for (const auto& vertex : mesh.vertices) {
        for (int axis = 0; axis < 3; ++axis) {
            const float value = vertex[axis];
            if (!std::isfinite(value))
                throw std::runtime_error("Mesh GLB position is not finite");
            append_f32(binary, value);
            accessor.minimum[static_cast<std::size_t>(axis)] = std::min(
                accessor.minimum[static_cast<std::size_t>(axis)], value);
            accessor.maximum[static_cast<std::size_t>(axis)] = std::max(
                accessor.maximum[static_cast<std::size_t>(axis)], value);
        }
    }
    accessors.push_back(accessor);
}

void write_vec3_attribute(
    Bytes& binary, std::vector<BufferView>& views,
    std::vector<Accessor>& accessors, const std::vector<Vec3f>& values) {
    const int view = add_view(
        binary, views, values.size() * 3U * sizeof(float), k_target_array);
    Accessor accessor;
    accessor.view = view;
    accessor.component_type = k_component_float;
    accessor.count = values.size();
    accessor.type = "VEC3";
    for (const auto& value : values) {
        for (int axis = 0; axis < 3; ++axis) {
            const float component = value[axis];
            if (!std::isfinite(component))
                throw std::runtime_error("Mesh GLB attribute is not finite");
            append_f32(binary, component);
        }
    }
    accessors.push_back(accessor);
}

void write_uvs(
    Bytes& binary, std::vector<BufferView>& views,
    std::vector<Accessor>& accessors, std::span<const Vec2f> uvs) {
    const int view = add_view(
        binary, views, uvs.size() * 2U * sizeof(float), k_target_array);
    Accessor accessor;
    accessor.view = view;
    accessor.component_type = k_component_float;
    accessor.count = uvs.size();
    accessor.type = "VEC2";
    for (const auto& uv : uvs) {
        if (!std::isfinite(uv.x()) || !std::isfinite(uv.y()))
            throw std::runtime_error("Mesh GLB UV is not finite");
        append_f32(binary, uv.x());
        append_f32(binary, uv.y());
    }
    accessors.push_back(accessor);
}

void write_indices(
    Bytes& binary, std::vector<BufferView>& views,
    std::vector<Accessor>& accessors, const Mesh& mesh) {
    const std::size_t count = mesh.faces.size() * 3U;
    const int view = add_view(
        binary, views, count * sizeof(std::uint32_t), k_target_element);
    Accessor accessor;
    accessor.view = view;
    accessor.component_type = k_component_uint;
    accessor.count = count;
    accessor.type = "SCALAR";
    const auto vertex_count = mesh.vertices.size();
    for (const auto& face : mesh.faces) {
        for (int corner = 0; corner < 3; ++corner) {
            if (face[corner] < 0 ||
                static_cast<std::size_t>(face[corner]) >= vertex_count)
                throw std::runtime_error("Mesh GLB face index is out of range");
            const auto index = static_cast<std::uint32_t>(face[corner]);
            append_bytes(binary, &index, sizeof(index));
        }
    }
    accessors.push_back(accessor);
}

int write_png(
    Bytes& binary, std::vector<BufferView>& views,
    std::span<const std::uint8_t> png) {
    const int view = add_view(binary, views, png.size(), 0);
    append_bytes(binary, png.data(), png.size());
    pad4(binary);
    return view;
}

std::string glb_json(
    const std::vector<BufferView>& views, const std::vector<Accessor>& accessors,
    const int position, const int normal, const int color, const int uv,
    const int indices, const int image_view, const std::size_t binary_size,
    const bool textured) {
    std::ostringstream json;
    json << std::setprecision(9);
    json << "{\"asset\":{\"generator\":\"AetherScan\",\"version\":\"2.0\"}";
    if (textured) json << ",\"extensionsUsed\":[\"KHR_materials_unlit\"]";
    json << ",\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
         << "\"nodes\":[{\"mesh\":0}],\"meshes\":[{\"primitives\":[{"
         << "\"attributes\":{\"POSITION\":" << position;
    if (normal >= 0) json << ",\"NORMAL\":" << normal;
    if (color >= 0) json << ",\"COLOR_0\":" << color;
    if (uv >= 0) json << ",\"TEXCOORD_0\":" << uv;
    json << "},\"indices\":" << indices << ",\"mode\":" << k_triangles
         << ",\"material\":0}]}],\"materials\":[{";
    if (textured) {
        json << "\"pbrMetallicRoughness\":{\"baseColorFactor\":[1,1,1,1],"
             << "\"baseColorTexture\":{\"index\":0},\"metallicFactor\":0,"
             << "\"roughnessFactor\":1},"
             << "\"extensions\":{\"KHR_materials_unlit\":{}}";
    } else {
        json << "\"pbrMetallicRoughness\":{\"baseColorFactor\":[1,1,1,1],"
             << "\"metallicFactor\":0,\"roughnessFactor\":1}";
    }
    json << "}]";
    if (textured) {
        json << ",\"samplers\":[{\"magFilter\":9729,\"minFilter\":9729,"
             << "\"wrapS\":33071,\"wrapT\":33071}],"
             << "\"textures\":[{\"sampler\":0,\"source\":0}],"
             << "\"images\":[{\"bufferView\":" << image_view
             << ",\"mimeType\":\"image/png\"}]";
    }
    json << ",\"buffers\":[{\"byteLength\":" << binary_size << "}],"
         << "\"bufferViews\":[";
    for (std::size_t i = 0; i < views.size(); ++i) {
        if (i != 0U) json << ',';
        json << "{\"buffer\":0,\"byteOffset\":" << views[i].offset
             << ",\"byteLength\":" << views[i].length;
        if (views[i].target != 0)
            json << ",\"target\":" << views[i].target;
        json << '}';
    }
    json << "],\"accessors\":[";
    for (std::size_t i = 0; i < accessors.size(); ++i) {
        if (i != 0U) json << ',';
        const auto& accessor = accessors[i];
        json << "{\"bufferView\":" << accessor.view
             << ",\"componentType\":" << accessor.component_type
             << ",\"count\":" << accessor.count << ",\"type\":\""
             << accessor.type << '\"';
        if (accessor.has_range) {
            json << ",\"min\":[" << accessor.minimum[0] << ','
                 << accessor.minimum[1] << ',' << accessor.minimum[2]
                 << "],\"max\":[" << accessor.maximum[0] << ','
                 << accessor.maximum[1] << ',' << accessor.maximum[2] << ']';
        }
        json << '}';
    }
    json << "]}";
    return json.str();
}

}  // namespace

void save_mesh_glb(
    const Mesh& mesh, const std::filesystem::path& path,
    std::span<const Vec2f> uvs, std::span<const std::uint8_t> albedo_png) {
    if (mesh.vertices.empty() || mesh.faces.empty())
        throw std::runtime_error("Cannot export an empty mesh to GLB");
    if (mesh.vertices.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("Mesh GLB vertex count exceeds 4,294,967,295");

    const bool has_normals = mesh.normals.size() == mesh.vertices.size();
    const bool has_colors = mesh.colors.size() == mesh.vertices.size();
    const bool has_uvs = uvs.size() == mesh.vertices.size();
    const bool textured = has_uvs && !albedo_png.empty();

    Bytes binary;
    std::vector<BufferView> views;
    std::vector<Accessor> accessors;
    views.reserve(8);
    accessors.reserve(6);

    write_positions(binary, views, accessors, mesh);
    const int position = 0;
    int normal = -1;
    int color = -1;
    int uv = -1;
    if (has_normals) {
        write_vec3_attribute(binary, views, accessors, mesh.normals);
        normal = static_cast<int>(accessors.size() - 1U);
    }
    if (has_colors) {
        write_vec3_attribute(binary, views, accessors, mesh.colors);
        color = static_cast<int>(accessors.size() - 1U);
    }
    if (textured) {
        write_uvs(binary, views, accessors, uvs);
        uv = static_cast<int>(accessors.size() - 1U);
    }
    write_indices(binary, views, accessors, mesh);
    const int indices = static_cast<int>(accessors.size() - 1U);
    int image_view = -1;
    if (textured) image_view = write_png(binary, views, albedo_png);
    pad4(binary);

    if (binary.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("Mesh GLB binary payload exceeds 4 GiB");

    std::string json = glb_json(
        views, accessors, position, normal, color, uv, indices, image_view,
        binary.size(), textured);
    while (json.size() % 4U != 0U) json.push_back(' ');

    const std::uint64_t total =
        12ULL + 8ULL + json.size() + 8ULL + binary.size();
    if (total > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("Mesh GLB file exceeds 4 GiB");

    Bytes output;
    output.reserve(static_cast<std::size_t>(total));
    append_u32(output, k_glb_magic);
    append_u32(output, 2U);
    append_u32(output, static_cast<std::uint32_t>(total));
    append_u32(output, static_cast<std::uint32_t>(json.size()));
    append_u32(output, k_json_chunk);
    output.insert(output.end(), json.begin(), json.end());
    append_u32(output, static_cast<std::uint32_t>(binary.size()));
    append_u32(output, k_bin_chunk);
    output.insert(output.end(), binary.begin(), binary.end());

    if (!path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("Failed to create GLB: " + path.string());
    out.write(
        reinterpret_cast<const char*>(output.data()),
        static_cast<std::streamsize>(output.size()));
    if (!out) throw std::runtime_error("Failed to write GLB: " + path.string());
}

}  // namespace aetherscan::io
