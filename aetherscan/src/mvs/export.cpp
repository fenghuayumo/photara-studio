#include "mvs/export.hpp"

#include "io/format_version.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace aetherscan::mvs {
namespace {

enum class PlyScalar {
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    float32,
    float64,
};

struct PlyProperty {
    std::string name;
    PlyScalar value_type{};
    bool list{};
    PlyScalar count_type{};
};

[[nodiscard]] PlyScalar parse_ply_scalar(const std::string& name) {
    if (name == "char" || name == "int8") return PlyScalar::int8;
    if (name == "uchar" || name == "uint8") return PlyScalar::uint8;
    if (name == "short" || name == "int16") return PlyScalar::int16;
    if (name == "ushort" || name == "uint16") return PlyScalar::uint16;
    if (name == "int" || name == "int32") return PlyScalar::int32;
    if (name == "uint" || name == "uint32") return PlyScalar::uint32;
    if (name == "float" || name == "float32") return PlyScalar::float32;
    if (name == "double" || name == "float64") return PlyScalar::float64;
    throw std::runtime_error("Unsupported PLY scalar type: " + name);
}

[[nodiscard]] std::size_t ply_scalar_size(const PlyScalar type) noexcept {
    switch (type) {
    case PlyScalar::int8:
    case PlyScalar::uint8: return 1;
    case PlyScalar::int16:
    case PlyScalar::uint16: return 2;
    case PlyScalar::int32:
    case PlyScalar::uint32:
    case PlyScalar::float32: return 4;
    case PlyScalar::float64: return 8;
    }
    return 0;
}

template <class T>
[[nodiscard]] T read_ply_binary_value(
    const std::vector<char>& bytes, std::size_t& offset) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
        throw std::runtime_error("Unexpected end of binary PLY payload");
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    if constexpr (std::endian::native == std::endian::big) {
        std::array<char, sizeof(T)> swapped{};
        std::memcpy(swapped.data(), &value, sizeof(T));
        std::reverse(swapped.begin(), swapped.end());
        std::memcpy(&value, swapped.data(), sizeof(T));
    }
    return value;
}

[[nodiscard]] double read_ply_binary_scalar(
    const std::vector<char>& bytes, std::size_t& offset,
    const PlyScalar type) {
    switch (type) {
    case PlyScalar::int8:
        return read_ply_binary_value<std::int8_t>(bytes, offset);
    case PlyScalar::uint8:
        return read_ply_binary_value<std::uint8_t>(bytes, offset);
    case PlyScalar::int16:
        return read_ply_binary_value<std::int16_t>(bytes, offset);
    case PlyScalar::uint16:
        return read_ply_binary_value<std::uint16_t>(bytes, offset);
    case PlyScalar::int32:
        return read_ply_binary_value<std::int32_t>(bytes, offset);
    case PlyScalar::uint32:
        return read_ply_binary_value<std::uint32_t>(bytes, offset);
    case PlyScalar::float32:
        return read_ply_binary_value<float>(bytes, offset);
    case PlyScalar::float64:
        return read_ply_binary_value<double>(bytes, offset);
    }
    return 0.0;
}

[[nodiscard]] bool ply_integer_type(const PlyScalar type) noexcept {
    return type != PlyScalar::float32 && type != PlyScalar::float64;
}

void assign_dense_scalar(
    DensePoint& point, const std::string& name, const double raw,
    const PlyScalar type) {
    const float value = static_cast<float>(raw);
    if (name == "x") point.position.x() = value;
    else if (name == "y") point.position.y() = value;
    else if (name == "z") point.position.z() = value;
    else if (name == "nx") point.normal.x() = value;
    else if (name == "ny") point.normal.y() = value;
    else if (name == "nz") point.normal.z() = value;
    else if (name == "weight") point.weight = value;
    else if (name == "red" || name == "green" || name == "blue") {
        const int channel = name == "red" ? 0 : name == "green" ? 1 : 2;
        point.color[channel] = ply_integer_type(type) || value > 1.F
            ? value / 255.F
            : value;
    }
}

void write_ply_header(
    std::ostream& out, const std::size_t vertices, const std::size_t faces,
    const bool has_color, const bool has_normal, const bool has_visibility = false) {
    out << "ply\nformat binary_little_endian 1.0\n";
    out << "element vertex " << vertices << '\n';
    out << "property float x\nproperty float y\nproperty float z\n";
    if (has_normal)
        out << "property float nx\nproperty float ny\nproperty float nz\n";
    if (has_color)
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    if (has_visibility)
        out << "property float weight\nproperty list uint uint view_indices\n"
               "property list uint float view_weights\n";
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

DenseCloud load_dense_ply(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open dense PLY: " + path.string());
    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("Invalid PLY header: " + path.string());
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "ply")
        throw std::runtime_error("Invalid PLY header: " + path.string());

    bool ascii = false;
    bool binary_little = false;
    bool in_vertex = false;
    bool vertex_is_first_element = true;
    bool saw_element = false;
    bool saw_end_header = false;
    std::size_t vertex_count = 0;
    std::vector<PlyProperty> properties;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;
        if (keyword == "format") {
            std::string format;
            tokens >> format;
            ascii = format == "ascii";
            binary_little = format == "binary_little_endian";
            if (!ascii && !binary_little)
                throw std::runtime_error(
                    "Dense PLY must be ASCII or binary little endian");
        } else if (keyword == "element") {
            std::string name;
            std::size_t count = 0;
            tokens >> name >> count;
            in_vertex = name == "vertex";
            if (in_vertex) {
                vertex_is_first_element = !saw_element;
                vertex_count = count;
                properties.clear();
            }
            saw_element = true;
        } else if (keyword == "property" && in_vertex) {
            std::string type;
            tokens >> type;
            PlyProperty property;
            if (type == "list") {
                std::string count_type;
                std::string value_type;
                tokens >> count_type >> value_type >> property.name;
                property.list = true;
                property.count_type = parse_ply_scalar(count_type);
                property.value_type = parse_ply_scalar(value_type);
            } else {
                tokens >> property.name;
                property.value_type = parse_ply_scalar(type);
            }
            properties.push_back(std::move(property));
        } else if (keyword == "end_header") {
            saw_end_header = true;
            break;
        }
    }
    if (!saw_end_header || (!ascii && !binary_little) || vertex_count == 0 ||
        properties.empty())
        throw std::runtime_error("PLY has no readable vertex element: " + path.string());
    if (!vertex_is_first_element)
        throw std::runtime_error("PLY vertex element must precede other data elements");
    const auto has_property = [&](const char* name) {
        return std::any_of(
            properties.begin(), properties.end(),
            [name](const PlyProperty& property) {
                return !property.list && property.name == name;
            });
    };
    if (!has_property("x") || !has_property("y") || !has_property("z"))
        throw std::runtime_error("Dense PLY is missing x/y/z properties");

    DenseCloud cloud;
    cloud.points.reserve(vertex_count);
    std::vector<char> payload;
    std::size_t offset = 0;
    if (binary_little) {
        const auto start = in.tellg();
        in.seekg(0, std::ios::end);
        const auto end = in.tellg();
        if (start < 0 || end < start)
            throw std::runtime_error("Unable to size binary PLY payload");
        payload.resize(static_cast<std::size_t>(end - start));
        in.seekg(start);
        if (!payload.empty())
            in.read(payload.data(), static_cast<std::streamsize>(payload.size()));
        if (!in) throw std::runtime_error("Failed to read binary PLY payload");
    }

    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        DensePoint point;
        point.color = Vec3f::Constant(0.5F);
        for (const PlyProperty& property : properties) {
            if (!property.list) {
                double value = 0.0;
                if (ascii) {
                    if (!(in >> value))
                        throw std::runtime_error("Unexpected end of ASCII PLY payload");
                } else {
                    value = read_ply_binary_scalar(
                        payload, offset, property.value_type);
                }
                assign_dense_scalar(point, property.name, value, property.value_type);
                continue;
            }

            double count_value = 0.0;
            if (ascii) {
                if (!(in >> count_value))
                    throw std::runtime_error("Invalid ASCII PLY list count");
            } else {
                count_value = read_ply_binary_scalar(
                    payload, offset, property.count_type);
            }
            if (count_value < 0.0 || count_value > 100'000'000.0)
                throw std::runtime_error("Invalid PLY list length");
            const auto count = static_cast<std::size_t>(count_value);
            const bool read_views = property.name == "view_indices";
            const bool read_weights = property.name == "view_weights";
            if (read_views) point.views.reserve(count);
            if (read_weights) point.view_weights.reserve(count);
            for (std::size_t item = 0; item < count; ++item) {
                double value = 0.0;
                if (ascii) {
                    if (!(in >> value))
                        throw std::runtime_error("Invalid ASCII PLY list value");
                } else {
                    value = read_ply_binary_scalar(
                        payload, offset, property.value_type);
                }
                if (read_views && value >= 0.0 &&
                    value <= static_cast<double>((std::numeric_limits<Index>::max)()))
                    point.views.push_back(static_cast<Index>(value));
                else if (read_weights && std::isfinite(value))
                    point.view_weights.push_back(static_cast<float>(value));
            }
        }
        if (!point.position.allFinite() ||
            (point.position.array().abs() >= 1e10F).any())
            continue;
        point.color = point.color.cwiseMax(0.F).cwiseMin(1.F);
        if (!point.normal.allFinite()) point.normal.setZero();
        if (point.view_weights.size() != point.views.size())
            point.view_weights.clear();
        cloud.points.push_back(std::move(point));
    }
    if (cloud.points.empty())
        throw std::runtime_error("Dense PLY has no finite points: " + path.string());
    return cloud;
}

Mesh load_mesh_ply(const std::filesystem::path& path) {
    // Reuse the point loader for its validated vertex conversion, then make a
    // second streaming pass for the face element. Mesh PLYs are small enough
    // compared with the dense cloud that this keeps the parser straightforward
    // without adding a second set of subtly different vertex conversions.
    DenseCloud cloud = load_dense_ply(path);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open mesh PLY: " + path.string());

    std::string line;
    if (!std::getline(in, line))
        throw std::runtime_error("Invalid mesh PLY header: " + path.string());
    bool ascii = false;
    bool binary_little = false;
    bool in_vertex = false;
    bool in_face = false;
    bool saw_end_header = false;
    std::size_t vertex_count = 0;
    std::size_t face_count = 0;
    std::vector<PlyProperty> vertex_properties;
    std::vector<PlyProperty> face_properties;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;
        if (keyword == "format") {
            std::string format;
            tokens >> format;
            ascii = format == "ascii";
            binary_little = format == "binary_little_endian";
            if (!ascii && !binary_little)
                throw std::runtime_error(
                    "Mesh PLY must be ASCII or binary little endian");
        } else if (keyword == "element") {
            std::string name;
            std::size_t count = 0;
            tokens >> name >> count;
            in_vertex = name == "vertex";
            in_face = name == "face";
            if (in_vertex) vertex_count = count;
            if (in_face) face_count = count;
        } else if (keyword == "property" && (in_vertex || in_face)) {
            std::string type;
            tokens >> type;
            PlyProperty property;
            if (type == "list") {
                std::string count_type;
                std::string value_type;
                tokens >> count_type >> value_type >> property.name;
                property.list = true;
                property.count_type = parse_ply_scalar(count_type);
                property.value_type = parse_ply_scalar(value_type);
            } else {
                tokens >> property.name;
                property.value_type = parse_ply_scalar(type);
            }
            (in_vertex ? vertex_properties : face_properties)
                .push_back(std::move(property));
        } else if (keyword == "end_header") {
            saw_end_header = true;
            break;
        }
    }
    if (!saw_end_header || (!ascii && !binary_little) ||
        vertex_count != cloud.points.size() || face_count == 0 ||
        face_properties.empty())
        throw std::runtime_error(
            "PLY has no readable mesh face element: " + path.string());

    std::vector<char> payload;
    std::size_t offset = 0;
    if (binary_little) {
        const auto start = in.tellg();
        in.seekg(0, std::ios::end);
        const auto end = in.tellg();
        if (start < 0 || end < start)
            throw std::runtime_error("Unable to size binary mesh PLY payload");
        payload.resize(static_cast<std::size_t>(end - start));
        in.seekg(start);
        if (!payload.empty())
            in.read(payload.data(), static_cast<std::streamsize>(payload.size()));
        if (!in) throw std::runtime_error("Failed to read binary mesh PLY payload");
    }
    const auto read_scalar = [&](const PlyScalar type) {
        double value = 0.0;
        if (ascii) {
            if (!(in >> value))
                throw std::runtime_error("Unexpected end of ASCII mesh PLY");
        } else {
            value = read_ply_binary_scalar(payload, offset, type);
        }
        return value;
    };
    const auto read_property = [&](const PlyProperty& property) {
        std::vector<double> values;
        if (!property.list) {
            values.push_back(read_scalar(property.value_type));
            return values;
        }
        const double count_value = read_scalar(property.count_type);
        if (!std::isfinite(count_value) || count_value < 0.0 ||
            count_value > 100'000'000.0)
            throw std::runtime_error("Invalid mesh PLY list length");
        const auto count = static_cast<std::size_t>(count_value);
        values.reserve(count);
        for (std::size_t item = 0; item < count; ++item)
            values.push_back(read_scalar(property.value_type));
        return values;
    };

    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex)
        for (const PlyProperty& property : vertex_properties)
            static_cast<void>(read_property(property));

    Mesh mesh;
    mesh.vertices.reserve(vertex_count);
    const auto has_vertex_property = [&](const char* name) {
        return std::any_of(
            vertex_properties.begin(), vertex_properties.end(),
            [name](const PlyProperty& property) {
                return !property.list && property.name == name;
            });
    };
    const bool has_normals =
        has_vertex_property("nx") && has_vertex_property("ny") &&
        has_vertex_property("nz");
    const bool has_colors =
        has_vertex_property("red") && has_vertex_property("green") &&
        has_vertex_property("blue");
    if (has_normals) mesh.normals.reserve(vertex_count);
    if (has_colors) mesh.colors.reserve(vertex_count);
    for (DensePoint& point : cloud.points) {
        mesh.vertices.push_back(point.position);
        if (has_normals) mesh.normals.push_back(point.normal);
        if (has_colors) mesh.colors.push_back(point.color);
    }

    mesh.faces.reserve(face_count);
    for (std::size_t face = 0; face < face_count; ++face) {
        std::vector<Index> polygon;
        for (const PlyProperty& property : face_properties) {
            std::vector<double> values = read_property(property);
            if (!property.list ||
                (property.name != "vertex_indices" &&
                 property.name != "vertex_index"))
                continue;
            polygon.reserve(values.size());
            for (const double value : values) {
                if (!std::isfinite(value) || value < 0.0 ||
                    value >= static_cast<double>(vertex_count) ||
                    value > static_cast<double>(
                        (std::numeric_limits<Index>::max)()))
                    throw std::runtime_error("Mesh PLY face index is out of range");
                polygon.push_back(static_cast<Index>(value));
            }
        }
        if (polygon.size() < 3) continue;
        for (std::size_t corner = 1; corner + 1 < polygon.size(); ++corner)
            mesh.faces.emplace_back(
                static_cast<int>(polygon[0]),
                static_cast<int>(polygon[corner]),
                static_cast<int>(polygon[corner + 1]));
    }
    if (mesh.faces.empty())
        throw std::runtime_error("Mesh PLY contains no valid triangles");
    return mesh;
}

void save_subject_bounds(
    const OrientedBoundingBox& bounds, const std::filesystem::path& path) {
    if (!bounds.valid)
        throw std::invalid_argument("Cannot save invalid SubjectBounds");
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error(
            "Failed to create SubjectBounds: " + path.string());
    out.precision(9);
    out << bounds.center.x() << ' ' << bounds.center.y() << ' '
        << bounds.center.z();
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            out << ' ' << bounds.axes(row, column);
    out << ' ' << bounds.half_extent.x() << ' '
        << bounds.half_extent.y() << ' ' << bounds.half_extent.z() << '\n';
}

bool load_subject_bounds(
    OrientedBoundingBox& bounds, const std::filesystem::path& path) {
    bounds = {};
    std::ifstream in(path);
    if (!in) return false;
    OrientedBoundingBox loaded;
    in >> loaded.center.x() >> loaded.center.y() >> loaded.center.z();
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            in >> loaded.axes(row, column);
    in >> loaded.half_extent.x() >> loaded.half_extent.y() >>
        loaded.half_extent.z();
    if (!in) return false;
    loaded.valid = loaded.center.allFinite() &&
        loaded.half_extent.allFinite() &&
        (loaded.half_extent.array() > 0.F).all() &&
        loaded.axes.allFinite();
    if (!loaded.valid) return false;
    bounds = loaded;
    return true;
}

void save_dense_ply(const DenseCloud& cloud, const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to create PLY: " + path.string());
    const bool has_color = !cloud.points.empty();
    const bool has_normal = !cloud.points.empty();
    write_ply_header(out, cloud.points.size(), 0, has_color, has_normal, true);
    for (const auto& p : cloud.points) {
        for (int k = 0; k < 3; ++k) write_little_endian(out, p.position[k]);
        for (int k = 0; k < 3; ++k) write_little_endian(out, p.normal[k]);
        write_little_endian(out, to_u8(p.color.x()));
        write_little_endian(out, to_u8(p.color.y()));
        write_little_endian(out, to_u8(p.color.z()));
        write_little_endian(out, p.weight);
        write_little_endian(out, static_cast<std::uint32_t>(p.views.size()));
        for (const auto view : p.views) write_little_endian(out, static_cast<std::uint32_t>(view));
        write_little_endian(out, static_cast<std::uint32_t>(p.view_weights.size()));
        for (const float weight : p.view_weights) write_little_endian(out, weight);
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

std::vector<std::uint8_t> encode_mesh(const Mesh& mesh) {
    const bool has_normals = mesh.normals.size() == mesh.vertices.size();
    const bool has_colors = mesh.colors.size() == mesh.vertices.size();
    std::uint32_t flags = 0;
    if (has_normals) flags |= 1U;
    if (has_colors) flags |= 2U;
    const std::size_t vertex_bytes =
        mesh.vertices.size() * sizeof(float) * 3 +
        (has_normals ? mesh.normals.size() * sizeof(float) * 3 : 0) +
        (has_colors ? mesh.colors.size() * sizeof(float) * 3 : 0);
    const std::size_t face_bytes = mesh.faces.size() * sizeof(std::int32_t) * 3;
    std::vector<std::uint8_t> bytes(
        4 + 8 + 8 + 4 + vertex_bytes + face_bytes);
    std::uint8_t* cursor = bytes.data();
    const auto append = [&](const void* data, const std::size_t size) {
        std::memcpy(cursor, data, size);
        cursor += size;
    };
    const std::uint32_t version = k_mesh_chunk_version;
    const std::uint64_t vertex_count = mesh.vertices.size();
    const std::uint64_t face_count = mesh.faces.size();
    append(&version, sizeof(version));
    append(&vertex_count, sizeof(vertex_count));
    append(&face_count, sizeof(face_count));
    append(&flags, sizeof(flags));
    for (const auto& vertex : mesh.vertices)
        append(vertex.data(), sizeof(float) * 3);
    if (has_normals)
        for (const auto& normal : mesh.normals)
            append(normal.data(), sizeof(float) * 3);
    if (has_colors)
        for (const auto& color : mesh.colors)
            append(color.data(), sizeof(float) * 3);
    for (const auto& face : mesh.faces) {
        const std::int32_t indices[3] = {face.x(), face.y(), face.z()};
        append(indices, sizeof(indices));
    }
    return bytes;
}

Mesh decode_mesh(const std::span<const std::uint8_t> bytes) {
    if (bytes.size() < 4 + 8 + 8 + 4)
        throw std::runtime_error("Mesh chunk is too small");
    const std::uint8_t* cursor = bytes.data();
    const std::uint8_t* end = bytes.data() + bytes.size();
    const auto take = [&](const std::size_t size) {
        if (cursor + size > end)
            throw std::runtime_error("Truncated mesh chunk");
        const std::uint8_t* data = cursor;
        cursor += size;
        return data;
    };
    std::uint32_t version = 0;
    std::memcpy(&version, take(sizeof(version)), sizeof(version));
    if (version == 0 || version > k_mesh_chunk_version)
        throw std::runtime_error(io::unsupported_payload_version(
            "Mesh chunk", version, k_mesh_chunk_version));
    std::uint64_t vertex_count = 0;
    std::uint64_t face_count = 0;
    std::uint32_t flags = 0;
    std::memcpy(&vertex_count, take(sizeof(vertex_count)), sizeof(vertex_count));
    std::memcpy(&face_count, take(sizeof(face_count)), sizeof(face_count));
    std::memcpy(&flags, take(sizeof(flags)), sizeof(flags));
    if (vertex_count > 100'000'000ULL || face_count > 200'000'000ULL)
        throw std::runtime_error("Mesh chunk counts are unreasonable");
    Mesh mesh;
    mesh.vertices.resize(static_cast<std::size_t>(vertex_count));
    for (auto& vertex : mesh.vertices)
        std::memcpy(vertex.data(), take(sizeof(float) * 3), sizeof(float) * 3);
    if ((flags & 1U) != 0) {
        mesh.normals.resize(mesh.vertices.size());
        for (auto& normal : mesh.normals)
            std::memcpy(normal.data(), take(sizeof(float) * 3), sizeof(float) * 3);
    }
    if ((flags & 2U) != 0) {
        mesh.colors.resize(mesh.vertices.size());
        for (auto& color : mesh.colors)
            std::memcpy(color.data(), take(sizeof(float) * 3), sizeof(float) * 3);
    }
    mesh.faces.resize(static_cast<std::size_t>(face_count));
    for (auto& face : mesh.faces) {
        std::int32_t indices[3]{};
        std::memcpy(indices, take(sizeof(indices)), sizeof(indices));
        face = Eigen::Vector3i(indices[0], indices[1], indices[2]);
    }
    return mesh;
}

}  // namespace aetherscan::mvs
