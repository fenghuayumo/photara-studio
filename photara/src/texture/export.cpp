#include "texture/export.hpp"

#include "io/image.hpp"
#include "../io/binary_codec.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>

namespace photara::texture {
namespace {

[[nodiscard]] float linear_to_srgb(const float c) {
    const float x = std::clamp(c, 0.F, 1.F);
    return x <= 0.0031308F ? x * 12.92F
                           : 1.055F * std::pow(x, 1.F / 2.4F) - 0.055F;
}

[[nodiscard]] std::uint8_t to_u8(const float v) {
    return static_cast<std::uint8_t>(
        std::clamp(static_cast<int>(std::lround(v * 255.F)), 0, 255));
}

}  // namespace

void save_atlas_png(
    const TexturedMesh& mesh, const std::filesystem::path& path) {
    if (mesh.atlas_width == 0 || mesh.atlas_height == 0 ||
        mesh.atlas_rgb.size() !=
            static_cast<std::size_t>(mesh.atlas_width) * mesh.atlas_height *
                3U)
        throw std::runtime_error("Invalid atlas for PNG export");

    io::RgbImage image;
    image.width = mesh.atlas_width;
    image.height = mesh.atlas_height;
    image.pixels.resize(
        static_cast<std::size_t>(image.width) * image.height * 3U);
    for (std::size_t i = 0; i < image.pixels.size(); ++i)
        image.pixels[i] = to_u8(linear_to_srgb(mesh.atlas_rgb[i]));
    io::save_rgb_png(image, path);
}

void save_textured_obj(
    const TexturedMesh& mesh, const std::filesystem::path& stem) {
    if (mesh.positions.empty() || mesh.indices.size() < 3)
        throw std::runtime_error("Cannot export empty textured mesh");

    const auto obj_path = stem.string() + ".obj";
    const auto mtl_path = stem.string() + ".mtl";
    const auto png_name = stem.filename().string() + "_albedo.png";
    const auto png_path = stem.string() + "_albedo.png";

    save_atlas_png(mesh, png_path);

    {
        std::ofstream mtl(mtl_path);
        if (!mtl)
            throw std::runtime_error("Failed to create MTL: " + mtl_path);
        mtl << "newmtl atlas\n";
        mtl << "Ka 1.000 1.000 1.000\n";
        mtl << "Kd 1.000 1.000 1.000\n";
        mtl << "Ks 0.000 0.000 0.000\n";
        mtl << "d 1.0\n";
        mtl << "illum 1\n";
        mtl << "map_Kd " << png_name << '\n';
    }

    std::ofstream obj(obj_path);
    if (!obj) throw std::runtime_error("Failed to create OBJ: " + obj_path);
    obj << "# Photara textured mesh";
    if (mesh.delighted) obj << " (albedo / delighted)";
    obj << '\n';
    obj << "mtllib " << stem.filename().string() << ".mtl\n";
    obj << "usemtl atlas\n";

    for (const auto& v : mesh.positions)
        obj << "v " << v.x() << ' ' << v.y() << ' ' << v.z() << '\n';
    // OBJ V increases upward; UVAtlas/D3D-style V increases downward.
    for (const auto& uv : mesh.uvs)
        obj << "vt " << uv.x() << ' ' << (1.F - uv.y()) << '\n';
    if (mesh.normals.size() == mesh.positions.size()) {
        for (const auto& n : mesh.normals)
            obj << "vn " << n.x() << ' ' << n.y() << ' ' << n.z() << '\n';
    }

    const bool has_n = mesh.normals.size() == mesh.positions.size();
    for (std::size_t f = 0; f + 2 < mesh.indices.size(); f += 3) {
        const auto emit = [&](const std::uint32_t idx) {
            const int i = static_cast<int>(idx) + 1;
            if (has_n)
                obj << i << '/' << i << '/' << i;
            else
                obj << i << '/' << i;
        };
        obj << "f ";
        emit(mesh.indices[f]);
        obj << ' ';
        emit(mesh.indices[f + 1]);
        obj << ' ';
        emit(mesh.indices[f + 2]);
        obj << '\n';
    }
}

namespace {

[[nodiscard]] std::string read_binary_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Failed to read " + path.string());
    return {
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_binary_file(
    const std::filesystem::path& path, const std::string& bytes) {
    if (!path.parent_path().empty()) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("Failed to write " + path.string());
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void replace_token(std::string& text, const std::string& from, const std::string& to) {
    if (from.empty() || from == to) return;
    std::size_t at = 0;
    while ((at = text.find(from, at)) != std::string::npos) {
        text.replace(at, from.size(), to);
        at += to.size();
    }
}

void write_blob(io::binary::BufferWriter& writer, const std::string& bytes) {
    writer.value_size(bytes.size());
    writer.bytes(bytes.data(), bytes.size());
}

std::string read_blob(io::binary::BufferReader& reader) {
    std::string bytes(reader.size(1, 512ull * 1024ull * 1024ull), '\0');
    reader.bytes(bytes.data(), bytes.size());
    return bytes;
}

}  // namespace

bool textured_obj_exists(const std::filesystem::path& stem) {
    std::error_code error;
    return !stem.empty() &&
           std::filesystem::exists(textured_obj_path(stem), error) &&
           std::filesystem::exists(textured_albedo_path(stem), error);
}

void copy_textured_obj(
    const std::filesystem::path& source_stem,
    const std::filesystem::path& destination_stem) {
    if (source_stem.empty() || destination_stem.empty())
        throw std::runtime_error("Textured mesh copy requires a source and destination");
    if (!textured_obj_exists(source_stem))
        throw std::runtime_error(
            "No textured mesh at " + textured_obj_path(source_stem).string());

    std::string obj = read_binary_file(textured_obj_path(source_stem));
    std::string mtl;
    std::error_code mtl_error;
    if (std::filesystem::exists(textured_mtl_path(source_stem), mtl_error))
        mtl = read_binary_file(textured_mtl_path(source_stem));
    const std::string png = read_binary_file(textured_albedo_path(source_stem));
    const std::string dest_obj = destination_stem.filename().string() + ".obj";
    const std::string dest_mtl = destination_stem.filename().string() + ".mtl";
    const std::string dest_png =
        destination_stem.filename().string() + "_albedo.png";
    replace_token(obj, source_stem.filename().string() + ".mtl", dest_mtl);
    replace_token(mtl, source_stem.filename().string() + "_albedo.png", dest_png);
    write_binary_file(textured_obj_path(destination_stem), obj);
    write_binary_file(textured_mtl_path(destination_stem), mtl);
    write_binary_file(textured_albedo_path(destination_stem), png);
}

std::vector<std::uint8_t> encode_textured_obj(const std::filesystem::path& stem) {
    if (!textured_obj_exists(stem))
        throw std::runtime_error(
            "Cannot pack textured mesh; missing " +
            textured_obj_path(stem).string());
    io::binary::BufferWriter writer;
    writer.value(k_texture_chunk_version);
    writer.value(static_cast<std::uint32_t>(1));
    write_blob(writer, read_binary_file(textured_obj_path(stem)));
    std::error_code mtl_error;
    write_blob(
        writer,
        std::filesystem::exists(textured_mtl_path(stem), mtl_error)
            ? read_binary_file(textured_mtl_path(stem))
            : std::string{});
    write_blob(writer, read_binary_file(textured_albedo_path(stem)));
    return writer.take();
}

void decode_textured_obj(
    std::span<const std::uint8_t> bytes, const std::filesystem::path& stem) {
    if (stem.empty())
        throw std::runtime_error("Textured mesh decode requires a destination stem");
    io::binary::BufferReader reader(bytes);
    const auto version = reader.value<std::uint32_t>();
    if (version == 0 || version > k_texture_chunk_version)
        throw std::runtime_error("Unsupported textured mesh chunk version");
    static_cast<void>(reader.value<std::uint32_t>());
    const std::string obj = read_blob(reader);
    const std::string mtl = reader.remaining() > 0 ? read_blob(reader) : std::string{};
    const std::string png = reader.remaining() > 0 ? read_blob(reader) : std::string{};
    write_binary_file(textured_obj_path(stem), obj);
    if (!mtl.empty()) write_binary_file(textured_mtl_path(stem), mtl);
    write_binary_file(textured_albedo_path(stem), png);
}

}  // namespace photara::texture
