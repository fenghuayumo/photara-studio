#include "texture/export.hpp"

#include "io/image.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

namespace aetherscan::texture {
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
    obj << "# AetherScan textured mesh";
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

}  // namespace aetherscan::texture
