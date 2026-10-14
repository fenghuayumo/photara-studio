#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace editor {

enum class ModelImportKind { none, gaussians, mesh };

// PLY is shared by Gaussian models, meshes and point clouds. Inspect only
// the bounded header; never guess from filenames such as "model_splat".
inline ModelImportKind model_import_kind(const std::filesystem::path& path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension == ".obj") return ModelImportKind::mesh;
    if (extension == ".sog" || extension == ".spz" ||
        extension == ".glb" || extension == ".usdc")
        return ModelImportKind::gaussians;
    if (extension != ".ply") return ModelImportKind::none;

    std::ifstream input(path, std::ios::binary);
    std::string line, element;
    std::unordered_set<std::string> properties;
    bool faces = false;
    bool complete = false;
    std::size_t bytes = 0;
    bool first_line = true;
    // Character-by-character reads also bound malformed files without newlines.
    while (bytes < 64 * 1024 && input) {
        line.clear();
        char c;
        while (bytes < 64 * 1024 && input.get(c)) {
            ++bytes;
            if (c == '\n') break;
            if (c != '\r') line += c;
        }
        if (first_line) {
            if (line != "ply") throw std::runtime_error("Invalid PLY header");
            first_line = false;
        }
        std::istringstream row(line);
        std::string token;
        row >> token;
        if (token == "end_header") { complete = true; break; }
        if (token == "element") {
            std::size_t count = 0;
            row >> element >> count;
            if (element == "face" && count > 0) faces = true;
        } else if (token == "property" && element == "vertex") {
            std::string type, name;
            row >> type >> name;
            if (type != "list") properties.insert(name);
        }
    }
    if (!complete) throw std::runtime_error("Invalid or oversized PLY header");
    bool gaussian = true;
    for (const char* name : {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                            "opacity", "scale_0", "scale_1", "scale_2",
                            "rot_0", "rot_1", "rot_2", "rot_3"})
        gaussian = gaussian && properties.contains(name);
    if (gaussian) return ModelImportKind::gaussians;
    if (faces) return ModelImportKind::mesh;
    throw std::runtime_error("PLY must contain Gaussian parameters or mesh faces");
}

} // namespace editor
