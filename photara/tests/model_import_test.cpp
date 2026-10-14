#include "model_import.hpp"

#include <chrono>
#include <iostream>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main() {
    const auto directory = std::filesystem::temp_directory_path() /
        ("photara_model_import_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    try {
        using editor::ModelImportKind;
        const auto classify = [&](const std::string& header) {
            const auto path = directory / "model.PLY";
            { std::ofstream output(path, std::ios::binary); output << header; }
            return editor::model_import_kind(path);
        };
        const auto reject = [&](const std::string& header) {
            bool rejected = false;
            try { classify(header); } catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "Invalid/unsupported PLY was accepted");
        };
        const std::string vertices = "ply\nformat ascii 1.0\nelement vertex 3\n"
            "property float x\nproperty float y\nproperty float z\n";
        require(classify(vertices + "element face 1\nproperty list uchar int vertex_indices\nend_header\n")
                    == ModelImportKind::mesh, "Mesh PLY misclassified");
        std::string gaussian = vertices;
        for (const char* name : {"f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0",
                                "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"})
            gaussian += std::string("property float ") + name + "\n";
        require(classify(gaussian + "end_header\n") == ModelImportKind::gaussians,
                "Gaussian PLY misclassified");
        std::string crlf;
        for (char c : gaussian + "end_header\n") {
            if (c == '\n') crlf += '\r';
            crlf += c;
        }
        require(classify(crlf) == ModelImportKind::gaussians, "CRLF header failed");
        reject(vertices + "end_header\n");
        reject(vertices + "element face 0\nend_header\n");
        reject(vertices + "property float opacity\nend_header\n");
        reject(gaussian);
        reject("not a ply\nend_header\n");
        reject("ply\n" + std::string(70 * 1024, 'x'));
        reject(vertices + "element custom 1\n" + gaussian.substr(vertices.size()) + "end_header\n");
        require(editor::model_import_kind("model.OBJ") == ModelImportKind::mesh, "OBJ not recognized");
        for (const char* name : {"model.SOG", "model.SPZ", "model.GLB", "model.USDC"})
            require(editor::model_import_kind(name) == ModelImportKind::gaussians,
                    "Gaussian format not recognized");
        for (const char* name : {"images", "photo.JPG", "scene.ascan", "scene.asfm", "model.stl"})
            require(editor::model_import_kind(name) == ModelImportKind::none,
                    "Non-model input intercepted");
        std::filesystem::remove_all(directory);
        std::cout << "Model import classification passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(directory);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
