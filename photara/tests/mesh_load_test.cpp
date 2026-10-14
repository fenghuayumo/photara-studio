#include "mvs/export.hpp"

#include <chrono>
#include <bit>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace photara::mvs;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    const auto path = std::filesystem::temp_directory_path() /
        ("photara_mesh_load_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".ply");
    try {
        if (argc > 1 && std::string(argv[1]) == "--benchmark") {
            const int count = argc > 2 ? std::stoi(argv[2]) : 1'000'000;
            Mesh source;
            source.vertices.reserve(count + 2);
            source.normals.reserve(count + 2);
            for (int i = 0; i < count + 2; ++i) {
                source.vertices.emplace_back(float(i % 1000), float(i / 1000), float(i % 7));
                source.normals.emplace_back(0.F, 0.F, 1.F);
            }
            for (int i = 0; i < count; ++i) source.faces.emplace_back(i, i + 1, i + 2);
            save_mesh_ply(source, path);
            source = {};
            for (int run = 0; run < 3; ++run) {
                const auto begin = std::chrono::steady_clock::now();
                const auto loaded = load_mesh_ply(path);
                const auto ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
                std::cout << loaded.vertices.size() << " vertices, " << loaded.faces.size()
                          << " faces: " << ms << " ms\n";
                require(loaded.faces.size() == static_cast<std::size_t>(count), "benchmark face count");
            }
        } else {
            Mesh source;
            source.vertices = {{0.F, 0.F, 0.F}, {1.F, 0.F, 0.F}, {0.F, 1.F, 0.F}};
            source.normals.assign(3, Vec3f{0.F, 0.F, 1.F});
            source.colors = {{1.F, 0.F, 0.F}, {0.F, 1.F, 0.F}, {0.F, 0.F, 1.F}};
            source.faces.emplace_back(0, 1, 2);
            save_mesh_ply(source, path);
            auto loaded = load_mesh_ply(path);
            require(loaded.vertices == source.vertices && loaded.faces == source.faces, "binary round trip geometry");
            require(loaded.normals == source.normals && loaded.colors == source.colors, "binary attributes");
            const auto write = [&](const std::string& text) {
                std::ofstream output(path, std::ios::binary); output << text;
            };
            // Reordered attributes, integer and floating colors, unknown lists,
            // and a quad exercise the general parser rather than the writer layout.
            const std::string header = "ply\nformat ascii 1.0\nelement vertex 4\n"
                "property float z\nproperty double x\nproperty float y\n"
                "property uchar red\nproperty float green\nproperty uchar blue\n"
                "property list uchar int extra\nelement face 1\n"
                "property int material\nproperty list uchar int vertex_index\nend_header\n";
            write(header + "0 0 0 255 0.5 0 2 9 10\n0 1 0 0 1 255 0\n"
                           "0 1 1 0 0 255 1 5\n0 0 1 255 1 0 0\n7 4 0 1 2 3\n");
            loaded = load_mesh_ply(path);
            require(loaded.vertices.size() == 4 && loaded.faces.size() == 2, "ASCII lists/triangulation");
            require(loaded.faces[1] == Eigen::Vector3i(0, 2, 3), "quad winding");
            require(loaded.colors[0] == Vec3f(1.F, 0.5F, 0.F), "color conversion");
            const std::string binary_header = "ply\nformat binary_little_endian 1.0\n"
                "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
                "property list uchar int ignored\nelement edge 1\nproperty list uchar int ignored\n"
                "element face 1\nproperty ushort material\nproperty list uchar int vertex_indices\nend_header\n";
            std::string payload;
            const auto append = [&]<class T>(T value) {
                auto bytes = std::bit_cast<std::array<char, sizeof(T)>>(value);
                if constexpr (std::endian::native == std::endian::big)
                    std::reverse(bytes.begin(), bytes.end());
                payload.append(bytes.data(), bytes.size());
            };
            for (int i = 0; i < 3; ++i) {
                append(float(i == 1)); append(float(i == 2)); append(0.F);
                append(std::uint8_t{2}); append(std::int32_t{8}); append(std::int32_t{9});
            }
            append(std::uint8_t{2}); append(std::int32_t{0}); append(std::int32_t{1});
            append(std::uint16_t{42}); append(std::uint8_t{3});
            append(std::int32_t{0}); append(std::int32_t{1}); append(std::int32_t{2});
            write(binary_header + payload);
            loaded = load_mesh_ply(path);
            require(loaded.vertices == source.vertices && loaded.faces == source.faces &&
                    loaded.colors.empty() && loaded.normals.empty(), "binary skipped lists/elements");
            const auto reject = [&](const std::string& text) {
                write(text);
                bool rejected = false;
                try { load_mesh_ply(path); } catch (const std::exception&) { rejected = true; }
                require(rejected, "malformed mesh accepted");
            };
            reject(header + "0 0 0 255 0 0 -1\n");
            reject(header + "0 0 0 255 0 0 0\n");
            reject(binary_header + payload.substr(0, 16));
            reject("not a ply\nend_header\n");
            save_mesh_ply(source, path);
            std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
            bool rejected = false;
            try { load_mesh_ply(path); } catch (const std::exception&) { rejected = true; }
            require(rejected, "truncated binary face accepted");
            std::cout << "Mesh loader checks passed\n";
        }
        std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
