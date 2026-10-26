#include "texture/mesh_preview.hpp"
#include "mvs/export.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

// Uses the same triangle Z-buffer and clay shading as Studio. Camera JSON
// comes from the splat audit so all compared meshes receive identical views.
int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("Usage: splat_mesh_preview_audit MESH AUDIT_JSON OUTPUT");
        const auto mesh = photara::mvs::load_mesh_ply(argv[1]);
        const auto audit = nlohmann::json::parse(std::ifstream(argv[2]));
        const std::filesystem::path output(argv[3]);
        std::filesystem::create_directories(output);
        std::vector<float> positions, normals;
        std::vector<std::uint32_t> indices;
        positions.reserve(mesh.vertices.size()*3);
        for (const auto& v : mesh.vertices) for (int a=0; a<3; ++a) positions.push_back(v[a]);
        normals.reserve(mesh.normals.size()*3);
        for (const auto& v : mesh.normals) for (int a=0; a<3; ++a) normals.push_back(v[a]);
        indices.reserve(mesh.faces.size()*3);
        for (const auto& face : mesh.faces) for (int a=0; a<3; ++a) indices.push_back(face[a]);
        photara::texture::MeshPreviewRasterizer renderer;
        for (const auto& view : audit.at("views")) {
            const auto index=view.at("index").get<unsigned>();
            if (index != 0 && index != 40) continue;
            photara::texture::MeshPreviewCamera camera;
            camera.world_to_camera=view.at("world_to_camera_column_major").get<std::array<float,16>>();
            camera.position=view.at("position").get<std::array<float,3>>();
            camera.fx=view.at("fx"); camera.fy=view.at("fy");
            camera.cx=view.at("cx"); camera.cy=view.at("cy");
            camera.width=view.at("width"); camera.height=view.at("height");
            const auto image=renderer.render(positions,normals,{},indices,camera,{});
            photara::io::save_rgb_png(image,output/("mesh_"+std::to_string(index)+".png"));
        }
        std::cout << "Mesh preview completed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
