#include "sparse_view.hpp"
#include "vulkan_backend.hpp"
#include "mvs/mesh_handoff.hpp"
#include <future>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int test_gpu_upload(const std::shared_ptr<const editor::gpu::MeshGpuData>& data,
                    const editor::gpu::MeshGpuTexture& albedo) {
    // Exercise the actual viewer renderer with an invisible window; no user
    // layout/preferences are loaded or written.
    if (!glfwInit()) return 77;
    if (!glfwVulkanSupported()) { glfwTerminate(); return 77; }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(64, 64, "Mesh preview test", nullptr, nullptr);
    require(window != nullptr, "hidden GPU test window");
    std::uint32_t count = 0;
    const auto required = glfwGetRequiredInstanceExtensions(&count);
    ImVector<const char*> extensions;
    for (std::uint32_t i = 0; i < count; ++i) extensions.push_back(required[i]);
    editor::gpu::create_context(extensions);
    VkSurfaceKHR surface{};
    editor::gpu::check(glfwCreateWindowSurface(editor::gpu::instance(), window, nullptr, &surface));
    editor::gpu::create_window(surface, 64, 64);
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui_ImplVulkan_InitInfo init{};
    init.Instance = editor::gpu::instance();
    init.PhysicalDevice = editor::gpu::physical_device();
    init.Device = editor::gpu::device();
    init.QueueFamily = editor::gpu::queue_family();
    init.Queue = editor::gpu::queue();
    init.DescriptorPool = editor::gpu::descriptor_pool();
    init.RenderPass = editor::gpu::window().RenderPass;
    init.MinImageCount = editor::gpu::k_min_images;
    init.ImageCount = editor::gpu::window().ImageCount;
    init.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init.CheckVkResultFn = editor::gpu::check;
    ImGui_ImplVulkan_Init(&init);
    {
        editor::gpu::MeshPreviewRenderer renderer;
        renderer.set_mesh(data);
        renderer.set_albedo(albedo);
        editor::gpu::MeshPreviewUniforms uniforms;
        uniforms.world_to_clip = uniforms.world_to_camera =
            {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        require(renderer.draw(64, 64, uniforms) && renderer.descriptor(), "solid mesh GPU draw");
        uniforms.textured = true;
        require(renderer.draw(64, 64, uniforms), "textured mesh GPU draw");
        uniforms.wireframe = true;
        require(renderer.draw(64, 64, uniforms), "first lazy wireframe draw");
        require(renderer.draw(64, 64, uniforms), "cached wireframe draw");
        renderer.set_mesh(data);
        require(renderer.draw(64, 64, uniforms), "wireframe rebuild after mesh replacement");
        uniforms.wireframe = false;
        require(renderer.draw(64, 64, uniforms), "switch back to solid mesh");
        renderer.set_mesh({});
        require(!renderer.draw(64, 64, uniforms), "empty mesh GPU reset");
    }
    editor::gpu::wait_idle();
    ImGui_ImplVulkan_Shutdown();
    ImGui::DestroyContext();
    editor::gpu::destroy_context();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

int main(int argc, char** argv) {
    const auto path = std::filesystem::temp_directory_path() /
        ("photara_mesh_preview_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".obj");
    auto atlas_path = path;
    atlas_path.replace_extension();
    atlas_path += "_albedo.png";
    try {
        const photara::io::RgbImage atlas{1, 1, {255, 128, 64}};
        photara::io::save_rgb_png(atlas, atlas_path);
        const auto write = [&](const std::string& text) {
            std::ofstream output(path); output << text;
        };
        const std::string vertices = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n";
        write(vertices + "f 1/1/1 2/2/1 3/3/1\nf -4/1/1 -2/3/1 -1/4/1\n");
        auto loaded = editor::load_preview_mesh(path);
        require(loaded.ok && loaded.mesh.vertices.size() == 4 && loaded.mesh.faces.size() == 2,
                "OBJ shared corners / negative indices");
        require(loaded.mesh.uvs[0].y == 1.F && loaded.mesh.uvs[2].y == 0.F, "OBJ UV orientation");
        require(loaded.mesh.faces[0][0] == loaded.mesh.faces[1][0], "OBJ vertex not shared");
        loaded = editor::prepare_mesh_preview(std::move(loaded), true);
        require(loaded.albedo.width == 1 && loaded.albedo.height == 1 &&
                loaded.albedo.pixels == std::vector<std::uint8_t>{255, 128, 64, 255},
                "background texture decode and RGBA packing");
        require(loaded.ok && loaded.gpu_data && loaded.reconstruction_bounds_valid, "background preparation");
        require(loaded.gpu_data->indices.size() == 6 && loaded.gpu_data->vertices.size() == 4,
                "packed buffer dimensions");
        for (std::size_t i = 0; i < loaded.mesh.vertices.size(); ++i) {
            const auto& gpu = loaded.gpu_data->vertices[i];
            require(gpu.px == loaded.mesh.vertices[i].x && gpu.py == loaded.mesh.vertices[i].y &&
                    gpu.pz == loaded.mesh.vertices[i].z, "packed positions");
            require(gpu.nz == 1.F && gpu.r == 1.F && gpu.u == loaded.mesh.uvs[i].x &&
                    gpu.v == loaded.mesh.uvs[i].y, "packed normal/color/UV attributes");
        }
        write(vertices + "vt 0.5 0.5\nvn 0 0 -1\nf 1/1/1 2/2/1 3/3/1\nf 1/5/2 3/3/2 4/4/2\n");
        loaded = editor::load_preview_mesh(path);
        require(loaded.ok && loaded.mesh.vertices.size() == 6, "OBJ seams or hard normal edges collapsed");
        write("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 1\nf 1 2 3\nf 1 3 4\n");
        loaded = editor::load_preview_mesh(path);
        require(loaded.ok && loaded.mesh.vertices.size() == 6 && loaded.mesh.uvs.empty(),
                "OBJ flat shading / missing UVs");
        loaded = editor::prepare_mesh_preview(std::move(loaded), false);
        require(loaded.ok && !loaded.reconstruction_bounds_valid && loaded.gpu_data,
                "reuse existing box should skip fitting");
        // The viewport path must work even when no mesh PLY was ever written.
        auto ready = path;
        ready += ".ready";
        {
            photara::mvs::Mesh mesh;
            mesh.vertices = {{0.F, 0.F, 0.F}, {1.F, 0.F, 0.F}, {0.F, 1.F, 0.F}};
            mesh.faces = {{0, 1, 2}};
            mesh.colors.assign(3, {1.F, 0.F, 0.F});
            photara::mvs::MeshHandoff publisher(mesh, ready);
            auto pending = std::async(std::launch::async, [ready] {
                return editor::prepare_mesh_preview(editor::load_preview_mesh_handoff(ready), true);
            });
            loaded = pending.get();
            require(publisher.wait_for_reader(std::chrono::milliseconds(1)), "viewport handoff acknowledgment");
            require(loaded.ok && loaded.gpu_data && loaded.reconstruction_bounds_valid &&
                loaded.mesh.vertices.size() == 3 && loaded.mesh.faces.size() == 1 &&
                loaded.gpu_data->indices == std::vector<std::uint32_t>{0, 1, 2}, "shared mesh preview preparation");
            require(loaded.gpu_data->vertices[0].r == 1.F && loaded.gpu_data->vertices[0].g == 0.F,
                "shared mesh preview color packing");
        }
        std::filesystem::remove(ready);
        ready += ".ack";
        std::filesystem::remove(ready);
        if (argc > 1 && std::string(argv[1]) == "--vulkan") {
            const int result = test_gpu_upload(loaded.gpu_data, loaded.albedo);
            if (result != 0) {
                std::filesystem::remove(path);
                std::filesystem::remove(atlas_path);
                return result;
            }
        }
        editor::MeshLoad failed;
        failed.error = "test failure";
        auto prepared = editor::prepare_mesh_preview(std::move(failed), true);
        require(!prepared.ok && !prepared.gpu_data && prepared.error == "test failure", "load failure propagation");
        std::filesystem::remove(path);
        std::filesystem::remove(atlas_path);
        std::cout << "Mesh preview checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::filesystem::remove(atlas_path);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
