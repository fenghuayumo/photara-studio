#include "app.hpp"
#include "file_dialogs.hpp"
#include "i18n.hpp"
#include "ui.hpp"

#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#ifndef IMGUI_HAS_DOCK
#error "The editor requires Dear ImGui built from the docking branch."
#endif

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

using editor::Action;
using editor::App;
using editor::ClearResultsAction;
using editor::k_preview_extent;
using editor::i18n::tr;

// ImGui stores IniFilename as a raw pointer, so this has to outlive the
// context. Smoke tests keep ini disabled so they cannot clobber a saved layout.
std::string g_editor_ini;

int main(const int argc, char** argv) {
    App app;
    app.smoke_mode = argc > 1 && std::string_view(argv[1]) == "--interop-smoke";
    if (!app.smoke_mode)
        editor::i18n::load(
            editor::resolve_editor_ini().parent_path() / "editor.language");

    if (app.smoke_mode) {
        std::snprintf(
            app.settings.images_dir.data(), app.settings.images_dir.size(),
            "D:\\ScanVideo\\ori_img\\images");
        std::snprintf(
            app.settings.project_dir.data(), app.settings.project_dir.size(),
            "D:\\ProgramCode\\C++\\3dgs\\AetherScan\\artifacts\\cuda_vulkan_smoke");
        app.settings.iterations = 100;
    }

    glfwSetErrorCallback([](const int code, const char* text) {
        std::fprintf(stderr, "GLFW %d: %s\n", code, text);
    });
    if (!glfwInit() || !glfwVulkanSupported()) return 1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(
        1600, 940, tr("AetherScan Reconstruction Editor"), nullptr, nullptr);
    glfwSetWindowUserPointer(window, &app);
    glfwSetDropCallback(
        window, [](GLFWwindow* handle, const int count, const char** paths) {
            auto* state = static_cast<App*>(glfwGetWindowUserPointer(handle));
            if (state == nullptr || count <= 0 || paths == nullptr) return;
            state->dropped_paths.clear();
            state->dropped_paths.reserve(static_cast<std::size_t>(count));
            for (int i = 0; i < count; ++i) {
                if (paths[i] != nullptr && paths[i][0] != '\0')
                    state->dropped_paths.emplace_back(paths[i]);
            }
        });

    ImVector<const char*> extensions;
    std::uint32_t extension_count{};
    const char** required = glfwGetRequiredInstanceExtensions(&extension_count);
    for (std::uint32_t i = 0; i < extension_count; ++i)
        extensions.push_back(required[i]);
    editor::gpu::create_context(extensions);
    VkSurfaceKHR surface{};
    editor::gpu::check(
        glfwCreateWindowSurface(editor::gpu::instance(), window, nullptr, &surface));
    int width{};
    int height{};
    glfwGetFramebufferSize(window, &width, &height);
    editor::gpu::create_window(surface, width, height);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigDockingWithShift = false;
    if (!app.smoke_mode) {
        const std::filesystem::path ini = editor::resolve_editor_ini();
        std::error_code error;
        std::filesystem::create_directories(ini.parent_path(), error);
        g_editor_ini = ini.string();
        io.IniFilename = g_editor_ini.c_str();
        if (!std::filesystem::exists(ini, error)) app.reset_dock_layout = true;
    }
    const editor::theme::Fonts fonts = editor::theme::load_fonts(io);
    io.FontDefault = fonts.regular;
    editor::theme::apply_style();

    ImGui_ImplGlfw_InitForVulkan(window, true);
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

    editor::refresh_artifacts(app);
    app.preview.create(k_preview_extent, k_preview_extent);

    const auto smoke_begin = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        app.job.poll();
        app.viewer.poll();
        app.preview.poll();
        app.log.poll(app.fresh_lines);
        for (const std::string& line : app.fresh_lines)
            app.monitor.consume(line);
        if (app.job.consume_completion()) editor::on_job_finished(app);
        if (app.viewer.consume_completion()) {
            const int view_code = app.viewer.exit_code();
            if (view_code != 0 && view_code != 2)
                editor::set_message(
                    app,
                    "Splat viewer exited with code " +
                        std::to_string(view_code),
                    editor::theme::warning);
        }
        editor::poll_scene_load(app);
        editor::poll_mesh_load(app);
        editor::poll_alignment_preview(app);
        editor::poll_align_live(app);
        editor::poll_camera_photos(app);

        if (app.smoke_mode) {
            if (!app.smoke_started && !app.job.running()) {
                app.smoke_started = true;
                editor::start_train(app, true);
                glfwIconifyWindow(window);
            } else if (
                app.smoke_started && !app.job.running() &&
                editor::gpu::consumed_timeline_value() >= 3) {
                app.smoke_success = app.job.exit_code() == 0;
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            } else if (
                std::chrono::steady_clock::now() - smoke_begin >
                std::chrono::seconds(60)) {
                app.job.stop();
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
        }

        glfwGetFramebufferSize(window, &width, &height);
        if (width > 0 && height > 0 &&
            (editor::gpu::swapchain_needs_rebuild() ||
             editor::gpu::window().Width != width ||
             editor::gpu::window().Height != height))
            editor::gpu::resize_window(width, height);
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
            app.preview.consume_without_present();
            editor::publish_preview_ack(app);
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        editor::consume_dropped_paths(app);

        app.settings.iterations = std::max(app.settings.iterations, 1);
        app.settings.densification_cap = std::clamp(
            app.settings.densification_cap, 10'000, 50'000'000);
        app.settings.sh_degree = std::clamp(app.settings.sh_degree, 0, 3);
        app.settings.preview_interval =
            std::max(app.settings.preview_interval, 1);
        app.settings.max_features = std::max(app.settings.max_features, 512);
        app.settings.geometry_from_iter =
            std::max(app.settings.geometry_from_iter, 0);
        app.settings.video_fps = std::clamp(app.settings.video_fps, 0.05F, 60.F);
        app.settings.video_sharp_window =
            std::max(1, app.settings.video_sharp_window);
        app.settings.video_max_frames =
            std::max(0, app.settings.video_max_frames);
        app.settings.video_quality =
            std::clamp(app.settings.video_quality, -1, 100);
        app.settings.video_scale =
            std::clamp(app.settings.video_scale, 0.05F, 4.F);
        app.settings.video_rotate =
            std::clamp(app.settings.video_rotate / 90, 0, 3) * 90;
        app.settings.texture_quality =
            std::clamp(app.settings.texture_quality, 0, 2);
        app.settings.atlas_resolution =
            std::clamp(app.settings.atlas_resolution, 64, 8192);

        Action action = editor::draw_menu_bar(app);
        const Action toolbar_action = editor::draw_toolbar(app);
        if (action == Action::none) action = toolbar_action;

        editor::build_dock_space(app);
        editor::draw_scene_panel(app);
        editor::draw_viewport_panel(app);
        editor::draw_console_panel(app);
        const Action inspector_action = editor::draw_inspector(app);
        if (action == Action::none) action = inspector_action;
        editor::draw_status_bar(app);

        switch (action) {
            case Action::align:
                if (!app.smoke_mode) editor::start_align(app);
                break;
            case Action::train:
                if (!app.smoke_mode) editor::start_train(app, false);
                break;
            case Action::dense:
                if (!app.smoke_mode) editor::start_dense(app);
                break;
            case Action::texture:
                if (!app.smoke_mode) editor::start_texture(app);
                break;
            case Action::export_sfm:
                if (!app.smoke_mode) editor::start_export_sfm(app);
                break;
            case Action::pause:
                app.job.pause();
                if (app.job.paused()) {
                    app.monitor.pause_clock();
                    editor::set_message(
                        app, "Paused. Press Resume to continue.",
                        editor::theme::warning);
                } else {
                    editor::set_message(
                        app, "Could not pause the running job",
                        editor::theme::danger);
                }
                break;
            case Action::resume:
                app.job.resume();
                if (!app.job.paused()) {
                    app.monitor.resume_clock();
                    editor::set_message(app, "Resumed", editor::theme::accent);
                } else {
                    editor::set_message(
                        app, "Could not resume the paused job",
                        editor::theme::danger);
                }
                break;
            case Action::stop:
                app.job.stop();
                if (app.job.consume_completion())
                    editor::on_job_finished(app);
                else
                    editor::set_message(
                        app,
                        "Stopped. You can change images, video, or parameters "
                        "and run again.",
                        editor::theme::warning);
                break;
            case Action::reveal:
                editor::reveal_in_explorer(app.layout.root);
                break;
            case Action::none:
                break;
        }

        editor::draw_controls_window(app);
        editor::draw_mesh_export_modal(app);
        editor::draw_alignment_export_modal(app);
        editor::draw_splat_export_modal(app);
        const ClearResultsAction clear_results_action =
            editor::draw_clear_results_modal(app);
        if (app.close_requested) glfwSetWindowShouldClose(window, GLFW_TRUE);
        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (draw_data->DisplaySize.x > 0.F && draw_data->DisplaySize.y > 0.F)
            editor::gpu::present(draw_data, editor::theme::surface_0);
        editor::publish_preview_ack(app);

        if (app.pending_align_viewport_clear) {
            app.pending_align_viewport_clear = false;
            editor::clear_viewport_scene(app);
        }

        if (clear_results_action == ClearResultsAction::clear_view) {
            editor::clear_loaded_result(app);
            app.suppress_scene_auto_load = true;
            editor::set_message(
                app, "Loaded reconstruction cleared from view",
                editor::theme::success);
        } else if (
            clear_results_action == ClearResultsAction::delete_generated) {
            editor::delete_reconstruction_results(app);
        }
    }

    if (app.viewer.running()) app.viewer.stop();
    if (app.job.running()) app.job.stop();
    vkDeviceWaitIdle(editor::gpu::device());
    app.image_qa_session.clear();
    app.photos.clear();
    app.atlas_preview.reset();
    app.mesh_renderer.reset();
    app.preview.reset();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    editor::gpu::destroy_context();
    glfwDestroyWindow(window);
    glfwTerminate();
    return app.smoke_mode && !app.smoke_success ? 4 : 0;
}
