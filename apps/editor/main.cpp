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
#include <cmath>
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
    const std::string_view smoke_arg = argc > 1 ? argv[1] : "";
    const bool mesh_smoke = smoke_arg == "--interop-mesh-smoke";
    const bool stop_smoke = smoke_arg == "--interop-stop-smoke" ||
                            smoke_arg == "--interop-stop-smoke-vulkan";
    const bool retrain_smoke = smoke_arg == "--interop-retrain-smoke" ||
                               smoke_arg == "--interop-retrain-smoke-vulkan";
    app.smoke_mode = smoke_arg == "--interop-smoke" ||
                     smoke_arg == "--interop-smoke-vulkan" || stop_smoke ||
                     retrain_smoke || mesh_smoke;
    if (!app.smoke_mode)
        editor::i18n::load(
            editor::resolve_editor_ini().parent_path() / "editor.language");
    if (!app.smoke_mode) editor::load_editor_cache_dir(app);
    if (!app.smoke_mode) editor::load_editor_preferences(app);

    if (app.smoke_mode) {
        // The interop smoke harness needs a fixed dataset so runs stay
        // comparable, but developer-machine paths do not belong in the
        // source. Point PHOTARA_SMOKE_IMAGES at the COLMAP images directory;
        // optionally set PHOTARA_SMOKE_PROJECT for the output project folder
        // (default: "<dataset>/photara_smoke" beside the images directory).
        const char* smoke_images = std::getenv("PHOTARA_SMOKE_IMAGES");
        if (smoke_images == nullptr || smoke_images[0] == '\0') {
            std::fprintf(
                stderr,
                "--interop-* smoke requires PHOTARA_SMOKE_IMAGES (COLMAP "
                "images directory); optionally PHOTARA_SMOKE_PROJECT\n");
            return 2;
        }
        const char* smoke_project = std::getenv("PHOTARA_SMOKE_PROJECT");
        const std::filesystem::path project_path =
            smoke_project != nullptr && smoke_project[0] != '\0'
                ? editor::path_from_utf8_field(smoke_project)
                : editor::path_from_utf8_field(smoke_images).parent_path() /
                      "photara_smoke";
        const std::u8string project_u8 = project_path.u8string();
        std::snprintf(
            app.settings.images_dir.data(), app.settings.images_dir.size(),
            "%s", smoke_images);
        std::snprintf(
            app.settings.project_dir.data(), app.settings.project_dir.size(),
            "%s", reinterpret_cast<const char*>(project_u8.c_str()));
        app.settings.build_mesh = mesh_smoke;
        app.settings.iterations = stop_smoke ? 30'000 : 100;
        if (smoke_arg == "--interop-smoke-vulkan" ||
            smoke_arg == "--interop-stop-smoke-vulkan" ||
            smoke_arg == "--interop-retrain-smoke-vulkan")
            app.settings.training_backend = 1;
    }

    glfwSetErrorCallback([](const int code, const char* text) {
        std::fprintf(stderr, "GLFW %d: %s\n", code, text);
    });
    if (!glfwInit() || !glfwVulkanSupported()) return 1;
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(
        1600, 940, tr("Photara Studio"), nullptr, nullptr);
    editor::apply_studio_window_icon(window);
    glfwSetWindowUserPointer(window, &app);
    glfwSetWindowCloseCallback(window, [](GLFWwindow* handle) {
        auto* state = static_cast<App*>(glfwGetWindowUserPointer(handle));
        if (!state || state->smoke_mode || state->close_approved) return;
        glfwSetWindowShouldClose(handle, GLFW_FALSE);
        state->close_requested = true;
    });
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
    editor::prepare_cache_session(app);
    app.cache_usage = editor::scan_cache_usage(
        app.settings, app.layout.working_sfm.parent_path());
    if (app.settings.cache_retention_days > 0) {
        const std::size_t removed = editor::clean_unused_caches(
            app.settings, app.layout.working_sfm.parent_path(),
            std::chrono::minutes(60 * 24 * app.settings.cache_retention_days));
        if (removed > 0) {
            editor::set_message(
                app,
                "Cache sweep removed " + std::to_string(removed) +
                    " unused folders",
                editor::theme::text_muted);
            app.cache_usage = editor::scan_cache_usage(
                app.settings, app.layout.working_sfm.parent_path());
        }
    }
    app.preview.create(k_preview_extent, k_preview_extent);

    const auto smoke_begin = std::chrono::steady_clock::now();
    // Retrain smoke: the first run must complete, then Train 3DGS is started
    // again exactly like a user click, after the frame that still references
    // the old preview descriptors has been submitted.
    int smoke_completed_runs = 0;
    bool retrain_pending = false;
    bool retrain_requested = false;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        app.job.poll();
        editor::poll_project_save(app);
        app.preview.poll();
        app.log.poll(app.fresh_lines);
        for (const std::string& line : app.fresh_lines)
            app.monitor.consume(line);
        editor::poll_splat_model_load(app);
        editor::poll_mesh_load(app);
        if (app.job.consume_completion()) editor::on_job_finished(app);
        editor::poll_splat_model_load(app);
        editor::poll_scene_load(app);
        editor::poll_mesh_load(app);
        editor::poll_alignment_preview(app);
        editor::poll_align_live(app);
        editor::poll_camera_photos(app);
        editor::poll_splat_simplify(app);
        editor::poll_gaussian_merge(app);
        editor::poll_splat_export(app);

        if (app.smoke_mode) {
            if (!app.smoke_started && !app.job.running()) {
                app.smoke_started = true;
                editor::start_train(app, true);
                // The retrain flow has to keep presenting frames: the bug was
                // a draw list referencing descriptors destroyed by the second
                // start_train().
                if (!stop_smoke && !retrain_smoke)
                    glfwIconifyWindow(window);
            } else if (
                app.smoke_started && !app.job.running() &&
                !retrain_pending &&
                (stop_smoke || app.final_splat_ready) &&
                (!mesh_smoke || app.final_mesh_ready) &&
                (stop_smoke || editor::gpu::consumed_timeline_value() >= 3)) {
                ++smoke_completed_runs;
                if (retrain_smoke && smoke_completed_runs == 1 &&
                    app.job.exit_code() == 0) {
                    retrain_pending = true;
                } else {
                    app.smoke_success = stop_smoke
                        ? app.job.exit_code() == 2 &&
                              !app.preview.display.descriptor &&
                              app.active_job == editor::JobKind::none
                        : app.job.exit_code() == 0;
                    glfwSetWindowShouldClose(window, GLFW_TRUE);
                }
            } else if (
                std::chrono::steady_clock::now() - smoke_begin >
                std::chrono::seconds(retrain_smoke || mesh_smoke ? 240 : 60)) {
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
        if (!app.pending_train_prepare.valid()) editor::consume_dropped_paths(app);

        app.settings.sanitize();
        app.settings.iterations = std::max(app.settings.iterations, 1);
        app.settings.densification_cap = std::clamp(
            app.settings.densification_cap, 10'000, 50'000'000);
        app.settings.sh_degree = std::clamp(app.settings.sh_degree, 0, 3);
        app.settings.mask_mode = app.settings.mask_mode == 1 ? 1 : 0;
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

        ImGui::BeginDisabled(app.pending_train_prepare.valid());
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

        // Exercise the same action as the Stop button after the viewport has
        // added a live preview texture to this frame's draw list.
        if (stop_smoke && app.job.running() &&
            editor::gpu::consumed_timeline_value() >= 3)
            action = Action::stop;
        if (retrain_pending) {
            retrain_requested = true;
            retrain_pending = false;
        }

        bool stop_requested = false;
        // start_train() recreates the shared preview image and drops the
        // splat descriptor sets that this frame's draw list may still
        // reference. Like Stop, the request has to wait until the frame is
        // submitted; queue idle cannot protect an unsubmitted draw list.
        bool train_requested = false;
        bool resume_training_requested = false;
        bool dense_requested = false;
        if (app.saving_training) action = Action::none;
        switch (action) {
            case Action::align:
                if (!app.smoke_mode) editor::start_align(app);
                break;
            case Action::train:
                train_requested = !app.smoke_mode;
                break;
            case Action::continue_training:
                train_requested = !app.smoke_mode;
                resume_training_requested = true;
                break;
            case Action::dense:
                // The Gaussians path falls through to start_train().
                dense_requested = !app.smoke_mode;
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
                stop_requested = true;
                break;
            case Action::reveal:
                editor::reveal_in_explorer(app.layout.root);
                break;
            case Action::none:
                break;
        }

        editor::draw_controls_window(app);
        editor::draw_about_window(app);
        editor::draw_preferences_window(app);
        editor::draw_mesh_export_modal(app);
        editor::draw_alignment_export_modal(app);
        editor::draw_splat_export_modal(app);
        editor::draw_sam_license_modal(app);
        const ClearResultsAction clear_results_action =
            editor::draw_clear_results_modal(app);
        ImGui::EndDisabled();
        if (app.close_requested && !app.pending_train_prepare.valid()) {
            app.close_requested = false;
            if (editor::has_unsaved_project_changes(app)) app.show_close_confirmation = true;
            else app.close_approved = true;
        }
        const std::string train_preparation_popup = std::string(tr("Preparing Training")) + "###train_preparation";
        const bool show_train_preparation = app.pending_train_prepare.valid() && app.train_preparation &&
            std::chrono::steady_clock::now() - app.train_preparation->started >= std::chrono::milliseconds(250);
        if (show_train_preparation) {
            ImGui::OpenPopup(train_preparation_popup.c_str());
            if (ImGui::BeginPopupModal(train_preparation_popup.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
                const auto preparation = app.train_preparation;
                const auto stage = preparation ? preparation->stage.load() : editor::TrainPrepareStage::checking;
                const char* label = "Checking training inputs...";
                if (stage == editor::TrainPrepareStage::reading_sfm) label = "Reading updated camera data...";
                if (stage == editor::TrainPrepareStage::encoding_sfm) label = "Preparing cached camera data...";
                if (stage == editor::TrainPrepareStage::restoring_progress) label = "Restoring saved training progress...";
                ImGui::TextUnformatted(tr(label));
                if (preparation) {
                    ImGui::Text(tr("Preparation elapsed: %.1f s"),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - preparation->started).count());
                    if (preparation->cancelled) {
                        ImGui::TextUnformatted(tr("Cancelling after the current step..."));
                    } else if (ImGui::Button(tr("Cancel"))) {
                        preparation->cancelled = true;
                    }
                }
                ImGui::EndPopup();
            }
        } else if (ImGui::BeginPopupModal(train_preparation_popup.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }
        editor::draw_project_lifecycle_modals(app);
        if (app.close_approved) glfwSetWindowShouldClose(window, GLFW_TRUE);
        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (draw_data->DisplaySize.x > 0.F && draw_data->DisplaySize.y > 0.F)
            editor::gpu::present(draw_data, editor::theme::surface_0);
        editor::publish_preview_ack(app);

        // Preparation runs off the UI thread; launch only after the submitted
        // frame no longer references preview resources that start_train drops.
        if (app.close_requested && app.train_preparation) app.train_preparation->cancelled = true;
        editor::poll_train_start(app);

        // Now the frame is submitted and the stale preview descriptors are
        // safe to destroy.
        if (train_requested) editor::start_train(app, false, resume_training_requested);
        if (retrain_requested) {
            retrain_requested = false;
            editor::start_train(app, true);
        }
        if (dense_requested) editor::start_dense(app);

        // Finish submitting the draw list and acknowledging its shared image
        // before terminating the producer. on_job_finished() destroys the
        // preview texture; queue idle cannot protect an unsubmitted draw list.
        if (stop_requested) {
            app.job.stop();
            if (app.job.consume_completion())
                editor::on_job_finished(app);
            else
                editor::set_message(
                    app,
                    "Stopped. You can change images, video, or parameters "
                    "and run again.",
                    editor::theme::warning);
        }

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

    if (app.job.running()) app.job.stop();
    if (app.splat_export.cancel) app.splat_export.cancel->store(true);
    if (app.splat_export.job.valid()) app.splat_export.job.wait();
    editor::wait_splat_model_loads(app);
    editor::cleanup_cache_session(app);
    editor::gpu::wait_idle();
    app.image_qa_session.clear();
    app.photos.clear();
    app.atlas_preview.reset();
    app.about_icon.reset();
    editor::release_splat_preview(app);
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
