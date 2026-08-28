#include "pipeline.hpp"
#include "console_view.hpp"
#include "icons.hpp"
#include "sparse_view.hpp"
#include "theme.hpp"
#include "viewport_gizmo.hpp"
#include "vulkan_backend.hpp"

#include "imgui_impl_glfw.h"
#include "imgui_internal.h"

#ifndef IMGUI_HAS_DOCK
#error "The editor requires Dear ImGui built from the docking branch."
#endif

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#if defined(_WIN32)
#include <shellapi.h>
#include <shlobj.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef AETHERSCAN_CLI_PATH
#define AETHERSCAN_CLI_PATH "aetherscan"
#endif

namespace {

using namespace editor;

constexpr std::uint32_t k_preview_extent = 1920;

enum class ViewportTab { sparse, training };

enum class StepState { pending, active, done, skipped, failed };

struct App {
    ProjectSettings settings;
    ProjectLayout layout;

    ProcessJob job;
    JobKind active_job{JobKind::none};
    RunMonitor monitor;
    LogStream log;
    ConsoleView console;
    std::vector<std::string> fresh_lines;

    gpu::ExternalPreview preview;

    SparseScene scene;
    OrbitCamera camera;
    ViewOptions view_options;
    ViewportGizmoState gizmo;
    SceneRenderer renderer;
    std::future<SceneLoad> pending_load;
    bool loading_scene{};
    std::string scene_source;

    bool has_sparse{};
    bool has_mvs{};
    bool has_model{};
    bool has_mesh{};
    ViewportTab tab{ViewportTab::sparse};
    unsigned preview_view{};
    std::uint64_t preview_camera_revision{};
    bool preview_follow_view{true};
    OrbitCamera last_preview_orbit{};
    bool has_last_preview_orbit{};
    std::uint32_t preview_raster_width{k_preview_extent};
    std::uint32_t preview_raster_height{k_preview_extent};

    std::string message;
    ImVec4 message_colour{theme::text_muted};

    // Manual CUDA/Vulkan interop verification harness.
    bool smoke_mode{};
    bool smoke_started{};
    bool smoke_success{};
    bool close_requested{};
    bool show_controls{};
    bool show_clear_results{};
    bool project_folder_automatic{};

    bool show_scene{true};
    bool show_viewport{true};
    bool show_console{true};
    bool show_inspector{true};
    bool show_status_bar{true};
    bool reset_dock_layout{};
};

constexpr float k_toolbar_height = 52.F;
constexpr float k_status_height = 34.F;

// ImGui stores IniFilename as a raw pointer, so this has to outlive the
// context. Smoke tests keep ini disabled so they cannot clobber a saved layout.
std::string g_editor_ini;

std::filesystem::path resolve_editor_ini() {
#if defined(_WIN32)
    if (const char* appdata = std::getenv("APPDATA"))
        return std::filesystem::path(appdata) / "AetherScan" / "editor.ini";
#endif
    return std::filesystem::current_path() / "aetherscan_editor.ini";
}

void set_message(App& app, std::string text, const ImVec4& colour) {
    app.message = std::move(text);
    app.message_colour = colour;
}

bool directory_has_images(const std::filesystem::path& directory) {
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) return false;
    for (const auto& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (!entry.is_regular_file(error)) continue;
        std::string extension = entry.path().extension().string();
        std::transform(
            extension.begin(), extension.end(), extension.begin(),
            [](const unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
        if (extension == ".jpg" || extension == ".jpeg" ||
            extension == ".png" || extension == ".tif" ||
            extension == ".tiff" || extension == ".bmp")
            return true;
    }
    return false;
}

#if defined(_WIN32)
// Native folder picker. Failure simply leaves the text field untouched.
bool pick_folder(const wchar_t* title, std::array<char, 1024>& destination) {
    bool picked = false;
    const HRESULT initialised =
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    IFileOpenDialog* dialog = nullptr;
    if (SUCCEEDED(CoCreateInstance(
            CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(&dialog)))) {
        DWORD options = 0;
        dialog->GetOptions(&options);
        dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_PATHMUSTEXIST);
        dialog->SetTitle(title);
        if (SUCCEEDED(dialog->Show(nullptr))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dialog->GetResult(&item))) {
                PWSTR wide = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &wide))) {
                    const int bytes = WideCharToMultiByte(
                        CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
                    if (bytes > 0 &&
                        static_cast<std::size_t>(bytes) <= destination.size()) {
                        WideCharToMultiByte(
                            CP_UTF8, 0, wide, -1, destination.data(), bytes,
                            nullptr, nullptr);
                        picked = true;
                    }
                    CoTaskMemFree(wide);
                }
                item->Release();
            }
        }
        dialog->Release();
    }
    if (SUCCEEDED(initialised)) CoUninitialize();
    return picked;
}

void reveal_in_explorer(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::exists(path, error)) return;
    ShellExecuteW(nullptr, L"open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}
#else
bool pick_folder(const wchar_t*, std::array<char, 1024>&) { return false; }
void reveal_in_explorer(const std::filesystem::path&) {}
#endif

// ---------------------------------------------------------------------------
// Project actions

void refresh_artifacts(App& app) {
    // A normal editor session starts without a project. Do not resolve an
    // empty path relative to the process working directory, otherwise a
    // stray sparse.ply next to the executable could be treated as the active
    // reconstruction.
    if (app.settings.project_dir[0] == '\0') {
        app.layout = {};
        app.has_sparse = false;
        app.has_mvs = false;
        app.has_model = false;
        app.has_mesh = false;
        return;
    }
    app.layout = resolve_layout(app.settings);
    std::error_code error;
    app.has_sparse = std::filesystem::exists(app.layout.sparse_ply, error);
    app.has_mvs = std::filesystem::exists(app.layout.sparse_mvs, error);
    app.has_model = std::filesystem::exists(app.layout.splat_ply, error);
    app.has_mesh = std::filesystem::exists(app.layout.mesh_ply, error);
}

void assign_default_project_folder(App& app) {
    if (app.settings.images_dir[0] == '\0' ||
        app.settings.project_dir[0] != '\0')
        return;

    const std::filesystem::path images(app.settings.images_dir.data());
    const std::filesystem::path parent = images.parent_path();
    const std::filesystem::path project =
        (parent.empty() ? images : parent) / "aetherscan_gui";
    const std::string text = project.string();
    std::snprintf(
        app.settings.project_dir.data(), app.settings.project_dir.size(), "%s",
        text.c_str());
    app.project_folder_automatic = true;
}

void clear_loaded_result(App& app) {
    app.scene.clear();
    app.scene_source.clear();
    app.camera = {};
    app.tab = ViewportTab::sparse;
    app.monitor.reset();
    app.log.clear();
    app.console = {};
    app.fresh_lines.clear();
}

void select_image_folder(App& app) {
    if (app.job.running() || app.loading_scene) return;
    if (!pick_folder(
            L"Select the capture image folder", app.settings.images_dir))
        return;

    clear_loaded_result(app);
    if (app.project_folder_automatic) {
        app.settings.project_dir.fill('\0');
        app.project_folder_automatic = false;
    }
    assign_default_project_folder(app);
    refresh_artifacts(app);
    set_message(
        app, "Image dataset selected; previous viewport result cleared",
        theme::text_muted);
}

void request_scene_load(
    App& app, const std::filesystem::path& cloud,
    const std::filesystem::path& poses, std::string label) {
    if (app.loading_scene) return;
    app.loading_scene = true;
    app.scene_source = std::move(label);
    app.pending_load = std::async(
        std::launch::async,
        [cloud, poses] { return load_sparse_scene(cloud, poses); });
}

void ensure_sparse_loaded(App& app) {
    if (!app.has_sparse || app.scene.has_points() || app.loading_scene) return;
    request_scene_load(
        app, app.layout.sparse_ply, app.layout.sparse_poses, "Sparse cloud");
}

const ViewPose* first_registered_view(const SparseScene& scene) {
    for (const ViewPose& pose : scene.views) {
        if (pose.registered) return &pose;
    }
    return scene.views.empty() ? nullptr : &scene.views.front();
}

bool orbit_pose_changed(const OrbitCamera& a, const OrbitCamera& b) {
    const auto differs = [](const float left, const float right) {
        return std::abs(left - right) > 1e-5F;
    };
    return differs(a.yaw, b.yaw) || differs(a.pitch, b.pitch) ||
           differs(a.distance, b.distance) || differs(a.fov_degrees, b.fov_degrees) ||
           differs(a.target.x, b.target.x) || differs(a.target.y, b.target.y) ||
           differs(a.target.z, b.target.z);
}

void fit_preview_raster(
    const float viewport_w, const float viewport_h, std::uint32_t& width,
    std::uint32_t& height) {
    const float view_w = std::max(1.F, viewport_w);
    const float view_h = std::max(1.F, viewport_h);
    if (view_w >= view_h) {
        width = k_preview_extent;
        height = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(std::lround(
                   static_cast<double>(k_preview_extent) * view_h / view_w)));
    } else {
        height = k_preview_extent;
        width = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(std::lround(
                   static_cast<double>(k_preview_extent) * view_w / view_h)));
    }
}

void sync_live_preview_camera(
    App& app, const bool force, std::uint32_t width, std::uint32_t height) {
    const bool live =
        app.job.running() && app.active_job == JobKind::train;
    if (!force && !live) return;
    if (app.layout.preview_camera_file.empty()) return;
    width = std::max<std::uint32_t>(1, width);
    height = std::max<std::uint32_t>(1, height);
    if (!force && app.has_last_preview_orbit &&
        !orbit_pose_changed(app.camera, app.last_preview_orbit) &&
        width == app.preview_raster_width && height == app.preview_raster_height)
        return;
    ++app.preview_camera_revision;
    app.preview_raster_width = width;
    app.preview_raster_height = height;
    const SplatPreviewCamera preview =
        make_preview_camera(app.camera, width, height);
    write_preview_camera_file(
        app.layout.preview_camera_file, preview, app.preview_camera_revision);
    app.last_preview_orbit = app.camera;
    app.has_last_preview_orbit = true;
}

void snap_preview_to_index(App& app, const unsigned index) {
    app.preview_view = index;
    app.preview_follow_view = true;
    write_preview_view_index(app.layout, app.preview_view);
    if (index < app.scene.views.size())
        snap_orbit_to_view(app.camera, app.scene.views[index]);
    sync_live_preview_camera(
        app, true, app.preview_raster_width, app.preview_raster_height);
}

void select_project_folder(App& app) {
    if (app.job.running() || app.loading_scene) return;
    if (!pick_folder(
            L"Select the AetherScan project folder",
            app.settings.project_dir))
        return;

    // Switching projects is an explicit scene transition. Clear the previous
    // reconstruction immediately, then load the selected project's sparse
    // result when one is available.
    clear_loaded_result(app);
    app.project_folder_automatic = false;
    refresh_artifacts(app);
    if (app.has_sparse)
        request_scene_load(
            app, app.layout.sparse_ply, app.layout.sparse_poses,
            "Sparse cloud");
    else
        set_message(app, "Project selected; no sparse cloud yet", theme::text_muted);
}

bool has_reconstruction_result(const App& app) {
    if (app.scene.has_points() || app.has_sparse || app.has_mvs ||
        app.has_model || app.has_mesh)
        return true;
    if (app.layout.cache.empty()) return false;
    std::error_code error;
    return std::filesystem::exists(app.layout.cache, error);
}

void delete_reconstruction_results(App& app) {
    if (app.job.running() || app.loading_scene || app.layout.root.empty()) return;

    clear_loaded_result(app);
    const std::array<std::filesystem::path, 9> generated_files = {
        app.layout.sparse_ply, app.layout.sparse_mvs, app.layout.sparse_poses,
        app.layout.model_output, app.layout.splat_ply, app.layout.mesh_ply,
        app.layout.align_log, app.layout.train_log, app.layout.export_log};

    std::uintmax_t removed = 0;
    std::string failure;
    for (const std::filesystem::path& path : generated_files) {
        if (path.empty()) continue;
        std::error_code error;
        if (std::filesystem::remove(path, error)) ++removed;
        if (error && failure.empty()) failure = error.message();
    }

    // resolve_layout() always makes this exact root/cache child. Re-check the
    // relationship before a recursive removal so a malformed path can never
    // broaden the deletion target.
    const std::filesystem::path root = app.layout.root.lexically_normal();
    const std::filesystem::path cache = app.layout.cache.lexically_normal();
    const bool safe_cache = !root.empty() && !cache.empty() && cache != root &&
                            cache.parent_path() == root &&
                            cache.filename() == "cache";
    if (safe_cache) {
        std::error_code error;
        removed += std::filesystem::remove_all(cache, error);
        if (error && failure.empty()) failure = error.message();
    }

    refresh_artifacts(app);
    if (failure.empty()) {
        set_message(
            app,
            "Reconstruction results cleared (" + std::to_string(removed) +
                " generated entries removed)",
            theme::success);
    } else {
        set_message(
            app, "Some reconstruction results could not be removed: " + failure,
            theme::danger);
    }
}

void poll_scene_load(App& app) {
    if (!app.loading_scene || !app.pending_load.valid()) return;
    if (app.pending_load.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
        return;
    SceneLoad loaded = app.pending_load.get();
    app.loading_scene = false;
    if (!loaded.ok) {
        set_message(app, "Point cloud: " + loaded.error, theme::danger);
        return;
    }
    app.scene = std::move(loaded.scene);
    const bool live_train =
        app.job.running() && app.active_job == JobKind::train;
    if (!live_train) app.camera.frame(app.scene);
    if (app.tab != ViewportTab::training) app.tab = ViewportTab::sparse;
    set_message(
        app,
        app.scene_source + ": " + format_count(app.scene.points.size()) +
            " points, " + std::to_string(app.scene.registered_views) + " / " +
            std::to_string(app.scene.total_views) + " views registered",
        theme::success);
}

void start_align(App& app) {
    if (app.job.running()) return;
    assign_default_project_folder(app);
    if (app.settings.project_dir[0] == '\0') {
        set_message(app, "Select a project output folder first", theme::warning);
        return;
    }
    refresh_artifacts(app);
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (error) {
        set_message(app, "Cannot create project directory", theme::danger);
        return;
    }
    {
        std::error_code stale;
        std::filesystem::remove(app.layout.sparse_mvs, stale);
    }
    if (!directory_has_images(app.settings.images_dir.data())) {
        set_message(
            app, "No images found in the selected source folder", theme::danger);
        return;
    }
    try {
        app.monitor.begin(JobKind::align);
        app.log.open(app.layout.align_log);
        app.job.start(
            build_align_command(
                AETHERSCAN_CLI_PATH, app.settings, app.layout),
            app.layout.align_log);
        app.active_job = JobKind::align;
        set_message(app, "Aligning cameras...", theme::accent);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}

bool alignment_cache_present(const App& app) {
    if (app.layout.cache.empty()) return false;
    std::error_code error;
    return std::filesystem::exists(app.layout.cache, error);
}

bool can_export_sfm(const App& app) {
    return app.settings.images_dir[0] != '\0' &&
           app.settings.project_dir[0] != '\0' &&
           (app.has_sparse || alignment_cache_present(app));
}

const char* running_job_caption(const JobKind kind) {
    switch (kind) {
        case JobKind::align: return "ALIGNING";
        case JobKind::export_sfm: return "EXPORTING";
        case JobKind::train: return "TRAINING";
        case JobKind::none: return "READY";
    }
    return "READY";
}

const char* stop_job_label(const JobKind kind) {
    switch (kind) {
        case JobKind::align: return "Stop Alignment";
        case JobKind::export_sfm: return "Stop Export";
        default: return "Stop Training";
    }
}

void start_export_sfm(App& app) {
    if (app.job.running()) return;
    assign_default_project_folder(app);
    if (app.settings.project_dir[0] == '\0') {
        set_message(app, "Select a project output folder first", theme::warning);
        return;
    }
    refresh_artifacts(app);
    if (!can_export_sfm(app)) {
        set_message(
            app, "Align photos before exporting SfM alignment", theme::warning);
        return;
    }
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (error) {
        set_message(app, "Cannot create project directory", theme::danger);
        return;
    }
    try {
        app.monitor.begin(JobKind::export_sfm);
        app.log.open(app.layout.export_log);
        app.job.start(
            build_export_sfm_command(
                AETHERSCAN_CLI_PATH, app.settings, app.layout),
            app.layout.export_log);
        app.active_job = JobKind::export_sfm;
        set_message(app, "Exporting SfM alignment...", theme::accent);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}

void start_train(App& app, const bool smoke) {
    if (app.job.running()) return;
    assign_default_project_folder(app);
    if (app.settings.project_dir[0] == '\0') {
        set_message(app, "Select a project output folder first", theme::warning);
        return;
    }
    refresh_artifacts(app);
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    app.preview_view = 0;
    app.preview_follow_view = true;
    load_view_poses(app.layout.sparse_poses, app.scene);
    if (const ViewPose* pose = first_registered_view(app.scene))
        snap_orbit_to_view(app.camera, *pose);
    write_preview_view_index(app.layout, app.preview_view);
    sync_live_preview_camera(
        app, true, app.preview_raster_width, app.preview_raster_height);
    ensure_sparse_loaded(app);

    // The trainer inherits the shared allocation, so the exported handles are
    // recreated per run: the timeline counter has to restart from zero.
    app.preview.create(k_preview_extent, k_preview_extent);
    PreviewHandles handles;
    handles.memory =
        reinterpret_cast<std::uintptr_t>(app.preview.memory_handle);
    handles.semaphore =
        reinterpret_cast<std::uintptr_t>(app.preview.semaphore_handle);
    handles.allocation_size = app.preview.allocation_size;
    handles.width = app.preview.width;
    handles.height = app.preview.height;
    handles.device_luid = gpu::device_luid();
    handles.device_node_mask = gpu::device_node_mask();

    std::string command;
    if (smoke) {
        // Fixed COLMAP dataset used only by --interop-smoke.
        std::ostringstream smoke_command;
        smoke_command
            << '"' << AETHERSCAN_CLI_PATH << "\" --images \""
            << app.settings.images_dir.data() << "\" --output \""
            << app.layout.model_output.string()
            << "\" --splat-dataset \"D:\\ScanVideo\\ori_img\""
            << " --splat-format colmap --splat-use-mask false"
            << " --splat --splat-strategy adc_plus --splat-iterations "
            << app.settings.iterations << " --splat-preview-interval "
            << app.settings.preview_interval
            << " --splat-preview-view 0 --splat-preview-view-file \""
            << app.layout.preview_view_file.string() << '"'
            << " --splat-preview-camera-file \""
            << app.layout.preview_camera_file.string() << '"'
            << " --splat-preview-vk-memory-handle " << handles.memory
            << " --splat-preview-vk-semaphore-handle " << handles.semaphore
            << " --splat-preview-vk-allocation-size " << handles.allocation_size
            << " --splat-preview-vk-width " << handles.width
            << " --splat-preview-vk-height " << handles.height
            << " --splat-preview-vk-device-luid " << handles.device_luid
            << " --splat-preview-vk-device-node-mask "
            << handles.device_node_mask;
        command = smoke_command.str();
    } else {
        command = build_train_command(
            AETHERSCAN_CLI_PATH, app.settings, app.layout, handles);
    }

    try {
        app.monitor.begin(JobKind::train);
        app.log.open(app.layout.train_log);
        app.job.start(command, app.layout.train_log);
        app.active_job = JobKind::train;
        app.tab = ViewportTab::training;
        set_message(
            app,
            app.settings.build_mesh
                ? "Training with depth/normal geometry supervision..."
                : "Training Gaussians...",
            theme::accent);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
    // The child has inherited the handles; the parent copies are no longer
    // needed and must not leak across runs.
    app.preview.close_export_handles();
}

void on_job_finished(App& app) {
    const int code = app.job.exit_code();
    app.monitor.mark_finished(code);
    const JobKind kind = app.active_job;
    app.active_job = JobKind::none;
    refresh_artifacts(app);

    if (code != 0) {
        std::string reason = app.monitor.last_error();
        if (reason.empty())
            reason = code == 2 ? "Stopped by user"
                               : "Exited with code " + std::to_string(code);
        set_message(
            app,
            std::string(job_name(kind)) + " failed: " + reason,
            code == 2 ? theme::warning : theme::danger);
        return;
    }

    if (kind == JobKind::align) {
        if (app.has_sparse) {
            request_scene_load(
                app, app.layout.sparse_ply, app.layout.sparse_poses,
                "Sparse cloud");
        } else {
            set_message(
                app, "Alignment finished but no sparse cloud was written",
                theme::warning);
        }
        return;
    }
    if (kind == JobKind::export_sfm) {
        set_message(
            app,
            app.has_mvs ? "Exported SfM alignment to sparse.mvs"
                        : "Export finished but sparse.mvs was not written",
            app.has_mvs ? theme::success : theme::warning);
        return;
    }
    set_message(
        app,
        app.settings.build_mesh
            ? (app.has_mesh ? "Training and mesh extraction finished"
                            : "Training finished, mesh extraction produced no "
                              "surface")
            : "Training finished",
        app.settings.build_mesh && !app.has_mesh ? theme::warning
                                                 : theme::success);
}

// ---------------------------------------------------------------------------
// UI fragments

enum class Action { none, align, train, export_sfm, stop, reveal };

int workflow_step(const App& app) {
    const bool training =
        app.job.running() && app.active_job == JobKind::train;
    const bool aligning =
        app.job.running() && app.active_job == JobKind::align;
    if (training && app.monitor.stage() == Stage::meshing) return 3;
    if (training) return 2;
    if (aligning) return 1;
    if (app.has_model) return 2;
    if (app.has_sparse) return 1;
    return 0;
}

void apply_default_dock_layout(const ImGuiID dockspace_id, const ImVec2 size) {
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, size);

    ImGuiID dock_main = dockspace_id;
    ImGuiID dock_left = 0;
    ImGuiID dock_right = 0;
    ImGuiID dock_bottom = 0;
    ImGui::DockBuilderSplitNode(
        dock_main, ImGuiDir_Left, 0.18F, &dock_left, &dock_main);
    ImGui::DockBuilderSplitNode(
        dock_main, ImGuiDir_Right, 0.24F, &dock_right, &dock_main);
    ImGui::DockBuilderSplitNode(
        dock_main, ImGuiDir_Down, 0.28F, &dock_bottom, &dock_main);

    ImGui::DockBuilderDockWindow("Scene", dock_left);
    ImGui::DockBuilderDockWindow("Viewport", dock_main);
    ImGui::DockBuilderDockWindow("Console", dock_bottom);
    ImGui::DockBuilderDockWindow("Inspector", dock_right);
    if (ImGuiDockNode* node = ImGui::DockBuilderGetNode(dock_main))
        node->LocalFlags |= ImGuiDockNodeFlags_AutoHideTabBar;
    ImGui::DockBuilderFinish(dockspace_id);
}

// Host window covering the work area under the toolbar and above the status
// bar. The default split matches the previous fixed layout so first launch
// still reads as Scene | Viewport+Console | Inspector.
void build_dock_space(App& app) {
    if (!(ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DockingEnable)) return;

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float status_h = app.show_status_bar ? k_status_height : 0.F;
    const ImVec2 pos{
        viewport->WorkPos.x, viewport->WorkPos.y + k_toolbar_height};
    const ImVec2 size{
        viewport->WorkSize.x,
        std::max(1.F, viewport->WorkSize.y - k_toolbar_height - status_h)};

    ImGui::SetNextWindowPos(pos);
    ImGui::SetNextWindowSize(size);
    ImGui::SetNextWindowViewport(viewport->ID);

    constexpr ImGuiWindowFlags host_flags =
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground |
        ImGuiWindowFlags_NoSavedSettings;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.F, 0.F));
    ImGui::Begin("##EditorDockSpaceHost", nullptr, host_flags);
    ImGui::PopStyleVar(3);

    // Bump the id when the default window set changes so old ini layouts
    // migrate instead of restoring a missing split.
    const ImGuiID dockspace_id = ImGui::GetID("EditorDockSpace_v1");
    if (app.reset_dock_layout ||
        ImGui::DockBuilderGetNode(dockspace_id) == nullptr) {
        app.show_scene = true;
        app.show_viewport = true;
        app.show_console = true;
        app.show_inspector = true;
        apply_default_dock_layout(dockspace_id, ImGui::GetContentRegionAvail());
        app.reset_dock_layout = false;
    }

    ImGui::DockSpace(
        dockspace_id, ImVec2(0.F, 0.F), ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();
}

Action draw_menu_bar(App& app) {
    Action action = Action::none;
    const bool busy = app.job.running();
    const ImGuiIO& io = ImGui::GetIO();
    if (!io.WantTextInput && io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O) &&
        !busy) {
        select_image_folder(app);
    }
    if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Escape) && busy)
        action = Action::stop;

    if (!ImGui::BeginMainMenuBar()) return action;
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Select Image Folder...", "Ctrl+O", false, !busy)) {
            select_image_folder(app);
        }
        if (ImGui::MenuItem("Set Project Folder...", nullptr, false, !busy)) {
            select_project_folder(app);
        }
        ImGui::Separator();
        if (ImGui::MenuItem(
                "Load Sparse Cloud", nullptr, false,
                app.has_sparse && !app.loading_scene)) {
            request_scene_load(
                app, app.layout.sparse_ply, app.layout.sparse_poses,
                "Sparse cloud");
        }
        if (ImGui::MenuItem(
                "Export SfM Alignment...", nullptr, false,
                !busy && can_export_sfm(app)))
            action = Action::export_sfm;
        if (ImGui::MenuItem(
                "Open Output Folder", nullptr, false,
                app.layout.root.has_filename()))
            action = Action::reveal;
        ImGui::Separator();
        if (ImGui::MenuItem("Exit", "Alt+F4")) app.close_requested = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Reconstruction")) {
        if (ImGui::MenuItem(
                app.has_sparse ? "Re-align Photos" : "Align Photos", nullptr,
                false, !busy && app.settings.images_dir[0] != '\0'))
            action = Action::align;
        if (ImGui::MenuItem(
                "Train 3DGS", nullptr, false,
                !busy && app.settings.images_dir[0] != '\0'))
            action = Action::train;
        if (ImGui::MenuItem(
                "Export SfM Alignment...", nullptr, false,
                !busy && can_export_sfm(app)))
            action = Action::export_sfm;
        ImGui::Separator();
        if (ImGui::MenuItem(
                "Clear Reconstruction Results...", nullptr, false,
                !busy && !app.loading_scene &&
                    has_reconstruction_result(app)))
            app.show_clear_results = true;
        ImGui::Separator();
        if (ImGui::MenuItem("Stop Active Job", "Esc", false, busy))
            action = Action::stop;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
        ImGui::MenuItem("Scene", nullptr, &app.show_scene);
        ImGui::MenuItem("Viewport", nullptr, &app.show_viewport);
        ImGui::MenuItem("Inspector", nullptr, &app.show_inspector);
        ImGui::MenuItem("Console", nullptr, &app.show_console);
        ImGui::MenuItem("Status Bar", nullptr, &app.show_status_bar);
        ImGui::Separator();
        if (ImGui::MenuItem("Reset Layout")) {
            app.show_scene = true;
            app.show_viewport = true;
            app.show_console = true;
            app.show_inspector = true;
            app.show_status_bar = true;
            app.reset_dock_layout = true;
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("Viewport Controls")) app.show_controls = true;
        ImGui::EndMenu();
    }

    ImGui::SameLine(0.F, 28.F);
    ImGui::PushStyleColor(ImGuiCol_Text, theme::accent);
    ImGui::TextUnformatted("AETHER");
    ImGui::PopStyleColor();
    ImGui::SameLine(0.F, 1.F);
    ImGui::TextUnformatted("SCAN");

    const int current = workflow_step(app);
    const std::array<const char*, 4> steps{
        {"Images", "Alignment", "Gaussians", "Mesh"}};
    ImGui::SameLine(0.F, 22.F);
    for (int i = 0; i < static_cast<int>(steps.size()); ++i) {
        if (i > 0) {
            ImGui::SameLine(0.F, 8.F);
            theme::caption("\xE2\x80\xBA");
            ImGui::SameLine(0.F, 8.F);
        }
        const bool reached = i <= current;
        ImGui::PushStyleColor(
            ImGuiCol_Text, i == current ? theme::accent
                                        : (reached ? theme::text_muted
                                                   : theme::text_faint));
        ImGui::TextUnformatted(steps[static_cast<std::size_t>(i)]);
        ImGui::PopStyleColor();
    }

    const std::string project = app.layout.root.filename().empty()
        ? std::string("Untitled Project")
        : app.layout.root.filename().string();
    const float project_width = ImGui::CalcTextSize(project.c_str()).x;
    const float project_x =
        std::max(0.F, ImGui::GetWindowWidth() - project_width - 16.F);
    if (project_x > ImGui::GetCursorPosX() + 24.F) {
        ImGui::SameLine(project_x);
        theme::caption(project.c_str());
    }

    ImGui::EndMainMenuBar();
    return action;
}

void draw_controls_window(App& app) {
    if (!app.show_controls) return;
    ImGui::SetNextWindowSize({420.F, 0.F}, ImGuiCond_Appearing);
    if (ImGui::Begin("Viewport Controls", &app.show_controls,
                     ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("Camera");
        ImGui::Separator();
        ImGui::BulletText("LMB drag: orbit");
        ImGui::BulletText("MMB or Shift+LMB drag: pan");
        ImGui::BulletText("RMB drag: fly look");
        ImGui::BulletText("RMB + WASD/QE: fly; Shift accelerates");
        ImGui::BulletText("Mouse wheel: dolly; F: frame reconstruction");
    }
    ImGui::End();
}

void draw_clear_results_modal(App& app) {
    constexpr const char* popup = "Clear Reconstruction Results";
    if (app.show_clear_results) {
        ImGui::OpenPopup(popup);
        app.show_clear_results = false;
    }

    ImGui::SetNextWindowSize({470.F, 0.F}, ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(
            popup, nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        return;

    ImGui::TextUnformatted("Clear the current reconstruction?");
    ImGui::Spacing();
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 440.F);
    ImGui::TextColored(
        theme::warning,
        "Clear View Only removes the old result from the editor but keeps all "
        "project files.");
    ImGui::TextColored(
        theme::danger,
        "Delete Generated Results permanently removes sparse clouds, trained "
        "models, meshes, logs, and the reconstruction cache from the current "
        "project. Source images are never deleted.");
    ImGui::PopTextWrapPos();
    if (!app.layout.root.empty()) {
        ImGui::Spacing();
        theme::caption("Current project");
        ImGui::TextWrapped("%s", app.layout.root.string().c_str());
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (theme::toolbar_button("Clear View Only", {126.F, 30.F})) {
        clear_loaded_result(app);
        set_message(app, "Loaded reconstruction cleared from view", theme::success);
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (theme::danger_button("Delete Generated Results", {190.F, 30.F})) {
        delete_reconstruction_results(app);
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", {92.F, 30.F})) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

Action draw_toolbar(App& app) {
    Action action = Action::none;
    const bool busy = app.job.running();
    const bool images_ready = app.settings.images_dir[0] != '\0';

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize({viewport->WorkSize.x, k_toolbar_height});
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.F, 10.F));
    ImGui::PushStyleColor(
        ImGuiCol_WindowBg, ImVec4(0.106F, 0.110F, 0.122F, 1.F));
    constexpr ImGuiWindowFlags toolbar_flags =
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoScrollbar;
    ImGui::Begin("##toolbar", nullptr, toolbar_flags);
    ImGui::PopStyleVar(3);

    if (icons::labeled_button(
            "##image_folder", icons::Icon::folder, "Image Folder",
            {124.F, 32.F}, icons::ButtonStyle::normal, !busy, false,
            "Select capture image folder")) {
        select_image_folder(app);
    }
    ImGui::SameLine();

    // Step 1: multi-view alignment.
    const bool aligning = busy && app.active_job == JobKind::align;
    if (aligning) {
        icons::labeled_button(
            "##aligning", icons::Icon::align, "Aligning...", {132.F, 32.F},
            icons::ButtonStyle::primary, false, true);
    } else if (icons::labeled_button(
                   "##align", icons::Icon::align,
                   app.has_sparse ? "Re-align Photos" : "Align Photos",
                   {132.F, 32.F}, icons::ButtonStyle::primary,
                   !busy && images_ready)) {
        action = Action::align;
    }
    ImGui::SameLine();

    // Step 2: Gaussian optimisation, gated on a reviewed alignment.
    const bool training = busy && app.active_job == JobKind::train;
    if (training) {
        icons::labeled_button(
            "##training", icons::Icon::train, "Training...", {126.F, 32.F},
            icons::ButtonStyle::primary, false, true);
    } else {
        const bool ready = !busy && images_ready;
        if (app.has_sparse) {
            if (icons::labeled_button(
                    "##train", icons::Icon::train, "Train 3DGS",
                    {126.F, 32.F}, icons::ButtonStyle::primary, ready))
                action = Action::train;
        } else if (icons::labeled_button(
                       "##train", icons::Icon::train, "Train 3DGS",
                       {126.F, 32.F}, icons::ButtonStyle::normal, ready)) {
            action = Action::train;
        }
        if (!app.has_sparse && ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "No alignment yet. Training will run Structure from Motion "
                "first, then optimise Gaussians.");
    }
    ImGui::SameLine();

    ImGui::AlignTextToFramePadding();
    ImGui::Checkbox("Build Mesh", &app.settings.build_mesh);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Extract a surface after training.\n"
            "Enables depth-normal consistency and multi-view geometry/NCC\n"
            "supervision during optimisation, which the mesh needs.");
    ImGui::SameLine(0.F, 14.F);

    if (icons::labeled_button(
            "##open_output", icons::Icon::output, "Open Output",
            {116.F, 32.F}, icons::ButtonStyle::normal,
            app.layout.root.has_filename()))
        action = Action::reveal;
    if (busy) {
        ImGui::SameLine();
        if (icons::labeled_button(
                "##stop", icons::Icon::stop, "Stop", {76.F, 32.F},
                icons::ButtonStyle::danger))
            action = Action::stop;
    }

    const char* transport = "CUDA / Vulkan  |  external memory";
    const float transport_width = ImGui::CalcTextSize(transport).x;
    ImGui::SameLine(
        std::max(0.F, ImGui::GetWindowWidth() - transport_width - 16.F));
    ImGui::SetCursorPosY(18.F);
    theme::caption(transport);

    ImGui::End();
    ImGui::PopStyleColor();
    return action;
}

void draw_step(
    const char* index, const char* label, const StepState state,
    const char* note) {
    ImVec4 colour = theme::inactive;
    switch (state) {
        case StepState::done: colour = theme::success; break;
        case StepState::active: {
            const float pulse =
                0.55F + 0.45F * std::abs(std::sin(
                                    static_cast<float>(ImGui::GetTime()) * 2.4F));
            colour = theme::fade(theme::accent, pulse);
            break;
        }
        case StepState::failed: colour = theme::danger; break;
        case StepState::skipped: colour = theme::fade(theme::inactive, 0.4F); break;
        case StepState::pending: break;
    }

    ImGui::SetCursorPosX(14.F);
    theme::status_dot(colour);
    ImGui::SameLine(0.F, 8.F);
    theme::caption(index);
    ImGui::SameLine(0.F, 8.F);
    if (state == StepState::done || state == StepState::active) {
        ImGui::TextUnformatted(label);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Text, theme::text_muted);
        ImGui::TextUnformatted(label);
        ImGui::PopStyleColor();
    }
    if (note) {
        ImGui::SetCursorPosX(44.F);
        theme::caption(note);
    }
}

void draw_scene_panel(App& app) {
    if (!app.show_scene) return;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.F, 0.F));
    const bool open = ImGui::Begin("Scene", &app.show_scene);
    ImGui::PopStyleVar();
    if (!open) {
        ImGui::End();
        return;
    }

    const float wrap = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - 20.F;

    theme::section_header("SCENE");
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {7.F, 7.F});
    ImGui::SetCursorPosX(10.F);
    if (ImGui::TreeNodeEx(
            app.layout.root.filename().empty()
                ? "Capture"
                : app.layout.root.filename().string().c_str(),
            ImGuiTreeNodeFlags_DefaultOpen |
                ImGuiTreeNodeFlags_SpanAvailWidth)) {
        const auto leaf = [](const char* label, const bool present,
                             const bool selected) {
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf |
                                       ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                       ImGuiTreeNodeFlags_SpanAvailWidth;
            if (selected) flags |= ImGuiTreeNodeFlags_Selected;
            if (!present) ImGui::PushStyleColor(ImGuiCol_Text, theme::text_faint);
            ImGui::TreeNodeEx(label, flags);
            if (!present) ImGui::PopStyleColor();
        };
        leaf("Input Images", app.settings.images_dir[0] != '\0', false);
        leaf("Camera Poses", !app.scene.views.empty(), false);
        leaf(
            "Sparse Point Cloud", app.has_sparse || app.scene.has_points(),
            app.tab == ViewportTab::sparse);
        if (ImGui::IsItemClicked() && app.has_sparse) {
            app.tab = ViewportTab::sparse;
            ensure_sparse_loaded(app);
        }
        leaf("OpenMVS Export", app.has_mvs, false);
        leaf(
            "Gaussian Model", app.has_model,
            app.tab == ViewportTab::training);
        if (ImGui::IsItemClicked()) app.tab = ViewportTab::training;
        leaf("Reconstructed Mesh", app.has_mesh, false);
        ImGui::TreePop();
    }
    ImGui::PopStyleVar();
    ImGui::Dummy({0, 6.F});

    theme::section_header("PIPELINE");
    const bool busy = app.job.running();
    const Stage stage = app.monitor.stage();
    const bool aligning = busy && app.active_job == JobKind::align;
    const bool training = busy && app.active_job == JobKind::train;

    draw_step(
        "01", "Select images",
        app.settings.images_dir[0] != '\0' ? StepState::done : StepState::pending,
        nullptr);
    draw_step(
        "02", "Align cameras",
        aligning ? StepState::active
                 : (app.has_sparse ? StepState::done : StepState::pending),
        aligning ? stage_name(stage) : nullptr);
    draw_step(
        "03", "Review sparse cloud",
        app.scene.has_points() ? StepState::done
                               : (app.has_sparse ? StepState::active
                                                 : StepState::pending),
        app.scene.has_points()
            ? nullptr
            : (app.has_sparse
                   ? (app.loading_scene ? "Loading" : "On disk")
                   : nullptr));
    draw_step(
        "04", "Optimise Gaussians",
        training && stage != Stage::meshing
            ? StepState::active
            : (app.has_model ? StepState::done : StepState::pending),
        app.settings.build_mesh ? "Geometry constraints on" : nullptr);
    draw_step(
        "05", "Extract mesh",
        !app.settings.build_mesh
            ? StepState::skipped
            : (training && stage == Stage::meshing
                   ? StepState::active
                   : (app.has_mesh ? StepState::done : StepState::pending)),
        app.settings.build_mesh ? nullptr : "Disabled");

    ImGui::Dummy({0, 8.F});
    theme::section_header("SOURCE");
    ImGui::Indent(14.F);
    theme::caption("IMAGES");
    ImGui::Spacing();
    ImGui::PushTextWrapPos(wrap);
    ImGui::TextUnformatted(
        app.settings.images_dir[0] != '\0' ? app.settings.images_dir.data()
                                          : "(not selected)");
    ImGui::PopTextWrapPos();
    ImGui::Dummy({0, 6.F});
    theme::caption("PROJECT");
    ImGui::Spacing();
    ImGui::PushTextWrapPos(wrap);
    ImGui::TextUnformatted(
        app.settings.project_dir[0] != '\0' ? app.settings.project_dir.data()
                                           : "(not selected)");
    ImGui::PopTextWrapPos();
    ImGui::Unindent(14.F);

    ImGui::End();
}


void draw_empty_viewport(
    ImDrawList* draw, const ImVec2 min, const ImVec2 max, const char* headline,
    const char* hint) {
    draw->PushClipRect(min, max, true);
    const float headline_width = ImGui::CalcTextSize(headline).x;
    const float hint_width = ImGui::CalcTextSize(hint).x;
    const float centre_x = (min.x + max.x) * 0.5F;
    const float y = min.y + (max.y - min.y) * 0.18F;
    draw->AddText(
        {centre_x - headline_width * 0.5F, y},
        theme::u32(theme::text_muted), headline);
    draw->AddText(
        {centre_x - hint_width * 0.5F, y + 20.F},
        theme::u32(theme::text_faint), hint);
    draw->PopClipRect();
}

void draw_viewport_overlay(
    ImDrawList* draw, const ImVec2 min, const char* label, const ImVec4& dot) {
    const float text_width = ImGui::CalcTextSize(label).x;
    const ImVec2 origin{min.x + 14.F, min.y + 14.F};
    const ImVec2 max{origin.x + text_width + 42.F, origin.y + 30.F};
    draw->AddRectFilled(origin, max, IM_COL32(16, 18, 23, 214), 5.F);
    draw->AddRect(origin, max, theme::u32(theme::border, 0.7F), 5.F);
    draw->AddCircleFilled(
        {origin.x + 16.F, origin.y + 15.F}, 4.F, theme::u32(dot));
    draw->AddText(
        {origin.x + 28.F, origin.y + 15.F - ImGui::GetTextLineHeight() * 0.5F},
        theme::u32(theme::text_bright), label);
}

void draw_sparse_tab(App& app, const ImVec2 min, const ImVec2 max) {
    ensure_sparse_loaded(app);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilledMultiColor(
        min, max, IM_COL32(9, 11, 16, 255), IM_COL32(9, 11, 16, 255),
        IM_COL32(18, 22, 32, 255), IM_COL32(18, 22, 32, 255));

    ImGui::SetCursorScreenPos(min);
    ImGui::InvisibleButton(
        "##sparse_view",
        {std::max(1.F, max.x - min.x), std::max(1.F, max.y - min.y)},
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
            ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();

    const SceneDrawStats stats = app.renderer.draw(
        draw, min, max, app.scene, app.camera, app.view_options, hovered);

    const bool gizmo_captures =
        draw_viewport_gizmo(app.gizmo, app.camera, min, max);
    const bool frame_key = hovered && !gizmo_captures &&
                           !ImGui::GetIO().WantTextInput &&
                           ImGui::IsKeyPressed(ImGuiKey_F);
    if (frame_key) app.camera.frame(app.scene);
    update_orbit_camera(
        app.camera, hovered && !gizmo_captures, app.scene.radius);

    const char* overlay = "NO ALIGNMENT";
    ImVec4 overlay_dot = theme::inactive;
    if (app.loading_scene) {
        overlay = "LOADING";
        overlay_dot = theme::warning;
    } else if (app.scene.has_points()) {
        overlay = app.scene_source.empty() ? "SPARSE POINT CLOUD"
                                           : app.scene_source.c_str();
        overlay_dot = theme::accent;
    } else if (app.has_sparse) {
        overlay = "CLOUD READY";
    }
    draw_viewport_overlay(draw, min, overlay, overlay_dot);

    if (app.scene.has_points()) {
        char readout[192];
        std::snprintf(
            readout, sizeof(readout),
            "%s pts drawn  |  %zu / %zu cameras shown  |  %s pts total",
            format_count(stats.drawn_points).c_str(), stats.drawn_views,
            app.scene.registered_views,
            format_count(app.scene.points.size()).c_str());
        draw->AddText(
            {min.x + 16.F, max.y - 42.F}, theme::u32(theme::text_muted), readout);
    } else if (!app.loading_scene) {
        const char* hint = app.has_sparse
            ? "Reading sparse.ply..."
            : "Pick an image folder, then Align Photos";
        draw->AddText(
            {min.x + 16.F, max.y - 42.F}, theme::u32(theme::text_faint), hint);
    }
    draw->AddText(
        {min.x + 16.F, max.y - 24.F}, theme::u32(theme::text_faint),
        "LMB orbit  |  MMB pan  |  RMB + WASD/QE fly  |  wheel dolly  |  F frame");

    if (!gizmo_captures && stats.hovered_view >= 0 &&
        static_cast<std::size_t>(stats.hovered_view) < app.scene.views.size()) {
        const ViewPose& pose = app.scene.views[stats.hovered_view];
        ImGui::SetTooltip(
            "%s\n%u x %u  |  f %.1f px\n%zu observations  |  p95 %.2f px",
            pose.name.c_str(), pose.width, pose.height, pose.fx,
            pose.observations, pose.reprojection_p95);
    }
}

unsigned preview_camera_count(const App& app) {
    if (!app.scene.views.empty())
        return static_cast<unsigned>(app.scene.views.size());
    return std::max(1U, app.preview_view + 1U);
}

void step_preview_view(App& app, const int delta) {
    unsigned next = app.preview_view;
    if (!app.scene.views.empty()) {
        const int count = static_cast<int>(app.scene.views.size());
        int wrapped = (static_cast<int>(app.preview_view) + delta) % count;
        if (wrapped < 0) wrapped += count;
        next = static_cast<unsigned>(wrapped);
    } else {
        next = static_cast<unsigned>(
            std::max(0, static_cast<int>(app.preview_view) + delta));
    }
    if (next == app.preview_view && app.preview_follow_view) return;
    snap_preview_to_index(app, next);
}

void handle_preview_view_input(App& app, const bool hovered) {
    if (!hovered || ImGui::GetIO().WantTextInput) return;
    if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) return;
    if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))
        step_preview_view(app, -1);
    if (ImGui::IsKeyPressed(ImGuiKey_RightArrow))
        step_preview_view(app, 1);
}

void draw_training_tab(App& app, const ImVec2 min, const ImVec2 max) {
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(min, max, theme::u32(theme::viewport_bg));

    ImGui::SetCursorScreenPos(min);
    ImGui::InvisibleButton(
        "##training_view",
        {std::max(1.F, max.x - min.x), std::max(1.F, max.y - min.y)},
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
            ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();

    const bool training = app.job.running() && app.active_job == JobKind::train;
    const bool has_frame = app.preview.display.descriptor &&
                           gpu::consumed_timeline_value() > 0;
    const unsigned view_count = preview_camera_count(app);
    char camera_label[64];
    if (!app.preview_follow_view || app.scene.views.empty())
        std::snprintf(camera_label, sizeof(camera_label), "orbit camera");
    else
        std::snprintf(
            camera_label, sizeof(camera_label), "camera %u / %u",
            app.preview_view + 1, view_count);

    const char* controls = training
        ? "LMB orbit  |  MMB pan  |  RMB + WASD fly  |  arrows snap capture"
        : "Resume Train 3DGS to move this camera";

    if (!has_frame) {
        draw_empty_viewport(
            draw, min, max,
            training ? "Waiting for the first rendered iteration..."
                     : "No live training preview",
            training ? "Orbit the view; the first frame uses this camera"
                     : "Run Train 3DGS to stream the optimiser output");
    } else {
        draw->AddImage(
            reinterpret_cast<ImTextureID>(app.preview.display.descriptor),
            min, max);
    }

    const bool gizmo_captures =
        draw_viewport_gizmo(app.gizmo, app.camera, min, max);
    update_orbit_camera(
        app.camera, hovered && !gizmo_captures, app.scene.radius);
    if (app.camera.interacting ||
        (hovered && !gizmo_captures && ImGui::GetIO().MouseWheel != 0.F))
        app.preview_follow_view = false;
    handle_preview_view_input(app, hovered && !gizmo_captures);
    std::uint32_t raster_w = app.preview_raster_width;
    std::uint32_t raster_h = app.preview_raster_height;
    fit_preview_raster(max.x - min.x, max.y - min.y, raster_w, raster_h);
    sync_live_preview_camera(app, false, raster_w, raster_h);

    draw_viewport_overlay(
        draw, min,
        has_frame ? (training ? "LIVE TRAINING PREVIEW" : "LAST TRAINING FRAME")
                  : (training ? "TRAINING" : "IDLE"),
        has_frame ? (training ? theme::success : theme::inactive)
                  : (training ? theme::warning : theme::inactive));

    const TrainingStats& stats = app.monitor.training();
    if (has_frame && stats.valid) {
        char readout[192];
        std::snprintf(
            readout, sizeof(readout),
            "iter %u / %u  |  %s gaussians  |  loss %.4f  |  %.1f ms/step  |  %s",
            stats.iteration, stats.total_iterations,
            format_count(stats.gaussians).c_str(), stats.loss,
            stats.step_milliseconds, camera_label);
        draw->AddText(
            {min.x + 16.F, max.y - 42.F}, theme::u32(theme::text_muted),
            readout);
    } else if (has_frame) {
        draw->AddText(
            {min.x + 16.F, max.y - 42.F}, theme::u32(theme::text_muted),
            camera_label);
    }
    draw->AddText(
        {min.x + 16.F, max.y - 24.F}, theme::u32(theme::text_faint),
        controls);
}

void draw_viewport_panel(App& app) {
    if (!app.show_viewport) return;
    constexpr ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.F, 0.F));
    const bool open = ImGui::Begin("Viewport", &app.show_viewport, flags);
    ImGui::PopStyleVar();
    if (!open) {
        ImGui::End();
        return;
    }

    // SetCursorPos is window-relative and includes the dock tab bar. The
    // Sparse Cloud / Live Training buttons used to sit at (8, 5) under the
    // "Viewport" tab. Layout from the content origin Begin() already set.
    if (ImGuiDockNode* node = ImGui::GetWindowDockNode())
        node->LocalFlags |= ImGuiDockNodeFlags_AutoHideTabBar;

    constexpr float k_header_height = 36.F;
    const ImVec2 content_start = ImGui::GetCursorPos();
    const ImVec2 header_origin = ImGui::GetCursorScreenPos();
    const float header_width = ImGui::GetContentRegionAvail().x;
    ImGui::GetWindowDrawList()->AddRectFilled(
        header_origin,
        {header_origin.x + header_width, header_origin.y + k_header_height},
        theme::u32(theme::surface_3));
    ImGui::GetWindowDrawList()->AddLine(
        {header_origin.x, header_origin.y + k_header_height - 1.F},
        {header_origin.x + header_width, header_origin.y + k_header_height - 1.F},
        theme::u32(theme::border));

    ImGui::SetCursorPos({content_start.x + 8.F, content_start.y + 5.F});
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12.F, 4.F));
    if (theme::toolbar_button(
            "Sparse Cloud", {0, 26.F}, true, app.tab == ViewportTab::sparse))
        app.tab = ViewportTab::sparse;
    ImGui::SameLine(0.F, 4.F);
    if (theme::toolbar_button(
            "Live Training", {0, 26.F}, true, app.tab == ViewportTab::training))
        app.tab = ViewportTab::training;
    ImGui::PopStyleVar();

    const char* state = app.job.running()
        ? running_job_caption(app.active_job)
        : "READY";
    const float state_width = ImGui::CalcTextSize(state).x;
    ImGui::SetCursorPos({
        content_start.x + std::max(8.F, header_width - state_width - 14.F),
        content_start.y + 10.F});
    theme::caption(state);
    ImGui::SetCursorPos({content_start.x, content_start.y + k_header_height});

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
    ImGui::BeginChild(
        "##view", {0, 0}, false,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar();
    const ImVec2 view_min = ImGui::GetCursorScreenPos();
    const ImVec2 region = ImGui::GetContentRegionAvail();
    const ImVec2 view_max{view_min.x + region.x, view_min.y + region.y};
    if (app.tab == ViewportTab::sparse)
        draw_sparse_tab(app, view_min, view_max);
    else
        draw_training_tab(app, view_min, view_max);
    ImGui::EndChild();
    ImGui::End();
}

void draw_console_panel(App& app) {
    if (!app.show_console) return;
    editor::draw_console(
        app.show_console, app.console, app.log, app.job.running(),
        app.active_job, app.monitor.stage());
}

Action draw_inspector(App& app) {
    Action action = Action::none;
    if (!app.show_inspector) return action;
    if (!ImGui::Begin("Inspector", &app.show_inspector)) {
        ImGui::End();
        return action;
    }
    const bool busy = app.job.running();

    if (ImGui::CollapsingHeader("Project", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        theme::caption("Image source");
        ImGui::SetNextItemWidth(-30.F);
        if (ImGui::InputText(
                "##images", app.settings.images_dir.data(),
                app.settings.images_dir.size())) {
            clear_loaded_result(app);
            if (app.project_folder_automatic) {
                app.settings.project_dir.fill('\0');
                app.project_folder_automatic = false;
            }
            assign_default_project_folder(app);
            refresh_artifacts(app);
        }
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button("...##pick_images", {24.F, 0})) {
            select_image_folder(app);
        }
        theme::caption("Project directory");
        ImGui::SetNextItemWidth(-30.F);
        if (ImGui::InputText(
                "##project", app.settings.project_dir.data(),
                app.settings.project_dir.size())) {
            clear_loaded_result(app);
            app.project_folder_automatic = false;
            refresh_artifacts(app);
        }
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button("...##pick_project", {24.F, 0})) {
            select_project_folder(app);
        }
        ImGui::Spacing();
        if (theme::danger_button(
                "Clear Reconstruction Results...", {-1.F, 28.F},
                !busy && !app.loading_scene &&
                    has_reconstruction_result(app)))
            app.show_clear_results = true;
        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader(
            "Camera Alignment", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        ImGui::BeginDisabled(busy);
        theme::caption("Solver");
        ImGui::SetNextItemWidth(-1.F);
        const char* modes[] = {"Global", "Incremental", "Hierarchical"};
        ImGui::Combo("##sfm_mode", &app.settings.sfm_mode, modes, 3);
        theme::caption("Max features per image");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::InputInt("##max_features", &app.settings.max_features, 1000, 5000);
        ImGui::Checkbox("Reuse cached alignment", &app.settings.reuse_cache);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Reuse the Structure-from-Motion cache so Train 3DGS can\n"
                "reload cameras and sparse points into memory instead of\n"
                "solving poses again. Uncheck to rebuild from the images.\n"
                "OpenMVS files are not written unless you export.");
        ImGui::EndDisabled();

        if (theme::toolbar_button(
                "Export SfM Alignment", {-1.F, 28.F},
                !busy && can_export_sfm(app)))
            action = Action::export_sfm;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Write cameras and sparse points to sparse.mvs.\n"
                "Align Photos and Train 3DGS keep the reconstruction in\n"
                "memory (via the alignment cache) and do not write this file.");
        if (app.has_mvs) {
            ImGui::Spacing();
            theme::metric_coloured(
                "OpenMVS file", "sparse.mvs", theme::success);
        }

        if (!app.scene.views.empty()) {
            ImGui::Spacing();
            theme::metric_coloured(
                "Registered views",
                (std::to_string(app.scene.registered_views) + " / " +
                 std::to_string(app.scene.total_views))
                    .c_str(),
                app.scene.registered_views == app.scene.total_views
                    ? theme::success
                    : theme::warning);
            char buffer[64];
            std::snprintf(
                buffer, sizeof(buffer), "%.3f px", app.scene.mean_reprojection);
            theme::metric("Mean reprojection", buffer);
            theme::metric(
                "Sparse points",
                format_count(app.scene.points.size()).c_str());
        }
        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader(
            "Gaussian Splatting", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        ImGui::BeginDisabled(busy);
        theme::caption("Capture type");
        ImGui::SetNextItemWidth(-1.F);
        const char* capture[] = {"Object", "Scene"};
        int capture_index = app.settings.scene_mode ? 1 : 0;
        if (ImGui::Combo("##capture", &capture_index, capture, 2))
            app.settings.scene_mode = capture_index == 1;
        theme::caption("Densification strategy");
        ImGui::SetNextItemWidth(-1.F);
        const char* strategies[] = {
            "Default", "ADC Plus", "ADC IGS", "Dense adaptive"};
        ImGui::Combo("##strategy", &app.settings.strategy, strategies, 4);
        theme::caption("Iterations");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::InputInt("##iterations", &app.settings.iterations, 1000, 5000);
        theme::caption("Max training resolution");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::InputInt(
            "##resolution", &app.settings.max_resolution, 128, 512);
        theme::caption("Live preview cadence");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::InputInt("##cadence", &app.settings.preview_interval, 10, 50);
        ImGui::Checkbox(
            "Coarse-to-fine resolution", &app.settings.progressive_resolution);
        ImGui::Checkbox("Foreground mask training", &app.settings.use_mask);
        ImGui::Checkbox("Normal field", &app.settings.normal_field);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "GaussianWrapping's learned normal field.\n"
                "Off (default) trains the GGGS path.");
        ImGui::EndDisabled();
        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader(
            "Mesh & Geometry", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        ImGui::BeginDisabled(busy);
        ImGui::Checkbox(
            "Build mesh after training", &app.settings.build_mesh);
        ImGui::Spacing();
        if (app.settings.build_mesh) {
            ImGui::PushTextWrapPos(0.F);
            ImGui::PushStyleColor(ImGuiCol_Text, theme::accent);
            ImGui::TextUnformatted(
                "Training will add depth-normal consistency and multi-view "
                "geometry/NCC supervision so the learned depth is metric "
                "enough to fuse.");
            ImGui::PopStyleColor();
            ImGui::PopTextWrapPos();
            ImGui::Spacing();
            theme::caption("Surface backend");
            ImGui::SetNextItemWidth(-1.F);
            const char* methods[] = {"Auto", "TSDF", "Delaunay", "PAM"};
            ImGui::Combo("##mesh_method", &app.settings.mesh_method, methods, 4);
            theme::caption("Depth-normal weight");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::DragFloat(
                "##depth_normal", &app.settings.depth_normal_weight, 0.005F,
                0.F, 1.F, "%.3f");
            theme::caption("Multi-view geometry weight");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::DragFloat(
                "##mv_geo", &app.settings.multi_view_geo_weight, 0.005F, 0.F,
                1.F, "%.3f");
            theme::caption("Multi-view NCC weight");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::DragFloat(
                "##mv_ncc", &app.settings.multi_view_ncc_weight, 0.01F, 0.F,
                2.F, "%.2f");
            theme::caption("Geometry loss start iteration");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputInt(
                "##geo_from", &app.settings.geometry_from_iter, 500, 2000);
        } else {
            ImGui::PushTextWrapPos(0.F);
            theme::caption(
                "Appearance-only training. Geometry losses stay off and no "
                "surface is extracted.");
            ImGui::PopTextWrapPos();
        }
        ImGui::EndDisabled();
        ImGui::Spacing();
    }

    if (app.tab == ViewportTab::sparse &&
        ImGui::CollapsingHeader("Sparse View", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        if (theme::toolbar_button(
                "Load Sparse Cloud", {-1.F, 28.F},
                app.has_sparse && !app.loading_scene))
            request_scene_load(
                app, app.layout.sparse_ply, app.layout.sparse_poses,
                "Sparse cloud");
        if (theme::toolbar_button(
                "Load Trained Model", {-1.F, 28.F},
                app.has_model && !app.loading_scene))
            request_scene_load(
                app, app.layout.splat_ply, app.layout.sparse_poses,
                "Gaussian centres");
        ImGui::Spacing();
        theme::caption("Point size");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderFloat(
            "##point_size", &app.view_options.point_size, 1.F, 6.F, "%.1f px");
        theme::caption("Rendered point budget");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderInt(
            "##budget", &app.view_options.point_budget, 20'000, 600'000,
            "%d");
        theme::caption("Camera marker size");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderFloat(
            "##view_scale", &app.view_options.view_scale, 0.02F, 0.4F, "%.2f");
        theme::caption("Fly speed");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderFloat(
            "##fly_speed", &app.camera.move_speed, 0.1F, 10.F, "%.1fx");
        ImGui::Spacing();
        ImGui::Checkbox("Colour by depth", &app.view_options.colour_by_depth);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Replace sampled point colours with a near-to-far ramp.");
        ImGui::Checkbox("Show cameras", &app.view_options.show_views);
        ImGui::Checkbox("Show trajectory", &app.view_options.show_trajectory);
        ImGui::Checkbox("Show ground grid", &app.view_options.show_grid);
        ImGui::Spacing();
    }

    const TrainingStats& stats = app.monitor.training();
    if (stats.valid &&
        ImGui::CollapsingHeader(
            "Training Telemetry", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        char buffer[64];
        std::snprintf(
            buffer, sizeof(buffer), "%u / %u", stats.iteration,
            stats.total_iterations);
        theme::metric("Iteration", buffer);
        const float ratio = stats.total_iterations > 0
            ? static_cast<float>(stats.iteration) /
                  static_cast<float>(stats.total_iterations)
            : 0.F;
        theme::progress_track({-1.F, 6.F}, ratio, theme::accent);
        ImGui::Spacing();
        theme::metric("Gaussians", format_count(stats.gaussians).c_str());
        std::snprintf(buffer, sizeof(buffer), "%.4f", stats.loss);
        theme::metric("Total loss", buffer);
        std::snprintf(buffer, sizeof(buffer), "%.4f", stats.rgb_loss);
        theme::metric("RGB", buffer);
        if (app.settings.build_mesh) {
            std::snprintf(buffer, sizeof(buffer), "%.4f", stats.depth_loss);
            theme::metric("Depth", buffer);
            std::snprintf(buffer, sizeof(buffer), "%.4f", stats.normal_loss);
            theme::metric("Normal", buffer);
            std::snprintf(
                buffer, sizeof(buffer), "%.4f",
                stats.multi_view_geometry_loss);
            theme::metric("Multi-view geo", buffer);
            std::snprintf(
                buffer, sizeof(buffer), "%.4f", stats.multi_view_ncc_loss);
            theme::metric("Multi-view NCC", buffer);
        }
        std::snprintf(buffer, sizeof(buffer), "%.1f ms", stats.step_milliseconds);
        theme::metric("Step time", buffer);
        std::snprintf(
            buffer, sizeof(buffer), "%.2fx", stats.resolution_scale);
        theme::metric("Resolution scale", buffer);
        std::snprintf(
            buffer, sizeof(buffer), "%llu",
            static_cast<unsigned long long>(gpu::consumed_timeline_value()));
        theme::metric("Preview timeline", buffer);
        ImGui::Spacing();
    }

    // Context-appropriate primary action pinned to the bottom of the panel.
    ImGui::Dummy({0, 10.F});
    if (busy) {
        if (theme::danger_button(
                stop_job_label(app.active_job), {-1.F, 40.F}))
            action = Action::stop;
    } else if (!app.has_sparse) {
        if (theme::primary_button(
                "Align Photos", {-1.F, 40.F},
                app.settings.images_dir[0] != '\0'))
            action = Action::align;
    } else {
        if (theme::primary_button(
                app.settings.build_mesh ? "Train 3DGS + Mesh" : "Train 3DGS",
                {-1.F, 40.F}, app.settings.images_dir[0] != '\0'))
            action = Action::train;
    }

    ImGui::End();
    return action;
}

float status_segment_width(const char* text) {
    return 15.F + 6.F + ImGui::CalcTextSize(text).x;
}

void draw_status_segment(
    const float x, const float y, const icons::Icon icon, const char* text,
    const ImVec4 colour = theme::text_muted) {
    ImGui::SetCursorPos({x, y});
    icons::inline_icon(icon, theme::u32(colour), 15.F);
    ImGui::SameLine(0.F, 6.F);
    ImGui::SetCursorPosY(y);
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

void draw_status_separator(const float x, const float height) {
    ImGui::GetWindowDrawList()->AddLine(
        {ImGui::GetWindowPos().x + x, ImGui::GetWindowPos().y + 8.F},
        {ImGui::GetWindowPos().x + x, ImGui::GetWindowPos().y + height - 8.F},
        theme::u32(theme::border, 0.8F));
}

void draw_status_bar(const App& app) {
    if (!app.show_status_bar) return;

    const bool busy = app.job.running();
    const ImVec4 background = busy
        ? ImVec4(0.027F, 0.208F, 0.325F, 1.F)
        : ImVec4(0.086F, 0.090F, 0.102F, 1.F);
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(
        viewport->WorkPos.x,
        viewport->WorkPos.y + viewport->WorkSize.y - k_status_height));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, k_status_height));
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.F, 0.F));
    ImGui::PushStyleColor(ImGuiCol_WindowBg, background);
    constexpr ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoBringToFrontOnFocus;
    ImGui::Begin("##StatusBar", nullptr, flags);
    ImGui::PopStyleVar(3);

    const float height = k_status_height;
    const float centre_y = (height - ImGui::GetTextLineHeight()) * 0.5F;
    ImGui::SetCursorPos({12.F, centre_y});
    ImVec4 state_colour = theme::inactive;
    icons::Icon state_icon = icons::Icon::cube;
    if (busy) {
        state_colour = theme::accent;
        state_icon = app.active_job == JobKind::train ? icons::Icon::train
                                                      : icons::Icon::align;
    } else if (app.monitor.stage() == Stage::failed) {
        state_colour = theme::danger;
        state_icon = icons::Icon::stop;
    } else if (app.monitor.stage() == Stage::complete) {
        state_colour = theme::success;
        state_icon = app.scene.has_points() ? icons::Icon::points
                                            : icons::Icon::cube;
    } else if (app.scene.has_points()) {
        state_icon = icons::Icon::points;
    }
    icons::inline_icon(state_icon, theme::u32(state_colour), 16.F);

    ImGui::SameLine(0.F, 8.F);
    ImGui::SetCursorPosY(centre_y);
    if (busy) {
        const std::string headline = app.monitor.headline();
        ImGui::TextUnformatted(headline.c_str());
    } else if (!app.message.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, app.message_colour);
        ImGui::TextUnformatted(app.message.c_str());
        ImGui::PopStyleColor();
    } else {
        ImGui::TextUnformatted("Ready");
    }

    // Persistent right-hand telemetry: backend and build tag always remain
    // visible; idle mode also shows viewport, point and active-tool state.
    const char* tag = "AetherScan 0.2";
    const float tag_width = ImGui::CalcTextSize(tag).x;
    float right = ImGui::GetWindowWidth() - tag_width - 14.F;
    ImGui::SetCursorPos({right, centre_y});
    ImGui::PushStyleColor(
        ImGuiCol_Text, busy ? theme::text_bright : theme::text_faint);
    ImGui::TextUnformatted(tag);
    ImGui::PopStyleColor();

    const char* backend = "CUDA / Vulkan";
    const float backend_width = status_segment_width(backend);
    right -= backend_width + 22.F;
    draw_status_separator(right + backend_width + 11.F, height);
    draw_status_segment(
        right, centre_y, icons::Icon::gpu, backend,
        busy ? theme::text_bright : theme::text_muted);

    if (busy) {
        const float fraction = app.monitor.fraction();
        const double eta = app.monitor.eta_seconds();
        char trailing[96];
        if (fraction >= 0.F && eta >= 0.0)
            std::snprintf(
                trailing, sizeof(trailing), "%3.0f%%   ETA %s", fraction * 100.F,
                format_duration(eta).c_str());
        else if (fraction >= 0.F)
            std::snprintf(trailing, sizeof(trailing), "%3.0f%%", fraction * 100.F);
        else
            std::snprintf(trailing, sizeof(trailing), "working");

        const float trailing_width = ImGui::CalcTextSize(trailing).x;
        constexpr float bar_width = 220.F;
        const float bar_x = std::max(
            240.F, right - trailing_width - 14.F - bar_width);
        ImGui::SameLine(bar_x);
        ImGui::SetCursorPosY((height - 7.F) * 0.5F);
        theme::progress_track(
            {bar_width, 7.F}, fraction,
            fraction < 0.F ? theme::accent : ImVec4(0.55F, 0.84F, 1.F, 1.F));
        ImGui::SameLine(0.F, 12.F);
        ImGui::SetCursorPosY(centre_y);
        ImGui::TextUnformatted(trailing);
    } else {
        if (app.scene.has_points()) {
            const std::string points =
                format_count(app.scene.points.size()) + " pts";
            const float points_width = status_segment_width(points.c_str());
            right -= points_width + 22.F;
            draw_status_separator(right + points_width + 11.F, height);
            draw_status_segment(
                right, centre_y, icons::Icon::points, points.c_str());
        }

        const char* view = app.tab == ViewportTab::sparse ? "Sparse" : "Training";
        const float view_width = status_segment_width(view);
        right -= view_width + 22.F;
        if (right > 520.F) {
            draw_status_separator(right + view_width + 11.F, height);
            draw_status_segment(
                right, centre_y,
                app.tab == ViewportTab::sparse ? icons::Icon::cube
                                               : icons::Icon::train,
                view);
        }
    }

    ImGui::End();
    ImGui::PopStyleColor();
}

}  // namespace

int main(const int argc, char** argv) {
    App app;
    app.smoke_mode = argc > 1 && std::string_view(argv[1]) == "--interop-smoke";

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
        1600, 940, "AetherScan Reconstruction Editor", nullptr, nullptr);

    ImVector<const char*> extensions;
    std::uint32_t extension_count{};
    const char** required = glfwGetRequiredInstanceExtensions(&extension_count);
    for (std::uint32_t i = 0; i < extension_count; ++i)
        extensions.push_back(required[i]);
    gpu::create_context(extensions);
    VkSurfaceKHR surface{};
    gpu::check(
        glfwCreateWindowSurface(gpu::instance(), window, nullptr, &surface));
    int width{};
    int height{};
    glfwGetFramebufferSize(window, &width, &height);
    gpu::create_window(surface, width, height);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigDockingWithShift = false;
    if (!app.smoke_mode) {
        const std::filesystem::path ini = resolve_editor_ini();
        std::error_code error;
        std::filesystem::create_directories(ini.parent_path(), error);
        g_editor_ini = ini.string();
        io.IniFilename = g_editor_ini.c_str();
        if (!std::filesystem::exists(ini, error)) app.reset_dock_layout = true;
    }
    const theme::Fonts fonts = theme::load_fonts(io);
    io.FontDefault = fonts.regular;
    theme::apply_style();

    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init{};
    init.Instance = gpu::instance();
    init.PhysicalDevice = gpu::physical_device();
    init.Device = gpu::device();
    init.QueueFamily = gpu::queue_family();
    init.Queue = gpu::queue();
    init.DescriptorPool = gpu::descriptor_pool();
    init.RenderPass = gpu::window().RenderPass;
    init.MinImageCount = gpu::k_min_images;
    init.ImageCount = gpu::window().ImageCount;
    init.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init.CheckVkResultFn = gpu::check;
    ImGui_ImplVulkan_Init(&init);

    refresh_artifacts(app);
    app.preview.create(k_preview_extent, k_preview_extent);

    const auto smoke_begin = std::chrono::steady_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        app.job.poll();
        app.preview.poll();
        app.log.poll(app.fresh_lines);
        for (const std::string& line : app.fresh_lines)
            app.monitor.consume(line);
        if (app.job.consume_completion()) on_job_finished(app);
        poll_scene_load(app);

        if (app.smoke_mode) {
            if (!app.smoke_started && !app.job.running()) {
                app.smoke_started = true;
                start_train(app, true);
                glfwIconifyWindow(window);
            } else if (
                app.smoke_started && !app.job.running() &&
                gpu::consumed_timeline_value() >= 3) {
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
            (gpu::swapchain_needs_rebuild() || gpu::window().Width != width ||
             gpu::window().Height != height))
            gpu::resize_window(width, height);
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED)) {
            app.preview.consume_without_present();
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        app.settings.iterations = std::max(app.settings.iterations, 1);
        app.settings.preview_interval =
            std::max(app.settings.preview_interval, 1);
        app.settings.max_features = std::max(app.settings.max_features, 512);
        app.settings.geometry_from_iter =
            std::max(app.settings.geometry_from_iter, 0);

        Action action = draw_menu_bar(app);
        const Action toolbar_action = draw_toolbar(app);
        if (action == Action::none) action = toolbar_action;

        build_dock_space(app);
        draw_scene_panel(app);
        draw_viewport_panel(app);
        draw_console_panel(app);
        const Action inspector_action = draw_inspector(app);
        if (action == Action::none) action = inspector_action;
        draw_status_bar(app);

        switch (action) {
            case Action::align:
                if (!app.smoke_mode) start_align(app);
                break;
            case Action::train:
                if (!app.smoke_mode) start_train(app, false);
                break;
            case Action::export_sfm:
                if (!app.smoke_mode) start_export_sfm(app);
                break;
            case Action::stop:
                app.job.stop();
                set_message(app, "Stopping...", theme::warning);
                break;
            case Action::reveal: reveal_in_explorer(app.layout.root); break;
            case Action::none: break;
        }

        draw_controls_window(app);
        draw_clear_results_modal(app);
        if (app.close_requested) glfwSetWindowShouldClose(window, GLFW_TRUE);
        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (draw_data->DisplaySize.x > 0.F && draw_data->DisplaySize.y > 0.F)
            gpu::present(draw_data, theme::surface_0);
    }

    if (app.job.running()) app.job.stop();
    vkDeviceWaitIdle(gpu::device());
    app.preview.reset();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    gpu::destroy_context();
    glfwDestroyWindow(window);
    glfwTerminate();
    return app.smoke_mode && !app.smoke_success ? 4 : 0;
}
