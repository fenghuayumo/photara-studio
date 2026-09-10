#include "pipeline.hpp"
#include "console_view.hpp"
#include "icons.hpp"
#include "image_qa_view.hpp"
#include "sparse_view.hpp"
#include "theme.hpp"
#include "viewport_gizmo.hpp"
#include "vulkan_backend.hpp"

#include "io/image.hpp"
#include "io/video_frames.hpp"
#include "project/archive.hpp"
#include "project/document.hpp"
#include "sfm/asfm.hpp"
#include "sfm/export_mvs.hpp"
#include "splat/trainer.hpp"
#include "splat/visualize.hpp"
#if defined(AETHERSCAN_HAS_TEXTURE)
#include "texture/mesh_preview.hpp"
#endif

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
#include <memory>
#include <optional>
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

enum class VisualizationMode { points, splat, rings, mesh };

enum class ViewportWorkspace { scene_3d, image_2d };

enum class StepState { pending, active, done, skipped, failed };

enum class ClearResultsAction { none, clear_view, delete_generated };

struct App {
    ProjectSettings settings;
    ProjectLayout layout;

    ProcessJob job;
    ProcessJob viewer;
    JobKind active_job{JobKind::none};
    RunMonitor monitor;
    LogStream log;
    ConsoleView console;
    std::vector<std::string> fresh_lines;

    gpu::ExternalPreview preview;
    gpu::CameraPhotoCache photos;

    SparseScene scene;
    PreviewMesh mesh;
    std::future<MeshLoad> pending_mesh_load;
    bool frame_mesh_on_load{};
    bool mesh_load_failed{};
#if defined(AETHERSCAN_HAS_TEXTURE)
    std::unique_ptr<aetherscan::texture::MeshPreviewRasterizer>
        mesh_rasterizer;
    gpu::PreviewTexture mesh_preview;
    std::vector<float> mesh_gpu_positions;
    std::vector<float> mesh_gpu_normals;
    std::vector<float> mesh_gpu_colours;
    std::vector<std::uint32_t> mesh_gpu_indices;
    SplatPreviewCamera mesh_gpu_camera{};
    bool mesh_gpu_wireframe{};
    bool mesh_gpu_vertex_colour{};
    bool mesh_gpu_failed{};
#endif
    OrbitCamera camera;
    ViewOptions view_options;
    ViewportGizmoState gizmo;
    SceneRenderer renderer;
    std::future<SceneLoad> pending_load;
    std::future<SceneLoad> alignment_preview_load;
    std::filesystem::file_time_type alignment_preview_stamp{};
    std::chrono::steady_clock::time_point alignment_preview_poll{};
    bool alignment_preview_seen{};
    unsigned alignment_preview_generation{};
    unsigned alignment_preview_load_generation{};
    // Bumped to drop an in-flight scene load (e.g. Re-align must not restore
    // the previous cloud after the viewport has already been cleared).
    unsigned scene_load_generation{};
    unsigned pending_scene_load_generation{};
    bool loading_scene{};
    std::string scene_source;
    // Signature of the external dataset alignment already loaded (or whose
    // load failed) so the viewport does not retry an unchanged bad path.
    std::string dataset_scene_key;

    bool has_sparse{};
    bool has_asfm{};
    bool has_mvs{};
    bool has_model{};
    bool has_mesh{};
    std::uint32_t project_writer_version{};
    std::uint32_t project_min_reader_version{};
    VisualizationMode view_mode{VisualizationMode::points};
    ViewportWorkspace workspace{ViewportWorkspace::scene_3d};
    ImageQaState image_qa;
    ImageQaSession image_qa_session;
    unsigned qa_preview_view{~0U};
    std::chrono::steady_clock::time_point qa_metrics_after{};
    // Timeline value already consumed when the QA capture camera was written.
    // Compare frames are only trusted once a newer preview arrives, so an
    // orbit-camera frame can never be wiped against a training photo.
    std::uint64_t qa_camera_timeline{};
    bool qa_camera_valid{};
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
    // A user-cleared viewport must stay empty until an explicit load or a new
    // reconstruction completes. Otherwise draw_sparse_tab() reloads it on the
    // very next frame merely because an artifact still exists on disk.
    bool suppress_scene_auto_load{};
    // Re-align draws camera photos into this frame's list. Destroy those
    // descriptors only after present() has submitted them.
    bool pending_align_viewport_clear{};

    bool show_scene{true};
    bool show_viewport{true};
    bool show_console{true};
    bool show_inspector{true};
    bool show_status_bar{true};
    bool reset_dock_layout{};

    // OS file drops arrive on the GLFW callback and are consumed against the
    // last viewport rectangle on the following frame.
    std::vector<std::string> dropped_paths;
    ImVec2 viewport_min{};
    ImVec2 viewport_max{};
    bool viewport_bounds_valid{};
};

void stop_splat_view(App& app) {
    if (app.viewer.running()) app.viewer.stop();
}

void start_splat_view(App& app);
void sync_live_preview_camera(
    App& app, bool force, std::uint32_t width, std::uint32_t height);
void set_viewport_workspace(App& app, ViewportWorkspace workspace);

bool has_external_dataset(const App& app) {
    return app.settings.dataset_source[0] != '\0';
}

bool alignment_ready(const App& app) {
    return has_external_dataset(app) || app.has_sparse || app.scene.has_points();
}

bool waiting_for_train_preview(const App& app) {
    return app.job.running() && app.active_job == JobKind::train &&
           gpu::consumed_timeline_value() == 0;
}

std::filesystem::path existing_splat_model(const App& app) {
    const std::filesystem::path imported(app.settings.splat_model_source.data());
    std::error_code error;
    if (!imported.empty() && std::filesystem::exists(imported, error))
        return imported;
    const std::array<std::filesystem::path, 5> candidates = {
        app.layout.splat_model, app.layout.splat_ply, app.layout.splat_sog,
        app.layout.splat_spz, app.layout.splat_glb};
    for (const auto& candidate : candidates)
        if (std::filesystem::exists(candidate, error)) return candidate;
    return {};
}

bool live_preview_active(const App& app) {
    return (app.job.running() && app.active_job == JobKind::train) ||
           app.viewer.running();
}

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

std::string lower_path_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](const unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    return extension;
}

bool is_supported_image_extension(std::string extension) {
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](const unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    return extension == ".jpg" || extension == ".jpeg" ||
           extension == ".png" || extension == ".tif" ||
           extension == ".tiff" || extension == ".bmp";
}

bool directory_has_images(const std::filesystem::path& directory) {
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) return false;
    for (const auto& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (!entry.is_regular_file(error)) continue;
        if (is_supported_image_extension(entry.path().extension().string()))
            return true;
    }
    return false;
}

std::filesystem::path path_from_drop(const std::string& text) {
#if defined(_WIN32)
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
#else
    return std::filesystem::path(text);
#endif
}

std::filesystem::path normalized_directory(const std::filesystem::path& path) {
    std::error_code error;
    auto canonical = std::filesystem::weakly_canonical(path, error);
    if (error) canonical = path.lexically_normal();
    return canonical;
}

bool same_directory(
    const std::filesystem::path& left, const std::filesystem::path& right) {
    std::error_code error;
    if (std::filesystem::equivalent(left, right, error)) return true;
    return normalized_directory(left) == normalized_directory(right);
}

bool is_supported_image_file(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) &&
           is_supported_image_extension(path.extension().string());
}

std::optional<std::filesystem::path> resolve_dropped_image_directory(
    const std::vector<std::string>& dropped, std::string& error) {
    std::vector<std::filesystem::path> folders;
    std::vector<std::filesystem::path> image_folders;
    std::vector<std::filesystem::path> images;
    folders.reserve(dropped.size());
    images.reserve(dropped.size());
    for (const std::string& text : dropped) {
        if (text.empty()) continue;
        const std::filesystem::path path = path_from_drop(text);
        std::error_code status;
        if (std::filesystem::is_directory(path, status)) {
            const auto folder = normalized_directory(path);
            folders.push_back(folder);
            if (directory_has_images(folder)) image_folders.push_back(folder);
        } else if (is_supported_image_file(path)) {
            images.push_back(path);
        }
    }

    if (image_folders.size() > 1) {
        error = "Drop a single image folder";
        return std::nullopt;
    }
    if (image_folders.size() == 1) {
        const auto& folder = image_folders.front();
        for (const auto& image : images) {
            if (!same_directory(image.parent_path(), folder)) {
                error = "Dropped items must belong to one image folder";
                return std::nullopt;
            }
        }
        return folder;
    }
    if (!folders.empty() && images.empty()) {
        error = "Dropped folder contains no supported images";
        return std::nullopt;
    }
    if (images.empty()) {
        error = "Drop an image folder, photos, a video, .asfm, or .ascan project";
        return std::nullopt;
    }

    const auto parent = normalized_directory(images.front().parent_path());
    for (const auto& image : images) {
        if (!same_directory(image.parent_path(), parent)) {
            error = "Dropped photos must come from the same folder";
            return std::nullopt;
        }
    }
    return parent;
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

bool copy_wide_path(const wchar_t* wide, std::array<char, 1024>& destination) {
    const int bytes = WideCharToMultiByte(
        CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (bytes <= 0 || static_cast<std::size_t>(bytes) > destination.size())
        return false;
    WideCharToMultiByte(
        CP_UTF8, 0, wide, -1, destination.data(), bytes, nullptr, nullptr);
    return true;
}

enum class FilePickKind { project, dataset, point_cloud, splat_model, video };

bool pick_file(
    const wchar_t* title, std::array<char, 1024>& destination,
    const bool save, const FilePickKind kind) {
    bool picked = false;
    const HRESULT initialised =
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    IFileDialog* dialog = nullptr;
    const HRESULT created = save
        ? CoCreateInstance(
              CLSID_FileSaveDialog, nullptr, CLSCTX_INPROC_SERVER,
              IID_PPV_ARGS(&dialog))
        : CoCreateInstance(
              CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
              IID_PPV_ARGS(&dialog));
    if (SUCCEEDED(created) && dialog) {
        DWORD options = 0;
        dialog->GetOptions(&options);
        if (save)
            dialog->SetOptions(options | FOS_OVERWRITEPROMPT);
        else
            dialog->SetOptions(options | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST);
        dialog->SetTitle(title);
        COMDLG_FILTERSPEC project_filters[] = {
            {L"AetherScan Project (*.ascan)", L"*.ascan"},
            {L"All files (*.*)", L"*.*"}};
        COMDLG_FILTERSPEC dataset_filters[] = {
            {L"Camera datasets (*.csv;*.mvs)", L"*.csv;*.mvs"},
            {L"All files (*.*)", L"*.*"}};
        COMDLG_FILTERSPEC point_cloud_filters[] = {
            {L"Point cloud (*.ply)", L"*.ply"},
            {L"All files (*.*)", L"*.*"}};
        COMDLG_FILTERSPEC splat_model_filters[] = {
            {L"Gaussian splat (*.ply;*.sog;*.spz;*.glb)", L"*.ply;*.sog;*.spz;*.glb"},
            {L"All files (*.*)", L"*.*"}};
        COMDLG_FILTERSPEC video_filters[] = {
            {L"Video files (*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.insv;*.wmv)",
             L"*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.insv;*.wmv;*.mts;*.m2ts;*.360"},
            {L"All files (*.*)", L"*.*"}};
        const COMDLG_FILTERSPEC* filters = project_filters;
        if (kind == FilePickKind::dataset) filters = dataset_filters;
        if (kind == FilePickKind::point_cloud) filters = point_cloud_filters;
        if (kind == FilePickKind::splat_model) filters = splat_model_filters;
        if (kind == FilePickKind::video) filters = video_filters;
        dialog->SetFileTypes(2, filters);
        if (kind == FilePickKind::project && save)
            dialog->SetDefaultExtension(L"ascan");
        if (SUCCEEDED(dialog->Show(nullptr))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dialog->GetResult(&item))) {
                PWSTR wide = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &wide))) {
                    picked = copy_wide_path(wide, destination);
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

bool pick_project_file(
    const wchar_t* title, std::array<char, 1024>& destination, const bool save) {
    return pick_file(title, destination, save, FilePickKind::project);
}

bool pick_dataset_file(
    const wchar_t* title, std::array<char, 1024>& destination) {
    return pick_file(title, destination, false, FilePickKind::dataset);
}

bool pick_point_cloud_file(
    const wchar_t* title, std::array<char, 1024>& destination) {
    return pick_file(title, destination, false, FilePickKind::point_cloud);
}
bool pick_splat_model_file(
    const wchar_t* title, std::array<char, 1024>& destination) {
    return pick_file(title, destination, false, FilePickKind::splat_model);
}
bool pick_video_file(
    const wchar_t* title, std::array<char, 1024>& destination) {
    return pick_file(title, destination, false, FilePickKind::video);
}

void reveal_in_explorer(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::exists(path, error)) return;
    ShellExecuteW(nullptr, L"open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}
#else
bool pick_folder(const wchar_t*, std::array<char, 1024>&) { return false; }
bool pick_project_file(const wchar_t*, std::array<char, 1024>&, bool) {
    return false;
}
bool pick_dataset_file(const wchar_t*, std::array<char, 1024>&) {
    return false;
}
bool pick_point_cloud_file(const wchar_t*, std::array<char, 1024>&) {
    return false;
}
bool pick_splat_model_file(const wchar_t*, std::array<char, 1024>&) {
    return false;
}
bool pick_video_file(const wchar_t*, std::array<char, 1024>&) {
    return false;
}
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
        app.has_asfm = false;
        app.has_mvs = false;
        app.has_model = false;
        app.has_mesh = false;
        app.project_writer_version = 0;
        app.project_min_reader_version = 0;
        return;
    }
    app.layout = resolve_layout(app.settings);
    std::error_code error;
    app.has_asfm = std::filesystem::exists(app.layout.sparse_asfm, error);
    app.has_mvs = std::filesystem::exists(app.layout.sparse_mvs, error);
    app.has_sparse =
        std::filesystem::exists(app.layout.sparse_ply, error) ||
        std::filesystem::exists(app.layout.working_sfm, error);
    app.has_model = !existing_splat_model(app).empty();
    app.has_mesh =
        std::filesystem::exists(app.layout.mesh_ply, error) ||
        std::filesystem::exists(app.layout.mvs_mesh_ply, error) ||
        std::filesystem::exists(app.layout.mvs_raw_mesh_ply, error);
    app.project_writer_version = 0;
    app.project_min_reader_version = 0;
    if (app.layout.project_file.empty()) return;
    try {
        if (!std::filesystem::exists(app.layout.project_file, error)) return;
        const auto archive =
            aetherscan::project::Archive::open(app.layout.project_file);
        app.project_writer_version = archive.writer_version();
        app.project_min_reader_version = archive.min_reader_version();
        app.has_sparse =
            archive.has(aetherscan::project::ChunkType::sfm) || app.has_sparse;
        app.has_model =
            archive.has(aetherscan::project::ChunkType::gaussians) ||
            app.has_model;
        app.has_mesh =
            archive.has(aetherscan::project::ChunkType::mesh) || app.has_mesh;
    } catch (...) {
    }
}

std::filesystem::path reconstruction_images_path(const App& app) {
    return reconstruction_images_dir(app.settings);
}

void store_utf8_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path);

void assign_default_project_folder(App& app) {
    if (app.settings.images_dir[0] == '\0' ||
        app.settings.project_dir[0] != '\0')
        return;

    const std::filesystem::path images =
        path_from_utf8_field(app.settings.images_dir.data());
    const std::filesystem::path parent = images.parent_path();
    std::filesystem::path project_name = is_video_source(app.settings)
        ? images.stem()
        : images.filename();
    project_name += ".ascan";
    const std::filesystem::path project =
        parent.empty() ? project_name : parent / project_name;
    store_utf8_path_field(app.settings.project_dir, project);
    app.project_folder_automatic = true;
}

void store_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path) {
    const std::string text = path_to_utf8(path);
    if (text.size() + 1 > field.size()) return;
    std::memcpy(field.data(), text.data(), text.size());
    field[text.size()] = '\0';
}

void store_utf8_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path) {
    store_path_field(field, path);
}

aetherscan::sfm::Scene load_working_sfm(
    const std::filesystem::path& path,
    const std::filesystem::path& images_dir) {
    aetherscan::sfm::AsfmOptions options;
    options.path_base = images_dir;
    return aetherscan::sfm::load_asfm(path, options);
}

aetherscan::project::Settings collect_project_settings(const App& app) {
    aetherscan::project::Settings settings;
    settings.name = app.layout.project_file.empty()
        ? std::string("Untitled")
        : path_to_utf8(app.layout.project_file.stem());
    settings.image_directory =
        path_from_utf8_field(app.settings.images_dir.data());
    settings.dataset_source =
        path_from_utf8_field(app.settings.dataset_source.data());
    settings.dataset_format = app.settings.dataset_format == 1
        ? "colmap"
        : app.settings.dataset_format == 2
            ? "realitycapture"
            : app.settings.dataset_format == 3 ? "openmvs" : "auto";
    settings.dataset_initial_cloud =
        path_from_utf8_field(app.settings.dataset_initial_cloud.data());
    settings.splat_model_source =
        path_from_utf8_field(app.settings.splat_model_source.data());
    settings.video_frames_dir =
        path_from_utf8_field(app.settings.video_frames_dir.data());
    settings.video_fps = app.settings.video_fps;
    settings.video_sharp_window = app.settings.video_sharp_window;
    settings.video_max_frames = app.settings.video_max_frames;
    settings.video_quality = app.settings.video_quality;
    settings.video_scale = app.settings.video_scale;
    settings.video_rotate = app.settings.video_rotate;
    settings.splat_output_format = app.settings.splat_format == 1
        ? "ply"
        : app.settings.splat_format == 2
            ? "sog"
            : app.settings.splat_format == 3 ? "spz"
            : app.settings.splat_format == 4 ? "glb" : "auto";
    settings.camera_model = app.settings.camera_model;
    settings.sfm_mode = app.settings.sfm_mode;
    settings.reuse_cache = app.settings.reuse_cache;
    settings.max_features = static_cast<unsigned>(
        std::max(0, app.settings.max_features));
    settings.scene_mode = app.settings.scene_mode;
    settings.iterations = app.settings.iterations;
    settings.preview_interval = app.settings.preview_interval;
    settings.strategy = app.settings.strategy;
    settings.max_resolution = app.settings.max_resolution;
    settings.progressive_resolution = app.settings.progressive_resolution;
    settings.use_mask = app.settings.use_mask;
    settings.build_mesh = app.settings.build_mesh;
    settings.mesh_source = app.settings.mesh_source;
    settings.mesh_method = app.settings.mesh_method;
    settings.depth_normal_weight = app.settings.depth_normal_weight;
    settings.multi_view_geo_weight = app.settings.multi_view_geo_weight;
    settings.multi_view_ncc_weight = app.settings.multi_view_ncc_weight;
    settings.geometry_from_iter = app.settings.geometry_from_iter;
    settings.normal_field = app.settings.normal_field;
    return settings;
}

void apply_project_settings(
    App& app, const aetherscan::project::Settings& settings) {
    if (!settings.image_directory.empty())
        store_path_field(app.settings.images_dir, settings.image_directory);
    store_path_field(app.settings.dataset_source, settings.dataset_source);
    if (settings.dataset_format == "colmap")
        app.settings.dataset_format = 1;
    else if (settings.dataset_format == "realitycapture")
        app.settings.dataset_format = 2;
    else if (settings.dataset_format == "openmvs")
        app.settings.dataset_format = 3;
    else
        app.settings.dataset_format = 0;
    store_path_field(
        app.settings.dataset_initial_cloud, settings.dataset_initial_cloud);
    store_path_field(app.settings.splat_model_source, settings.splat_model_source);
    store_path_field(app.settings.video_frames_dir, settings.video_frames_dir);
    app.settings.video_fps = settings.video_fps;
    app.settings.video_sharp_window = settings.video_sharp_window;
    app.settings.video_max_frames = settings.video_max_frames;
    app.settings.video_quality = settings.video_quality;
    app.settings.video_scale = settings.video_scale;
    app.settings.video_rotate = settings.video_rotate;
    if (settings.splat_output_format == "ply")
        app.settings.splat_format = 1;
    else if (settings.splat_output_format == "sog")
        app.settings.splat_format = 2;
    else if (settings.splat_output_format == "spz")
        app.settings.splat_format = 3;
    else if (settings.splat_output_format == "glb")
        app.settings.splat_format = 4;
    else
        app.settings.splat_format = 0;
    app.settings.camera_model = settings.camera_model;
    app.settings.sfm_mode = settings.sfm_mode;
    app.settings.reuse_cache = settings.reuse_cache;
    app.settings.max_features = static_cast<int>(settings.max_features);
    app.settings.scene_mode = settings.scene_mode;
    app.settings.iterations = settings.iterations;
    app.settings.preview_interval = settings.preview_interval;
    app.settings.strategy = settings.strategy;
    app.settings.max_resolution = settings.max_resolution;
    app.settings.progressive_resolution = settings.progressive_resolution;
    app.settings.use_mask = settings.use_mask;
    app.settings.build_mesh = settings.build_mesh;
    app.settings.mesh_source = settings.mesh_source == 1 ? 1 : 0;
    app.settings.mesh_method = settings.mesh_method;
    app.settings.depth_normal_weight = settings.depth_normal_weight;
    app.settings.multi_view_geo_weight = settings.multi_view_geo_weight;
    app.settings.multi_view_ncc_weight = settings.multi_view_ncc_weight;
    app.settings.geometry_from_iter = settings.geometry_from_iter;
    app.settings.normal_field = settings.normal_field;
}

void request_asfm_scene_load(
    App& app, const std::filesystem::path& asfm, std::string label) {
    if (app.loading_scene || asfm.empty()) return;
    app.suppress_scene_auto_load = false;
    const std::filesystem::path images = reconstruction_images_path(app);
    app.pending_scene_load_generation = app.scene_load_generation;
    app.loading_scene = true;
    app.scene_source = std::move(label);
    app.pending_load = std::async(
        std::launch::async, [asfm, images] {
            SceneLoad loaded;
            try {
                return sparse_scene_from_sfm(load_working_sfm(asfm, images));
            } catch (const std::exception& failure) {
                loaded.error = failure.what();
                return loaded;
            }
        });
}

std::string external_dataset_signature(const App& app) {
    std::string key(app.settings.dataset_source.data());
    key += '|';
    key += std::to_string(app.settings.dataset_format);
    key += '|';
    key += app.settings.dataset_initial_cloud.data();
    key += '|';
    key += app.settings.images_dir.data();
    return key;
}

std::string external_dataset_format(const App& app) {
    switch (app.settings.dataset_format) {
        case 1: return "colmap";
        case 2: return "realitycapture";
        case 3: return "openmvs";
        default: return "auto";
    }
}

// Loads the imported camera alignment into the editor scene. Without those
// view poses the 2D QA workspace has no capture cameras to snap its compare
// render or feature overlay to, and the trainer keeps previewing from the
// orbit camera instead of the photo being inspected.
void request_dataset_scene_load(App& app) {
    if (app.loading_scene || !has_external_dataset(app)) return;
    const std::filesystem::path source(app.settings.dataset_source.data());
    const std::string format = external_dataset_format(app);
    const std::filesystem::path initial_cloud(
        app.settings.dataset_initial_cloud.data());
    const std::filesystem::path images = reconstruction_images_path(app);
    app.dataset_scene_key = external_dataset_signature(app);
    app.suppress_scene_auto_load = false;
    app.pending_scene_load_generation = app.scene_load_generation;
    app.loading_scene = true;
    app.scene_source = "External dataset";
    app.pending_load = std::async(
        std::launch::async, [source, format, initial_cloud, images] {
            return sparse_scene_from_dataset(
                source, format, initial_cloud, images);
        });
}

std::filesystem::path inferred_images_dir_near(const std::filesystem::path& asfm);
void clear_loaded_result(App& app);

void apply_external_dataset_selection(App& app) {
    clear_loaded_result(app);
    refresh_artifacts(app);
    if (!has_external_dataset(app)) return;
    const std::filesystem::path source(app.settings.dataset_source.data());
    if (app.settings.images_dir[0] == '\0') {
        auto inferred = inferred_images_dir_near(source);
        if (inferred.empty()) {
            std::error_code error;
            if (std::filesystem::is_directory(source, error)) {
                if (directory_has_images(source)) inferred = source;
                else if (directory_has_images(source / "images"))
                    inferred = source / "images";
            }
        }
        if (!inferred.empty())
            store_utf8_path_field(app.settings.images_dir, inferred);
    }
    assign_default_project_folder(app);
    request_dataset_scene_load(app);
    set_message(
        app,
        "External cameras loaded. Align Photos is skipped; Train 3DGS or "
        "Dense MVS can run next.",
        theme::accent);
}

void ensure_dataset_scene_loaded(App& app) {
    if (app.suppress_scene_auto_load || !has_external_dataset(app) ||
        app.loading_scene)
        return;
    // A Gaussian-centre scene merges the imported views when no internal
    // poses CSV exists, so it also counts as showing the dataset cameras.
    const bool imported_views_visible =
        !app.scene.views.empty() &&
        (app.scene_source == "External dataset" ||
         app.scene_source == "Gaussian centres");
    if (imported_views_visible) return;
    if (external_dataset_signature(app) == app.dataset_scene_key) return;
    request_dataset_scene_load(app);
}

void request_ascan_scene_load(App& app) {
    if (app.loading_scene) return;
    const auto ascan = app.layout.project_file;
    const auto working = app.layout.working_sfm;
    const auto asfm = app.layout.sparse_asfm;
    const std::filesystem::path images = reconstruction_images_path(app);
    if (ascan.empty() && working.empty() && asfm.empty()) return;
    app.suppress_scene_auto_load = false;
    app.pending_scene_load_generation = app.scene_load_generation;
    app.loading_scene = true;
    app.scene_source = "Project SfM";
    app.pending_load = std::async(
        std::launch::async, [ascan, working, asfm, images] {
            SceneLoad loaded;
            std::error_code error;
            try {
                // Live Align writes the cache working copy, not .ascan.
                // Relative photo paths in that file are stored against the
                // image folder and must be resolved with the same base.
                if (!working.empty() &&
                    std::filesystem::exists(working, error)) {
                    return sparse_scene_from_sfm(
                        load_working_sfm(working, images));
                }
                if (!ascan.empty() && std::filesystem::exists(ascan, error)) {
                    const auto archive =
                        aetherscan::project::Archive::open(ascan);
                    const auto scene = aetherscan::project::read_sfm(archive);
                    if (scene) return sparse_scene_from_sfm(*scene);
                }
                if (!asfm.empty() && std::filesystem::exists(asfm, error))
                    return sparse_scene_from_sfm(load_working_sfm(asfm, images));
                loaded.error = "Project has no SfM stage yet";
            } catch (const std::exception& failure) {
                loaded.error = failure.what();
            }
            return loaded;
        });
}

bool save_project_to_path(App& app, const std::filesystem::path& path) {
    try {
        std::error_code error;
        aetherscan::project::Archive archive =
            aetherscan::project::Archive::create();
        const auto current = app.layout.project_file;
        if (!current.empty() && std::filesystem::exists(current, error))
            archive = aetherscan::project::Archive::open(current);
        else if (std::filesystem::exists(path, error))
            archive = aetherscan::project::Archive::open(path);
        aetherscan::project::write_settings(
            archive, collect_project_settings(app), path);
        std::error_code working_error;
        if (!app.layout.working_sfm.empty() &&
            std::filesystem::exists(app.layout.working_sfm, working_error)) {
            const auto scene = load_working_sfm(
                app.layout.working_sfm, reconstruction_images_path(app));
            aetherscan::project::write_sfm(archive, scene, path);
        }
        archive.save(path);
        store_path_field(app.settings.project_dir, path);
        app.project_folder_automatic = false;
        refresh_artifacts(app);
        return true;
    } catch (const std::exception& error) {
        set_message(app, error.what(), theme::danger);
        return false;
    }
}

void clear_viewport_scene(App& app) {
    app.photos.clear();
    app.image_qa_session.clear();
    app.image_qa = {};
    app.qa_preview_view = ~0U;
    app.qa_camera_valid = false;
    app.scene.clear();
    app.mesh.clear();
    app.mesh_load_failed = false;
#if defined(AETHERSCAN_HAS_TEXTURE)
    app.mesh_gpu_positions.clear();
    app.mesh_gpu_normals.clear();
    app.mesh_gpu_colours.clear();
    app.mesh_gpu_indices.clear();
    app.mesh_gpu_failed = false;
    app.mesh_preview.reset();
#endif
    app.scene_source.clear();
    app.camera = {};
    app.view_mode = VisualizationMode::points;
    ++app.scene_load_generation;
    app.loading_scene = false;
}

void clear_loaded_result(App& app) {
    clear_viewport_scene(app);
    app.monitor.reset();
    app.log.clear();
    app.console = {};
    app.fresh_lines.clear();
}

bool alignment_job_running(const App& app) {
    return app.job.running() && app.active_job == JobKind::align;
}

void new_project(App& app) {
    if (app.job.running() || app.loading_scene) return;
    stop_splat_view(app);
    clear_loaded_result(app);
    app.settings = {};
    app.layout = {};
    app.has_sparse = false;
    app.has_asfm = false;
    app.has_mvs = false;
    app.has_model = false;
    app.has_mesh = false;
    app.project_writer_version = 0;
    app.project_min_reader_version = 0;
    app.project_folder_automatic = false;
    set_message(app, "New project", theme::text_muted);
}

void apply_image_directory_selection(App& app) {
    stop_splat_view(app);
    clear_loaded_result(app);
    // An image-folder selection starts a different reconstruction. Retaining
    // an explicitly opened project here makes refresh_artifacts() immediately
    // reload that project's old SfM result into the new dataset.
    app.settings.project_dir.fill('\0');
    app.project_folder_automatic = false;
    assign_default_project_folder(app);
    refresh_artifacts(app);
    set_message(
        app, "Image dataset selected; previous viewport result cleared",
        theme::text_muted);
}

void select_image_folder(App& app) {
    if (app.job.running() || app.loading_scene) return;
    if (!pick_folder(
            L"Select the capture image folder", app.settings.images_dir))
        return;
    app.settings.video_frames_dir.fill('\0');
    apply_image_directory_selection(app);
}

void apply_video_selection(App& app) {
    stop_splat_view(app);
    clear_loaded_result(app);
    app.settings.video_frames_dir.fill('\0');
    app.settings.project_dir.fill('\0');
    app.project_folder_automatic = false;
    assign_default_project_folder(app);
    refresh_artifacts(app);
    set_message(
        app, "Video selected; Align Photos will extract sharp frames, then run SfM",
        theme::text_muted);
}

void select_video_file(App& app) {
    if (app.job.running() || app.loading_scene) return;
    if (!pick_video_file(
            L"Select a capture video", app.settings.images_dir))
        return;
    apply_video_selection(app);
}

void open_project_from_path(App& app, const std::filesystem::path& path);
void open_asfm_from_path(App& app, const std::filesystem::path& asfm);

void apply_dropped_image_source(
    App& app, const std::vector<std::string>& dropped) {
    if (app.job.running() || app.loading_scene) {
        set_message(
            app, "Stop the running job first to change images or video",
            theme::warning);
        return;
    }
    std::vector<std::filesystem::path> videos;
    for (const std::string& text : dropped) {
        if (text.empty()) continue;
        const std::filesystem::path path = path_from_drop(text);
        std::error_code status;
        if (std::filesystem::is_regular_file(path, status) &&
            aetherscan::io::is_video_path(path))
            videos.push_back(path);
    }
    if (videos.size() > 1) {
        set_message(app, "Drop a single video file", theme::danger);
        return;
    }
    if (videos.size() == 1) {
        if (dropped.size() != 1) {
            set_message(
                app, "Drop a single video, or a folder of photos",
                theme::danger);
            return;
        }
        store_utf8_path_field(app.settings.images_dir, videos.front());
        apply_video_selection(app);
        return;
    }
    std::string error;
    const auto directory = resolve_dropped_image_directory(dropped, error);
    if (!directory) {
        set_message(app, error, theme::danger);
        return;
    }
    store_utf8_path_field(app.settings.images_dir, *directory);
    apply_image_directory_selection(app);
}

void apply_dropped_paths(App& app, const std::vector<std::string>& dropped) {
    std::vector<std::filesystem::path> projects;
    std::vector<std::filesystem::path> asfms;
    std::vector<std::string> remainder;
    projects.reserve(dropped.size());
    asfms.reserve(dropped.size());
    remainder.reserve(dropped.size());
    for (const std::string& text : dropped) {
        if (text.empty()) continue;
        const std::filesystem::path path = path_from_drop(text);
        const std::string extension = lower_path_extension(path);
        if (extension == ".ascan")
            projects.push_back(path);
        else if (extension == ".asfm")
            asfms.push_back(path);
        else
            remainder.push_back(text);
    }

    if (projects.size() > 1) {
        set_message(app, "Drop a single .ascan project file", theme::danger);
        return;
    }
    if (projects.size() == 1) {
        open_project_from_path(app, projects.front());
        return;
    }
    if (asfms.size() > 1) {
        set_message(app, "Drop a single .asfm file", theme::danger);
        return;
    }
    if (asfms.size() == 1) {
        if (!remainder.empty()) {
            std::string error;
            if (const auto directory =
                    resolve_dropped_image_directory(remainder, error))
                store_utf8_path_field(app.settings.images_dir, *directory);
        }
        open_asfm_from_path(app, asfms.front());
        return;
    }
    apply_dropped_image_source(app, remainder.empty() ? dropped : remainder);
}

bool mouse_over_viewport(const App& app) {
    if (!app.viewport_bounds_valid) return false;
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    return mouse.x >= app.viewport_min.x && mouse.y >= app.viewport_min.y &&
           mouse.x < app.viewport_max.x && mouse.y < app.viewport_max.y;
}

void consume_dropped_paths(App& app) {
    if (app.dropped_paths.empty()) return;
    std::vector<std::string> dropped;
    dropped.swap(app.dropped_paths);
    if (!mouse_over_viewport(app)) {
        set_message(
            app,
            "Drop photos, a video, an .asfm scene, or an .ascan project on the viewport",
            theme::warning);
        return;
    }
    apply_dropped_paths(app, dropped);
}

void request_scene_load(
    App& app, const std::filesystem::path& cloud,
    const std::filesystem::path& poses, std::string label) {
    if (app.loading_scene) return;
    app.suppress_scene_auto_load = false;
    app.pending_scene_load_generation = app.scene_load_generation;
    app.loading_scene = true;
    app.scene_source = std::move(label);
    app.pending_load = std::async(
        std::launch::async,
        [cloud, poses] { return load_sparse_scene(cloud, poses); });
}

void request_gaussian_scene_load(App& app) {
    if (app.loading_scene) return;
    const auto model_path = existing_splat_model(app);
    const auto ascan = app.layout.project_file;
    const auto poses = app.layout.sparse_poses;
    if (model_path.empty() && ascan.empty()) return;
    const bool dataset = has_external_dataset(app);
    const std::filesystem::path dataset_source(app.settings.dataset_source.data());
    const std::string dataset_format = external_dataset_format(app);
    const std::filesystem::path dataset_initial_cloud(
        app.settings.dataset_initial_cloud.data());
    const std::filesystem::path dataset_images = reconstruction_images_path(app);
    app.suppress_scene_auto_load = false;
    app.pending_scene_load_generation = app.scene_load_generation;
    app.loading_scene = true;
    app.scene_source = "Gaussian centres";
    app.pending_load = std::async(
        std::launch::async,
        [model_path, ascan, poses, dataset, dataset_source, dataset_format,
         dataset_initial_cloud, dataset_images] {
            const auto merge_dataset_views = [&](SceneLoad& loaded) {
                // External-dataset training uses the imported cameras, not
                // an internal poses CSV. Always replace so compare / feature
                // QA snap to the same views the trainer used.
                if (!dataset || !loaded.ok) return;
                SceneLoad imported = sparse_scene_from_dataset(
                    dataset_source, dataset_format, dataset_initial_cloud,
                    dataset_images);
                if (!imported.ok) return;
                loaded.scene.views = std::move(imported.scene.views);
                loaded.scene.registered_views =
                    imported.scene.registered_views;
                loaded.scene.total_views = imported.scene.total_views;
            };
            try {
                if (!model_path.empty())
                {
                    SceneLoad loaded = load_gaussian_scene(model_path, poses);
                    merge_dataset_views(loaded);
                    return loaded;
                }
                std::error_code error;
                if (std::filesystem::exists(ascan, error)) {
                    const auto archive =
                        aetherscan::project::Archive::open(ascan);
                    if (archive.has(aetherscan::project::ChunkType::gaussians))
                    {
                        SceneLoad loaded = gaussian_scene_from_model(
                            aetherscan::splat::decode_gaussians(
                                archive.chunk(
                                    aetherscan::project::ChunkType::gaussians)),
                            poses);
                        merge_dataset_views(loaded);
                        return loaded;
                    }
                }
                SceneLoad loaded;
                loaded.error = "Project has no trained Gaussian model yet";
                return loaded;
            } catch (const std::exception& failure) {
                SceneLoad loaded;
                loaded.error = failure.what();
                return loaded;
            }
        });
}

void ensure_sparse_loaded(App& app) {
    if (alignment_job_running(app) || app.suppress_scene_auto_load ||
        !app.has_sparse || app.scene.has_points() || app.loading_scene)
        return;
    if (!app.layout.project_file.empty() || !app.layout.working_sfm.empty())
        request_ascan_scene_load(app);
    else
        request_scene_load(
            app, app.layout.sparse_ply, app.layout.sparse_poses, "Sparse cloud");
}

void ensure_gaussian_scene(App& app) {
    if (app.scene.has_gaussians() || app.loading_scene || !app.has_model) return;
    request_gaussian_scene_load(app);
}

std::filesystem::path existing_mesh_path(const App& app) {
    std::error_code error;
    const auto exists = [&](const std::filesystem::path& path) {
        return !path.empty() && std::filesystem::exists(path, error);
    };
    const std::array<std::filesystem::path, 3> preferred = {
        mesh_from_mvs(app.settings) ? app.layout.mvs_mesh_ply
                                    : app.layout.mesh_ply,
        app.layout.mesh_ply, app.layout.mvs_mesh_ply};
    for (const auto& path : preferred)
        if (exists(path)) return path;
    if (exists(app.layout.mvs_raw_mesh_ply)) return app.layout.mvs_raw_mesh_ply;
    return {};
}

void request_mesh_load(App& app, const bool frame_when_ready) {
    if (app.loading_scene) return;
    const auto ply = existing_mesh_path(app);
    const auto ascan = app.layout.project_file;
    if (ply.empty() && ascan.empty()) {
        app.mesh_load_failed = true;
        set_message(app, "No mesh file found next to the project", theme::warning);
        return;
    }
    app.loading_scene = true;
    app.mesh_load_failed = false;
    app.frame_mesh_on_load = frame_when_ready;
    app.pending_scene_load_generation = app.scene_load_generation;
    app.pending_mesh_load = std::async(
        std::launch::async,
        [ply, ascan] { return load_preview_mesh(ply, ascan); });
}

void ensure_mesh_loaded(App& app) {
    if (app.mesh.has() || app.loading_scene || app.mesh_load_failed ||
        !app.has_mesh)
        return;
    request_mesh_load(app, true);
}

void pack_mesh_gpu_buffers(App& app) {
#if defined(AETHERSCAN_HAS_TEXTURE)
    app.mesh_gpu_positions.clear();
    app.mesh_gpu_normals.clear();
    app.mesh_gpu_colours.clear();
    app.mesh_gpu_indices.clear();
    app.mesh_gpu_camera = {};
    if (!app.mesh.has()) return;
    const std::size_t count = app.mesh.vertices.size();
    app.mesh_gpu_positions.resize(count * 3U);
    for (std::size_t i = 0; i < count; ++i) {
        app.mesh_gpu_positions[3U * i] = app.mesh.vertices[i].x;
        app.mesh_gpu_positions[3U * i + 1U] = app.mesh.vertices[i].y;
        app.mesh_gpu_positions[3U * i + 2U] = app.mesh.vertices[i].z;
    }
    if (app.mesh.normals.size() == count) {
        app.mesh_gpu_normals.resize(count * 3U);
        for (std::size_t i = 0; i < count; ++i) {
            app.mesh_gpu_normals[3U * i] = app.mesh.normals[i].x;
            app.mesh_gpu_normals[3U * i + 1U] = app.mesh.normals[i].y;
            app.mesh_gpu_normals[3U * i + 2U] = app.mesh.normals[i].z;
        }
    }
    if (app.mesh.colours.size() == count) {
        app.mesh_gpu_colours.resize(count * 3U);
        for (std::size_t i = 0; i < count; ++i) {
            const std::uint32_t packed = app.mesh.colours[i];
            app.mesh_gpu_colours[3U * i] =
                static_cast<float>((packed >> IM_COL32_R_SHIFT) & 255U) / 255.F;
            app.mesh_gpu_colours[3U * i + 1U] =
                static_cast<float>((packed >> IM_COL32_G_SHIFT) & 255U) / 255.F;
            app.mesh_gpu_colours[3U * i + 2U] =
                static_cast<float>((packed >> IM_COL32_B_SHIFT) & 255U) / 255.F;
        }
    }
    app.mesh_gpu_indices.reserve(app.mesh.faces.size() * 3U);
    for (const auto& face : app.mesh.faces) {
        app.mesh_gpu_indices.push_back(face[0]);
        app.mesh_gpu_indices.push_back(face[1]);
        app.mesh_gpu_indices.push_back(face[2]);
    }
#else
    (void)app;
#endif
}

bool update_gpu_mesh_preview(App& app, const ImVec2 min, const ImVec2 max) {
#if defined(AETHERSCAN_HAS_TEXTURE)
    if (!app.mesh.has() || app.mesh_gpu_failed ||
        app.mesh_gpu_indices.empty())
        return false;
    std::uint32_t width = static_cast<std::uint32_t>(
        std::max(1.F, std::floor(max.x - min.x)));
    std::uint32_t height = static_cast<std::uint32_t>(
        std::max(1.F, std::floor(max.y - min.y)));
    const std::uint32_t cap = app.camera.interacting ? 960U : 1600U;
    if (width > cap || height > cap) {
        const float scale = static_cast<float>(cap) /
                            static_cast<float>(std::max(width, height));
        width = std::max(1U, static_cast<std::uint32_t>(width * scale));
        height = std::max(1U, static_cast<std::uint32_t>(height * scale));
    }
    const SplatPreviewCamera camera =
        make_preview_camera(app.camera, width, height);
    const bool same =
        app.mesh_preview.descriptor != nullptr &&
        camera.width == app.mesh_gpu_camera.width &&
        camera.height == app.mesh_gpu_camera.height &&
        camera.fx == app.mesh_gpu_camera.fx &&
        camera.cx == app.mesh_gpu_camera.cx &&
        camera.cy == app.mesh_gpu_camera.cy &&
        camera.world_to_camera == app.mesh_gpu_camera.world_to_camera &&
        app.mesh_gpu_wireframe == app.view_options.mesh_wireframe &&
        app.mesh_gpu_vertex_colour == app.view_options.mesh_vertex_colour;
    if (same) return true;
    try {
        if (!app.mesh_rasterizer)
            app.mesh_rasterizer =
                std::make_unique<aetherscan::texture::MeshPreviewRasterizer>();
        aetherscan::texture::MeshPreviewCamera gpu_camera;
        gpu_camera.world_to_camera = camera.world_to_camera;
        gpu_camera.position = camera.position;
        gpu_camera.fx = camera.fx;
        gpu_camera.fy = camera.fy;
        gpu_camera.cx = camera.cx;
        gpu_camera.cy = camera.cy;
        gpu_camera.width = camera.width;
        gpu_camera.height = camera.height;
        aetherscan::texture::MeshPreviewOptions options;
        options.wireframe = app.view_options.mesh_wireframe;
        options.vertex_colour = app.view_options.mesh_vertex_colour;
        const aetherscan::io::RgbImage image = app.mesh_rasterizer->render(
            app.mesh_gpu_positions, app.mesh_gpu_normals,
            app.mesh_gpu_colours, app.mesh_gpu_indices, gpu_camera, options);
        app.mesh_preview.upload(image);
        app.mesh_gpu_camera = camera;
        app.mesh_gpu_wireframe = options.wireframe;
        app.mesh_gpu_vertex_colour = options.vertex_colour;
        return app.mesh_preview.descriptor != nullptr;
    } catch (const std::exception& failure) {
        app.mesh_gpu_failed = true;
        set_message(
            app, std::string("Mesh GPU rasterizer: ") + failure.what(),
            theme::warning);
        return false;
    }
#else
    (void)app;
    (void)min;
    (void)max;
    return false;
#endif
}

void show_mesh_view(App& app, const bool frame_when_ready) {
    app.view_mode = VisualizationMode::mesh;
    stop_splat_view(app);
    if (app.mesh.has()) {
        app.camera.frame(app.mesh.centroid, app.mesh.radius);
        return;
    }
    app.mesh_load_failed = false;
    request_mesh_load(app, frame_when_ready);
}

aetherscan::splat::VisualizeOptions editor_visualize_options(const App& app) {
    aetherscan::splat::VisualizeOptions options;
    options.mode = app.view_mode == VisualizationMode::points
        ? aetherscan::splat::VisualizationMode::points
        : app.view_mode == VisualizationMode::rings
            ? aetherscan::splat::VisualizationMode::rings
            : aetherscan::splat::VisualizationMode::splat;
    options.point_size_px = app.view_options.point_size;
    options.ring_scale = app.view_options.ring_scale;
    return options;
}

void write_preview_vis(App& app) {
    if (app.layout.preview_vis_file.empty()) return;
    aetherscan::splat::write_visualization_sidecar(
        app.layout.preview_vis_file, editor_visualize_options(app),
        app.preview_camera_revision);
}

void publish_preview_vis(App& app) {
    ++app.preview_camera_revision;
    write_preview_vis(app);
}

void set_visualization_mode(App& app, const VisualizationMode mode) {
    app.view_mode = mode;
    // Changing the rail mode is an explicit request to show scene data again.
    if (mode == VisualizationMode::points)
        app.suppress_scene_auto_load = false;
    if (mode == VisualizationMode::mesh) {
        show_mesh_view(app, !app.mesh.has());
        write_preview_vis(app);
        return;
    }
    const bool training =
        app.job.running() && app.active_job == JobKind::train;
    if (app.has_model && !training)
        start_splat_view(app);
    else if (mode == VisualizationMode::points && !live_preview_active(app) &&
             !app.scene.has_points())
        ensure_sparse_loaded(app);
    write_preview_vis(app);
    sync_live_preview_camera(
        app, true, app.preview_raster_width, app.preview_raster_height);
}

constexpr float k_view_rail_pad = 10.F;
constexpr float k_view_rail_top = 52.F;
constexpr float k_view_rail_width = 44.F;
constexpr float k_view_rail_height = 156.F;
constexpr float k_scene_toggle_gap = 8.F;
constexpr float k_scene_toggle_height = 84.F;

ImRect view_mode_rail_rect(const ImVec2 view_min) {
    const ImVec2 origin{
        view_min.x + k_view_rail_pad, view_min.y + k_view_rail_top};
    return {
        origin.x, origin.y, origin.x + k_view_rail_width,
        origin.y + k_view_rail_height};
}

ImRect scene_toggle_rail_rect(const ImVec2 view_min) {
    const ImVec2 origin{
        view_min.x + k_view_rail_pad,
        view_min.y + k_view_rail_top + k_view_rail_height + k_scene_toggle_gap};
    return {
        origin.x, origin.y, origin.x + k_view_rail_width,
        origin.y + k_scene_toggle_height};
}

bool view_mode_rail_contains(const ImVec2 view_min, const ImVec2 mouse) {
    return view_mode_rail_rect(view_min).Contains(mouse) ||
           scene_toggle_rail_rect(view_min).Contains(mouse);
}

void set_camera_overlays(ViewOptions& options, const bool visible) {
    options.show_views = visible;
    options.show_camera_photos = visible;
}

// One click path on top of the full-viewport InvisibleButton. The previous
// ghost_button + raw hit test both fired, so toggles flipped twice and looked
// stuck.
bool rail_icon_button(
    const char* id, const icons::Icon icon, const ImVec2 min, const ImVec2 size,
    const bool active, const bool enabled, const char* tooltip) {
    ImGui::SetCursorScreenPos(min);
    ImGui::SetNextItemAllowOverlap();
    if (!enabled) ImGui::BeginDisabled();
    ImGui::InvisibleButton(id, size);
    if (!enabled) ImGui::EndDisabled();
    const bool hovered = ImGui::IsItemHovered();
    const bool pressed = enabled && hovered && ImGui::IsItemClicked();

    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 max{min.x + size.x, min.y + size.y};
    if (enabled && active) {
        draw->AddRectFilled(
            min, max, theme::u32(theme::fade(theme::accent, 0.20F)), 6.F);
        draw->AddRect(min, max, theme::u32(theme::accent, 0.70F), 6.F);
    } else if (enabled && hovered) {
        draw->AddRectFilled(min, max, theme::u32(theme::surface_3), 6.F);
    }
    const ImVec4 icon_colour = !enabled
        ? theme::text_faint
        : (active ? theme::accent
                  : (hovered ? theme::text_bright : theme::text_muted));
    const float pad = std::max(7.F, std::min(size.x, size.y) * 0.22F);
    icons::draw(
        draw, icon, {min.x + pad, min.y + pad}, {max.x - pad, max.y - pad},
        theme::u32(icon_colour), 1.8F);
    if (hovered && tooltip) ImGui::SetTooltip("%s", tooltip);
    return pressed;
}

bool draw_view_mode_rail(App& app, const ImVec2 view_min) {
    const ImRect rail = view_mode_rail_rect(view_min);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(
        rail.Min, rail.Max, IM_COL32(16, 18, 23, 214), 8.F);
    draw->AddRect(
        rail.Min, rail.Max, theme::u32(theme::border, 0.7F), 8.F);

    constexpr float k_btn = 32.F;
    constexpr float k_inner = 6.F;
    constexpr float k_gap = 4.F;
    const ImVec2 button_size{k_btn, k_btn};
    bool hovered = false;

    const bool training =
        app.job.running() && app.active_job == JobKind::train;
    const bool splat_ok = app.has_model || training;
    const bool rings_ok = app.has_model || training;

    struct RailItem {
        const char* id;
        icons::Icon icon;
        VisualizationMode mode;
        bool enabled;
        const char* tooltip;
    };
    const bool mesh_ok = app.has_mesh || app.mesh.has();
    const RailItem items[] = {
        {"##viz_points", icons::Icon::points, VisualizationMode::points, true,
         "Point Cloud"},
        {"##viz_splat", icons::Icon::splat, VisualizationMode::splat, splat_ok,
         splat_ok ? "Splat" : "Train 3DGS to view the splat"},
        {"##viz_rings", icons::Icon::rings, VisualizationMode::rings, rings_ok,
         rings_ok ? "Rings"
                  : "Available while training or after a Gaussian model exists"},
        {"##viz_mesh", icons::Icon::cube, VisualizationMode::mesh, mesh_ok,
         mesh_ok ? "Mesh"
                 : "Build a mesh to inspect the reconstructed surface"},
    };

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0.F, k_gap});
    for (int i = 0; i < 4; ++i) {
        const ImVec2 button_min{
            rail.Min.x + k_inner,
            rail.Min.y + k_inner + static_cast<float>(i) * (k_btn + k_gap)};
        if (rail_icon_button(
                items[i].id, items[i].icon, button_min, button_size,
                app.view_mode == items[i].mode, items[i].enabled,
                items[i].tooltip))
            set_visualization_mode(app, items[i].mode);
        hovered = hovered || ImGui::IsItemHovered();
    }
    ImGui::PopStyleVar();
    return hovered;
}

bool draw_scene_toggle_rail(App& app, const ImVec2 view_min) {
    const ImRect rail = scene_toggle_rail_rect(view_min);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(rail.Min, rail.Max, IM_COL32(16, 18, 23, 214), 8.F);
    draw->AddRect(rail.Min, rail.Max, theme::u32(theme::border, 0.7F), 8.F);

    constexpr float k_btn = 32.F;
    constexpr float k_inner = 6.F;
    constexpr float k_gap = 4.F;
    const ImVec2 button_size{k_btn, k_btn};
    bool hovered = false;

    const ImVec2 cameras_min{rail.Min.x + k_inner, rail.Min.y + k_inner};
    if (rail_icon_button(
            "##tog_cameras", icons::Icon::frustum, cameras_min, button_size,
            app.view_options.show_views, true,
            app.view_options.show_views
                ? "Hide camera frustums"
                : "Show camera frustums"))
        set_camera_overlays(app.view_options, !app.view_options.show_views);
    hovered = hovered || ImGui::IsItemHovered();

    const ImVec2 grid_min{
        rail.Min.x + k_inner, rail.Min.y + k_inner + k_btn + k_gap};
    if (rail_icon_button(
            "##tog_grid", icons::Icon::grid, grid_min, button_size,
            app.view_options.show_grid, true,
            app.view_options.show_grid ? "Hide ground grid"
                                      : "Show ground grid"))
        app.view_options.show_grid = !app.view_options.show_grid;
    hovered = hovered || ImGui::IsItemHovered();
    return hovered;
}

void draw_workspace_toggle(App& app, const ImVec2 origin) {
    struct Item {
        const char* id;
        icons::Icon icon;
        const char* label;
        ViewportWorkspace workspace;
    };
    const Item items[] = {
        {"##ws_3d", icons::Icon::cube, "3D", ViewportWorkspace::scene_3d},
        {"##ws_2d", icons::Icon::view2d, "2D", ViewportWorkspace::image_2d},
    };
    ImGui::SetCursorScreenPos({origin.x + 10.F, origin.y + 5.F});
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 cluster_min = ImGui::GetCursorScreenPos();
    const ImVec2 cluster_max{cluster_min.x + 118.F, cluster_min.y + 26.F};
    draw->AddRectFilled(
        cluster_min, cluster_max, theme::u32(theme::surface_2), 7.F);
    draw->AddRect(
        cluster_min, cluster_max, theme::u32(theme::border, 0.7F), 7.F);
    for (int i = 0; i < 2; ++i) {
        const ImVec2 chip_min{
            cluster_min.x + 3.F + static_cast<float>(i) * 56.F,
            cluster_min.y + 2.F};
        const ImVec2 chip_max{chip_min.x + 54.F, cluster_min.y + 24.F};
        ImGui::SetCursorScreenPos(chip_min);
        ImGui::InvisibleButton(items[i].id, {54.F, 22.F});
        const bool active = app.workspace == items[i].workspace;
        const bool hovered = ImGui::IsItemHovered();
        if (active)
            draw->AddRectFilled(
                chip_min, chip_max, theme::u32(theme::fade(theme::accent, 0.20F)),
                5.F);
        else if (hovered)
            draw->AddRectFilled(
                chip_min, chip_max, theme::u32(theme::surface_3), 5.F);
        const ImVec4 colour = active ? theme::accent : theme::text_muted;
        icons::draw(
            draw, items[i].icon, {chip_min.x + 6.F, chip_min.y + 3.F},
            {chip_min.x + 20.F, chip_min.y + 17.F}, theme::u32(colour), 1.5F);
        draw->AddText(
            {chip_min.x + 24.F, chip_min.y + 3.F}, theme::u32(colour),
            items[i].label);
        if (ImGui::IsItemClicked())
            set_viewport_workspace(app, items[i].workspace);
        if (hovered)
            ImGui::SetTooltip(
                items[i].workspace == ViewportWorkspace::scene_3d
                    ? "3D scene view"
                    : "2D image QA — features, GT vs 3DGS, error map");
    }
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
    // 2D QA owns the sidecar while that workspace is open. Writing the orbit
    // camera here would wipe the capture pose the compare view is waiting on.
    if (app.workspace == ViewportWorkspace::image_2d) return;
    const bool live = live_preview_active(app);
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
    const auto vis = editor_visualize_options(app);
    write_preview_camera_file(
        app.layout.preview_camera_file, preview, app.preview_camera_revision,
        aetherscan::splat::visualization_mode_name(vis.mode),
        vis.point_size_px, vis.ring_scale);
    // The shared camera sidecar now describes the orbit camera, not the QA
    // capture pose; drop the QA freshness marker until it is rewritten.
    app.qa_camera_valid = false;
    write_preview_vis(app);
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

// Double-click a training frustum to look through that capture; otherwise
// focus the orbit pivot on the point (or splat pixel) under the cursor.
bool handle_viewport_double_click(
    App& app, const bool accepts_input, const SceneDrawStats& stats,
    const ImVec2 min, const ImVec2 max) {
    if (!accepts_input || ImGui::GetIO().WantTextInput) return false;
    if (!ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) return false;
    if (stats.hovered_view >= 0 &&
        static_cast<std::size_t>(stats.hovered_view) < app.scene.views.size() &&
        app.scene.views[static_cast<std::size_t>(stats.hovered_view)].registered) {
        snap_preview_to_index(app, static_cast<unsigned>(stats.hovered_view));
        app.camera.interacting = false;
        return true;
    }
    Vec3 point;
    if (!pick_orbit_focus_point(
            app.scene, app.camera, min, max, ImGui::GetIO().MousePos, point,
            app.view_mode == VisualizationMode::mesh ? &app.mesh : nullptr))
        return false;
    app.camera.focus_on(point);
    app.preview_follow_view = false;
    app.camera.interacting = false;
    return true;
}

[[nodiscard]] bool qa_capture_frame_ready(const App& app) {
    if (!app.qa_camera_valid || app.image_qa.selected < 0) return false;
    if (app.qa_preview_view != static_cast<unsigned>(app.image_qa.selected))
        return false;
    const std::uint64_t consumed = gpu::consumed_timeline_value();
    return consumed > 0 && consumed > app.qa_camera_timeline;
}

void sync_qa_preview_camera(App& app) {
    if (app.layout.preview_camera_file.empty()) return;
    if (app.image_qa.selected < 0 ||
        static_cast<std::size_t>(app.image_qa.selected) >= app.scene.views.size())
        return;
    const ViewPose& pose = app.scene.views[static_cast<std::size_t>(app.image_qa.selected)];
    std::uint32_t width = k_preview_extent;
    std::uint32_t height = k_preview_extent;
    if (pose.width > 0 && pose.height > 0) {
        if (pose.width >= pose.height) {
            width = k_preview_extent;
            height = std::max<std::uint32_t>(
                1, static_cast<std::uint32_t>(std::lround(
                       static_cast<double>(k_preview_extent) * pose.height /
                       pose.width)));
        } else {
            height = k_preview_extent;
            width = std::max<std::uint32_t>(
                1, static_cast<std::uint32_t>(std::lround(
                       static_cast<double>(k_preview_extent) * pose.width /
                       pose.height)));
        }
    }
    const unsigned view = static_cast<unsigned>(app.image_qa.selected);
    if (app.qa_camera_valid && app.qa_preview_view == view &&
        app.preview_raster_width == width &&
        app.preview_raster_height == height)
        return;
    std::error_code error;
    std::filesystem::create_directories(
        app.layout.preview_camera_file.parent_path(), error);
    ++app.preview_camera_revision;
    app.preview_raster_width = width;
    app.preview_raster_height = height;
    app.preview_view = view;
    app.preview_follow_view = true;
    const SplatPreviewCamera preview =
        make_preview_camera_from_view(pose, width, height);
    if (!write_preview_camera_file(
            app.layout.preview_camera_file, preview, app.preview_camera_revision,
            "splat", app.view_options.point_size, app.view_options.ring_scale)) {
        app.qa_camera_valid = false;
        return;
    }
    app.qa_preview_view = view;
    app.qa_camera_timeline = gpu::consumed_timeline_value();
    app.qa_camera_valid = true;
    write_preview_view_index(app.layout, app.preview_view);
    aetherscan::splat::VisualizeOptions vis;
    vis.mode = aetherscan::splat::VisualizationMode::splat;
    vis.point_size_px = app.view_options.point_size;
    vis.ring_scale = app.view_options.ring_scale;
    aetherscan::splat::write_visualization_sidecar(
        app.layout.preview_vis_file, vis, app.preview_camera_revision);
    app.qa_metrics_after =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(220);
}

void capture_qa_render(App& app) {
    if (app.workspace != ViewportWorkspace::image_2d) return;
    if (!image_qa_needs_render(app.image_qa.mode)) return;
    // The wipe itself samples GPU textures. Readback is only for PSNR / error.
    if (app.image_qa.dragging_wipe) return;
    if (!app.image_qa.metrics_dirty) return;
    if (app.image_qa_session.metrics_busy()) return;
    if (!app.preview.display.descriptor || !qa_capture_frame_ready(app))
        return;
    if (!app.image_qa_session.has_gt() ||
        app.image_qa_session.loaded_view() != app.image_qa.selected)
        return;
    if (std::chrono::steady_clock::now() < app.qa_metrics_after) return;

    const std::uint64_t revision = gpu::consumed_timeline_value();
    if (app.image_qa_session.has_render_pixels() &&
        app.image_qa_session.render_view() == app.image_qa.selected &&
        app.image_qa_session.metrics().valid) {
        app.image_qa.metrics_dirty = false;
        return;
    }
    aetherscan::io::RgbImage render;
    if (!app.preview.display.download_rgb(render)) return;
    app.image_qa_session.set_render(
        std::move(render), app.image_qa.selected, revision);
    app.image_qa.metrics_dirty = false;
}

void set_viewport_workspace(App& app, const ViewportWorkspace workspace) {
    if (app.workspace == workspace) return;
    app.workspace = workspace;
    app.qa_preview_view = ~0U;
    app.qa_camera_valid = false;
    if (workspace == ViewportWorkspace::image_2d) {
        refresh_image_qa_folder(
            app.image_qa, reconstruction_images_path(app));
        const int count = image_qa_count(app.image_qa, app.scene);
        app.qa_metrics_after =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
        app.image_qa.metrics_dirty = true;
        if (app.image_qa.selected < 0 && count > 0) {
            select_image_qa_view(
                app.image_qa, static_cast<int>(app.preview_view), count);
        }
    } else if (live_preview_active(app)) {
        sync_live_preview_camera(
            app, true, app.preview_raster_width, app.preview_raster_height);
    }
}

void ensure_qa_preview(App& app) {
    if (app.workspace != ViewportWorkspace::image_2d) return;
    // Publish the capture pose before starting the viewer so the first
    // streamed frame is already the photo being inspected, not the orbit eye.
    sync_qa_preview_camera(app);
    if (image_qa_needs_render(app.image_qa.mode) && app.has_model &&
        !live_preview_active(app) && !app.job.running())
        start_splat_view(app);
    sync_qa_preview_camera(app);
    capture_qa_render(app);
}

void sync_qa_selection_to_preview(App& app, const int previous) {
    if (app.image_qa.selected == previous) return;
    if (app.image_qa.selected < 0 ||
        static_cast<std::size_t>(app.image_qa.selected) >= app.scene.views.size())
        return;
    app.preview_view = static_cast<unsigned>(app.image_qa.selected);
    app.preview_follow_view = true;
    write_preview_view_index(app.layout, app.preview_view);
    snap_orbit_to_view(
        app.camera, app.scene.views[static_cast<std::size_t>(app.image_qa.selected)]);
    app.qa_preview_view = ~0U;
    app.qa_camera_valid = false;
    app.image_qa.metrics_dirty = true;
    app.qa_metrics_after =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(220);
}

void apply_opened_project(App& app) {
    stop_splat_view(app);
    clear_loaded_result(app);
    app.project_folder_automatic = false;
    refresh_artifacts(app);
    try {
        const auto archive =
            aetherscan::project::Archive::open(app.layout.project_file);
        apply_project_settings(app, aetherscan::project::read_settings(archive));
        refresh_artifacts(app);
    } catch (const std::exception& error) {
        set_message(app, error.what(), theme::danger);
        return;
    }
    if (has_external_dataset(app))
        // The imported alignment drives training for this project; internal
        // SfM artifacts, if any, describe a different capture setup.
        request_dataset_scene_load(app);
    else if (app.has_sparse)
        request_ascan_scene_load(app);
    else if (app.has_asfm)
        request_asfm_scene_load(
            app, app.layout.sparse_asfm, app.layout.sparse_asfm.filename().string());
    else
        set_message(app, "Project opened; no SfM stage yet", theme::text_muted);
}

void select_project_folder(App& app) {
    if (app.job.running() || app.loading_scene) return;
    if (!pick_project_file(
            L"Open AetherScan Project", app.settings.project_dir, false))
        return;
    apply_opened_project(app);
}

void open_project_from_path(App& app, const std::filesystem::path& path) {
    if (app.job.running() || app.loading_scene) {
        set_message(
            app, "Cannot open a project while a job is running", theme::warning);
        return;
    }
    store_utf8_path_field(app.settings.project_dir, path);
    apply_opened_project(app);
}

std::filesystem::path inferred_images_dir_near(
    const std::filesystem::path& asfm) {
    const auto parent = asfm.parent_path();
    if (directory_has_images(parent)) return parent;
    const auto images = parent / "images";
    if (directory_has_images(images)) return images;
    return {};
}

void infer_images_dir_from_scene(App& app) {
    if (app.settings.images_dir[0] != '\0') return;
    std::optional<std::filesystem::path> parent;
    for (const ViewPose& view : app.scene.views) {
        if (view.image_path.empty() || view.image_path.is_relative()) continue;
        const auto directory = view.image_path.parent_path();
        if (directory.empty()) continue;
        if (!parent)
            parent = directory;
        else if (!same_directory(*parent, directory))
            return;
    }
    if (parent && directory_has_images(*parent))
        store_utf8_path_field(app.settings.images_dir, *parent);
}

void open_asfm_from_path(App& app, const std::filesystem::path& asfm) {
    if (app.job.running() || app.loading_scene) {
        set_message(
            app, "Cannot open SfM while a job is running", theme::warning);
        return;
    }
    stop_splat_view(app);
    clear_loaded_result(app);

    const auto sibling = asfm.parent_path() / (asfm.stem().string() + ".ascan");
    std::error_code error;
    if (std::filesystem::exists(sibling, error)) {
        store_utf8_path_field(app.settings.project_dir, sibling);
        app.project_folder_automatic = false;
        try {
            refresh_artifacts(app);
            const auto archive =
                aetherscan::project::Archive::open(app.layout.project_file);
            apply_project_settings(
                app, aetherscan::project::read_settings(archive));
        } catch (...) {
        }
    } else {
        store_utf8_path_field(
            app.settings.project_dir,
            asfm.parent_path() / (asfm.stem().string() + ".ascan"));
        app.project_folder_automatic = true;
    }

    if (app.settings.images_dir[0] == '\0') {
        const auto inferred = inferred_images_dir_near(asfm);
        if (!inferred.empty())
            store_utf8_path_field(app.settings.images_dir, inferred);
    }
    if (app.settings.project_dir[0] != '\0') refresh_artifacts(app);
    app.has_sparse = true;
    app.has_asfm = true;
    request_asfm_scene_load(app, asfm, asfm.filename().string());
}

void save_project_as(App& app) {
    if (app.job.running()) return;
    if (!pick_project_file(
            L"Save AetherScan Project", app.settings.project_dir, true))
        return;
    {
        std::filesystem::path path =
            path_from_utf8_field(app.settings.project_dir.data());
        std::string extension = path.extension().string();
        std::transform(
            extension.begin(), extension.end(), extension.begin(),
            [](const unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
        if (extension != ".ascan") path.replace_extension(".ascan");
        store_path_field(app.settings.project_dir, path);
    }
    app.project_folder_automatic = false;
    refresh_artifacts(app);
    if (save_project_to_path(app, app.layout.project_file))
        set_message(app, "Project saved", theme::success);
}

void save_project(App& app) {
    if (app.job.running()) return;
    assign_default_project_folder(app);
    refresh_artifacts(app);
    if (app.layout.project_file.empty()) {
        save_project_as(app);
        return;
    }
    if (save_project_to_path(app, app.layout.project_file))
        set_message(app, "Project saved", theme::success);
}

bool has_reconstruction_result(const App& app) {
    if (app.scene.has_points() || app.has_sparse || app.has_asfm ||
        app.has_mvs || app.has_model || app.has_mesh)
        return true;
    std::error_code dense_error;
    if (!app.layout.dense_ply.empty() &&
        std::filesystem::exists(app.layout.dense_ply, dense_error))
        return true;
    if (app.layout.cache.empty()) return false;
    std::error_code error;
    return std::filesystem::exists(app.layout.cache, error);
}

void delete_reconstruction_results(App& app) {
    if (app.job.running() || app.loading_scene || app.layout.root.empty()) return;

    stop_splat_view(app);
    clear_loaded_result(app);
    app.suppress_scene_auto_load = true;
    const std::array<std::filesystem::path, 17> generated_files = {
        app.layout.sparse_ply, app.layout.sparse_asfm, app.layout.sparse_mvs,
        app.layout.sparse_poses, app.layout.splat_ply, app.layout.splat_sog,
        app.layout.splat_spz, app.layout.splat_glb, app.layout.mesh_ply,
        app.layout.mvs_mesh_ply, app.layout.mvs_raw_mesh_ply,
        app.layout.dense_ply, app.layout.align_log, app.layout.train_log,
        app.layout.dense_log, app.layout.export_log, app.layout.view_log};

    std::uintmax_t removed = 0;
    std::string failure;
    for (const std::filesystem::path& path : generated_files) {
        if (path.empty()) continue;
        std::error_code error;
        if (std::filesystem::remove(path, error)) ++removed;
        if (error && failure.empty()) failure = error.message();
    }

    if (!app.layout.project_file.empty()) {
        try {
            if (std::filesystem::exists(app.layout.project_file)) {
                auto archive =
                    aetherscan::project::Archive::open(app.layout.project_file);
                archive.erase_chunk(aetherscan::project::ChunkType::sfm);
                archive.erase_chunk(aetherscan::project::ChunkType::gaussians);
                archive.erase_chunk(aetherscan::project::ChunkType::mesh);
                archive.erase_chunk(aetherscan::project::ChunkType::texture);
                aetherscan::project::write_settings(
                    archive, collect_project_settings(app),
                    app.layout.project_file);
                archive.save(app.layout.project_file);
            }
        } catch (const std::exception& error) {
            if (failure.empty()) failure = error.what();
        }
    }

    // resolve_layout() keeps cache as a child of the project directory.
    // relationship before a recursive removal so a malformed path can never
    // broaden the deletion target.
    const std::filesystem::path root = app.layout.root.lexically_normal();
    const std::filesystem::path cache = app.layout.cache.lexically_normal();
    const bool safe_cache = !root.empty() && !cache.empty() && cache != root &&
                            cache.parent_path() == root &&
                            (cache.filename() == "cache" ||
                             cache.extension() == ".cache");
    if (safe_cache) {
        std::error_code error;
        removed += std::filesystem::remove_all(cache, error);
        if (error && failure.empty()) failure = error.message();
    }

    // With reuse_cache disabled, resolve_layout() places the live sfm.bin and
    // preview sidecars under %TEMP%/AetherScan/<project-hash>, not in
    // layout.cache. Delete that exact namespaced directory as well; leaving it
    // behind makes refresh_artifacts() resurrect the supposedly deleted scene.
    const std::filesystem::path runtime =
        app.layout.working_sfm.parent_path().lexically_normal();
    std::error_code temp_error;
    const std::filesystem::path temp_root =
        (std::filesystem::temp_directory_path(temp_error) / "AetherScan")
            .lexically_normal();
    const bool safe_runtime =
        !temp_error && !runtime.empty() && runtime != temp_root &&
        runtime.parent_path() == temp_root;
    if (safe_runtime) {
        std::error_code error;
        removed += std::filesystem::remove_all(runtime, error);
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

void poll_camera_photos(App& app) {
    if (app.scene.views.empty()) {
        if (app.photos.size() != 0) app.photos.clear();
        return;
    }
    attach_view_image_paths(app.scene, reconstruction_images_path(app));
    app.photos.resize(app.scene.views.size());
    if (app.workspace == ViewportWorkspace::image_2d) {
        const int count = static_cast<int>(app.scene.views.size());
        const int selected = std::clamp(app.image_qa.selected, 0, count - 1);
        const int begin = std::max(0, selected - 18);
        const int end = std::min(count, selected + 19);
        const int strip_begin = std::max(0, app.image_qa.filmstrip_first);
        const int strip_end = std::min(count, app.image_qa.filmstrip_last + 1);
        for (int i = begin; i < end; ++i)
            app.photos.request(
                static_cast<std::size_t>(i), app.scene.views[static_cast<std::size_t>(i)].image_path);
        for (int i = strip_begin; i < strip_end; ++i)
            app.photos.request(
                static_cast<std::size_t>(i), app.scene.views[static_cast<std::size_t>(i)].image_path);
    } else if (app.view_options.show_views && app.view_options.show_camera_photos) {
        std::vector<std::size_t> markers;
        sampled_view_indices(app.scene, markers);
        for (const std::size_t index : markers)
            app.photos.request(index, app.scene.views[index].image_path);
    }
    app.photos.poll();
}

void poll_mesh_load(App& app) {
    if (!app.pending_mesh_load.valid()) return;
    if (app.pending_mesh_load.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
        return;
    MeshLoad loaded = app.pending_mesh_load.get();
    app.loading_scene = false;
    if (app.pending_scene_load_generation != app.scene_load_generation) return;
    if (!loaded.ok) {
        app.mesh_load_failed = true;
        set_message(app, "Mesh: " + loaded.error, theme::danger);
        return;
    }
    app.mesh = std::move(loaded.mesh);
    app.mesh_load_failed = false;
    pack_mesh_gpu_buffers(app);
    app.camera.frame(app.mesh.centroid, app.mesh.radius);
    app.frame_mesh_on_load = false;
    app.view_mode = VisualizationMode::mesh;
    set_message(
        app,
        "Mesh: " + format_count(app.mesh.vertices.size()) + " vertices, " +
            format_count(app.mesh.faces.size()) + " faces",
        theme::success);
}

void poll_scene_load(App& app) {
    if (!app.loading_scene || !app.pending_load.valid()) return;
    if (app.pending_load.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
        return;
    SceneLoad loaded = app.pending_load.get();
    app.loading_scene = false;
    if (app.pending_scene_load_generation != app.scene_load_generation) return;
    if (!loaded.ok) {
        set_message(app, "Point cloud: " + loaded.error, theme::danger);
        return;
    }
    app.photos.clear();
    app.image_qa_session.clear();
    app.qa_preview_view = ~0U;
    app.qa_camera_valid = false;
    app.scene = std::move(loaded.scene);
    app.alignment_preview_seen = false;
    // When an internal SfM or Gaussian scene replaces the imported cameras,
    // allow the next ensure pass to restore the dataset view poses, which are
    // the ones training actually uses.
    if (app.scene_source != "External dataset" &&
        app.scene_source != "Gaussian centres")
        app.dataset_scene_key.clear();
    if (app.image_qa.selected >= static_cast<int>(app.scene.views.size()))
        app.image_qa.selected = app.scene.views.empty() ? -1 : 0;
    infer_images_dir_from_scene(app);
    attach_view_image_paths(app.scene, reconstruction_images_path(app));
    if (!live_preview_active(app) && app.view_mode != VisualizationMode::mesh)
        app.camera.frame(app.scene);
    if (app.view_mode != VisualizationMode::splat &&
        app.view_mode != VisualizationMode::rings &&
        app.view_mode != VisualizationMode::mesh)
        app.view_mode = VisualizationMode::points;
    set_message(
        app,
        app.scene_source + ": " + format_count(app.scene.points.size()) +
            " points, " + std::to_string(app.scene.registered_views) + " / " +
            std::to_string(app.scene.total_views) + " views registered",
        theme::success);
}

void poll_alignment_preview(App& app) {
    const bool aligning = app.job.running() && app.active_job == JobKind::align;
    if (app.alignment_preview_load.valid()) {
        if (app.alignment_preview_load.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
        auto loaded = app.alignment_preview_load.get();
        if (aligning && !app.loading_scene && loaded.ok &&
            app.alignment_preview_load_generation == app.alignment_preview_generation) {
            const bool first = !app.alignment_preview_seen;
            app.scene = std::move(loaded.scene);
            app.scene_source = "Alignment preview";
            attach_view_image_paths(app.scene, reconstruction_images_path(app));
            app.alignment_preview_seen = true;
            if (first) {
                app.photos.clear();
                app.camera.frame(app.scene);
                app.view_mode = VisualizationMode::points;
            }
            app.image_qa.metrics_dirty = true;
            set_message(app, "Aligning: " + std::to_string(app.scene.registered_views) +
                " cameras (partial result)", theme::accent);
        }
    }
    if (!aligning || app.loading_scene || app.layout.working_sfm.empty()) return;
    const auto now = std::chrono::steady_clock::now();
    if (now < app.alignment_preview_poll) return;
    app.alignment_preview_poll = now + std::chrono::milliseconds(500);
    auto path = app.layout.working_sfm;
    path += ".preview.asfm";
    std::error_code error;
    const auto stamp = std::filesystem::last_write_time(path, error);
    if (error || stamp == app.alignment_preview_stamp) return;
    app.alignment_preview_stamp = stamp;
    app.alignment_preview_load_generation = app.alignment_preview_generation;
    const std::filesystem::path images = reconstruction_images_path(app);
    app.alignment_preview_load = std::async(std::launch::async, [path, images] {
        try { return sparse_scene_from_sfm(load_working_sfm(path, images), false); }
        catch (const std::exception& error) {
            SceneLoad result;
            result.error = error.what();
            return result;
        }
    });
}

void start_align(App& app) {
    if (app.job.running()) return;
    stop_splat_view(app);
    if (has_external_dataset(app)) {
        set_message(
            app,
            "External dataset selected; use Train 3DGS instead of Align Photos",
            theme::warning);
        return;
    }
    assign_default_project_folder(app);
    if (app.settings.project_dir[0] == '\0') {
        set_message(app, "Save or choose a project file first", theme::warning);
        return;
    }
    refresh_artifacts(app);
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (error) {
        set_message(app, "Cannot create project directory", theme::danger);
        return;
    }
    if (!app.layout.working_sfm.empty()) {
        std::error_code cache_error;
        std::filesystem::create_directories(
            app.layout.working_sfm.parent_path(), cache_error);
    }
    {
        std::error_code stale;
        std::filesystem::remove(app.layout.sparse_mvs, stale);
    }
    if (is_video_source(app.settings)) {
        std::error_code video_error;
        const std::filesystem::path video =
            path_from_utf8_field(app.settings.images_dir.data());
        if (!std::filesystem::is_regular_file(video, video_error)) {
            set_message(app, "Video file not found", theme::danger);
            return;
        }
        const auto frames = reconstruction_images_path(app);
        if (!directory_has_images(frames) &&
            !aetherscan::io::ffmpeg_available()) {
            set_message(
                app,
                "ffmpeg was not found. Install ffmpeg and add it to PATH.",
                theme::danger);
            return;
        }
    } else if (!directory_has_images(
                   path_from_utf8_field(app.settings.images_dir.data()))) {
        set_message(
            app, "No images found in the selected source folder", theme::danger);
        return;
    }
    try {
        app.monitor.begin(JobKind::align);
        auto preview_path = app.layout.working_sfm;
        preview_path += ".preview.asfm";
        std::error_code preview_error;
        std::filesystem::remove(preview_path, preview_error);
        app.alignment_preview_stamp = {};
        if (preview_error) {
            app.alignment_preview_stamp = std::filesystem::last_write_time(preview_path, preview_error);
            if (preview_error) app.alignment_preview_stamp = {};
        }
        app.alignment_preview_seen = false;
        ++app.alignment_preview_generation;
        app.alignment_preview_poll = {};
        app.log.open(app.layout.align_log);
        app.job.start(
            build_align_command(
                AETHERSCAN_CLI_PATH, app.settings, app.layout),
            app.layout.align_log);
        app.active_job = JobKind::align;
        // Drop the previous cloud so the next frame only shows this run.
        // The actual GPU/photo teardown waits until after present(); this
        // frame's draw list still references those descriptors.
        ++app.scene_load_generation;
        app.suppress_scene_auto_load = true;
        app.pending_align_viewport_clear = true;
        set_message(app, "Aligning cameras...", theme::accent);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}

bool alignment_cache_present(const App& app) {
    if (app.layout.cache.empty()) return false;
    std::error_code error;
    return std::filesystem::exists(app.layout.working_sfm, error) ||
           std::filesystem::exists(app.layout.cache, error);
}

bool can_export_sfm(const App& app) {
    return !has_external_dataset(app) &&
           app.settings.images_dir[0] != '\0' &&
           app.settings.project_dir[0] != '\0' &&
           (app.has_sparse || alignment_cache_present(app));
}

const char* running_job_caption(const JobKind kind) {
    switch (kind) {
        case JobKind::align: return "ALIGNING";
        case JobKind::export_sfm: return "EXPORTING";
        case JobKind::train: return "TRAINING";
        case JobKind::dense: return "DENSE MVS";
        case JobKind::none: return "READY";
    }
    return "READY";
}

const char* job_state_caption(const App& app) {
    if (!app.job.running()) return app.viewer.running() ? "VIEWING" : "READY";
    if (app.job.paused()) return "PAUSED";
    return running_job_caption(app.active_job);
}

const char* pause_job_label(const JobKind kind) {
    switch (kind) {
        case JobKind::align: return "Pause Alignment";
        case JobKind::export_sfm: return "Pause Export";
        case JobKind::dense: return "Pause Dense MVS";
        default: return "Pause Training";
    }
}

const char* resume_job_label(const JobKind kind) {
    switch (kind) {
        case JobKind::align: return "Resume Alignment";
        case JobKind::export_sfm: return "Resume Export";
        case JobKind::dense: return "Resume Dense MVS";
        default: return "Resume Training";
    }
}

const char* stop_job_label(const JobKind kind) {
    switch (kind) {
        case JobKind::align: return "Stop Alignment";
        case JobKind::export_sfm: return "Stop Export";
        case JobKind::dense: return "Stop Dense MVS";
        default: return "Stop Training";
    }
}

constexpr const char* k_stop_job_tooltip =
    "Abort the running job so you can change images, video, or parameters "
    "and start again.\nProgress since the last saved artifact is discarded.\n"
    "Shift+Esc";

constexpr const char* k_pause_job_tooltip =
    "Freeze the running job without discarding progress. Resume to continue.";

constexpr const char* k_busy_change_capture_tooltip =
    "Stop the running job first to choose a different image folder or video.";

void start_export_sfm(App& app) {
    if (app.job.running()) return;
    stop_splat_view(app);
    assign_default_project_folder(app);
    if (app.settings.project_dir[0] == '\0') {
        set_message(app, "Save or choose a project file first", theme::warning);
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
        if (!app.layout.working_sfm.empty()) {
            std::error_code exists_error;
            if (std::filesystem::exists(app.layout.working_sfm, exists_error)) {
                const auto scene = load_working_sfm(
                    app.layout.working_sfm, reconstruction_images_path(app));
                aetherscan::sfm::save_asfm(scene, app.layout.sparse_asfm);
                aetherscan::sfm::export_openmvs_interface(
                    scene, app.layout.sparse_mvs);
                refresh_artifacts(app);
                set_message(
                    app, "Exported SfM to .asfm and OpenMVS .mvs",
                    theme::success);
                return;
            }
        }
        if (!app.layout.project_file.empty()) {
            std::error_code exists_error;
            if (std::filesystem::exists(app.layout.project_file, exists_error)) {
                const auto archive =
                    aetherscan::project::Archive::open(app.layout.project_file);
                const auto scene = aetherscan::project::read_sfm(archive);
                if (scene) {
                    aetherscan::sfm::save_asfm(*scene, app.layout.sparse_asfm);
                    aetherscan::sfm::export_openmvs_interface(
                        *scene, app.layout.sparse_mvs);
                    refresh_artifacts(app);
                    set_message(
                        app, "Exported SfM to .asfm and OpenMVS .mvs",
                        theme::success);
                    return;
                }
            }
        }
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

void start_splat_view(App& app) {
    if (app.job.running() || app.viewer.running() || !app.has_model) return;
    if (app.settings.images_dir[0] == '\0' ||
        app.settings.project_dir[0] == '\0')
        return;
    refresh_artifacts(app);
    if (!app.has_model) return;
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (!app.layout.preview_camera_file.empty()) {
        std::error_code preview_error;
        std::filesystem::create_directories(
            app.layout.preview_camera_file.parent_path(), preview_error);
    }
    sync_live_preview_camera(
        app, true, app.preview_raster_width, app.preview_raster_height);

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
    try {
        app.viewer.start(
            build_view_command(
                AETHERSCAN_CLI_PATH, app.settings, app.layout, handles),
            app.layout.view_log);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
    app.preview.close_export_handles();
}

void start_train(App& app, const bool smoke) {
    if (app.job.running()) return;
    stop_splat_view(app);
    assign_default_project_folder(app);
    if (app.settings.project_dir[0] == '\0') {
        set_message(app, "Save or choose a project file first", theme::warning);
        return;
    }
    refresh_artifacts(app);
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (!app.layout.preview_view_file.empty()) {
        std::error_code preview_error;
        std::filesystem::create_directories(
            app.layout.preview_view_file.parent_path(), preview_error);
    }
    app.preview_view = 0;
    app.preview_follow_view = true;
    if (has_external_dataset(app)) {
        // Keep the already-reviewed imported cameras and cloud on screen.
        // The child CLI reloads the same dataset; wiping the viewport made
        // Train look like a reset.
        if (!app.scene.has_points() && app.scene.views.empty())
            request_dataset_scene_load(app);
    } else {
        load_view_poses(app.layout.sparse_poses, app.scene);
        attach_view_image_paths(app.scene, reconstruction_images_path(app));
    }
    if (const ViewPose* pose = first_registered_view(app.scene))
        snap_orbit_to_view(app.camera, *pose);
    write_preview_view_index(app.layout, app.preview_view);
    app.view_mode = VisualizationMode::splat;
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
            << " --dataset-format colmap --splat-use-mask false"
            << " --splat --splat-strategy adc_plus --splat-iterations "
            << app.settings.iterations << " --splat-preview-interval "
            << app.settings.preview_interval
            << " --splat-preview-view 0 --splat-preview-view-file \""
            << app.layout.preview_view_file.string() << '"'
            << " --splat-preview-camera-file \""
            << app.layout.preview_camera_file.string() << '"'
            << " --splat-preview-vis-file \""
            << app.layout.preview_vis_file.string() << '"'
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
        set_message(
            app,
            has_external_dataset(app)
                ? "Training from imported cameras..."
                : (mesh_from_gaussians(app.settings)
                       ? "Training with depth/normal geometry supervision..."
                       : "Training from aligned cameras..."),
            theme::accent);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
    // The child has inherited the handles; the parent copies are no longer
    // needed and must not leak across runs.
    app.preview.close_export_handles();
}

void start_dense(App& app) {
    if (app.job.running()) return;
    stop_splat_view(app);
    assign_default_project_folder(app);
    if (app.settings.project_dir[0] == '\0') {
        set_message(app, "Save or choose a project file first", theme::warning);
        return;
    }
    if (!alignment_ready(app) && !alignment_cache_present(app)) {
        set_message(
            app,
            "Align photos or load an external camera dataset before Dense MVS",
            theme::warning);
        return;
    }
    refresh_artifacts(app);
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (error) {
        set_message(app, "Cannot create project directory", theme::danger);
        return;
    }
    if (has_external_dataset(app)) {
        if (!app.scene.has_points() && app.scene.views.empty())
            request_dataset_scene_load(app);
    }
    app.view_mode = VisualizationMode::points;
    try {
        app.monitor.begin(JobKind::dense);
        app.log.open(app.layout.dense_log);
        app.job.start(
            build_dense_command(
                AETHERSCAN_CLI_PATH, app.settings, app.layout),
            app.layout.dense_log);
        app.active_job = JobKind::dense;
        set_message(
            app,
            mesh_from_mvs(app.settings)
                ? (has_external_dataset(app)
                       ? "Building photogrammetry mesh from imported cameras..."
                       : "Building photogrammetry mesh from aligned cameras...")
                : (has_external_dataset(app)
                       ? "Dense MVS from imported cameras..."
                       : "Dense MVS from aligned cameras..."),
            theme::accent);
    } catch (const std::exception& failure) {
        set_message(app, failure.what(), theme::danger);
    }
}

void on_job_finished(App& app) {
    const int code = app.job.exit_code();
    app.monitor.mark_finished(code);
    const JobKind kind = app.active_job;
    app.active_job = JobKind::none;
    refresh_artifacts(app);

    if (code == 2) {
        set_message(
            app,
            std::string(job_name(kind)) +
                " stopped. You can change images, video, or parameters and run again.",
            theme::warning);
        return;
    }
    if (code != 0) {
        std::string reason = app.monitor.last_error();
        if (reason.empty())
            reason = "Exited with code " + std::to_string(code);
        set_message(
            app, std::string(job_name(kind)) + " failed: " + reason,
            theme::danger);
        return;
    }

    if (kind == JobKind::align) {
        app.suppress_scene_auto_load = false;
        if (app.has_sparse) {
            request_ascan_scene_load(app);
        } else {
            set_message(
                app, "Alignment finished but no SfM stage was written",
                theme::warning);
        }
        return;
    }
    if (kind == JobKind::dense) {
        std::error_code error;
        const bool has_dense =
            std::filesystem::exists(app.layout.dense_ply, error);
        if (mesh_from_mvs(app.settings)) {
            set_message(
                app,
                app.has_mesh ? "Photogrammetry mesh finished"
                             : (has_dense
                                    ? "Dense MVS finished but no mesh was written"
                                    : "Dense MVS finished but no surface was written"),
                app.has_mesh ? theme::success : theme::warning);
        } else {
            set_message(
                app,
                has_dense ? "Dense MVS finished"
                          : "Dense MVS finished but no dense.ply was written",
                has_dense ? theme::success : theme::warning);
        }
        if (app.has_mesh) {
            app.mesh.clear();
            show_mesh_view(app, true);
        } else if (app.has_sparse || has_external_dataset(app)) {
            ensure_sparse_loaded(app);
        }
        return;
    }
    if (kind == JobKind::export_sfm) {
        std::error_code error;
        const bool has_asfm =
            std::filesystem::exists(app.layout.sparse_asfm, error);
        set_message(
            app,
            has_asfm ? "Exported SfM alignment to .asfm"
                     : (app.has_mvs
                            ? "Exported SfM alignment to sparse.mvs"
                            : "Export finished but no SfM file was written"),
            has_asfm || app.has_mvs ? theme::success : theme::warning);
        return;
    }
    set_message(
        app,
        mesh_from_gaussians(app.settings)
            ? (app.has_mesh ? "Training and mesh extraction finished"
                            : "Training finished, mesh extraction produced no "
                              "surface")
            : "Training finished",
        mesh_from_gaussians(app.settings) && !app.has_mesh ? theme::warning
                                                           : theme::success);
    if (app.has_mesh) {
        app.mesh.clear();
        show_mesh_view(app, true);
    } else if (!app.smoke_mode && app.has_model) {
        start_splat_view(app);
    }
}

// ---------------------------------------------------------------------------
// UI fragments

enum class Action {
    none,
    align,
    train,
    dense,
    export_sfm,
    pause,
    resume,
    stop,
    reveal
};

int workflow_step(const App& app) {
    const bool training =
        app.job.running() && app.active_job == JobKind::train;
    const bool aligning =
        app.job.running() && app.active_job == JobKind::align;
    const bool show_mesh = app.settings.build_mesh || app.has_mesh;
    if (show_mesh &&
        ((training && app.monitor.stage() == Stage::meshing) ||
         (app.job.running() && app.active_job == JobKind::dense &&
          mesh_from_mvs(app.settings)) ||
         app.has_mesh))
        return 3;
    if (training) return 2;
    if (aligning) return 1;
    if (app.has_model) return 2;
    if (app.has_sparse || has_external_dataset(app)) return 1;
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
    if (!io.WantTextInput && io.KeyCtrl && !io.KeyShift &&
        ImGui::IsKeyPressed(ImGuiKey_N) && !busy) {
        new_project(app);
    }
    if (!io.WantTextInput && io.KeyCtrl && !io.KeyShift &&
        ImGui::IsKeyPressed(ImGuiKey_O) && !busy) {
        select_image_folder(app);
    }
    if (!io.WantTextInput && io.KeyCtrl && io.KeyShift &&
        ImGui::IsKeyPressed(ImGuiKey_O) && !busy) {
        select_project_folder(app);
    }
    if (!io.WantTextInput && io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S) &&
        !busy) {
        if (io.KeyShift) save_project_as(app);
        else save_project(app);
    }
    if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Escape) && busy) {
        if (io.KeyShift) action = Action::stop;
        else
            action = app.job.paused() ? Action::resume : Action::pause;
    }
    if (!io.WantTextInput && !io.KeyCtrl && !io.KeyAlt &&
        ImGui::IsKeyPressed(ImGuiKey_2))
        set_viewport_workspace(app, ViewportWorkspace::image_2d);
    if (!io.WantTextInput && !io.KeyCtrl && !io.KeyAlt &&
        ImGui::IsKeyPressed(ImGuiKey_3))
        set_viewport_workspace(app, ViewportWorkspace::scene_3d);

    if (!ImGui::BeginMainMenuBar()) return action;
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("New Project", "Ctrl+N", false, !busy)) {
            new_project(app);
        }
        if (ImGui::MenuItem("Select Image Folder...", "Ctrl+O", false, !busy)) {
            select_image_folder(app);
        }
        if (busy && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", k_busy_change_capture_tooltip);
        if (ImGui::MenuItem("Select Video...", nullptr, false, !busy)) {
            select_video_file(app);
        }
        if (busy && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", k_busy_change_capture_tooltip);
        if (ImGui::MenuItem("Open Project...", "Ctrl+Shift+O", false, !busy)) {
            select_project_folder(app);
        }
        if (ImGui::MenuItem("Save Project", "Ctrl+S", false, !busy)) {
            save_project(app);
        }
        if (ImGui::MenuItem("Save Project As...", "Ctrl+Shift+S", false, !busy)) {
            save_project_as(app);
        }
        ImGui::Separator();
        if (ImGui::MenuItem(
                "Load Sparse Cloud", nullptr, false,
                app.has_sparse && !app.loading_scene)) {
            app.view_mode = VisualizationMode::points;
            if (!app.layout.project_file.empty())
                request_ascan_scene_load(app);
            else
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
                false,
                !busy && app.settings.images_dir[0] != '\0' &&
                    !has_external_dataset(app)))
            action = Action::align;
        if (ImGui::MenuItem(
                mesh_from_gaussians(app.settings) ? "Train 3DGS + Mesh"
                                                  : "Train 3DGS",
                nullptr, false,
                !busy && (app.settings.images_dir[0] != '\0' ||
                          has_external_dataset(app))))
            action = Action::train;
        if (ImGui::MenuItem(
                mesh_from_mvs(app.settings) ? "Build Mesh" : "Dense MVS",
                nullptr, false, !busy && alignment_ready(app)))
            action = Action::dense;
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
        if (app.job.paused()) {
            if (ImGui::MenuItem("Resume Job", "Esc", false, busy))
                action = Action::resume;
        } else if (ImGui::MenuItem("Pause Job", "Esc", false, busy)) {
            action = Action::pause;
        }
        if (ImGui::MenuItem("Stop Job", "Shift+Esc", false, busy))
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
        if (ImGui::MenuItem(
                "3D Scene", "3",
                app.workspace == ViewportWorkspace::scene_3d))
            set_viewport_workspace(app, ViewportWorkspace::scene_3d);
        if (ImGui::MenuItem(
                "2D Image QA", "2",
                app.workspace == ViewportWorkspace::image_2d))
            set_viewport_workspace(app, ViewportWorkspace::image_2d);
        ImGui::Separator();
        if (ImGui::MenuItem(
                "Show Cameras", nullptr, app.view_options.show_views))
            set_camera_overlays(
                app.view_options, !app.view_options.show_views);
        ImGui::MenuItem("Show Ground Grid", nullptr, &app.view_options.show_grid);
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
    const bool show_mesh_crumb = app.settings.build_mesh || app.has_mesh;
    const std::array<const char*, 4> steps{
        {"Images", "Alignment", "Gaussians", "Mesh"}};
    const int step_count = show_mesh_crumb ? 4 : 3;
    ImGui::SameLine(0.F, 22.F);
    for (int i = 0; i < step_count; ++i) {
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

    const std::string project = app.layout.project_file.empty()
        ? (app.layout.root.filename().empty()
               ? std::string("Untitled Project")
               : app.layout.root.filename().string())
        : app.layout.project_file.filename().string();
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
        ImGui::TextUnformatted("3D camera");
        ImGui::Separator();
        ImGui::BulletText("LMB drag: orbit");
        ImGui::BulletText("MMB or Shift+LMB drag: pan");
        ImGui::BulletText("RMB drag: fly look");
        ImGui::BulletText("RMB + WASD/QE: fly; Shift accelerates");
        ImGui::BulletText("Mouse wheel: dolly; F: frame reconstruction");
        ImGui::BulletText("Double-click a point: orbit around that point");
        ImGui::BulletText("Double-click a camera frustum: look through it");
        ImGui::Spacing();
        ImGui::TextUnformatted("2D image QA");
        ImGui::Separator();
        ImGui::BulletText("Wheel: zoom; LMB/MMB drag: pan; double-click or F: fit");
        ImGui::BulletText("Left / Right: previous and next capture");
        ImGui::BulletText("Compare: drag the vertical handle to wipe GT vs 3DGS");
        ImGui::BulletText("2 / 3: switch 2D image QA and 3D scene");
    }
    ImGui::End();
}

ClearResultsAction draw_clear_results_modal(App& app) {
    ClearResultsAction action = ClearResultsAction::none;
    constexpr const char* popup = "Clear Reconstruction Results";
    if (app.show_clear_results) {
        ImGui::OpenPopup(popup);
        app.show_clear_results = false;
    }

    ImGui::SetNextWindowSize({470.F, 0.F}, ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(
            popup, nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        return action;

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
        const std::string root_utf8 = path_to_utf8(app.layout.root);
        ImGui::TextWrapped("%s", root_utf8.c_str());
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (theme::toolbar_button("Clear View Only", {126.F, 30.F})) {
        action = ClearResultsAction::clear_view;
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (theme::danger_button("Delete Generated Results", {190.F, 30.F})) {
        action = ClearResultsAction::delete_generated;
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", {92.F, 30.F})) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
    return action;
}

Action draw_toolbar(App& app) {
    Action action = Action::none;
    const bool busy = app.job.running();
    const bool images_ready = app.settings.images_dir[0] != '\0';
    const bool external_dataset = has_external_dataset(app);

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
            busy ? k_busy_change_capture_tooltip
                 : "Select capture image folder, or drop photos / .asfm / .ascan on the viewport")) {
        select_image_folder(app);
    }
    ImGui::SameLine();
    if (icons::labeled_button(
            "##video_file", icons::Icon::camera, "Video",
            {88.F, 32.F}, icons::ButtonStyle::normal, !busy, false,
            busy ? k_busy_change_capture_tooltip
                 : "Select a capture video. Align Photos extracts sharp frames, then runs SfM.")) {
        select_video_file(app);
    }
    ImGui::SameLine();

    // Step 1: multi-view alignment.
    const bool aligning = busy && app.active_job == JobKind::align;
    if (aligning) {
        icons::labeled_button(
            "##aligning", icons::Icon::align,
            app.job.paused() ? "Paused" : "Aligning...", {132.F, 32.F},
            icons::ButtonStyle::primary, false, true,
            "Use Stop to abort alignment so you can change images or parameters.");
    } else if (icons::labeled_button(
                   "##align", icons::Icon::align,
                   app.has_sparse ? "Re-align Photos" : "Align Photos",
                   {132.F, 32.F}, icons::ButtonStyle::primary,
                   !busy && images_ready && !external_dataset)) {
        action = Action::align;
    }
    ImGui::SameLine();

    // Step 2: Gaussian optimisation, gated on a reviewed alignment.
    const bool training = busy && app.active_job == JobKind::train;
    if (training) {
        icons::labeled_button(
            "##training", icons::Icon::train,
            app.job.paused() ? "Paused" : "Training...", {126.F, 32.F},
            icons::ButtonStyle::primary, false, true,
            "Use Stop to abort training so you can change images or parameters.");
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
        if (ImGui::IsItemHovered()) {
            if (external_dataset)
                ImGui::SetTooltip(
                    "Train directly from the imported cameras. SfM is skipped.");
            else if (!app.has_sparse)
                ImGui::SetTooltip(
                    "No alignment yet. Training will run Structure from Motion "
                    "first, then optimise Gaussians.");
            else
                ImGui::SetTooltip(
                    "Start 3DGS from the current aligned cameras and sparse cloud.");
        }
    }
    ImGui::SameLine();

    ImGui::AlignTextToFramePadding();
    ImGui::BeginDisabled(busy);
    ImGui::Checkbox("Build Mesh", &app.settings.build_mesh);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        const char* mesh_tip = busy
            ? "Stop the running job first to change reconstruction options."
            : (app.settings.mesh_source == 1
                   ? "Extract a surface with dense multi-view stereo after alignment.\n"
                     "Choose Photogrammetry or From Gaussians in the Mesh panel."
                   : "Extract a surface after 3DGS training.\n"
                     "From Gaussians enables depth-normal and multi-view geometry\n"
                     "losses during optimisation. Switch to Photogrammetry in the\n"
                     "Mesh panel for a dense MVS surface.");
        ImGui::SetTooltip("%s", mesh_tip);
    }
    ImGui::SameLine(0.F, 14.F);

    if (icons::labeled_button(
            "##open_output", icons::Icon::output, "Open Output",
            {116.F, 32.F}, icons::ButtonStyle::normal,
            app.layout.root.has_filename()))
        action = Action::reveal;
    if (busy) {
        ImGui::SameLine();
        if (app.job.paused()) {
            if (icons::labeled_button(
                    "##resume", icons::Icon::play, "Resume", {96.F, 32.F},
                    icons::ButtonStyle::primary, true, false,
                    "Continue the paused reconstruction."))
                action = Action::resume;
        } else if (icons::labeled_button(
                       "##pause", icons::Icon::pause, "Pause", {96.F, 32.F},
                       icons::ButtonStyle::normal, true, false,
                       k_pause_job_tooltip)) {
            action = Action::pause;
        }
        ImGui::SameLine();
        if (icons::labeled_button(
                "##stop", icons::Icon::stop, "Stop", {88.F, 32.F},
                icons::ButtonStyle::danger, true, false, k_stop_job_tooltip))
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
    const bool external_dataset = has_external_dataset(app);

    theme::section_header("SCENE");
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {7.F, 7.F});
    ImGui::SetCursorPosX(10.F);
    const std::string project_label = app.layout.project_file.empty()
        ? (app.layout.root.filename().empty()
               ? std::string("Untitled Project")
               : app.layout.root.filename().string())
        : app.layout.project_file.filename().string();
    const bool has_cloud = app.has_sparse || app.scene.has_points();
    const bool has_gaussians = app.has_model;
    const bool has_mesh = app.has_mesh;
    if (ImGui::TreeNodeEx(
            project_label.c_str(),
            ImGuiTreeNodeFlags_DefaultOpen |
                ImGuiTreeNodeFlags_SpanAvailWidth)) {
        const auto object_row = [](const char* label, const bool selected) {
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf |
                                       ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                       ImGuiTreeNodeFlags_SpanAvailWidth;
            if (selected) flags |= ImGuiTreeNodeFlags_Selected;
            ImGui::TreeNodeEx(label, flags);
            return ImGui::IsItemClicked();
        };
        if (!has_cloud && !has_gaussians && !has_mesh) {
            ImGui::SetCursorPosX(22.F);
            ImGui::PushTextWrapPos(wrap);
            theme::caption(
                external_dataset
                    ? "External camera dataset selected. Train 3DGS to use it."
                    : "Align photos to add a sparse cloud.");
            ImGui::PopTextWrapPos();
        } else {
            if (has_cloud &&
                object_row(
                    "Sparse Cloud",
                    app.view_mode == VisualizationMode::points &&
                        !live_preview_active(app))) {
                app.view_mode = VisualizationMode::points;
                if (!alignment_job_running(app)) {
                    app.suppress_scene_auto_load = false;
                    if (app.has_sparse) ensure_sparse_loaded(app);
                }
                write_preview_vis(app);
                sync_live_preview_camera(
                    app, true, app.preview_raster_width,
                    app.preview_raster_height);
            }
            if (has_gaussians &&
                object_row(
                    "Gaussians",
                    app.view_mode == VisualizationMode::splat ||
                        app.view_mode == VisualizationMode::rings ||
                        live_preview_active(app))) {
                set_visualization_mode(app, VisualizationMode::splat);
            }
            if (has_mesh &&
                object_row(
                    "Mesh", app.view_mode == VisualizationMode::mesh)) {
                set_visualization_mode(app, VisualizationMode::mesh);
            }
        }
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
        app.settings.images_dir[0] != '\0' || external_dataset
            ? StepState::done
            : StepState::pending,
        nullptr);
    draw_step(
        "02", "Align cameras",
        aligning ? StepState::active
                 : (app.has_sparse || external_dataset
                        ? StepState::done
                        : StepState::pending),
        aligning
            ? stage_name(stage)
            : (external_dataset ? "Imported cameras (SfM skipped)" : nullptr));
    draw_step(
        "03", "Review sparse cloud",
        app.scene.has_points()
            ? StepState::done
            : ((external_dataset || app.has_sparse)
                   ? StepState::active
                   : StepState::pending),
        app.scene.has_points()
            ? nullptr
            : (aligning
                   ? "Recomputing"
                   : (app.loading_scene
                          ? "Loading"
                          : (external_dataset ? "Imported cameras"
                                              : (app.has_sparse ? "On disk"
                                                                : nullptr)))));
    draw_step(
        "04", "Optimise Gaussians",
        training && stage != Stage::meshing
            ? StepState::active
            : (app.has_model ? StepState::done : StepState::pending),
        mesh_from_gaussians(app.settings) ? "Geometry constraints on" : nullptr);
    if (app.settings.build_mesh) {
        const bool extracting =
            (mesh_from_gaussians(app.settings) && training &&
             stage == Stage::meshing) ||
            (mesh_from_mvs(app.settings) && busy &&
             app.active_job == JobKind::dense);
        draw_step(
            "05", "Extract mesh",
            extracting ? StepState::active
                       : (app.has_mesh ? StepState::done : StepState::pending),
            extracting ? stage_name(stage)
                       : (mesh_from_mvs(app.settings) ? "Photogrammetry"
                                                      : "From Gaussians"));
    }

    ImGui::Dummy({0, 8.F});
    theme::section_header("SOURCE");
    ImGui::Indent(14.F);
    theme::caption(is_video_source(app.settings) ? "VIDEO" : "IMAGES");
    ImGui::Spacing();
    ImGui::PushTextWrapPos(wrap);
    ImGui::TextUnformatted(
        app.settings.images_dir[0] != '\0' ? app.settings.images_dir.data()
                                          : "(not selected)");
    ImGui::PopTextWrapPos();
    if (busy) {
        ImGui::Dummy({0, 4.F});
        ImGui::PushTextWrapPos(wrap);
        theme::caption(
            "Stop the running job to choose a different capture.");
        ImGui::PopTextWrapPos();
    }
    if (is_video_source(app.settings)) {
        ImGui::Dummy({0, 4.F});
        theme::caption("EXTRACTED FRAMES");
        ImGui::Spacing();
        ImGui::PushTextWrapPos(wrap);
        const std::string frames_utf8 =
            path_to_utf8(reconstruction_images_path(app));
        ImGui::TextUnformatted(frames_utf8.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::Dummy({0, 6.F});
    theme::caption("PROJECT");
    ImGui::Spacing();
    ImGui::PushTextWrapPos(wrap);
    ImGui::TextUnformatted(
        app.settings.project_dir[0] != '\0' ? app.settings.project_dir.data()
                                           : "(not selected)");
    ImGui::PopTextWrapPos();
    if (external_dataset) {
        ImGui::Dummy({0, 6.F});
        theme::caption("EXTERNAL DATASET");
        ImGui::Spacing();
        ImGui::PushTextWrapPos(wrap);
        ImGui::TextUnformatted(app.settings.dataset_source.data());
        ImGui::PopTextWrapPos();
    }
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
    const bool aligning = alignment_job_running(app);
    if (app.view_mode == VisualizationMode::mesh)
        ensure_mesh_loaded(app);
    if (app.view_mode == VisualizationMode::rings)
        ensure_gaussian_scene(app);
    // Only fill an empty viewport here. Replacing an explicitly loaded cloud
    // would fight the user; the QA workspace enforces dataset cameras itself.
    else if (has_external_dataset(app) && app.scene.views.empty() &&
             !app.scene.has_points())
        ensure_dataset_scene_loaded(app);
    else if (app.view_mode != VisualizationMode::mesh || !app.loading_scene)
        ensure_sparse_loaded(app);
    app.view_options.draw_rings = app.view_mode == VisualizationMode::rings;
    app.view_options.draw_mesh = app.view_mode == VisualizationMode::mesh;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilledMultiColor(
        min, max, IM_COL32(9, 11, 16, 255), IM_COL32(9, 11, 16, 255),
        IM_COL32(18, 22, 32, 255), IM_COL32(18, 22, 32, 255));
    const bool gpu_mesh =
        app.view_mode == VisualizationMode::mesh &&
        update_gpu_mesh_preview(app, min, max);
    if (gpu_mesh) {
        app.view_options.draw_mesh = false;
#if defined(AETHERSCAN_HAS_TEXTURE)
        if (app.mesh_preview.descriptor) {
            draw->AddImage(
                reinterpret_cast<ImTextureID>(app.mesh_preview.descriptor),
                min, max);
        }
#endif
    }

    ImGui::SetCursorScreenPos(min);
    ImGui::SetNextItemAllowOverlap();
    ImGui::InvisibleButton(
        "##sparse_view",
        {std::max(1.F, max.x - min.x), std::max(1.F, max.y - min.y)},
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
            ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered =
        ImGui::IsItemHovered() &&
        !view_mode_rail_contains(min, ImGui::GetIO().MousePos);

    if (!app.loading_scene && !app.scene.has_points() && !app.mesh.has() &&
        (aligning || !app.has_sparse || app.suppress_scene_auto_load)) {
        draw_empty_viewport(
            draw, min, max,
            aligning ? "Aligning photos..."
                     : app.settings.images_dir[0] != '\0'
                    ? "Ready to align cameras"
                    : "Drop photos, a video, or a reconstruction",
            aligning
                ? "Cameras and points appear as soon as geometry is available"
                : app.settings.images_dir[0] != '\0'
                ? "Run Align Photos, or drop a different folder, video, .asfm, or .ascan"
                : "Drop an image folder, photos, a video, .asfm, or .ascan onto this view");
    }

    ViewOptions draw_options = app.view_options;
    if (app.view_mode == VisualizationMode::mesh) draw_options.show_cloud = false;
    const SceneDrawStats stats = app.renderer.draw(
        draw, min, max, app.scene, app.camera, draw_options, hovered,
        app.photos.ids(), app.photos.size(),
        app.view_mode == VisualizationMode::mesh ? &app.mesh : nullptr);

    const bool gizmo_captures =
        draw_viewport_gizmo(app.gizmo, app.camera, min, max);
    const bool viewport_input = hovered && !gizmo_captures;
    const bool frame_key = viewport_input &&
                           !ImGui::GetIO().WantTextInput &&
                           ImGui::IsKeyPressed(ImGuiKey_F);
    if (frame_key) {
        if (app.view_mode == VisualizationMode::mesh && app.mesh.has())
            app.camera.frame(app.mesh.centroid, app.mesh.radius);
        else
            app.camera.frame(app.scene);
    }
    if (!handle_viewport_double_click(app, viewport_input, stats, min, max))
        update_orbit_camera(app.camera, viewport_input, app.scene.radius);

    const char* overlay = app.settings.images_dir[0] != '\0' ? "NO ALIGNMENT"
                                                            : "NO IMAGES";
    ImVec4 overlay_dot = theme::inactive;
    if (aligning) {
        overlay = app.alignment_preview_seen ? "ALIGNING / PARTIAL RESULT" : "ALIGNING";
        overlay_dot = theme::accent;
    } else if (waiting_for_train_preview(app)) {
        overlay = "PREPARING 3DGS";
        overlay_dot = theme::warning;
    } else if (app.job.running() && app.active_job == JobKind::dense) {
        overlay = "DENSE MVS";
        overlay_dot = theme::accent;
    } else if (app.loading_scene) {
        overlay = "LOADING";
        overlay_dot = theme::warning;
    } else if (app.alignment_preview_seen) {
        overlay = "PARTIAL ALIGNMENT / NOT FINAL";
        overlay_dot = theme::warning;
    } else if (app.view_mode == VisualizationMode::mesh) {
        overlay = app.mesh.has()
            ? (gpu_mesh ? "MESH (GPU)" : "MESH")
            : "NO MESH";
        overlay_dot = app.mesh.has() ? theme::accent : theme::inactive;
    } else if (app.view_mode == VisualizationMode::rings) {
        overlay = app.scene.has_gaussians() ? "GAUSSIAN RINGS" : "NO GAUSSIANS";
        overlay_dot = app.scene.has_gaussians() ? theme::accent : theme::inactive;
    } else if (app.scene.has_points()) {
        overlay = app.scene_source.empty() ? "SPARSE POINT CLOUD"
                                           : app.scene_source.c_str();
        overlay_dot = theme::accent;
    } else if (app.has_sparse && !app.suppress_scene_auto_load) {
        overlay = "CLOUD READY";
    }
    draw_viewport_overlay(draw, min, overlay, overlay_dot);

    if (app.view_mode == VisualizationMode::mesh && app.mesh.has()) {
        char readout[192];
        if (gpu_mesh) {
            std::snprintf(
                readout, sizeof(readout),
                "GPU raster  |  %s vertices  |  %s faces  |  %zu / %zu cameras",
                format_count(app.mesh.vertices.size()).c_str(),
                format_count(app.mesh.faces.size()).c_str(), stats.drawn_views,
                app.scene.registered_views);
        } else if (stats.drawn_faces == 0) {
            std::snprintf(
                readout, sizeof(readout),
                "Mesh loaded (%s faces) but none are in view — press F to frame",
                format_count(app.mesh.faces.size()).c_str());
        } else {
            std::snprintf(
                readout, sizeof(readout),
                "%s faces drawn  |  %s vertices  |  %s faces  |  %zu / %zu cameras",
                format_count(stats.drawn_faces).c_str(),
                format_count(app.mesh.vertices.size()).c_str(),
                format_count(app.mesh.faces.size()).c_str(), stats.drawn_views,
                app.scene.registered_views);
        }
        draw->AddText(
            {min.x + 16.F, max.y - 42.F}, theme::u32(theme::text_muted), readout);
    } else if (app.scene.has_points()) {
        char readout[192];
        std::snprintf(
            readout, sizeof(readout),
            app.view_mode == VisualizationMode::rings &&
                    app.scene.has_gaussians()
                ? "%s rings drawn  |  %zu / %zu cameras shown  |  %s gaussians"
                : "%s pts drawn  |  %zu / %zu cameras shown  |  %s pts total",
            format_count(stats.drawn_points).c_str(), stats.drawn_views,
            app.scene.registered_views,
            format_count(app.scene.points.size()).c_str());
        draw->AddText(
            {min.x + 16.F, max.y - 42.F}, theme::u32(theme::text_muted), readout);
    } else if (!app.loading_scene) {
        const char* hint = aligning
            ? "Waiting for cameras from this alignment"
            : app.suppress_scene_auto_load
            ? "Previous result cleared"
            : app.has_sparse
            ? "Reading sparse.ply..."
            : "Drop a photo folder, .asfm, or .ascan here";
        draw->AddText(
            {min.x + 16.F, max.y - 42.F}, theme::u32(theme::text_faint), hint);
    }
    draw->AddText(
        {min.x + 16.F, max.y - 24.F}, theme::u32(theme::text_faint),
        "LMB orbit  |  MMB pan  |  RMB + WASD/QE fly  |  wheel dolly  |  F frame  |  double-click focus");

    if (!gizmo_captures && stats.hovered_view >= 0 &&
        static_cast<std::size_t>(stats.hovered_view) < app.scene.views.size()) {
        const ViewPose& pose = app.scene.views[stats.hovered_view];
        ImGui::SetTooltip(
            "%s\n%u x %u  |  f %.1f px\n%zu observations  |  p95 %.2f px\n"
            "Double-click to look through this camera",
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
    ImGui::SetNextItemAllowOverlap();
    ImGui::InvisibleButton(
        "##training_view",
        {std::max(1.F, max.x - min.x), std::max(1.F, max.y - min.y)},
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
            ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered =
        ImGui::IsItemHovered() &&
        !view_mode_rail_contains(min, ImGui::GetIO().MousePos);

    const bool training = app.job.running() && app.active_job == JobKind::train;
    const bool viewing = app.viewer.running();
    const bool live = training || viewing;
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

    const char* controls = live
        ? "LMB orbit  |  MMB pan  |  RMB + WASD fly  |  arrows snap capture  |  double-click focus"
        : (app.has_model
               ? "Select a visualization mode to start the live preview"
               : "Train 3DGS to move this camera");

    if (!has_frame) {
        draw_empty_viewport(
            draw, min, max,
            live ? "Waiting for the first rendered view..."
                 : "No live preview",
            live ? "Orbit the view; the first frame uses this camera"
                 : (app.has_model
                        ? "Select Points, Splat, or Rings to render the model"
                        : "Run Train 3DGS to stream the optimiser output"));
    } else {
        draw->AddImage(
            reinterpret_cast<ImTextureID>(app.preview.display.descriptor),
            min, max);
    }

    ViewOptions overlay = app.view_options;
    overlay.show_cloud = false;
    overlay.draw_rings = false;
    const SceneDrawStats overlay_stats = app.renderer.draw(
        draw, min, max, app.scene, app.camera, overlay, hovered,
        app.photos.ids(), app.photos.size());

    const bool gizmo_captures =
        draw_viewport_gizmo(app.gizmo, app.camera, min, max);
    const bool viewport_input = hovered && !gizmo_captures;
    const bool used_double_click = handle_viewport_double_click(
        app, viewport_input, overlay_stats, min, max);
    if (!used_double_click)
        update_orbit_camera(app.camera, viewport_input, app.scene.radius);
    if (!used_double_click &&
        (app.camera.interacting ||
         (viewport_input && ImGui::GetIO().MouseWheel != 0.F)))
        app.preview_follow_view = false;
    handle_preview_view_input(app, viewport_input);
    std::uint32_t raster_w = app.preview_raster_width;
    std::uint32_t raster_h = app.preview_raster_height;
    fit_preview_raster(max.x - min.x, max.y - min.y, raster_w, raster_h);
    sync_live_preview_camera(app, false, raster_w, raster_h);

    const char* overlay_label = "IDLE";
    if (has_frame) {
        if (app.view_mode == VisualizationMode::points)
            overlay_label = "GAUSSIAN CENTRES";
        else if (app.view_mode == VisualizationMode::rings)
            overlay_label = "GAUSSIAN RINGS";
        else if (training)
            overlay_label = "LIVE TRAINING PREVIEW";
        else
            overlay_label = viewing ? "LIVE SPLAT VIEW" : "LAST TRAINING FRAME";
    } else if (live) {
        overlay_label = training ? "TRAINING" : "VIEWING";
    }
    draw_viewport_overlay(
        draw, min, overlay_label,
        has_frame
            ? (live ? theme::success : theme::inactive)
            : (live ? theme::warning : theme::inactive));

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

    if (!gizmo_captures && overlay_stats.hovered_view >= 0 &&
        static_cast<std::size_t>(overlay_stats.hovered_view) <
            app.scene.views.size()) {
        const ViewPose& pose = app.scene.views[overlay_stats.hovered_view];
        ImGui::SetTooltip(
            "%s\n%u x %u  |  f %.1f px\n%zu observations  |  p95 %.2f px\n"
            "Double-click to look through this camera",
            pose.name.c_str(), pose.width, pose.height, pose.fx,
            pose.observations, pose.reprojection_p95);
    }
}

void draw_viewport_panel(App& app) {
    if (!app.show_viewport) {
        app.viewport_bounds_valid = false;
        return;
    }
    constexpr ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar |
        ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.F, 0.F));
    const bool open = ImGui::Begin("Viewport", &app.show_viewport, flags);
    ImGui::PopStyleVar();
    if (!open) {
        app.viewport_bounds_valid = false;
        ImGui::End();
        return;
    }

    if (ImGuiDockNode* node = ImGui::GetWindowDockNode())
        node->LocalFlags |= ImGuiDockNodeFlags_AutoHideTabBar;

    app.viewport_min = ImGui::GetWindowPos();
    const ImVec2 viewport_size = ImGui::GetWindowSize();
    app.viewport_max = {
        app.viewport_min.x + viewport_size.x,
        app.viewport_min.y + viewport_size.y};
    app.viewport_bounds_valid = true;

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

    draw_workspace_toggle(app, header_origin);

    const char* state = app.workspace == ViewportWorkspace::image_2d
        ? "IMAGE QA"
        : job_state_caption(app);
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
    if (app.workspace == ViewportWorkspace::image_2d) {
        refresh_image_qa_folder(
            app.image_qa, reconstruction_images_path(app));
        const int previous = app.image_qa.selected;
        ImageQaDrawInput input;
        input.scene = &app.scene;
        input.photos = &app.photos;
        input.render = app.preview.display.descriptor
            ? reinterpret_cast<ImTextureID>(app.preview.display.descriptor)
            : ImTextureID{};
        input.has_render = app.preview.display.descriptor &&
                           qa_capture_frame_ready(app);
        input.render_live = live_preview_active(app);
        input.has_model = app.has_model;
        input.external_alignment = has_external_dataset(app);
        draw_image_qa(
            app.image_qa, app.image_qa_session, input, view_min, view_max);
        sync_qa_selection_to_preview(app, previous);
        ensure_dataset_scene_loaded(app);
        ensure_qa_preview(app);
    } else if (app.view_mode == VisualizationMode::mesh) {
        draw_sparse_tab(app, view_min, view_max);
        draw_view_mode_rail(app, view_min);
        draw_scene_toggle_rail(app, view_min);
    } else if (live_preview_active(app) && !waiting_for_train_preview(app)) {
        draw_training_tab(app, view_min, view_max);
        draw_view_mode_rail(app, view_min);
        draw_scene_toggle_rail(app, view_min);
    } else {
        draw_sparse_tab(app, view_min, view_max);
        draw_view_mode_rail(app, view_min);
        draw_scene_toggle_rail(app, view_min);
    }
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
        if (busy) {
            ImGui::PushTextWrapPos(0.F);
            ImGui::TextColored(
                theme::warning,
                "A reconstruction is running. Stop it to change images, "
                "video, or parameters.");
            ImGui::PopTextWrapPos();
            ImGui::Spacing();
        }
        ImGui::BeginDisabled(busy);
        theme::caption("Image source");
        ImGui::SetNextItemWidth(-138.F);
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
        if (ImGui::IsItemDeactivatedAfterEdit() &&
            is_video_source(app.settings))
            app.settings.video_frames_dir.fill('\0');
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button("Folder##pick_images", {58.F, 0})) {
            select_image_folder(app);
        }
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button("Video##pick_video", {46.F, 0})) {
            select_video_file(app);
        }
        if (is_video_source(app.settings)) {
            ImGui::Spacing();
            theme::caption("Video extraction");
            ImGui::PushTextWrapPos(0.F);
            theme::caption(
                "Align Photos extracts the sharpest stills, then runs SfM.");
            ImGui::PopTextWrapPos();
            const auto ffmpeg = aetherscan::io::locate_ffmpeg();
            if (ffmpeg.empty()) {
                theme::metric_coloured("ffmpeg", "Not found", theme::danger);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Install ffmpeg and add it to PATH. AetherScan also "
                        "looks next to the app and in common install folders.");
            } else {
                theme::metric_coloured("ffmpeg", "Ready", theme::success);
                if (ImGui::IsItemHovered()) {
                    const std::string located = path_to_utf8(ffmpeg);
                    ImGui::SetTooltip("%s", located.c_str());
                }
            }
            theme::caption("Target FPS");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputFloat("##video_fps", &app.settings.video_fps, 0.5F, 1.F, "%.2f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Kept frames per second of source time.\n"
                    "2 FPS is a good default for handheld scans.");
            theme::caption("Max frames (0 = no cap)");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputInt("##video_max_frames", &app.settings.video_max_frames);
            const std::string frames_utf8 =
                path_to_utf8(reconstruction_images_path(app));
            theme::metric("Will write", frames_utf8.c_str());
            if (ImGui::TreeNodeEx("Advanced##video", ImGuiTreeNodeFlags_SpanAvailWidth)) {
                theme::caption("Sharpness window");
                ImGui::SetNextItemWidth(-1.F);
                ImGui::InputInt("##video_sharp_window", &app.settings.video_sharp_window);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Keep the sharpest of N consecutive candidates.\n"
                        "1 disables blur selection. 3 is the default.");
                theme::caption("JPEG quality");
                ImGui::SetNextItemWidth(-1.F);
                ImGui::InputInt("##video_quality", &app.settings.video_quality);
                theme::caption("Scale");
                ImGui::SetNextItemWidth(-1.F);
                ImGui::InputFloat("##video_scale", &app.settings.video_scale, 0.1F, 0.25F, "%.2f");
                theme::caption("Rotate");
                ImGui::SetNextItemWidth(-1.F);
                const char* rotations[] = {"0°", "90°", "180°", "270°"};
                int rotate_choice = std::clamp(app.settings.video_rotate / 90, 0, 3);
                if (ImGui::Combo("##video_rotate", &rotate_choice, rotations, 4))
                    app.settings.video_rotate = rotate_choice * 90;
                theme::caption("Frames folder (optional)");
                ImGui::SetNextItemWidth(-30.F);
                ImGui::InputText(
                    "##video_frames_dir", app.settings.video_frames_dir.data(),
                    app.settings.video_frames_dir.size());
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Empty uses <video_stem>/images next to the file.");
                ImGui::SameLine(0.F, 4.F);
                if (ImGui::Button("...##pick_video_frames", {24.F, 0})) {
                    pick_folder(
                        L"Select extracted frames folder",
                        app.settings.video_frames_dir);
                }
                ImGui::TreePop();
            }
        }
        theme::caption("Project file");
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
        ImGui::EndDisabled();
        if (app.project_writer_version != 0) {
            ImGui::Spacing();
            const std::string format =
                "ascan v" + std::to_string(app.project_writer_version);
            theme::metric("Project format", format.c_str());
        }
        ImGui::Spacing();
        if (theme::danger_button(
                "Clear Reconstruction Results...", {-1.F, 28.F},
                !busy && !app.loading_scene &&
                    has_reconstruction_result(app)))
            app.show_clear_results = true;

        ImGui::BeginDisabled(busy);
        theme::caption("External SfM dataset (optional)");
        ImGui::SetNextItemWidth(-138.F);
        if (ImGui::InputText(
                "##dataset_source", app.settings.dataset_source.data(),
                app.settings.dataset_source.size())) {
        }
        if (ImGui::IsItemDeactivatedAfterEdit())
            apply_external_dataset_selection(app);
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button("Folder##pick_dataset_folder", {58.F, 0.F}) &&
            pick_folder(
                L"Select external SfM dataset folder",
                app.settings.dataset_source)) {
            apply_external_dataset_selection(app);
        }
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button("File##pick_dataset_file", {46.F, 0.F}) &&
            pick_dataset_file(
                L"Select external camera dataset file",
                app.settings.dataset_source)) {
            apply_external_dataset_selection(app);
        }
        if (has_external_dataset(app)) {
            theme::caption(
                "Imported cameras replace Align Photos. Review the cloud, then "
                "run Train 3DGS or Dense MVS.");
            theme::caption("Dataset format");
            ImGui::SetNextItemWidth(-1.F);
            const char* formats[] = {
                "Auto detect", "COLMAP", "RealityCapture", "OpenMVS"};
            ImGui::Combo(
                "##dataset_format", &app.settings.dataset_format, formats, 4);

            theme::caption("Initial point cloud (optional)");
            ImGui::SetNextItemWidth(-82.F);
            if (ImGui::InputText(
                    "##dataset_initial_cloud",
                    app.settings.dataset_initial_cloud.data(),
                    app.settings.dataset_initial_cloud.size()))
                clear_loaded_result(app);
            ImGui::SameLine(0.F, 4.F);
            if (ImGui::Button("File##pick_initial_cloud", {46.F, 0.F}) &&
                pick_point_cloud_file(
                    L"Select initial point cloud",
                    app.settings.dataset_initial_cloud))
                clear_loaded_result(app);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Optional dense PLY initializer. COLMAP/OpenMVS sparse points "
                    "are used automatically when available.");
            if (ImGui::Button("Clear external dataset", {-1.F, 26.F})) {
                app.settings.dataset_source.fill('\0');
                app.settings.dataset_initial_cloud.fill('\0');
                app.settings.dataset_format = 0;
                clear_loaded_result(app);
                refresh_artifacts(app);
                if (app.has_sparse) request_ascan_scene_load(app);
            }
        }
        ImGui::EndDisabled();
        ImGui::BeginDisabled(busy);
        theme::caption("Trained splat model (optional)");
        ImGui::SetNextItemWidth(-138.F);
        if (ImGui::InputText(
                "##splat_model_source", app.settings.splat_model_source.data(),
                app.settings.splat_model_source.size())) {
            clear_loaded_result(app);
            refresh_artifacts(app);
        }
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button("File##pick_splat_model", {46.F, 0.F}) &&
            pick_splat_model_file(
                L"Select trained Gaussian splat model",
                app.settings.splat_model_source)) {
            clear_loaded_result(app);
            refresh_artifacts(app);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Load an existing PLY, SOG, SPZ, or GLB model for preview. "
                "When set, it takes precedence over generated sidecars.");
        ImGui::EndDisabled();
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
        theme::caption("Camera model");
        ImGui::SetNextItemWidth(-1.F);
        const char* camera_models[] = {"Auto", "Pinhole", "OpenCV Fisheye"};
        int camera_choice = app.settings.camera_model == 2 ? 0 : app.settings.camera_model + 1;
        if (ImGui::Combo("##camera_model", &camera_choice, camera_models, 3))
            app.settings.camera_model = camera_choice == 0 ? 2 : camera_choice - 1;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Auto compares matched image geometry and EXIF lens hints.\n"
                              "Ambiguous results use Pinhole. You can override the model.");
        std::string result_model;
        for (const auto& view : app.scene.views) {
            if (!view.registered) continue;
            if (result_model.empty()) result_model = view.camera_model;
            else if (result_model != view.camera_model) { result_model = "Mixed"; break; }
        }
        theme::metric("Result camera", result_model.empty() ? "Pending alignment" : result_model.c_str());
        theme::caption("Max features per image");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::InputInt("##max_features", &app.settings.max_features, 1000, 5000);
        ImGui::Checkbox("Reuse cached alignment", &app.settings.reuse_cache);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Write a project .cache folder so later Align/Train can\n"
                "reuse extracted features. Off by default: skip that folder.\n"
                "Align/Train do not write .ascan or .asfm unless you Save\n"
                "Project or Export SfM.");
        ImGui::EndDisabled();

        if (theme::toolbar_button(
                "Export SfM Alignment", {-1.F, 28.F},
                !busy && can_export_sfm(app)))
            action = Action::export_sfm;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Write the SfM stage to a standalone .asfm file and an\n"
                "OpenMVS .mvs sidecar. Align/Train no longer write these.");
        if (app.has_asfm) {
            ImGui::Spacing();
            const std::string asfm_name =
                app.layout.sparse_asfm.filename().string();
            theme::metric_coloured(
                "SfM scene", asfm_name.c_str(), theme::success);
        }
        if (app.has_mvs) {
            ImGui::Spacing();
            const std::string mvs_name =
                app.layout.sparse_mvs.filename().string();
            theme::metric_coloured(
                "OpenMVS file", mvs_name.c_str(), theme::success);
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
        theme::caption("Splat output format");
        ImGui::SetNextItemWidth(-1.F);
        const char* splat_formats[] = {
            "Auto (PLY)", "PLY", "SOG", "SPZ", "GLB"};
        ImGui::BeginDisabled(busy);
        ImGui::Combo(
            "##splat_format", &app.settings.splat_format, splat_formats, 5);
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Format of the trained sidecar: PLY, PlayCanvas SOG, Niantic "
                "SPZ, or Khronos KHR_gaussian_splatting GLB. Auto writes PLY.");
        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader(
            "Mesh", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        ImGui::BeginDisabled(busy);
        ImGui::Checkbox("Build mesh", &app.settings.build_mesh);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip(
                "Include a surface mesh in the reconstruction.\n"
                "Choose From Gaussians or Photogrammetry below.");
        if (app.settings.build_mesh) {
            ImGui::Spacing();
            theme::caption("Method");
            if (ImGui::RadioButton(
                    "From Gaussians", app.settings.mesh_source == 0))
                app.settings.mesh_source = 0;
            ImGui::PushTextWrapPos(0.F);
            theme::caption(
                "Train 3DGS with depth and normal losses, then extract a "
                "surface from the Gaussians.");
            ImGui::PopTextWrapPos();
            ImGui::Spacing();
            if (ImGui::RadioButton(
                    "Photogrammetry", app.settings.mesh_source == 1)) {
                app.settings.mesh_source = 1;
                if (app.settings.mesh_method == 3)
                    app.settings.mesh_method = 0;
            }
            ImGui::PushTextWrapPos(0.F);
            theme::caption(
                "Dense multi-view stereo from aligned cameras, then fuse a "
                "mesh. 3DGS training stays appearance-only.");
            ImGui::PopTextWrapPos();
            ImGui::Spacing();
            theme::caption("Surface");
            ImGui::SetNextItemWidth(-1.F);
            if (mesh_from_gaussians(app.settings)) {
                const char* methods[] = {"Auto", "TSDF", "Delaunay", "PAM"};
                ImGui::Combo(
                    "##mesh_method", &app.settings.mesh_method, methods, 4);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "How depth is fused into triangles.\n"
                        "PAM is GaussianWrapping occupancy meshing.");
            } else {
                if (app.settings.mesh_method == 3)
                    app.settings.mesh_method = 0;
                const char* methods[] = {"Auto", "TSDF", "Delaunay"};
                ImGui::Combo(
                    "##mesh_method", &app.settings.mesh_method, methods, 3);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "How MVS depth maps are fused into triangles.");
            }
            if (mesh_from_gaussians(app.settings) &&
                ImGui::TreeNodeEx(
                    "Geometry training", ImGuiTreeNodeFlags_SpanAvailWidth)) {
                theme::caption("Depth-normal weight");
                ImGui::SetNextItemWidth(-1.F);
                ImGui::DragFloat(
                    "##depth_normal", &app.settings.depth_normal_weight,
                    0.005F, 0.F, 1.F, "%.3f");
                theme::caption("Multi-view geometry weight");
                ImGui::SetNextItemWidth(-1.F);
                ImGui::DragFloat(
                    "##mv_geo", &app.settings.multi_view_geo_weight, 0.005F,
                    0.F, 1.F, "%.3f");
                theme::caption("Multi-view NCC weight");
                ImGui::SetNextItemWidth(-1.F);
                ImGui::DragFloat(
                    "##mv_ncc", &app.settings.multi_view_ncc_weight, 0.01F,
                    0.F, 2.F, "%.2f");
                theme::caption("Geometry loss start iteration");
                ImGui::SetNextItemWidth(-1.F);
                ImGui::InputInt(
                    "##geo_from", &app.settings.geometry_from_iter, 500,
                    2000);
                ImGui::TreePop();
            }
        } else {
            ImGui::PushTextWrapPos(0.F);
            theme::caption(
                "Appearance-only. Train 3DGS without extracting a surface.");
            ImGui::PopTextWrapPos();
        }
        ImGui::EndDisabled();
        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        if (ImGui::Checkbox("Show cameras", &app.view_options.show_views))
            app.view_options.show_camera_photos = app.view_options.show_views;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Wire frustums and capture photos in the 3D / training view.");
        ImGui::Checkbox("Show trajectory", &app.view_options.show_trajectory);
        ImGui::Checkbox("Show ground grid", &app.view_options.show_grid);
        ImGui::Spacing();
    }

    if (app.workspace == ViewportWorkspace::image_2d &&
        ImGui::CollapsingHeader("Image QA", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        const int count = image_qa_count(app.image_qa, app.scene);
        if (app.image_qa.selected >= 0 && app.image_qa.selected < count &&
            static_cast<std::size_t>(app.image_qa.selected) <
                app.scene.views.size()) {
            const ViewPose& pose =
                app.scene.views[static_cast<std::size_t>(app.image_qa.selected)];
            theme::metric("Capture", pose.name.c_str());
            char res[32];
            std::snprintf(
                res, sizeof(res), "%u × %u", pose.width, pose.height);
            theme::metric("Resolution", res);
            theme::metric(
                "Registered", pose.registered ? "yes" : "no");
            theme::metric(
                "Observations", std::to_string(pose.observations).c_str());
            theme::metric(
                "Features", std::to_string(pose.features.size()).c_str());
            theme::metric(
                "Triangulated",
                std::to_string(pose.triangulated_features).c_str());
        } else if (count > 0) {
            theme::metric("Images", std::to_string(count).c_str());
        } else {
            theme::caption("Align photos to inspect cameras and features.");
        }
        ImGui::Spacing();
        ImGui::Checkbox("Show triangulated", &app.image_qa.show_triangulated);
        ImGui::Checkbox("Show untracked keypoints", &app.image_qa.show_untracked);
        if (app.image_qa_session.metrics().valid) {
            ImGui::Spacing();
            theme::section_header("COMPARE");
            const ImageQaMetrics& metrics = app.image_qa_session.metrics();
            char buffer[32];
            std::snprintf(buffer, sizeof(buffer), "%.2f dB", metrics.psnr);
            theme::metric("PSNR", buffer);
            std::snprintf(buffer, sizeof(buffer), "%.4f", metrics.ssim);
            theme::metric("SSIM", buffer);
            std::snprintf(buffer, sizeof(buffer), "%.4f", metrics.mae);
            theme::metric("MAE", buffer);
            std::snprintf(buffer, sizeof(buffer), "%.4f", metrics.rmse);
            theme::metric("RMSE", buffer);
        }
        ImGui::Spacing();
    }

    if (app.view_mode != VisualizationMode::splat &&
        ImGui::CollapsingHeader("Viewport", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        if (theme::toolbar_button(
                "Load Sparse Cloud", {-1.F, 28.F},
                app.has_sparse && !app.loading_scene)) {
            app.view_mode = VisualizationMode::points;
            if (!app.layout.project_file.empty())
                request_ascan_scene_load(app);
            else
                request_scene_load(
                    app, app.layout.sparse_ply, app.layout.sparse_poses,
                    "Sparse cloud");
        }
        if (theme::toolbar_button(
                "Load Trained Model", {-1.F, 28.F},
                app.has_model && !app.loading_scene))
            request_gaussian_scene_load(app);
        if (theme::toolbar_button(
                "Load Mesh", {-1.F, 28.F},
                app.has_mesh && !app.loading_scene)) {
            app.mesh.clear();
            show_mesh_view(app, true);
        }
        if (app.view_mode == VisualizationMode::mesh) {
            ImGui::Spacing();
            theme::metric(
                "Vertices", format_count(app.mesh.vertices.size()).c_str());
            theme::metric(
                "Faces", format_count(app.mesh.faces.size()).c_str());
            ImGui::Checkbox("Wireframe", &app.view_options.mesh_wireframe);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Draw triangle edges on top of the shaded surface.");
            ImGui::Checkbox(
                "Vertex colour", &app.view_options.mesh_vertex_colour);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Use PLY vertex colours instead of clay shading.");
            theme::caption("Rendered face budget");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::SliderInt(
                "##mesh_budget", &app.view_options.mesh_face_budget, 20'000,
                600'000, "%d");
        }
        ImGui::Spacing();
        if (app.view_mode != VisualizationMode::mesh) {
        theme::caption("Point size");
        ImGui::SetNextItemWidth(-1.F);
        if (ImGui::SliderFloat(
                "##point_size", &app.view_options.point_size, 1.F, 6.F,
                "%.1f px") &&
            live_preview_active(app))
            publish_preview_vis(app);
        theme::caption("Rendered point budget");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderInt(
            "##budget", &app.view_options.point_budget, 20'000, 600'000,
            "%d");
        }
        if (app.view_mode == VisualizationMode::rings) {
            theme::caption("Ring budget");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::SliderInt(
                "##ring_budget", &app.view_options.ring_budget, 1'000, 40'000,
                "%d");
            theme::caption("Ring scale");
            ImGui::SetNextItemWidth(-1.F);
            if (ImGui::SliderFloat(
                    "##ring_scale", &app.view_options.ring_scale, 1.F, 4.F,
                    "%.1f σ") &&
                live_preview_active(app))
                publish_preview_vis(app);
        }
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
        if (mesh_from_gaussians(app.settings)) {
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
        if (app.job.paused()) {
            if (icons::labeled_button(
                    "##resume_job", icons::Icon::play,
                    resume_job_label(app.active_job), {-1.F, 40.F},
                    icons::ButtonStyle::primary, true, false,
                    "Continue the paused reconstruction."))
                action = Action::resume;
        } else if (icons::labeled_button(
                       "##pause_job", icons::Icon::pause,
                       pause_job_label(app.active_job), {-1.F, 40.F},
                       icons::ButtonStyle::normal, true, false,
                       k_pause_job_tooltip)) {
            action = Action::pause;
        }
        ImGui::Dummy({0, 6.F});
        if (icons::labeled_button(
                "##stop_job", icons::Icon::stop,
                stop_job_label(app.active_job), {-1.F, 36.F},
                icons::ButtonStyle::danger, true, false, k_stop_job_tooltip))
            action = Action::stop;
    } else if (has_external_dataset(app)) {
        if (mesh_from_mvs(app.settings)) {
            if (theme::primary_button("Build Mesh", {-1.F, 40.F}, true))
                action = Action::dense;
            ImGui::Dummy({0, 6.F});
            if (theme::toolbar_button("Train 3DGS", {-1.F, 32.F}))
                action = Action::train;
        } else {
            if (theme::primary_button(
                    mesh_from_gaussians(app.settings)
                        ? "Train External 3DGS + Mesh"
                        : "Train External 3DGS",
                    {-1.F, 40.F}, true))
                action = Action::train;
            ImGui::Dummy({0, 6.F});
            if (theme::toolbar_button("Dense MVS", {-1.F, 32.F}))
                action = Action::dense;
        }
    } else if (!app.has_sparse) {
        if (theme::primary_button(
                "Align Photos", {-1.F, 40.F},
                app.settings.images_dir[0] != '\0'))
            action = Action::align;
    } else if (mesh_from_mvs(app.settings)) {
        if (theme::primary_button(
                "Build Mesh", {-1.F, 40.F},
                app.settings.images_dir[0] != '\0'))
            action = Action::dense;
        ImGui::Dummy({0, 6.F});
        if (theme::toolbar_button("Train 3DGS", {-1.F, 32.F}))
            action = Action::train;
    } else {
        if (theme::primary_button(
                mesh_from_gaussians(app.settings) ? "Train 3DGS + Mesh"
                                                  : "Train 3DGS",
                {-1.F, 40.F}, app.settings.images_dir[0] != '\0'))
            action = Action::train;
        ImGui::Dummy({0, 6.F});
        if (theme::toolbar_button("Dense MVS", {-1.F, 32.F}))
            action = Action::dense;
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
    const bool paused = busy && app.job.paused();
    const ImVec4 background = paused
        ? ImVec4(0.220F, 0.140F, 0.040F, 1.F)
        : (busy ? ImVec4(0.027F, 0.208F, 0.325F, 1.F)
                : ImVec4(0.086F, 0.090F, 0.102F, 1.F));
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
    if (paused) {
        state_colour = theme::warning;
        state_icon = icons::Icon::pause;
    } else if (busy) {
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
        const std::string headline = paused
            ? ("Paused  |  " + app.monitor.headline())
            : app.monitor.headline();
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

    const double elapsed = app.monitor.elapsed_seconds();
    if (elapsed >= 0.0) {
        const std::string elapsed_text =
            "Elapsed " + format_duration(elapsed);
        const float elapsed_width = status_segment_width(elapsed_text.c_str());
        right -= elapsed_width + 22.F;
        draw_status_separator(right + elapsed_width + 11.F, height);
        draw_status_segment(
            right, centre_y, icons::Icon::clock, elapsed_text.c_str(),
            busy ? theme::text_bright : theme::text_muted);
    }

    if (busy) {
        const float fraction = app.monitor.fraction();
        const double eta = app.monitor.eta_seconds();
        char trailing[96];
        if (paused) {
            if (fraction >= 0.F)
                std::snprintf(
                    trailing, sizeof(trailing), "%3.0f%%   paused",
                    fraction * 100.F);
            else
                std::snprintf(trailing, sizeof(trailing), "paused");
        } else if (fraction >= 0.F && eta >= 0.0)
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
            {bar_width, 7.F},
            paused && fraction < 0.F ? 0.F : fraction,
            paused ? theme::warning
                   : (fraction < 0.F ? theme::accent
                                     : ImVec4(0.55F, 0.84F, 1.F, 1.F)));
        ImGui::SameLine(0.F, 12.F);
        ImGui::SetCursorPosY(centre_y);
        ImGui::TextUnformatted(trailing);
    } else {
        if (app.view_mode == VisualizationMode::mesh && app.mesh.has()) {
            const std::string faces =
                format_count(app.mesh.faces.size()) + " faces";
            const float faces_width = status_segment_width(faces.c_str());
            right -= faces_width + 22.F;
            draw_status_separator(right + faces_width + 11.F, height);
            draw_status_segment(
                right, centre_y, icons::Icon::cube, faces.c_str());
        } else if (app.scene.has_points()) {
            const std::string points =
                format_count(app.scene.points.size()) + " pts";
            const float points_width = status_segment_width(points.c_str());
            right -= points_width + 22.F;
            draw_status_separator(right + points_width + 11.F, height);
            draw_status_segment(
                right, centre_y, icons::Icon::points, points.c_str());
        }

        const char* view = "Points";
        icons::Icon view_icon = icons::Icon::points;
        if (app.workspace == ViewportWorkspace::image_2d) {
            view = "2D Image";
            view_icon = icons::Icon::view2d;
        } else if (app.view_mode == VisualizationMode::splat) {
            view = "Splat";
            view_icon = icons::Icon::splat;
        } else if (app.view_mode == VisualizationMode::rings) {
            view = "Rings";
            view_icon = icons::Icon::rings;
        } else if (app.view_mode == VisualizationMode::mesh) {
            view = "Mesh";
            view_icon = icons::Icon::cube;
        }
        const float view_width = status_segment_width(view);
        right -= view_width + 22.F;
        if (right > 520.F) {
            draw_status_separator(right + view_width + 11.F, height);
            draw_status_segment(right, centre_y, view_icon, view);
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
        app.viewer.poll();
        app.preview.poll();
        app.log.poll(app.fresh_lines);
        for (const std::string& line : app.fresh_lines)
            app.monitor.consume(line);
        if (app.job.consume_completion()) on_job_finished(app);
        if (app.viewer.consume_completion()) {
            const int view_code = app.viewer.exit_code();
            if (view_code != 0 && view_code != 2)
                set_message(
                    app,
                    "Splat viewer exited with code " +
                        std::to_string(view_code),
                    theme::warning);
        }
        poll_scene_load(app);
        poll_mesh_load(app);
        poll_alignment_preview(app);
        poll_camera_photos(app);

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
        consume_dropped_paths(app);

        app.settings.iterations = std::max(app.settings.iterations, 1);
        app.settings.preview_interval =
            std::max(app.settings.preview_interval, 1);
        app.settings.max_features = std::max(app.settings.max_features, 512);
        app.settings.dataset_format = std::clamp(
            app.settings.dataset_format, 0, 3);
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
            case Action::dense:
                if (!app.smoke_mode) start_dense(app);
                break;
            case Action::export_sfm:
                if (!app.smoke_mode) start_export_sfm(app);
                break;
            case Action::pause:
                app.job.pause();
                if (app.job.paused()) {
                    app.monitor.pause_clock();
                    set_message(
                        app, "Paused. Press Resume to continue.",
                        theme::warning);
                } else {
                    set_message(
                        app, "Could not pause the running job", theme::danger);
                }
                break;
            case Action::resume:
                app.job.resume();
                if (!app.job.paused()) {
                    app.monitor.resume_clock();
                    set_message(app, "Resumed", theme::accent);
                } else {
                    set_message(
                        app, "Could not resume the paused job", theme::danger);
                }
                break;
            case Action::stop:
                app.job.stop();
                if (app.job.consume_completion())
                    on_job_finished(app);
                else
                    set_message(
                        app,
                        "Stopped. You can change images, video, or parameters "
                        "and run again.",
                        theme::warning);
                break;
            case Action::reveal: reveal_in_explorer(app.layout.root); break;
            case Action::none: break;
        }

        draw_controls_window(app);
        const ClearResultsAction clear_results_action =
            draw_clear_results_modal(app);
        if (app.close_requested) glfwSetWindowShouldClose(window, GLFW_TRUE);
        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (draw_data->DisplaySize.x > 0.F && draw_data->DisplaySize.y > 0.F)
            gpu::present(draw_data, theme::surface_0);

        if (app.pending_align_viewport_clear) {
            app.pending_align_viewport_clear = false;
            clear_viewport_scene(app);
        }

        // The viewport draw list can reference camera-photo descriptors until
        // present() has submitted this frame. Clear only afterwards so the
        // confirmation click cannot invalidate resources used by that frame.
        if (clear_results_action == ClearResultsAction::clear_view) {
            clear_loaded_result(app);
            // draw_sparse_tab() normally restores an available sparse result on
            // every frame. Keep a user-requested clear stable until they
            // explicitly load/select the cloud again.
            app.suppress_scene_auto_load = true;
            set_message(
                app, "Loaded reconstruction cleared from view", theme::success);
        } else if (
            clear_results_action == ClearResultsAction::delete_generated) {
            delete_reconstruction_results(app);
        }
    }

    if (app.viewer.running()) app.viewer.stop();
    if (app.job.running()) app.job.stop();
    vkDeviceWaitIdle(gpu::device());
    app.image_qa_session.clear();
    app.photos.clear();
#if defined(AETHERSCAN_HAS_TEXTURE)
    app.mesh_preview.reset();
    app.mesh_rasterizer.reset();
#endif
    app.preview.reset();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    gpu::destroy_context();
    glfwDestroyWindow(window);
    glfwTerminate();
    return app.smoke_mode && !app.smoke_success ? 4 : 0;
}
