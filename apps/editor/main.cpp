#include "pipeline.hpp"
#include "console_view.hpp"
#include "icons.hpp"
#include "sparse_view.hpp"
#include "theme.hpp"
#include "viewport_gizmo.hpp"
#include "vulkan_backend.hpp"

#include "project/archive.hpp"
#include "project/document.hpp"
#include "sfm/asfm.hpp"
#include "sfm/export_mvs.hpp"
#include "splat/trainer.hpp"
#include "splat/visualize.hpp"

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

enum class VisualizationMode { points, splat, rings };

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
    OrbitCamera camera;
    ViewOptions view_options;
    ViewportGizmoState gizmo;
    SceneRenderer renderer;
    std::future<SceneLoad> pending_load;
    bool loading_scene{};
    std::string scene_source;

    bool has_sparse{};
    bool has_asfm{};
    bool has_mvs{};
    bool has_model{};
    bool has_mesh{};
    std::uint32_t project_writer_version{};
    std::uint32_t project_min_reader_version{};
    VisualizationMode view_mode{VisualizationMode::points};
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

bool has_external_dataset(const App& app) {
    return app.settings.dataset_source[0] != '\0';
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
        error = "Drop an image folder, photos, .asfm, or .ascan project";
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

enum class FilePickKind { project, dataset, point_cloud, splat_model };

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
        const COMDLG_FILTERSPEC* filters = project_filters;
        if (kind == FilePickKind::dataset) filters = dataset_filters;
        if (kind == FilePickKind::point_cloud) filters = point_cloud_filters;
        if (kind == FilePickKind::splat_model) filters = splat_model_filters;
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
    app.has_mesh = std::filesystem::exists(app.layout.mesh_ply, error);
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

void assign_default_project_folder(App& app) {
    if (app.settings.images_dir[0] == '\0' ||
        app.settings.project_dir[0] != '\0')
        return;

    const std::filesystem::path images(app.settings.images_dir.data());
    const std::filesystem::path parent = images.parent_path();
    const std::filesystem::path project =
        (parent.empty() ? images : parent) / (images.filename().string() + ".ascan");
    const std::string text = project.string();
    std::snprintf(
        app.settings.project_dir.data(), app.settings.project_dir.size(), "%s",
        text.c_str());
    app.project_folder_automatic = true;
}

void store_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path) {
    const std::string text = path.string();
    std::snprintf(field.data(), field.size(), "%s", text.c_str());
}

void store_utf8_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path) {
#if defined(_WIN32)
    const std::u8string utf8 = path.u8string();
    if (utf8.size() + 1 > field.size()) return;
    std::memcpy(field.data(), utf8.data(), utf8.size());
    field[utf8.size()] = '\0';
#else
    store_path_field(field, path);
#endif
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
        : app.layout.project_file.stem().string();
    settings.image_directory = app.settings.images_dir.data();
    settings.dataset_source = app.settings.dataset_source.data();
    settings.dataset_format = app.settings.dataset_format == 1
        ? "colmap"
        : app.settings.dataset_format == 2
            ? "realitycapture"
            : app.settings.dataset_format == 3 ? "openmvs" : "auto";
    settings.dataset_initial_cloud = app.settings.dataset_initial_cloud.data();
    settings.splat_model_source = app.settings.splat_model_source.data();
    settings.splat_output_format = app.settings.splat_format == 1
        ? "ply"
        : app.settings.splat_format == 2
            ? "sog"
            : app.settings.splat_format == 3 ? "spz"
            : app.settings.splat_format == 4 ? "glb" : "auto";
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
    const std::filesystem::path images(app.settings.images_dir.data());
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

void request_ascan_scene_load(App& app) {
    if (app.loading_scene) return;
    const auto ascan = app.layout.project_file;
    const auto working = app.layout.working_sfm;
    const auto asfm = app.layout.sparse_asfm;
    const std::filesystem::path images(app.settings.images_dir.data());
    if (ascan.empty() && working.empty() && asfm.empty()) return;
    app.suppress_scene_auto_load = false;
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
                app.layout.working_sfm, app.settings.images_dir.data());
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

void clear_loaded_result(App& app) {
    app.photos.clear();
    app.scene.clear();
    app.scene_source.clear();
    app.camera = {};
    app.view_mode = VisualizationMode::points;
    app.monitor.reset();
    app.log.clear();
    app.console = {};
    app.fresh_lines.clear();
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
    apply_image_directory_selection(app);
}

void open_project_from_path(App& app, const std::filesystem::path& path);
void open_asfm_from_path(App& app, const std::filesystem::path& asfm);

void apply_dropped_image_source(
    App& app, const std::vector<std::string>& dropped) {
    if (app.job.running() || app.loading_scene) {
        set_message(
            app, "Cannot change images while a job is running", theme::warning);
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
            "Drop photos, an .asfm scene, or an .ascan project on the viewport",
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
    app.suppress_scene_auto_load = false;
    app.loading_scene = true;
    app.scene_source = "Gaussian centres";
    app.pending_load = std::async(
        std::launch::async, [model_path, ascan, poses] {
            try {
                if (!model_path.empty())
                    return load_gaussian_scene(model_path, poses);
                std::error_code error;
                if (std::filesystem::exists(ascan, error)) {
                    const auto archive =
                        aetherscan::project::Archive::open(ascan);
                    if (archive.has(aetherscan::project::ChunkType::gaussians))
                        return gaussian_scene_from_model(
                            aetherscan::splat::decode_gaussians(
                                archive.chunk(
                                    aetherscan::project::ChunkType::gaussians)),
                            poses);
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
    if (app.suppress_scene_auto_load || !app.has_sparse ||
        app.scene.has_points() || app.loading_scene)
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
constexpr float k_view_rail_height = 120.F;

ImRect view_mode_rail_rect(const ImVec2 view_min) {
    const ImVec2 origin{
        view_min.x + k_view_rail_pad, view_min.y + k_view_rail_top};
    return {
        origin.x, origin.y, origin.x + k_view_rail_width,
        origin.y + k_view_rail_height};
}

bool view_mode_rail_contains(const ImVec2 view_min, const ImVec2 mouse) {
    return view_mode_rail_rect(view_min).Contains(mouse);
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
    const RailItem items[] = {
        {"##viz_points", icons::Icon::points, VisualizationMode::points, true,
         "Point Cloud"},
        {"##viz_splat", icons::Icon::splat, VisualizationMode::splat, splat_ok,
         splat_ok ? "Splat" : "Train 3DGS to view the splat"},
        {"##viz_rings", icons::Icon::rings, VisualizationMode::rings, rings_ok,
         rings_ok ? "Rings"
                  : "Available while training or after a Gaussian model exists"},
    };

    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const bool clicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0.F, k_gap});
    for (int i = 0; i < 3; ++i) {
        const ImVec2 button_min{
            rail.Min.x + k_inner,
            rail.Min.y + k_inner + static_cast<float>(i) * (k_btn + k_gap)};
        ImGui::SetCursorScreenPos(button_min);
        ImGui::SetNextItemAllowOverlap();
        const bool pressed = icons::ghost_button(
            items[i].id, items[i].icon, button_size,
            app.view_mode == items[i].mode, items[i].enabled,
            items[i].tooltip);
        const bool hit =
            items[i].enabled && clicked &&
            ImRect{button_min, {button_min.x + k_btn, button_min.y + k_btn}}
                .Contains(mouse);
        if (pressed || hit)
            set_visualization_mode(app, items[i].mode);
        hovered = hovered || ImGui::IsItemHovered() ||
                  ImRect{button_min, {button_min.x + k_btn, button_min.y + k_btn}}
                      .Contains(mouse);
    }
    ImGui::PopStyleVar();
    return hovered;
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
    if (app.has_sparse)
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
        std::filesystem::path path(app.settings.project_dir.data());
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
    if (app.layout.cache.empty()) return false;
    std::error_code error;
    return std::filesystem::exists(app.layout.cache, error);
}

void delete_reconstruction_results(App& app) {
    if (app.job.running() || app.loading_scene || app.layout.root.empty()) return;

    stop_splat_view(app);
    clear_loaded_result(app);
    app.suppress_scene_auto_load = true;
    const std::array<std::filesystem::path, 13> generated_files = {
        app.layout.sparse_ply, app.layout.sparse_asfm, app.layout.sparse_mvs,
        app.layout.sparse_poses, app.layout.splat_ply, app.layout.splat_sog,
        app.layout.splat_spz, app.layout.splat_glb, app.layout.mesh_ply,
        app.layout.align_log, app.layout.train_log, app.layout.export_log,
        app.layout.view_log};

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
    attach_view_image_paths(
        app.scene, std::filesystem::path(app.settings.images_dir.data()));
    app.photos.resize(app.scene.views.size());
    if (app.view_options.show_views && app.view_options.show_camera_photos) {
        std::vector<std::size_t> markers;
        sampled_view_indices(app.scene, markers);
        for (const std::size_t index : markers)
            app.photos.request(index, app.scene.views[index].image_path);
    }
    app.photos.poll();
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
    app.photos.clear();
    app.scene = std::move(loaded.scene);
    infer_images_dir_from_scene(app);
    attach_view_image_paths(
        app.scene, std::filesystem::path(app.settings.images_dir.data()));
    if (!live_preview_active(app)) app.camera.frame(app.scene);
    if (app.view_mode != VisualizationMode::splat &&
        app.view_mode != VisualizationMode::rings)
        app.view_mode = VisualizationMode::points;
    set_message(
        app,
        app.scene_source + ": " + format_count(app.scene.points.size()) +
            " points, " + std::to_string(app.scene.registered_views) + " / " +
            std::to_string(app.scene.total_views) + " views registered",
        theme::success);
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
                    app.layout.working_sfm, app.settings.images_dir.data());
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
        // The external dataset is consumed by the child CLI. Do not leave a
        // stale internal SfM scene visible while that job is running.
        app.photos.clear();
        app.scene.clear();
    } else {
        load_view_poses(app.layout.sparse_poses, app.scene);
        attach_view_image_paths(
            app.scene, std::filesystem::path(app.settings.images_dir.data()));
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
            request_ascan_scene_load(app);
        } else {
            set_message(
                app, "Alignment finished but no SfM stage was written",
                theme::warning);
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
        app.settings.build_mesh
            ? (app.has_mesh ? "Training and mesh extraction finished"
                            : "Training finished, mesh extraction produced no "
                              "surface")
            : "Training finished",
        app.settings.build_mesh && !app.has_mesh ? theme::warning
                                                 : theme::success);
    if (!app.smoke_mode && app.has_model) start_splat_view(app);
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
    if (!io.WantTextInput && ImGui::IsKeyPressed(ImGuiKey_Escape) && busy)
        action = Action::stop;

    if (!ImGui::BeginMainMenuBar()) return action;
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("New Project", "Ctrl+N", false, !busy)) {
            new_project(app);
        }
        if (ImGui::MenuItem("Select Image Folder...", "Ctrl+O", false, !busy)) {
            select_image_folder(app);
        }
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
        ImGui::TextWrapped("%s", app.layout.root.string().c_str());
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
            "Select capture image folder, or drop photos / .asfm / .ascan on the viewport")) {
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
                   !busy && images_ready && !external_dataset)) {
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
        if (ImGui::IsItemHovered()) {
            if (external_dataset)
                ImGui::SetTooltip(
                    "Train directly from the selected external camera dataset.");
            else if (!app.has_sparse)
                ImGui::SetTooltip(
                    "No alignment yet. Training will run Structure from Motion "
                    "first, then optimise Gaussians.");
        }
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
                app.suppress_scene_auto_load = false;
                if (app.has_sparse) ensure_sparse_loaded(app);
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
            if (has_mesh) object_row("Mesh", false);
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
        app.settings.images_dir[0] != '\0' ? StepState::done : StepState::pending,
        nullptr);
    draw_step(
        "02", "Align cameras",
        aligning ? StepState::active
                 : (app.has_sparse || external_dataset
                        ? StepState::done
                        : StepState::pending),
        aligning
            ? stage_name(stage)
            : (external_dataset ? "External dataset" : nullptr));
    draw_step(
        "03", "Review sparse cloud",
        app.scene.has_points() ? StepState::done
                               : (external_dataset ? StepState::skipped
                                                   : (app.has_sparse
                                                          ? StepState::active
                                                          : StepState::pending)),
        app.scene.has_points()
            ? nullptr
            : (external_dataset
                   ? "Provided by external dataset"
                   : (app.has_sparse
                          ? (app.loading_scene ? "Loading" : "On disk")
                          : nullptr)));
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
    if (app.view_mode == VisualizationMode::rings)
        ensure_gaussian_scene(app);
    else
        ensure_sparse_loaded(app);
    app.view_options.draw_rings = app.view_mode == VisualizationMode::rings;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilledMultiColor(
        min, max, IM_COL32(9, 11, 16, 255), IM_COL32(9, 11, 16, 255),
        IM_COL32(18, 22, 32, 255), IM_COL32(18, 22, 32, 255));

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

    if (!app.loading_scene && !app.scene.has_points() && !app.has_sparse) {
        draw_empty_viewport(
            draw, min, max,
            app.settings.images_dir[0] != '\0' ? "Ready to align cameras"
                                               : "Drop photos or a reconstruction",
            app.settings.images_dir[0] != '\0'
                ? "Run Align Photos, or drop a different folder, .asfm, or .ascan"
                : "Drop an image folder, photos, .asfm, or .ascan onto this view");
    }

    const SceneDrawStats stats = app.renderer.draw(
        draw, min, max, app.scene, app.camera, app.view_options, hovered,
        app.photos.ids(), app.photos.size());

    const bool gizmo_captures =
        draw_viewport_gizmo(app.gizmo, app.camera, min, max);
    const bool frame_key = hovered && !gizmo_captures &&
                           !ImGui::GetIO().WantTextInput &&
                           ImGui::IsKeyPressed(ImGuiKey_F);
    if (frame_key) app.camera.frame(app.scene);
    update_orbit_camera(
        app.camera, hovered && !gizmo_captures, app.scene.radius);

    const char* overlay = app.settings.images_dir[0] != '\0' ? "NO ALIGNMENT"
                                                            : "NO IMAGES";
    ImVec4 overlay_dot = theme::inactive;
    if (app.loading_scene) {
        overlay = "LOADING";
        overlay_dot = theme::warning;
    } else if (app.view_mode == VisualizationMode::rings) {
        overlay = app.scene.has_gaussians() ? "GAUSSIAN RINGS" : "NO GAUSSIANS";
        overlay_dot = app.scene.has_gaussians() ? theme::accent : theme::inactive;
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
        const char* hint = app.has_sparse
            ? "Reading sparse.ply..."
            : "Drop a photo folder, .asfm, or .ascan here";
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
        ? "LMB orbit  |  MMB pan  |  RMB + WASD fly  |  arrows snap capture"
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

    const char* state = app.job.running()
        ? running_job_caption(app.active_job)
        : (app.viewer.running() ? "VIEWING" : "READY");
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
    if (live_preview_active(app))
        draw_training_tab(app, view_min, view_max);
    else
        draw_sparse_tab(app, view_min, view_max);
    draw_view_mode_rail(app, view_min);
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
            clear_loaded_result(app);
            refresh_artifacts(app);
        }
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button("Folder##pick_dataset_folder", {58.F, 0.F}) &&
            pick_folder(
                L"Select external SfM dataset folder",
                app.settings.dataset_source)) {
            clear_loaded_result(app);
            refresh_artifacts(app);
        }
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button("File##pick_dataset_file", {46.F, 0.F}) &&
            pick_dataset_file(
                L"Select external camera dataset file",
                app.settings.dataset_source)) {
            clear_loaded_result(app);
            refresh_artifacts(app);
        }
        if (has_external_dataset(app)) {
            theme::caption(
                "Train 3DGS will use the imported cameras and skip internal SfM.");
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
        ImGui::Spacing();
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
        ImGui::Checkbox("Show cameras", &app.view_options.show_views);
        ImGui::BeginDisabled(!app.view_options.show_views);
        ImGui::Checkbox("Show camera photos", &app.view_options.show_camera_photos);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip(
                "Map each capture onto the far plane of its view frustum.");
        ImGui::EndDisabled();
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
    } else if (has_external_dataset(app)) {
        if (theme::primary_button(
                app.settings.build_mesh ? "Train External 3DGS + Mesh"
                                        : "Train External 3DGS",
                {-1.F, 40.F},
                app.settings.images_dir[0] != '\0'))
            action = Action::train;
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

        const char* view = "Points";
        icons::Icon view_icon = icons::Icon::points;
        if (app.view_mode == VisualizationMode::splat) {
            view = "Splat";
            view_icon = icons::Icon::splat;
        } else if (app.view_mode == VisualizationMode::rings) {
            view = "Rings";
            view_icon = icons::Icon::rings;
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
        const ClearResultsAction clear_results_action =
            draw_clear_results_modal(app);
        if (app.close_requested) glfwSetWindowShouldClose(window, GLFW_TRUE);
        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        if (draw_data->DisplaySize.x > 0.F && draw_data->DisplaySize.y > 0.F)
            gpu::present(draw_data, theme::surface_0);

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
    app.photos.clear();
    app.preview.reset();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    gpu::destroy_context();
    glfwDestroyWindow(window);
    glfwTerminate();
    return app.smoke_mode && !app.smoke_success ? 4 : 0;
}
