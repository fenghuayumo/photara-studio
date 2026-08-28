#include "pipeline.hpp"
#include "sparse_view.hpp"
#include "theme.hpp"
#include "vulkan_backend.hpp"

#include "imgui_impl_glfw.h"

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
#include <cstdio>
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
    std::vector<std::string> fresh_lines;

    gpu::ExternalPreview preview;

    SparseScene scene;
    OrbitCamera camera;
    ViewOptions view_options;
    SceneRenderer renderer;
    std::future<SceneLoad> pending_load;
    bool loading_scene{};
    std::string scene_source;

    bool has_sparse{};
    bool has_model{};
    bool has_mesh{};
    ViewportTab tab{ViewportTab::sparse};

    std::string message;
    ImVec4 message_colour{theme::text_muted};

    // Manual CUDA/Vulkan interop verification harness.
    bool smoke_mode{};
    bool smoke_started{};
    bool smoke_success{};
};

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
    app.layout = resolve_layout(app.settings);
    std::error_code error;
    app.has_sparse = std::filesystem::exists(app.layout.sparse_ply, error);
    app.has_model = std::filesystem::exists(app.layout.splat_ply, error);
    app.has_mesh = std::filesystem::exists(app.layout.mesh_ply, error);
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
    app.camera.frame(app.scene);
    app.tab = ViewportTab::sparse;
    set_message(
        app,
        app.scene_source + ": " + format_count(app.scene.points.size()) +
            " points, " + std::to_string(app.scene.registered_views) + " / " +
            std::to_string(app.scene.total_views) + " views registered",
        theme::success);
}

void start_align(App& app) {
    if (app.job.running()) return;
    refresh_artifacts(app);
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);
    if (error) {
        set_message(app, "Cannot create project directory", theme::danger);
        return;
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

void start_train(App& app, const bool smoke) {
    if (app.job.running()) return;
    refresh_artifacts(app);
    std::error_code error;
    std::filesystem::create_directories(app.layout.root, error);

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
            std::string(kind == JobKind::align ? "Alignment" : "Training") +
                " failed: " + reason,
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

// Brand strip plus a breadcrumb of the four workflow stages, so the current
// position in the pipeline is readable without scanning the side panels.
void draw_title_bar(const App& app) {
    const bool training =
        app.job.running() && app.active_job == JobKind::train;
    const bool aligning =
        app.job.running() && app.active_job == JobKind::align;
    const int current = training && app.monitor.stage() == Stage::meshing ? 3
        : training                                                        ? 2
        : aligning                                                        ? 1
        : app.has_model                                                   ? 2
        : app.has_sparse                                                  ? 1
                                                                          : 0;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.078F, 0.080F, 0.090F, 1.F));
    ImGui::BeginChild("##titlebar", {0, 30.F}, false, ImGuiWindowFlags_NoScrollbar);
    ImGui::SetCursorPos({14.F, 6.F});
    ImGui::PushStyleColor(ImGuiCol_Text, theme::accent);
    ImGui::TextUnformatted("AETHER");
    ImGui::PopStyleColor();
    ImGui::SameLine(0.F, 1.F);
    ImGui::TextUnformatted("SCAN");

    const std::array<const char*, 4> steps{
        {"Images", "Alignment", "Gaussians", "Mesh"}};
    ImGui::SameLine(0.F, 26.F);
    for (int i = 0; i < static_cast<int>(steps.size()); ++i) {
        if (i > 0) {
            ImGui::SameLine(0.F, 8.F);
            theme::caption("\xE2\x80\xBA");  // single right angle quote
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
    const float width = ImGui::CalcTextSize(project.c_str()).x;
    ImGui::SameLine(std::max(0.F, ImGui::GetWindowWidth() - width - 16.F));
    theme::caption(project.c_str());
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

// Returns the action requested from the toolbar, if any.
enum class Action { none, align, train, stop, reveal, frame_scene };

Action draw_toolbar(App& app) {
    Action action = Action::none;
    const bool busy = app.job.running();
    const bool images_ready = app.settings.images_dir[0] != '\0';

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.106F, 0.110F, 0.122F, 1.F));
    ImGui::BeginChild("##toolbar", {0, 52.F}, false, ImGuiWindowFlags_NoScrollbar);
    ImGui::SetCursorPos({12.F, 10.F});

    if (theme::toolbar_button("Image Folder", {114.F, 32.F}, !busy)) {
        if (pick_folder(L"Select the capture image folder", app.settings.images_dir))
            refresh_artifacts(app);
    }
    ImGui::SameLine();

    // Step 1: multi-view alignment.
    const bool aligning = busy && app.active_job == JobKind::align;
    if (aligning) {
        theme::toolbar_button("Aligning...", {124.F, 32.F}, false, true);
    } else if (theme::primary_button(
                   app.has_sparse ? "Re-align Photos" : "Align Photos",
                   {124.F, 32.F}, !busy && images_ready)) {
        action = Action::align;
    }
    ImGui::SameLine();

    // Step 2: Gaussian optimisation, gated on a reviewed alignment.
    const bool training = busy && app.active_job == JobKind::train;
    if (training) {
        theme::toolbar_button("Training...", {124.F, 32.F}, false, true);
    } else {
        const bool ready = !busy && images_ready;
        if (app.has_sparse) {
            if (theme::primary_button("Train 3DGS", {124.F, 32.F}, ready))
                action = Action::train;
        } else if (theme::toolbar_button("Train 3DGS", {124.F, 32.F}, ready)) {
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

    if (theme::toolbar_button(
            "Open Output", {104.F, 32.F}, app.layout.root.has_filename()))
        action = Action::reveal;
    if (busy) {
        ImGui::SameLine();
        if (theme::danger_button("Stop", {74.F, 32.F})) action = Action::stop;
    }

    const char* transport = "CUDA / Vulkan  ·  external memory";
    const float transport_width = ImGui::CalcTextSize(transport).x;
    ImGui::SameLine(ImGui::GetWindowWidth() - transport_width - 16.F);
    ImGui::SetCursorPosY(18.F);
    theme::caption(transport);

    ImGui::EndChild();
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

void draw_left_panel(App& app, const float width) {
    theme::begin_panel("LeftPanel", {width, 0}, nullptr, nullptr, {0, 0});

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
            "Sparse Point Cloud", app.scene.has_points(),
            app.tab == ViewportTab::sparse && app.scene.has_points());
        leaf(
            "Gaussian Model", app.has_model,
            app.tab == ViewportTab::training);
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
            : (app.has_sparse ? "Ready to load" : nullptr));
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
    ImGui::PushTextWrapPos(width - 20.F);
    ImGui::TextUnformatted(
        app.settings.images_dir[0] != '\0' ? app.settings.images_dir.data()
                                          : "(not selected)");
    ImGui::PopTextWrapPos();
    ImGui::Dummy({0, 6.F});
    theme::caption("PROJECT");
    ImGui::Spacing();
    ImGui::PushTextWrapPos(width - 20.F);
    ImGui::TextUnformatted(
        app.settings.project_dir[0] != '\0' ? app.settings.project_dir.data()
                                           : "(not selected)");
    ImGui::PopTextWrapPos();
    ImGui::Unindent(14.F);

    theme::end_panel();
}

void draw_empty_viewport(
    ImDrawList* draw, const ImVec2 min, const ImVec2 max, const char* headline,
    const char* hint) {
    // A restrained perspective construction grid so the empty stage still
    // reads as a 3D workspace.
    const float horizon = min.y + (max.y - min.y) * 0.46F;
    const ImVec2 vanishing{(min.x + max.x) * 0.5F, horizon};
    draw->PushClipRect(min, max, true);
    draw->AddLine({min.x, horizon}, {max.x, horizon}, IM_COL32(30, 34, 42, 255));
    constexpr int rays = 16;
    for (int i = -rays; i <= rays; ++i) {
        const float x =
            vanishing.x + i * (max.x - min.x) / static_cast<float>(rays);
        draw->AddLine(
            vanishing, {x, max.y},
            i == 0 ? IM_COL32(52, 62, 74, 200) : IM_COL32(30, 34, 42, 170));
    }
    for (int i = 0; i < 16; ++i) {
        const float t = static_cast<float>(i) / 15.F;
        const float y = horizon + t * t * (max.y - horizon);
        draw->AddLine({min.x, y}, {max.x, y}, IM_COL32(30, 34, 42, 170));
    }
    draw->PopClipRect();

    const float headline_width = ImGui::CalcTextSize(headline).x;
    const float hint_width = ImGui::CalcTextSize(hint).x;
    const float centre_x = (min.x + max.x) * 0.5F;
    draw->AddText(
        {centre_x - headline_width * 0.5F, horizon - 44.F},
        theme::u32(theme::text_muted), headline);
    draw->AddText(
        {centre_x - hint_width * 0.5F, horizon - 22.F},
        theme::u32(theme::text_faint), hint);
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
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(min, max, theme::u32(theme::viewport_bg));

    if (!app.scene.has_points()) {
        const char* headline = app.loading_scene
            ? "Loading sparse reconstruction..."
            : (app.has_sparse ? "Sparse cloud ready to load"
                              : "No alignment yet");
        const char* hint = app.has_sparse
            ? "Use Load Sparse Cloud in the inspector"
            : "Pick an image folder, then run Align Photos";
        draw_empty_viewport(draw, min, max, headline, hint);
        return;
    }

    const bool hovered = ImGui::IsWindowHovered();
    update_orbit_camera(app.camera, hovered);
    const SceneDrawStats stats = app.renderer.draw(
        draw, min, max, app.scene, app.camera, app.view_options, hovered);

    draw_viewport_overlay(
        draw, min,
        app.scene_source.empty() ? "SPARSE POINT CLOUD" : app.scene_source.c_str(),
        theme::accent);

    // Bottom-left readout: what is on screen and how to navigate.
    char readout[192];
    std::snprintf(
        readout, sizeof(readout), "%s pts drawn  ·  %zu cameras  ·  %s pts total",
        format_count(stats.drawn_points).c_str(), stats.drawn_views,
        format_count(app.scene.points.size()).c_str());
    draw->AddText(
        {min.x + 16.F, max.y - 42.F}, theme::u32(theme::text_muted), readout);
    draw->AddText(
        {min.x + 16.F, max.y - 24.F}, theme::u32(theme::text_faint),
        "LMB orbit  ·  RMB pan  ·  wheel zoom");

    if (stats.hovered_view >= 0 &&
        static_cast<std::size_t>(stats.hovered_view) < app.scene.views.size()) {
        const ViewPose& pose = app.scene.views[stats.hovered_view];
        ImGui::SetTooltip(
            "%s\n%u x %u  ·  f %.1f px\n%zu observations  ·  p95 %.2f px",
            pose.name.c_str(), pose.width, pose.height, pose.fx,
            pose.observations, pose.reprojection_p95);
    }
}

void draw_training_tab(App& app, const ImVec2 min, const ImVec2 max) {
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(min, max, theme::u32(theme::viewport_bg));

    const bool training = app.job.running() && app.active_job == JobKind::train;
    const bool has_frame = app.preview.display.descriptor &&
                           gpu::consumed_timeline_value() > 0;
    if (!has_frame) {
        draw_empty_viewport(
            draw, min, max,
            training ? "Waiting for the first rendered iteration..."
                     : "No live training preview",
            training ? "Frames arrive over Vulkan external memory"
                     : "Run Train 3DGS to stream the optimiser output");
        draw_viewport_overlay(
            draw, min, training ? "TRAINING" : "IDLE",
            training ? theme::warning : theme::inactive);
        return;
    }

    const ImVec2 available{max.x - min.x, max.y - min.y};
    const float scale = std::min(
        available.x / static_cast<float>(app.preview.width),
        available.y / static_cast<float>(app.preview.height));
    const ImVec2 size{
        app.preview.width * scale, app.preview.height * scale};
    const ImVec2 origin{
        min.x + (available.x - size.x) * 0.5F,
        min.y + (available.y - size.y) * 0.5F};
    draw->AddImage(
        reinterpret_cast<ImTextureID>(app.preview.display.descriptor), origin,
        {origin.x + size.x, origin.y + size.y});

    draw_viewport_overlay(
        draw, min, training ? "LIVE TRAINING PREVIEW" : "LAST TRAINING FRAME",
        training ? theme::success : theme::inactive);

    const TrainingStats& stats = app.monitor.training();
    if (stats.valid) {
        char readout[192];
        std::snprintf(
            readout, sizeof(readout),
            "iter %u / %u  ·  %s gaussians  ·  loss %.4f  ·  %.1f ms/step",
            stats.iteration, stats.total_iterations,
            format_count(stats.gaussians).c_str(), stats.loss,
            stats.step_milliseconds);
        draw->AddText(
            {min.x + 16.F, max.y - 26.F}, theme::u32(theme::text_muted),
            readout);
    }
}

void draw_centre_column(App& app, const float width, const float console_height) {
    ImGui::BeginChild("CentreStack", {width, 0}, false, ImGuiWindowFlags_NoScrollbar);
    const float viewport_height =
        ImGui::GetContentRegionAvail().y - console_height - 4.F;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::BeginChild(
        "Viewport", {0, viewport_height}, true, ImGuiWindowFlags_NoScrollbar);
    ImGui::PopStyleVar();

    // Header with the viewport tabs on the left and state on the right.
    const ImVec2 header_origin = ImGui::GetCursorScreenPos();
    const float header_width = ImGui::GetContentRegionAvail().x;
    ImGui::GetWindowDrawList()->AddRectFilled(
        header_origin, {header_origin.x + header_width, header_origin.y + 36.F},
        theme::u32(theme::surface_3));
    ImGui::GetWindowDrawList()->AddLine(
        {header_origin.x, header_origin.y + 35.F},
        {header_origin.x + header_width, header_origin.y + 35.F},
        theme::u32(theme::border));

    ImGui::SetCursorPos({8.F, 5.F});
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
        ? (app.active_job == JobKind::align ? "ALIGNING" : "TRAINING")
        : "READY";
    const float state_width = ImGui::CalcTextSize(state).x;
    ImGui::SameLine(std::max(0.F, ImGui::GetWindowWidth() - state_width - 14.F));
    ImGui::SetCursorPosY(10.F);
    theme::caption(state);
    ImGui::SetCursorPos({0, 36.F});

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
    ImGui::EndChild();

    // Console.
    const char* trailing = app.job.running() ? "FOLLOWING OUTPUT" : nullptr;
    theme::begin_panel("Console", {0, 0}, "CONSOLE", trailing, {0, 0});
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.031F, 0.033F, 0.039F, 1.F));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10.F, 8.F));
    ImGui::BeginChild("Log", {0, 0}, false, ImGuiWindowFlags_HorizontalScrollbar);
    ImGui::PopStyleVar();
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.66F, 0.68F, 0.72F, 1.F));
    if (app.log.console().empty())
        ImGui::TextUnformatted("Reconstruction output appears here.");
    else
        ImGui::TextUnformatted(app.log.console().c_str());
    ImGui::PopStyleColor();
    if (app.job.running()) ImGui::SetScrollHereY(1.F);
    ImGui::EndChild();
    ImGui::PopStyleColor();
    theme::end_panel();

    ImGui::EndChild();
}

Action draw_inspector(App& app, const float width) {
    Action action = Action::none;
    const bool busy = app.job.running();
    theme::begin_panel("Inspector", {width, 0}, "INSPECTOR");

    if (ImGui::CollapsingHeader("Project", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        theme::caption("Image source");
        ImGui::SetNextItemWidth(-30.F);
        ImGui::InputText(
            "##images", app.settings.images_dir.data(),
            app.settings.images_dir.size());
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button("...##pick_images", {24.F, 0})) {
            if (pick_folder(L"Select the capture image folder",
                            app.settings.images_dir))
                refresh_artifacts(app);
        }
        theme::caption("Project directory");
        ImGui::SetNextItemWidth(-30.F);
        if (ImGui::InputText(
                "##project", app.settings.project_dir.data(),
                app.settings.project_dir.size()))
            refresh_artifacts(app);
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button("...##pick_project", {24.F, 0})) {
            if (pick_folder(L"Select the project output folder",
                            app.settings.project_dir))
                refresh_artifacts(app);
        }
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
                "Shares a checkpoint directory between runs so training reuses\n"
                "the alignment you just reviewed instead of re-solving it.");
        ImGui::EndDisabled();

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
        ImGui::Checkbox("Learn normal field", &app.settings.normal_field);
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
        if (theme::toolbar_button(
                "Frame Scene", {-1.F, 28.F}, app.scene.has_points()))
            action = Action::frame_scene;
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
        ImGui::Spacing();
        ImGui::Checkbox("Colour by depth", &app.view_options.colour_by_depth);
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
                app.active_job == JobKind::align ? "Stop Alignment"
                                                 : "Stop Training",
                {-1.F, 40.F}))
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

    theme::end_panel();
    return action;
}

void draw_status_bar(const App& app, const float height) {
    const bool busy = app.job.running();
    const ImVec4 background = busy
        ? ImVec4(0.027F, 0.208F, 0.325F, 1.F)
        : ImVec4(0.086F, 0.090F, 0.102F, 1.F);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, background);
    ImGui::BeginChild("##status", {0, height}, false, ImGuiWindowFlags_NoScrollbar);

    const float centre_y = (height - ImGui::GetTextLineHeight()) * 0.5F;
    ImGui::SetCursorPos({12.F, centre_y});
    ImVec4 dot = theme::inactive;
    if (busy) dot = theme::accent;
    else if (app.monitor.stage() == Stage::failed) dot = theme::danger;
    else if (app.monitor.stage() == Stage::complete) dot = theme::success;
    theme::status_dot(dot, 7.F);

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

    // Progress bar plus percentage and ETA, right-aligned before the build tag.
    const char* tag = "AetherScan 0.2";
    const float tag_width = ImGui::CalcTextSize(tag).x;
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
            240.F, ImGui::GetWindowWidth() - tag_width - 28.F -
                       trailing_width - 12.F - bar_width);
        ImGui::SameLine(bar_x);
        ImGui::SetCursorPosY((height - 7.F) * 0.5F);
        theme::progress_track(
            {bar_width, 7.F}, fraction,
            fraction < 0.F ? theme::accent : ImVec4(0.55F, 0.84F, 1.F, 1.F));
        ImGui::SameLine(0.F, 12.F);
        ImGui::SetCursorPosY(centre_y);
        ImGui::TextUnformatted(trailing);
    }

    ImGui::SameLine(std::max(0.F, ImGui::GetWindowWidth() - tag_width - 14.F));
    ImGui::SetCursorPosY(centre_y);
    ImGui::PushStyleColor(
        ImGuiCol_Text, busy ? theme::text_bright : theme::text_faint);
    ImGui::TextUnformatted(tag);
    ImGui::PopStyleColor();

    ImGui::EndChild();
    ImGui::PopStyleColor();
}

}  // namespace

int main(const int argc, char** argv) {
    App app;
    app.smoke_mode = argc > 1 && std::string_view(argv[1]) == "--interop-smoke";

    std::snprintf(
        app.settings.images_dir.data(), app.settings.images_dir.size(),
        "D:\\ScanVideo\\ori_img\\images");
    std::snprintf(
        app.settings.project_dir.data(), app.settings.project_dir.size(),
        "D:\\ScanVideo\\ori_img\\aetherscan_gui");
    if (app.smoke_mode) {
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
    if (app.has_sparse)
        request_scene_load(
            app, app.layout.sparse_ply, app.layout.sparse_poses,
            "Sparse cloud");

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

        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin(
            "AetherScan", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse |
                ImGuiWindowFlags_NoBringToFrontOnFocus);

        app.settings.iterations = std::max(app.settings.iterations, 1);
        app.settings.preview_interval =
            std::max(app.settings.preview_interval, 1);
        app.settings.max_features = std::max(app.settings.max_features, 512);
        app.settings.geometry_from_iter =
            std::max(app.settings.geometry_from_iter, 0);

        draw_title_bar(app);
        Action action = draw_toolbar(app);

        constexpr float status_height = 34.F;
        constexpr float left_width = 272.F;
        constexpr float right_width = 336.F;
        constexpr float console_height = 178.F;
        const float workspace_height =
            ImGui::GetContentRegionAvail().y - status_height;

        ImGui::BeginChild(
            "##workspace", {0, workspace_height}, false,
            ImGuiWindowFlags_NoScrollbar);
        draw_left_panel(app, left_width);
        ImGui::SameLine(0.F, 4.F);
        const float centre_width =
            ImGui::GetContentRegionAvail().x - right_width - 4.F;
        draw_centre_column(app, centre_width, console_height);
        ImGui::SameLine(0.F, 4.F);
        const Action inspector_action = draw_inspector(app, right_width);
        if (action == Action::none) action = inspector_action;
        ImGui::EndChild();

        draw_status_bar(app, status_height);

        switch (action) {
            case Action::align:
                if (!app.smoke_mode) start_align(app);
                break;
            case Action::train:
                if (!app.smoke_mode) start_train(app, false);
                break;
            case Action::stop:
                app.job.stop();
                set_message(app, "Stopping...", theme::warning);
                break;
            case Action::reveal: reveal_in_explorer(app.layout.root); break;
            case Action::frame_scene: app.camera.frame(app.scene); break;
            case Action::none: break;
        }

        ImGui::End();
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
