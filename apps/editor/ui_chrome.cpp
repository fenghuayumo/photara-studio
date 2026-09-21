#include "ui.hpp"
#include "app.hpp"
#include "file_dialogs.hpp"
#include "i18n.hpp"
#include "icons.hpp"

#include "imgui_internal.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#ifndef PHOTARA_CLI_PATH
#define PHOTARA_CLI_PATH "photara"
#endif
#ifndef PHOTARA_VERSION
#define PHOTARA_VERSION "0.2.0"
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace editor {
using i18n::tr;

struct WorkflowCrumbs {
    std::array<const char*, 5> labels{};
    int count{1};
    int current{};
};

WorkflowCrumbs workflow_crumbs(const App& app) {
    WorkflowCrumbs crumbs;
    crumbs.labels[0] = tr("Images");
    const bool have_images =
        app.settings.images_dir[0] != '\0' || has_external_dataset(app);
    if (!have_images) return crumbs;

    crumbs.labels[1] = tr("Alignment");
    crumbs.count = 2;
    crumbs.current = 1;

    const bool training =
        app.job.running() && app.active_job == JobKind::train;
    const bool aligning =
        app.job.running() && app.active_job == JobKind::align;
    const bool densing =
        app.job.running() && app.active_job == JobKind::dense;
    const bool texturing =
        app.job.running() && app.active_job == JobKind::texture;
    const bool aligned = app.has_sparse || has_external_dataset(app);
    if (aligning) {
        crumbs.current = 1;
        return crumbs;
    }
    if (!aligned && !training && !densing && !texturing) return crumbs;

    const bool mvs_route =
        !training && (mesh_from_mvs(app.settings) || densing ||
                      (app.settings.mesh_source == 1 && app.settings.build_mesh));
    if (mvs_route) {
        crumbs.labels[2] = tr("MVS Mesh");
        crumbs.labels[3] = tr("Texture Baking");
        crumbs.count = 4;
        if (texturing || app.has_texture) crumbs.current = 3;
        else if (densing || app.has_mesh) crumbs.current = 2;
        else crumbs.current = 1;
        return crumbs;
    }

    crumbs.labels[2] = tr("3DGS");
    crumbs.labels[3] = tr("Extract Mesh");
    crumbs.labels[4] = tr("Texture Baking");
    crumbs.count = 5;
    const bool extracting =
        training && (app.monitor.stage() == Stage::meshing ||
                     app.monitor.stage() == Stage::texturing);
    if (texturing || app.has_texture) crumbs.current = 4;
    else if (app.has_mesh || extracting) crumbs.current = 3;
    else if (training || app.has_model) crumbs.current = 2;
    else crumbs.current = 1;
    return crumbs;
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

    ImGui::DockBuilderDockWindow("###Scene", dock_left);
    ImGui::DockBuilderDockWindow("###Viewport", dock_main);
    ImGui::DockBuilderDockWindow("###Console", dock_bottom);
    ImGui::DockBuilderDockWindow("###Inspector", dock_right);
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
    if (ImGui::BeginMenu(tr("File"))) {
        if (ImGui::MenuItem(tr("New Project"), "Ctrl+N", false, !busy)) {
            new_project(app);
        }
        if (ImGui::MenuItem(tr("Select Image Folder..."), "Ctrl+O", false, !busy)) {
            select_image_folder(app);
        }
        if (busy && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", tr(k_busy_change_capture_tooltip));
        if (ImGui::MenuItem(tr("Select Video..."), nullptr, false, !busy)) {
            select_video_file(app);
        }
        if (busy && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", tr(k_busy_change_capture_tooltip));
        if (ImGui::MenuItem(tr("Open Project..."), "Ctrl+Shift+O", false, !busy)) {
            select_project_folder(app);
        }
        if (ImGui::MenuItem(tr("Save Project"), "Ctrl+S", false, !busy)) {
            save_project(app);
        }
        if (ImGui::MenuItem(tr("Save Project As..."), "Ctrl+Shift+S", false, !busy)) {
            save_project_as(app);
        }
        ImGui::Separator();
        if (ImGui::MenuItem(
                tr("Export SfM Alignment..."), nullptr, false,
                !busy && can_export_alignment(app)))
            open_alignment_export_panel(app);
        if (ImGui::MenuItem(
                tr("Export Splat..."), nullptr, false,
                !busy && can_export_model(app)))
            open_splat_export_panel(app);
        if (ImGui::MenuItem(
                tr("Export Mesh..."), nullptr, false,
                !busy && can_export_mesh_file(app)))
            open_mesh_export_panel(app);
        if (ImGui::MenuItem(
                tr("Open Output Folder"), nullptr, false,
                app.layout.root.has_filename()))
            action = Action::reveal;
        ImGui::Separator();
        if (ImGui::MenuItem(tr("Exit"), "Alt+F4")) app.close_requested = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu(tr("Reconstruction"))) {
        if (ImGui::MenuItem(
                app.has_sparse ? tr("Re-align Photos") : tr("Align Photos"), nullptr,
                false,
                !busy && app.settings.images_dir[0] != '\0' &&
                    !has_external_dataset(app)))
            action = Action::align;
        if (ImGui::MenuItem(
                mesh_from_gaussians(app.settings) ? tr("Train 3DGS + Mesh")
                                                  : tr("Train 3DGS"),
                nullptr, false,
                !busy && (app.settings.images_dir[0] != '\0' ||
                          has_external_dataset(app))))
            action = Action::train;
        if (ImGui::MenuItem(
                tr("Extract Mesh"), nullptr, false,
                !busy && alignment_ready(app)))
            action = Action::dense;
        if (ImGui::MenuItem(
                app.has_texture ? tr("Re-bake Texture") : tr("Bake Texture"), nullptr,
                false, !busy && app.has_mesh))
            action = Action::texture;
        if (ImGui::MenuItem(
                tr("Export SfM Alignment..."), nullptr, false,
                !busy && can_export_alignment(app)))
            open_alignment_export_panel(app);
        ImGui::Separator();
        if (ImGui::MenuItem(
                tr("Clear Reconstruction Results..."), nullptr, false,
                !busy && !app.loading_scene &&
                    has_reconstruction_result(app)))
            app.show_clear_results = true;
        ImGui::Separator();
        if (app.job.paused()) {
            if (ImGui::MenuItem(tr("Resume Job"), "Esc", false, busy))
                action = Action::resume;
        } else if (ImGui::MenuItem(tr("Pause Job"), "Esc", false, busy)) {
            action = Action::pause;
        }
        if (ImGui::MenuItem(tr("Stop Job"), "Shift+Esc", false, busy))
            action = Action::stop;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu(tr("View"))) {
        ImGui::MenuItem(tr("Scene"), nullptr, &app.show_scene);
        ImGui::MenuItem(tr("Viewport"), nullptr, &app.show_viewport);
        ImGui::MenuItem(tr("Inspector"), nullptr, &app.show_inspector);
        ImGui::MenuItem(tr("Console"), nullptr, &app.show_console);
        ImGui::MenuItem(tr("Status Bar"), nullptr, &app.show_status_bar);
        ImGui::Separator();
        if (ImGui::MenuItem(
                tr("3D Scene"), "3",
                app.workspace == ViewportWorkspace::scene_3d))
            set_viewport_workspace(app, ViewportWorkspace::scene_3d);
        if (ImGui::MenuItem(
                tr("2D Image QA"), "2",
                app.workspace == ViewportWorkspace::image_2d))
            set_viewport_workspace(app, ViewportWorkspace::image_2d);
        ImGui::Separator();
        if (ImGui::MenuItem(
                tr("Show Cameras"), nullptr, app.view_options.show_views))
            set_camera_overlays(
                app.view_options, !app.view_options.show_views);
        ImGui::MenuItem(tr("Show Ground Grid"), nullptr, &app.view_options.show_grid);
        ImGui::MenuItem(tr("Show Origin Axes"), nullptr, &app.view_options.show_axes);
        ImGui::MenuItem(
            tr("Show Reconstruction Region"), nullptr,
            &app.view_options.show_region);
        ImGui::Separator();
        if (ImGui::BeginMenu(tr("Language"))) {
            for (int i = 0; i < 4; ++i) {
                const auto lang = static_cast<i18n::Language>(i);
                if (ImGui::MenuItem(
                        i18n::native_name(lang), nullptr,
                        i18n::language() == lang)) {
                    i18n::set_language(lang);
                    if (GLFWwindow* window = glfwGetCurrentContext())
                        glfwSetWindowTitle(
                            window, tr("Photara Studio"));
                }
            }
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem(tr("Reset Layout"))) {
            app.show_scene = true;
            app.show_viewport = true;
            app.show_console = true;
            app.show_inspector = true;
            app.show_status_bar = true;
            app.reset_dock_layout = true;
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu(tr("Help"))) {
        if (ImGui::MenuItem(tr("Viewport Controls"))) app.show_controls = true;
        ImGui::Separator();
        if (ImGui::MenuItem(tr("About Photara Studio"))) app.show_about = true;
        ImGui::EndMenu();
    }

    ImGui::SameLine(0.F, 28.F);
    ImGui::PushStyleColor(ImGuiCol_Text, theme::accent);
    ImGui::TextUnformatted("PHOTARA");
    ImGui::PopStyleColor();
    ImGui::SameLine(0.F, 6.F);
    ImGui::TextUnformatted("STUDIO");

    const WorkflowCrumbs crumbs = workflow_crumbs(app);
    ImGui::SameLine(0.F, 22.F);
    for (int i = 0; i < crumbs.count; ++i) {
        if (i > 0) {
            ImGui::SameLine(0.F, 8.F);
            theme::caption("\xE2\x80\xBA");
            ImGui::SameLine(0.F, 8.F);
        }
        const bool reached = i <= crumbs.current;
        ImGui::PushStyleColor(
            ImGuiCol_Text, i == crumbs.current ? theme::accent
                                               : (reached ? theme::text_muted
                                                          : theme::text_faint));
        ImGui::TextUnformatted(crumbs.labels[static_cast<std::size_t>(i)]);
        ImGui::PopStyleColor();
    }

    const std::string project = app.layout.project_file.empty()
        ? (app.layout.root.filename().empty()
               ? std::string(tr("Untitled Project"))
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
    if (ImGui::Begin(i18n::id("Viewport Controls", "###ViewportControls"), &app.show_controls,
                     ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted(tr("3D camera"));
        ImGui::Separator();
        ImGui::BulletText("%s", tr("LMB drag: orbit"));
        ImGui::BulletText("%s", tr("MMB or Shift+LMB drag: pan"));
        ImGui::BulletText("%s", tr("RMB drag: fly look"));
        ImGui::BulletText("%s", tr("RMB + WASD/QE: fly; Shift accelerates"));
        ImGui::BulletText("%s", tr("Mouse wheel: dolly; F: frame reconstruction"));
        ImGui::BulletText(
            "%s", tr("Region gizmo: drag RGB arrows to move the box, face dots to resize"));
        ImGui::BulletText("%s", tr("Shift: fine control; Ctrl: snap"));
        ImGui::BulletText("%s", tr("Double-click a point: orbit around it"));
        ImGui::BulletText("%s", tr("Double-click a camera frustum: look through it"));
        ImGui::Spacing();
        ImGui::TextUnformatted(tr("2D image QA"));
        ImGui::Separator();
        ImGui::BulletText("%s", tr("Wheel: zoom; LMB/MMB drag: pan; double-click or F: fit"));
        ImGui::BulletText("%s", tr("Left / Right: previous and next capture"));
        ImGui::BulletText("%s", tr("Compare: drag the vertical handle to wipe GT vs 3DGS"));
        ImGui::BulletText("%s", tr("2 / 3: switch 2D image QA and 3D scene"));
    }
    ImGui::End();
}

std::filesystem::path studio_icon_path() {
    std::filesystem::path exe_dir;
#if defined(_WIN32)
    wchar_t buffer[MAX_PATH]{};
    const DWORD n = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    if (n > 0 && n < MAX_PATH)
        exe_dir = std::filesystem::path(buffer).parent_path();
#endif
    const std::filesystem::path candidates[] = {
        exe_dir / "icon.png",
        exe_dir / "icons" / "icon.png",
        exe_dir / ".." / ".." / ".." / "apps" / "icons" / "icon.png",
    };
    std::error_code error;
    for (const auto& path : candidates) {
        if (!path.empty() && std::filesystem::is_regular_file(path, error))
            return path;
    }
    return {};
}

void ensure_about_icon(App& app) {
    if (app.about_icon_ready || app.about_icon.descriptor != nullptr)
        return;
    app.about_icon_ready = true;
    const auto path = studio_icon_path();
    if (path.empty()) return;
    try {
        app.about_icon.upload(photara::io::load_rgb(path));
    } catch (const std::exception&) {
        app.about_icon.reset();
    }
}

void draw_about_window(App& app) {
    if (!app.show_about) return;
    ensure_about_icon(app);
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        viewport->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));
    ImGui::SetNextWindowSize({420.F, 0.F}, ImGuiCond_Appearing);
    if (!ImGui::Begin(
            i18n::id("About Photara Studio", "###AboutPhotaraStudio"),
            &app.show_about, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::End();
        return;
    }

    const float width = ImGui::GetContentRegionAvail().x;
    if (app.about_icon.descriptor != nullptr) {
        const float icon = 96.F;
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const ImVec2 min{origin.x + (width - icon) * 0.5F, origin.y};
        const ImVec2 max{min.x + icon, min.y + icon};
        ImGui::GetWindowDrawList()->AddImageRounded(
            reinterpret_cast<ImTextureID>(app.about_icon.descriptor), min, max,
            {0.F, 0.F}, {1.F, 1.F}, IM_COL32_WHITE, 20.F);
        ImGui::Dummy({width, icon});
        ImGui::Spacing();
    }

    auto centred = [width](const char* text, const ImVec4& colour) {
        const float text_w = ImGui::CalcTextSize(text).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (width - text_w) * 0.5F);
        ImGui::PushStyleColor(ImGuiCol_Text, colour);
        ImGui::TextUnformatted(text);
        ImGui::PopStyleColor();
    };
    centred("Photara Studio", theme::text_bright);
    centred(
        tr("Photogrammetry and 3D Gaussian splatting"),
        theme::text_muted);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    theme::metric("Version", PHOTARA_VERSION);
    theme::metric("Pipeline", "SfM / MVS / 3DGS / mesh / texture");
    theme::metric("Backends", "CUDA / Vulkan");
    theme::metric("Developer", "Photara");
    theme::metric("Copyright", "© 2026 Photara");
    ImGui::Spacing();
    const float button_w = 120.F;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (width - button_w) * 0.5F);
    if (theme::toolbar_button(tr("Close"), {button_w, 0.F}))
        app.show_about = false;
    ImGui::End();
}

ClearResultsAction draw_clear_results_modal(App& app) {
    ClearResultsAction action = ClearResultsAction::none;
    constexpr const char* popup_id = "###ClearReconstructionResults";
    if (app.show_clear_results) {
        ImGui::OpenPopup(popup_id);
        app.show_clear_results = false;
    }

    ImGui::SetNextWindowSize({470.F, 0.F}, ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(
            i18n::id("Clear Reconstruction Results", popup_id), nullptr,
            ImGuiWindowFlags_AlwaysAutoResize))
        return action;

    ImGui::TextUnformatted(tr("Clear the current reconstruction?"));
    ImGui::Spacing();
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 440.F);
    ImGui::TextColored(
        theme::warning, "%s",
        tr("Clear View Only removes the old result from the editor but keeps all "
           "project files."));
    ImGui::TextColored(
        theme::danger, "%s",
        tr("Delete Generated Results permanently removes sparse clouds, trained "
           "models, meshes, logs, and the reconstruction cache from the current "
           "project. Source images are never deleted."));
    ImGui::PopTextWrapPos();
    if (!app.layout.root.empty()) {
        ImGui::Spacing();
        theme::caption(tr("Current project"));
        const std::string root_utf8 = path_to_utf8(app.layout.root);
        ImGui::TextWrapped("%s", root_utf8.c_str());
    }
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (theme::toolbar_button(tr("Clear View Only"), {126.F, 30.F})) {
        action = ClearResultsAction::clear_view;
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (theme::danger_button(tr("Delete Generated Results"), {190.F, 30.F})) {
        action = ClearResultsAction::delete_generated;
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button(tr("Cancel"), {92.F, 30.F})) ImGui::CloseCurrentPopup();
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
            "##image_folder", icons::Icon::folder, tr("Image Folder"),
            {124.F, 32.F}, icons::ButtonStyle::normal, !busy, false,
            busy ? tr(k_busy_change_capture_tooltip)
                 : tr("Select capture image folder, or drop photos / .asfm / .ascan on the viewport"))) {
        select_image_folder(app);
    }
    ImGui::SameLine();
    if (icons::labeled_button(
            "##video_file", icons::Icon::camera, tr("Video"),
            {88.F, 32.F}, icons::ButtonStyle::normal, !busy, false,
            busy ? tr(k_busy_change_capture_tooltip)
                 : tr("Select a capture video. Align Photos extracts sharp frames, then runs SfM."))) {
        select_video_file(app);
    }
    ImGui::SameLine();

    // Step 1: multi-view alignment.
    const bool aligning = busy && app.active_job == JobKind::align;
    if (aligning) {
        icons::labeled_button(
            "##aligning", icons::Icon::align,
            app.job.paused() ? tr("Paused") : tr("Aligning..."), {132.F, 32.F},
            icons::ButtonStyle::primary, false, true,
            tr("Use Stop to abort alignment so you can change images or parameters."));
    } else if (icons::labeled_button(
                   "##align", icons::Icon::align,
                   app.has_sparse ? tr("Re-align Photos") : tr("Align Photos"),
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
            app.job.paused() ? tr("Paused") : tr("Training..."), {126.F, 32.F},
            icons::ButtonStyle::primary, false, true,
            tr("Use Stop to abort training so you can change images or parameters."));
    } else {
        const bool ready = !busy && images_ready;
        if (app.has_sparse) {
            if (icons::labeled_button(
                    "##train", icons::Icon::train, tr("Train 3DGS"),
                    {126.F, 32.F}, icons::ButtonStyle::primary, ready))
                action = Action::train;
        } else if (icons::labeled_button(
                       "##train", icons::Icon::train, tr("Train 3DGS"),
                       {126.F, 32.F}, icons::ButtonStyle::normal, ready)) {
            action = Action::train;
        }
        if (ImGui::IsItemHovered()) {
            if (external_dataset)
                ImGui::SetTooltip(
                    "%s", tr("Train directly from the imported cameras. SfM is skipped."));
            else if (!app.has_sparse)
                ImGui::SetTooltip(
                    "%s",
                    tr("No alignment yet. Training will run Structure from Motion "
                       "first, then optimise Gaussians."));
            else
                ImGui::SetTooltip(
                    "%s",
                    tr("Start 3DGS from the current aligned cameras and sparse cloud."));
        }
    }
    ImGui::SameLine();

    ImGui::AlignTextToFramePadding();
    ImGui::BeginDisabled(busy);
    ImGui::Checkbox(tr("Build Mesh"), &app.settings.build_mesh);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        const char* mesh_tip = busy
            ? tr("Stop the running job first to change reconstruction options.")
            : (app.settings.mesh_source == 1
                   ? tr("Extract a surface with the MVS mesh pipeline after alignment.\n"
                     "Choose Extract Mesh (MVS) or From Gaussians in the Mesh panel.")
                   : tr("Extract a surface after 3DGS training.\n"
                     "From Gaussians enables depth-normal and multi-view geometry\n"
                     "losses during optimisation. Switch to Extract Mesh (MVS) in the\n"
                     "Mesh panel for a photogrammetry surface."));
        ImGui::SetTooltip("%s", mesh_tip);
    }
    ImGui::SameLine(0.F, 14.F);

    if (icons::labeled_button(
            "##open_output", icons::Icon::output, tr("Open Output"),
            {116.F, 32.F}, icons::ButtonStyle::normal,
            app.layout.root.has_filename()))
        action = Action::reveal;
    if (busy) {
        ImGui::SameLine();
        if (app.job.paused()) {
            if (icons::labeled_button(
                    "##resume", icons::Icon::play, tr("Resume"), {96.F, 32.F},
                    icons::ButtonStyle::primary, true, false,
                    tr("Continue the paused reconstruction.")))
                action = Action::resume;
        } else if (icons::labeled_button(
                       "##pause", icons::Icon::pause, tr("Pause"), {96.F, 32.F},
                       icons::ButtonStyle::normal, true, false,
                       tr(k_pause_job_tooltip))) {
            action = Action::pause;
        }
        ImGui::SameLine();
        if (icons::labeled_button(
                "##stop", icons::Icon::stop, tr("Stop"), {88.F, 32.F},
                icons::ButtonStyle::danger, true, false, tr(k_stop_job_tooltip)))
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
void draw_console_panel(App& app) {
    if (!app.show_console) return;
    editor::draw_console(
        app.show_console, app.console, app.log, app.job.running(),
        app.active_job, app.monitor.stage());
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
            ? (std::string(tr("Paused")) + "  |  " + app.monitor.headline())
            : app.monitor.headline();
        ImGui::TextUnformatted(headline.c_str());
    } else if (!app.message.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, app.message_colour);
        ImGui::TextUnformatted(app.message.c_str());
        ImGui::PopStyleColor();
    } else {
        ImGui::TextUnformatted(tr("Ready"));
    }

    // Persistent right-hand telemetry: backend and build tag always remain
    // visible; idle mode also shows viewport, point and active-tool state.
    const char* tag = "Photara Studio 0.2";
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
        char elapsed_text[64];
        std::snprintf(
            elapsed_text, sizeof(elapsed_text), tr("Elapsed %s"),
            format_duration(elapsed).c_str());
        const float elapsed_width = status_segment_width(elapsed_text);
        right -= elapsed_width + 22.F;
        draw_status_separator(right + elapsed_width + 11.F, height);
        draw_status_segment(
            right, centre_y, icons::Icon::clock, elapsed_text,
            busy ? theme::text_bright : theme::text_muted);
    }

    if (busy) {
        const float fraction = app.monitor.fraction();
        const double eta = app.monitor.eta_seconds();
        char trailing[96];
        if (paused) {
            if (fraction >= 0.F)
                std::snprintf(
                    trailing, sizeof(trailing), tr("%3.0f%%   paused"),
                    fraction * 100.F);
            else
                std::snprintf(trailing, sizeof(trailing), "%s", tr("paused"));
        } else if (fraction >= 0.F && eta >= 0.0)
            std::snprintf(
                trailing, sizeof(trailing), tr("%3.0f%%   ETA %s"), fraction * 100.F,
                format_duration(eta).c_str());
        else if (fraction >= 0.F)
            std::snprintf(trailing, sizeof(trailing), "%3.0f%%", fraction * 100.F);
        else
            std::snprintf(trailing, sizeof(trailing), "%s", tr("working"));

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
            char faces[64];
            std::snprintf(
                faces, sizeof(faces), tr("%s faces"),
                format_count(app.mesh.faces.size()).c_str());
            const float faces_width = status_segment_width(faces);
            right -= faces_width + 22.F;
            draw_status_separator(right + faces_width + 11.F, height);
            draw_status_segment(
                right, centre_y, icons::Icon::cube, faces);
        } else if (app.scene.has_points()) {
            char points[64];
            std::snprintf(
                points, sizeof(points), tr("%s pts"),
                format_count(app.scene.points.size()).c_str());
            const float points_width = status_segment_width(points);
            right -= points_width + 22.F;
            draw_status_separator(right + points_width + 11.F, height);
            draw_status_segment(
                right, centre_y, icons::Icon::points, points);
        }

        const char* view = tr("Points");
        icons::Icon view_icon = icons::Icon::points;
        if (app.workspace == ViewportWorkspace::image_2d) {
            view = tr("2D Image");
            view_icon = icons::Icon::view2d;
        } else if (app.view_mode == VisualizationMode::splat) {
            view = tr("Splat");
            view_icon = icons::Icon::splat;
        } else if (app.view_mode == VisualizationMode::rings) {
            view = tr("Rings");
            view_icon = icons::Icon::rings;
        } else if (app.view_mode == VisualizationMode::mesh) {
            view = tr("Mesh");
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

}  // namespace editor
