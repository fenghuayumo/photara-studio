#include "ui.hpp"
#include "app.hpp"
#include "file_dialogs.hpp"
#include "i18n.hpp"
#include "icons.hpp"
#include "vulkan_backend.hpp"
#include "core/version.hpp"
#include "io/image.hpp"

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

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace editor {
using i18n::tr;

struct WorkflowCrumbs {
    std::array<const char*, 8> labels{};
    int count{1};
    int current{};
};

WorkflowCrumbs workflow_crumbs(const App& app) {
    WorkflowCrumbs crumbs;
    crumbs.labels[0] = tr("Images");
    const bool have_images =
        app.settings.images_dir[0] != '\0' || has_external_dataset(app);
    if (!have_images) return crumbs;

    const bool video = is_video_source(app.settings);
    const bool masks = app.settings.sam_masks && !has_external_dataset(app);
    int index = 1;
    int frames_index = -1;
    int masks_index = -1;
    if (video) {
        frames_index = index;
        crumbs.labels[static_cast<std::size_t>(index++)] = tr("Frames");
    }
    if (masks) {
        masks_index = index;
        crumbs.labels[static_cast<std::size_t>(index++)] = tr("Masks");
    }
    const int align_index = index;
    crumbs.labels[static_cast<std::size_t>(index++)] = tr("Alignment");
    crumbs.count = index;
    crumbs.current = video ? frames_index : (masks ? masks_index : align_index);

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
        const Stage stage = app.monitor.stage();
        if (stage == Stage::preparing && frames_index >= 0)
            crumbs.current = frames_index;
        else if (stage == Stage::masking && masks_index >= 0)
            crumbs.current = masks_index;
        else
            crumbs.current = align_index;
        return crumbs;
    }
    if (!aligned && !training && !densing && !texturing) return crumbs;

    // Gaussian extraction runs as a dense job too, but must not read as MVS.
    const bool mvs_route =
        !training && (mesh_from_mvs(app.settings) ||
                      (densing && app.settings.mesh_source == 1));
    if (mvs_route) {
        crumbs.labels[static_cast<std::size_t>(index++)] = tr("MVS Mesh");
        crumbs.labels[static_cast<std::size_t>(index++)] = tr("Texture Baking");
        crumbs.count = index;
        if (texturing || app.has_texture) crumbs.current = index - 1;
        else if (densing || app.has_mesh) crumbs.current = index - 2;
        else crumbs.current = align_index;
        return crumbs;
    }

    crumbs.labels[static_cast<std::size_t>(index++)] = tr("3DGS");
    crumbs.labels[static_cast<std::size_t>(index++)] = tr("Extract Mesh");
    crumbs.labels[static_cast<std::size_t>(index++)] = tr("Texture Baking");
    crumbs.count = index;
    const bool extracting =
        training && (app.monitor.stage() == Stage::meshing ||
                     app.monitor.stage() == Stage::texturing);
    if (texturing || app.has_texture) crumbs.current = index - 1;
    else if (app.has_mesh || extracting) crumbs.current = index - 2;
    else if (training || app.has_model) crumbs.current = index - 3;
    else crumbs.current = align_index;
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
        dock_main, ImGuiDir_Down, 0.34F, &dock_bottom, &dock_main);

    ImGui::DockBuilderDockWindow("###Scene", dock_left);
    ImGui::DockBuilderDockWindow("###Viewport", dock_main);
    ImGui::DockBuilderDockWindow("###Console", dock_bottom);
    ImGui::DockBuilderDockWindow("###CameraPath", dock_bottom);
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
        viewport->WorkPos.x, viewport->WorkPos.y + (model_viewer_active(app) ? 0.F : k_toolbar_height)};
    const ImVec2 size{
        viewport->WorkSize.x,
        std::max(1.F, viewport->WorkSize.y - (model_viewer_active(app) ? 0.F : k_toolbar_height) - status_h)};

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
    // Direct keyboard shortcuts do not inherit ImGui's disabled widget state.
    shortcuts::suspended = app.show_preferences || app.show_welcome;
    if (app.pending_train_prepare.valid()) return action;
    const bool busy = app.job.running();
    const ImGuiIO& io = ImGui::GetIO();
    if (!io.WantTextInput && !shortcuts::suspended) {
        if (shortcuts::pressed("new_project") && !busy) new_project(app);
        if (shortcuts::pressed("open_images") && !busy) select_image_folder(app);
        if (shortcuts::pressed("open_project") && !busy) select_project_folder(app);
        if (shortcuts::pressed("save") && (!busy || app.active_job == JobKind::train)) save_project(app);
        if (shortcuts::pressed("save_as") && !busy) save_project_as(app);
        if (ImGui::IsKeyPressed(ImGuiKey_Escape) && busy)
            action = io.KeyShift ? Action::stop : app.job.paused() ? Action::resume : Action::pause;
        if (shortcuts::pressed("workspace_2d")) set_viewport_workspace(app, ViewportWorkspace::image_2d);
        if (shortcuts::pressed("workspace_3d")) set_viewport_workspace(app, ViewportWorkspace::scene_3d);
        if (shortcuts::pressed("preferences")) open_preferences(app, -1);
    }

    if (!ImGui::BeginMainMenuBar()) return action;
    if (ImGui::BeginMenu(tr("Workspace"))) {
        if (ImGui::MenuItem(tr("Model viewer"), nullptr, model_viewer_active(app))) app.model_workspace = 1;
        if (ImGui::MenuItem(tr("Reconstruction"), nullptr, !model_viewer_active(app))) app.model_workspace = 0;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu(tr("File"))) {
        if (ImGui::MenuItem(tr("New Project"), shortcuts::text("new_project"), false, !busy)) {
            new_project(app);
        }
        if (ImGui::MenuItem(tr("Select Image Folder..."), shortcuts::text("open_images"), false, !busy)) {
            select_image_folder(app);
        }
        if (busy && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", tr(k_busy_change_capture_tooltip));
        if (ImGui::MenuItem(tr("Select Video..."), nullptr, false, !busy)) {
            select_video_file(app);
        }
        if (busy && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", tr(k_busy_change_capture_tooltip));
        if (ImGui::MenuItem(tr("Open Project..."), shortcuts::text("open_project"), false, !busy)) {
            select_project_folder(app);
        }
        if (ImGui::MenuItem(tr("Save Project"), shortcuts::text("save"), false, !app.saving_training && (!busy || app.active_job == JobKind::train))) {
            save_project(app);
        }
        if (ImGui::MenuItem(tr("Save Project As..."), shortcuts::text("save_as"), false, !busy)) {
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
        if (ImGui::MenuItem(tr("Preferences..."), shortcuts::text("preferences")))
            open_preferences(app, -1);
        if (ImGui::MenuItem(tr("Exit"), "Alt+F4")) app.close_requested = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu(tr("Selection"))) {
        const bool splats =
            app.splat_edit.bound_to(app.splat_renderer->source_key()) &&
            app.splat_edit.gaussian_count() > 0;
        const bool any_selected = splats && app.splat_edit.selected_count() > 0;
        if (ImGui::MenuItem(tr("Select All"), shortcuts::text("select_all"), false, splats))
            app.splat_edit.select_all();
        if (ImGui::MenuItem(tr("Deselect All"), shortcuts::text("clear"), false, any_selected))
            app.splat_edit.clear_selection();
        if (ImGui::MenuItem(tr("Invert Selection"), shortcuts::text("invert"), false, splats))
            app.splat_edit.invert_selection();
        ImGui::Separator();
        if (ImGui::MenuItem(tr("Delete Selected"), "Delete", false, any_selected))
            app.splat_edit.delete_selected(app);
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
        if (!busy && app.training_checkpoint_info &&
            ImGui::MenuItem(tr("Resume Training")))
            action = Action::continue_training;
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
            if (ImGui::MenuItem(resume_job_label(app.active_job), "Esc", false, busy))
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
        ImGui::MenuItem(tr("Reconstruction settings"), nullptr, &app.show_inspector);
        ImGui::MenuItem(tr("Console"), nullptr, &app.show_console);
        ImGui::MenuItem(tr("Camera Path"), nullptr, &app.show_camera_path);
        ImGui::MenuItem(tr("Status Bar"), nullptr, &app.show_status_bar);
        ImGui::Separator();
        if (ImGui::MenuItem(
                tr("Video Preview"), nullptr, app.video_window.show) &&
            app.video_window.show)
            open_video_window(app);
        if (ImGui::MenuItem(
                tr("Mask Editor"), nullptr, app.mask_editor.show) &&
            app.mask_editor.show)
            open_mask_editor(app);
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
        if (ImGui::MenuItem(tr("Welcome to Photara"))) app.show_welcome = true;
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
               : path_to_utf8(app.layout.root.filename()))
        : path_to_utf8(app.layout.project_file.filename());
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
#else
    char buffer[4096]{};
    const ssize_t n = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (n > 0)
        exe_dir = std::filesystem::path(
                      std::string(buffer, buffer + n))
                      .parent_path();
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

void apply_studio_window_icon(GLFWwindow* window) {
    if (window == nullptr) return;
    const auto path = studio_icon_path();
    if (path.empty()) return;
    photara::io::RgbImage image;
    try {
        image = photara::io::load_rgb(path);
    } catch (const std::exception&) {
        return;
    }
    const std::size_t count =
        static_cast<std::size_t>(image.width) * image.height;
    if (image.width == 0 || image.height == 0 ||
        image.pixels.size() < count * 3)
        return;
    std::vector<unsigned char> rgba(count * 4);
    for (std::size_t i = 0; i < count; ++i) {
        rgba[i * 4] = image.pixels[i * 3];
        rgba[i * 4 + 1] = image.pixels[i * 3 + 1];
        rgba[i * 4 + 2] = image.pixels[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    GLFWimage icon{};
    icon.width = static_cast<int>(image.width);
    icon.height = static_cast<int>(image.height);
    icon.pixels = rgba.data();
    glfwSetWindowIcon(window, 1, &icon);
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

namespace {
// Native vector artwork: a reconstructed terrain made of scan lines and points.
// No textures or GPU uploads are needed by the startup window.
void draw_welcome_cover(ImDrawList* draw, ImVec2 origin, float width, float height) {
    const ImVec2 end{origin.x + width, origin.y + height};
    draw->PushClipRect(origin, end, true);
    draw->AddRectFilledMultiColor(origin, end, IM_COL32(15, 32, 47, 255),
        IM_COL32(21, 52, 68, 255), IM_COL32(9, 22, 34, 255), IM_COL32(12, 20, 30, 255));
    for (int row = 0; row < 27; ++row) {
        ImVec2 previous{};
        for (int column = 0; column < 74; ++column) {
            const float x = static_cast<float>(column) / 73.F;
            const float z = static_cast<float>(row) / 26.F;
            const float ridge = std::exp(-std::pow((x - 0.68F) * 5.F, 2.F)) *
                (0.55F + 0.22F * std::sin(x * 24.F + z * 4.F) +
                 0.15F * std::cos(x * 43.F - z * 5.F));
            const ImVec2 point{origin.x + width * (0.26F + x * 0.83F - z * 0.12F),
                origin.y + height * (0.51F + z * 0.64F - ridge * (0.82F - z * 0.40F))};
            const int alpha = 35 + static_cast<int>(z * 90.F);
            if (column > 0)
                draw->AddLine(previous, point, IM_COL32(66, 167, 190, alpha / 2));
            draw->AddCircleFilled(point, row % 3 == 0 ? 1.3F : 0.8F,
                column > 44 && column < 53 ? IM_COL32(222, 185, 119, alpha + 55)
                                         : IM_COL32(94, 204, 228, alpha + 35), 5);
            previous = point;
        }
    }
    // A soft left-hand veil keeps the wordmark readable over the point cloud.
    draw->AddRectFilledMultiColor(origin, end, IM_COL32(12, 23, 34, 235),
        IM_COL32(12, 23, 34, 0), IM_COL32(12, 23, 34, 0), IM_COL32(12, 23, 34, 235));
    draw->PopClipRect();
}

bool welcome_action(const char* id, icons::Icon icon, const char* title,
                    const char* subtitle, float width, float scale) {
    const ImVec2 start = ImGui::GetCursorScreenPos();
    const ImVec2 size{width, 62.F * scale};
    const bool clicked = ImGui::Button(id, size);
    const bool hover = ImGui::IsItemHovered();
    auto* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(start, {start.x + size.x, start.y + size.y},
        theme::u32(hover ? theme::surface_3 : theme::surface_2), 7.F * scale);
    draw->AddRect(start, {start.x + size.x, start.y + size.y},
        theme::u32(hover ? theme::accent : theme::border), 7.F * scale);
    icons::draw(draw, icon, {start.x + 14.F * scale, start.y + 19.F * scale},
        {start.x + 36.F * scale, start.y + 41.F * scale}, theme::u32(theme::accent));
    draw->AddText({start.x + 49.F * scale, start.y + 10.F * scale},
        theme::u32(theme::text_bright), title);
    draw->AddText(theme::small_font(), 12.F * scale,
        {start.x + 49.F * scale, start.y + 34.F * scale},
        theme::u32(theme::text_muted), subtitle);
    if (ImGui::IsItemFocused())
        draw->AddRect(start, {start.x + size.x, start.y + size.y}, theme::u32(theme::accent), 7.F * scale);
    return clicked;
}
}  // namespace

void draw_welcome_window(App& app) {
    if (!app.show_welcome || app.smoke_mode) return;
    const auto* viewport = ImGui::GetMainViewport();
    const float scale = std::clamp(ImGui::GetFontSize() / 14.F, 0.85F, 2.F);
    const float width = std::min(820.F * scale, viewport->WorkSize.x - 24.F);
    const float height = std::min(610.F * scale, viewport->WorkSize.y - 24.F);
    ImGui::SetNextWindowPos({viewport->WorkPos.x + viewport->WorkSize.x * 0.5F,
        viewport->WorkPos.y + viewport->WorkSize.y * 0.5F}, ImGuiCond_Always, {0.5F, 0.5F});
    ImGui::SetNextWindowSize({width, height});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0.F, 0.F});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 12.F * scale);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, theme::surface_1);
    const char* popup = "###StudioWelcome";
    if (!ImGui::IsPopupOpen(popup)) ImGui::OpenPopup(popup);
    if (ImGui::BeginPopupModal(popup, nullptr, ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings)) {
        const auto origin = ImGui::GetCursorScreenPos();
        const float cover_height = 216.F * scale;
        auto* draw = ImGui::GetWindowDrawList();
        draw_welcome_cover(draw, origin, width, cover_height);
        draw->AddText(ImGui::GetFont(), 31.F * scale,
            {origin.x + 30.F * scale, origin.y + 48.F * scale},
            theme::u32(theme::text_bright), "PHOTARA");
        draw->AddText(ImGui::GetFont(), 14.F * scale,
            {origin.x + 32.F * scale, origin.y + 88.F * scale},
            theme::u32(theme::accent), "S T U D I O");
        draw->AddText({origin.x + 32.F * scale, origin.y + 157.F * scale},
            theme::u32(theme::text_bright), tr("Every perspective becomes a world."));
        draw->AddText(theme::small_font(), 12.F * scale,
            {origin.x + 32.F * scale, origin.y + 183.F * scale},
            theme::u32(theme::text_muted), "PHOTOGRAMMETRY  /  GAUSSIAN SPLATTING");
        draw->AddText({origin.x + width - 112.F * scale, origin.y + 19.F * scale},
            theme::u32(theme::text_muted), PHOTARA_VERSION);
        ImGui::SetCursorPos({width - 40.F * scale, 9.F * scale});
        if (ImGui::Button("X##close-welcome", {28.F * scale, 28.F * scale}))
            app.show_welcome = false;
        ImGui::SetCursorPos({28.F * scale, cover_height + 23.F * scale});
        ImGui::TextUnformatted(tr("Welcome to Photara"));
        ImGui::SetCursorPosX(28.F * scale);
        ImGui::TextColored(theme::text_muted, "%s", tr("A new capture. A new dimension."));
        ImGui::Spacing();

        const float gap = 24.F * scale;
        const float content_width = width - 56.F * scale;
        const bool columns = content_width >= 570.F * scale;
        const float left_width = columns ? content_width * 0.44F : content_width;
        ImGui::SetCursorPosX(28.F * scale);
        ImGui::BeginChild("welcome-body", {content_width, 240.F * scale});
        ImGui::BeginGroup();
        ImGui::TextColored(theme::text_muted, "%s", tr("Get started"));
        ImGui::Spacing();
        const bool busy = app.job.running() || app.loading_scene || app.pending_train_prepare.valid();
        ImGui::BeginDisabled(busy);
        if (welcome_action("##new", icons::Icon::photo, tr("New Project"),
                tr("Start with a clean workspace"), left_width, scale)) {
            app.show_welcome = false;
            new_project(app);
        }
        if (welcome_action("##open", icons::Icon::folder, tr("Open Project..."),
                tr("Continue from a saved .ascan project"), left_width, scale))
            select_project_folder(app);
        if (welcome_action("##images", icons::Icon::camera, tr("Open Images..."),
                tr("Bring your photographs into 3D"), left_width, scale)) {
            app.show_welcome = false;
            select_image_folder(app);
        }
        ImGui::EndDisabled();
        ImGui::EndGroup();
        if (columns) ImGui::SameLine(0.F, gap);
        else ImGui::Spacing();
        ImGui::BeginGroup();
        ImGui::TextColored(theme::text_muted, "%s", tr("Recent projects"));
        ImGui::Spacing();
        const float right_width = columns ? content_width - left_width - gap : content_width;
        ImGui::BeginChild("recent-projects", {right_width, 198.F * scale});
        if (app.recent_projects.empty()) {
            ImGui::Dummy({0.F, 19.F * scale});
            icons::inline_icon(icons::Icon::clock, theme::u32(theme::text_faint), 24.F * scale);
            ImGui::Spacing();
            ImGui::TextUnformatted(tr("Your next creation starts here"));
            ImGui::PushStyleColor(ImGuiCol_Text, theme::text_muted);
            ImGui::TextWrapped("%s", tr("Projects you open or save will appear here."));
            ImGui::PopStyleColor();
        }
        std::filesystem::path selected;
        for (std::size_t index = 0; index < app.recent_projects.size(); ++index) {
            const auto& path = app.recent_projects[index];
            const auto name = path.stem().u8string();
            const auto location = path.u8string();
            ImGui::PushID(static_cast<int>(index));
            ImGui::BeginDisabled(busy);
            if (ImGui::Selectable("##recent", false, 0, {right_width - 14.F * scale, 44.F * scale}))
                selected = path;
            const auto min = ImGui::GetItemRectMin();
            auto* list = ImGui::GetWindowDrawList();
            list->PushClipRect(min, {min.x + right_width - 20.F * scale, min.y + 44.F * scale}, true);
            list->AddText({min.x + 6.F * scale, min.y + 3.F * scale}, theme::u32(theme::text_bright),
                reinterpret_cast<const char*>(name.c_str()));
            list->AddText(theme::small_font(), 12.F * scale,
                {min.x + 6.F * scale, min.y + 25.F * scale}, theme::u32(theme::text_faint),
                reinterpret_cast<const char*>(location.c_str()));
            list->PopClipRect();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", reinterpret_cast<const char*>(location.c_str()));
            ImGui::EndDisabled();
            ImGui::PopID();
        }
        ImGui::EndChild();
        ImGui::EndGroup();
        ImGui::EndChild();
        if (!selected.empty()) {
            std::error_code error;
            if (std::filesystem::is_regular_file(selected, error))
                open_project_from_path(app, selected);
            else set_message(app, tr("Project file is unavailable. It may have been moved or deleted."), theme::warning);
        }
        if (!app.message.empty()) {
            ImGui::SetCursorPosX(28.F * scale);
            ImGui::PushTextWrapPos(width - 28.F * scale);
            ImGui::TextColored(app.message_colour, "%s", tr(app.message.c_str()));
            ImGui::PopTextWrapPos();
        }
        ImGui::SetCursorPosX(28.F * scale);
        ImGui::Separator();
        ImGui::SetCursorPosX(28.F * scale);
        if (ImGui::Checkbox(tr("Show on startup"), &app.welcome_on_startup)) store_welcome_state(app);
        ImGui::SameLine();
        ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), width - 190.F * scale));
        if (ImGui::Button(tr("Enter workspace"), {162.F * scale, 28.F * scale})) app.show_welcome = false;
        if (ImGui::IsKeyPressed(ImGuiKey_Escape)) app.show_welcome = false;
        if (!app.show_welcome) ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
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
    theme::metric("Developer", "Bingyang Hu");
    theme::metric("Copyright", "© 2026 Bingyang Hu");
    theme::metric("License", "Apache-2.0");
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
    if (model_viewer_active(app)) return Action::none;
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
                 : tr(app.settings.sam_masks
                        ? "Select a capture video. Align Photos extracts frames, generates masks, then aligns cameras."
                        : "Select a capture video. Align Photos extracts sharp frames, then runs SfM."))) {
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
            app.job.paused() ? tr("Paused") :
                app.monitor.saving_checkpoint() ? tr("Saving training progress...") :
                app.monitor.training().valid ? tr("Training...") : tr("Initializing training..."), {126.F, 32.F},
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
            if (app.training_checkpoint_info)
                ImGui::SetTooltip("%s", tr("Start a new training run and replace the previous checkpoint. Use Resume Training to resume instead."));
            else if (external_dataset)
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
    if ((!busy && app.training_checkpoint_info) || (training && app.job.paused())) {
        ImGui::SameLine();
        if (icons::labeled_button("##resume_training", icons::Icon::play,
                tr("Resume Training"), {150.F, 32.F}, icons::ButtonStyle::normal, true))
            action = training ? Action::resume : Action::continue_training;
        if (ImGui::IsItemHovered()) {
            if (training) ImGui::SetTooltip("%s", tr("Continue the paused reconstruction."));
            else {
                const auto& info = *app.training_checkpoint_info;
                ImGui::SetTooltip("%s: %u / %u", tr("Saved Training Progress"),
                    info.iteration, info.total_iterations);
                ImGui::SetTooltip("%s", tr("Continue from the saved PLY model. Optimizer state will be reinitialized."));
            }
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
                     "Choose From MVS or From Gaussians in the Mesh panel.")
                   : tr("Extract a surface after 3DGS training.\n"
                     "From Gaussians enables depth-normal and multi-view geometry\n"
                     "losses during optimisation. Switch to From MVS in the\n"
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
        if (app.job.paused() && !training) {
            ImGui::SameLine();
            if (icons::labeled_button(
                    "##resume", icons::Icon::play, resume_job_label(app.active_job), {150.F, 32.F},
                    icons::ButtonStyle::primary, true, false,
                    tr("Continue the paused reconstruction.")))
                action = Action::resume;
        } else if (!app.job.paused()) {
            ImGui::SameLine();
            if (icons::labeled_button(
                       "##pause", icons::Icon::pause, tr("Pause"), {96.F, 32.F},
                       icons::ButtonStyle::normal, true, false,
                       tr(k_pause_job_tooltip)))
                action = Action::pause;
        }
        ImGui::SameLine();
        if (icons::labeled_button(
                "##stop", icons::Icon::stop, tr("Stop"), {88.F, 32.F},
                icons::ButtonStyle::danger, true, false, tr(k_stop_job_tooltip)))
            action = Action::stop;
    }

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

void draw_status_bar(App& app) {
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
    const char* tag = PHOTARA_STUDIO_VERSION_STRING;
    const float tag_width = ImGui::CalcTextSize(tag).x;
    float right = ImGui::GetWindowWidth() - tag_width - 14.F;
    ImGui::SetCursorPos({right, centre_y});
    ImGui::PushStyleColor(
        ImGuiCol_Text, busy ? theme::text_bright : theme::text_faint);
    ImGui::TextUnformatted(tag);
    ImGui::PopStyleColor();

    const char* device =
        app.settings.training_backend == 1 ? "Vulkan" : "CUDA";
    char backend[64];
    std::snprintf(
        backend, sizeof(backend), "%s  \xC2\xB7  %s", tr("Train"), device);
    const float backend_width = status_segment_width(backend);
    right -= backend_width + 22.F;
    draw_status_separator(right + backend_width + 11.F, height);
    ImGui::SetCursorPos({right, 0.F});
    ImGui::InvisibleButton("##open_training_compute", {backend_width, height});
    const bool backend_hovered = ImGui::IsItemHovered();
    const bool backend_clicked = ImGui::IsItemClicked();
    const ImVec4 backend_colour =
        busy || backend_hovered ? theme::text_bright : theme::text_muted;
    const ImVec2 origin = ImGui::GetWindowPos();
    const float line = ImGui::GetTextLineHeight();
    constexpr float icon_size = 15.F;
    const float icon_y = origin.y + centre_y + (line - icon_size) * 0.5F;
    icons::draw(
        ImGui::GetWindowDrawList(), icons::Icon::gpu,
        {origin.x + right, icon_y},
        {origin.x + right + icon_size, icon_y + icon_size},
        theme::u32(backend_colour));
    ImGui::GetWindowDrawList()->AddText(
        {origin.x + right + icon_size + 6.F, origin.y + centre_y},
        theme::u32(backend_colour), backend);
    if (backend_hovered) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::SetTooltip(
            "%s", tr("Training compute. Click to open Preferences."));
    }
    if (backend_clicked) open_preferences(app, 0);

    std::uint64_t vram_used = 0;
    std::uint64_t vram_total = 0;
    const TrainingStats& train_stats = app.monitor.training();
    const bool training_vram = busy && app.active_job == JobKind::train &&
        train_stats.vram_total > 0;
    if (training_vram) {
        vram_used = train_stats.vram_used;
        vram_total = train_stats.vram_total;
    } else {
        gpu::query_vram(vram_used, vram_total);
    }
    if (vram_total > 0) {
        const auto format_gb = [](const std::uint64_t bytes) {
            char buf[24];
            std::snprintf(
                buf, sizeof(buf), "%.1f GB",
                static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
            return std::string(buf);
        };
        const std::string used_gb = format_gb(vram_used);
        const std::string total_gb = format_gb(vram_total);
        const std::string free_gb = format_gb(vram_total - vram_used);
        const double pct = 100.0 * static_cast<double>(vram_used) /
                           static_cast<double>(vram_total);
        const float fraction = static_cast<float>(
            std::clamp(vram_used / static_cast<double>(vram_total), 0.0, 1.0));
        char vram_text[80];
        std::snprintf(
            vram_text, sizeof(vram_text), tr("VRAM %s / %s  %.0f%%"),
            used_gb.c_str(), total_gb.c_str(), pct);
        constexpr float k_vram_icon = 15.F;
        constexpr float k_vram_gap = 6.F;
        constexpr float k_vram_bar_w = 84.F;
        constexpr float k_vram_bar_h = 8.F;
        const float vram_width =
            k_vram_icon + k_vram_gap + k_vram_bar_w + k_vram_gap +
            ImGui::CalcTextSize(vram_text).x;
        right -= vram_width + 22.F;
        if (right > 240.F) {
            draw_status_separator(right + vram_width + 11.F, height);
            ImGui::SetCursorPos({right, 0.F});
            ImGui::InvisibleButton("##vram_status", {vram_width, height});
            if (ImGui::IsItemHovered()) {
                char tip[128];
                std::snprintf(
                    tip, sizeof(tip), tr("Used %s  ·  Free %s  ·  %s total"),
                    used_gb.c_str(), free_gb.c_str(), total_gb.c_str());
                ImGui::SetTooltip("%s", tip);
            }
            const ImVec4 fill = pct >= 90.0
                ? theme::danger
                : (pct >= 75.0 ? theme::warning : theme::accent);
            const ImVec4 vram_colour = pct >= 90.0
                ? theme::danger
                : (pct >= 75.0 ? theme::warning
                               : (busy ? theme::text_bright : theme::text_muted));
            ImGui::SetCursorPos({right, centre_y});
            icons::inline_icon(
                icons::Icon::gpu, theme::u32(vram_colour), k_vram_icon);
            const ImVec2 origin = ImGui::GetWindowPos();
            const float bar_x = origin.x + right + k_vram_icon + k_vram_gap;
            const float bar_y = origin.y + (height - k_vram_bar_h) * 0.5F;
            const ImVec2 bar_min{bar_x, bar_y};
            const ImVec2 bar_max{bar_x + k_vram_bar_w, bar_y + k_vram_bar_h};
            const float rounding = k_vram_bar_h * 0.5F;
            ImDrawList* draw = ImGui::GetWindowDrawList();
            draw->AddRectFilled(
                bar_min, bar_max, IM_COL32(255, 255, 255, 22), rounding);
            draw->AddRect(
                bar_min, bar_max, theme::u32(theme::border, 0.95F), rounding);
            if (fraction > 0.F) {
                const float filled = std::max(
                    k_vram_bar_h, k_vram_bar_w * fraction);
                draw->PushClipRect(bar_min, bar_max, true);
                draw->AddRectFilled(
                    bar_min, {bar_min.x + filled, bar_max.y}, theme::u32(fill),
                    rounding);
                draw->PopClipRect();
            }
            ImGui::SetCursorPos(
                {right + k_vram_icon + k_vram_gap + k_vram_bar_w + k_vram_gap,
                 centre_y});
            ImGui::PushStyleColor(ImGuiCol_Text, vram_colour);
            ImGui::TextUnformatted(vram_text);
            ImGui::PopStyleColor();
        }
    }

    const double training_elapsed = app.monitor.training_elapsed_seconds();
    const double elapsed = training_elapsed >= 0.0
        ? training_elapsed : app.monitor.elapsed_seconds();
    if (elapsed >= 0.0) {
        char elapsed_text[64];
        std::snprintf(
            elapsed_text, sizeof(elapsed_text),
            tr(training_elapsed >= 0.0 ? "Training elapsed %s" :
                busy && app.active_job == JobKind::train ? "Preparation elapsed %s" : "Elapsed %s"),
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


void draw_project_lifecycle_modals(App& app) {
    if (app.show_project_switch_confirmation) {
        app.show_project_switch_confirmation = false;
        ImGui::OpenPopup(tr("Save before switching?"));
    }
    if (ImGui::BeginPopupModal(tr("Save before switching?"), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("%s", tr("Save the current project before changing the dataset or opening another project?"));
        if (!app.checkpoint_save_error.empty()) ImGui::TextWrapped("%s", app.checkpoint_save_error.c_str());
        if (ImGui::Button(tr("Save"), {120.F, 32.F})) {
            request_project_save(app, false);
            if (!app.pending_project_action || app.project_transition_ready)
                ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button(tr("Don't Save"), {120.F, 32.F})) {
            ImGui::CloseCurrentPopup();
            complete_project_transition(app);
        }
        ImGui::SameLine();
        if (ImGui::Button(tr("Cancel"), {100.F, 32.F})) {
            app.pending_project_action = {};
            app.project_transition_ready = false;
            app.checkpoint_save_error.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    if (app.show_close_confirmation) {
        app.show_close_confirmation = false;
        ImGui::OpenPopup(tr("Save before closing?"));
    }
    ImGui::SetNextWindowSize({560.F, 0.F}, ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal(tr("Save before closing?"), nullptr,
            ImGuiWindowFlags_AlwaysAutoResize)) {
        if (app.saving_training) {
            ImGui::TextWrapped("%s", tr("Saving project and training progress. Please wait."));
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - app.checkpoint_save_started).count();
            if (seconds > 15)
                ImGui::TextWrapped("%s", tr("Waiting for the current iteration or checkpoint write to finish. You can cancel and keep working."));
        } else {
            ImGui::TextWrapped("%s", tr(app.job.running() && app.active_job == JobKind::train
                ? "Save this project and its current training progress? You can continue training when you open it again."
                : app.job.running()
                ? "Save the project before closing? The running task will stop; only 3DGS training supports restoring iteration progress."
                : "Save the project before closing?"));
        }
        if (!app.checkpoint_save_error.empty()) {
            ImGui::Spacing();
            ImGui::TextWrapped("%s", app.checkpoint_save_error.c_str());
        }
        ImGui::Spacing();
        ImGui::BeginDisabled(app.saving_training);
        if (ImGui::Button(tr("Save and Exit"), {160.F, 32.F})) request_project_save(app, true);
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button(tr("Exit Without Saving"), {185.F, 32.F})) {
            cancel_project_save(app);
            app.close_approved = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button(tr("Cancel"), {100.F, 32.F})) {
            cancel_project_save(app);
            app.checkpoint_save_error.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

}  // namespace editor
