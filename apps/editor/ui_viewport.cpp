#include "ui.hpp"
#include "app.hpp"
#include "i18n.hpp"
#include "icons.hpp"
#include "viewport_gizmo.hpp"

#include "imgui_impl_vulkan.h"
#include "imgui_internal.h"
#include "io/image.hpp"
#include "splat/visualize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace editor {
using i18n::tr;

constexpr float k_view_rail_pad = 10.F;
constexpr float k_view_rail_top = 52.F;
constexpr float k_view_rail_width = 44.F;
constexpr float k_view_rail_height = 156.F;
constexpr float k_scene_toggle_gap = 8.F;
constexpr float k_scene_toggle_height = 116.F;

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
         tr("Point Cloud")},
        {"##viz_splat", icons::Icon::splat, VisualizationMode::splat, splat_ok,
         splat_ok ? tr("Splat") : tr("Train 3DGS to view the splat")},
        {"##viz_rings", icons::Icon::rings, VisualizationMode::rings, rings_ok,
         rings_ok ? tr("Rings")
                  : tr("Available while training or after a Gaussian model exists")},
        {"##viz_mesh", icons::Icon::cube, VisualizationMode::mesh, mesh_ok,
         mesh_ok ? tr("Mesh")
                 : tr("Build a mesh to inspect the reconstructed surface")},
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
                ? tr("Hide camera frustums")
                : tr("Show camera frustums")))
        set_camera_overlays(app.view_options, !app.view_options.show_views);
    hovered = hovered || ImGui::IsItemHovered();

    const ImVec2 grid_min{
        rail.Min.x + k_inner, rail.Min.y + k_inner + k_btn + k_gap};
    if (rail_icon_button(
            "##tog_grid", icons::Icon::grid, grid_min, button_size,
            app.view_options.show_grid, true,
            app.view_options.show_grid ? tr("Hide ground grid")
                                      : tr("Show ground grid")))
        app.view_options.show_grid = !app.view_options.show_grid;
    hovered = hovered || ImGui::IsItemHovered();

    const ImVec2 region_min{
        rail.Min.x + k_inner,
        rail.Min.y + k_inner + 2.F * (k_btn + k_gap)};
    if (rail_icon_button(
            "##tog_region", icons::Icon::frame, region_min, button_size,
            app.view_options.show_region, true,
            app.view_options.show_region
                ? tr("Hide reconstruction region")
                : tr("Show reconstruction region")))
        app.view_options.show_region = !app.view_options.show_region;
    hovered = hovered || ImGui::IsItemHovered();
    return hovered;
}

void draw_align_step_bar(const App& app, const ImVec2 origin, const float width) {
    struct Step {
        const char* label;
        Stage stage;
    };
    Step steps[6];
    int step_count = 0;
    auto add_step = [&](const char* label, const Stage stage) {
        if (step_count >= 6) return;
        steps[step_count++] = {label, stage};
    };
    if (is_video_source(app.settings))
        add_step(tr("Frames"), Stage::preparing);
    if (app.settings.sam_masks)
        add_step(tr("Masks"), Stage::masking);
    add_step(tr("Features"), Stage::features);
    add_step(tr("Matching"), Stage::matching);
    add_step(tr("Tracks"), Stage::tracks);
    add_step(tr("Poses"), Stage::mapping);

    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 bar_max{origin.x + width, origin.y + 36.F};
    draw->AddRectFilled(origin, bar_max, theme::u32(theme::surface_2));
    draw->AddLine(
        {origin.x, bar_max.y - 1.F}, {bar_max.x, bar_max.y - 1.F},
        theme::u32(theme::border));

    const Stage stage = app.monitor.stage();
    const Stage ranked =
        stage == Stage::exporting ? Stage::mapping : stage;
    int current = 0;
    for (int i = 0; i < step_count; ++i) {
        if (steps[i].stage == ranked) {
            current = i;
            break;
        }
    }
    const float pad = 10.F;
    const bool show_count = app.monitor.task().total > 0 && width > 640.F;
    const float reserve = show_count ? 96.F : 0.F;
    const float inner = std::max(1.F, width - pad * 2.F - reserve);
    const float cell = inner / static_cast<float>(std::max(1, step_count));
    ImFont* small = theme::small_font();
    for (int i = 0; i < step_count; ++i) {
        const float x = origin.x + pad + cell * static_cast<float>(i);
        const bool active = i == current;
        const bool done = current > i;
        if (i + 1 < step_count) {
            const ImVec2 a{x + cell - 18.F, origin.y + 18.F};
            const ImVec2 b{x + cell + 6.F, origin.y + 18.F};
            draw->AddLine(
                a, b,
                theme::u32(done ? theme::success : theme::border, 0.85F), 1.6F);
        }
        const ImVec2 chip_min{x + 2.F, origin.y + 6.F};
        const ImVec2 chip_max{
            x + std::min(cell - 20.F, 118.F), origin.y + 30.F};
        if (active) {
            draw->AddRectFilled(
                chip_min, chip_max, theme::u32(theme::fade(theme::accent, 0.22F)),
                6.F);
            draw->AddRect(
                chip_min, chip_max, theme::u32(theme::accent, 0.8F), 6.F);
        } else if (done) {
            draw->AddRectFilled(
                chip_min, chip_max, theme::u32(theme::fade(theme::success, 0.12F)),
                6.F);
        }
        char index[4];
        std::snprintf(index, sizeof(index), "%d", i + 1);
        const ImVec4 colour = active
            ? theme::accent
            : (done ? theme::success : theme::text_faint);
        draw->AddCircleFilled(
            {chip_min.x + 12.F, origin.y + 18.F}, 7.F, theme::u32(colour, 0.9F));
        const ImVec2 ns = ImGui::CalcTextSize(index);
        draw->AddText(
            {chip_min.x + 12.F - ns.x * 0.5F,
             origin.y + 18.F - ns.y * 0.5F},
            theme::u32(theme::surface_0), index);
        draw->AddText(
            small, small->FontSize, {chip_min.x + 24.F, origin.y + 11.F},
            theme::u32(colour), steps[i].label);
    }

    const SubTask& task = app.monitor.task();
    if (show_count) {
        char progress[48];
        std::snprintf(
            progress, sizeof(progress), "%llu / %llu",
            static_cast<unsigned long long>(task.completed),
            static_cast<unsigned long long>(task.total));
        const ImVec2 size = ImGui::CalcTextSize(progress);
        draw->AddText(
            {bar_max.x - size.x - 14.F, origin.y + 10.F},
            theme::u32(theme::text_muted), progress);
    }
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
                    ? tr("3D scene view")
                    : tr("2D image QA — features, GT vs 3DGS, error map"));
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
           differs(a.ortho_height, b.ortho_height) ||
           differs(a.fisheye_k1, b.fisheye_k1) ||
           a.projection != b.projection ||
           differs(a.target.x, b.target.x) || differs(a.target.y, b.target.y) ||
           differs(a.target.z, b.target.z);
}

void fit_preview_raster(
    const float viewport_w, const float viewport_h, const bool interacting,
    std::uint32_t& width, std::uint32_t& height) {
    const float view_w = std::max(1.F, viewport_w);
    const float view_h = std::max(1.F, viewport_h);
    // Raster at the pixels the viewport actually shows, capped by the shared
    // image. While the camera moves the frame is replaced as soon as the mouse
    // moves again, so half resolution trades detail that is about to change for
    // latency in the live view and GPU time returned to the optimizer.
    constexpr std::uint32_t k_min_extent = 256;
    const float extent = std::min(
        static_cast<float>(k_preview_extent),
        std::max(view_w, view_h) * (interacting ? 0.5F : 1.F));
    if (view_w >= view_h) {
        width = std::max<std::uint32_t>(
            k_min_extent, static_cast<std::uint32_t>(std::lround(extent)));
        height = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(std::lround(
                   static_cast<double>(width) * view_h / view_w)));
    } else {
        height = std::max<std::uint32_t>(
            k_min_extent, static_cast<std::uint32_t>(std::lround(extent)));
        width = std::max<std::uint32_t>(
            1, static_cast<std::uint32_t>(std::lround(
                   static_cast<double>(height) * view_w / view_h)));
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
    const auto raster_delta = [](const std::uint32_t a, const std::uint32_t b) {
        return a > b ? a - b : b - a;
    };
    if (!force && app.has_last_preview_orbit &&
        !orbit_pose_changed(app.camera, app.last_preview_orbit) &&
        raster_delta(width, app.preview_raster_width) <= 2 &&
        raster_delta(height, app.preview_raster_height) <= 2)
        return;
    ++app.preview_camera_revision;
    app.preview_raster_width = width;
    app.preview_raster_height = height;
    SplatPreviewCamera preview =
        make_preview_camera(app.camera, width, height);
    const auto vis = editor_visualize_options(app);
    write_preview_camera_file(
        app.layout.preview_camera_file, preview, app.preview_camera_revision,
        photara::splat::visualization_mode_name(vis.mode),
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
    photara::splat::VisualizeOptions vis;
    vis.mode = photara::splat::VisualizationMode::splat;
    vis.point_size_px = app.view_options.point_size;
    vis.ring_scale = app.view_options.ring_scale;
    photara::splat::write_visualization_sidecar(
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
    const bool training =
        app.job.running() && app.active_job == JobKind::train;
    if (training) {
        if (!app.preview.display.descriptor || !qa_capture_frame_ready(app))
            return;
    } else if (app.splat_renderer.splat_count() == 0) {
        return;
    }
    if (!app.image_qa_session.has_gt() ||
        app.image_qa_session.loaded_view() != app.image_qa.selected)
        return;
    if (std::chrono::steady_clock::now() < app.qa_metrics_after) return;

    const std::uint64_t revision = training
        ? gpu::consumed_timeline_value()
        : app.splat_renderer.frames_epoch();
    if (app.image_qa_session.has_render_pixels() &&
        app.image_qa_session.render_view() == app.image_qa.selected &&
        app.image_qa_session.metrics().valid) {
        app.image_qa.metrics_dirty = false;
        return;
    }
    photara::io::RgbImage render;
    if (training) {
        if (!app.preview.display.download_rgb(render, k_image_qa_metric_extent))
            return;
    } else {
        std::vector<std::uint8_t> pixels;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        if (!app.splat_renderer.download_rgb(
                pixels, width, height, k_image_qa_metric_extent))
            return;
        render.width = width;
        render.height = height;
        render.pixels = std::move(pixels);
    }
    app.image_qa_session.set_render(
        std::move(render), app.image_qa.selected, revision);
    app.image_qa.metrics_dirty = false;
}

void set_viewport_workspace(
    App& app, const ViewportWorkspace workspace, const bool user_driven) {
    if (user_driven && alignment_job_running(app) && app.workspace != workspace)
        app.alignment_workspace_user_override = true;
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
    // Training still publishes the capture pose to the CUDA child. After
    // training, the 2D compare draws that pose with the Vulkan rasterizer.
    if (app.job.running() && app.active_job == JobKind::train)
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
        if (app.mesh_renderer.descriptor()) {
            draw->AddImage(
                reinterpret_cast<ImTextureID>(app.mesh_renderer.descriptor()),
                min, max);
        }
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
        const char* headline = tr("Drop photos, a video, or a reconstruction");
        const char* hint = tr(
            "Drop an image folder, photos, a video, .asfm, or .ascan onto this view");
        if (aligning) {
            const Stage stage = app.monitor.stage();
            if (stage == Stage::preparing) {
                headline = tr("Extracting frames");
                hint = tr("Sharp frames are chosen before masks and alignment");
            } else if (stage == Stage::masking) {
                headline = tr("Generating masks");
                hint = tr(
                    "Each frame gets a foreground mask, then alignment uses it");
            } else if (stage == Stage::features) {
                headline = tr("Extracting features in 2D");
                hint = tr("Switch to 2D to watch keypoints appear on each photo");
            } else if (stage == Stage::matching) {
                headline = tr("Matching images in 2D");
                hint = tr("Switch to 2D to watch correspondence lines between pairs");
            } else {
                headline = tr("Aligning photos...");
                hint = tr("Cameras and points appear as soon as geometry is available");
            }
        } else if (app.settings.images_dir[0] != '\0') {
            headline = tr("Ready to align cameras");
            hint = tr(
                "Run Align Photos, or drop a different folder, video, .asfm, or .ascan");
        }
        draw_empty_viewport(draw, min, max, headline, hint);
    }

    ViewOptions draw_options = app.view_options;
    if (app.view_mode == VisualizationMode::mesh) draw_options.show_cloud = false;
    ensure_reconstruction_box(app);
    const SceneDrawStats stats = app.renderer.draw(
        draw, min, max, app.scene, app.camera, draw_options, hovered,
        app.photos.ids(), app.photos.size(),
        app.view_mode == VisualizationMode::mesh ? &app.mesh : nullptr,
        &app.reconstruction_box);

    const bool gizmo_captures = draw_viewport_gizmo(
        app.gizmo, app.camera, min, max, &app.reconstruction_box,
        app.view_options.show_region && app.reconstruction_box.valid,
        reconstruction_local_radius(app));
    const bool viewport_input = hovered && !gizmo_captures;
    const bool frame_key = viewport_input &&
                           !ImGui::GetIO().WantTextInput &&
                           ImGui::IsKeyPressed(ImGuiKey_F);
    if (frame_key) frame_reconstruction(app);
    if (!handle_viewport_double_click(app, viewport_input, stats, min, max))
        update_orbit_camera(
            app.camera, viewport_input, reconstruction_local_radius(app));

    const char* overlay = app.settings.images_dir[0] != '\0' ? tr("NO ALIGNMENT")
                                                            : tr("NO IMAGES");
    ImVec4 overlay_dot = theme::inactive;
    if (aligning) {
        overlay = app.alignment_preview_seen ? tr("ALIGNING / PARTIAL RESULT") : tr("ALIGNING");
        overlay_dot = theme::accent;
    } else if (waiting_for_train_preview(app)) {
        overlay = tr("PREPARING 3DGS");
        overlay_dot = theme::warning;
    } else if (app.job.running() && app.active_job == JobKind::dense) {
        overlay = tr("EXTRACT MESH");
        overlay_dot = theme::accent;
    } else if (app.job.running() && app.active_job == JobKind::texture) {
        overlay = tr("BAKE TEXTURE");
        overlay_dot = theme::accent;
    } else if (app.loading_scene) {
        overlay = tr("LOADING");
        overlay_dot = theme::warning;
    } else if (app.alignment_preview_seen) {
        overlay = tr("PARTIAL ALIGNMENT / NOT FINAL");
        overlay_dot = theme::warning;
    } else if (app.view_mode == VisualizationMode::mesh) {
        overlay = app.mesh.has()
            ? (gpu_mesh ? tr("MESH (GPU)") : tr("Mesh"))
            : tr("NO MESH");
        overlay_dot = app.mesh.has() ? theme::accent : theme::inactive;
    } else if (app.view_mode == VisualizationMode::rings) {
        overlay = app.scene.has_gaussians() ? tr("GAUSSIAN RINGS") : tr("NO GAUSSIANS");
        overlay_dot = app.scene.has_gaussians() ? theme::accent : theme::inactive;
    } else if (app.scene.has_points()) {
        overlay = app.scene_source.empty() ? tr("SPARSE POINT CLOUD")
                                           : app.scene_source.c_str();
        overlay_dot = theme::accent;
    } else if (app.has_sparse && !app.suppress_scene_auto_load) {
        overlay = tr("CLOUD READY");
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
            ? tr("Waiting for cameras from this alignment")
            : app.suppress_scene_auto_load
            ? tr("Previous result cleared")
            : app.has_sparse
            ? tr("Reading sparse.ply...")
            : tr("Drop a photo folder, .asfm, or .ascan here");
        draw->AddText(
            {min.x + 16.F, max.y - 42.F}, theme::u32(theme::text_faint), hint);
    }
    const bool align_waiting = aligning && !app.scene.has_points();
    draw->AddText(
        {min.x + 16.F, max.y - 24.F}, theme::u32(theme::text_faint),
        align_waiting
            ? tr("2 / 3: switch 2D image QA and 3D scene")
            : tr("LMB orbit  |  MMB pan  |  RMB + WASD/QE fly  |  F frame  |  drag region gizmo  |  double-click focus"));

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
    if (!app.preview_follow_view || app.scene.views.empty()) {
        if (app.camera.projection == EditorProjection::fisheye)
            std::snprintf(
                camera_label, sizeof(camera_label), "orbit · fisheye %.0f°",
                app.camera.fov_degrees);
        else if (app.camera.projection == EditorProjection::orthographic)
            std::snprintf(
                camera_label, sizeof(camera_label), "orbit · ortho %.2f",
                app.camera.ortho_height);
        else if (app.camera.projection == EditorProjection::panorama)
            std::snprintf(camera_label, sizeof(camera_label), "orbit · panorama");
        else
            std::snprintf(
                camera_label, sizeof(camera_label), "orbit · %.0f°",
                app.camera.fov_degrees);
    } else
        std::snprintf(
            camera_label, sizeof(camera_label), "camera %u / %u",
            app.preview_view + 1, view_count);

    const char* controls = live
        ? "LMB orbit  |  MMB pan  |  RMB + WASD fly  |  drag region gizmo  |  arrows snap capture"
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
    ensure_reconstruction_box(app);
    const SceneDrawStats overlay_stats = app.renderer.draw(
        draw, min, max, app.scene, app.camera, overlay, hovered,
        app.photos.ids(), app.photos.size(), nullptr, &app.reconstruction_box);

    const bool gizmo_captures = draw_viewport_gizmo(
        app.gizmo, app.camera, min, max, &app.reconstruction_box,
        app.view_options.show_region && app.reconstruction_box.valid,
        reconstruction_local_radius(app));
    const bool viewport_input = hovered && !gizmo_captures;
    const bool used_double_click = handle_viewport_double_click(
        app, viewport_input, overlay_stats, min, max);
    if (!used_double_click)
        update_orbit_camera(
            app.camera, viewport_input, reconstruction_local_radius(app));
    if (!used_double_click &&
        (app.camera.interacting ||
         (viewport_input && ImGui::GetIO().MouseWheel != 0.F)))
        app.preview_follow_view = false;
    handle_preview_view_input(app, viewport_input);
    std::uint32_t raster_w = app.preview_raster_width;
    std::uint32_t raster_h = app.preview_raster_height;
    // During optimization, a smaller interactive raster keeps camera motion
    // responsive without taking too much GPU time away from training. Once
    // training has handed the model to the standalone viewer, the GPU is free
    // to render camera motion at the viewport's full resolution.
    const bool use_interactive_downscale =
        training && app.camera.interacting;
    fit_preview_raster(
        max.x - min.x, max.y - min.y, use_interactive_downscale, raster_w,
        raster_h);
    sync_live_preview_camera(app, false, raster_w, raster_h);

    const char* overlay_label = tr("IDLE");
    if (has_frame) {
        if (app.view_mode == VisualizationMode::points)
            overlay_label = tr("GAUSSIAN CENTRES");
        else if (app.view_mode == VisualizationMode::rings)
            overlay_label = tr("GAUSSIAN RINGS");
        else if (training)
            overlay_label = tr("LIVE TRAINING PREVIEW");
        else
            overlay_label = viewing ? tr("LIVE SPLAT VIEW") : tr("LAST TRAINING FRAME");
    } else if (live) {
        overlay_label = training ? tr("TRAINING") : tr("VIEWING");
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

std::uint32_t gut_projection(const EditorProjection projection) {
    switch (projection) {
        case EditorProjection::orthographic:
            return splat_render::k_camera_orthographic;
        case EditorProjection::fisheye:
            return splat_render::k_camera_fisheye;
        case EditorProjection::panorama:
            return splat_render::k_camera_equirectangular;
        case EditorProjection::perspective:
        default:
            return splat_render::k_camera_pinhole;
    }
}

std::uint32_t gut_projection(const photara::CameraModel model) {
    switch (model) {
        case photara::CameraModel::opencv_fisheye:
            return splat_render::k_camera_fisheye;
        case photara::CameraModel::equirectangular:
            return splat_render::k_camera_equirectangular;
        case photara::CameraModel::pinhole:
        case photara::CameraModel::automatic:
        default:
            return splat_render::k_camera_pinhole;
    }
}

splat_render::Camera gut_camera_from(const SplatPreviewCamera& preview) {
    splat_render::Camera camera;
    camera.world_to_camera = preview.world_to_camera;
    camera.position = preview.position;
    camera.fx = preview.fx;
    camera.fy = preview.fy;
    camera.cx = preview.cx;
    camera.cy = preview.cy;
    camera.k1 = preview.k1;
    camera.k2 = preview.k2;
    camera.k3 = preview.k3;
    camera.k4 = preview.k4;
    camera.width = preview.width;
    camera.height = preview.height;
    camera.model = gut_projection(preview.model);
    return camera;
}

void release_splat_preview_sets(App& app) {
    for (VkDescriptorSet& set : app.splat_preview_sets) {
        if (set) ImGui_ImplVulkan_RemoveTexture(set);
        set = VK_NULL_HANDLE;
    }
    app.splat_preview_views.fill(VK_NULL_HANDLE);
}

VkDescriptorSet splat_preview_texture(
    App& app, const splat_render::FrameTarget& frame) {
    if (app.splat_frames_epoch != app.splat_renderer.frames_epoch()) {
        release_splat_preview_sets(app);
        app.splat_frames_epoch = app.splat_renderer.frames_epoch();
    }
    if (frame.slot < 0 || frame.slot >= 3 || !frame.view || !frame.sampler)
        return VK_NULL_HANDLE;
    VkImageView& cached = app.splat_preview_views[static_cast<std::size_t>(frame.slot)];
    VkDescriptorSet& set = app.splat_preview_sets[static_cast<std::size_t>(frame.slot)];
    if (cached == frame.view && set) return set;
    if (set) ImGui_ImplVulkan_RemoveTexture(set);
    set = ImGui_ImplVulkan_AddTexture(frame.sampler, frame.view, frame.layout);
    cached = frame.view;
    return set;
}

void release_splat_preview(App& app) {
    release_splat_preview_sets(app);
    app.splat_frames_epoch = 0;
    app.splat_renderer.reset();
}

splat_render::Camera gut_camera_for_orbit(
    const SplatPreviewCamera& preview, const OrbitCamera& orbit) {
    splat_render::Camera camera = gut_camera_from(preview);
    camera.model = gut_projection(orbit.projection);
    if (orbit.projection != EditorProjection::orthographic) return camera;
    const float height = std::max(1.F, static_cast<float>(camera.height));
    const float pixels_per_unit =
        height / std::max(1e-4F, orbit.ortho_height);
    camera.fx = pixels_per_unit;
    camera.fy = pixels_per_unit;
    camera.cx = 0.5F * static_cast<float>(camera.width);
    camera.cy = 0.5F * height;
    camera.k1 = camera.k2 = camera.k3 = camera.k4 = 0.F;
    return camera;
}

bool draw_gut_image(
    App& app, const SplatPreviewCamera& preview, ImTextureID& texture,
    const OrbitCamera* orbit = nullptr) {
    texture = ImTextureID{};
    if (!ensure_splat_renderer(app)) return false;
    const splat_render::Camera camera = orbit == nullptr
        ? gut_camera_from(preview)
        : gut_camera_for_orbit(preview, *orbit);
    splat_render::FrameTarget target;
    if (!app.splat_renderer.draw(camera, target)) return false;
    const VkDescriptorSet set = splat_preview_texture(app, target);
    if (!set) return false;
    texture = reinterpret_cast<ImTextureID>(set);
    return true;
}

bool qa_preview_camera(const App& app, SplatPreviewCamera& camera) {
    if (app.image_qa.selected < 0 ||
        static_cast<std::size_t>(app.image_qa.selected) >= app.scene.views.size())
        return false;
    const ViewPose& pose =
        app.scene.views[static_cast<std::size_t>(app.image_qa.selected)];
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
    camera = make_preview_camera_from_view(pose, width, height);
    return true;
}

void draw_splat_render_tab(App& app, const ImVec2 min, const ImVec2 max) {
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(min, max, theme::u32(theme::viewport_bg));
    ImGui::SetCursorScreenPos(min);
    ImGui::SetNextItemAllowOverlap();
    ImGui::InvisibleButton(
        "##gut_view",
        {std::max(1.F, max.x - min.x), std::max(1.F, max.y - min.y)},
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight |
            ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered =
        ImGui::IsItemHovered() &&
        !view_mode_rail_contains(min, ImGui::GetIO().MousePos);

    std::uint32_t raster_w = app.preview_raster_width;
    std::uint32_t raster_h = app.preview_raster_height;
    fit_preview_raster(max.x - min.x, max.y - min.y, false, raster_w, raster_h);
    const SplatPreviewCamera preview =
        make_preview_camera(app.camera, raster_w, raster_h);
    ImTextureID texture{};
    const bool shown = draw_gut_image(app, preview, texture, &app.camera);
    if (shown) draw->AddImage(texture, min, max);
    else {
        const std::string& failure = app.splat_renderer.failure();
        draw_empty_viewport(
            draw, min, max, tr("VULKAN 3DGUT"),
            failure.empty() ? tr("Train 3DGS to view the splat") : failure.c_str());
    }

    ViewOptions overlay = app.view_options;
    overlay.show_cloud = false;
    overlay.draw_rings = false;
    ensure_reconstruction_box(app);
    const SceneDrawStats stats = app.renderer.draw(
        draw, min, max, app.scene, app.camera, overlay, hovered,
        app.photos.ids(), app.photos.size(), nullptr, &app.reconstruction_box);
    const bool gizmo_captures = draw_viewport_gizmo(
        app.gizmo, app.camera, min, max, &app.reconstruction_box,
        app.view_options.show_region && app.reconstruction_box.valid,
        reconstruction_local_radius(app));
    const bool viewport_input = hovered && !gizmo_captures;
    if (!handle_viewport_double_click(app, viewport_input, stats, min, max))
        update_orbit_camera(
            app.camera, viewport_input, reconstruction_local_radius(app));
    if (app.camera.interacting ||
        (viewport_input && ImGui::GetIO().MouseWheel != 0.F))
        app.preview_follow_view = false;
    handle_preview_view_input(app, viewport_input);

    draw_viewport_overlay(
        draw, min, tr("VULKAN 3DGUT"),
        shown ? theme::success : theme::warning);
    char readout[160];
    std::snprintf(
        readout, sizeof(readout), "%s gaussians  |  %u x %u",
        format_count(app.splat_renderer.splat_count()).c_str(),
        preview.width, preview.height);
    draw->AddText(
        {min.x + 16.F, max.y - 42.F}, theme::u32(theme::text_muted), readout);
    draw->AddText(
        {min.x + 16.F, max.y - 24.F}, theme::u32(theme::text_faint),
        tr("LMB orbit  |  MMB pan  |  RMB + WASD fly  |  arrows snap capture"));
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
    const bool open = ImGui::Begin(i18n::id("Viewport", "###Viewport"), &app.show_viewport, flags);
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
    constexpr float k_align_steps_h = 36.F;
    const bool align_steps = alignment_job_running(app);
    const float top_h = k_header_height + (align_steps ? k_align_steps_h : 0.F);
    const ImVec2 content_start = ImGui::GetCursorPos();
    const ImVec2 header_origin = ImGui::GetCursorScreenPos();
    const float header_width = ImGui::GetContentRegionAvail().x;
    ImGui::GetWindowDrawList()->AddRectFilled(
        header_origin,
        {header_origin.x + header_width, header_origin.y + k_header_height},
        theme::u32(theme::surface_3));
    if (!align_steps)
        ImGui::GetWindowDrawList()->AddLine(
            {header_origin.x, header_origin.y + k_header_height - 1.F},
            {header_origin.x + header_width,
             header_origin.y + k_header_height - 1.F},
            theme::u32(theme::border));

    draw_workspace_toggle(app, header_origin);

    const char* state = job_state_caption(app);
    if (app.workspace == ViewportWorkspace::image_2d) {
        if (alignment_job_running(app) &&
            app.monitor.stage() == Stage::preparing)
            state = tr("Extracting frames");
        else if (alignment_job_running(app) &&
                 app.monitor.stage() == Stage::masking)
            state = tr("Generating masks");
        else if (alignment_job_running(app) &&
                 app.monitor.stage() == Stage::features)
            state = tr("Extracting features");
        else if (alignment_job_running(app) &&
                 app.monitor.stage() == Stage::matching)
            state = app.align_live.kind ==
                    photara::sfm::AlignLiveKind::inliers
                ? tr("Verified pairs")
                : tr("Matching views");
        else
            state = tr("IMAGE QA");
    }
    const float state_width = ImGui::CalcTextSize(state).x;
    ImGui::SetCursorPos({
        content_start.x + std::max(8.F, header_width - state_width - 14.F),
        content_start.y + 10.F});
    theme::caption(state);
    if (align_steps)
        draw_align_step_bar(
            app, {header_origin.x, header_origin.y + k_header_height},
            header_width);
    ImGui::SetCursorPos({content_start.x, content_start.y + top_h});

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
        const bool training_now =
            app.job.running() && app.active_job == JobKind::train;
        ImTextureID gut_texture{};
        bool gut_ready = false;
        if (!training_now && image_qa_needs_render(app.image_qa.mode) &&
            app.has_model) {
            SplatPreviewCamera qa_camera;
            if (qa_preview_camera(app, qa_camera))
                gut_ready = draw_gut_image(app, qa_camera, gut_texture);
        }
        ImageQaDrawInput input;
        input.scene = &app.scene;
        input.photos = &app.photos;
        input.render = gut_ready
            ? gut_texture
            : (app.preview.display.descriptor
                   ? reinterpret_cast<ImTextureID>(app.preview.display.descriptor)
                   : ImTextureID{});
        input.has_render = gut_ready ||
                           (app.preview.display.descriptor &&
                            qa_capture_frame_ready(app));
        input.render_live = training_now && live_preview_active(app);
        input.has_model = app.has_model;
        input.external_alignment = has_external_dataset(app);
        if (alignment_job_running(app) &&
            app.align_live.kind != photara::sfm::AlignLiveKind::none) {
            const Stage stage = app.monitor.stage();
            const auto kind = app.align_live.kind;
            const bool pair = photara::sfm::is_live_pair_kind(kind);
            const bool features =
                kind == photara::sfm::AlignLiveKind::features;
            if (pair && stage == Stage::matching) {
                input.live = &app.align_live;
                input.pair_session = &app.align_match_session;
            } else if (
                features &&
                (stage == Stage::features || stage == Stage::preparing ||
                 stage == Stage::matching)) {
                input.live = &app.align_live;
            }
        }
        draw_image_qa(
            app.image_qa, app.image_qa_session, input, view_min, view_max);
        sync_qa_selection_to_preview(app, previous);
        ensure_dataset_scene_loaded(app);
        ensure_qa_preview(app);
    } else if (app.view_mode == VisualizationMode::mesh) {
        draw_sparse_tab(app, view_min, view_max);
        draw_view_mode_rail(app, view_min);
        draw_scene_toggle_rail(app, view_min);
    } else if (app.view_mode == VisualizationMode::splat && app.has_model &&
               !(app.job.running() && app.active_job == JobKind::train)) {
        draw_splat_render_tab(app, view_min, view_max);
        draw_view_mode_rail(app, view_min);
        draw_scene_toggle_rail(app, view_min);
    } else if (live_preview_active(app) && !waiting_for_train_preview(app)) {
        draw_training_tab(app, view_min, view_max);
        draw_view_mode_rail(app, view_min);
        draw_scene_toggle_rail(app, view_min);
    } else {
        draw_sparse_tab(app, view_min, view_max);
        const bool align_waiting =
            alignment_job_running(app) && !app.scene.has_points();
        if (!align_waiting) {
            draw_view_mode_rail(app, view_min);
            draw_scene_toggle_rail(app, view_min);
        }
    }
    ImGui::EndChild();
    ImGui::End();
}

}  // namespace editor
