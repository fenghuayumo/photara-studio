#include "ui.hpp"
#include "i18n.hpp"

#include <cstdio>
#include <string>

namespace editor {
using i18n::tr;

// View-only controls live beside the canvas, independently of the selected
// reconstruction step. Image QA remains available in the 2D workspace.
void draw_view_settings(App& app) {
    if (app.workspace != ViewportWorkspace::image_2d &&
        ImGui::CollapsingHeader(tr("Scene overlays"), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        if (ImGui::Checkbox(tr("Show cameras"), &app.view_options.show_views))
            app.view_options.show_camera_photos = app.view_options.show_views;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Wire frustums and capture photos in the 3D / training view.");
        ImGui::Checkbox(tr("Show trajectory"), &app.view_options.show_trajectory);
        ImGui::Checkbox(tr("Show ground grid"), &app.view_options.show_grid);
        ImGui::Checkbox(tr("Show origin axes"), &app.view_options.show_axes);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "RGB axes at the world origin (X red, Y green, Z blue).");
        ImGui::Checkbox(
            tr("Show reconstruction region"), &app.view_options.show_region);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tr(
                "Display-only overlay of the reconstruction region.\n"
                "Drag the center arrows to move the box, or a face dot\n"
                "to resize. Training follows it live only while the Focus\n"
                "Region checkbox is on; hiding the box never disables it."));
        if (app.reconstruction_box.valid) {
            const Vec3 size = app.reconstruction_box.size();
            ImGui::Text("Size  %.3f × %.3f × %.3f", size.x, size.y, size.z);
            if (theme::toolbar_button(
                    "Fit Region to Cloud", {-1.F, 28.F},
                    app.reconstruction_box.user_set)) {
                app.reconstruction_box.user_set = false;
                app.gizmo.box = {};
                invalidate_reconstruction_box(app.reconstruction_box);
                ensure_reconstruction_box(app);
                write_working_subject_bounds(app);
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr(
                    "Reset to the SfM-estimated SubjectBounds for the "
                    "focus region."));
        }
        ImGui::Spacing();
    }

    if (app.workspace == ViewportWorkspace::image_2d &&
        ImGui::CollapsingHeader(
            tr("Image QA"), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        const bool live_pair =
            alignment_job_running(app) &&
            app.monitor.stage() == Stage::matching &&
            photara::sfm::is_live_pair_kind(app.align_live.kind);
        const int count = image_qa_count(app.image_qa, app.scene);
        if (live_pair) {
            const auto& live = app.align_live;
            const std::string name_a = live.path_a.empty()
                ? std::string{"—"}
                : path_to_utf8(live.path_a.filename());
            const std::string name_b = live.path_b.empty()
                ? std::string{"—"}
                : path_to_utf8(live.path_b.filename());
            theme::metric("A", name_a.c_str());
            theme::metric("B", name_b.c_str());
            theme::metric(
                live.kind == photara::sfm::AlignLiveKind::inliers
                    ? tr("inliers")
                    : tr("matches"),
                std::to_string(live.total_matches).c_str());
        } else if (app.image_qa.selected >= 0 && app.image_qa.selected < count &&
            static_cast<std::size_t>(app.image_qa.selected) <
                app.scene.views.size()) {
            const ViewPose& pose =
                app.scene.views[static_cast<std::size_t>(app.image_qa.selected)];
            theme::metric("Capture", pose.name.c_str());
            theme::metric("Camera", pose.camera_model.c_str());
            char res[32];
            std::snprintf(
                res, sizeof(res), "%u × %u", pose.width, pose.height);
            theme::metric("Resolution", res);
            theme::metric(
                "Registered", pose.registered ? tr("yes") : tr("no"));
            theme::metric_coloured("Position support",
                tr(pose.position_audit == photara::sfm::PositionAudit::unverified
                    ? "Needs position review"
                    : pose.position_audit == photara::sfm::PositionAudit::graph_supported
                        ? "Graph supported" : "Not audited"),
                pose.position_audit == photara::sfm::PositionAudit::unverified
                    ? theme::warning : theme::text_muted);
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
        if (live_pair) {
            ImGui::Checkbox(
                tr("Show unmatched keypoints"),
                &app.image_qa.show_live_keypoints);
        } else {
            ImGui::Checkbox(
                tr("Show triangulated"), &app.image_qa.show_triangulated);
            ImGui::Checkbox(
                tr("Show untracked keypoints"), &app.image_qa.show_untracked);
        }
        if (app.image_qa_session.metrics().valid) {
            ImGui::Spacing();
            theme::section_header("COMPARE");
            const ImageQaMetrics& metrics = app.image_qa_session.metrics();
            ImGui::TextDisabled("%s", metrics.masked
                ? tr("Metrics: foreground only") : tr("Metrics: full image"));
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

    if (app.workspace != ViewportWorkspace::image_2d &&
        ImGui::CollapsingHeader(tr("Navigation camera"), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        theme::caption("Projection");
        ImGui::SetNextItemWidth(-1.F);
        int projection = static_cast<int>(app.camera.projection);
        const char* projections[] = {
            tr("Perspective"), tr("Orthographic"), tr("Fisheye"), tr("Panorama")};
        if (ImGui::Combo("##editor_projection", &projection, projections, 4)) {
            set_editor_projection(
                app.camera, static_cast<EditorProjection>(projection));
            app.preview_follow_view = false;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Perspective — standard lens with field of view.\n"
                "Orthographic — parallel view, no foreshortening.\n"
                "Fisheye — OpenCV equidistant, same model as training.\n"
                "Panorama — 360° × 180° spherical view.");

        if (app.camera.projection == EditorProjection::perspective ||
            app.camera.projection == EditorProjection::fisheye) {
            theme::caption(
                app.camera.projection == EditorProjection::fisheye
                    ? "Fisheye field of view"
                    : "Field of view");
            ImGui::SetNextItemWidth(-1.F);
            const float fov_min =
                app.camera.projection == EditorProjection::fisheye ? 80.F : 10.F;
            const float fov_max =
                app.camera.projection == EditorProjection::fisheye ? 179.F
                                                                   : 120.F;
            ImGui::SliderFloat(
                "##editor_fov", &app.camera.fov_degrees, fov_min, fov_max,
                "%.0f°");
        }
        if (app.camera.projection == EditorProjection::fisheye) {
            theme::caption("Lens distortion");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::SliderFloat(
                "##fisheye_k1", &app.camera.fisheye_k1, -0.2F, 0.4F, "%.3f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "OpenCV fisheye k1. Zero is a pure equidistant lens.");
        }
        if (app.camera.projection == EditorProjection::orthographic) {
            theme::caption("View height");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::DragFloat(
                "##ortho_height", &app.camera.ortho_height, 0.05F, 0.01F,
                1.0e5F, "%.3f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "World-space height visible in the viewport.\n"
                    "Scroll the view to zoom.");
        }
        if (app.camera.projection == EditorProjection::panorama)
            theme::caption("Full 360° × 180° spherical view around the camera.");
        theme::caption("Fly speed");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderFloat(
            "##fly_speed", &app.camera.move_speed, 0.1F, 10.F, "%.1fx");
        ImGui::Spacing();
    }

    if (app.workspace != ViewportWorkspace::image_2d &&
        app.view_mode != VisualizationMode::splat &&
        ImGui::CollapsingHeader(tr("Display & performance"), ImGuiTreeNodeFlags_None)) {
        ImGui::Spacing();
        if (app.view_mode == VisualizationMode::mesh) {
            ImGui::Spacing();
            theme::metric(
                "Vertices", format_count(app.mesh.vertices.size()).c_str());
            theme::metric(
                "Faces", format_count(app.mesh.faces.size()).c_str());
            if (app.has_texture)
                theme::metric(
                    "Texture",
                    app.settings.texture_delight ? "Albedo atlas" : "Projected atlas");
            ImGui::BeginDisabled(!app.mesh.has_texture());
            ImGui::Checkbox(tr("Albedo texture"), &app.view_options.mesh_texture);
            ImGui::EndDisabled();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip(
                    app.mesh.has_texture()
                        ? "Sample the baked atlas with mesh UVs."
                        : "Bake Texture to preview the albedo on the mesh.");
            ImGui::Checkbox(tr("Wireframe"), &app.view_options.mesh_wireframe);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Draw triangle edges on top of the shaded surface.");
            ImGui::Checkbox(
                tr("Vertex colour"), &app.view_options.mesh_vertex_colour);
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
            const bool gpu_rings =
                app.has_model &&
                !(app.job.running() && app.active_job == JobKind::train);
            if (!gpu_rings) {
                theme::caption("Ring budget");
                ImGui::SetNextItemWidth(-1.F);
                ImGui::SliderInt(
                    "##ring_budget", &app.view_options.ring_budget, 1'000, 40'000,
                    "%d");
                theme::caption("Ring radius");
                ImGui::SetNextItemWidth(-1.F);
                if (ImGui::SliderFloat(
                        "##ring_scale", &app.view_options.ring_scale, 1.F, 4.F,
                        "%.2f σ") &&
                    live_preview_active(app))
                    publish_preview_vis(app);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Ellipse size in projected sigmas. 2.83 matches SuperSplat's ring contour.");
            }
        }
        theme::caption("Camera marker size");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderFloat(
            "##view_scale", &app.view_options.view_scale, 0.02F, 0.4F, "%.2f");
        ImGui::Spacing();
        ImGui::Checkbox(tr("Colour by depth"), &app.view_options.colour_by_depth);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Replace sampled point colours with a near-to-far ramp.");
        ImGui::Spacing();
    }

}

}  // namespace editor
