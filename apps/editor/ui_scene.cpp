#include "ui.hpp"
#include "app.hpp"
#include "file_dialogs.hpp"
#include "i18n.hpp"
#include "icons.hpp"

#include "imgui_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef PHOTARA_CLI_PATH
#define PHOTARA_CLI_PATH "photara"
#endif

namespace editor {
using i18n::tr;

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
    const bool open = ImGui::Begin(i18n::id("Scene", "###Scene"), &app.show_scene);
    ImGui::PopStyleVar();
    if (!open) {
        ImGui::End();
        return;
    }

    const float wrap = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - 20.F;
    const bool external_dataset = has_external_dataset(app);

    theme::section_header(tr("SCENE"));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {7.F, 7.F});
    ImGui::SetCursorPosX(10.F);
    const std::string project_label = app.layout.project_file.empty()
        ? (app.layout.root.filename().empty()
               ? std::string(tr("Untitled Project"))
               : app.layout.root.filename().string())
        : app.layout.project_file.filename().string();
    const bool has_cloud = app.has_sparse || app.scene.has_points();
    const bool has_gaussians = app.has_model;
    const bool has_mesh = app.has_mesh;
    const bool has_reconstruction = has_cloud || has_gaussians || has_mesh;
    if (ImGui::TreeNodeEx(
            project_label.c_str(),
            ImGuiTreeNodeFlags_DefaultOpen |
                ImGuiTreeNodeFlags_SpanAvailWidth)) {
        if (!has_reconstruction) {
            ImGui::SetCursorPosX(22.F);
            ImGui::PushTextWrapPos(wrap);
            theme::caption(
                external_dataset
                    ? tr("External camera dataset selected. Train 3DGS or run Extract Mesh next.")
                    : tr("Align photos to start the reconstruction."));
            ImGui::PopTextWrapPos();
        } else {
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf |
                                       ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                       ImGuiTreeNodeFlags_SpanAvailWidth |
                                       ImGuiTreeNodeFlags_Selected;
            ImGui::TreeNodeEx(tr("Reconstruction"), flags);
            if (ImGui::IsItemClicked()) {
                if (app.view_mode == VisualizationMode::mesh)
                    show_mesh_view(app, !app.mesh.has());
                else if (
                    (app.view_mode == VisualizationMode::splat ||
                     app.view_mode == VisualizationMode::rings) &&
                    app.has_model)
                    set_visualization_mode(app, app.view_mode);
                else if (!alignment_job_running(app)) {
                    app.suppress_scene_auto_load = false;
                    if (app.has_sparse) ensure_sparse_loaded(app);
                }
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "%s",
                    tr("One reconstruction. Switch Sparse / Gaussians / Mesh\n"
                    "with the visualization buttons in the viewport."));
            std::string representations;
            if (has_cloud) representations = tr("Sparse");
            if (has_gaussians) {
                if (!representations.empty()) representations += "  ·  ";
                representations += tr("Gaussians");
            }
            if (has_mesh) {
                if (!representations.empty()) representations += "  ·  ";
                representations += tr("Mesh");
            }
            if (app.has_texture) {
                if (!representations.empty()) representations += "  ·  ";
                representations += tr("Texture");
            }
            ImGui::SetCursorPosX(44.F);
            theme::caption(representations.c_str());
        }
        ImGui::TreePop();
    }
    ImGui::PopStyleVar();
    ImGui::Dummy({0, 6.F});

    theme::section_header(tr("PIPELINE"));
    const bool busy = app.job.running();
    const Stage stage = app.monitor.stage();
    const bool aligning = busy && app.active_job == JobKind::align;
    const bool training = busy && app.active_job == JobKind::train;

    draw_step(
        "01", tr("Select images"),
        app.settings.images_dir[0] != '\0' || external_dataset
            ? StepState::done
            : StepState::pending,
        nullptr);
    draw_step(
        "02", tr("Align cameras"),
        aligning ? StepState::active
                 : (app.has_sparse || external_dataset
                        ? StepState::done
                        : StepState::pending),
        aligning
            ? stage_name(stage)
            : (external_dataset ? tr("Imported cameras (SfM skipped)") : nullptr));
    draw_step(
        "03", tr("Review sparse cloud"),
        app.scene.has_points()
            ? StepState::done
            : ((external_dataset || app.has_sparse)
                   ? StepState::active
                   : StepState::pending),
        app.scene.has_points()
            ? nullptr
            : (aligning
                   ? tr("Recomputing")
                   : (app.loading_scene
                          ? tr("Loading")
                          : (external_dataset ? tr("Imported cameras")
                                              : (app.has_sparse ? tr("On disk")
                                                                : nullptr)))));
    const bool densing = busy && app.active_job == JobKind::dense;
    const bool texturing = busy && app.active_job == JobKind::texture;
    const bool aligned = app.has_sparse || external_dataset;
    const bool mvs_route =
        !training && (mesh_from_mvs(app.settings) || densing ||
                      (app.settings.mesh_source == 1 && app.settings.build_mesh));
    const char* texture_note = nullptr;
    if (texturing)
        texture_note = stage_name(stage);
    else if (app.has_texture)
        texture_note = app.settings.texture_delight ? tr("Albedo atlas")
                                                    : tr("Projected atlas");
    else if (app.has_mesh)
        texture_note = tr("Project photos onto the mesh");
    const StepState texture_state = texturing
        ? StepState::active
        : (app.has_texture
               ? StepState::done
               : (app.has_mesh ? StepState::pending : StepState::pending));

    if (mvs_route) {
        const char* mesh_note = densing ? stage_name(stage)
                                        : (app.has_mesh ? tr("Photogrammetry") : nullptr);
        draw_step(
            "04", tr("MVS Mesh"),
            densing ? StepState::active
                    : (app.has_mesh
                           ? StepState::done
                           : (aligned ? StepState::pending : StepState::pending)),
            mesh_note);
        draw_step("05", tr("Texture Baking"), texture_state, texture_note);
    } else {
        const char* gaussian_note = nullptr;
        if (training)
            gaussian_note = stage_name(stage);
        else if (mesh_from_gaussians(app.settings) && app.has_model &&
                 app.has_mesh)
            gaussian_note = tr("Mesh extracted");
        else if (mesh_from_gaussians(app.settings))
            gaussian_note = tr("Then extract mesh");
        draw_step(
            "04", tr("Train 3DGS"),
            training ? StepState::active
                     : (app.has_model ? StepState::done : StepState::pending),
            gaussian_note);
        const bool extracting =
            training && (stage == Stage::meshing || stage == Stage::texturing);
        draw_step(
            "05", tr("Extract Mesh"),
            extracting ? StepState::active
                       : (app.has_mesh
                              ? StepState::done
                              : (mesh_from_gaussians(app.settings)
                                     ? StepState::pending
                                     : StepState::skipped)),
            app.has_mesh ? tr("From Gaussians") : nullptr);
        draw_step("06", tr("Texture Baking"), texture_state, texture_note);
    }

    ImGui::Dummy({0, 8.F});
    theme::section_header(tr("SOURCE"));
    ImGui::Indent(14.F);
    theme::caption(is_video_source(app.settings) ? tr("VIDEO") : tr("IMAGES"));
    ImGui::Spacing();
    ImGui::PushTextWrapPos(wrap);
    ImGui::TextUnformatted(
        app.settings.images_dir[0] != '\0' ? app.settings.images_dir.data()
                                          : tr("(not selected)"));
    ImGui::PopTextWrapPos();
    if (busy) {
        ImGui::Dummy({0, 4.F});
        ImGui::PushTextWrapPos(wrap);
        theme::caption(
            tr("Stop the running job to choose a different capture."));
        ImGui::PopTextWrapPos();
    }
    if (is_video_source(app.settings)) {
        ImGui::Dummy({0, 4.F});
        theme::caption(tr("EXTRACTED FRAMES"));
        ImGui::Spacing();
        ImGui::PushTextWrapPos(wrap);
        const std::string frames_utf8 =
            path_to_utf8(reconstruction_images_path(app));
        ImGui::TextUnformatted(frames_utf8.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::Dummy({0, 6.F});
    theme::caption(tr("PROJECT"));
    ImGui::Spacing();
    ImGui::PushTextWrapPos(wrap);
    ImGui::TextUnformatted(
        app.settings.project_dir[0] != '\0' ? app.settings.project_dir.data()
                                           : tr("(not selected)"));
    ImGui::PopTextWrapPos();
    if (external_dataset) {
        ImGui::Dummy({0, 6.F});
        theme::caption(tr("EXTERNAL DATASET"));
        ImGui::Spacing();
        ImGui::PushTextWrapPos(wrap);
        ImGui::TextUnformatted(app.settings.dataset_source.data());
        ImGui::PopTextWrapPos();
    }
    ImGui::Unindent(14.F);

    ImGui::End();
}

}  // namespace editor
