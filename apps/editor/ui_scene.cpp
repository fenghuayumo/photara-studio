#include "ui.hpp"
#include "app.hpp"
#include "file_dialogs.hpp"
#include "i18n.hpp"
#include "icons.hpp"

#include "imgui_internal.h"

#include <atomic>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace editor {
using i18n::tr;

namespace {
void draw_workflow_step(App& app, const char* index, const char* label,
    const InspectorSection section, const StepState state, const char* note) {
    const bool selected = current_inspector_section(app) == section;
    ImVec4 colour = theme::inactive;
    if (state == StepState::done) colour = theme::success;
    else if (state == StepState::active) colour = theme::accent;
    else if (state == StepState::failed) colour = theme::danger;
    ImGui::PushID(index);
    ImGui::SetCursorPosX(8.F);
    const ImVec2 min = ImGui::GetCursorScreenPos();
    const float width = std::max(1.F, ImGui::GetContentRegionAvail().x - 8.F);
    const float height = ImGui::GetTextLineHeight() + 22.F;
    ImGui::PushStyleColor(ImGuiCol_Header, theme::fade(theme::accent, 0.14F));
    if (ImGui::Selectable("##workflow_step", selected, 0, {width, height}))
        select_inspector_section(app, section);
    ImGui::PopStyleColor();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const float text_y = min.y + (height - ImGui::GetTextLineHeight()) * 0.5F;
    draw->AddCircleFilled({min.x + 12.F, min.y + height * 0.5F}, 3.5F,
        theme::u32(colour));
    draw->AddText({min.x + 26.F, text_y}, theme::u32(theme::text_muted), index);
    const ImVec2 text_min{min.x + 51.F, text_y};
    const ImVec2 text_max{min.x + width - 8.F, min.y + height};
    ImGui::PushStyleColor(ImGuiCol_Text, selected ? theme::accent : theme::text_bright);
    ImGui::RenderTextEllipsis(draw, text_min, text_max, text_max.x, text_max.x,
        label, nullptr, nullptr);
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(label);
        if (note) ImGui::TextUnformatted(note);
        ImGui::EndTooltip();
    }
    if (note) {
        ImGui::SetCursorPosX(34.F);
        ImGui::PushTextWrapPos(ImGui::GetWindowContentRegionMax().x - 12.F);
        theme::caption(note);
        ImGui::PopTextWrapPos();
    }
    ImGui::PopID();
}
}  // namespace

void draw_gaussian_models_panel(App& app) {
    if (app.gaussian_models.empty()) return;

    ImGui::Dummy({0, 6.F});
    theme::section_header(tr("GAUSSIAN MODELS"));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {7.F, 6.F});
    const bool merge_busy = app.gaussian_merge.running;
    // Unsaved screen-space edits bind to the composed cloud; changing
    // the composition now would silently discard them.
    const bool edits_locked = app.splat_edit.dirty();
    bool composition_changed = false;
    for (std::size_t index = 0; index < app.gaussian_models.size(); ++index) {
        GaussianModelEntry& entry = app.gaussian_models[index];
        ImGui::PushID(static_cast<int>(index));
        ImGui::SetCursorPosX(10.F);
        ImGui::BeginDisabled(merge_busy || edits_locked);
        if (ImGui::Checkbox("##gaussian_model_visible", &entry.visible))
            composition_changed = true;
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "%s", tr("Show or hide this model in the merged preview."));
        ImGui::SameLine(0.F, 6.F);
        const std::string label = entry.name + "###gaussian_model";
        const ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth |
            ImGuiTreeNodeFlags_AllowOverlap;
        const bool open = ImGui::TreeNodeEx(label.c_str(), flags);
        std::string tooltip = path_to_utf8(entry.path);
        if (entry.splat_count > 0)
            tooltip += "\n" + format_count(entry.splat_count) + " splats";
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tooltip.c_str());
        ImGui::SameLine();
        const float remove_width = ImGui::CalcTextSize("X").x + 14.F;
        ImGui::SetCursorPosX(
            ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x -
            remove_width);
        ImGui::BeginDisabled(merge_busy || edits_locked);
        if (ImGui::Button("X##remove_gaussian_model")) {
            remove_gaussian_model(app, index);
            // The entry was erased from the vector; stop touching it.
            ImGui::EndDisabled();
            if (open) ImGui::TreePop();
            ImGui::PopID();
            break;
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tr("Remove this model from the session."));
        if (open) {
            ImGui::Indent(16.F);
            ImGui::BeginDisabled(merge_busy || edits_locked);
            ImGui::DragFloat3(
                tr("Offset"), entry.offset, 0.05F, -100000.F, 100000.F,
                "%.2f");
            if (ImGui::IsItemDeactivatedAfterEdit())
                composition_changed = true;
            ImGui::DragFloat(
                tr("Scale"), &entry.scale, 0.005F, 0.01F, 1000.F, "%.3f",
                ImGuiSliderFlags_Logarithmic);
            if (ImGui::IsItemDeactivatedAfterEdit())
                composition_changed = true;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "%s",
                    tr("Uniform scale and translation applied before merge.\n"
                    "The preview reloads when the drag ends."));
            ImGui::EndDisabled();
            ImGui::Unindent(16.F);
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    if (composition_changed) refresh_gaussian_composition(app);

    ImGui::Dummy({0, 2.F});
    ImGui::SetCursorPosX(10.F);
    ImGui::BeginDisabled(merge_busy || edits_locked);
    if (ImGui::Button(tr("Add Model..."), {116.F, 0.F})) {
        std::vector<std::filesystem::path> picked;
        if (pick_splat_model_files(L"Add Gaussian splat models", picked)) {
            if (app.gaussian_models.empty()) clear_loaded_result(app);
            import_gaussian_models(app, picked);
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine(0.F, 6.F);
    const std::size_t visible_models = visible_gaussian_model_count(app);
    ImGui::BeginDisabled(merge_busy || visible_models < 2);
    if (ImGui::Button(tr("Merge Models..."), {-1.F, 0.F})) {
        std::array<char, 1024> output{};
        std::filesystem::path directory;
        if (!app.gaussian_models.empty()) {
            std::error_code error;
            directory = app.gaussian_models.front().path.parent_path();
            if (directory.empty() ||
                !std::filesystem::is_directory(directory, error))
                directory = std::filesystem::current_path(error);
        }
        store_path_field(output, directory / "merged_gaussians.ply");
        if (pick_export_path(
                L"Merge Gaussian Models", output, FilePickKind::splat_model,
                L"merged_gaussians.ply", L"ply")) {
            start_gaussian_merge(app, path_from_utf8_field(output.data()));
        }
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip(
            "%s",
            tr("Write every visible model (with its placement) into one new "
            "splat file, then load that file as the session's single model."));

    if (merge_busy) {
        ImGui::SetCursorPosX(10.F);
        const float width = ImGui::GetContentRegionAvail().x - 10.F;
        char overlay[64]{};
        std::snprintf(
            overlay, sizeof(overlay), "%s %u/%u", tr("Merging models"),
            static_cast<unsigned>(
                app.gaussian_merge.progress * app.gaussian_merge.input_count),
            app.gaussian_merge.input_count);
        ImGui::ProgressBar(app.gaussian_merge.progress, {width, 0.F}, overlay);
    }
    ImGui::PopStyleVar();
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
               : path_to_utf8(app.layout.root.filename()))
        : path_to_utf8(app.layout.project_file.filename());
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
                    show_mesh_view(app, false);
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
    draw_gaussian_models_panel(app);
    ImGui::Dummy({0, 6.F});

    theme::section_header(tr("PIPELINE"));
    const bool busy = app.job.running();
    const Stage stage = app.monitor.stage();
    const bool aligning = busy && app.active_job == JobKind::align;
    const bool training = busy && app.active_job == JobKind::train;
    const bool densing = busy && app.active_job == JobKind::dense;
    const bool texturing = busy && app.active_job == JobKind::texture;
    const bool extracting = training && stage == Stage::meshing;
    draw_workflow_step(app, "01", tr("Camera Alignment"), InspectorSection::alignment,
        aligning ? StepState::active : (alignment_ready(app) ? StepState::done : StepState::pending),
        aligning ? stage_name(stage) : (external_dataset ? tr("Imported cameras (SfM skipped)") : nullptr));
    draw_workflow_step(app, "02", tr("Train 3DGS"), InspectorSection::training,
        training && !extracting ? StepState::active : (app.has_model ? StepState::done : StepState::pending),
        training && !extracting ? stage_name(stage) : nullptr);
    draw_workflow_step(app, "03", tr("Extract Mesh"), InspectorSection::mesh,
        densing || extracting ? StepState::active : (app.has_mesh ? StepState::done : StepState::pending),
        densing || extracting ? stage_name(stage) : nullptr);
    draw_workflow_step(app, "04", tr("Texture Baking"), InspectorSection::texture,
        texturing ? StepState::active : (app.has_texture ? StepState::done : StepState::pending),
        texturing ? stage_name(stage) : nullptr);
    ImGui::Dummy({0.F, 10.F});
    ImGui::SetCursorPosX(10.F);
    if (icons::labeled_button("##project_settings", icons::Icon::sliders,
            tr("Project settings"), {std::max(1.F, ImGui::GetContentRegionAvail().x - 10.F), 32.F},
            icons::ButtonStyle::normal, true,
            current_inspector_section(app) == InspectorSection::project))
        select_inspector_section(app, InspectorSection::project);

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
