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
#include <filesystem>
#include <string>
#include <vector>

#ifndef AETHERSCAN_CLI_PATH
#define AETHERSCAN_CLI_PATH "aetherscan"
#endif

namespace editor {
using i18n::tr;

std::string mesh_export_stem_utf8(const App& app) {
    return path_to_utf8(
               app.layout.project_file.empty()
                   ? std::filesystem::path("project")
                   : app.layout.project_file.stem()) +
           "_mesh";
}

void draw_mesh_export_modal(App& app) {
    constexpr const char* popup = "###ExportMesh";
    if (app.mesh_export.show) {
        ImGui::OpenPopup(popup);
        app.mesh_export.show = false;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        viewport->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));
    ImGui::SetNextWindowSize({448.F, 0.F}, ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(
            i18n::id("Export Mesh", popup), nullptr,
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings))
        return;

    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 416.F);
    theme::caption("Choose a format, then pick where to save.");
    ImGui::PopTextWrapPos();
    ImGui::Spacing();

    theme::caption("Format");
    const float gap = 8.F;
    const float tile_w =
        (ImGui::GetContentRegionAvail().x - gap * 2.F) / 3.F;
    const ImVec2 tile{tile_w, 58.F};
    if (theme::choice_tile(
            "##fmt_ply", "PLY", "Geometry", app.mesh_export.format == 0, tile))
        app.mesh_export.format = 0;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Binary mesh with vertex colour.\n"
            "Best for CloudCompare and research tools.");
    ImGui::SameLine(0.F, gap);
    if (theme::choice_tile(
            "##fmt_obj", "OBJ", "DCC", app.mesh_export.format == 1, tile))
        app.mesh_export.format = 1;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Wavefront OBJ. Texture writes MTL + PNG alongside.\n"
            "Opens in Blender, Maya, and MeshLab.");
    ImGui::SameLine(0.F, gap);
    if (theme::choice_tile(
            "##fmt_glb", "GLB", "glTF", app.mesh_export.format == 2, tile))
        app.mesh_export.format = 2;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Single-file glTF 2.0 binary.\n"
            "Texture is packed into the file.");

    ImGui::Spacing();
    const bool ply = app.mesh_export.format == 0;
    const bool has_texture = can_export_textured_mesh(app);
    const bool texture_enabled = !ply && has_texture;
    ImGui::BeginDisabled(!texture_enabled);
    ImGui::Checkbox(tr("Include texture"), &app.mesh_export.include_texture);
    ImGui::EndDisabled();
    if (ply)
        theme::caption("PLY keeps vertex colour only.");
    else if (!has_texture)
        theme::caption("Bake Texture to include an albedo atlas.");
    else if (app.mesh_export.format == 1)
        theme::caption("Writes OBJ, MTL, and albedo PNG next to each other.");
    else
        theme::caption("Embeds the albedo atlas in the GLB.");

    ImGui::Spacing();
    const bool with_texture = mesh_export_wants_texture(app);
    const std::string stem = mesh_export_stem_utf8(app);
    std::vector<std::string> files;
    if (app.mesh_export.format == 0)
        files.push_back(stem + ".ply");
    else if (app.mesh_export.format == 1) {
        files.push_back(stem + ".obj");
        if (with_texture) {
            files.push_back(stem + ".mtl");
            files.push_back(stem + "_albedo.png");
        }
    } else {
        files.push_back(stem + ".glb");
    }

    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::surface_2);
    const float writes_h =
        ImGui::GetTextLineHeightWithSpacing() *
            static_cast<float>(files.size() + 1) +
        14.F;
    ImGui::BeginChild(
        "##export_writes", ImVec2(-1.F, writes_h), true,
        ImGuiWindowFlags_NoScrollbar);
    theme::caption("Writes");
    ImGui::PushFont(theme::mono_font());
    for (const std::string& file : files)
        ImGui::TextUnformatted(file.c_str());
    ImGui::PopFont();
    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    const float export_w = 108.F;
    const float cancel_w = 88.F;
    const float avail = ImGui::GetContentRegionAvail().x;
    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX() + std::max(0.F, avail - export_w - cancel_w - 8.F));
    if (theme::toolbar_button("Cancel", {cancel_w, 30.F}))
        ImGui::CloseCurrentPopup();
    ImGui::SameLine(0.F, 8.F);
    if (theme::primary_button("Export...", {export_w, 30.F})) {
        ImGui::CloseCurrentPopup();
        export_mesh_file(app);
    }
    ImGui::EndPopup();
}

void draw_alignment_export_modal(App& app) {
    constexpr const char* popup = "###ExportAlignment";
    if (app.alignment_export.show) {
        ImGui::OpenPopup(popup);
        app.alignment_export.show = false;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        viewport->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));
    ImGui::SetNextWindowSize({448.F, 0.F}, ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(
            i18n::id("Export SfM Alignment", popup), nullptr,
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings))
        return;

    theme::caption("Format");
    ImGui::SetNextItemWidth(-1.F);
    const char* formats[] = {
        "AetherScan ASFM", "COLMAP", "Nerfstudio / Blender", "OpenMVS"};
    if (ImGui::Combo(
            "##alignment_format", &app.alignment_export.format, formats, 4))
        sync_alignment_export_path(app, true);
    const auto format = alignment_export_format(app);
    const bool mvs_ok = alignment_mvs_supported(app);
    if (format == AlignmentExportFormat::asfm)
        theme::caption("Native scene: cameras, keypoints, and tracks.");
    else if (format == AlignmentExportFormat::colmap)
        theme::caption("cameras.txt, images.txt, and points3D.txt in a folder.");
    else if (format == AlignmentExportFormat::nerfstudio)
        theme::caption("transforms.json for Nerfstudio and Blender.");
    else if (!mvs_ok)
        theme::caption("OpenMVS needs rectified pinhole images.");
    else
        theme::caption("OpenMVS interface for Viewer and densify import.");

    ImGui::Spacing();
    theme::caption(
        format == AlignmentExportFormat::colmap ? "Folder" : "Location");
    ImGui::SetNextItemWidth(-78.F);
    ImGui::InputText(
        "##alignment_path", app.alignment_export.path.data(),
        app.alignment_export.path.size());
    ImGui::SameLine(0.F, 8.F);
    if (ImGui::Button(tr("Browse"), {70.F, 0.F}))
        browse_alignment_export_path(app);

    ImGui::Spacing();
    ImGui::Checkbox(tr("Export point cloud"), &app.alignment_export.write_ply);
    if (format == AlignmentExportFormat::colmap)
        theme::caption("Writes points3D.txt and points3D.ply.");
    else
        theme::caption("Writes a coloured PLY next to the scene file.");

    ImGui::Spacing();
    const auto out = path_from_utf8_field(app.alignment_export.path.data());
    const std::string file_name = path_to_utf8(
        out.filename().empty() ? alignment_export_stem_name(app)
                               : out.filename());
    std::vector<std::string> files;
    if (format == AlignmentExportFormat::colmap) {
        files.push_back("cameras.txt");
        files.push_back("images.txt");
        files.push_back("points3D.txt");
        if (app.alignment_export.write_ply)
            files.push_back("points3D.ply");
    } else {
        files.push_back(file_name);
        if (app.alignment_export.write_ply) {
            auto ply = out.empty() ? alignment_export_stem_name(app) : out;
            ply.replace_extension();
            ply += "_sparse.ply";
            files.push_back(path_to_utf8(ply.filename()));
        }
    }

    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::surface_2);
    const float writes_h =
        ImGui::GetTextLineHeightWithSpacing() *
            static_cast<float>(files.size() + 1) +
        14.F;
    ImGui::BeginChild(
        "##alignment_writes", ImVec2(-1.F, writes_h), true,
        ImGuiWindowFlags_NoScrollbar);
    theme::caption("Writes");
    ImGui::PushFont(theme::mono_font());
    for (const std::string& file : files)
        ImGui::TextUnformatted(file.c_str());
    ImGui::PopFont();
    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    const float export_w = 108.F;
    const float cancel_w = 88.F;
    const float avail = ImGui::GetContentRegionAvail().x;
    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX() + std::max(0.F, avail - export_w - cancel_w - 8.F));
    if (theme::toolbar_button("Cancel", {cancel_w, 30.F}))
        ImGui::CloseCurrentPopup();
    ImGui::SameLine(0.F, 8.F);
    const bool can_go =
        app.alignment_export.path[0] != '\0' &&
        (format != AlignmentExportFormat::openmvs || mvs_ok);
    if (theme::primary_button("Export", {export_w, 30.F}, can_go)) {
        ImGui::CloseCurrentPopup();
        export_alignment(app);
    }
    ImGui::EndPopup();
}

void draw_splat_export_modal(App& app) {
    constexpr const char* popup = "###ExportSplat";
    if (app.splat_export.show) {
        ImGui::OpenPopup(popup);
        app.splat_export.show = false;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        viewport->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));
    ImGui::SetNextWindowSize({448.F, 0.F}, ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(
            i18n::id("Export Splat", popup), nullptr,
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings))
        return;

    theme::caption("Format");
    ImGui::SetNextItemWidth(-1.F);
    const char* formats[] = {"PLY", "SOG", "SPZ", "GLB"};
    if (ImGui::Combo("##splat_export_format", &app.splat_export.format, formats, 4))
        sync_splat_export_path(app, true);
    if (app.splat_export.format == 1)
        theme::caption("PlayCanvas compressed Gaussians.");
    else if (app.splat_export.format == 2)
        theme::caption("Niantic SPZ.");
    else if (app.splat_export.format == 3)
        theme::caption("Khronos KHR_gaussian_splatting GLB.");
    else
        theme::caption("Standard 3DGS PLY.");

    ImGui::Spacing();
    theme::caption("SH degree");
    ImGui::SetNextItemWidth(-1.F);
    const char* degrees[] = {"0", "1", "2", "3"};
    ImGui::Combo(
        "##splat_sh_degree", &app.splat_export.sh_degree, degrees, 4);
    theme::caption("0 keeps only base colour. 3 is the training default.");

    ImGui::Spacing();
    theme::caption("Location");
    ImGui::SetNextItemWidth(-78.F);
    ImGui::InputText(
        "##splat_path", app.splat_export.path.data(),
        app.splat_export.path.size());
    ImGui::SameLine(0.F, 8.F);
    if (ImGui::Button(tr("Browse"), {70.F, 0.F}))
        browse_splat_export_path(app);

    ImGui::Spacing();
    const auto out = path_from_utf8_field(app.splat_export.path.data());
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::surface_2);
    const float writes_h = ImGui::GetTextLineHeightWithSpacing() * 2.F + 14.F;
    ImGui::BeginChild(
        "##splat_writes", ImVec2(-1.F, writes_h), true,
        ImGuiWindowFlags_NoScrollbar);
    theme::caption("Writes");
    ImGui::PushFont(theme::mono_font());
    ImGui::TextUnformatted(
        path_to_utf8(out.filename().empty() ? std::filesystem::path("splat")
                                            : out.filename())
            .c_str());
    ImGui::PopFont();
    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    const float export_w = 108.F;
    const float cancel_w = 88.F;
    const float avail = ImGui::GetContentRegionAvail().x;
    ImGui::SetCursorPosX(
        ImGui::GetCursorPosX() + std::max(0.F, avail - export_w - cancel_w - 8.F));
    if (theme::toolbar_button("Cancel", {cancel_w, 30.F}))
        ImGui::CloseCurrentPopup();
    ImGui::SameLine(0.F, 8.F);
    if (theme::primary_button(
            "Export", {export_w, 30.F}, app.splat_export.path[0] != '\0')) {
        ImGui::CloseCurrentPopup();
        export_trained_model(app);
    }
    ImGui::EndPopup();
}

}  // namespace editor
