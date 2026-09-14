#include "ui.hpp"
#include "app.hpp"
#include "file_dialogs.hpp"
#include "i18n.hpp"
#include "icons.hpp"

#include "io/image.hpp"
#include "io/video_frames.hpp"

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

Action draw_inspector(App& app) {
    Action action = Action::none;
    if (!app.show_inspector) return action;
    if (!ImGui::Begin(i18n::id("Inspector", "###Inspector"), &app.show_inspector)) {
        ImGui::End();
        return action;
    }
    const bool busy = app.job.running();

    if (ImGui::CollapsingHeader(tr("Project"), ImGuiTreeNodeFlags_DefaultOpen)) {
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
        if (ImGui::Button(i18n::id("Folder", "##pick_images"), {58.F, 0})) {
            select_image_folder(app);
        }
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button(i18n::id("Video", "##pick_video"), {46.F, 0})) {
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
                if (ImGui::TreeNodeEx(
                    i18n::id("Advanced", "##video"), ImGuiTreeNodeFlags_SpanAvailWidth)) {
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
        if (ImGui::Button(i18n::id("Folder", "##pick_dataset_folder"), {58.F, 0.F}) &&
            pick_folder(
                L"Select external SfM dataset folder",
                app.settings.dataset_source)) {
            apply_external_dataset_selection(app);
        }
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button(i18n::id("File", "##pick_dataset_file"), {46.F, 0.F}) &&
            pick_dataset_file(
                L"Select external camera dataset file",
                app.settings.dataset_source)) {
            apply_external_dataset_selection(app);
        }
        if (has_external_dataset(app)) {
            theme::caption(
                "Imported cameras replace Align Photos. Review the cloud, then "
                "run Train 3DGS or Extract Mesh.");
            theme::caption("Initial point cloud (optional)");
            ImGui::SetNextItemWidth(-82.F);
            if (ImGui::InputText(
                    "##dataset_initial_cloud",
                    app.settings.dataset_initial_cloud.data(),
                    app.settings.dataset_initial_cloud.size()))
                clear_loaded_result(app);
            ImGui::SameLine(0.F, 4.F);
            if (ImGui::Button(i18n::id("File", "##pick_initial_cloud"), {46.F, 0.F}) &&
                pick_point_cloud_file(
                    L"Select initial point cloud",
                    app.settings.dataset_initial_cloud))
                clear_loaded_result(app);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Optional dense PLY initializer. COLMAP/OpenMVS sparse points "
                    "are used automatically when available.");
            if (ImGui::Button(tr("Clear external dataset"), {-1.F, 26.F})) {
                app.settings.dataset_source.fill('\0');
                app.settings.dataset_initial_cloud.fill('\0');
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
        if (ImGui::Button(i18n::id("File", "##pick_splat_model"), {46.F, 0.F}) &&
            pick_splat_model_file(
                L"Select trained Gaussian splat model",
                app.settings.splat_model_source)) {
            clear_loaded_result(app);
            refresh_artifacts(app);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Load an existing PLY, SOG, SPZ, or GLB model for preview. "
                "When set, it takes precedence over the trained working copy.");
        ImGui::EndDisabled();
        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader(
            tr("Camera Alignment"), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        ImGui::BeginDisabled(busy);
        theme::caption("Solver");
        ImGui::SetNextItemWidth(-1.F);
        const char* modes[] = {tr("Global"), tr("Incremental"), tr("Hierarchical")};
        ImGui::Combo("##sfm_mode", &app.settings.sfm_mode, modes, 3);
        theme::caption("Camera model");
        ImGui::SetNextItemWidth(-1.F);
        const char* camera_models[] = {tr("Auto"), tr("Pinhole"), tr("OpenCV Fisheye")};
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
        ImGui::Checkbox(tr("Reuse cached alignment"), &app.settings.reuse_cache);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Write a project .cache folder so later Align/Train can\n"
                "reuse extracted features. Off by default: skip that folder.\n"
                "Align/Train keep working copies for preview. They do not\n"
                "write .ascan, .asfm, or PLY files unless you Save Project\n"
                "or Export.");
        ImGui::EndDisabled();

        if (theme::toolbar_button(
                "Export SfM Alignment", {-1.F, 28.F},
                !busy && can_export_alignment(app)))
            open_alignment_export_panel(app);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Save cameras and tracks as ASFM, COLMAP,\n"
                "Nerfstudio / Blender, or OpenMVS.");
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
            tr("Gaussian Splatting"), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        ImGui::BeginDisabled(busy);
        theme::caption("Capture type");
        ImGui::SetNextItemWidth(-1.F);
        const char* capture[] = {tr("Object"), tr("Scene")};
        int capture_index = app.settings.scene_mode ? 1 : 0;
        if (ImGui::Combo("##capture", &capture_index, capture, 2))
            app.settings.scene_mode = capture_index == 1;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Object uses the reconstruction region as Splat SubjectBounds "
                "(focus region).\n"
                "Scene trains unbounded and ignores the box.");
        theme::caption("Densification strategy");
        ImGui::SetNextItemWidth(-1.F);
        const char* strategies[] = {
            "ADC IGS", "ADC Plus"};
        int strategy_index = app.settings.strategy == 1 ? 1 : 0;
        ImGui::Combo("##strategy", &strategy_index, strategies, 2);
        app.settings.strategy = strategy_index;
        theme::caption("Iterations");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::InputInt("##iterations", &app.settings.iterations, 1000, 5000);
        theme::caption("Densification cap");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::InputInt(
            "##densification_cap",
            &app.settings.densification_cap, 100'000, 1'000'000);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tr(
                "Maximum Gaussian count during densification.\n"
                "Initialization uses the full source cloud.\n"
                "Default 1,000,000."));
        theme::caption("SH degree");
        ImGui::SetNextItemWidth(-1.F);
        const char* sh_degrees[] = {"0", "1", "2", "3"};
        ImGui::Combo(
            "##sh_degree", &app.settings.sh_degree, sh_degrees, 4);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Spherical-harmonic colour bands.\n"
                "0 is diffuse only. 3 is the training default.");
        theme::caption("Max training resolution");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::InputInt(
            "##resolution", &app.settings.max_resolution, 128, 512);
        theme::caption("Live preview cadence");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::InputInt("##cadence", &app.settings.preview_interval, 10, 50);
        ImGui::Checkbox(
            tr("Coarse-to-fine resolution"), &app.settings.progressive_resolution);
        ImGui::Checkbox(tr("Foreground mask training"), &app.settings.use_mask);
        ImGui::Checkbox(tr("Normal field"), &app.settings.normal_field);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "GaussianWrapping's learned normal field.\n"
                "Off (default) trains the GGGS path.");
        ImGui::EndDisabled();
        if (theme::toolbar_button(
                "Export Splat", {-1.F, 28.F},
                !busy && can_export_model(app)))
            open_splat_export_panel(app);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Save the trained Gaussians. Choose PLY, SOG, SPZ, or GLB\n"
                "and the spherical-harmonics degree.");
        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader(
            tr("Mesh"), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        ImGui::BeginDisabled(busy);
        ImGui::Checkbox(tr("Build mesh"), &app.settings.build_mesh);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip(
                "Include a surface mesh in the reconstruction.\n"
                "Choose From Gaussians or Extract Mesh (MVS) below.");
        if (app.settings.build_mesh) {
            ImGui::Spacing();
            theme::caption("Method");
            if (ImGui::RadioButton(
                    tr("From Gaussians"), app.settings.mesh_source == 0))
                app.settings.mesh_source = 0;
            ImGui::PushTextWrapPos(0.F);
            theme::caption(
                "Train 3DGS with depth and normal losses, then extract a "
                "surface from the Gaussians.");
            ImGui::PopTextWrapPos();
            ImGui::Spacing();
            if (ImGui::RadioButton(
                    tr("Extract Mesh (MVS)"), app.settings.mesh_source == 1)) {
                app.settings.mesh_source = 1;
                if (app.settings.mesh_method == 3)
                    app.settings.mesh_method = 0;
            }
            ImGui::PushTextWrapPos(0.F);
            theme::caption(
                "PatchMatch stereo from aligned cameras, then fuse a mesh. "
                "This is the MVS mesh pipeline. 3DGS stays appearance-only.");
            ImGui::PopTextWrapPos();
            ImGui::Spacing();
            theme::caption("Surface");
            ImGui::SetNextItemWidth(-1.F);
            if (mesh_from_gaussians(app.settings)) {
                const char* methods[] = {tr("Auto"), "TSDF", tr("Delaunay"), "PAM"};
                ImGui::Combo(
                    "##mesh_method", &app.settings.mesh_method, methods, 4);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "How depth is fused into triangles.\n"
                        "PAM is GaussianWrapping occupancy meshing.");
            } else {
                if (app.settings.mesh_method == 3)
                    app.settings.mesh_method = 0;
                const char* methods[] = {tr("Auto"), "TSDF", tr("Delaunay")};
                ImGui::Combo(
                    "##mesh_method", &app.settings.mesh_method, methods, 3);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "How MVS depth maps are fused into triangles.");
            }
            if (mesh_from_gaussians(app.settings) &&
                ImGui::TreeNodeEx(
                    tr("Geometry training"), ImGuiTreeNodeFlags_SpanAvailWidth)) {
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
        if (theme::toolbar_button(
                "Export Mesh", {-1.F, 28.F},
                !busy && can_export_mesh_file(app)))
            open_mesh_export_panel(app);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Save the reconstructed surface. Choose PLY, OBJ, or GLB,\n"
                "and whether to include the baked texture.");
        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader(
            tr("Texture"), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        ImGui::PushTextWrapPos(0.F);
        theme::caption(
            "Last reconstruction step. Unwrap the mesh, project calibrated "
            "photos with aether_drender, then optionally refine the atlas.");
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
#if !defined(AETHERSCAN_HAS_TEXTURE)
        theme::caption(
            "This build was compiled without texture baking (Vulkan + aether_drender).");
#else
        ImGui::BeginDisabled(busy);
        theme::caption("Quality");
        ImGui::SetNextItemWidth(-1.F);
        const char* texture_qualities[] = {tr("Fast"), tr("Standard"), tr("High")};
        if (ImGui::Combo(
                "##tex_quality", &app.settings.texture_quality,
                texture_qualities, 3))
            apply_texture_quality_preset(app.settings);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Fast: 1024 atlas, projection only.\n"
                "Standard: 2048 atlas + photometric refine.\n"
                "High: 4096 atlas + longer refine.");
        theme::caption("Atlas size");
        ImGui::SetNextItemWidth(-1.F);
        int atlas_index = app.settings.atlas_resolution >= 8192
            ? 3
            : (app.settings.atlas_resolution >= 4096
                   ? 2
                   : (app.settings.atlas_resolution >= 2048 ? 1 : 0));
        const char* atlas_sizes[] = {"1024", "2048", "4096", "8192"};
        if (ImGui::Combo("##atlas", &atlas_index, atlas_sizes, 4)) {
            const int sizes[] = {1024, 2048, 4096, 8192};
            app.settings.atlas_resolution = sizes[atlas_index];
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Square albedo atlas written with the textured OBJ.");
        ImGui::Checkbox(tr("Remove lighting (albedo)"), &app.settings.texture_delight);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Run Intrinsic delighter on the photos before projection.\n"
                "Needs ONNX Runtime and the Intrinsic stage_*.onnx models.");
        ImGui::Checkbox(tr("Refine atlas"), &app.settings.texture_optimize);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Photometric + seam optimization in aether_drender after the "
                "projective bake. Off is faster; on cleans view seams.");
        ImGui::EndDisabled();
        if (app.has_texture) {
            ImGui::Spacing();
            theme::metric(
                "Status",
                app.settings.texture_delight ? "Albedo ready" : "Texture ready");
            const auto albedo = textured_albedo_path(app.layout.working_texture);
            if (app.atlas_preview_path != albedo) {
                app.atlas_preview.reset();
                app.atlas_preview_path = albedo;
                std::error_code atlas_error;
                if (std::filesystem::exists(albedo, atlas_error)) {
                    try {
                        app.atlas_preview.upload(aetherscan::io::load_rgb(albedo));
                    } catch (...) {
                        app.atlas_preview.reset();
                    }
                }
            }
            if (app.atlas_preview.descriptor != VK_NULL_HANDLE &&
                app.atlas_preview.width > 0 && app.atlas_preview.height > 0) {
                const float width = ImGui::GetContentRegionAvail().x;
                const float height = width *
                    static_cast<float>(app.atlas_preview.height) /
                    static_cast<float>(app.atlas_preview.width);
                ImGui::Image(
                    reinterpret_cast<ImTextureID>(app.atlas_preview.descriptor),
                    {width, (std::min)(height, 168.F)});
            }
        } else if (app.has_mesh) {
            ImGui::Spacing();
            theme::caption("Mesh is ready. Bake Texture to project the photos.");
        }
        if (theme::toolbar_button(
                app.has_texture ? "Re-bake Texture" : "Bake Texture",
                {-1.F, 28.F}, !busy && app.has_mesh))
            action = Action::texture;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "UV unwrap + multi-view projection onto the current mesh.\n"
                "Use Export Mesh to save the atlas with the model.");
#endif
        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader(tr("Display"), ImGuiTreeNodeFlags_DefaultOpen)) {
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
            ImGui::SetTooltip(
                "Splat object-mode SubjectBounds / focus region.\n"
                "Drag the center arrows to move the box, or a face dot\n"
                "to resize. Object training and mesh extraction use this volume.");
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
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Reset to the SfM SubjectBounds used by Splat object mode.");
        }
        ImGui::Spacing();
    }

    if (app.workspace == ViewportWorkspace::image_2d &&
        ImGui::CollapsingHeader(
            tr("Image QA"), ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Spacing();
        const int count = image_qa_count(app.image_qa, app.scene);
        if (app.image_qa.selected >= 0 && app.image_qa.selected < count &&
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
        ImGui::Checkbox(tr("Show triangulated"), &app.image_qa.show_triangulated);
        ImGui::Checkbox(tr("Show untracked keypoints"), &app.image_qa.show_untracked);
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

    if (app.workspace != ViewportWorkspace::image_2d &&
        ImGui::CollapsingHeader(tr("Camera"), ImGuiTreeNodeFlags_DefaultOpen)) {
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

    if (app.view_mode != VisualizationMode::splat &&
        ImGui::CollapsingHeader(tr("Viewport"), ImGuiTreeNodeFlags_DefaultOpen)) {
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
        ImGui::Spacing();
        ImGui::Checkbox(tr("Colour by depth"), &app.view_options.colour_by_depth);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Replace sampled point colours with a near-to-far ramp.");
        ImGui::Spacing();
    }

    const TrainingStats& stats = app.monitor.training();
    if (stats.valid &&
        ImGui::CollapsingHeader(
            tr("Training Telemetry"), ImGuiTreeNodeFlags_DefaultOpen)) {
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
                    tr("Continue the paused reconstruction.")))
                action = Action::resume;
        } else if (icons::labeled_button(
                       "##pause_job", icons::Icon::pause,
                       pause_job_label(app.active_job), {-1.F, 40.F},
                       icons::ButtonStyle::normal, true, false,
                       tr(k_pause_job_tooltip))) {
            action = Action::pause;
        }
        ImGui::Dummy({0, 6.F});
        if (icons::labeled_button(
                "##stop_job", icons::Icon::stop,
                stop_job_label(app.active_job), {-1.F, 36.F},
                icons::ButtonStyle::danger, true, false, tr(k_stop_job_tooltip)))
            action = Action::stop;
    } else if (app.has_mesh) {
        if (theme::primary_button(
                app.has_texture ? "Re-bake Texture" : "Bake Texture",
                {-1.F, 40.F}, true))
            action = Action::texture;
        ImGui::Dummy({0, 6.F});
        if (mesh_from_mvs(app.settings)) {
            if (theme::toolbar_button("Extract Mesh", {-1.F, 32.F}))
                action = Action::dense;
            ImGui::Dummy({0, 6.F});
            if (theme::toolbar_button("Train 3DGS", {-1.F, 32.F}))
                action = Action::train;
        } else {
            if (theme::toolbar_button(
                    mesh_from_gaussians(app.settings) ? "Train 3DGS + Mesh"
                                                      : "Train 3DGS",
                    {-1.F, 32.F}))
                action = Action::train;
            ImGui::Dummy({0, 6.F});
            if (theme::toolbar_button("Extract Mesh", {-1.F, 32.F}))
                action = Action::dense;
        }
    } else if (has_external_dataset(app)) {
        if (mesh_from_mvs(app.settings)) {
            if (theme::primary_button("Extract Mesh", {-1.F, 40.F}, true))
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
            if (theme::toolbar_button("Extract Mesh", {-1.F, 32.F}))
                action = Action::dense;
        }
    } else if (!app.has_sparse) {
        if (theme::primary_button(
                "Align Photos", {-1.F, 40.F},
                app.settings.images_dir[0] != '\0'))
            action = Action::align;
    } else if (mesh_from_mvs(app.settings)) {
        if (theme::primary_button(
                "Extract Mesh", {-1.F, 40.F},
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
        if (theme::toolbar_button("Extract Mesh", {-1.F, 32.F}))
            action = Action::dense;
    }

    ImGui::End();
    return action;
}

}  // namespace editor
