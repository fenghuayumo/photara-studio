#include "ui.hpp"
#include "ui.hpp"
#include "file_dialogs.hpp"
#include "i18n.hpp"
#include "icons.hpp"
#include "sam_prompt_suggestions.hpp"

#include "io/image.hpp"
#include "io/video_frames.hpp"
#include "sam/model_cache.hpp"
#include "splat/training_preset.hpp"

#include "imgui_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace editor {
using i18n::tr;

namespace {
using photara::splat::LossGroup;
using photara::splat::TrainingLossSettings;

void objective_note(const char* message) {
    ImGui::PushTextWrapPos(0.F);
    ImGui::TextDisabled("%s", tr(message));
    ImGui::PopTextWrapPos();
}

bool objective_field_visible(const ProjectSettings& settings,
    const photara::splat::TrainingLossField& field) {
    const auto member = field.value;
    if (member == &TrainingLossSettings::shape_max_axis_ratio)
        return settings.shape_axis_ratio_reg > 0.F;
    if (field.group != LossGroup::colour) return true;
    if (member == &TrainingLossSettings::bilateral_grid_tv_weight ||
        member == &TrainingLossSettings::bilateral_grid_deviation_limit)
        return settings.bilateral_grid;
    if (settings.ppisp_layout == 0) return false;
    if (member == &TrainingLossSettings::ppisp_reg_vig_center ||
        member == &TrainingLossSettings::ppisp_reg_vig_non_pos ||
        member == &TrainingLossSettings::ppisp_reg_vig_channel_var)
        return settings.ppisp_model >= 1;
    if (member == &TrainingLossSettings::ppisp_reg_crf_channel_var)
        return settings.ppisp_model == 2;
    return true;
}

void objective_field_tooltip(const photara::splat::TrainingLossField& field) {
    ImGui::BeginTooltip();
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 26.F);
    ImGui::TextUnformatted(tr(field.help));
    ImGui::Spacing();
    if (field.default_value == 0.F)
        ImGui::Text("%s: %s (0)", tr("Default value"), tr("Off"));
    else
        ImGui::Text("%s: %.5g", tr("Default value"), field.default_value);
    if (field.has_tuning_range()) {
        ImGui::Text("%s: %.5g - %.5g", tr("Suggested range"),
            field.tuning_minimum, field.tuning_maximum);
    }
    if (field.tuning_note) ImGui::TextUnformatted(tr(field.tuning_note));
    ImGui::Spacing();
    if (field.maximum == std::numeric_limits<float>::max())
        ImGui::TextDisabled("%s: >= %.5g", tr("Allowed range"), field.minimum);
    else
        ImGui::TextDisabled("%s: %.5g - %.5g", tr("Allowed range"), field.minimum, field.maximum);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

void objective_fields(ProjectSettings& settings, LossGroup group, bool advanced) {
    if (!ImGui::BeginTable("##objective_fields", 2,
            ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings)) return;
    ImGui::TableSetupColumn("Label", ImGuiTableColumnFlags_WidthStretch, 0.62F);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 0.38F);
    for (const auto& field : photara::splat::k_training_loss_fields) {
        if (field.group != group || field.advanced != advanced ||
            !objective_field_visible(settings, field)) continue;
        if (settings.training_mode == 0 && !photara::splat::is_rgb_loss(field)) continue;
        ImGui::PushID(field.cli);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::AlignTextToFramePadding();
        ImGui::BeginGroup();
        ImGui::PushTextWrapPos(0.F);
        ImGui::TextUnformatted(tr(field.label));
        if (field.has_tuning_range())
            ImGui::TextDisabled("%s %.5g - %.5g", tr(field.default_value == 0.F
                ? "When enabled" : "Suggested"), field.tuning_minimum, field.tuning_maximum);
        ImGui::PopTextWrapPos();
        ImGui::EndGroup();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) objective_field_tooltip(field);
        ImGui::TableNextColumn();
        const bool alpha_disabled = field.value == &TrainingLossSettings::match_alpha_weight &&
            (!(settings.use_mask || settings.sam_masks) || settings.mask_mode != 1);
        const bool normal_disabled = (field.value == &TrainingLossSettings::normal_field_weight ||
            field.value == &TrainingLossSettings::normal_field_depth_ratio) && !settings.normal_field;
        ImGui::BeginDisabled(alpha_disabled || normal_disabled);
        ImGui::SetNextItemWidth(-1.F);
        ImGui::InputFloat("##value", &(settings.*field.value), 0.F, 0.F, "%.5g");
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            objective_field_tooltip(field);
        ImGui::EndDisabled();
        ImGui::PopID();
    }
    ImGui::EndTable();
}

void draw_ai_priors(App& app) {
    auto& settings=app.settings;
    ImGui::Separator();
    ImGui::Checkbox(tr("AI depth & normal priors"), &settings.ai_priors);
    objective_note("AI priors are independent of surface consistency and multi-view losses. Zero disables each prior.");
    if (!settings.ai_priors) return;
    const char* sources[]{tr("Generate with MoGe"), tr("Existing aligned cache")};
    theme::caption("Prior source");
    ImGui::SetNextItemWidth(-1.F);
    ImGui::Combo("##ai_prior_source", &settings.ai_prior_source, sources, 2);
    if (settings.ai_prior_source == 0) {
        objective_note("MoGe-2 ViT-B Normal, MIT license, 400 MiB. Downloaded automatically before first training and reused across projects.");
#if defined(PHOTARA_HAS_INFERENCE)
        if (moge_model_ready()) {
            ImGui::TextColored(theme::success,"%s",tr("MoGe model is ready"));
        } else if (app.moge_download.state()==ModelDownload::State::running) {
            ImGui::ProgressBar(std::max(0.F,app.moge_download.progress()),{-1.F,0.F});
            if (ImGui::SmallButton(tr("Show download"))) app.show_moge_download=true;
        } else if (ImGui::Button(tr("Download model"))) {
            request_moge_download(app);
        }
#else
        objective_note("MoGe inference is unavailable in this build. Choose an existing aligned cache.");
#endif
        theme::caption("Prior token budget");
        ImGui::SetNextItemWidth(-1.F);
        ImGui::InputInt("##ai_prior_tokens", &settings.ai_prior_tokens, 100, 400);
        settings.ai_prior_tokens = std::clamp(settings.ai_prior_tokens, 1, 10000);
    }
    theme::caption(settings.ai_prior_source == 0 ? "Prior cache folder (optional)" : "Aligned prior cache folder");
    ImGui::SetNextItemWidth(-68.F);
    ImGui::InputText("##ai_prior_directory", settings.ai_prior_directory.data(), settings.ai_prior_directory.size());
    ImGui::SameLine(0.F, 4.F);
    if (ImGui::Button(i18n::id("Folder", "##ai_prior_folder_pick"), {64.F,0.F}))
        pick_folder(L"Select aligned AI prior cache folder", settings.ai_prior_directory);
    if (settings.ai_prior_source == 0)
        objective_note("Training prepares priors automatically and reuses compatible maps. Leave the folder empty to use the task cache.");
    else
        objective_note("Select the folder containing aligned view_*.prior files for these cameras and images.");
    const auto weight = [](const char* label, const char* id, float& value) {
        theme::caption(label);
        ImGui::SetNextItemWidth(-1.F);
        ImGui::InputFloat(id, &value, 0.F, 0.F, "%.5g");
        if (!std::isfinite(value) || value < 0.F) value = 0.F;
    };
    weight("AI depth weight", "##ai_depth_weight", settings.ai_depth_weight);
    weight("AI normal weight", "##ai_normal_weight", settings.ai_normal_weight);
    if (settings.ai_depth_weight == 0.F && settings.ai_normal_weight == 0.F)
        objective_note("Both weights are zero. Prior preparation and loading are skipped.");
    objective_note("Requires pinhole views. Fisheye images are rectified; panoramas are not supported.");
}

void draw_training_objective(App& app) {
    if (!ImGui::TreeNodeEx(tr("Loss & constraints"), ImGuiTreeNodeFlags_SpanAvailWidth)) return;
    auto& settings = app.settings;
    if (settings.training_mode == 0) {
        photara::splat::disable_training_constraints(settings);
        settings.normal_field = false;
        settings.ppisp_layout = 0;
        settings.bilateral_grid = false;
        objective_note("Fast mode uses RGB image fitting (L1 + SSIM) only. Switch to Standard or High quality to enable constraints.");
        objective_fields(settings, LossGroup::image, false);
        settings.sanitize();
        ImGui::TreePop();
        return;
    }
    objective_note("Fit the photos, stabilize geometry and limit overfitting. Zero disables a penalty.");
    if (ImGui::BeginTabBar("##objective_tabs", ImGuiTabBarFlags_FittingPolicyScroll)) {
        const struct { const char* title; LossGroup group; } tabs[]{
            {"Image fitting", LossGroup::image}, {"Geometry", LossGroup::geometry},
            {"Gaussians", LossGroup::gaussian}, {"Colour", LossGroup::colour}};
        for (const auto& tab : tabs) {
            if (!ImGui::BeginTabItem(tr(tab.title))) continue;
            ImGui::PushID(static_cast<int>(tab.group));
            const bool geometry = settings.geometry_training || mesh_from_gaussians(settings);
            if (tab.group == LossGroup::image)
                objective_note("Image loss = image weight x [(1 - structure balance) x L1 + structure balance x SSIM loss].");
            if (tab.group == LossGroup::image)
                objective_note("Mask opacity fitting requires Transparent mask mode.");
            if (tab.group == LossGroup::gaussian)
                objective_note("Soft constraints on individual Gaussians. Start with the defaults and change one strength at a time.");
            if (tab.group == LossGroup::geometry) {
                bool enabled = geometry;
                ImGui::BeginDisabled(mesh_from_gaussians(settings));
                if (ImGui::Checkbox(tr("Surface consistency"), &enabled)) settings.geometry_training = enabled;
                ImGui::EndDisabled();
                if (mesh_from_gaussians(settings))
                    objective_note("Enabled automatically for Gaussian mesh extraction.");
                else if (!geometry)
                    objective_note("Enable surface consistency to train geometry without extracting a mesh.");
            }
            if (tab.group == LossGroup::colour) {
                bool ppisp = settings.ppisp_layout != 0;
                if (ImGui::Checkbox(tr("Exposure & white balance"), &ppisp)) settings.ppisp_layout = ppisp ? 1 : 0;
                if (ppisp) {
                    const char* models[]{tr("Exposure & white balance"), tr("With vignetting"), tr("With camera response")};
                    ImGui::SetNextItemWidth(-1.F);
                    ImGui::Combo("##ppisp_model", &settings.ppisp_model, models, 3);
                }
                ImGui::Checkbox(tr("Spatial colour correction"), &settings.bilateral_grid);
                objective_note("Colour correction is used during training only. Exported Gaussians keep their canonical colour.");
            }
            ImGui::BeginDisabled(tab.group == LossGroup::geometry && !geometry);
            objective_fields(settings, tab.group, false);
            ImGui::EndDisabled();
            if (tab.group == LossGroup::geometry) {
                theme::caption("Geometry start step");
                ImGui::SetNextItemWidth(-1.F);
                ImGui::InputInt("##geometry_start", &settings.geometry_from_iter, 500, 2000);
                if (settings.geometry_from_iter > settings.iterations)
                    objective_note("Starts after this training run.");
            }
            if (tab.group == LossGroup::geometry)
                ImGui::Checkbox(tr("Normal field"), &settings.normal_field);
            if (tab.group == LossGroup::geometry) draw_ai_priors(app);
            const bool has_advanced = std::any_of(
                photara::splat::k_training_loss_fields.begin(),
                photara::splat::k_training_loss_fields.end(), [&](const auto& field) {
                    return field.group == tab.group && field.advanced &&
                        objective_field_visible(settings, field);
                });
            if (has_advanced && ImGui::TreeNodeEx(
                    tr("Advanced constraints"), ImGuiTreeNodeFlags_SpanAvailWidth)) {
                objective_fields(settings, tab.group, true);
                if (tab.group == LossGroup::geometry) {
                    ImGui::BeginDisabled(!settings.normal_field);
                    theme::caption("Normal-field start step");
                    ImGui::SetNextItemWidth(-1.F);
                    ImGui::InputInt("##normal_field_start", &settings.normal_field_from_iter, 1000, 5000);
                    if (settings.normal_field_from_iter >= settings.iterations)
                        objective_note("Starts after this training run.");
                    ImGui::EndDisabled();
                }
                ImGui::TreePop();
            }
            if (ImGui::SmallButton(tr("Reset weights"))) {
                const TrainingLossSettings defaults;
                for (const auto& field : photara::splat::k_training_loss_fields)
                    if (field.group == tab.group) settings.*field.value = defaults.*field.value;
            }
            ImGui::PopID();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    settings.sanitize();
    settings.geometry_from_iter = std::max(settings.geometry_from_iter, 0);
    ImGui::TreePop();
}
}  // namespace

InspectorSection current_inspector_section(const App& app) {
    if (app.inspector_section != InspectorSection::automatic)
        return app.inspector_section;
    if (app.job.running()) {
        switch (app.active_job) {
            case JobKind::align: return InspectorSection::alignment;
            case JobKind::train: return InspectorSection::training;
            case JobKind::dense: return InspectorSection::mesh;
            case JobKind::texture: return InspectorSection::texture;
            default: break;
        }
    }
    if (app.has_mesh) return InspectorSection::texture;
    if (app.has_model) return InspectorSection::mesh;
    if (alignment_ready(app)) return mesh_from_mvs(app.settings)
        ? InspectorSection::mesh : InspectorSection::training;
    return app.settings.images_dir[0] != '\0'
        ? InspectorSection::alignment : InspectorSection::project;
}

void select_inspector_section(App& app, const InspectorSection section) {
    app.inspector_section = section;
    app.show_inspector = true;
}

namespace {
void inspector_field(const char* label) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::AlignTextToFramePadding();
    ImGui::PushTextWrapPos(0.F);
    ImGui::TextUnformatted(tr(label));
    ImGui::PopTextWrapPos();
    ImGui::TableSetColumnIndex(1);
}

const char* inspector_title(const InspectorSection section) {
    switch (section) {
        case InspectorSection::project: return "Project settings";
        case InspectorSection::alignment: return "Camera Alignment";
        case InspectorSection::training: return "Train 3DGS";
        case InspectorSection::mesh: return "Extract Mesh";
        case InspectorSection::texture: return "Texture Baking";
        default: return "Reconstruction settings";
    }
}

const char* inspector_description(const InspectorSection section) {
    switch (section) {
        case InspectorSection::project: return "Choose capture sources and project files.";
        case InspectorSection::alignment: return "Estimate capture cameras and the sparse cloud.";
        case InspectorSection::training: return "Train a Gaussian model from the aligned photographs.";
        case InspectorSection::mesh: return "Build a surface from Gaussians or MVS depth maps.";
        case InspectorSection::texture: return "Unwrap the mesh and project calibrated photographs.";
        default: return "Select a reconstruction step in the scene panel.";
    }
}
// Labels in the alignment form need more contrast than secondary captions.
void alignment_label(const char* label) {
    ImGui::TextColored(theme::text_muted, "%s", tr(label));
}

void alignment_method_combo(const char* id, std::string& value,
    const char* const* labels, const char* const* tokens, const int count,
    const int experimental_begin) {
    int selected = -1;
    for (int i = 0; i < count; ++i)
        if (value == tokens[i]) selected = i;
    std::string preview = selected >= 0 ? tr(labels[selected]) : value;
    if (selected < 0 || selected >= experimental_begin)
        preview += std::string(" · ") + tr("Unavailable");
    ImGui::SetNextItemWidth(-1.F);
    if (ImGui::BeginCombo(id, preview.c_str())) {
        for (int i = 0; i < count; ++i) {
            const bool available = i < experimental_begin;
            std::string label = tr(labels[i]);
            if (!available) label += std::string(" · ") + tr("Not ready");
            ImGui::BeginDisabled(!available);
            if (ImGui::Selectable(label.c_str(), selected == i)) value = tokens[i];
            ImGui::EndDisabled();
            if (!available && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(ImGui::GetFontSize() * 23.F);
                ImGui::TextUnformatted(tr(
                    "Experimental support is incomplete. This option is temporarily unavailable."));
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
        }
        ImGui::EndCombo();
    }
}

void draw_alignment_configuration(App& app) {
    auto& settings = app.settings;
    alignment_label("Capture type");
    const float half_width = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5F;
    if (theme::toolbar_button(tr("Unordered photos"), {half_width, 30.F}, true,
            settings.alignment_capture_type == 0))
        settings.alignment_capture_type = 0;
    ImGui::SameLine();
    if (theme::toolbar_button(tr("Video sequence"), {half_width, 30.F}, true,
            settings.alignment_capture_type == 1))
        settings.alignment_capture_type = 1;
    if (settings.alignment_capture_type == 1) {
        if (ImGui::Button(tr("Import frames..."), {half_width, 26.F})) select_image_folder(app);
        ImGui::SameLine();
        if (ImGui::Button(tr("Import video..."), {half_width, 26.F})) select_video_file(app);
    } else if (ImGui::Button(tr("Import photos..."), {-1.F, 26.F})) {
        select_image_folder(app);
    }
    if (settings.images_dir[0] != '\0') {
        const auto source = path_from_utf8_field(settings.images_dir.data());
        const auto filename = path_to_utf8(source.filename());
        ImGui::TextWrapped("%s", filename.c_str());
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", settings.images_dir.data());
    } else {
        alignment_label("No source selected");
    }
    ImGui::Spacing();
    alignment_label("Alignment quality");
    const char* qualities[] = {tr("Fast"), tr("Standard"), tr("High")};
    const int quality = std::clamp(settings.alignment_quality, 0, 2);
    const int expected_features[] = {12'000, 27'000, 54'000};
    const int expected_overlap[] = {3, 6, 10};
    const bool customized = settings.max_features != expected_features[quality] ||
        settings.sequential_overlap != expected_overlap[quality] ||
        !settings.retrieval_augmentation || settings.refine_weak_cameras != (quality == 2);
    std::string preview = qualities[quality];
    if (customized) preview += std::string(" · ") + tr("Customized");
    ImGui::SetNextItemWidth(-1.F);
    if (ImGui::BeginCombo("##alignment_quality", preview.c_str())) {
        for (int i = 0; i < 3; ++i)
            if (ImGui::Selectable(qualities[i], quality == i)) apply_alignment_quality_preset(settings, i);
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", tr("Quality sets the feature and matching budgets. Advanced settings allow overrides."));
    ImGui::Spacing();
    if (!ImGui::TreeNodeEx(tr("Advanced alignment settings"), ImGuiTreeNodeFlags_SpanAvailWidth)) return;

    if (ImGui::CollapsingHeader(tr("Cameras and lenses"))) {
        alignment_label("Camera sharing");
        const char* sharing[] = {tr("Per folder"), tr("All images"), tr("Each image")};
        ImGui::SetNextItemWidth(-1.F);
        ImGui::Combo("##camera_sharing", &settings.camera_sharing, sharing, 3);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tr("Per folder groups images by folder and EXIF lens identity. All images shares intrinsics globally. Each image estimates its own intrinsics."));
        alignment_label("Camera model");
        const char* models[] = {tr("Auto"), tr("Pinhole"), tr("OpenCV Fisheye"), tr("Equirectangular 360")};
        const int model_values[] = {2, 0, 1, 3};
        int model = 0;
        for (int i = 0; i < 4; ++i) if (settings.camera_model == model_values[i]) model = i;
        ImGui::SetNextItemWidth(-1.F);
        if (ImGui::Combo("##camera_model", &model, models, 4)) settings.camera_model = model_values[model];
        theme::metric("Focal initialization", settings.initial_focal_pixels > 0.0 ? tr("Manual") : tr("Automatic"));
        if (ImGui::TreeNodeEx(tr("Manual camera calibration"), ImGuiTreeNodeFlags_SpanAvailWidth)) {
            ImGui::TextWrapped("%s", tr("Leave focal length automatic unless you have camera calibration data."));
            alignment_label("Initial focal length (pixels, 0 = automatic)");
            ImGui::SetNextItemWidth(-1.F);
            if (ImGui::InputDouble("##initial_focal", &settings.initial_focal_pixels, 10.0, 100.0, "%.1f"))
                settings.initial_focal_pixels = std::max(0.0, settings.initial_focal_pixels);
            ImGui::BeginDisabled(settings.initial_focal_pixels <= 0.0);
            ImGui::Checkbox(tr("Lock calibrated focal length"), &settings.trust_initial_focal);
            ImGui::EndDisabled();
            if (settings.initial_focal_pixels <= 0.0) settings.trust_initial_focal = false;
            if (settings.initial_focal_pixels > 0.0 && ImGui::Button(tr("Use automatic focal length"))) {
                settings.initial_focal_pixels = 0.0;
                settings.trust_initial_focal = false;
            }
            ImGui::TreePop();
        }
    }
    if (ImGui::CollapsingHeader(tr("Pairing and features"))) {
        alignment_label("Image matching");
        const char* strategies[] = {tr("Automatic"), tr("Sequential"), tr("Exhaustive")};
        ImGui::SetNextItemWidth(-1.F);
        ImGui::Combo("##matching_strategy", &settings.matching_strategy, strategies, 3);
        if (alignment_uses_sequential_pairs(settings)) {
            alignment_label("Sequential overlap");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::SliderInt("##sequential_overlap", &settings.sequential_overlap, 1, 20);
        }
        if (settings.matching_strategy != 2) {
            ImGui::Checkbox(tr("Add image retrieval pairs"), &settings.retrieval_augmentation);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr("Retrieval adds likely distant-view and loop-closure pairs."));
            if (!settings.retrieval_augmentation && !alignment_uses_sequential_pairs(settings))
                ImGui::TextWrapped("%s", tr("With retrieval disabled, unordered photos use exhaustive matching."));
        }
        ImGui::Spacing();
        alignment_label("Feature extractor");
        const bool supported_extractor = settings.feature_extractor == "siftgpu" ||
            settings.feature_extractor == "vulkan_sift" || settings.feature_extractor == "sift";
        const char* extractors[] = {"SIFT", "SuperPoint", "DISK", "ALIKED"};
        const char* extractor_tokens[] = {"siftgpu", "superpoint", "disk", "aliked"};
        std::string extractor_algorithm = supported_extractor ? "siftgpu" : settings.feature_extractor;
        const auto previous_extractor = extractor_algorithm;
        alignment_method_combo("##feature_extractor", extractor_algorithm, extractors, extractor_tokens, 4, 1);
        if (extractor_algorithm != previous_extractor) settings.feature_extractor = extractor_algorithm;
        alignment_label("Max features per image");
        ImGui::SetNextItemWidth(-1.F);
        if (ImGui::InputInt("##max_features", &settings.max_features, 1000, 5000))
            settings.max_features = std::max(1, settings.max_features);
        alignment_label("Matcher");
        const bool supported_matcher = settings.feature_matcher == "gpu_mutual_ratio" ||
            settings.feature_matcher == "vulkan_mutual_ratio" || settings.feature_matcher == "mutual_ratio";
        const char* matchers[] = {"Brute force", "LightGlue", "Hybrid LightGlue"};
        const char* matcher_tokens[] = {"gpu_mutual_ratio", "lightglue", "hybrid_lightglue"};
        std::string matcher_algorithm = supported_matcher ? "gpu_mutual_ratio" : settings.feature_matcher;
        const auto previous_matcher = matcher_algorithm;
        alignment_method_combo("##feature_matcher", matcher_algorithm, matchers, matcher_tokens, 3, 1);
        if (matcher_algorithm != previous_matcher) settings.feature_matcher = matcher_algorithm;
        if (!supported_extractor || !supported_matcher) {
            ImGui::TextWrapped("%s", tr("This project uses an unavailable experimental method. Select a SIFT method to continue."));
            if (ImGui::Button(tr("Use SIFT defaults"), {-1.F, 0.F})) {
                settings.feature_extractor = "siftgpu";
                settings.feature_matcher = "gpu_mutual_ratio";
            }
        }
    }
    if (ImGui::CollapsingHeader(tr("Solver and cache"))) {
        alignment_label("Solver type");
        const char* solvers[] = {tr("Global"), tr("Hierarchical")};
        int solver = settings.sfm_mode == 2 ? 1 : 0;
        ImGui::SetNextItemWidth(-1.F);
        if (ImGui::Combo("##sfm_mode", &solver, solvers, 2)) settings.sfm_mode = solver == 1 ? 2 : 0;
        ImGui::Checkbox(tr("Refine weak cameras"), &settings.refine_weak_cameras);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tr("Spend extra time refining weak registered cameras. Keeps the original pose when independent checks do not improve."));
        ImGui::Checkbox(tr("Reuse cached alignment"), &settings.reuse_cache);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tr("Reuse compatible alignment stages from the project cache.\n"
                "Changed images, parameters, or algorithm versions invalidate affected stages.\n"
                "Off by default for new projects. Off recomputes alignment;\n"
                "working cameras remain available for training and preview."));
        draw_cache_folder_link(app);
    }
    ImGui::TreePop();
}
}  // namespace

Action draw_inspector(App& app) {
    Action action = Action::none;
    if (!app.show_inspector) return action;
    if (!ImGui::Begin(i18n::id(model_viewer_active(app) ? "Model properties" : "Reconstruction settings", "###Inspector"), &app.show_inspector)) {
        ImGui::End();
        return action;
    }
    if (model_viewer_active(app)) {
        ImGui::BeginChild("##model_properties", {0.F,0.F}, ImGuiChildFlags_AlwaysUseWindowPadding);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8.F,12.F});
        draw_model_properties(app);
        ImGui::PopStyleVar();
        ImGui::EndChild(); ImGui::End(); return action;
    }
    const bool busy = app.job.running();
    const InspectorSection section = current_inspector_section(app);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {16.F, 12.F});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {8.F, 10.F});
    // Keep job controls reachable even when a step has many advanced fields.
    const bool checkpoint = !busy && section == InspectorSection::training &&
        app.training_checkpoint_info.has_value();
    const float footer_height = busy ? 150.F : (checkpoint ? 230.F : 118.F);
    char parameters_id[64];
    std::snprintf(parameters_id, sizeof(parameters_id), "##inspector_parameters_%d",
        static_cast<int>(section));
    ImGui::BeginChild(parameters_id,
        {0.F, -std::min(footer_height, ImGui::GetContentRegionAvail().y * 0.55F)},
        ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PushID(static_cast<int>(section));
    theme::caption(section == InspectorSection::project ? "PROJECT" : "PIPELINE");
    ImGui::Spacing();
    ImGui::TextUnformatted(tr(inspector_title(section)));
    ImGui::PushTextWrapPos(0.F);
    ImGui::TextDisabled("%s", tr(inspector_description(section)));
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (section == InspectorSection::project) {
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
        ImGui::BeginDisabled(busy || app.pending_train_prepare.valid());
        theme::caption("Image source");
        ImGui::SetNextItemWidth(-138.F);
        auto image_source = app.settings.images_dir;
        if (ImGui::InputText("##images", image_source.data(), image_source.size(), ImGuiInputTextFlags_EnterReturnsTrue))
            change_image_source(app, path_from_utf8_field(image_source.data()));
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
                app.settings.sam_masks
                    ? "Align Photos extracts frames, generates masks, then aligns cameras."
                    : "Align Photos extracts the sharpest stills, then runs SfM.");
            ImGui::PopTextWrapPos();
            const auto ffmpeg = photara::io::locate_ffmpeg();
            if (ffmpeg.empty()) {
                theme::metric_coloured("ffmpeg", "Not found", theme::danger);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip(
                        "Install ffmpeg and add it to PATH. Photara also "
                        "looks next to the app and in common install folders.");
            } else {
                theme::metric_coloured("ffmpeg", "Ready", theme::success);
                if (ImGui::IsItemHovered()) {
                    const std::string located = path_to_utf8(ffmpeg);
                    ImGui::SetTooltip("%s", located.c_str());
                }
            }
            if (ImGui::TreeNodeEx(
                    tr("Video extraction settings"),
                    ImGuiTreeNodeFlags_SpanAvailWidth)) {
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
                ImGui::TreePop();
            }
            if (theme::toolbar_button(
                    tr("Open Video Window..."), {-1.F, 28.F}))
                open_video_window(app);
        }
#if defined(PHOTARA_HAS_SAM)
        ImGui::Spacing();
        theme::caption(tr("Model masks"));
        if (ImGui::Checkbox(tr("Generate model masks"), &app.settings.sam_masks)) {
            if (app.settings.sam_masks) {
                app.settings.use_mask = true;
            }
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tr(
                "The selected model generates text-prompted masks before alignment.\n"
                "Masks are used for 3DGS training. Post-alignment refinement is optional.\n"
                "Use the option below to enable masks for alignment.\n"
                "Choose a mask model and download its missing files below."));
        ImGui::BeginDisabled(!app.settings.sam_masks || has_external_dataset(app));
        ImGui::Checkbox(tr("Use masks for alignment"), &app.settings.sam_use_mask_for_sfm);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", tr("Off (recommended): align using the full image, including background features.\n"
                "On: ignore features outside the first-pass SAM3 masks.\n"
                "Useful for moving backgrounds; small or incomplete masks can reduce alignment quality.\n"
                "Applies to the next alignment. 3DGS training and mask refinement are unchanged."));
        ImGui::Checkbox(tr("Refine masks after alignment"), &app.settings.sam_sfm_guided);
        ImGui::TextWrapped("%s", tr("Off by default. Enabling reruns SAM after alignment using the sparse point cloud, overwrites masks, and increases runtime and memory / VRAM use."));
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", tr("Runs when this alignment generated new SAM masks. Existing masks reused from disk are not automatically refined."));
        ImGui::EndDisabled();
        if (app.settings.sam_masks) {
            draw_mask_model_controls(app);
            theme::caption(tr("Prompt"));
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputText(
                "##sam_text", app.settings.sam_text.data(),
                app.settings.sam_text.size());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr("Several phrases, separated by semicolons."));
            draw_sam_prompt_suggestions(app.settings.sam_text);
            theme::caption(tr("Negative prompt"));
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputText(
                "##sam_neg", app.settings.sam_negative_text.data(),
                app.settings.sam_negative_text.size());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr(
                    "Regions matching a negative phrase are removed from the mask."));
            ImGui::Checkbox(
                tr("Keep prompted subject"), &app.settings.sam_keep_prompted);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr(
                    "On: the prompt names the subject to keep.\n"
                    "Off: the prompt names distractors to remove."));
            ImGui::BeginDisabled(app.settings.sam_engine == 1);
            ImGui::Checkbox(tr("Track across frames"), &app.settings.sam_video);
            ImGui::EndDisabled();

        }
#else
        ImGui::Spacing();
        theme::caption(tr("Model masks"));
        ImGui::TextWrapped("%s", tr(
            "This build does not include SAM mask generation."));
#endif
        ImGui::Spacing();
        if (theme::toolbar_button(
                tr("Open Mask Editor..."), {-1.F, 28.F}))
            open_mask_editor(app);
        theme::caption("Project file");
        if (app.layout.project_file.empty())
            theme::caption(tr("Unsaved session. Press Ctrl+S to save a project."));
        ImGui::SetNextItemWidth(-30.F);
        ImGui::InputText(
                "##project", app.settings.project_dir.data(),
                app.settings.project_dir.size(), ImGuiInputTextFlags_ReadOnly);
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

        ImGui::BeginDisabled(busy || app.pending_train_prepare.valid());
        theme::caption("External SfM dataset (optional)");
        ImGui::SetNextItemWidth(-138.F);
        auto dataset_source = app.settings.dataset_source;
        if (ImGui::InputText("##dataset_source", dataset_source.data(), dataset_source.size(), ImGuiInputTextFlags_EnterReturnsTrue))
            change_external_dataset(app, path_from_utf8_field(dataset_source.data()));
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button(i18n::id("Folder", "##pick_dataset_folder"), {58.F, 0.F}) &&
            pick_folder(
                L"Select external SfM dataset folder",
                dataset_source)) {
            change_external_dataset(app, path_from_utf8_field(dataset_source.data()));
        }
        ImGui::SameLine(0.F, 4.F);
        if (ImGui::Button(i18n::id("File", "##pick_dataset_file"), {46.F, 0.F}) &&
            pick_dataset_file(
                L"Select external camera dataset file",
                dataset_source)) {
            change_external_dataset(app, path_from_utf8_field(dataset_source.data()));
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
            if (ImGui::Button(tr("Remove external dataset"), {-1.F, 26.F})) {
                change_external_dataset(app, {});
            }
        }
        ImGui::EndDisabled();
        ImGui::BeginDisabled(busy || app.pending_train_prepare.valid());
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

    if (section == InspectorSection::alignment) {
        ImGui::Spacing();
        ImGui::BeginDisabled(busy || app.pending_train_prepare.valid());
        draw_alignment_configuration(app);
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
                path_to_utf8(app.layout.sparse_asfm.filename());
            theme::metric_coloured(
                "SfM scene", asfm_name.c_str(), theme::success);
        }
        if (app.has_mvs) {
            ImGui::Spacing();
            const std::string mvs_name =
                path_to_utf8(app.layout.sparse_mvs.filename());
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
            const auto audit = app.scene.position_audit_counts();
            if (audit.unverified > 0) {
                theme::metric_coloured("Positions needing review",
                    std::to_string(audit.unverified).c_str(), theme::warning);
                ImGui::PushStyleColor(ImGuiCol_Text, theme::warning);
                ImGui::TextWrapped("%s", tr("Inspect flagged cameras before accepting alignment. Low reprojection error does not verify their positions."));
                ImGui::PopStyleColor();
            } else {
                theme::metric("Position audit", tr(audit.audited == 0 ? "Not audited" : "Graph supported"));
            }
            if (audit.unaudited > 0 && audit.audited > 0)
                theme::metric("Positions not audited", std::to_string(audit.unaudited).c_str());
            theme::metric(
                "Sparse points",
                format_count(app.scene.points.size()).c_str());
        }
        ImGui::Spacing();
    }

    if (section == InspectorSection::training) {
        ImGui::Spacing();
        draw_training_compute_link(app);
        ImGui::Spacing();
        ImGui::BeginDisabled(busy || app.pending_train_prepare.valid());
        if (ImGui::BeginTable("##training_properties", 2,
                ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthStretch, 0.42F);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 0.58F);
            inspector_field("Training mode");
            ImGui::SetNextItemWidth(-1.F);
            const char* modes[] = {tr("Fast"), tr("Standard"), tr("High quality")};
            const int mode = std::clamp(app.settings.training_mode, 0, 2);
            const auto preset = photara::splat::training_preset(
                static_cast<photara::splat::TrainingMode>(mode));
            const bool customized = app.settings.iterations != static_cast<int>(preset.iterations) ||
                app.settings.densification_cap != static_cast<int>(preset.gaussian_cap) ||
                app.settings.speedy_pruning != preset.speedy_pruning;
            if (ImGui::BeginCombo("##training_mode", customized ? tr("Custom") : modes[mode])) {
                for (int index = 0; index < 3; ++index) {
                    if (ImGui::Selectable(modes[index], !customized && mode == index)) {
                        const auto selected = photara::splat::training_preset(
                            static_cast<photara::splat::TrainingMode>(index));
                        photara::splat::apply_training_loss_preset(app.settings,
                            static_cast<photara::splat::TrainingMode>(index));
                        app.settings.normal_field = false;
                        app.settings.ppisp_layout = 0;
                        app.settings.bilateral_grid = false;
                        app.settings.training_mode = index;
                        app.settings.iterations = static_cast<int>(selected.iterations);
                        app.settings.densification_cap = static_cast<int>(selected.gaussian_cap);
                        app.settings.speedy_pruning = selected.speedy_pruning;
                    }
                }
                ImGui::EndCombo();
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s\n\n%s", tr("Fast: 30,000 steps, 1M Gaussians, no pruning.\n"
                    "Standard: 30,000 steps, 3M Gaussians, no pruning.\n"
                    "High quality: 50,000 steps, 5M Gaussians, no pruning."),
                    tr("Fast keeps RGB loss only. Standard and High quality default to scale 0.01 and needle suppression 0.001. Selecting a mode resets loss settings."));
            inspector_field("Capture type");
            ImGui::SetNextItemWidth(-1.F);
            const char* capture[] = {tr("Object"), tr("Scene")};
            int capture_index = app.settings.scene_mode ? 1 : 0;
            if (ImGui::Combo("##capture", &capture_index, capture, 2))
                app.settings.scene_mode = capture_index == 1;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr(
                    "Object captures a bounded subject; Scene keeps distant\n"
                    "background geometry."));
            inspector_field("Iterations");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputInt("##iterations", &app.settings.iterations, 1000, 5000);
            inspector_field("Densification cap");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputInt(
                "##densification_cap",
                &app.settings.densification_cap, 100'000, 1'000'000);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr(
                    "Maximum Gaussian count during densification.\n"
                    "Initialization uses the full source cloud.\n"
                    "Default 3,000,000."));
            inspector_field("Pruning");
            ImGui::Checkbox("##pruning", &app.settings.speedy_pruning);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr("Prune 30% during growth and 20% after growth.\n"
                    "At most two soft and two hard prunes; off by default in all modes.\n"
                    "30,000 steps: soft at 9,000/12,000; hard at 18,000/24,000.\n"
                    "Other run lengths scale with the growth and fine-tuning phases.\n"
                    "Requires full training resolution."));
            ImGui::EndTable();
        }
        if (ImGui::Checkbox(
                tr("Focus Region"), &app.settings.focus_region) &&
            app.settings.focus_region)
            app.view_options.show_region = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tr(
                "Constrain splat training to the reconstruction region.\n"
                "Works with both Object and Scene capture. Turning it on\n"
                "shows the editable box; hiding the box later only changes\n"
                "the display and never disables the training constraint."));
        if (ImGui::TreeNodeEx(tr("Advanced training settings"),
                ImGuiTreeNodeFlags_SpanAvailWidth)) {
            theme::caption("Densification strategy");
            ImGui::SetNextItemWidth(-1.F);
            const char* strategies[] = {
                "ADC IGS", "ADC Plus"};
            int strategy_index = app.settings.strategy == 1 ? 1 : 0;
            ImGui::Combo("##strategy", &strategy_index, strategies, 2);
            app.settings.strategy = strategy_index;
            theme::caption("SH degree");
            ImGui::SetNextItemWidth(-1.F);
            const char* sh_degrees[] = {"0", "1", "2", "3"};
            ImGui::Combo(
                "##sh_degree", &app.settings.sh_degree, sh_degrees, 4);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr(
                    "Spherical-harmonic colour bands.\n"
                    "0 is diffuse only. 3 is the training default."));
            theme::caption("Max training resolution");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputInt(
                "##resolution", &app.settings.max_resolution, 128, 512);
            theme::caption("Live preview cadence");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::InputInt("##cadence", &app.settings.preview_interval, 10, 50);
            ImGui::Checkbox(
                tr("Coarse-to-fine resolution"), &app.settings.progressive_resolution);
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("%s", tr(
                    "Starts training at a lower image resolution and increases it\n"
                    "in stages. This reduces early GPU memory and compute cost.\n"
                    "Turn it off to use the maximum resolution from the start."));
            ImGui::Checkbox(tr("Foreground mask training"), &app.settings.use_mask);
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("%s", tr(
                    "Uses each view's mask or source alpha during 3DGS training.\n"
                    "A mask is required for every training image.\n"
                    "Choose masked or transparent below."));
            if (app.settings.use_mask || app.settings.sam_masks) {
                theme::caption(tr("Mask mode"));
                ImGui::SetNextItemWidth(-1.F);
                const char* mask_modes[] = {
                    tr("masked"),
                    tr("transparent"),
                };
                int mask_mode = app.settings.mask_mode == 1 ? 1 : 0;
                if (ImGui::Combo("##mask_mode", &mask_mode, mask_modes, 2))
                    app.settings.mask_mode = mask_mode;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", tr(
                        "masked: those rays are missing observations.\n"
                        "Colour is trained only where the mask is valid, and alpha is left alone.\n"
                        "Use this when a moving person hides static geometry.\n"
                        "transparent: the mask is the target opacity.\n"
                        "Valid pixels stay opaque and masked pixels become transparent."));
            }
            draw_training_objective(app);
            ImGui::TreePop();
        }
        ImGui::EndDisabled();
        const bool simplify_open = app.splat_simplify.show;
        if (theme::toolbar_button(
                tr("Simplify"), {-1.F, 28.F},
                !busy && can_export_model(app), simplify_open))
            app.splat_simplify.show = !app.splat_simplify.show;
        if (app.splat_simplify.show && !app.splat_simplify.running &&
            !app.splat_simplify.has_baseline)
            app.splat_simplify.source_count = app.splat_edit.gaussian_count();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tr(
                "Reduce Gaussian count after training. Nearby splats are merged.\n"
                "Does not retrain. Undo restores the previous model."));
        if (app.splat_simplify.show && can_export_model(app)) {
            ImGui::Spacing();
            const bool simplifying = app.splat_simplify.running;
            const std::uint32_t source = app.splat_simplify.source_count
                ? app.splat_simplify.source_count
                : app.splat_edit.gaussian_count();
            ImGui::BeginDisabled(simplifying);
            theme::caption("Keep");
            float keep_percent = app.splat_simplify.keep_ratio * 100.F;
            ImGui::SetNextItemWidth(-1.F);
            if (ImGui::SliderFloat(
                    "##simplify_keep", &keep_percent, 10.F, 100.F, "%.0f%%"))
                app.splat_simplify.keep_ratio = keep_percent / 100.F;
            theme::caption("Opacity floor");
            ImGui::SetNextItemWidth(-1.F);
            ImGui::SliderFloat(
                "##simplify_opacity", &app.splat_simplify.min_opacity,
                0.F, 0.3F, "%.2f");
            ImGui::EndDisabled();
            const auto target = static_cast<std::uint32_t>(std::llround(
                static_cast<double>(source) *
                static_cast<double>(app.splat_simplify.keep_ratio)));
            char change[96];
            std::snprintf(
                change, sizeof(change), "%s  →  %s",
                format_count(source).c_str(), format_count(target).c_str());
            theme::metric("Gaussians", change);
            if (simplifying) {
                theme::progress_track(
                    {-1.F, 6.F}, app.splat_simplify.progress, theme::accent);
                if (theme::toolbar_button(tr("Cancel"), {-1.F, 28.F})) {
                    if (app.splat_simplify.cancel)
                        app.splat_simplify.cancel->store(true);
                }
            } else if (theme::primary_button(
                           tr("Apply"), {-1.F, 30.F}, source > 0 && !busy)) {
                start_splat_simplify(app);
            }
            if (!app.splat_simplify.error.empty()) {
                ImGui::PushTextWrapPos(0.F);
                ImGui::TextColored(
                    theme::danger, "%s", app.splat_simplify.error.c_str());
                ImGui::PopTextWrapPos();
            }
        }
        if (theme::toolbar_button(
                tr("Export Splat"), {-1.F, 28.F},
                !busy && can_export_model(app)))
            open_splat_export_panel(app);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tr(
                "Save the trained Gaussians. Choose PLY, SOG, SPZ, or GLB\n"
                "and the spherical-harmonics degree."));
        ImGui::Spacing();
    }

    if (section == InspectorSection::mesh) {
        ImGui::Spacing();
        ImGui::BeginDisabled(busy || app.pending_train_prepare.valid());
        ImGui::Checkbox(tr("Build mesh"), &app.settings.build_mesh);
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip(
                "Also extract a surface automatically at the end of 3DGS training.\n"
                "Uncheck it to train a splat-only model; Extract Mesh still works.");
        // The method and quality settings also configure the explicit
        // Extract Mesh action, so they stay available without Build Mesh.
        {
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
                    tr("From MVS"), app.settings.mesh_source == 1)) {
                app.settings.mesh_source = 1;
                if (app.settings.mesh_method == 3)
                    app.settings.mesh_method = 0;
            }
            ImGui::PushTextWrapPos(0.F);
            theme::caption(
                "PatchMatch stereo from aligned cameras, then fuse a mesh "
                "from the dense cloud. 3DGS stays appearance-only.");
            ImGui::PopTextWrapPos();
            ImGui::Spacing();
            theme::caption("Mesh algorithm");
            ImGui::SetNextItemWidth(-1.F);
            if (mesh_source_gaussians(app.settings)) {
                const char* methods[] = {
                    tr(app.settings.normal_field
                           ? "Auto: PAM" : "Auto: TSDF"),
                    "TSDF", tr("Delaunay"), "PAM"};
                ImGui::Combo(
                    "##mesh_method", &app.settings.mesh_method, methods, 4);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", tr(
                        "Auto uses PAM with Normal field enabled, otherwise TSDF.\n"
                        "Select an algorithm to override Auto.\n"
                        "PAM requires a model trained with Normal field."));
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
            ImGui::Spacing();
            theme::caption("Mesh quality");
            ImGui::SetNextItemWidth(-1.F);
            const char* mesh_qualities[] = {tr("Fast"), tr("Standard"), tr("High")};
            ImGui::Combo("##mesh_quality", &app.settings.mesh_quality, mesh_qualities, 3);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr(
                    "Fast prioritizes speed and may lose small details.\n"
                    "Standard balances detail and time. High uses more samples and refinement.\n"
                    "Mesh quality does not change 3DGS training settings."));
            if (mesh_source_gaussians(app.settings))
                objective_note("Geometry weights are in Gaussian Splatting > Loss & constraints > Geometry.");
        }
        ImGui::Spacing();
        if (ImGui::TreeNodeEx(tr("Mesh postprocessing"),
                ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth)) {
            ImGui::PushTextWrapPos(0.F);
            theme::caption("Applied during mesh extraction and texture baking. Manifold repair runs before UV unwrapping.");
            ImGui::PopTextWrapPos();
            theme::caption("Postprocessing quality");
            const char* qualities[] = {tr("Fast"), tr("Standard"), tr("High")};
            ImGui::SetNextItemWidth(-1.F);
            if (ImGui::Combo("##mesh_post_quality", &app.settings.mesh_postprocess_quality,
                    qualities, 3))
                apply_mesh_postprocess_preset(app.settings);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr("Fast: 500,000 faces, no remesh smoothing.\nStandard: 1,000,000 faces, 1 smoothing iteration.\nHigh: 2,000,000 faces, 2 smoothing iterations.\nPresets set budgets; enable simplification or remeshing separately."));
            ImGui::Checkbox(tr("Simplify mesh"), &app.settings.mesh_simplify);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr("Reduce faces only when the mesh exceeds the target. Keep useful disconnected components."));
            ImGui::BeginDisabled(!app.settings.mesh_simplify);
            theme::caption("Target triangle count");
            ImGui::SetNextItemWidth(-1.F);
            if (ImGui::InputScalar("##mesh_target", ImGuiDataType_U64, &app.settings.mesh_target_faces))
                app.settings.mesh_target_faces = std::max<std::uint64_t>(4, app.settings.mesh_target_faces);
            ImGui::EndDisabled();
            ImGui::Checkbox(tr("Remesh surface"), &app.settings.mesh_remesh);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr("Rebuild triangle layout with Instant Meshes. Changes geometry and can affect thin structures. Runs independently of simplification."));
            if (app.settings.mesh_remesh) {
                theme::caption("Remesh triangle budget");
                ImGui::SetNextItemWidth(-1.F);
                if (ImGui::InputScalar("##mesh_remesh_target", ImGuiDataType_U64,
                        &app.settings.mesh_remesh_target_faces))
                    app.settings.mesh_remesh_target_faces = std::max<std::uint64_t>(4, app.settings.mesh_remesh_target_faces);
                theme::caption("Remesh smoothing iterations");
                ImGui::SetNextItemWidth(-1.F);
                if (ImGui::InputInt("##mesh_remesh_smooth", &app.settings.mesh_remesh_smooth_iterations))
                    app.settings.mesh_remesh_smooth_iterations = std::clamp(app.settings.mesh_remesh_smooth_iterations, 0, 10);
            }
            ImGui::TreePop();
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

    if (section == InspectorSection::texture) {
        ImGui::Spacing();
        ImGui::PushTextWrapPos(0.F);
        theme::caption(
            "Last reconstruction step. Unwrap the mesh, project calibrated "
            "photos with photara_drender, then optionally refine the atlas.");
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
#if !defined(PHOTARA_HAS_TEXTURE)
        theme::caption(
            "This build was compiled without texture baking (Vulkan + photara_drender).");
#else
        ImGui::BeginDisabled(busy || app.pending_train_prepare.valid());
        if (ImGui::BeginTable("##texture_properties", 2,
                ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthStretch, 0.42F);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 0.58F);
            inspector_field("Quality");
            ImGui::SetNextItemWidth(-1.F);
            const char* texture_qualities[] = {tr("Fast"), tr("Standard"), tr("High")};
            if (ImGui::Combo(
                    "##tex_quality", &app.settings.texture_quality,
                    texture_qualities, 3))
                apply_texture_quality_preset(app.settings);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Fast: 1024 atlas, projection only.\n"
                    "Standard: 2048 atlas, 3000 photometric + 60 seam iterations.\n"
                    "High: 4096 atlas, 6000 photometric + 120 seam iterations.");
            inspector_field("Atlas size");
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
            inspector_field("Blend space");
            ImGui::SetNextItemWidth(-1.F);
            int blend_space = app.settings.texture_blend_linear ? 1 : 0;
            const char* blend_spaces[] = {tr("sRGB"), tr("Linear")};
            if (ImGui::Combo("##tex_blend", &blend_space, blend_spaces, 2))
                app.settings.texture_blend_linear = blend_space == 1;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "%s",
                    tr("sRGB blends like the photographs, so the textured mesh matches them.\n"
                       "Linear blends in scene-linear light, then encodes sRGB on export."));
            ImGui::EndTable();
        }
        ImGui::Checkbox(tr("Remove lighting (albedo)"), &app.settings.texture_delight);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Run Intrinsic delighter on the photos before projection.\n"
                "Needs ONNX Runtime and the Intrinsic stage_*.onnx models.");
        ImGui::Checkbox(tr("Refine atlas"), &app.settings.texture_optimize);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Photometric + seam optimization in photara_drender after the "
                "projective bake. Off is faster; on cleans view seams.");
        if (app.settings.texture_optimize) {
            theme::caption("Photometric iterations");
            ImGui::SetNextItemWidth(-1.F);
            if (ImGui::InputInt("##tex_photo_steps", &app.settings.texture_photometric_steps, 100, 1000))
                app.settings.texture_photometric_steps = std::max(1, app.settings.texture_photometric_steps);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr("Iterations to fit the atlas colors to the photographs."));
            theme::caption("Seam iterations");
            ImGui::SetNextItemWidth(-1.F);
            if (ImGui::InputInt("##tex_seam_steps", &app.settings.texture_seam_steps, 10, 100))
                app.settings.texture_seam_steps = std::max(0, app.settings.texture_seam_steps);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("%s", tr("Iterations to match colors across UV seams. Set to 0 to skip."));
        }
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
                        app.atlas_preview.upload(photara::io::load_rgb(albedo));
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
#endif
        ImGui::Spacing();
    }

    const TrainingStats& stats = app.monitor.training();
    if (section == InspectorSection::training && stats.valid &&
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
        theme::metric("Image & geometry loss", buffer);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", tr("Weighted image and geometry losses. Optimizer regularization is applied separately and is not included in this value."));
        std::snprintf(buffer, sizeof(buffer), "%.4f", stats.rgb_loss);
        theme::metric("RGB", buffer);
        if ((app.settings.use_mask || app.settings.sam_masks) && app.settings.mask_mode == 1) {
            std::snprintf(buffer, sizeof(buffer), "%.4f", stats.alpha_loss);
            theme::metric("Mask opacity", buffer);
        }
        if (mesh_from_gaussians(app.settings) || app.settings.geometry_training || app.settings.normal_field ||
            (app.settings.ai_priors && app.settings.training_mode != 0)) {
            std::snprintf(buffer, sizeof(buffer), "%.4f", stats.depth_loss);
            theme::metric("Depth", buffer);
            std::snprintf(buffer, sizeof(buffer), "%.4f", stats.normal_loss);
            theme::metric("Normal", buffer);
            std::snprintf(
                buffer, sizeof(buffer), "%.4f",
                stats.multi_view_geometry_loss);
            theme::metric("Multi-view depth", buffer);
            std::snprintf(
                buffer, sizeof(buffer), "%.4f", stats.multi_view_ncc_loss);
            theme::metric("Multi-view appearance", buffer);
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

    ImGui::PopID();
    ImGui::EndChild();
    ImGui::Separator();
    ImGui::BeginChild("##inspector_actions", {0.F, 0.F},
        ImGuiChildFlags_AlwaysUseWindowPadding);
    // Context-appropriate primary action pinned to the bottom of the panel.
    const auto hover_tip = [](const char* english) {
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", tr(english));
    };
    const char* const train_tip = mesh_from_gaussians(app.settings)
        ? "Train 3DGS with depth and normal losses, then extract a "
          "surface from the Gaussians."
        : "Train 3DGS from the aligned cameras using the selected losses and priors.";
    const char* const extract_tip = mesh_source_gaussians(app.settings)
        ? (app.has_model
               ? "Extract a surface from the trained 3DGS model.\n"
                 "Uses the saved Gaussians. Training is not started again."
               : "Extract a surface from the Gaussians.\n"
                 "Trains with depth and normal losses, then builds the selected surface.")
        : "Build a mesh from the MVS dense cloud.\n"
          "PatchMatch stereo from the aligned cameras, then fuse triangles.";
    const char* const bake_tip = app.has_texture
        ? "Unwrap UVs again and reproject the calibrated photos onto the current mesh."
        : "Unwrap UVs and project the calibrated photos onto the current mesh.";
    if (checkpoint) {
        const auto& info = *app.training_checkpoint_info;
        ImGui::Text("%s: %u / %u", tr("Saved Training Progress"), info.iteration, info.total_iterations);
        ImGui::TextWrapped("%s", tr("Continue from the saved PLY model. Optimizer state will be reinitialized."));
        if (theme::primary_button("Resume Training##checkpoint", {-1.F, 40.F},
                !app.pending_train_prepare.valid())) {
            action = Action::continue_training;
        }
    }
    if (busy && app.saving_training) {
        ImGui::TextWrapped("%s", tr("Saving project and training progress. Please wait."));
        if (ImGui::Button(tr("Cancel"), {-1.F, 36.F})) cancel_project_save(app);
    } else if (busy) {
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
    } else {
        const bool aligned = alignment_ready(app);
        const char* label = "Align Photos";
        const char* tip = "Choose an image folder or video in Project settings.";
        Action next = Action::align;
        bool enabled = app.settings.images_dir[0] != '\0' && !has_external_dataset(app);
        switch (section) {
            case InspectorSection::training:
                label = mesh_from_gaussians(app.settings)
                    ? "Train 3DGS + Mesh" : "Train 3DGS";
                tip = train_tip;
                next = Action::train;
                enabled = aligned;
                break;
            case InspectorSection::mesh:
                label = "Extract Mesh";
                tip = extract_tip;
                next = Action::dense;
                enabled = aligned;
                break;
            case InspectorSection::texture:
                label = app.has_texture ? "Re-bake Texture" : "Bake Texture";
                tip = bake_tip;
                next = Action::texture;
                enabled = app.has_mesh;
#if !defined(PHOTARA_HAS_TEXTURE)
                enabled = false;
#endif
                break;
            default: break;
        }
        // Preparation runs asynchronously before a ProcessJob exists.
        enabled = enabled && !app.pending_train_prepare.valid();
        ImGui::PushTextWrapPos(0.F);
        theme::caption(app.pending_train_prepare.valid() ? "Preparing training..."
            : (section == InspectorSection::texture && !app.has_mesh)
                ? "Extract a mesh before baking textures."
            : ((section == InspectorSection::training || section == InspectorSection::mesh) && !aligned)
                ? "Align photos or import a camera dataset first."
            : (has_external_dataset(app) && (section == InspectorSection::project || section == InspectorSection::alignment))
                ? "Imported cameras are ready. Select training or mesh extraction."
            : enabled ? "Ready" : "Choose an image folder or video in Project settings.");
        ImGui::PopTextWrapPos();
        const bool pressed = checkpoint
            ? theme::toolbar_button(label, {-1.F, 32.F}, enabled)
            : theme::primary_button(label, {-1.F, 40.F}, enabled);
        if (pressed) action = next;
        hover_tip(tip);
    }
    ImGui::EndChild();
    ImGui::PopStyleVar(2);

    ImGui::End();
    return action;
}

}  // namespace editor
