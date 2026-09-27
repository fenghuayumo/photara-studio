#include "ui.hpp"

#include "i18n.hpp"

#include "imgui.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdio>
#include <string>

namespace editor {
namespace {

using i18n::tr;

void preference_heading(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, theme::text_bright);
    ImGui::TextUnformatted(tr(text));
    ImGui::PopStyleColor();
}

void preferences_footer() {
    ImGui::Dummy({0.F, 8.F});
    theme::caption(
        "Saved on this computer. Projects keep their own training parameters.");
}

bool compute_card(
    const char* id, const char* title, const char* subtitle,
    const bool selected, const bool enabled, const ImVec2 size) {
    ImGui::BeginDisabled(!enabled);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const bool pressed = ImGui::InvisibleButton(id, size);
    const bool hovered = ImGui::IsItemHovered();
    ImGui::EndDisabled();

    const ImVec2 max{origin.x + size.x, origin.y + size.y};
    const ImVec4 fill = selected ? theme::fade(theme::accent, enabled ? 0.16F : 0.10F)
        : !enabled ? theme::surface_2
                   : (hovered ? theme::surface_3 : theme::surface_2);
    const ImVec4 outline = selected ? theme::accent : theme::border;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(origin, max, theme::u32(fill), 6.F);
    draw->AddRect(
        origin, max, theme::u32(outline, selected ? 0.9F : 1.F), 6.F);

    ImFont* title_font = ImGui::GetFont();
    ImFont* sub_font = theme::small_font();
    const ImVec4 title_colour = selected ? theme::accent
        : !enabled ? theme::text_faint
                   : theme::text_bright;
    const float text_x = origin.x + 14.F;
    const float title_y = origin.y + 16.F;
    draw->AddText(
        title_font, title_font->FontSize, {text_x, title_y},
        theme::u32(title_colour), tr(title));
    draw->AddText(
        sub_font, sub_font->FontSize,
        {text_x, title_y + title_font->FontSize + 6.F},
        theme::u32(theme::text_faint), tr(subtitle));
    return pressed && enabled;
}

bool preference_nav(const char* english, const bool selected) {
    const char* label = tr(english);
    ImGui::PushID(english);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    constexpr float height = 36.F;
    const bool pressed = ImGui::InvisibleButton("##row", {width, height});
    const bool hovered = ImGui::IsItemHovered();
    ImGui::PopID();

    const ImVec2 max{origin.x + width, origin.y + height};
    ImDrawList* draw = ImGui::GetWindowDrawList();
    if (selected || hovered) {
        draw->AddRectFilled(
            origin, max,
            theme::u32(selected ? theme::surface_2 : theme::surface_1));
    }
    if (selected) {
        draw->AddRectFilled(
            {origin.x, origin.y + 8.F}, {origin.x + 2.F, max.y - 8.F},
            theme::u32(theme::accent));
    }
    ImFont* font = ImGui::GetFont();
    const float text_y = origin.y + (height - font->FontSize) * 0.5F;
    const ImVec4 colour = selected || hovered ? theme::text_bright
                                              : theme::text_muted;
    draw->AddText(
        font, font->FontSize, {origin.x + 18.F, text_y}, theme::u32(colour),
        label);
    return pressed;
}

bool choice_row(
    const char* group, const char* const* labels, const int count, int& value,
    const bool* enabled) {
    ImGui::PushID(group);
    const float gap = 6.F;
    const float width =
        (ImGui::GetContentRegionAvail().x - gap * static_cast<float>(count - 1)) /
        static_cast<float>(count);
    bool changed = false;
    for (int index = 0; index < count; ++index) {
        if (index > 0) ImGui::SameLine(0.F, gap);
        ImGui::PushID(index);
        if (theme::toolbar_button(
                labels[index], {width, 30.F}, enabled[index], value == index) &&
            value != index) {
            value = index;
            changed = true;
        }
        ImGui::PopID();
    }
    ImGui::PopID();
    return changed;
}

void draw_compute_page(App& app) {
    const bool busy = app.job.running();
    preference_heading("Training compute");
    theme::caption("The next Train 3DGS job uses this device.");
    theme::caption("The viewport preview stays on Vulkan.");
    if (busy)
        theme::caption("The running job keeps the backend it started with.");
    ImGui::Spacing();

    const bool vulkan_ok = studio_vulkan_training();
    const float gap = 8.F;
    const float card_w = (ImGui::GetContentRegionAvail().x - gap) * 0.5F;
    const ImVec2 card_size{card_w, 74.F};
    if (compute_card(
            "##train_cuda", "CUDA", "NVIDIA default",
            app.settings.training_backend != 1, !busy, card_size) &&
        app.settings.training_backend != 0) {
        app.settings.training_backend = 0;
        store_editor_preferences(app);
    }
    ImGui::SameLine(0.F, gap);
    if (compute_card(
            "##train_vulkan", "Vulkan",
            vulkan_ok ? "Shared trainer" : "Not in this build",
            app.settings.training_backend == 1, !busy && vulkan_ok,
            card_size) &&
        app.settings.training_backend != 1) {
        app.settings.training_backend = 1;
        store_editor_preferences(app);
    }

    ImGui::Spacing();
    ImGui::Spacing();
    preference_heading("Photo alignment");
    const char* alignment_labels[] = {"Automatic", "CPU", "CUDA"};
    const bool alignment_enabled[] = {
        !busy, !busy, !busy && studio_ba_cuda()};
    if (choice_row(
            "alignment", alignment_labels, 3, app.settings.alignment_backend,
            alignment_enabled))
        store_editor_preferences(app);
    theme::caption(
        "Bundle adjustment. Automatic uses CUDA when this computer has it.");

    ImGui::Spacing();
    preference_heading("Subject masks");
    const bool sam = studio_has_sam();
    const char* sam_labels[] = {"Automatic", "CUDA", "Vulkan"};
    const bool sam_enabled[] = {
        !busy && sam,
        !busy && sam && studio_sam_cuda(),
        !busy && sam && studio_sam_vulkan(),
    };
    if (choice_row(
            "sam", sam_labels, 3, app.settings.sam_backend, sam_enabled))
        store_editor_preferences(app);
    theme::caption(
        sam ? "SAM 3. Prompts and whether masks are generated stay with the project."
            : "SAM is not in this build.");
    preferences_footer();
}

void draw_storage_page(App& app) {
    const bool busy = app.job.running();
    preference_heading("Cache folder");
    ImGui::PushTextWrapPos(0.F);
    theme::caption(
        "Leave empty to keep the working copies next to the project\n"
        "(<project folder>/<project name>.cache). Set a folder to\n"
        "collect every dataset's cache on another drive.");
    ImGui::PopTextWrapPos();
    if (busy)
        theme::caption("Stop the running job before changing the cache folder.");
    ImGui::Spacing();

    ImGui::BeginDisabled(busy);
    const float browse_w = ImGui::CalcTextSize(tr("Browse")).x + 28.F;
    ImGui::SetNextItemWidth(std::max(40.F, ImGui::GetContentRegionAvail().x - browse_w - 8.F));
    if (ImGui::InputText(
            "##cache_dir", app.settings.cache_dir.data(),
            app.settings.cache_dir.size())) {
        clear_loaded_result(app);
        refresh_artifacts(app);
        app.cache_usage.valid = false;
    }
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        store_editor_cache_dir(app);
        app.cache_usage.valid = false;
    }
    ImGui::SameLine(0.F, 8.F);
    if (theme::toolbar_button("Browse", {browse_w, 0.F}))
        select_cache_folder(app);
    ImGui::EndDisabled();

    if (!app.layout.working_sfm.empty())
        theme::caption(path_to_utf8(app.layout.working_sfm.parent_path()).c_str());

    ImGui::Spacing();
    ImGui::Spacing();
    preference_heading("Cache cleanup");
    if (!app.cache_usage.valid)
        app.cache_usage = scan_cache_usage(
            app.settings, app.layout.working_sfm.parent_path());
    char summary[192];
    std::snprintf(
        summary, sizeof(summary), "%s  \xC2\xB7  %zu %s",
        format_cache_bytes(app.cache_usage.bytes).c_str(),
        app.cache_usage.unused_folders, tr("unused"));
    theme::caption(summary);
    ImGui::BeginDisabled(busy);
    if (theme::toolbar_button("Clean unused caches", {-1.F, 30.F})) {
        std::uintmax_t freed = 0;
        const std::size_t removed = clean_unused_caches(
            app.settings, app.layout.working_sfm.parent_path(),
            std::chrono::minutes(30), &freed);
        app.cache_usage = scan_cache_usage(
            app.settings, app.layout.working_sfm.parent_path());
        if (removed == 0) {
            set_message(app, "No unused cache folders to clean", theme::text_muted);
        } else {
            char text[200];
            std::snprintf(
                text, sizeof(text),
                tr("Cleaned %zu cache folders, freed %s"), removed,
                format_cache_bytes(freed).c_str());
            set_message(app, text, theme::success);
        }
    }
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip(
            "%s",
            tr("Removes the working copies of datasets that are not open\n"
               "here. Unsaved results live in those folders: save the\n"
               "project first if you want to keep them."));
    bool auto_clean = app.settings.cache_retention_days > 0;
    if (ImGui::Checkbox(tr("Auto-clean unused caches"), &auto_clean)) {
        app.settings.cache_retention_days = auto_clean ? 30 : 0;
        store_editor_cache_dir(app);
    }
    if (auto_clean) {
        ImGui::SetNextItemWidth(88.F);
        if (ImGui::InputInt(
                "##cache_retention_days", &app.settings.cache_retention_days, 1, 7)) {
            app.settings.cache_retention_days = std::clamp(
                app.settings.cache_retention_days, 1, 3650);
            store_editor_cache_dir(app);
        }
        ImGui::SameLine(0.F, 8.F);
        theme::caption(tr("days unused"));
    }
    preferences_footer();
}

void apply_language(const i18n::Language language) {
    i18n::set_language(language);
    if (GLFWwindow* window = glfwGetCurrentContext())
        glfwSetWindowTitle(window, tr("Photara Studio"));
}

void draw_language_page() {
    preference_heading("Interface language");
    ImGui::Spacing();
    const char* names[] = {
        i18n::native_name(i18n::Language::en),
        i18n::native_name(i18n::Language::zh_cn),
        i18n::native_name(i18n::Language::ja),
        i18n::native_name(i18n::Language::ko),
    };
    int current = static_cast<int>(i18n::language());
    ImGui::SetNextItemWidth(-1.F);
    if (ImGui::Combo("##language", &current, names, 4))
        apply_language(static_cast<i18n::Language>(current));
    preferences_footer();
}

}  // namespace

void open_preferences(App& app, const int page) {
    app.show_preferences = true;
    if (page >= 0 && page <= 2) app.preferences_page = page;
    app.preferences_raise = true;
}

void draw_training_compute_link(App& app) {
    const char* device =
        app.settings.training_backend == 1 ? "Vulkan" : "CUDA";
    const char* key = tr("Training");
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = std::max(1.F, ImGui::GetContentRegionAvail().x);
    const float height = ImGui::GetFrameHeight();
    ImGui::InvisibleButton("##training_compute_link", {width, height});
    const bool hovered = ImGui::IsItemHovered();
    const bool clicked = ImGui::IsItemClicked();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    if (hovered) {
        draw->AddRectFilled(
            origin, {origin.x + width, origin.y + height},
            theme::u32(theme::surface_3), 4.F);
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::SetTooltip(
            "%s", tr("Training compute. Click to open Preferences."));
    }
    ImFont* font = ImGui::GetFont();
    const float text_y = origin.y + (height - font->FontSize) * 0.5F;
    draw->AddText(
        font, font->FontSize, {origin.x + 2.F, text_y},
        theme::u32(theme::text_faint), key);
    const float value_width = ImGui::CalcTextSize(device).x;
    draw->AddText(
        font, font->FontSize,
        {origin.x + width - value_width - 2.F, text_y},
        theme::u32(hovered ? theme::accent : theme::text_bright), device);
    if (clicked) open_preferences(app, 0);
}

void draw_cache_folder_link(App& app) {
    const bool custom = app.settings.cache_dir[0] != '\0';
    std::string folder;
    if (custom) {
        folder = path_to_utf8(
            path_from_utf8_field(app.settings.cache_dir.data()).filename());
        if (folder.empty()) folder = app.settings.cache_dir.data();
    }
    const char* value = custom ? folder.c_str() : tr("Beside the project");
    const char* key = tr("Cache folder");
    std::string tip;
    if (!app.layout.working_sfm.empty())
        tip = path_to_utf8(app.layout.working_sfm.parent_path());
    else if (custom)
        tip = app.settings.cache_dir.data();

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = std::max(1.F, ImGui::GetContentRegionAvail().x);
    const float height = ImGui::GetFrameHeight();
    ImGui::InvisibleButton("##cache_folder_link", {width, height});
    const bool hovered = ImGui::IsItemHovered();
    const bool clicked = ImGui::IsItemClicked();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    if (hovered) {
        draw->AddRectFilled(
            origin, {origin.x + width, origin.y + height},
            theme::u32(theme::surface_3), 4.F);
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }
    ImFont* font = ImGui::GetFont();
    const float text_y = origin.y + (height - font->FontSize) * 0.5F;
    const float key_w = font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.F, key).x;
    draw->AddText(
        font, font->FontSize, {origin.x + 2.F, text_y},
        theme::u32(theme::text_faint), key);
    const float value_max = std::max(24.F, width - key_w - 16.F);
    std::string shown = value;
    if (font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.F, shown.c_str()).x > value_max) {
        while (shown.size() > 1 &&
               font->CalcTextSizeA(
                   font->FontSize, FLT_MAX, 0.F, ("\xE2\x80\xA6" + shown).c_str()).x >
                   value_max) {
            const auto next = shown.find_first_of("\\/", 1);
            if (next == std::string::npos) {
                shown.erase(shown.begin());
                while (!shown.empty() &&
                       (static_cast<unsigned char>(shown.front()) & 0xC0) == 0x80)
                    shown.erase(shown.begin());
            } else {
                shown.erase(0, next);
            }
        }
        shown.insert(0, "\xE2\x80\xA6");
    }
    const float value_w =
        font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.F, shown.c_str()).x;
    draw->AddText(
        font, font->FontSize,
        {origin.x + width - value_w - 2.F, text_y},
        theme::u32(hovered ? theme::accent : theme::text_bright), shown.c_str());
    if (hovered) {
        const std::string hover = tip.empty()
            ? std::string(tr("Cache folder. Click to open Preferences."))
            : tip + "\n" + tr("Cache folder. Click to open Preferences.");
        ImGui::SetTooltip("%s", hover.c_str());
    }
    if (clicked) open_preferences(app, 2);
}

void draw_preferences_window(App& app) {
    if (!app.show_preferences) return;
    if (app.preferences_page < 0 || app.preferences_page > 2)
        app.preferences_page = 0;

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        viewport->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));
    ImGui::SetNextWindowSize({680.F, 480.F}, ImGuiCond_Appearing);
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(640.F, 420.F), ImVec2(960.F, 760.F));
    if (app.preferences_raise) {
        ImGui::SetNextWindowFocus();
        app.preferences_raise = false;
    }

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.F, 0.F));
    const bool visible = ImGui::Begin(
        i18n::id("Preferences", "###Preferences"), &app.show_preferences,
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse);
    ImGui::PopStyleVar();
    if (!visible) {
        ImGui::End();
        return;
    }

    constexpr float nav_w = 168.F;
    const ImVec2 seam_top = ImGui::GetCursorScreenPos();
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::surface_0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.F, 0.F));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.F);
    ImGui::BeginChild(
        "##prefs_nav", {nav_w, 0.F}, ImGuiChildFlags_None,
        ImGuiWindowFlags_NoScrollbar);
    ImGui::Dummy({0.F, 10.F});
    if (preference_nav("Compute", app.preferences_page == 0))
        app.preferences_page = 0;
    if (preference_nav("Language", app.preferences_page == 1))
        app.preferences_page = 1;
    if (preference_nav("Storage", app.preferences_page == 2))
        app.preferences_page = 2;
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();

    ImGui::SameLine(0.F, 0.F);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::surface_1);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(22.F, 18.F));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.F);
    ImGui::BeginChild(
        "##prefs_page", {0.F, 0.F},
        ImGuiChildFlags_AlwaysUseWindowPadding);
    if (app.preferences_page == 1) draw_language_page();
    else if (app.preferences_page == 2) draw_storage_page(app);
    else draw_compute_page(app);
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();

    const ImVec2 window_pos = ImGui::GetWindowPos();
    const ImVec2 window_size = ImGui::GetWindowSize();
    ImGui::GetWindowDrawList()->AddLine(
        {window_pos.x + nav_w, seam_top.y},
        {window_pos.x + nav_w, window_pos.y + window_size.y},
        theme::u32(theme::border));
    ImGui::End();
}

}  // namespace editor
