#include "theme.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <filesystem>
#include <string>

namespace editor::theme {
namespace {

const char* pick_ui_font() {
    static const char* const candidates[] = {
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    };
    for (const char* candidate : candidates)
        if (std::filesystem::exists(candidate)) return candidate;
    return nullptr;
}

const char* pick_mono_font() {
    static const char* const candidates[] = {
        "C:\\Windows\\Fonts\\CascadiaMono.ttf",
        "C:\\Windows\\Fonts\\cascadiamono.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
        "C:\\Windows\\Fonts\\lucon.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    };
    for (const char* candidate : candidates)
        if (std::filesystem::exists(candidate)) return candidate;
    return nullptr;
}

constexpr float header_height = 32.F;

ImFont* g_small_font{};
ImFont* g_mono_font{};

ImFont* load_ui_font(ImGuiIO& io, const char* path, const float size) {
    ImFontConfig config;
    // msyh.ttc is a collection; FontNo 0 is the regular face.
    config.FontNo = 0;
    config.PixelSnapH = true;
    return io.Fonts->AddFontFromFileTTF(
        path, size, &config, io.Fonts->GetGlyphRangesChineseFull());
}

ImFont* load_mono_font(ImGuiIO& io, const char* path, const float size) {
    ImFontConfig config;
    config.PixelSnapH = true;
    config.OversampleH = 2;
    return io.Fonts->AddFontFromFileTTF(
        path, size, &config, io.Fonts->GetGlyphRangesDefault());
}

}  // namespace

ImU32 u32(const ImVec4& colour, const float alpha_scale) {
    ImVec4 scaled = colour;
    scaled.w *= alpha_scale;
    return ImGui::ColorConvertFloat4ToU32(scaled);
}

ImVec4 fade(const ImVec4& colour, const float alpha) {
    return ImVec4(colour.x, colour.y, colour.z, alpha);
}

Fonts load_fonts(ImGuiIO& io) {
    Fonts fonts;
    const char* path = pick_ui_font();
    fonts.regular = path ? load_ui_font(io, path, 15.F) : nullptr;
    if (!fonts.regular) {
        fonts.regular = io.Fonts->AddFontDefault();
        fonts.small = fonts.regular;
    } else {
        fonts.small = load_ui_font(io, path, 12.F);
        if (!fonts.small) fonts.small = fonts.regular;
    }
    g_small_font = fonts.small;
    if (const char* mono = pick_mono_font())
        fonts.mono = load_mono_font(io, mono, 13.F);
    if (!fonts.mono) fonts.mono = fonts.small;
    g_mono_font = fonts.mono;
    return fonts;
}

ImFont* small_font() {
    return g_small_font ? g_small_font : ImGui::GetFont();
}

ImFont* mono_font() {
    return g_mono_font ? g_mono_font : ImGui::GetFont();
}

void apply_style() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding = {10.F, 8.F};
    style.FramePadding = {9, 6};
    style.CellPadding = {8, 5};
    style.ItemSpacing = {8, 7};
    style.ItemInnerSpacing = {6, 5};
    style.IndentSpacing = 18.F;
    style.ScrollbarSize = 11.F;
    style.GrabMinSize = 9.F;
    style.WindowBorderSize = 1.F;
    style.DockingSeparatorSize = 3.F;
    style.ChildBorderSize = 1.F;
    style.PopupBorderSize = 1.F;
    style.FrameBorderSize = 1.F;
    style.WindowRounding = 0.F;
    style.ChildRounding = 5.F;
    style.FrameRounding = 4.F;
    style.PopupRounding = 5.F;
    style.ScrollbarRounding = 8.F;
    style.GrabRounding = 4.F;
    style.TabRounding = 4.F;
    style.SeparatorTextBorderSize = 1.F;

    auto& c = style.Colors;
    c[ImGuiCol_Text] = text_bright;
    c[ImGuiCol_TextDisabled] = text_faint;
    c[ImGuiCol_WindowBg] = surface_0;
    c[ImGuiCol_ChildBg] = surface_1;
    c[ImGuiCol_PopupBg] = ImVec4(0.106F, 0.110F, 0.122F, 0.98F);
    c[ImGuiCol_Border] = border;
    c[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_FrameBg] = surface_2;
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.153F, 0.157F, 0.176F, 1.F);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.184F, 0.188F, 0.212F, 1.F);
    c[ImGuiCol_TitleBg] = surface_2;
    c[ImGuiCol_TitleBgActive] = surface_3;
    c[ImGuiCol_MenuBarBg] = surface_2;
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ScrollbarGrab] = ImVec4(0.220F, 0.227F, 0.251F, 1.F);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.290F, 0.298F, 0.325F, 1.F);
    c[ImGuiCol_ScrollbarGrabActive] = accent_deep;
    c[ImGuiCol_CheckMark] = accent;
    c[ImGuiCol_SliderGrab] = accent_deep;
    c[ImGuiCol_SliderGrabActive] = accent;
    c[ImGuiCol_Button] = ImVec4(0.129F, 0.133F, 0.149F, 1.F);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.192F, 0.200F, 0.224F, 1.F);
    c[ImGuiCol_ButtonActive] = accent_deep;
    c[ImGuiCol_Header] = ImVec4(0.145F, 0.149F, 0.169F, 1.F);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.184F, 0.192F, 0.216F, 1.F);
    c[ImGuiCol_HeaderActive] = ImVec4(0.208F, 0.216F, 0.243F, 1.F);
    c[ImGuiCol_Separator] = border;
    c[ImGuiCol_SeparatorHovered] = accent_deep;
    c[ImGuiCol_SeparatorActive] = accent;
    c[ImGuiCol_ResizeGrip] = ImVec4(0, 0, 0, 0);
    c[ImGuiCol_ResizeGripHovered] = fade(accent, 0.5F);
    c[ImGuiCol_ResizeGripActive] = accent;
    c[ImGuiCol_Tab] = ImVec4(0.086F, 0.090F, 0.102F, 1.F);
    c[ImGuiCol_TabHovered] = ImVec4(0.165F, 0.171F, 0.192F, 1.F);
    c[ImGuiCol_TabSelected] = surface_3;
    c[ImGuiCol_TabSelectedOverline] = accent;
    c[ImGuiCol_TabDimmed] = ImVec4(0.071F, 0.073F, 0.082F, 1.F);
    c[ImGuiCol_TabDimmedSelected] = ImVec4(0.125F, 0.129F, 0.145F, 1.F);
    c[ImGuiCol_DockingPreview] = fade(accent, 0.45F);
    c[ImGuiCol_DockingEmptyBg] = surface_0;
    c[ImGuiCol_PlotHistogram] = accent;
    c[ImGuiCol_TextSelectedBg] = fade(accent, 0.35F);
}

void section_header(const char* title, const char* trailing) {
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 max{origin.x + width, origin.y + header_height};
    draw->AddRectFilled(origin, max, u32(surface_3));
    draw->AddLine({origin.x, max.y - 1.F}, {max.x, max.y - 1.F}, u32(border));

    draw->AddRectFilled(
        {origin.x, origin.y + 9.F}, {origin.x + 2.F, max.y - 9.F},
        u32(accent, 0.85F));

    ImFont* font = g_small_font ? g_small_font : ImGui::GetFont();
    const float size = font->FontSize;
    const float text_y = origin.y + (header_height - size) * 0.5F;
    draw->AddText(font, size, {origin.x + 12.F, text_y}, u32(text_muted), title);
    if (trailing) {
        const float trailing_width =
            font->CalcTextSizeA(size, FLT_MAX, 0.F, trailing).x;
        draw->AddText(
            font, size, {max.x - trailing_width - 12.F, text_y},
            u32(text_faint), trailing);
    }
    ImGui::Dummy({width, header_height});
}

void begin_panel(
    const char* id, const ImVec2 size, const char* title, const char* trailing,
    const ImVec2 body_padding) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::BeginChild(id, size, true, ImGuiWindowFlags_NoScrollbar);
    ImGui::PopStyleVar();
    if (title) section_header(title, trailing);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, body_padding);
    ImGui::BeginChild("##body", ImVec2(0, 0), false);
}

void end_panel() {
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImGui::EndChild();
}

void status_dot(const ImVec4& colour, const float diameter) {
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float radius = diameter * 0.5F;
    const float line = ImGui::GetTextLineHeight();
    const ImVec2 centre{origin.x + radius, origin.y + line * 0.5F};
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddCircleFilled(centre, radius + 2.5F, u32(colour, 0.18F));
    draw->AddCircleFilled(centre, radius, u32(colour));
    ImGui::Dummy({diameter, line});
}

void pill(const char* text, const ImVec4& text_colour, const ImVec4& fill) {
    const ImVec2 text_size = ImGui::CalcTextSize(text);
    const ImVec2 padding{8.F, 3.F};
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 max{
        origin.x + text_size.x + padding.x * 2.F,
        origin.y + text_size.y + padding.y * 2.F};
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(origin, max, u32(fill), 4.F);
    draw->AddText({origin.x + padding.x, origin.y + padding.y}, u32(text_colour), text);
    ImGui::Dummy({max.x - origin.x, max.y - origin.y});
}

void progress_track(const ImVec2 size, const float fraction, const ImVec4& fill) {
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = size.x > 0.F ? size.x
                                     : std::max(1.F, ImGui::GetContentRegionAvail().x + size.x);
    const float height = std::max(3.F, size.y);
    const ImVec2 max{origin.x + width, origin.y + height};
    const float rounding = height * 0.5F;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(origin, max, u32(surface_2), rounding);

    if (fraction < 0.F) {
        // Indeterminate: a short highlight sweeping the full track.
        const float span = std::max(48.F, width * 0.28F);
        const float period = 1.8F;
        const float phase = std::fmod(
            static_cast<float>(ImGui::GetTime()), period) / period;
        const float head = -span + phase * (width + span);
        const float begin = origin.x + std::max(0.F, head);
        const float end = origin.x + std::min(width, head + span);
        if (end > begin) {
            draw->PushClipRect(origin, max, true);
            draw->AddRectFilledMultiColor(
                {begin, origin.y}, {end, max.y}, u32(fill, 0.05F),
                u32(fill, 0.95F), u32(fill, 0.95F), u32(fill, 0.05F));
            draw->PopClipRect();
        }
    } else if (fraction > 0.F) {
        const float filled = std::max(height, width * std::min(fraction, 1.F));
        draw->AddRectFilled(
            origin, {origin.x + filled, max.y}, u32(fill), rounding);
        // A brighter leading edge reads as motion without animating.
        draw->AddRectFilled(
            {origin.x + filled - rounding * 2.F, origin.y},
            {origin.x + filled, max.y}, u32(fill, 0.45F), rounding);
    }
    ImGui::Dummy({width, height});
}

namespace {

bool styled_button(
    const char* label, const ImVec2 size, const bool enabled,
    const ImVec4& base, const ImVec4& hovered, const ImVec4& text_colour) {
    ImGui::PushStyleColor(ImGuiCol_Button, base);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hovered);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, hovered);
    ImGui::PushStyleColor(ImGuiCol_Text, text_colour);
    if (!enabled) ImGui::BeginDisabled();
    const bool pressed = ImGui::Button(label, size);
    if (!enabled) ImGui::EndDisabled();
    ImGui::PopStyleColor(4);
    return pressed && enabled;
}

}  // namespace

bool primary_button(const char* label, const ImVec2 size, const bool enabled) {
    return styled_button(
        label, size, enabled, accent_deep, accent_hover, text_bright);
}

bool toolbar_button(
    const char* label, const ImVec2 size, const bool enabled, const bool active) {
    const ImVec4 base = active ? ImVec4(0.176F, 0.204F, 0.243F, 1.F)
                               : ImVec4(0.129F, 0.133F, 0.149F, 1.F);
    const ImVec4 hovered = ImVec4(0.196F, 0.204F, 0.229F, 1.F);
    const bool pressed = styled_button(
        label, size, enabled, base, hovered,
        active ? accent : text_bright);
    if (active) {
        const ImVec2 min = ImGui::GetItemRectMin();
        const ImVec2 max = ImGui::GetItemRectMax();
        ImGui::GetWindowDrawList()->AddRect(
            min, max, u32(accent, 0.45F), ImGui::GetStyle().FrameRounding);
    }
    return pressed;
}

bool danger_button(const char* label, const ImVec2 size, const bool enabled) {
    return styled_button(
        label, size, enabled, ImVec4(0.235F, 0.100F, 0.106F, 1.F),
        ImVec4(0.313F, 0.129F, 0.137F, 1.F), danger);
}

bool choice_tile(
    const char* id, const char* title, const char* subtitle, const bool selected,
    const ImVec2 size) {
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton(id, size);
    const bool pressed = ImGui::IsItemClicked();
    const bool hovered = ImGui::IsItemHovered();
    const ImVec2 max{origin.x + size.x, origin.y + size.y};
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec4 fill = selected ? fade(accent, 0.16F)
        : (hovered ? surface_3 : surface_2);
    const ImVec4 outline = selected ? accent
        : (hovered ? ImVec4(0.28F, 0.29F, 0.33F, 1.F) : border);
    draw->AddRectFilled(origin, max, u32(fill), 6.F);
    draw->AddRect(origin, max, u32(outline, selected ? 0.9F : 1.F), 6.F);

    ImFont* title_font = ImGui::GetFont();
    ImFont* sub_font = small_font();
    const float title_size = title_font->FontSize;
    const float sub_size = sub_font->FontSize;
    const ImVec2 title_extent =
        title_font->CalcTextSizeA(title_size, FLT_MAX, 0.F, title);
    const ImVec2 sub_extent =
        sub_font->CalcTextSizeA(sub_size, FLT_MAX, 0.F, subtitle);
    const float stack = title_extent.y + 4.F + sub_extent.y;
    const float text_y = origin.y + (size.y - stack) * 0.5F;
    draw->AddText(
        title_font, title_size,
        {origin.x + (size.x - title_extent.x) * 0.5F, text_y},
        u32(selected ? accent : text_bright), title);
    draw->AddText(
        sub_font, sub_size,
        {origin.x + (size.x - sub_extent.x) * 0.5F, text_y + title_extent.y + 4.F},
        u32(text_faint), subtitle);
    return pressed;
}

void caption(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, text_faint);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

void metric(const char* key, const char* value) {
    metric_coloured(key, value, text_bright);
}

void metric_coloured(
    const char* key, const char* value, const ImVec4& colour) {
    caption(key);
    const float value_width = ImGui::CalcTextSize(value).x;
    const float right = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
    ImGui::SameLine(std::max(ImGui::GetCursorPosX(), right - value_width));
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::TextUnformatted(value);
    ImGui::PopStyleColor();
}

}  // namespace editor::theme
