#include "console_view.hpp"

#include "icons.hpp"
#include "theme.hpp"

#include "imgui.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string_view>

namespace editor {
namespace {

constexpr float k_toolbar_height = 42.F;
constexpr float k_filter_height = 40.F;
constexpr float k_header_height = k_toolbar_height + k_filter_height;
constexpr float k_gutter = 4.F;
constexpr float k_time_width = 108.F;
constexpr float k_level_width = 52.F;
constexpr float k_row_pad = 16.F;

bool contains_ci(const std::string_view hay, const std::string_view needle) {
    if (needle.empty()) return true;
    if (needle.size() > hay.size()) return false;
    const auto fold = [](const unsigned char value) {
        return static_cast<char>(std::tolower(value));
    };
    return std::search(
               hay.begin(), hay.end(), needle.begin(), needle.end(),
               [&](const char a, const char b) {
                   return fold(static_cast<unsigned char>(a)) ==
                          fold(static_cast<unsigned char>(b));
               }) != hay.end();
}

int find_ci(const std::string_view hay, const std::string_view needle) {
    if (needle.empty() || needle.size() > hay.size()) return -1;
    const auto fold = [](const unsigned char value) {
        return static_cast<char>(std::tolower(value));
    };
    const auto found = std::search(
        hay.begin(), hay.end(), needle.begin(), needle.end(),
        [&](const char a, const char b) {
            return fold(static_cast<unsigned char>(a)) ==
                   fold(static_cast<unsigned char>(b));
        });
    return found == hay.end()
        ? -1
        : static_cast<int>(found - hay.begin());
}

bool matches_filter(const ConsoleSeverity severity, const ConsoleFilter filter) {
    switch (filter) {
        case ConsoleFilter::warning:
            return severity == ConsoleSeverity::warning;
        case ConsoleFilter::error:
            return severity == ConsoleSeverity::error;
        case ConsoleFilter::info:
            return severity != ConsoleSeverity::warning &&
                   severity != ConsoleSeverity::error;
        case ConsoleFilter::all:
            return true;
    }
    return true;
}

const char* level_tag(const ConsoleSeverity severity) {
    switch (severity) {
        case ConsoleSeverity::error: return "ERR";
        case ConsoleSeverity::warning: return "WRN";
        case ConsoleSeverity::info: return "INF";
        case ConsoleSeverity::debug: return "DBG";
        case ConsoleSeverity::other: return "OUT";
    }
    return "OUT";
}

ImVec4 level_colour(const ConsoleSeverity severity) {
    switch (severity) {
        case ConsoleSeverity::error: return theme::danger;
        case ConsoleSeverity::warning: return theme::warning;
        case ConsoleSeverity::info: return theme::accent;
        case ConsoleSeverity::debug: return theme::text_faint;
        case ConsoleSeverity::other: return theme::text_muted;
    }
    return theme::text_muted;
}

ImVec4 message_colour(const ConsoleSeverity severity) {
    switch (severity) {
        case ConsoleSeverity::error: return theme::danger;
        case ConsoleSeverity::warning: return ImVec4(0.93F, 0.82F, 0.58F, 1.F);
        case ConsoleSeverity::debug: return theme::text_faint;
        default: return ImVec4(0.78F, 0.80F, 0.84F, 1.F);
    }
}

std::string format_line(const ConsoleLine& line) {
    std::string text;
    if (!line.time.empty()) {
        text += line.time;
        text += "  ";
    }
    text += level_tag(line.severity);
    text += "  ";
    text += line.message;
    return text;
}

void copy_text(const std::string& text) {
    ImGui::SetClipboardText(text.c_str());
}

void rebuild_visible(ConsoleView& view, const LogStream& log) {
    const std::string query = view.search.data();
    if (view.cached_generation == log.generation() &&
        view.cached_filter == view.filter && view.cached_query == query)
        return;

    view.visible.clear();
    const std::vector<ConsoleLine>& lines = log.lines();
    view.visible.reserve(lines.size());
    for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
        const ConsoleLine& line = lines[static_cast<std::size_t>(i)];
        if (!matches_filter(line.severity, view.filter)) continue;
        if (!query.empty() && !contains_ci(line.message, query) &&
            !contains_ci(line.time, query) &&
            !contains_ci(level_tag(line.severity), query))
            continue;
        view.visible.push_back(i);
    }
    view.cached_generation = log.generation();
    view.cached_filter = view.filter;
    view.cached_query = query;
    if (view.selected >= static_cast<int>(lines.size())) view.selected = -1;
}

bool filter_chip(
    const char* id, const char* label, const int count, const bool active,
    const ImVec4& colour) {
    ImFont* font = theme::small_font();
    ImGui::PushFont(font);
    char count_text[16];
    std::snprintf(count_text, sizeof(count_text), "%d", count);
    const ImVec2 label_size = ImGui::CalcTextSize(label);
    const ImVec2 count_size = ImGui::CalcTextSize(count_text);
    const ImVec2 size{label_size.x + count_size.x + 28.F, 28.F};
    const bool pressed = ImGui::InvisibleButton(id, size);
    const bool hovered = ImGui::IsItemHovered();
    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    ImDrawList* draw = ImGui::GetWindowDrawList();

    const ImVec4 fill = active ? theme::fade(colour, 0.18F)
                               : (hovered ? theme::surface_3 : theme::surface_1);
    draw->AddRectFilled(min, max, theme::u32(fill), 6.F);
    if (active)
        draw->AddRect(min, max, theme::u32(colour, 0.50F), 6.F);
    else if (hovered)
        draw->AddRect(min, max, theme::u32(theme::border, 0.8F), 6.F);

    const float text_y = min.y + (size.y - font->FontSize) * 0.5F;
    draw->AddText(
        font, font->FontSize, {min.x + 11.F, text_y},
        theme::u32(active ? colour : theme::text_muted), label);
    draw->AddText(
        font, font->FontSize, {min.x + 11.F + label_size.x + 8.F, text_y},
        theme::u32(active || count > 0 ? colour : theme::text_faint),
        count_text);
    ImGui::PopFont();
    return pressed;
}

void draw_status_badge(const bool running, const Stage stage) {
    ImVec4 colour = theme::inactive;
    const char* label = "IDLE";
    if (running) {
        colour = theme::accent;
        label = "LIVE";
        const float pulse =
            0.55F + 0.45F * (0.5F + 0.5F * std::sinf(
                static_cast<float>(ImGui::GetTime()) * 3.4F));
        colour = theme::fade(theme::accent, pulse);
    } else if (stage == Stage::failed) {
        colour = theme::danger;
        label = "FAILED";
    } else if (stage == Stage::complete) {
        colour = theme::success;
        label = "DONE";
    }

    theme::status_dot(colour, 8.F);
    ImGui::SameLine(0.F, 10.F);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2.F);
    ImGui::PushFont(theme::small_font());
    ImGui::PushStyleColor(
        ImGuiCol_Text, running ? theme::accent
                               : (stage == Stage::failed
                                      ? theme::danger
                                      : (stage == Stage::complete ? theme::success
                                                                  : theme::text_faint)));
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void draw_unseen_badge(const int count) {
    if (count <= 0) return;
    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    char text[8];
    if (count > 99)
        std::snprintf(text, sizeof(text), "99+");
    else
        std::snprintf(text, sizeof(text), "%d", count);
    ImFont* font = theme::small_font();
    const ImVec2 size = font->CalcTextSizeA(10.F, FLT_MAX, 0.F, text);
    const ImVec2 centre{max.x - 1.F, min.y + 2.F};
    const float radius = std::max(7.F, size.x * 0.5F + 4.F);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddCircleFilled(centre, radius, theme::u32(theme::accent));
    draw->AddText(
        font, 10.F, {centre.x - size.x * 0.5F, centre.y - size.y * 0.5F},
        theme::u32(theme::text_bright), text);
}

void draw_empty_state(
    ImDrawList* draw, const ImVec2 min, const ImVec2 max, const char* title,
    const char* hint) {
    ImFont* font = ImGui::GetFont();
    const float title_w =
        font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.F, title).x;
    const float hint_w =
        font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.F, hint).x;
    const float centre_x = (min.x + max.x) * 0.5F;
    const float centre_y = (min.y + max.y) * 0.5F;
    draw->AddText(
        {centre_x - title_w * 0.5F, centre_y - 12.F},
        theme::u32(theme::text_muted), title);
    draw->AddText(
        {centre_x - hint_w * 0.5F, centre_y + 8.F},
        theme::u32(theme::text_faint), hint);
}

void draw_message(
    ImDrawList* draw, ImFont* font, const float size, const ImVec2 pos,
    const ImU32 colour, const std::string_view text, const std::string_view query,
    const float clip_max_x, const float row_bottom) {
    draw->PushClipRect(pos, {clip_max_x, row_bottom}, true);
    draw->AddText(
        font, size, pos, colour, text.data(), text.data() + text.size());
    const int match = find_ci(text, query);
    if (match >= 0 && !query.empty()) {
        const float prefix_w = font->CalcTextSizeA(
            size, FLT_MAX, 0.F, text.data(), text.data() + match).x;
        const float match_w = font->CalcTextSizeA(
            size, FLT_MAX, 0.F, text.data() + match,
            text.data() + match + query.size()).x;
        const ImVec2 hi_min{pos.x + prefix_w, pos.y - 1.F};
        const ImVec2 hi_max{pos.x + prefix_w + match_w, pos.y + size + 1.F};
        draw->AddRectFilled(hi_min, hi_max, theme::u32(theme::accent, 0.32F), 2.F);
        draw->AddText(
            font, size, {pos.x + prefix_w, pos.y}, theme::u32(theme::text_bright),
            text.data() + match, text.data() + match + query.size());
    }
    draw->PopClipRect();
}

void draw_log_row(
    const ConsoleLine& line, const int line_index, const bool selected,
    const float width, const float row_height, const std::string_view query,
    ConsoleView& view) {
    ImGui::PushID(line_index);
    ImGui::InvisibleButton("##row", {width, row_height});
    const bool hovered = ImGui::IsItemHovered();
    const bool pressed = ImGui::IsItemClicked(ImGuiMouseButton_Left);
    if (pressed) view.selected = line_index;
    if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
        copy_text(format_line(line));

    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec4 level = level_colour(line.severity);

    ImU32 background = 0;
    if (selected)
        background = theme::u32(theme::accent, 0.16F);
    else if (hovered)
        background = theme::u32(theme::surface_3, 0.90F);
    else if (line.severity == ConsoleSeverity::error)
        background = theme::u32(theme::danger, 0.07F);
    else if (line.severity == ConsoleSeverity::warning)
        background = theme::u32(theme::warning, 0.05F);
    else if ((line_index & 1) == 1)
        background = theme::u32(theme::surface_1, 0.55F);
    if (background != 0) draw->AddRectFilled(min, max, background);

    draw->AddRectFilled(
        min, {min.x + k_gutter, max.y}, theme::u32(level, selected ? 1.F : 0.85F));

    ImFont* mono = theme::mono_font();
    ImFont* small = theme::small_font();
    const float text_y = min.y + (row_height - mono->FontSize) * 0.5F;
    float x = min.x + k_row_pad;

    if (!line.time.empty()) {
        draw->AddText(
            mono, mono->FontSize, {x, text_y}, theme::u32(theme::text_faint),
            line.time.c_str());
        x += k_time_width;
    }

    const char* tag = level_tag(line.severity);
    const ImVec2 tag_size =
        small->CalcTextSizeA(small->FontSize, FLT_MAX, 0.F, tag);
    const ImVec2 badge_min{x, min.y + (row_height - tag_size.y - 8.F) * 0.5F};
    const ImVec2 badge_max{
        badge_min.x + tag_size.x + 12.F, badge_min.y + tag_size.y + 8.F};
    draw->AddRectFilled(badge_min, badge_max, theme::u32(level, 0.16F), 4.F);
    draw->AddText(
        small, small->FontSize,
        {badge_min.x + 6.F, badge_min.y + 4.F}, theme::u32(level), tag);
    x += k_level_width + 12.F;

    draw_message(
        draw, mono, mono->FontSize, {x, text_y}, theme::u32(message_colour(line.severity)),
        line.message, query, max.x - k_row_pad, max.y);

    const float message_width = mono->CalcTextSizeA(
        mono->FontSize, FLT_MAX, 0.F, line.message.c_str()).x;
    if (hovered && message_width > max.x - x - k_row_pad)
        ImGui::SetTooltip("%s", line.message.c_str());

    ImGui::PopID();
}

}  // namespace

void draw_console(
    bool& open, ConsoleView& view, LogStream& log, const bool running,
    const JobKind job, const Stage stage) {
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar;
    if (running) flags |= ImGuiWindowFlags_UnsavedDocument;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.F, 0.F));
    const bool visible = ImGui::Begin("Console", &open, flags);
    ImGui::PopStyleVar();
    if (!visible) {
        ImGui::End();
        return;
    }

    rebuild_visible(view, log);
    const ConsoleCounts& counts = log.counts();
    const std::vector<ConsoleLine>& lines = log.lines();
    const ImGuiIO& io = ImGui::GetIO();
    const bool window_focused =
        ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);

    if (window_focused && io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_F) &&
        !io.WantTextInput)
        view.request_search_focus = true;

    const ImVec2 header_origin = ImGui::GetCursorScreenPos();
    const ImVec2 content_start = ImGui::GetCursorPos();
    const float header_width = ImGui::GetContentRegionAvail().x;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(
        header_origin,
        {header_origin.x + header_width, header_origin.y + k_toolbar_height},
        theme::u32(theme::surface_3));
    draw->AddRectFilled(
        {header_origin.x, header_origin.y + k_toolbar_height},
        {header_origin.x + header_width, header_origin.y + k_header_height},
        theme::u32(theme::surface_2));
    draw->AddLine(
        {header_origin.x, header_origin.y + k_toolbar_height},
        {header_origin.x + header_width, header_origin.y + k_toolbar_height},
        theme::u32(theme::border, 0.85F));
    draw->AddLine(
        {header_origin.x, header_origin.y + k_header_height - 1.F},
        {header_origin.x + header_width, header_origin.y + k_header_height - 1.F},
        theme::u32(theme::border));

    constexpr float k_btn = 30.F;
    constexpr float k_btn_gap = 8.F;
    constexpr float k_side_pad = 14.F;
    const float toolbar_y =
        content_start.y + (k_toolbar_height - k_btn) * 0.5F;

    ImGui::SetCursorPos({content_start.x + k_side_pad, toolbar_y + 4.F});
    draw_status_badge(running, stage);
    if (running) {
        ImGui::SameLine(0.F, 12.F);
        ImGui::SetCursorPosY(toolbar_y + 7.F);
        ImGui::PushFont(theme::small_font());
        ImGui::PushStyleColor(ImGuiCol_Text, theme::text_faint);
        ImGui::TextUnformatted(job_name(job));
        ImGui::PopStyleColor();
        ImGui::PopFont();
    }

    const float actions_width = k_btn * 3.F + k_btn_gap * 2.F;
    ImGui::SetCursorPos({
        content_start.x + header_width - k_side_pad - actions_width, toolbar_y});
    const int unseen = view.at_bottom
        ? 0
        : std::max(
              0, static_cast<int>(lines.size()) -
                     static_cast<int>(view.seen_count));
    if (icons::ghost_button(
            "##follow", icons::Icon::follow, {k_btn, k_btn},
            view.follow && view.at_bottom, true,
            view.follow ? "Follow output" : "Jump to latest")) {
        view.follow = true;
        view.jump_to_end = true;
    }
    draw_unseen_badge(unseen);
    ImGui::SameLine(0.F, k_btn_gap);
    if (icons::ghost_button(
            "##copy", icons::Icon::copy, {k_btn, k_btn}, false,
            !view.visible.empty(), "Copy visible lines")) {
        std::string text;
        for (const int index : view.visible) {
            text += format_line(lines[static_cast<std::size_t>(index)]);
            text += '\n';
        }
        copy_text(text);
    }
    ImGui::SameLine(0.F, k_btn_gap);
    if (icons::ghost_button(
            "##clear", icons::Icon::trash, {k_btn, k_btn}, false,
            counts.total > 0, "Clear console")) {
        log.clear_display();
        view.selected = -1;
        view.seen_count = 0;
        view.visible.clear();
        view.cached_generation = log.generation();
    }

    const float filter_y =
        content_start.y + k_toolbar_height + (k_filter_height - 28.F) * 0.5F;
    ImGui::SetCursorPos({content_start.x + k_side_pad, filter_y});
    if (filter_chip(
            "##all", "All", counts.total, view.filter == ConsoleFilter::all,
            theme::accent))
        view.filter = ConsoleFilter::all;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show every log line");
    ImGui::SameLine(0.F, 8.F);
    if (filter_chip(
            "##info", "Info", counts.info, view.filter == ConsoleFilter::info,
            ImVec4(0.55F, 0.72F, 0.86F, 1.F)))
        view.filter = view.filter == ConsoleFilter::info ? ConsoleFilter::all
                                                         : ConsoleFilter::info;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Info and debug output");
    ImGui::SameLine(0.F, 8.F);
    if (filter_chip(
            "##warn", "Warn", counts.warning,
            view.filter == ConsoleFilter::warning, theme::warning))
        view.filter = view.filter == ConsoleFilter::warning
            ? ConsoleFilter::all
            : ConsoleFilter::warning;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Warnings only");
    ImGui::SameLine(0.F, 8.F);
    if (filter_chip(
            "##err", "Error", counts.error, view.filter == ConsoleFilter::error,
            theme::danger))
        view.filter = view.filter == ConsoleFilter::error ? ConsoleFilter::all
                                                          : ConsoleFilter::error;
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Errors only");

    const float chips_end =
        ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x;
    const float search_right = header_width - k_side_pad;
    const float search_width =
        std::clamp(search_right - chips_end - 20.F, 0.F, 280.F);
    if (search_width >= 120.F) {
        ImGui::SetCursorPos({
            content_start.x + search_right - search_width, filter_y});
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(30.F, 6.F));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.F);
        ImGui::PushFont(theme::small_font());
        ImGui::PushItemWidth(search_width);
        if (view.request_search_focus) {
            ImGui::SetKeyboardFocusHere();
            view.request_search_focus = false;
        }
        ImGui::InputTextWithHint(
            "##console_search", "Filter logs", view.search.data(),
            view.search.size());
        ImGui::PopItemWidth();
        ImGui::PopFont();
        ImGui::PopStyleVar(2);
        const ImVec2 field_min = ImGui::GetItemRectMin();
        const ImVec2 field_max = ImGui::GetItemRectMax();
        icons::draw(
            draw, icons::Icon::search,
            {field_min.x + 9.F, field_min.y + 6.F},
            {field_min.x + 23.F, field_max.y - 6.F},
            theme::u32(theme::text_faint), 1.8F);
        if (view.search[0] != '\0' && ImGui::IsItemFocused() &&
            ImGui::IsKeyPressed(ImGuiKey_Escape))
            view.search[0] = '\0';
    }

    ImGui::SetCursorPos({content_start.x, content_start.y + k_header_height});
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::console_bg);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.F, 0.F));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.F, 0.F));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.F);
    ImGui::BeginChild("##log", {0.F, 0.F}, false);

    const ImVec2 log_min = ImGui::GetCursorScreenPos();
    const ImVec2 log_region = ImGui::GetContentRegionAvail();
    const ImVec2 log_max{log_min.x + log_region.x, log_min.y + log_region.y};
    const float row_height = std::floor(theme::mono_font()->FontSize + 17.F);
    const std::string_view query = view.search.data();

    if (lines.empty()) {
        draw_empty_state(
            ImGui::GetWindowDrawList(), log_min, log_max,
            running ? "Listening for process output..."
                    : "No reconstruction output yet",
            running ? "Live logs from the active job stream here"
                    : "Run Align Photos or Train 3DGS to stream logs");
        ImGui::Dummy(log_region);
    } else if (view.visible.empty()) {
        draw_empty_state(
            ImGui::GetWindowDrawList(), log_min, log_max,
            "No matching log lines",
            "Clear the filter or search to see all output");
        ImGui::Dummy(log_region);
    } else {
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(view.visible.size()), row_height);
        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                const int line_index = view.visible[static_cast<std::size_t>(i)];
                draw_log_row(
                    lines[static_cast<std::size_t>(line_index)], line_index,
                    view.selected == line_index, log_region.x, row_height, query,
                    view);
            }
        }
    }

    if (ImGui::BeginPopupContextWindow("##console_ctx")) {
        if (view.selected >= 0 &&
            view.selected < static_cast<int>(lines.size()) &&
            ImGui::MenuItem("Copy Line"))
            copy_text(format_line(lines[static_cast<std::size_t>(view.selected)]));
        if (ImGui::MenuItem("Copy Visible", nullptr, false, !view.visible.empty())) {
            std::string text;
            for (const int index : view.visible) {
                text += format_line(lines[static_cast<std::size_t>(index)]);
                text += '\n';
            }
            copy_text(text);
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Clear Console", nullptr, false, counts.total > 0)) {
            log.clear_display();
            view.selected = -1;
            view.seen_count = 0;
        }
        ImGui::EndPopup();
    }

    if (window_focused && !io.WantTextInput && !view.visible.empty()) {
        int visible_at = -1;
        for (int i = 0; i < static_cast<int>(view.visible.size()); ++i)
            if (view.visible[static_cast<std::size_t>(i)] == view.selected) {
                visible_at = i;
                break;
            }
        int next = visible_at;
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow))
            next = std::max(0, (visible_at < 0 ? static_cast<int>(view.visible.size()) : visible_at) - 1);
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow))
            next = std::min(
                static_cast<int>(view.visible.size()) - 1,
                visible_at < 0 ? 0 : visible_at + 1);
        if (next != visible_at && next >= 0) {
            view.selected = view.visible[static_cast<std::size_t>(next)];
            ImGui::SetScrollY(static_cast<float>(next) * row_height);
            view.follow = false;
        }
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_C) && view.selected >= 0 &&
            view.selected < static_cast<int>(lines.size()))
            copy_text(format_line(lines[static_cast<std::size_t>(view.selected)]));
    }

    const bool at_bottom =
        ImGui::GetScrollMaxY() <= 1.F ||
        ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.F;
    if (ImGui::IsWindowHovered()) {
        if (io.MouseWheel > 0.F)
            view.follow = false;
        else if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && !at_bottom)
            view.follow = false;
        if (io.MouseWheel < 0.F && at_bottom) view.follow = true;
    }
    if (view.jump_to_end || (view.follow && at_bottom))
        ImGui::SetScrollHereY(1.F);
    view.jump_to_end = false;
    view.at_bottom =
        ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.F ||
        ImGui::GetScrollMaxY() <= 1.F;
    if (view.at_bottom) view.seen_count = lines.size();

    ImGui::EndChild();
    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor();
    ImGui::End();
}

}  // namespace editor
