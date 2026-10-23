#pragma once

#include "i18n.hpp"
#include "theme.hpp"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <iterator>
#include <string>
#include <string_view>

namespace editor {
namespace detail {

inline std::string normalize_prompt_phrase(const std::string_view phrase) {
    std::string normalized;
    normalized.reserve(phrase.size());
    for (const unsigned char value : phrase)
        if (!std::isspace(value))
            normalized.push_back(static_cast<char>(std::tolower(value)));
    return normalized;
}

inline bool append_prompt_phrase(
    std::array<char, 512>& prompt, const std::string_view phrase) {
    std::string current(prompt.data());
    const std::string wanted = normalize_prompt_phrase(phrase);
    std::size_t start = 0;
    while (start <= current.size()) {
        const std::size_t end = current.find(';', start);
        const std::string existing = current.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        if (normalize_prompt_phrase(existing) == wanted) return true;
        if (end == std::string::npos) break;
        start = end + 1;
    }

    const std::string separator = current.empty() ? "" : "; ";
    if (current.size() + separator.size() + phrase.size() >= prompt.size())
        return false;
    current += separator;
    current += phrase;
    std::memcpy(prompt.data(), current.c_str(), current.size() + 1);
    return true;
}

}  // namespace detail

inline void draw_sam_prompt_suggestions(std::array<char, 512>& prompt) {
    struct Suggestion {
        const char* label;
        const char* prompt;
    };
    static constexpr Suggestion suggestions[] = {
        {"pedestrian", "pedestrian"},
        {"person", "person"},
        {"hand", "hand"},
        {"tourist", "tourist"},
        {"visitor", "visitor"},
    };

    ImGui::TextDisabled("%s", i18n::tr("Quick prompts"));
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", i18n::tr(
            "Click a suggestion to add its English noun phrase to the prompt."));
    const float line_width = ImGui::GetContentRegionAvail().x;
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    float used_width = 0.F;
    for (std::size_t i = 0; i < std::size(suggestions); ++i) {
        const auto& suggestion = suggestions[i];
        const char* label = suggestion.label;
        const float button_width = ImGui::CalcTextSize(label).x +
            ImGui::GetStyle().FramePadding.x * 2.F;
        if (i > 0) {
            if (used_width + spacing + button_width > line_width) {
                used_width = 0.F;
            } else {
                ImGui::SameLine();
                used_width += spacing;
            }
        }
        const std::string id = std::string(label) + "##sam_prompt_" +
            std::to_string(i);
        if (theme::toolbar_button(id.c_str(), {0.F, 0.F}))
            detail::append_prompt_phrase(prompt, suggestion.prompt);
        used_width += button_width;
    }
}

}  // namespace editor
