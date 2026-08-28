#pragma once

#include "imgui.h"

namespace editor::icons {

enum class Icon {
    select,
    translate,
    rotate,
    scale,
    cube,
    world,
    snap,
    grid,
    camera,
    frame,
    folder,
    align,
    train,
    output,
    stop,
    points,
    gpu,
    copy,
    trash,
    search,
    follow,
};

enum class ButtonStyle { normal, primary, danger };

void draw(
    ImDrawList* draw_list, Icon icon, ImVec2 min, ImVec2 max, ImU32 colour,
    float thickness = 0.F);

bool button(
    const char* id, Icon icon, ImVec2 size, bool active = false,
    bool enabled = true, const char* tooltip = nullptr);

bool labeled_button(
    const char* id, Icon icon, const char* label, ImVec2 size,
    ButtonStyle style = ButtonStyle::normal, bool enabled = true,
    bool active = false, const char* tooltip = nullptr);

// Quiet toolbar control: no fill until hover, extra padding around the glyph.
bool ghost_button(
    const char* id, Icon icon, ImVec2 size, bool active = false,
    bool enabled = true, const char* tooltip = nullptr);

// Small non-interactive icon that participates in the current ImGui layout.
void inline_icon(Icon icon, ImU32 colour, float size = 15.F);

}  // namespace editor::icons
