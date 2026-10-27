#pragma once

#include "imgui.h"

namespace editor::icons {

enum class Icon {
    eye,
    eye_off,
    model_mesh,
    model_gaussian,
    select,
    translate,
    rotate,
    scale,
    cube,
    world,
    snap,
    grid,
    camera,
    render_image,
    render_video,
    frame,
    folder,
    align,
    train,
    output,
    stop,
    pause,
    play,
    points,
    splat,
    rings,
    gpu,
    copy,
    trash,
    search,
    follow,
    clock,
    view2d,
    compare,
    heatmap,
    features,
    add,
    more,
    previous_keyframe,
    next_keyframe,
    chevron_left,
    chevron_right,
    frustum,
    photo,
    undo,
    redo,
    target,
    marquee,
    circle_select,
    polygon_select,
    brush,
    depth_front,
    depth_through,
    box_select,
    sphere_select,
    edit,
    sliders,
    lasso,
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
