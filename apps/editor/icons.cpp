#include "icons.hpp"

#include "theme.hpp"

#include <algorithm>
#include <cmath>

namespace editor::icons {
namespace {

ImVec2 point(const ImVec2 min, const ImVec2 max, const float x, const float y) {
    return {min.x + (max.x - min.x) * x, min.y + (max.y - min.y) * y};
}

void line(
    ImDrawList* draw_list, const ImVec2 min, const ImVec2 max, const float ax,
    const float ay, const float bx, const float by, const ImU32 colour,
    const float thickness) {
    draw_list->AddLine(
        point(min, max, ax, ay), point(min, max, bx, by), colour, thickness);
}

void arrow_head(
    ImDrawList* draw_list, const ImVec2 tip, const ImVec2 a, const ImVec2 b,
    const ImU32 colour, const float thickness) {
    draw_list->AddLine(tip, a, colour, thickness);
    draw_list->AddLine(tip, b, colour, thickness);
}

struct ButtonColours {
    ImVec4 base;
    ImVec4 hovered;
    ImVec4 active;
    ImVec4 icon;
};

ButtonColours button_colours(
    const ButtonStyle style, const bool selected, const bool enabled) {
    if (!enabled)
        return {
            theme::surface_2, theme::surface_2, theme::surface_2,
            theme::text_faint};
    if (style == ButtonStyle::primary)
        return {
            theme::accent_deep, theme::accent_hover, theme::accent_hover,
            theme::text_bright};
    if (style == ButtonStyle::danger)
        return {
            ImVec4(0.235F, 0.100F, 0.106F, 1.F),
            ImVec4(0.313F, 0.129F, 0.137F, 1.F),
            ImVec4(0.36F, 0.14F, 0.15F, 1.F), theme::danger};
    return {
        selected ? ImVec4(0.176F, 0.204F, 0.243F, 1.F)
                 : ImVec4(0.129F, 0.133F, 0.149F, 1.F),
        ImVec4(0.196F, 0.204F, 0.229F, 1.F), theme::accent_deep,
        selected ? theme::accent : theme::text_bright};
}

bool begin_icon_button(
    const char* id, const ImVec2 size, const ButtonColours& colours,
    const bool enabled) {
    ImGui::PushStyleColor(ImGuiCol_Button, colours.base);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colours.hovered);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, colours.active);
    if (!enabled) ImGui::BeginDisabled();
    const bool pressed = ImGui::Button(id, size);
    if (!enabled) ImGui::EndDisabled();
    ImGui::PopStyleColor(3);
    return pressed && enabled;
}

void finish_button(
    const bool active, const char* tooltip, const ImVec4 icon_colour) {
    if (active) {
        ImGui::GetWindowDrawList()->AddRect(
            ImGui::GetItemRectMin(), ImGui::GetItemRectMax(),
            theme::u32(theme::accent, 0.5F),
            ImGui::GetStyle().FrameRounding);
    }
    if (tooltip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tooltip);
    (void)icon_colour;
}

}  // namespace

void draw(
    ImDrawList* draw_list, const Icon icon, const ImVec2 min, const ImVec2 max,
    const ImU32 colour, float thickness) {
    const float extent = std::min(max.x - min.x, max.y - min.y);
    if (thickness <= 0.F) thickness = std::max(1.25F, extent * 0.075F);
    const ImVec2 centre = point(min, max, 0.5F, 0.5F);

    switch (icon) {
        case Icon::select: {
            const ImVec2 cursor[] = {
                point(min, max, 0.24F, 0.18F),
                point(min, max, 0.72F, 0.55F),
                point(min, max, 0.50F, 0.59F),
                point(min, max, 0.62F, 0.82F),
                point(min, max, 0.50F, 0.88F),
                point(min, max, 0.39F, 0.64F),
                point(min, max, 0.24F, 0.80F)};
            draw_list->AddPolyline(cursor, 7, colour, 0, thickness);
            line(draw_list, min, max, 0.24F, 0.80F, 0.24F, 0.18F, colour,
                 thickness);
            break;
        }
        case Icon::translate: {
            line(draw_list, min, max, 0.18F, 0.50F, 0.82F, 0.50F, colour,
                 thickness);
            line(draw_list, min, max, 0.50F, 0.18F, 0.50F, 0.82F, colour,
                 thickness);
            arrow_head(
                draw_list, point(min, max, 0.18F, 0.50F),
                point(min, max, 0.31F, 0.39F),
                point(min, max, 0.31F, 0.61F), colour, thickness);
            arrow_head(
                draw_list, point(min, max, 0.82F, 0.50F),
                point(min, max, 0.69F, 0.39F),
                point(min, max, 0.69F, 0.61F), colour, thickness);
            arrow_head(
                draw_list, point(min, max, 0.50F, 0.18F),
                point(min, max, 0.39F, 0.31F),
                point(min, max, 0.61F, 0.31F), colour, thickness);
            arrow_head(
                draw_list, point(min, max, 0.50F, 0.82F),
                point(min, max, 0.39F, 0.69F),
                point(min, max, 0.61F, 0.69F), colour, thickness);
            break;
        }
        case Icon::rotate: {
            draw_list->PathArcTo(centre, extent * 0.31F, -2.65F, 2.15F, 20);
            draw_list->PathStroke(colour, 0, thickness);
            arrow_head(
                draw_list, point(min, max, 0.33F, 0.24F),
                point(min, max, 0.18F, 0.27F),
                point(min, max, 0.29F, 0.40F), colour, thickness);
            break;
        }
        case Icon::scale: {
            line(draw_list, min, max, 0.30F, 0.70F, 0.70F, 0.30F, colour,
                 thickness);
            draw_list->AddRect(
                point(min, max, 0.62F, 0.18F), point(min, max, 0.82F, 0.38F),
                colour, 1.F, 0, thickness);
            draw_list->AddRectFilled(
                point(min, max, 0.20F, 0.62F), point(min, max, 0.38F, 0.80F),
                colour, 1.F);
            break;
        }
        case Icon::cube: {
            const ImVec2 top = point(min, max, 0.50F, 0.17F);
            const ImVec2 left = point(min, max, 0.22F, 0.34F);
            const ImVec2 right = point(min, max, 0.78F, 0.34F);
            const ImVec2 bottom_left = point(min, max, 0.22F, 0.68F);
            const ImVec2 bottom = point(min, max, 0.50F, 0.84F);
            const ImVec2 bottom_right = point(min, max, 0.78F, 0.68F);
            const ImVec2 outline[] = {
                top, right, bottom_right, bottom, bottom_left, left};
            draw_list->AddPolyline(outline, 6, colour, ImDrawFlags_Closed,
                                   thickness);
            draw_list->AddLine(left, centre, colour, thickness);
            draw_list->AddLine(right, centre, colour, thickness);
            draw_list->AddLine(centre, bottom, colour, thickness);
            break;
        }
        case Icon::world: {
            draw_list->AddCircle(centre, extent * 0.33F, colour, 20, thickness);
            draw_list->AddEllipse(
                centre, {extent * 0.16F, extent * 0.33F}, colour, 0.F, 20,
                thickness);
            line(draw_list, min, max, 0.19F, 0.50F, 0.81F, 0.50F, colour,
                 thickness);
            break;
        }
        case Icon::snap: {
            draw_list->PathArcTo(centre, extent * 0.28F, 0.F, 3.14159265F, 16);
            draw_list->PathStroke(colour, 0, thickness * 2.F);
            draw_list->AddRectFilled(
                point(min, max, 0.20F, 0.47F), point(min, max, 0.35F, 0.76F),
                colour, 1.F);
            draw_list->AddRectFilled(
                point(min, max, 0.65F, 0.47F), point(min, max, 0.80F, 0.76F),
                colour, 1.F);
            break;
        }
        case Icon::grid: {
            for (int i = 0; i < 4; ++i) {
                const float t = 0.22F + i * 0.19F;
                line(draw_list, min, max, t, 0.22F, t, 0.78F, colour,
                     thickness * 0.75F);
                line(draw_list, min, max, 0.22F, t, 0.78F, t, colour,
                     thickness * 0.75F);
            }
            break;
        }
        case Icon::camera: {
            draw_list->AddRect(
                point(min, max, 0.17F, 0.33F), point(min, max, 0.83F, 0.76F),
                colour, 2.F, 0, thickness);
            draw_list->AddCircle(centre, extent * 0.14F, colour, 16, thickness);
            const ImVec2 top[] = {
                point(min, max, 0.29F, 0.33F),
                point(min, max, 0.38F, 0.21F),
                point(min, max, 0.60F, 0.21F),
                point(min, max, 0.69F, 0.33F)};
            draw_list->AddPolyline(top, 4, colour, 0, thickness);
            break;
        }
        case Icon::frame: {
            line(draw_list, min, max, 0.20F, 0.40F, 0.20F, 0.20F, colour,
                 thickness);
            line(draw_list, min, max, 0.20F, 0.20F, 0.40F, 0.20F, colour,
                 thickness);
            line(draw_list, min, max, 0.60F, 0.20F, 0.80F, 0.20F, colour,
                 thickness);
            line(draw_list, min, max, 0.80F, 0.20F, 0.80F, 0.40F, colour,
                 thickness);
            line(draw_list, min, max, 0.20F, 0.60F, 0.20F, 0.80F, colour,
                 thickness);
            line(draw_list, min, max, 0.20F, 0.80F, 0.40F, 0.80F, colour,
                 thickness);
            line(draw_list, min, max, 0.60F, 0.80F, 0.80F, 0.80F, colour,
                 thickness);
            line(draw_list, min, max, 0.80F, 0.60F, 0.80F, 0.80F, colour,
                 thickness);
            break;
        }
        case Icon::folder: {
            const ImVec2 outline[] = {
                point(min, max, 0.14F, 0.30F),
                point(min, max, 0.42F, 0.30F),
                point(min, max, 0.50F, 0.40F),
                point(min, max, 0.86F, 0.40F),
                point(min, max, 0.78F, 0.76F),
                point(min, max, 0.18F, 0.76F)};
            draw_list->AddPolyline(outline, 6, colour, ImDrawFlags_Closed,
                                   thickness);
            break;
        }
        case Icon::align: {
            draw_list->AddRect(
                point(min, max, 0.18F, 0.28F), point(min, max, 0.66F, 0.68F),
                colour, 1.F, 0, thickness);
            draw_list->AddRect(
                point(min, max, 0.34F, 0.38F), point(min, max, 0.82F, 0.78F),
                colour, 1.F, 0, thickness);
            line(draw_list, min, max, 0.40F, 0.61F, 0.52F, 0.50F, colour,
                 thickness);
            line(draw_list, min, max, 0.52F, 0.50F, 0.72F, 0.69F, colour,
                 thickness);
            break;
        }
        case Icon::train: {
            const ImVec2 triangle[] = {
                point(min, max, 0.28F, 0.22F),
                point(min, max, 0.28F, 0.78F),
                point(min, max, 0.72F, 0.50F)};
            draw_list->AddTriangleFilled(
                triangle[0], triangle[1], triangle[2], colour);
            draw_list->AddCircleFilled(
                point(min, max, 0.75F, 0.24F), extent * 0.055F, colour);
            break;
        }
        case Icon::output: {
            draw_list->AddRect(
                point(min, max, 0.19F, 0.48F), point(min, max, 0.81F, 0.78F),
                colour, 1.F, 0, thickness);
            line(draw_list, min, max, 0.50F, 0.18F, 0.50F, 0.60F, colour,
                 thickness);
            arrow_head(
                draw_list, point(min, max, 0.50F, 0.18F),
                point(min, max, 0.36F, 0.34F),
                point(min, max, 0.64F, 0.34F), colour, thickness);
            break;
        }
        case Icon::stop:
            draw_list->AddRectFilled(
                point(min, max, 0.28F, 0.28F), point(min, max, 0.72F, 0.72F),
                colour, 1.F);
            break;
        case Icon::pause:
            draw_list->AddRectFilled(
                point(min, max, 0.28F, 0.22F), point(min, max, 0.44F, 0.78F),
                colour, 1.F);
            draw_list->AddRectFilled(
                point(min, max, 0.56F, 0.22F), point(min, max, 0.72F, 0.78F),
                colour, 1.F);
            break;
        case Icon::play: {
            const ImVec2 triangle[] = {
                point(min, max, 0.30F, 0.20F),
                point(min, max, 0.30F, 0.80F),
                point(min, max, 0.78F, 0.50F)};
            draw_list->AddTriangleFilled(
                triangle[0], triangle[1], triangle[2], colour);
            break;
        }
        case Icon::points:
            draw_list->AddCircleFilled(
                point(min, max, 0.30F, 0.62F), extent * 0.09F, colour);
            draw_list->AddCircleFilled(
                point(min, max, 0.52F, 0.31F), extent * 0.09F, colour);
            draw_list->AddCircleFilled(
                point(min, max, 0.72F, 0.66F), extent * 0.09F, colour);
            break;
        case Icon::splat:
            draw_list->AddCircleFilled(
                point(min, max, 0.50F, 0.52F), extent * 0.22F, colour);
            draw_list->AddCircleFilled(
                point(min, max, 0.36F, 0.40F), extent * 0.13F, colour);
            break;
        case Icon::rings: {
            draw_list->AddEllipse(
                point(min, max, 0.50F, 0.50F),
                {extent * 0.30F, extent * 0.18F}, colour, -0.40F, 18,
                thickness);
            draw_list->AddEllipse(
                point(min, max, 0.50F, 0.50F),
                {extent * 0.16F, extent * 0.10F}, colour, -0.40F, 14,
                thickness);
            break;
        }
        case Icon::gpu: {
            draw_list->AddRect(
                point(min, max, 0.25F, 0.25F), point(min, max, 0.75F, 0.75F),
                colour, 2.F, 0, thickness);
            draw_list->AddRect(
                point(min, max, 0.38F, 0.38F), point(min, max, 0.62F, 0.62F),
                colour, 1.F, 0, thickness);
            for (int i = 0; i < 3; ++i) {
                const float t = 0.34F + i * 0.16F;
                line(draw_list, min, max, t, 0.14F, t, 0.25F, colour,
                     thickness * 0.7F);
                line(draw_list, min, max, t, 0.75F, t, 0.86F, colour,
                     thickness * 0.7F);
                line(draw_list, min, max, 0.14F, t, 0.25F, t, colour,
                     thickness * 0.7F);
                line(draw_list, min, max, 0.75F, t, 0.86F, t, colour,
                     thickness * 0.7F);
            }
            break;
        }
        case Icon::copy: {
            const float rounding = std::max(1.5F, extent * 0.11F);
            const float stroke = std::max(thickness, extent * 0.09F);
            draw_list->AddRect(
                point(min, max, 0.36F, 0.16F), point(min, max, 0.84F, 0.64F),
                colour, rounding, 0, stroke);
            draw_list->AddRect(
                point(min, max, 0.16F, 0.36F), point(min, max, 0.64F, 0.84F),
                colour, rounding, 0, stroke);
            break;
        }
        case Icon::trash: {
            const float stroke = std::max(thickness, extent * 0.09F);
            const float rounding = std::max(1.4F, extent * 0.10F);
            draw_list->AddLine(
                point(min, max, 0.40F, 0.16F), point(min, max, 0.60F, 0.16F),
                colour, stroke);
            draw_list->AddLine(
                point(min, max, 0.40F, 0.16F), point(min, max, 0.40F, 0.28F),
                colour, stroke);
            draw_list->AddLine(
                point(min, max, 0.60F, 0.16F), point(min, max, 0.60F, 0.28F),
                colour, stroke);
            draw_list->AddLine(
                point(min, max, 0.18F, 0.28F), point(min, max, 0.82F, 0.28F),
                colour, stroke);
            draw_list->AddRect(
                point(min, max, 0.28F, 0.34F), point(min, max, 0.72F, 0.84F),
                colour, rounding, ImDrawFlags_RoundCornersBottom, stroke);
            line(draw_list, min, max, 0.42F, 0.46F, 0.42F, 0.72F, colour,
                 stroke * 0.85F);
            line(draw_list, min, max, 0.58F, 0.46F, 0.58F, 0.72F, colour,
                 stroke * 0.85F);
            break;
        }
        case Icon::search: {
            const float stroke = std::max(thickness, extent * 0.09F);
            const ImVec2 lens = point(min, max, 0.40F, 0.40F);
            const float radius = extent * 0.23F;
            draw_list->AddCircle(lens, radius, colour, 24, stroke);
            const float diag = 0.70710678F;
            draw_list->AddLine(
                {lens.x + radius * diag, lens.y + radius * diag},
                {lens.x + radius * diag + extent * 0.28F,
                 lens.y + radius * diag + extent * 0.28F},
                colour, stroke);
            break;
        }
        case Icon::follow: {
            const float stroke = std::max(thickness, extent * 0.09F);
            draw_list->PathLineTo(point(min, max, 0.28F, 0.22F));
            draw_list->PathLineTo(point(min, max, 0.50F, 0.40F));
            draw_list->PathLineTo(point(min, max, 0.72F, 0.22F));
            draw_list->PathStroke(colour, 0, stroke);
            draw_list->PathLineTo(point(min, max, 0.28F, 0.44F));
            draw_list->PathLineTo(point(min, max, 0.50F, 0.62F));
            draw_list->PathLineTo(point(min, max, 0.72F, 0.44F));
            draw_list->PathStroke(colour, 0, stroke);
            draw_list->AddLine(
                point(min, max, 0.24F, 0.80F), point(min, max, 0.76F, 0.80F),
                colour, stroke);
            break;
        }
        case Icon::clock: {
            draw_list->AddCircle(
                centre, extent * 0.36F, colour, 0, thickness);
            line(draw_list, min, max, 0.50F, 0.50F, 0.50F, 0.28F, colour,
                 thickness);
            line(draw_list, min, max, 0.50F, 0.50F, 0.72F, 0.50F, colour,
                 thickness);
            break;
        }
        case Icon::view2d: {
            draw_list->AddRect(
                point(min, max, 0.16F, 0.24F), point(min, max, 0.84F, 0.76F),
                colour, 2.F, 0, thickness);
            draw_list->AddTriangleFilled(
                point(min, max, 0.24F, 0.64F), point(min, max, 0.42F, 0.44F),
                point(min, max, 0.58F, 0.64F), colour);
            draw_list->AddTriangleFilled(
                point(min, max, 0.48F, 0.64F), point(min, max, 0.66F, 0.38F),
                point(min, max, 0.78F, 0.64F), colour);
            draw_list->AddCircleFilled(
                point(min, max, 0.30F, 0.36F), extent * 0.055F, colour);
            break;
        }
        case Icon::compare: {
            draw_list->AddRect(
                point(min, max, 0.16F, 0.22F), point(min, max, 0.84F, 0.78F),
                colour, 2.F, 0, thickness);
            line(draw_list, min, max, 0.50F, 0.22F, 0.50F, 0.78F, colour,
                 thickness);
            draw_list->AddCircleFilled(
                point(min, max, 0.34F, 0.50F), extent * 0.07F, colour);
            draw_list->AddCircle(
                point(min, max, 0.66F, 0.50F), extent * 0.07F, colour, 12,
                thickness);
            break;
        }
        case Icon::heatmap: {
            draw_list->AddRect(
                point(min, max, 0.18F, 0.22F), point(min, max, 0.82F, 0.78F),
                colour, 2.F, 0, thickness);
            draw_list->AddRectFilled(
                point(min, max, 0.26F, 0.54F), point(min, max, 0.40F, 0.70F),
                colour, 1.F);
            draw_list->AddRectFilled(
                point(min, max, 0.43F, 0.42F), point(min, max, 0.57F, 0.70F),
                colour, 1.F);
            draw_list->AddRectFilled(
                point(min, max, 0.60F, 0.30F), point(min, max, 0.74F, 0.70F),
                colour, 1.F);
            break;
        }
        case Icon::features: {
            draw_list->AddCircle(
                point(min, max, 0.32F, 0.34F), extent * 0.10F, colour, 12,
                thickness);
            draw_list->AddCircleFilled(
                point(min, max, 0.68F, 0.40F), extent * 0.07F, colour);
            draw_list->AddCircleFilled(
                point(min, max, 0.42F, 0.68F), extent * 0.055F, colour);
            line(draw_list, min, max, 0.62F, 0.66F, 0.78F, 0.66F, colour,
                 thickness);
            line(draw_list, min, max, 0.70F, 0.58F, 0.70F, 0.74F, colour,
                 thickness);
            break;
        }
        case Icon::chevron_left: {
            line(draw_list, min, max, 0.60F, 0.24F, 0.34F, 0.50F, colour,
                 thickness);
            line(draw_list, min, max, 0.34F, 0.50F, 0.60F, 0.76F, colour,
                 thickness);
            break;
        }
        case Icon::chevron_right: {
            line(draw_list, min, max, 0.40F, 0.24F, 0.66F, 0.50F, colour,
                 thickness);
            line(draw_list, min, max, 0.66F, 0.50F, 0.40F, 0.76F, colour,
                 thickness);
            break;
        }
        case Icon::frustum: {
            const ImVec2 apex = point(min, max, 0.50F, 0.20F);
            const ImVec2 far[] = {
                point(min, max, 0.22F, 0.78F),
                point(min, max, 0.78F, 0.78F),
                point(min, max, 0.70F, 0.58F),
                point(min, max, 0.30F, 0.58F)};
            draw_list->AddPolyline(far, 4, colour, ImDrawFlags_Closed, thickness);
            draw_list->AddLine(apex, far[0], colour, thickness);
            draw_list->AddLine(apex, far[1], colour, thickness);
            draw_list->AddLine(apex, far[2], colour, thickness);
            draw_list->AddLine(apex, far[3], colour, thickness);
            break;
        }
        case Icon::photo: {
            draw_list->AddRect(
                point(min, max, 0.16F, 0.24F), point(min, max, 0.84F, 0.76F),
                colour, 2.F, 0, thickness);
            draw_list->AddCircle(
                point(min, max, 0.34F, 0.40F), extent * 0.07F, colour, 12,
                thickness);
            const ImVec2 hill[] = {
                point(min, max, 0.22F, 0.66F),
                point(min, max, 0.40F, 0.48F),
                point(min, max, 0.58F, 0.66F)};
            draw_list->AddPolyline(hill, 3, colour, 0, thickness);
            break;
        }
    }
}

bool button(
    const char* id, const Icon icon, const ImVec2 size, const bool active,
    const bool enabled, const char* tooltip) {
    const ButtonColours colours =
        button_colours(ButtonStyle::normal, active, enabled);
    const bool pressed = begin_icon_button(id, size, colours, enabled);
    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    const float pad = std::max(5.F, std::min(size.x, size.y) * 0.22F);
    draw(
        ImGui::GetWindowDrawList(), icon, {min.x + pad, min.y + pad},
        {max.x - pad, max.y - pad}, theme::u32(colours.icon));
    finish_button(active, tooltip, colours.icon);
    return pressed;
}

bool labeled_button(
    const char* id, const Icon icon, const char* label, const ImVec2 size,
    const ButtonStyle style, const bool enabled, const bool active,
    const char* tooltip) {
    const ButtonColours colours = button_colours(style, active, enabled);
    const bool pressed = begin_icon_button(id, size, colours, enabled);
    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    const float icon_size = std::min(18.F, size.y - 10.F);
    const ImVec2 icon_min{min.x + 10.F, min.y + (size.y - icon_size) * 0.5F};
    draw(
        ImGui::GetWindowDrawList(), icon, icon_min,
        {icon_min.x + icon_size, icon_min.y + icon_size},
        theme::u32(colours.icon));
    const ImVec2 text_size = ImGui::CalcTextSize(label);
    ImGui::GetWindowDrawList()->AddText(
        {icon_min.x + icon_size + 8.F, min.y + (size.y - text_size.y) * 0.5F},
        theme::u32(colours.icon), label);
    finish_button(active, tooltip, colours.icon);
    return pressed;
}

bool ghost_button(
    const char* id, const Icon icon, const ImVec2 size, const bool active,
    const bool enabled, const char* tooltip) {
    const ImVec4 transparent{0.F, 0.F, 0.F, 0.F};
    ButtonColours colours;
    if (!enabled) {
        colours = {transparent, transparent, transparent, theme::text_faint};
    } else if (active) {
        colours = {
            theme::fade(theme::accent, 0.16F),
            theme::fade(theme::accent, 0.24F),
            theme::fade(theme::accent, 0.24F), theme::accent};
    } else {
        colours = {
            transparent, theme::surface_3, theme::surface_2, theme::text_muted};
    }

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.F);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.F);
    const bool pressed = begin_icon_button(id, size, colours, enabled);
    ImGui::PopStyleVar(2);

    const bool hovered = ImGui::IsItemHovered();
    ImVec4 icon_colour = colours.icon;
    if (enabled && hovered && !active)
        icon_colour = icon == Icon::trash ? theme::danger : theme::text_bright;

    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    const float pad = std::max(7.F, std::min(size.x, size.y) * 0.26F);
    draw(
        ImGui::GetWindowDrawList(), icon, {min.x + pad, min.y + pad},
        {max.x - pad, max.y - pad}, theme::u32(icon_colour), 1.8F);
    finish_button(active, tooltip, icon_colour);
    return pressed;
}

void inline_icon(const Icon icon, const ImU32 colour, const float size) {
    const ImVec2 min = ImGui::GetCursorScreenPos();
    const float line_height = ImGui::GetTextLineHeight();
    const float y = min.y + (line_height - size) * 0.5F;
    draw(
        ImGui::GetWindowDrawList(), icon, {min.x, y}, {min.x + size, y + size},
        colour);
    ImGui::Dummy({size, line_height});
}

}  // namespace editor::icons
