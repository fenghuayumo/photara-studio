#include "viewport_gizmo.hpp"

#include "theme.hpp"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace editor {
namespace {

constexpr float k_gizmo_size = 108.F;
constexpr float k_gizmo_pad = 12.F;
constexpr float k_pi = 3.14159265358979323846F;

struct AxisEnd {
    ImVec2 position;
    Vec3 camera_offset;
    const char* label;
    ImU32 colour;
    bool positive{};
};

float squared_distance(const ImVec2 a, const ImVec2 b) {
    const float x = a.x - b.x;
    const float y = a.y - b.y;
    return x * x + y * y;
}

void orient_camera(OrbitCamera& camera, const Vec3 offset) {
    const float length = std::sqrt(
        offset.x * offset.x + offset.y * offset.y + offset.z * offset.z);
    if (length < 1e-6F) return;
    const Vec3 direction{
        offset.x / length, offset.y / length, offset.z / length};
    camera.pitch = std::clamp(
        std::asin(-direction.y), -k_pi * 0.487F, k_pi * 0.487F);
    camera.yaw = std::atan2(direction.x, direction.z);
}

bool draw_axes_gizmo(
    OrbitCamera& camera, const ImVec2 min, const ImVec2 max) {
    std::array<float, 16> view;
    camera_view_matrix(camera, min, max, view);

    const ImVec2 widget_min{
        max.x - k_gizmo_size - k_gizmo_pad, min.y + k_gizmo_pad};
    const ImVec2 widget_max{
        widget_min.x + k_gizmo_size, widget_min.y + k_gizmo_size};
    const ImVec2 centre{
        (widget_min.x + widget_max.x) * 0.5F,
        (widget_min.y + widget_max.y) * 0.5F};
    constexpr float axis_length = 31.F;

    // The reconstruction uses Y-down, so the navigation widget presents -Y
    // as the conventional positive/up direction shown in DCC applications.
    const Vec3 axes[3] = {{1, 0, 0}, {0, -1, 0}, {0, 0, 1}};
    const char* labels[3] = {"X", "Y", "Z"};
    const ImU32 colours[3] = {
        IM_COL32(244, 105, 147, 255), IM_COL32(175, 236, 63, 255),
        IM_COL32(73, 174, 239, 255)};

    std::array<AxisEnd, 6> ends;
    for (int axis = 0; axis < 3; ++axis) {
        const Vec3 direction = axes[axis];
        const float screen_x = view[0] * direction.x +
                               view[4] * direction.y +
                               view[8] * direction.z;
        const float screen_y = -(view[1] * direction.x +
                                 view[5] * direction.y +
                                 view[9] * direction.z);
        ends[axis * 2] = {
            {centre.x + screen_x * axis_length,
             centre.y + screen_y * axis_length},
            direction, labels[axis], colours[axis], true};
        ends[axis * 2 + 1] = {
            {centre.x - screen_x * axis_length,
             centre.y - screen_y * axis_length},
            {-direction.x, -direction.y, -direction.z}, nullptr,
            colours[axis], false};
    }

    ImGui::SetCursorScreenPos(widget_min);
    ImGui::InvisibleButton("##orientation_axes", {k_gizmo_size, k_gizmo_size});
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddCircleFilled(
        centre, 45.F, IM_COL32(196, 190, 181, 58), 32);
    draw->AddCircle(
        centre, 45.F, IM_COL32(235, 235, 235, 28), 32, 1.F);
    draw->AddCircleFilled(centre, 3.F, IM_COL32(180, 185, 195, 210));

    for (const AxisEnd& end : ends) {
        ImVec4 line = ImGui::ColorConvertU32ToFloat4(end.colour);
        line.w = end.positive ? 0.72F : 0.24F;
        const ImU32 line_colour = ImGui::ColorConvertFloat4ToU32(line);
        draw->AddLine(centre, end.position, line_colour, 1.5F);
    }

    const ImVec2 mouse = ImGui::GetIO().MousePos;
    int hot = -1;
    float best = 11.F * 11.F;
    if (hovered) {
        for (int i = 0; i < static_cast<int>(ends.size()); ++i) {
            const float distance = squared_distance(mouse, ends[i].position);
            if (distance < best) {
                best = distance;
                hot = i;
            }
        }
    }

    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < static_cast<int>(ends.size()); ++i) {
            const AxisEnd& end = ends[i];
            if ((pass == 0) != !end.positive) continue;
            const float radius = i == hot ? 8.F : (end.positive ? 7.F : 5.F);
            ImVec4 endpoint = ImGui::ColorConvertU32ToFloat4(end.colour);
            if (!end.positive) endpoint.w = i == hot ? 0.85F : 0.46F;
            draw->AddCircleFilled(
                end.position, radius,
                ImGui::ColorConvertFloat4ToU32(endpoint), 16);
            if (end.label) {
                const ImVec2 text = ImGui::CalcTextSize(end.label);
                draw->AddText(
                    {end.position.x - text.x * 0.5F,
                     end.position.y - text.y * 0.5F},
                    IM_COL32(30, 34, 40, 255), end.label);
            }
        }
    }

    if (hot >= 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        orient_camera(camera, ends[hot].camera_offset);
    return hovered;
}

}  // namespace

bool draw_viewport_gizmo(
    ViewportGizmoState& state, OrbitCamera& camera, const ImVec2 min,
    const ImVec2 max) {
    if (!state.visible) return false;
    const bool captures = draw_axes_gizmo(camera, min, max);
    if (captures) {
        ImGui::SetTooltip("Click an axis to change camera view");
    }
    return captures;
}

}  // namespace editor
