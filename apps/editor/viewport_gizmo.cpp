#include "viewport_gizmo.hpp"

#include "theme.hpp"

#include "ImGuizmo.h"
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
    ImGui::SetNextItemAllowOverlap();
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

void apply_imguizmo_style() {
    ImGuizmo::Style& style = ImGuizmo::GetStyle();
    style.TranslationLineThickness = 3.F;
    style.RotationLineThickness = 2.5F;
    style.ScaleLineThickness = 2.5F;
    style.CenterCircleSize = 6.F;
    style.Colors[ImGuizmo::DIRECTION_X] = {0.96F, 0.41F, 0.58F, 1.F};
    style.Colors[ImGuizmo::DIRECTION_Y] = {0.69F, 0.93F, 0.25F, 1.F};
    style.Colors[ImGuizmo::DIRECTION_Z] = {0.29F, 0.68F, 0.94F, 1.F};
    style.Colors[ImGuizmo::PLANE_X] = {0.96F, 0.41F, 0.58F, 0.22F};
    style.Colors[ImGuizmo::PLANE_Y] = {0.69F, 0.93F, 0.25F, 0.22F};
    style.Colors[ImGuizmo::PLANE_Z] = {0.29F, 0.68F, 0.94F, 0.22F};
    style.Colors[ImGuizmo::SELECTION] = {0.29F, 0.71F, 0.96F, 0.90F};
}

float vec_component(const Vec3& v, const int axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

void set_vec_component(Vec3& v, const int axis, const float value) {
    if (axis == 0) v.x = value;
    else if (axis == 1) v.y = value;
    else v.z = value;
}

Vec3 axis_vector(const int axis) {
    Vec3 v{};
    set_vec_component(v, axis, 1.F);
    return v;
}

struct BoxFace {
    int axis{};
    bool is_max{};
};

constexpr BoxFace k_box_faces[6] = {
    {0, false}, {0, true}, {1, false}, {1, true}, {2, false}, {2, true}};

Vec3 reconstruction_box_face_centre(
    const ReconstructionBox& box, const BoxFace face) {
    Vec3 centre = box.centre();
    set_vec_component(
        centre, face.axis,
        vec_component(face.is_max ? box.max : box.min, face.axis));
    return centre;
}

float squared_length(const Vec3 v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

Vec3 add3(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 sub3(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 scale3(const Vec3 a, const float s) {
    return {a.x * s, a.y * s, a.z * s};
}
float dot3(const Vec3 a, const Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec3 cross3(const Vec3 a, const Vec3 b) {
    return {
        a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
Vec3 norm3(const Vec3 v) {
    const float length = std::sqrt(squared_length(v));
    return length > 1e-20F ? scale3(v, 1.F / length) : Vec3{0.F, 0.F, 1.F};
}

bool intersect_plane(
    const Vec3 origin, const Vec3 direction, const Vec3 point,
    const Vec3 normal, Vec3& hit) {
    const float denom = dot3(direction, normal);
    if (std::abs(denom) < 1e-8F) return false;
    const float t = dot3(sub3(point, origin), normal) / denom;
    if (t <= 1e-4F) return false;
    hit = add3(origin, scale3(direction, t));
    return true;
}

bool draw_region_handles(
    ViewportGizmoState& state, ReconstructionBox& box,
    const ReconstructionTransform* xf, const OrbitCamera& camera,
    const ImVec2 min, const ImVec2 max, const float scene_radius) {
    if (!box.valid) {
        state.box = {};
        return false;
    }

    const ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    const bool in_view =
        mouse.x >= min.x && mouse.x <= max.x && mouse.y >= min.y &&
        mouse.y <= max.y;
    const ImVec2 orient_min{
        max.x - 108.F - 12.F, min.y + 12.F};
    const bool over_orient =
        mouse.x >= orient_min.x && mouse.x <= orient_min.x + 108.F &&
        mouse.y >= orient_min.y && mouse.y <= orient_min.y + 108.F;

    int hot = state.box.dragging ? state.box.face : -1;
    ImVec2 handle_screen[6];
    bool handle_ok[6]{};
    float handle_depth[6]{};
    float best = 12.F * 12.F;
    for (int i = 0; i < 6; ++i) {
        const Vec3 local = reconstruction_box_face_centre(box, k_box_faces[i]);
        const Vec3 world = xf ? transform_point(*xf, local) : local;
        if (!project_world_to_screen(
                camera, min, max, world, handle_screen[i], handle_depth[i]))
            continue;
        handle_ok[i] = true;
        if (state.box.dragging || !in_view || over_orient) continue;
        const float distance = squared_distance(mouse, handle_screen[i]);
        if (distance < best) {
            best = distance;
            hot = i;
        }
    }

    if (!state.box.dragging && hot >= 0 &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        const Vec3 local =
            reconstruction_box_face_centre(box, k_box_faces[hot]);
        state.box.dragging = true;
        state.box.face = hot;
        state.box.grab_point = xf ? transform_point(*xf, local) : local;
        state.box.start = box;
    }

    if (state.box.dragging) {
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            state.box.dragging = false;
            state.box.face = -1;
        } else {
            const BoxFace face = k_box_faces[state.box.face];
            Vec3 eye;
            Vec3 dir;
            camera_world_ray(camera, min, max, mouse, eye, dir);
            const Vec3 local_axis = axis_vector(face.axis);
            Vec3 axis_world =
                xf ? transform_vector(*xf, local_axis) : local_axis;
            const float axis_len = std::sqrt(squared_length(axis_world));
            if (axis_len > 1e-8F) {
                const Vec3 axis_n = scale3(axis_world, 1.F / axis_len);
                Vec3 plane_n = cross3(axis_n, sub3(state.box.grab_point, eye));
                if (squared_length(plane_n) < 1e-10F)
                    plane_n = cross3(axis_n, Vec3{0.F, 1.F, 0.F});
                plane_n = cross3(plane_n, axis_n);
                Vec3 hit;
                if (intersect_plane(
                        eye, dir, state.box.grab_point, norm3(plane_n), hit)) {
                    float along = dot3(sub3(hit, state.box.grab_point), axis_n);
                    if (io.KeyShift) along *= 0.1F;
                    float local_delta = along / axis_len;
                    box = state.box.start;
                    float& side = face.is_max
                        ? (face.axis == 0
                               ? box.max.x
                               : (face.axis == 1 ? box.max.y : box.max.z))
                        : (face.axis == 0
                               ? box.min.x
                               : (face.axis == 1 ? box.min.y : box.min.z));
                    side = vec_component(
                               face.is_max ? state.box.start.max
                                           : state.box.start.min,
                               face.axis) +
                           local_delta;
                    if (io.KeyCtrl) {
                        const float step = std::max(0.01F, scene_radius * 0.02F);
                        side = std::round(side / step) * step;
                    }
                    constexpr float k_min_span = 1e-3F;
                    if (face.is_max)
                        side = std::max(
                            side, vec_component(box.min, face.axis) + k_min_span);
                    else
                        side = std::min(
                            side, vec_component(box.max, face.axis) - k_min_span);
                    box.user_set = true;
                    box.valid = true;
                }
            }
        }
    }

    ImDrawList* draw = ImGui::GetWindowDrawList();
    if (hot >= 0) {
        const Vec3 mn = box.min;
        const Vec3 mx = box.max;
        const Vec3 local[8] = {
            {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mn.x, mx.y, mn.z},
            {mx.x, mx.y, mn.z}, {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z},
            {mn.x, mx.y, mx.z}, {mx.x, mx.y, mx.z}};
        constexpr int corners[6][4] = {
            {0, 2, 6, 4}, {1, 3, 7, 5}, {0, 1, 5, 4},
            {2, 3, 7, 6}, {0, 1, 3, 2}, {4, 5, 7, 6}};
        ImVec2 screen[4];
        bool visible = true;
        float depth{};
        for (int i = 0; i < 4; ++i) {
            const Vec3 world = xf ? transform_point(*xf, local[corners[hot][i]])
                                  : local[corners[hot][i]];
            visible = visible &&
                      project_world_to_screen(
                          camera, min, max, world, screen[i], depth);
        }
        if (visible) {
            draw->AddConvexPolyFilled(
                screen, 4, theme::u32(theme::warning, 0.16F));
            draw->AddPolyline(
                screen, 4, theme::u32(theme::warning, 0.95F),
                ImDrawFlags_Closed, 2.F);
        }
    }
    for (int i = 0; i < 6; ++i) {
        if (!handle_ok[i]) continue;
        const bool active = i == hot;
        const float radius = active ? 8.F : 6.F;
        const ImU32 fill = active
            ? theme::u32(theme::warning, 1.F)
            : theme::u32(theme::warning, 0.82F);
        draw->AddCircleFilled(handle_screen[i], radius + 1.4F, IM_COL32(20, 22, 28, 220), 12);
        draw->AddCircleFilled(handle_screen[i], radius, fill, 12);
    }

    if (hot >= 0) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::SetTooltip(
            "Drag a face to resize the reconstruction region\n"
            "Shift: fine  ·  Ctrl: snap");
    }
    return state.box.dragging || hot >= 0;
}

bool draw_object_gizmo(
    ViewportGizmoState& state, ReconstructionTransform& xf,
    const OrbitCamera& camera, const ImVec2 min, const ImVec2 max,
    const float scene_radius) {
    if (state.tool == TransformTool::orbit) return false;

    apply_imguizmo_style();
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::AllowAxisFlip(false);
    ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());
    ImGuizmo::SetRect(
        min.x, min.y, std::max(1.F, max.x - min.x),
        std::max(1.F, max.y - min.y));
    ImGuizmo::SetGizmoSizeClipSpace(0.15F);

    std::array<float, 16> view{};
    std::array<float, 16> projection{};
    std::array<float, 16> matrix{};
    camera_view_matrix(camera, min, max, view);
    camera_projection_matrix(camera, min, max, projection);
    reconstruction_gizmo_matrix(xf, matrix);

    ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
    if (state.tool == TransformTool::rotate)
        operation = ImGuizmo::ROTATE;
    else if (state.tool == TransformTool::scale)
        operation = ImGuizmo::SCALEU;

    const ImGuizmo::MODE mode = state.space == GizmoSpace::local
        ? ImGuizmo::LOCAL
        : ImGuizmo::WORLD;

    float snap[3] = {};
    const float* snap_ptr = nullptr;
    const ImGuiIO& io = ImGui::GetIO();
    if (io.KeyCtrl) {
        if (state.tool == TransformTool::rotate) {
            snap[0] = snap[1] = snap[2] = 15.F;
        } else if (state.tool == TransformTool::scale) {
            snap[0] = snap[1] = snap[2] = 0.1F;
        } else {
            const float step = std::max(0.01F, scene_radius * 0.02F);
            snap[0] = snap[1] = snap[2] = step;
        }
        snap_ptr = snap;
    }

    const bool changed = ImGuizmo::Manipulate(
        view.data(), projection.data(), operation, mode, matrix.data(),
        nullptr, snap_ptr);
    if (changed) reconstruction_from_gizmo_matrix(xf, matrix);

    if (ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        if (state.tool == TransformTool::translate)
            ImGui::SetTooltip("Move reconstruction  ·  Ctrl snaps");
        else if (state.tool == TransformTool::rotate)
            ImGui::SetTooltip("Rotate reconstruction  ·  Ctrl snaps 15°");
        else
            ImGui::SetTooltip("Scale reconstruction uniformly  ·  Ctrl snaps");
    }
    return ImGuizmo::IsOver() || ImGuizmo::IsUsing();
}

}  // namespace

bool draw_viewport_gizmo(
    ViewportGizmoState& state, OrbitCamera& camera, const ImVec2 min,
    const ImVec2 max, ReconstructionTransform* transform,
    const float scene_radius, const bool object_enabled,
    ReconstructionBox* region, const bool region_enabled) {
    if (!state.visible) return false;
    ImGuizmo::BeginFrame();
    bool captures = false;
    if (object_enabled && transform != nullptr)
        captures = draw_object_gizmo(
            state, *transform, camera, min, max, scene_radius);
    if (!captures && region_enabled && region != nullptr)
        captures = draw_region_handles(
            state, *region, transform, camera, min, max, scene_radius);
    else if (!region_enabled || region == nullptr)
        state.box = {};
    if (!captures) {
        captures = draw_axes_gizmo(camera, min, max);
        if (captures)
            ImGui::SetTooltip("Click an axis to change camera view");
    } else {
        draw_axes_gizmo(camera, min, max);
    }
    return captures;
}

}  // namespace editor
