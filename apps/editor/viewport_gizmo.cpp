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

ImU32 axis_colour(const int axis, const bool hot) {
    const ImU32 colours[3] = {
        IM_COL32(244, 105, 147, 255), IM_COL32(175, 236, 63, 255),
        IM_COL32(73, 174, 239, 255)};
    if (!hot) return colours[axis];
    const ImVec4 c = ImGui::ColorConvertU32ToFloat4(colours[axis]);
    return ImGui::ColorConvertFloat4ToU32(
        {std::min(1.F, c.x + 0.18F), std::min(1.F, c.y + 0.18F),
         std::min(1.F, c.z + 0.18F), 1.F});
}

float point_segment_distance(
    const ImVec2 point, const ImVec2 a, const ImVec2 b) {
    const float abx = b.x - a.x;
    const float aby = b.y - a.y;
    const float length2 = abx * abx + aby * aby;
    float t = 0.F;
    if (length2 > 1e-8F)
        t = std::clamp(
            ((point.x - a.x) * abx + (point.y - a.y) * aby) / length2, 0.F,
            1.F);
    const float dx = point.x - (a.x + abx * t);
    const float dy = point.y - (a.y + aby * t);
    return std::sqrt(dx * dx + dy * dy);
}

bool point_in_quad(const ImVec2 point, const ImVec2 quad[4]) {
    auto side = [](const ImVec2 a, const ImVec2 b, const ImVec2 p) {
        return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
    };
    const float s0 = side(quad[0], quad[1], point);
    const float s1 = side(quad[1], quad[2], point);
    const float s2 = side(quad[2], quad[3], point);
    const float s3 = side(quad[3], quad[0], point);
    return (s0 >= 0.F && s1 >= 0.F && s2 >= 0.F && s3 >= 0.F) ||
           (s0 <= 0.F && s1 <= 0.F && s2 <= 0.F && s3 <= 0.F);
}

void draw_arrow_head(
    ImDrawList* draw, const ImVec2 tip, const ImVec2 along, const ImU32 colour) {
    const float dx = tip.x - along.x;
    const float dy = tip.y - along.y;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length < 1.F) return;
    const float nx = dx / length;
    const float ny = dy / length;
    const float px = -ny;
    const float py = nx;
    constexpr float size = 11.F;
    draw->AddTriangleFilled(
        tip,
        {tip.x - nx * size + px * 5.5F, tip.y - ny * size + py * 5.5F},
        {tip.x - nx * size - px * 5.5F, tip.y - ny * size - py * 5.5F},
        colour);
}

float gizmo_world_length(
    const OrbitCamera& camera, const ImVec2 min, const ImVec2 max) {
    const float height = std::max(1.F, max.y - min.y);
    if (camera.projection == EditorProjection::orthographic)
        return 96.F * camera.ortho_height / std::max(1.F, height);
    if (camera.projection == EditorProjection::panorama)
        return camera.distance * 0.25F;
    const float half_fov = camera.fov_degrees * 0.5F * k_pi / 180.F;
    const float focal = height * 0.5F / std::max(1e-4F, std::tan(half_fov));
    return 96.F * camera.distance / std::max(1.F, focal);
}

Vec3 drag_axis_delta(
    const OrbitCamera& camera, const ImVec2 min, const ImVec2 max,
    const ImVec2 mouse, const Vec3 origin, const Vec3 grab, const Vec3 axis) {
    Vec3 eye;
    Vec3 dir;
    camera_world_ray(camera, min, max, mouse, eye, dir);
    const Vec3 axis_n = norm3(axis);
    Vec3 plane_n = cross3(axis_n, sub3(origin, eye));
    if (squared_length(plane_n) < 1e-10F)
        plane_n = cross3(axis_n, Vec3{0.F, 1.F, 0.F});
    plane_n = cross3(plane_n, axis_n);
    Vec3 hit;
    if (!intersect_plane(eye, dir, origin, norm3(plane_n), hit)) return {};
    const float along = dot3(sub3(hit, grab), axis_n);
    return scale3(axis_n, along);
}

Vec3 drag_plane_delta(
    const OrbitCamera& camera, const ImVec2 min, const ImVec2 max,
    const ImVec2 mouse, const Vec3 origin, const Vec3 grab, const Vec3 normal) {
    Vec3 eye;
    Vec3 dir;
    camera_world_ray(camera, min, max, mouse, eye, dir);
    Vec3 hit;
    if (!intersect_plane(eye, dir, origin, normal, hit)) return {};
    const Vec3 n = norm3(normal);
    return sub3(sub3(hit, grab), scale3(n, dot3(sub3(hit, grab), n)));
}

void translate_box(ReconstructionBox& box, const Vec3 delta) {
    box.min = add3(box.min, delta);
    box.max = add3(box.max, delta);
    box.user_set = true;
    box.valid = true;
}

constexpr int k_part_face = 0;
constexpr int k_part_axis = 6;
constexpr int k_part_plane = 9;

bool draw_region_handles(
    ViewportGizmoState& state, ReconstructionBox& box,
    const OrbitCamera& camera, const ImVec2 min, const ImVec2 max,
    const float scene_radius) {
    if (!box.valid) {
        state.box = {};
        return false;
    }

    const ImGuiIO& io = ImGui::GetIO();
    const ImVec2 mouse = io.MousePos;
    const bool in_view =
        mouse.x >= min.x && mouse.x <= max.x && mouse.y >= min.y &&
        mouse.y <= max.y;
    const ImVec2 orient_min{max.x - 108.F - 12.F, min.y + 12.F};
    const bool over_orient =
        mouse.x >= orient_min.x && mouse.x <= orient_min.x + 108.F &&
        mouse.y >= orient_min.y && mouse.y <= orient_min.y + 108.F;

    const Vec3 origin = box.centre();
    // Match the orientation widget and world axes: green is -Y (up) in this
    // Y-down reconstruction world.
    const Vec3 axes[3] = {{1.F, 0.F, 0.F}, {0.F, -1.F, 0.F}, {0.F, 0.F, 1.F}};
    const float handle_len = gizmo_world_length(camera, min, max);

    ImVec2 origin_screen;
    float origin_depth{};
    const bool origin_ok = project_world_to_screen(
        camera, min, max, origin, origin_screen, origin_depth);

    ImVec2 axis_ends[3];
    bool axis_ok[3]{};
    for (int i = 0; i < 3; ++i) {
        float depth{};
        axis_ok[i] = origin_ok &&
                     project_world_to_screen(
                         camera, min, max,
                         add3(origin, scale3(axes[i], handle_len)), axis_ends[i],
                         depth);
    }

    ImVec2 planes[3][4];
    bool plane_ok[3]{};
    const float plane_s = handle_len * 0.32F;
    const int plane_u[3] = {1, 2, 0};
    const int plane_v[3] = {2, 0, 1};
    for (int i = 0; i < 3; ++i) {
        const Vec3 u = scale3(axes[plane_u[i]], plane_s);
        const Vec3 v = scale3(axes[plane_v[i]], plane_s);
        const Vec3 pts[4] = {
            origin, add3(origin, u), add3(origin, add3(u, v)), add3(origin, v)};
        plane_ok[i] = true;
        for (int p = 0; p < 4; ++p) {
            float depth{};
            plane_ok[i] = plane_ok[i] &&
                          project_world_to_screen(
                              camera, min, max, pts[p], planes[i][p], depth);
        }
    }

    ImVec2 face_screen[6];
    bool face_ok[6]{};
    for (int i = 0; i < 6; ++i) {
        float depth{};
        face_ok[i] = project_world_to_screen(
            camera, min, max, reconstruction_box_face_centre(box, k_box_faces[i]),
            face_screen[i], depth);
    }

    int hot = state.box.dragging ? state.box.part : -1;
    if (!state.box.dragging && in_view && !over_orient && origin_ok) {
        float best = 14.F;
        for (int i = 0; i < 3; ++i) {
            if (!axis_ok[i]) continue;
            const float distance =
                point_segment_distance(mouse, origin_screen, axis_ends[i]);
            if (distance < best) {
                best = distance;
                hot = k_part_axis + i;
            }
        }
        if (hot < 0) {
            for (int i = 0; i < 3; ++i) {
                if (plane_ok[i] && point_in_quad(mouse, planes[i]))
                    hot = k_part_plane + i;
            }
        }
        if (hot < 0) {
            float best_face = 16.F * 16.F;
            for (int i = 0; i < 6; ++i) {
                if (!face_ok[i]) continue;
                const float distance = squared_distance(mouse, face_screen[i]);
                if (distance < best_face) {
                    best_face = distance;
                    hot = k_part_face + i;
                }
            }
        }
    }

    if (!state.box.dragging && hot >= 0 &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        Vec3 eye;
        Vec3 dir;
        camera_world_ray(camera, min, max, mouse, eye, dir);
        Vec3 grab = origin;
        if (hot >= k_part_axis && hot < k_part_plane) {
            const Vec3 axis = axes[hot - k_part_axis];
            Vec3 plane_n = cross3(axis, sub3(origin, eye));
            if (squared_length(plane_n) < 1e-10F)
                plane_n = cross3(axis, Vec3{0.F, 1.F, 0.F});
            plane_n = cross3(plane_n, axis);
            intersect_plane(eye, dir, origin, norm3(plane_n), grab);
        } else if (hot >= k_part_plane) {
            intersect_plane(
                eye, dir, origin, axes[hot - k_part_plane], grab);
        } else {
            grab = reconstruction_box_face_centre(box, k_box_faces[hot]);
        }
        state.box.dragging = true;
        state.box.part = hot;
        state.box.grab_point = grab;
        state.box.start = box;
    }

    if (state.box.dragging) {
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            state.box.dragging = false;
            state.box.part = -1;
        } else {
            box = state.box.start;
            const float precision = io.KeyShift ? 0.1F : 1.F;
            const float step = std::max(0.01F, scene_radius * 0.02F);
            const Vec3 drag_origin = state.box.start.centre();
            if (state.box.part >= k_part_axis && state.box.part < k_part_plane) {
                Vec3 delta = drag_axis_delta(
                    camera, min, max, mouse, drag_origin, state.box.grab_point,
                    axes[state.box.part - k_part_axis]);
                delta = scale3(delta, precision);
                if (io.KeyCtrl) {
                    const float along = dot3(
                        delta, axes[state.box.part - k_part_axis]);
                    delta = scale3(
                        axes[state.box.part - k_part_axis],
                        std::round(along / step) * step);
                }
                translate_box(box, delta);
            } else if (state.box.part >= k_part_plane) {
                Vec3 delta = drag_plane_delta(
                    camera, min, max, mouse, drag_origin, state.box.grab_point,
                    axes[state.box.part - k_part_plane]);
                delta = scale3(delta, precision);
                if (io.KeyCtrl) {
                    delta = {
                        std::round(delta.x / step) * step,
                        std::round(delta.y / step) * step,
                        std::round(delta.z / step) * step};
                }
                translate_box(box, delta);
            } else if (state.box.part >= 0 && state.box.part < 6) {
                const BoxFace face = k_box_faces[state.box.part];
                const Vec3 axis = axis_vector(face.axis);
                Vec3 delta = drag_axis_delta(
                    camera, min, max, mouse, state.box.grab_point,
                    state.box.grab_point, axis);
                float along = dot3(delta, axis) * precision;
                if (io.KeyCtrl) along = std::round(along / step) * step;
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
                       along;
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

    ImDrawList* draw = ImGui::GetWindowDrawList();
    const int active = state.box.dragging ? state.box.part : hot;

    if (active >= 0 && active < 6) {
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
            visible = visible &&
                      project_world_to_screen(
                          camera, min, max, local[corners[active][i]],
                          screen[i], depth);
        }
        if (visible) {
            draw->AddConvexPolyFilled(
                screen, 4, theme::u32(theme::warning, 0.16F));
            draw->AddPolyline(
                screen, 4, theme::u32(theme::warning, 0.95F),
                ImDrawFlags_Closed, 2.2F);
        }
    }

    if (origin_ok) {
        for (int i = 0; i < 3; ++i) {
            if (!plane_ok[i]) continue;
            const bool hot_plane = active == k_part_plane + i;
            ImVec4 fill = ImGui::ColorConvertU32ToFloat4(axis_colour(i, hot_plane));
            fill.w = hot_plane ? 0.32F : 0.14F;
            draw->AddConvexPolyFilled(
                planes[i], 4, ImGui::ColorConvertFloat4ToU32(fill));
            fill.w = hot_plane ? 0.95F : 0.55F;
            draw->AddPolyline(
                planes[i], 4, ImGui::ColorConvertFloat4ToU32(fill),
                ImDrawFlags_Closed, 1.4F);
        }
        for (int i = 0; i < 3; ++i) {
            if (!axis_ok[i]) continue;
            const bool hot_axis = active == k_part_axis + i;
            const ImU32 colour = axis_colour(i, hot_axis);
            draw->AddLine(
                origin_screen, axis_ends[i], colour, hot_axis ? 3.2F : 2.4F);
            draw_arrow_head(draw, axis_ends[i], origin_screen, colour);
        }
        draw->AddCircleFilled(
            origin_screen, 6.F, IM_COL32(245, 245, 245, 240), 16);
        draw->AddCircle(
            origin_screen, 6.F, IM_COL32(20, 22, 28, 200), 16, 1.4F);
    }

    for (int i = 0; i < 6; ++i) {
        if (!face_ok[i]) continue;
        const bool hot_face = active == i;
        const float radius = hot_face ? 8.F : 6.F;
        const ImU32 fill = hot_face
            ? theme::u32(theme::warning, 1.F)
            : theme::u32(theme::warning, 0.82F);
        draw->AddCircleFilled(
            face_screen[i], radius + 1.4F, IM_COL32(20, 22, 28, 220), 12);
        draw->AddCircleFilled(face_screen[i], radius, fill, 12);
    }

    if (active >= 0) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        if (active >= k_part_axis)
            ImGui::SetTooltip(
                "Drag arrows to move the reconstruction region\n"
                "Shift: fine  ·  Ctrl: snap");
        else
            ImGui::SetTooltip(
                "Drag a face handle to resize the reconstruction region\n"
                "Shift: fine  ·  Ctrl: snap");
    }
    return state.box.dragging || active >= 0;
}

}  // namespace

bool draw_viewport_gizmo(
    ViewportGizmoState& state, OrbitCamera& camera, const ImVec2 min,
    const ImVec2 max, ReconstructionBox* region, const bool region_enabled,
    const float scene_radius) {
    if (!state.visible) return false;
    bool captures = false;
    if (region_enabled && region != nullptr)
        captures = draw_region_handles(
            state, *region, camera, min, max, scene_radius);
    else
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
