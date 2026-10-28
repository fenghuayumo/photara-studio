#include "sparse_view.hpp"

#include "imgui_internal.h"
#include "theme.hpp"
#include "camera_timeline.hpp"
#include "camera_path_overlay.hpp"
#include "project/document.hpp"

#include "core/camera_projection.hpp"
#include "io/image.hpp"
#include "mvs/export.hpp"
#include "mvs/mesh_handoff.hpp"
#include "mvs/internal.hpp"
#include "mvs/types.hpp"
#include "parallel/thread_pool.hpp"
#include "project/archive.hpp"
#include "splat/dataset.hpp"
#include "splat/formats.hpp"
#include "splat/visualize.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>
#include <unordered_map>

namespace editor {
namespace {

// SfM reconstructions use a Y-down world, so world "up" is -Y.
constexpr Vec3 k_world_up{0.F, -1.F, 0.F};
constexpr float k_near_plane = 1e-4F;
constexpr float k_far_plane = 1e5F;
// Cap so a multi-million-point dense or splat PLY cannot exhaust host memory.
constexpr std::size_t k_max_loaded_points = 4'000'000;

Vec3 operator-(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 operator+(const Vec3 a, const Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 operator*(const Vec3 a, const float s) {
    return {a.x * s, a.y * s, a.z * s};
}
float dot(const Vec3 a, const Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec3 cross(const Vec3 a, const Vec3 b) {
    return {
        a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}
Vec3 normalize(const Vec3 v) {
    const float length = std::sqrt(dot(v, v));
    return length > 1e-20F ? v * (1.F / length) : Vec3{0.F, 0.F, 1.F};
}

Vec3 orbit_right(const Vec3 forward, const float yaw) {
    const Vec3 lateral = cross(forward, k_world_up);
    if (dot(lateral, lateral) > 1e-8F) return normalize(lateral);
    // At the exact top/bottom views the forward vector is parallel to world
    // up. Preserve a stable screen-right direction from the orbit yaw.
    return {-std::cos(yaw), 0.F, std::sin(yaw)};
}

Vec3 rotate_axis(Vec3 v, Vec3 axis, float angle) {
    axis = normalize(axis);
    return v * std::cos(angle) + cross(axis, v) * std::sin(angle) + axis * (dot(axis, v) * (1.F - std::cos(angle)));
}
Vec3 camera_right(const OrbitCamera& camera, Vec3 forward) {
    const Vec3 right = orbit_right(forward, camera.yaw);
    return right * std::cos(camera.roll) + cross(right, forward) * std::sin(camera.roll);
}
void set_camera_basis(OrbitCamera& camera, Vec3 forward, Vec3 right) {
    forward = normalize(forward);
    camera.pitch = std::asin(std::clamp(forward.y, -1.F, 1.F));
    if (std::hypot(forward.x, forward.z) > 1e-6F)
        camera.yaw = std::atan2(-forward.x, -forward.z);
    const Vec3 level_right = orbit_right(forward, camera.yaw);
    camera.roll = std::atan2(dot(right, cross(level_right, forward)), dot(right, level_right));
}

const char* camera_model_label(const photara::CameraModel model) {
    switch (model) {
        case photara::CameraModel::opencv_fisheye:
            return "OpenCV Fisheye";
        case photara::CameraModel::equirectangular:
            return "Equirectangular";
        default:
            return "Pinhole";
    }
}

photara::CameraModel parse_view_camera_model(const std::string& token) {
    if (token == "opencv_fisheye" || token == "fisheye")
        return photara::CameraModel::opencv_fisheye;
    if (token == "equirectangular" || token == "equirect")
        return photara::CameraModel::equirectangular;
    return photara::CameraModel::pinhole;
}

void apply_camera_model(ViewPose& pose, const photara::CameraModel model) {
    pose.model = model;
    pose.camera_model = camera_model_label(model);
}

constexpr float k_pi = 3.14159265F;

void orbit_intrinsics(
    const OrbitCamera& camera, const std::uint32_t width,
    const std::uint32_t height, float& fx, float& fy, float& cx, float& cy,
    photara::CameraModel& model, float& k1, float& k2, float& k3,
    float& k4) {
    const float w = static_cast<float>(std::max<std::uint32_t>(1, width));
    const float h = static_cast<float>(std::max<std::uint32_t>(1, height));
    cx = 0.5F * w;
    cy = 0.5F * h;
    k1 = k2 = k3 = k4 = 0.F;
    model = photara::CameraModel::pinhole;
    if (camera.projection == EditorProjection::panorama) {
        model = photara::CameraModel::equirectangular;
        fx = fy = w / (2.F * k_pi);
        return;
    }
    if (camera.projection == EditorProjection::fisheye) {
        model = photara::CameraModel::opencv_fisheye;
        k1 = camera.fisheye_k1;
        const float theta = std::clamp(camera.fov_degrees, 20.F, 179.F) * 0.5F *
                            k_pi / 180.F;
        const float t2 = theta * theta;
        const float theta_d = theta * (1.F + t2 * k1);
        fy = (h * 0.5F) / std::max(1e-4F, theta_d);
        fx = fy;
        return;
    }
    if (camera.projection == EditorProjection::orthographic) {
        const float view_h = std::max(1e-4F, camera.ortho_height);
        fy = h / view_h;
        fx = fy;
        return;
    }
    const float half = std::clamp(camera.fov_degrees, 10.F, 170.F) * 0.5F *
                       k_pi / 180.F;
    fy = h * 0.5F / std::max(1e-4F, std::tan(half));
    fx = fy;
}

// A camera-space basis plus the pixel focal length for the current viewport.
struct ViewFrame {
    Vec3 eye;
    Vec3 right;
    Vec3 up;
    Vec3 forward;
    float focal{};
    float half_x{};
    float half_y{};
    ImVec2 min{};
    ImVec2 centre;
    float fx{1.F};
    float fy{1.F};
    float cx{};
    float cy{};
    float k1{};
    float k2{};
    float k3{};
    float k4{};
    float width{1.F};
    float height{1.F};
    EditorProjection projection{EditorProjection::perspective};
    photara::CameraModel model{photara::CameraModel::pinhole};
    float ortho_height{4.F};
};

ViewFrame build_frame(
    const OrbitCamera& camera, const ImVec2 min, const ImVec2 max) {
    ViewFrame frame;
    const float pitch = std::clamp(camera.pitch, -k_pi * 0.5F, k_pi * 0.5F);
    const Vec3 offset{
        std::cos(pitch) * std::sin(camera.yaw), -std::sin(pitch),
        std::cos(pitch) * std::cos(camera.yaw)};
    frame.eye = camera.target + offset * camera.distance;
    frame.forward = normalize(camera.target - frame.eye);
    frame.right = camera_right(camera, frame.forward);
    frame.up = cross(frame.right, frame.forward);
    const float height = std::max(1.F, max.y - min.y);
    const float width = std::max(1.F, max.x - min.x);
    frame.min = min;
    frame.width = width;
    frame.height = height;
    frame.projection = camera.projection;
    frame.ortho_height = std::max(1e-4F, camera.ortho_height);
    orbit_intrinsics(
        camera, static_cast<std::uint32_t>(std::lround(width)),
        static_cast<std::uint32_t>(std::lround(height)), frame.fx, frame.fy,
        frame.cx, frame.cy, frame.model, frame.k1, frame.k2, frame.k3,
        frame.k4);
    frame.focal = frame.fy;
    frame.half_x = width * 0.5F;
    frame.half_y = height * 0.5F;
    frame.centre = {(min.x + max.x) * 0.5F, (min.y + max.y) * 0.5F};
    return frame;
}

// Returns false when the point sits at or behind the near plane.
bool project(
    const ViewFrame& frame, const Vec3 world, ImVec2& screen, float& depth) {
    const Vec3 relative = world - frame.eye;
    const float x = dot(relative, frame.right);
    const float y = -dot(relative, frame.up);
    depth = dot(relative, frame.forward);
    if (frame.projection == EditorProjection::panorama) {
        const auto pixel = photara::project_equirectangular_camera(
            x, y, depth, static_cast<int>(std::lround(frame.width)),
            static_cast<int>(std::lround(frame.height)));
        if (!pixel.valid) return false;
        screen = {frame.min.x + static_cast<float>(pixel.u),
                  frame.min.y + static_cast<float>(pixel.v)};
        depth = std::sqrt(x * x + y * y + depth * depth);
        return true;
    }
    if (depth <= k_near_plane) return false;
    if (frame.projection == EditorProjection::orthographic) {
        const float scale = frame.height / frame.ortho_height;
        screen.x = frame.centre.x + x * scale;
        screen.y = frame.centre.y + y * scale;
        return true;
    }
    if (frame.projection == EditorProjection::fisheye) {
        const auto pixel = photara::project_fisheye_camera(
            x, y, depth, frame.fx, frame.fy, frame.cx, frame.cy, frame.k1,
            frame.k2, frame.k3, frame.k4);
        if (!pixel.valid) return false;
        screen = {frame.min.x + static_cast<float>(pixel.u),
                  frame.min.y + static_cast<float>(pixel.v)};
        return true;
    }
    const float inverse = frame.focal / depth;
    screen.x = frame.centre.x + x * inverse;
    screen.y = frame.centre.y + y * inverse;
    return true;
}

// Keeps the parameter interval [t0, t1] of P(t)=P0+t(P1-P0) inside t*den >= num.
bool keep_halfspace(const float num, const float den, float& t0, float& t1) {
    constexpr float eps = 1e-12F;
    if (std::abs(den) < eps) return num <= 0.F;
    const float t = num / den;
    if (den > 0.F) {
        if (t > t1) return false;
        t0 = std::max(t0, t);
    } else {
        if (t < t0) return false;
        t1 = std::min(t1, t);
    }
    return t0 <= t1;
}

// Clips the segment to the view frustum so grazing lines near the camera
// cannot explode to huge screen coordinates and streak across the viewport.
void draw_segment(
    ImDrawList* draw, const ViewFrame& frame, const Vec3 a, const Vec3 b,
    const ImU32 colour, const float thickness = 1.F) {
    if (frame.projection == EditorProjection::fisheye ||
        frame.projection == EditorProjection::panorama) {
        // Curved viewport projections need subdivision; never connect across
        // the panorama seam with a screen-wide line.
        for (int i = 0; i < 8; ++i) {
            ImVec2 p, q;
            float dp{}, dq{};
            if (project(frame, a + (b - a) * (i / 8.F), p, dp) &&
                project(frame, a + (b - a) * ((i + 1) / 8.F), q, dq) &&
                (frame.projection != EditorProjection::panorama ||
                 std::abs(p.x - q.x) <= frame.width * 0.5F))
                draw->AddLine(p, q, colour, thickness);
        }
        return;
    }
    const Vec3 rel_a = a - frame.eye;
    const Vec3 rel_b = b - frame.eye;
    const float ax = dot(rel_a, frame.right);
    const float ay = dot(rel_a, frame.up);
    const float az = dot(rel_a, frame.forward);
    const float bx = dot(rel_b, frame.right);
    const float by = dot(rel_b, frame.up);
    const float bz = dot(rel_b, frame.forward);
    const float dx = bx - ax;
    const float dy = by - ay;
    const float dz = bz - az;
    float t0 = 0.F;
    float t1 = 1.F;
    if (frame.projection == EditorProjection::orthographic) {
        if (!keep_halfspace(k_near_plane - az, dz, t0, t1)) return;
        if (!keep_halfspace(az - k_far_plane, -dz, t0, t1)) return;
        const float scale = frame.height / frame.ortho_height;
        const auto to_screen = [&](const float t) {
            return ImVec2{
                frame.centre.x + (ax + dx * t) * scale,
                frame.centre.y - (ay + dy * t) * scale};
        };
        draw->AddLine(to_screen(t0), to_screen(t1), colour, thickness);
        return;
    }
    if (!keep_halfspace(k_near_plane - az, dz, t0, t1)) return;
    if (!keep_halfspace(az - k_far_plane, -dz, t0, t1)) return;
    const float pad = 1.15F;
    const float tx = frame.half_x * pad / std::max(1e-4F, frame.focal);
    const float ty = frame.half_y * pad / std::max(1e-4F, frame.focal);
    if (!keep_halfspace(ax - tx * az, tx * dz - dx, t0, t1)) return;
    if (!keep_halfspace(-(ax + tx * az), dx + tx * dz, t0, t1)) return;
    if (!keep_halfspace(ay - ty * az, ty * dz - dy, t0, t1)) return;
    if (!keep_halfspace(-(ay + ty * az), dy + ty * dz, t0, t1)) return;

    const float cx0 = ax + dx * t0;
    const float cy0 = ay + dy * t0;
    const float cz0 = az + dz * t0;
    const float cx1 = ax + dx * t1;
    const float cy1 = ay + dy * t1;
    const float cz1 = az + dz * t1;
    if (cz0 <= k_near_plane || cz1 <= k_near_plane) return;
    const ImVec2 screen_a{
        frame.centre.x + cx0 * frame.focal / cz0,
        frame.centre.y - cy0 * frame.focal / cz0};
    const ImVec2 screen_b{
        frame.centre.x + cx1 * frame.focal / cz1,
        frame.centre.y - cy1 * frame.focal / cz1};
    draw->AddLine(screen_a, screen_b, colour, thickness);
}

Vec3 marker_to_world(const ViewPose& pose, const Vec3 local, const float size) {
    const auto& r = pose.rotation;
    return {pose.centre.x + size * (r[0] * local.x + r[3] * local.y + r[6] * local.z),
            pose.centre.y + size * (r[1] * local.x + r[4] * local.y + r[7] * local.z),
            pose.centre.z + size * (r[2] * local.x + r[5] * local.y + r[8] * local.z)};
}

struct MarkerIntrinsics {
    float fx, fy, cx, cy;
    std::uint32_t width, height;
};

MarkerIntrinsics marker_intrinsics(const ViewPose& pose) {
    if (pose.src_width && pose.src_height)
        return {pose.src_fx, pose.src_fy, pose.src_cx, pose.src_cy, pose.src_width, pose.src_height};
    return {pose.fx, pose.fy, pose.cx >= 0 ? pose.cx : pose.width * 0.5F,
            pose.cy >= 0 ? pose.cy : pose.height * 0.5F, pose.width, pose.height};
}

std::array<std::uint32_t, 17> marker_calibration(const ViewPose& pose) {
    return {static_cast<std::uint32_t>(pose.model), pose.width, pose.height,
        std::bit_cast<std::uint32_t>(pose.fx), std::bit_cast<std::uint32_t>(pose.fy),
        std::bit_cast<std::uint32_t>(pose.cx), std::bit_cast<std::uint32_t>(pose.cy),
        std::bit_cast<std::uint32_t>(pose.k1), std::bit_cast<std::uint32_t>(pose.k2),
        std::bit_cast<std::uint32_t>(pose.k3), std::bit_cast<std::uint32_t>(pose.k4),
        pose.src_width, pose.src_height,
        std::bit_cast<std::uint32_t>(pose.src_fx), std::bit_cast<std::uint32_t>(pose.src_fy),
        std::bit_cast<std::uint32_t>(pose.src_cx), std::bit_cast<std::uint32_t>(pose.src_cy)};
}

void image_plane_corners(
    const ViewPose& pose, const float length, Vec3 corners[4]) {
    const float half_x = pose.fx > 1e-3F && pose.width > 0
        ? length * (pose.width * 0.5F) / pose.fx
        : length * 0.5F;
    const float half_y = pose.fy > 1e-3F && pose.height > 0
        ? length * (pose.height * 0.5F) / pose.fy
        : length * 0.35F;
    const float offset_x = pose.cx >= 0.F && pose.fx > 1e-3F
        ? length * (pose.width * 0.5F - pose.cx) / pose.fx : 0.F;
    const float offset_y = pose.cy >= 0.F && pose.fy > 1e-3F
        ? length * (pose.height * 0.5F - pose.cy) / pose.fy : 0.F;
    corners[0] = marker_to_world(pose, {-half_x + offset_x, -half_y + offset_y, length}, 1.F);
    corners[1] = marker_to_world(pose, {half_x + offset_x, -half_y + offset_y, length}, 1.F);
    corners[2] = marker_to_world(pose, {half_x + offset_x, half_y + offset_y, length}, 1.F);
    corners[3] = marker_to_world(pose, {-half_x + offset_x, half_y + offset_y, length}, 1.F);
}

bool point_in_convex_quad(const ImVec2 point, const ImVec2 quad[4]) {
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

// Lower is better. The image-plane quad is only hittable when the frustum is
// actually drawn, so undrawn cameras do not steal clicks through invisible
// photo planes. Apexes stay hittable so a camera on the trajectory can be
// selected even when it is not in the sampled marker set.
float view_pick_score(
    const ViewFrame& frame, const Vec3& apex, const Vec3 corners[4],
    const ImVec2 mouse, const bool include_image_plane) {
    float score = 1e9F;
    ImVec2 apex_screen;
    float apex_depth{};
    if (project(frame, apex, apex_screen, apex_depth)) {
        const float dx = apex_screen.x - mouse.x;
        const float dy = apex_screen.y - mouse.y;
        score = std::sqrt(dx * dx + dy * dy);
    }
    if (!include_image_plane) return score;
    ImVec2 corner_screen[4];
    bool plane_visible = true;
    for (int i = 0; i < 4; ++i) {
        float corner_depth{};
        plane_visible &=
            project(frame, corners[i], corner_screen[i], corner_depth);
    }
    if (plane_visible && point_in_convex_quad(mouse, corner_screen))
        score = std::min(score, 4.F);
    return score;
}

void draw_curved_camera_marker(
    ImDrawList* draw, const ViewFrame& frame, const ViewPose& pose, const float size,
    const CameraMarkerGeometry& mesh, const ImTextureID photo, const ImU32 colour,
    const bool active, const bool hovered) {
    struct Vertex { ImVec2 pixel; float depth{}; bool visible{}; };
    std::vector<Vertex> projected;
    // Only focused markers need photo tessellation; the default view draws
    // the cached sparse wire grid and leaves the scene unobscured.
    if (active) {
        projected.resize(mesh.vertices.size());
        for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
            auto& v = projected[i];
            v.visible = project(frame, marker_to_world(pose, mesh.vertices[i].position, size), v.pixel, v.depth);
        }
        struct Face { std::array<std::uint32_t,3> ids; float depth; };
        std::vector<Face> faces;
        faces.reserve(mesh.triangles.size());
        for (const auto& ids : mesh.triangles) {
            const auto& a = projected[ids[0]]; const auto& b = projected[ids[1]]; const auto& c = projected[ids[2]];
            if (!a.visible || !b.visible || !c.visible) continue;
            if (frame.projection == EditorProjection::panorama &&
                std::max({a.pixel.x,b.pixel.x,c.pixel.x}) - std::min({a.pixel.x,b.pixel.x,c.pixel.x}) > frame.width*0.5F)
                continue;
            faces.push_back({ids,(a.depth+b.depth+c.depth)/3.F});
        }
        std::sort(faces.begin(), faces.end(), [](const Face& a, const Face& b) { return a.depth > b.depth; });
        if (photo) draw->PushTextureID(photo);
        const bool inside = dot(frame.eye-pose.centre, frame.eye-pose.centre) < size*size;
        for (const auto& face : faces) {
            const auto& ids = face.ids;
            const Vec3 normal = (mesh.vertices[ids[0]].position + mesh.vertices[ids[1]].position +
                                 mesh.vertices[ids[2]].position) * (1.F/3.F);
            const Vec3 world = marker_to_world(pose, normal, size);
            const bool front = inside || dot(world-pose.centre, frame.eye-world) >= 0;
            const ImU32 tint = photo ? IM_COL32(255,255,255,front ? (hovered ? 215 : 180) : 36)
                : ((colour & ~IM_COL32_A_MASK) | IM_COL32(0,0,0,front ? 14 : 5));
            if (photo) {
                draw->PrimReserve(3,3);
                for (const auto id : ids) {
                    const auto uv = mesh.vertices[id].uv;
                    draw->PrimVtx(projected[id].pixel,{uv.x,uv.y},tint);
                }
            } else {
                draw->AddTriangleFilled(projected[ids[0]].pixel, projected[ids[1]].pixel,
                                        projected[ids[2]].pixel, tint);
            }
        }
        if (photo) draw->PopTextureID();
    }
    const float thickness = active ? 1.25F : 0.7F;
    for (const auto& line : mesh.lines)
        draw_segment(draw,frame,marker_to_world(pose,mesh.vertices[line[0]].position,size),
                     marker_to_world(pose,mesh.vertices[line[1]].position,size),colour,thickness);
    for (const auto id : mesh.rays)
        draw_segment(draw,frame,pose.centre,marker_to_world(pose,mesh.vertices[id].position,size),colour,thickness);
    // A panorama has no forward frustum. This arrow identifies its zero-yaw
    // direction; the separate up axis makes capture orientation unambiguous.
    const auto world = [&](Vec3 p) { return marker_to_world(pose,p,size); };
    draw_segment(draw,frame,pose.centre,world({0,0,1.35F}),colour,thickness);
    draw_segment(draw,frame,world({-.12F,0,1.12F}),world({0,0,1.35F}),colour,thickness);
    draw_segment(draw,frame,world({.12F,0,1.12F}),world({0,0,1.35F}),colour,thickness);
    ImVec2 centre;
    float depth{};
    if (project(frame,pose.centre,centre,depth)) {
        draw->AddCircleFilled(centre,active ? 3.5F : 1.5F,colour);
        if (active) {
            draw_segment(draw,frame,pose.centre,world({0,-1.25F,0}),colour,thickness);
            draw->AddText({centre.x+7.F,centre.y-16.F},colour,
                pose.model == photara::CameraModel::equirectangular ? "360" : "Fisheye");
        }
    }
}

float snap_grid_cell(const float desired) {
    const float value = std::max(desired, 1e-8F);
    const float exponent = std::floor(std::log10(value));
    const float base = std::pow(10.F, exponent);
    const float mantissa = value / base;
    if (mantissa < 1.5F) return base;
    if (mantissa < 3.5F) return 2.F * base;
    if (mantissa < 7.5F) return 5.F * base;
    return 10.F * base;
}

bool is_multiple(const float value, const float step) {
    if (step <= 1e-20F) return false;
    const float scaled = value / step;
    return std::abs(scaled - std::round(scaled)) < 1e-4F;
}

float grid_horizon_extent(
    const ViewFrame& frame, const OrbitCamera& camera, const float plane_y,
    const float focus_x, const float focus_z) {
    float extent = camera.projection == EditorProjection::orthographic
        ? camera.ortho_height * 4.F
        : camera.distance * 4.F;
    const float ortho_scale = frame.height / frame.ortho_height;
    const float xs[3] = {-frame.half_x, 0.F, frame.half_x};
    const float ys[3] = {-frame.half_y, 0.F, frame.half_y};
    for (const float sx : xs) {
        for (const float sy : ys) {
            Vec3 origin = frame.eye;
            Vec3 dir = normalize(
                frame.forward * frame.focal + frame.right * sx + frame.up * sy);
            if (camera.projection == EditorProjection::orthographic) {
                origin = frame.eye + frame.right * (sx / ortho_scale) +
                         frame.up * (sy / ortho_scale);
                dir = frame.forward;
            }
            if (std::abs(dir.y) < 1e-6F) continue;
            const float t = (plane_y - origin.y) / dir.y;
            if (t <= k_near_plane) continue;
            const Vec3 hit = origin + dir * t;
            const float dx = hit.x - focus_x;
            const float dz = hit.z - focus_z;
            extent = std::max(extent, std::sqrt(dx * dx + dz * dz));
        }
    }
    return extent * 1.25F;
}

// World-locked XZ ground grid. Dense cells around the look-at, major lines
// continuing to the horizon so the upper view is ground, not an empty void.
void draw_ground_grid(
    ImDrawList* draw, const ViewFrame& frame, const OrbitCamera& camera,
    const float plane_y) {
    const float world_per_pixel =
        camera.distance / std::max(1.F, frame.focal);
    const float minor = snap_grid_cell(world_per_pixel * 40.F);
    const float major = minor * 10.F;
    const float focus_x = camera.target.x;
    const float focus_z = camera.target.z;
    const float horizon = grid_horizon_extent(
        frame, camera, plane_y, focus_x, focus_z);
    constexpr int k_minor_cells = 36;
    const float inner = minor * static_cast<float>(k_minor_cells);
    const float extent =
        std::min(std::max(horizon, inner), major * 40.F);
    const float origin_x = std::floor(focus_x / minor) * minor;
    const float origin_z = std::floor(focus_z / minor) * minor;

    auto radial_fade = [&](const float x, const float z) {
        const float nx = (x - focus_x) / extent;
        const float nz = (z - focus_z) / extent;
        return std::clamp(1.F - nx * nx - nz * nz, 0.F, 1.F);
    };

    auto draw_faded_line = [&](
        const Vec3 start, const Vec3 end, const ImVec4& rgb, const float alpha,
        const float thickness, const int segments) {
        for (int s = 0; s < segments; ++s) {
            const float u0 = static_cast<float>(s) / segments;
            const float u1 = static_cast<float>(s + 1) / segments;
            const Vec3 p0 = start + (end - start) * u0;
            const Vec3 p1 = start + (end - start) * u1;
            const float fade =
                radial_fade((p0.x + p1.x) * 0.5F, (p0.z + p1.z) * 0.5F);
            if (fade * alpha < 0.02F) continue;
            draw_segment(
                draw, frame, p0, p1, theme::u32(rgb, alpha * fade), thickness);
        }
    };

    const ImVec4 minor_rgb{0.30F, 0.33F, 0.38F, 1.F};
    const ImVec4 major_rgb{0.42F, 0.45F, 0.52F, 1.F};
    const ImVec4 axis_x{0.89F, 0.32F, 0.32F, 1.F};
    const ImVec4 axis_z{0.31F, 0.60F, 0.92F, 1.F};

    auto draw_axis_family = [&](const bool along_z) {
        const int count = std::min(
            240, static_cast<int>(std::ceil(extent / minor)));
        const float snapped = along_z ? origin_x : origin_z;
        for (int i = -count; i <= count; ++i) {
            const float coord = snapped + static_cast<float>(i) * minor;
            const float delta = along_z ? coord - focus_x : coord - focus_z;
            if (std::abs(delta) > extent) continue;
            const float span = std::sqrt(
                std::max(0.F, extent * extent - delta * delta));
            const bool origin = std::abs(coord) <= minor * 0.25F;
            const bool major_line = origin || is_multiple(coord, major);
            const bool inner_minor = std::abs(delta) <= inner + minor;
            if (!major_line && !inner_minor) continue;
            const ImVec4 rgb = origin ? (along_z ? axis_z : axis_x)
                                      : (major_line ? major_rgb : minor_rgb);
            const float alpha = origin ? 0.72F : (major_line ? 0.40F : 0.18F);
            const int segments = major_line ? 8 : 4;
            if (along_z) {
                draw_faded_line(
                    {coord, plane_y, focus_z - span},
                    {coord, plane_y, focus_z + span}, rgb, alpha,
                    origin ? 1.6F : 1.F, segments);
            } else {
                draw_faded_line(
                    {focus_x - span, plane_y, coord},
                    {focus_x + span, plane_y, coord}, rgb, alpha,
                    origin ? 1.6F : 1.F, segments);
            }
        }
    };

    draw_axis_family(true);
    draw_axis_family(false);
}

ImU32 depth_ramp(const float t) {
    // Cool teal in the distance to a warm near highlight; reads well on the
    // dark viewport and gives the flat SfM cloud a sense of depth.
    const float clamped = std::clamp(t, 0.F, 1.F);
    const ImVec4 far{0.20F, 0.38F, 0.62F, 1.F};
    const ImVec4 mid{0.30F, 0.74F, 0.86F, 1.F};
    const ImVec4 near{0.94F, 0.95F, 0.88F, 1.F};
    ImVec4 colour;
    if (clamped < 0.5F) {
        const float local = clamped * 2.F;
        colour = {
            far.x + (mid.x - far.x) * local, far.y + (mid.y - far.y) * local,
            far.z + (mid.z - far.z) * local, 1.F};
    } else {
        const float local = (clamped - 0.5F) * 2.F;
        colour = {
            mid.x + (near.x - mid.x) * local, mid.y + (near.y - mid.y) * local,
            mid.z + (near.z - mid.z) * local, 1.F};
    }
    return ImGui::ColorConvertFloat4ToU32(colour);
}

void collect_cloud_points(
    const SparseScene& scene, const ViewFrame& frame,
    const ViewOptions& options, std::vector<ProjectedCloudPoint>& points) {
    points.clear();
    if (!options.show_cloud || scene.points.empty()) return;
    const std::size_t stride = std::max<std::size_t>(
        1, scene.points.size() / std::max(1, options.point_budget) + 1);
    points.reserve(scene.points.size() / stride + 1);
    const bool source_colours = !options.colour_by_depth &&
        scene.colours.size() == scene.points.size();
    float near_depth = std::numeric_limits<float>::max(), far_depth = 0.F;
    for (std::size_t index = 0; index < scene.points.size(); index += stride) {
        ImVec2 screen;
        float depth{};
        if (!project(frame, scene.world_point(index), screen, depth) ||
            !std::isfinite(screen.x) || !std::isfinite(screen.y) || !std::isfinite(depth) ||
            screen.x < frame.min.x || screen.x > frame.min.x + frame.width ||
            screen.y < frame.min.y || screen.y > frame.min.y + frame.height)
            continue;
        points.push_back({screen.x, screen.y, depth,
            source_colours ? scene.colours[index] : 0U});
        near_depth = std::min(near_depth, depth);
        far_depth = std::max(far_depth, depth);
    }
    if (!source_colours) {
        const float span = std::max(1e-6F, far_depth - near_depth);
        for (auto& point : points)
            point.colour = depth_ramp(1.F - (point.depth - near_depth) / span);
    }
}

struct ProjectedRing {
    ImVec2 centre;
    ImVec2 radius;
    float rotation{};
};

bool project_gaussian_ring(
    const ViewFrame& frame, const Vec3 mean, const GaussianPrimitive& gaussian,
    const float ring_scale, ProjectedRing& ring) {
    const Vec3 relative = mean - frame.eye;
    const float cam_x = dot(relative, frame.right);
    const float cam_y = dot(relative, frame.up);
    const float cam_z = dot(relative, frame.forward);
    if (cam_z <= k_near_plane) return false;

    float qw = gaussian.rotation[0];
    float qx = gaussian.rotation[1];
    float qy = gaussian.rotation[2];
    float qz = gaussian.rotation[3];
    const float qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    if (qn < 1e-8F) return false;
    qw /= qn;
    qx /= qn;
    qy /= qn;
    qz /= qn;
    const float xx = qx * qx;
    const float yy = qy * qy;
    const float zz = qz * qz;
    const float xy = qx * qy;
    const float xz = qx * qz;
    const float yz = qy * qz;
    const float wx = qw * qx;
    const float wy = qw * qy;
    const float wz = qw * qz;
    const Vec3 axis_x{
        (1.F - 2.F * (yy + zz)) * gaussian.scale.x,
        (2.F * (xy + wz)) * gaussian.scale.x,
        (2.F * (xz - wy)) * gaussian.scale.x};
    const Vec3 axis_y{
        (2.F * (xy - wz)) * gaussian.scale.y,
        (1.F - 2.F * (xx + zz)) * gaussian.scale.y,
        (2.F * (yz + wx)) * gaussian.scale.y};
    const Vec3 axis_z{
        (2.F * (xz + wy)) * gaussian.scale.z,
        (2.F * (yz - wx)) * gaussian.scale.z,
        (1.F - 2.F * (xx + yy)) * gaussian.scale.z};
    const Vec3 cam_ax{
        dot(axis_x, frame.right), dot(axis_x, frame.up),
        dot(axis_x, frame.forward)};
    const Vec3 cam_ay{
        dot(axis_y, frame.right), dot(axis_y, frame.up),
        dot(axis_y, frame.forward)};
    const Vec3 cam_az{
        dot(axis_z, frame.right), dot(axis_z, frame.up),
        dot(axis_z, frame.forward)};

    const float inv_z = 1.F / cam_z;
    const float inv_z2 = inv_z * inv_z;
    const float jx_x = frame.focal * inv_z;
    const float jx_z = -frame.focal * cam_x * inv_z2;
    const float jy_y = -frame.focal * inv_z;
    const float jy_z = frame.focal * cam_y * inv_z2;
    const auto project_axis = [&](const Vec3& axis) {
        return ImVec2{
            jx_x * axis.x + jx_z * axis.z, jy_y * axis.y + jy_z * axis.z};
    };
    const ImVec2 p0 = project_axis(cam_ax);
    const ImVec2 p1 = project_axis(cam_ay);
    const ImVec2 p2 = project_axis(cam_az);
    const float a = p0.x * p0.x + p1.x * p1.x + p2.x * p2.x;
    const float b = p0.x * p0.y + p1.x * p1.y + p2.x * p2.y;
    const float c = p0.y * p0.y + p1.y * p1.y + p2.y * p2.y;
    const float a_d = a + 0.3F;
    const float c_d = c + 0.3F;
    const float mid = 0.5F * (a_d + c_d);
    const float ext =
        0.5F * std::sqrt(std::max(0.F, (a_d - c_d) * (a_d - c_d) + 4.F * b * b));
    const float lambda0 = std::max(0.F, mid + ext);
    const float lambda1 = std::max(0.1F, mid - ext);
    float rx = ring_scale * std::sqrt(lambda0);
    float ry = ring_scale * std::sqrt(lambda1);
    const float cap = std::min(1024.F, std::min(frame.width, frame.height));
    const float major = std::max(rx, ry);
    if (major > cap && major > 1e-4F) {
        const float fit = cap / major;
        rx *= fit;
        ry *= fit;
    }
    ring.centre = {
        frame.centre.x + cam_x * frame.focal * inv_z,
        frame.centre.y - cam_y * frame.focal * inv_z};
    ring.radius = {rx, ry};
    ring.rotation = 0.5F * std::atan2(2.F * b, a - c);
    return major >= 0.75F;
}

std::size_t draw_gaussian_rings(
    ImDrawList* draw, const ViewFrame& frame, const SparseScene& scene,
    const ViewOptions& options, const ImVec2 min, const ImVec2 max) {
    const std::size_t count = scene.points.size();
    const std::size_t budget = static_cast<std::size_t>(
        std::max(1, options.ring_budget));
    const std::size_t stride =
        std::max<std::size_t>(1, count / budget + 1);
    const bool source_colours =
        !options.colour_by_depth && !scene.colours.empty();
    std::size_t drawn = 0;
    for (std::size_t i = 0; i < count; i += stride) {
        ProjectedRing ring;
        if (!project_gaussian_ring(
                frame, scene.world_point(i), scene.world_gaussian(i), options.ring_scale,
                ring))
            continue;
        if (ring.centre.x + ring.radius.x < min.x ||
            ring.centre.x - ring.radius.x > max.x ||
            ring.centre.y + ring.radius.y < min.y ||
            ring.centre.y - ring.radius.y > max.y)
            continue;
        const ImU32 fill = source_colours
            ? (scene.colours[i] & 0x00FFFFFF) | IM_COL32(0, 0, 0, 40)
            : theme::u32(theme::accent, 0.16F);
        const ImU32 colour = source_colours
            ? (scene.colours[i] & 0x00FFFFFF) | IM_COL32(0, 0, 0, 170)
            : theme::u32(theme::accent, 0.72F);
        const float radius = std::max(ring.radius.x, ring.radius.y);
        const int segments = std::clamp(static_cast<int>(radius * 0.65F), 24, 72);
        const float thickness = std::clamp(radius * 0.045F, 1.25F, 3.5F);
        draw->AddEllipseFilled(
            ring.centre, ring.radius, fill, ring.rotation, segments);
        draw->AddEllipse(
            ring.centre, ring.radius, colour, ring.rotation, segments, thickness);
        ++drawn;
    }
    return drawn;
}

// ---------------------------------------------------------------------------
// PLY parsing

enum class PlyType { unknown, i8, u8, i16, u16, i32, u32, f32, f64 };

PlyType parse_ply_type(const std::string& name) {
    if (name == "char" || name == "int8") return PlyType::i8;
    if (name == "uchar" || name == "uint8") return PlyType::u8;
    if (name == "short" || name == "int16") return PlyType::i16;
    if (name == "ushort" || name == "uint16") return PlyType::u16;
    if (name == "int" || name == "int32") return PlyType::i32;
    if (name == "uint" || name == "uint32") return PlyType::u32;
    if (name == "float" || name == "float32") return PlyType::f32;
    if (name == "double" || name == "float64") return PlyType::f64;
    return PlyType::unknown;
}

std::size_t ply_type_size(const PlyType type) {
    switch (type) {
        case PlyType::i8:
        case PlyType::u8: return 1;
        case PlyType::i16:
        case PlyType::u16: return 2;
        case PlyType::i32:
        case PlyType::u32:
        case PlyType::f32: return 4;
        case PlyType::f64: return 8;
        default: return 0;
    }
}

struct PlyProperty {
    std::string name;
    PlyType type{PlyType::unknown};
    std::size_t offset{};
};

double read_ply_value(
    const char* data, const PlyType type) {
    switch (type) {
        case PlyType::i8: {
            std::int8_t value{};
            std::memcpy(&value, data, 1);
            return value;
        }
        case PlyType::u8: {
            std::uint8_t value{};
            std::memcpy(&value, data, 1);
            return value;
        }
        case PlyType::i16: {
            std::int16_t value{};
            std::memcpy(&value, data, 2);
            return value;
        }
        case PlyType::u16: {
            std::uint16_t value{};
            std::memcpy(&value, data, 2);
            return value;
        }
        case PlyType::i32: {
            std::int32_t value{};
            std::memcpy(&value, data, 4);
            return value;
        }
        case PlyType::u32: {
            std::uint32_t value{};
            std::memcpy(&value, data, 4);
            return value;
        }
        case PlyType::f32: {
            float value{};
            std::memcpy(&value, data, 4);
            return value;
        }
        case PlyType::f64: {
            double value{};
            std::memcpy(&value, data, 8);
            return value;
        }
        default: return 0.0;
    }
}

const PlyProperty* find_property(
    const std::vector<PlyProperty>& properties,
    const std::initializer_list<const char*> names) {
    for (const char* name : names)
        for (const auto& property : properties)
            if (property.name == name) return &property;
    return nullptr;
}

std::string load_ply(const std::filesystem::path& path, SparseScene& scene) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        const auto name = path.filename().u8string();
        return "Cannot open " + std::string(name.begin(), name.end());
    }

    std::string line;
    if (!std::getline(input, line)) return "Empty PLY";
    if (line.rfind("ply", 0) != 0) return "Not a PLY file";

    bool binary = false;
    bool big_endian = false;
    std::string element;
    std::size_t vertex_count = 0;
    std::vector<PlyProperty> vertex_properties;
    std::size_t vertex_stride = 0;
    bool header_done = false;

    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;
        if (keyword == "format") {
            std::string format;
            tokens >> format;
            binary = format != "ascii";
            big_endian = format == "binary_big_endian";
        } else if (keyword == "element") {
            std::string name;
            std::size_t count = 0;
            tokens >> name >> count;
            element = name;
            if (name == "vertex") vertex_count = count;
        } else if (keyword == "property" && element == "vertex") {
            std::string type_name;
            std::string name;
            tokens >> type_name;
            if (type_name == "list") continue;
            tokens >> name;
            PlyProperty property;
            property.name = name;
            property.type = parse_ply_type(type_name);
            property.offset = vertex_stride;
            const std::size_t size = ply_type_size(property.type);
            if (size == 0) return "Unsupported PLY property type: " + type_name;
            vertex_stride += size;
            vertex_properties.push_back(property);
        } else if (keyword == "end_header") {
            header_done = true;
            break;
        }
    }
    if (!header_done) return "Truncated PLY header";
    if (big_endian) return "Big-endian PLY is not supported";
    if (vertex_count == 0) return "PLY contains no vertices";

    const PlyProperty* px = find_property(vertex_properties, {"x"});
    const PlyProperty* py = find_property(vertex_properties, {"y"});
    const PlyProperty* pz = find_property(vertex_properties, {"z"});
    if (!px || !py || !pz) return "PLY has no x/y/z properties";
    const PlyProperty* pr = find_property(
        vertex_properties, {"red", "r", "diffuse_red"});
    const PlyProperty* pg = find_property(
        vertex_properties, {"green", "g", "diffuse_green"});
    const PlyProperty* pb = find_property(
        vertex_properties, {"blue", "b", "diffuse_blue"});
    const bool has_colour = pr && pg && pb;

    const std::size_t stride = std::max<std::size_t>(
        1, (vertex_count + k_max_loaded_points - 1) / k_max_loaded_points);
    const std::size_t reserve = vertex_count / stride + 1;
    scene.points.clear();
    scene.colours.clear();
    scene.points.reserve(reserve);
    if (has_colour) scene.colours.reserve(reserve);

    const auto push = [&](const double x, const double y, const double z,
                          const double r, const double g, const double b) {
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return;
        scene.points.push_back(
            {static_cast<float>(x), static_cast<float>(y),
             static_cast<float>(z)});
        if (has_colour) {
            const auto channel = [](const double value) {
                return static_cast<int>(
                    std::clamp(value <= 1.0 ? value * 255.0 : value, 0.0, 255.0));
            };
            scene.colours.push_back(IM_COL32(
                channel(r), channel(g), channel(b), 255));
        }
    };

    if (binary) {
        std::vector<char> row(vertex_stride);
        for (std::size_t i = 0; i < vertex_count; ++i) {
            if (!input.read(row.data(), static_cast<std::streamsize>(vertex_stride)))
                break;
            if (i % stride != 0) continue;
            push(
                read_ply_value(row.data() + px->offset, px->type),
                read_ply_value(row.data() + py->offset, py->type),
                read_ply_value(row.data() + pz->offset, pz->type),
                has_colour ? read_ply_value(row.data() + pr->offset, pr->type) : 0.0,
                has_colour ? read_ply_value(row.data() + pg->offset, pg->type) : 0.0,
                has_colour ? read_ply_value(row.data() + pb->offset, pb->type) : 0.0);
        }
    } else {
        // ASCII rows are whitespace-separated in declared property order.
        std::vector<double> values(vertex_properties.size());
        const auto index_of = [&](const PlyProperty* property) {
            return static_cast<std::size_t>(property - vertex_properties.data());
        };
        for (std::size_t i = 0; i < vertex_count; ++i) {
            if (!std::getline(input, line)) break;
            if (i % stride != 0) continue;
            std::istringstream row(line);
            bool complete = true;
            for (auto& value : values)
                if (!(row >> value)) {
                    complete = false;
                    break;
                }
            if (!complete) continue;
            push(
                values[index_of(px)], values[index_of(py)], values[index_of(pz)],
                has_colour ? values[index_of(pr)] : 0.0,
                has_colour ? values[index_of(pg)] : 0.0,
                has_colour ? values[index_of(pb)] : 0.0);
        }
    }
    if (scene.points.empty()) return "PLY produced no finite points";
    return {};
}

// ---------------------------------------------------------------------------
// SfM diagnostics CSV

std::vector<std::string> split_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char character = line[i];
        if (quoted) {
            if (character == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    current += '"';
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                current += character;
            }
        } else if (character == '"') {
            quoted = true;
        } else if (character == ',') {
            fields.push_back(current);
            current.clear();
        } else {
            current += character;
        }
    }
    fields.push_back(current);
    return fields;
}

double field_as_double(
    const std::vector<std::string>& fields, const std::size_t index) {
    if (index >= fields.size() || fields[index].empty()) return 0.0;
    return std::strtod(fields[index].c_str(), nullptr);
}

// Rebuilds the world-to-camera rotation from the CSV quaternion.
std::array<float, 9> rotation_from_quaternion(
    const double w, const double x, const double y, const double z) {
    const double norm = std::sqrt(w * w + x * x + y * y + z * z);
    if (!(norm > 1e-12)) return {{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    const double nw = w / norm;
    const double nx = x / norm;
    const double ny = y / norm;
    const double nz = z / norm;
    return {{
        static_cast<float>(1 - 2 * (ny * ny + nz * nz)),
        static_cast<float>(2 * (nx * ny - nz * nw)),
        static_cast<float>(2 * (nx * nz + ny * nw)),
        static_cast<float>(2 * (nx * ny + nz * nw)),
        static_cast<float>(1 - 2 * (nx * nx + nz * nz)),
        static_cast<float>(2 * (ny * nz - nx * nw)),
        static_cast<float>(2 * (nx * nz - ny * nw)),
        static_cast<float>(2 * (ny * nz + nx * nw)),
        static_cast<float>(1 - 2 * (nx * nx + ny * ny)),
    }};
}

std::string load_poses(
    const std::filesystem::path& path, SparseScene& scene) {
    std::ifstream input(path);
    if (!input) return "No camera poses next to the sparse cloud";
    std::string line;
    if (!std::getline(input, line)) return "Empty camera pose table";

    const auto columns = split_csv_row(line);
    const auto model_column = std::find(columns.begin(), columns.end(), "camera_model");
    const std::size_t model_index = static_cast<std::size_t>(model_column-columns.begin());
    const auto audit_column = std::find(columns.begin(), columns.end(), "position_audit");
    const std::size_t audit_index = static_cast<std::size_t>(audit_column-columns.begin());
    double reprojection_sum = 0.0;
    std::size_t reprojection_count = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv_row(line);
        if (fields.size() < 28) continue;
        ViewPose pose;
        pose.name = fields[1];
        if (model_index < fields.size())
            apply_camera_model(pose, parse_view_camera_model(fields[model_index]));
        pose.registered = field_as_double(fields, 2) != 0.0;
        if (audit_column != columns.end() && audit_index < fields.size()) {
            if (fields[audit_index] == "unverified")
                pose.position_audit = photara::sfm::PositionAudit::unverified;
            else if (fields[audit_index] == "graph_supported")
                pose.position_audit = photara::sfm::PositionAudit::graph_supported;
        }
        pose.width = static_cast<std::uint32_t>(field_as_double(fields, 4));
        pose.height = static_cast<std::uint32_t>(field_as_double(fields, 5));
        pose.fx = static_cast<float>(field_as_double(fields, 6));
        pose.fy = static_cast<float>(field_as_double(fields, 7));
        pose.cx = static_cast<float>(field_as_double(fields, 8));
        pose.cy = static_cast<float>(field_as_double(fields, 9));
        pose.k1 = static_cast<float>(field_as_double(fields, 10));
        pose.k2 = static_cast<float>(field_as_double(fields, 11));
        pose.k3 = static_cast<float>(field_as_double(fields, 12));
        pose.k4 = static_cast<float>(field_as_double(fields, 13));
        pose.centre = {
            static_cast<float>(field_as_double(fields, 14)),
            static_cast<float>(field_as_double(fields, 15)),
            static_cast<float>(field_as_double(fields, 16))};
        pose.rotation = rotation_from_quaternion(
            field_as_double(fields, 17), field_as_double(fields, 18),
            field_as_double(fields, 19), field_as_double(fields, 20));
        pose.observations =
            static_cast<std::size_t>(field_as_double(fields, 21));
        pose.reprojection_p95 =
            static_cast<float>(field_as_double(fields, 24));
        if (pose.registered) {
            ++scene.registered_views;
            const double mean = field_as_double(fields, 22);
            if (mean > 0.0) {
                reprojection_sum += mean;
                ++reprojection_count;
            }
        }
        ++scene.total_views;
        scene.views.push_back(std::move(pose));
    }
    scene.mean_reprojection = reprojection_count == 0
        ? 0.F
        : static_cast<float>(reprojection_sum / static_cast<double>(reprojection_count));
    return {};
}

bool sample_photo_rgb(
    const photara::io::RgbImage& image, const float x, const float y,
    double& r, double& g, double& b) {
    if (image.width == 0 || image.height == 0 || image.pixels.size() < 3)
        return false;
    const int xi = static_cast<int>(std::lround(static_cast<double>(x)));
    const int yi = static_cast<int>(std::lround(static_cast<double>(y)));
    if (xi < 0 || yi < 0 || xi >= static_cast<int>(image.width) ||
        yi >= static_cast<int>(image.height))
        return false;
    const std::size_t offset =
        (static_cast<std::size_t>(yi) * image.width +
         static_cast<std::size_t>(xi)) *
        3;
    if (offset + 2 >= image.pixels.size()) return false;
    r = image.pixels[offset];
    g = image.pixels[offset + 1];
    b = image.pixels[offset + 2];
    return true;
}

// Photos are decoded at the coarsest JPEG DCT scale that still covers this many
// pixels per side: landmark colours come from averaging many observations, so
// 1/4-resolution samples are visually identical and decode several times
// faster.
constexpr std::uint32_t k_min_photo_sample = 640;
// Measured on a 739-photo iPhone set (16-core host): 39.5 s serial
// full-resolution, 17.7 s serial at DCT 1/4, 1.8 s on 16 threads. The pass only
// runs for files that genuinely lack colour, so it may use most of the machine.
constexpr unsigned k_max_colour_threads = 16;
// Sampling a handful of stray landmarks is not worth touching the photos at
// all; the pass only runs when a meaningful share of the cloud lacks colour.
constexpr double k_min_missing_share = 0.01;

// Samples photos for the landmarks in `missing` (indices into loaded.colours)
// and patches their colour in. Runs off the UI thread and in parallel, because
// a colourless 739-photo set otherwise costs tens of seconds of serial
// full-resolution decodes on every load.
void colour_missing_points(
    const photara::sfm::Scene& scene, SparseScene& loaded,
    const std::vector<std::size_t>& track_ids,
    const std::vector<std::size_t>& missing) {
    if (missing.empty()) return;

    std::vector<std::vector<std::pair<std::size_t, photara::sfm::Index>>>
        observations_by_image(scene.images.size());
    for (const std::size_t point : missing) {
        const auto& track = scene.tracks[track_ids[point]];
        const std::size_t inliers = std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        for (std::size_t i = 0; i < inliers; ++i) {
            const auto& observation = track.observations[i];
            if (observation.image_id >= scene.images.size()) continue;
            observations_by_image[observation.image_id].emplace_back(
                point, observation.feature_id);
        }
    }

    // Slot lookup so the reduction below stays independent of the scheduling
    // order of the decode workers.
    constexpr std::size_t k_no_slot = static_cast<std::size_t>(-1);
    std::vector<std::size_t> slot(loaded.points.size(), k_no_slot);
    for (std::size_t i = 0; i < missing.size(); ++i) slot[missing[i]] = i;

    std::vector<std::vector<std::uint32_t>> pixels_by_image(scene.images.size());
    const unsigned threads = std::min(
        k_max_colour_threads, photara::parallel::resolve_thread_count(0));
    photara::parallel::parallel_for(
        scene.images.size(), threads, [&](const std::size_t image_id) {
            const auto& observations = observations_by_image[image_id];
            if (observations.empty()) return;
            const auto& image = scene.images[image_id];
            if (image.path.empty()) return;
            photara::io::RgbImage rgb;
            try {
                rgb = photara::io::load_rgb_with_minimum_size(
                    image.path, k_min_photo_sample, k_min_photo_sample);
            } catch (...) {
                return;
            }
            if (rgb.width == 0 || rgb.height == 0) return;
            const float scale_x =
                image.features.image_width > 0
                    ? static_cast<float>(rgb.width) /
                          static_cast<float>(image.features.image_width)
                    : 1.F;
            const float scale_y =
                image.features.image_height > 0
                    ? static_cast<float>(rgb.height) /
                          static_cast<float>(image.features.image_height)
                    : 1.F;
            auto& pixels = pixels_by_image[image_id];
            pixels.assign(observations.size(), 0U);
            for (std::size_t i = 0; i < observations.size(); ++i) {
                const photara::sfm::Index feature_id = observations[i].second;
                if (feature_id >= image.features.keypoints.size()) continue;
                const auto& keypoint = image.features.keypoints[feature_id];
                double r = 0.0;
                double g = 0.0;
                double b = 0.0;
                if (!sample_photo_rgb(
                        rgb, keypoint.x * scale_x, keypoint.y * scale_y, r, g, b))
                    continue;
                pixels[i] = 0x01000000U | static_cast<std::uint32_t>(r) |
                            (static_cast<std::uint32_t>(g) << 8) |
                            (static_cast<std::uint32_t>(b) << 16);
            }
        });

    std::vector<double> sum_r(missing.size());
    std::vector<double> sum_g(missing.size());
    std::vector<double> sum_b(missing.size());
    std::vector<std::uint32_t> samples(missing.size());
    for (std::size_t image_id = 0; image_id < scene.images.size(); ++image_id) {
        const auto& pixels = pixels_by_image[image_id];
        const auto& observations = observations_by_image[image_id];
        for (std::size_t i = 0; i < pixels.size(); ++i) {
            const std::uint32_t pixel = pixels[i];
            if (!(pixel & 0x01000000U)) continue;
            const std::size_t point = observations[i].first;
            const std::size_t index = slot[point];
            if (index == k_no_slot) continue;
            sum_r[index] += static_cast<double>(pixel & 255U);
            sum_g[index] += static_cast<double>((pixel >> 8) & 255U);
            sum_b[index] += static_cast<double>((pixel >> 16) & 255U);
            ++samples[index];
        }
    }
    for (std::size_t i = 0; i < missing.size(); ++i) {
        if (samples[i] == 0) continue;  // keeps the neutral grey stored earlier
        const auto channel = [count = samples[i]](const double sum) {
            return static_cast<int>(
                std::clamp(std::lround(sum / count), 0L, 255L));
        };
        loaded.colours[missing[i]] = IM_COL32(
            channel(sum_r[i]), channel(sum_g[i]), channel(sum_b[i]), 255);
    }
}

}  // namespace

void SparseScene::clear() { *this = {}; }

SparseScene::PositionAuditCounts SparseScene::position_audit_counts() const {
    PositionAuditCounts counts;
    for (const auto& view : views) {
        if (!view.registered) continue;
        if (view.position_audit == photara::sfm::PositionAudit::not_audited)
            ++counts.unaudited;
        else {
            ++counts.audited;
            counts.unverified += view.position_audit == photara::sfm::PositionAudit::unverified;
        }
    }
    return counts;
}

void PreviewMesh::compute_normals() {
    normals.assign(vertices.size(), {});
    for (const auto& face : faces) {
        if (face[0] >= vertices.size() || face[1] >= vertices.size() ||
            face[2] >= vertices.size())
            continue;
        const Vec3 a = vertices[face[0]];
        const Vec3 ab = vertices[face[1]] - a;
        const Vec3 ac = vertices[face[2]] - a;
        const Vec3 n = cross(ab, ac);
        normals[face[0]] = normals[face[0]] + n;
        normals[face[1]] = normals[face[1]] + n;
        normals[face[2]] = normals[face[2]] + n;
    }
    for (Vec3& n : normals) n = normalize(n);
}

void PreviewMesh::compute_bounds() {
    if (vertices.empty()) {
        centroid = {};
        radius = 1.F;
        return;
    }
    Vec3 sum;
    for (const Vec3& p : vertices) sum = sum + p;
    centroid = sum * (1.F / static_cast<float>(vertices.size()));
    std::vector<float> distances;
    distances.reserve(vertices.size());
    for (const Vec3& p : vertices) {
        const Vec3 offset = p - centroid;
        distances.push_back(std::sqrt(dot(offset, offset)));
    }
    std::nth_element(
        distances.begin(),
        distances.begin() + static_cast<std::ptrdiff_t>(distances.size() * 95 / 100),
        distances.end());
    const std::size_t p95 = std::min(distances.size() - 1, distances.size() * 95 / 100);
    radius = std::max(1e-3F, distances[p95] * 1.25F);
}

PreviewMesh preview_mesh_from_mvs(const photara::mvs::Mesh& source) {
    PreviewMesh mesh;
    mesh.vertices.reserve(source.vertices.size());
    for (const auto& v : source.vertices)
        mesh.vertices.push_back({v.x(), v.y(), v.z()});
    if (source.normals.size() == source.vertices.size()) {
        mesh.normals.reserve(source.normals.size());
        for (const auto& n : source.normals)
            mesh.normals.push_back({n.x(), n.y(), n.z()});
    }
    if (source.colors.size() == source.vertices.size()) {
        mesh.colours.reserve(source.colors.size());
        for (const auto& c : source.colors) {
            const int r = static_cast<int>(std::lround(
                std::clamp(c.x(), 0.F, 1.F) * 255.F));
            const int g = static_cast<int>(std::lround(
                std::clamp(c.y(), 0.F, 1.F) * 255.F));
            const int b = static_cast<int>(std::lround(
                std::clamp(c.z(), 0.F, 1.F) * 255.F));
            mesh.colours.push_back(IM_COL32(r, g, b, 255));
        }
    }
    mesh.faces.reserve(source.faces.size());
    for (const auto& f : source.faces) {
        if (f.x() < 0 || f.y() < 0 || f.z() < 0) continue;
        const auto i0 = static_cast<std::uint32_t>(f.x());
        const auto i1 = static_cast<std::uint32_t>(f.y());
        const auto i2 = static_cast<std::uint32_t>(f.z());
        if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() ||
            i2 >= mesh.vertices.size() || i0 == i1 || i1 == i2 || i0 == i2)
            continue;
        mesh.faces.push_back({i0, i1, i2});
    }
    if (mesh.normals.size() != mesh.vertices.size()) mesh.compute_normals();
    mesh.compute_bounds();
    return mesh;
}

namespace {

int obj_index(const int value, const int count) {
    if (value > 0) return value - 1;
    if (value < 0) return count + value;
    return -1;
}

bool parse_obj_corner(
    const std::string& token, int& vertex, int& tex, int& normal) {
    vertex = tex = normal = 0;
    if (token.empty()) return false;
    const std::size_t first = token.find('/');
    if (first == std::string::npos) {
        vertex = std::atoi(token.c_str());
        return vertex != 0;
    }
    vertex = std::atoi(token.c_str());
    const std::size_t second = token.find('/', first + 1);
    if (second == std::string::npos) {
        tex = std::atoi(token.c_str() + first + 1);
        return vertex != 0;
    }
    if (second > first + 1) tex = std::atoi(token.c_str() + first + 1);
    if (second + 1 < token.size())
        normal = std::atoi(token.c_str() + second + 1);
    return vertex != 0;
}

PreviewMesh preview_mesh_from_obj(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Failed to open OBJ: " + path.string());
    std::vector<Vec3> positions, normals;
    std::vector<Vec2> texcoords;
    PreviewMesh mesh;
    struct CornerHash {
        std::size_t operator()(const std::array<int, 3>& corner) const noexcept {
            std::size_t hash = 0;
            for (int index : corner)
                hash ^= std::hash<int>{}(index) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
            return hash;
        }
    };
    std::unordered_map<std::array<int, 3>, std::uint32_t, CornerHash> corners;
    std::vector<std::array<int, 3>> polygon;
    bool have_normals = true, have_uvs = false;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        std::string tag;
        stream >> tag;
        if (tag == "v") {
            Vec3 p;
            stream >> p.x >> p.y >> p.z;
            positions.push_back(p);
        } else if (tag == "vt") {
            Vec2 uv;
            stream >> uv.x >> uv.y;
            // OBJ uses bottom-left UVs; image sampling uses top-left.
            uv.y = 1.F - uv.y;
            texcoords.push_back(uv);
        } else if (tag == "vn") {
            Vec3 n;
            stream >> n.x >> n.y >> n.z;
            normals.push_back(n);
        } else if (tag == "f") {
            if (mesh.vertices.empty()) {
                mesh.vertices.reserve(positions.size());
                mesh.normals.reserve(positions.size());
                mesh.uvs.reserve(positions.size());
            }
            polygon.clear();
            std::string token;
            while (stream >> token) {
                int vertex = 0, tex = 0, normal = 0;
                if (!parse_obj_corner(token, vertex, tex, normal)) continue;
                polygon.push_back({obj_index(vertex, static_cast<int>(positions.size())),
                    obj_index(tex, static_cast<int>(texcoords.size())),
                    obj_index(normal, static_cast<int>(normals.size()))});
            }
            for (std::size_t k = 1; k + 1 < polygon.size(); ++k) {
                const std::array<std::array<int, 3>, 3> tri{polygon[0], polygon[k], polygon[k + 1]};
                if (std::any_of(tri.begin(), tri.end(), [&](const auto& c) {
                    return c[0] < 0 || static_cast<std::size_t>(c[0]) >= positions.size();
                })) continue;
                std::array<std::uint32_t, 3> face;
                for (std::size_t c = 0; c < 3; ++c) {
                    const int vi = tri[c][0];
                    const int ti = tri[c][1] >= 0 && static_cast<std::size_t>(tri[c][1]) < texcoords.size()
                        ? tri[c][1] : -1;
                    const int ni = tri[c][2] >= 0 && static_cast<std::size_t>(tri[c][2]) < normals.size()
                        ? tri[c][2] : -1;
                    // Share only identical position/UV/normal tuples. Keep
                    // triangles without normals separate for the existing flat
                    // shading behavior; preserve UV seams and hard normal edges.
                    auto index = static_cast<std::uint32_t>(mesh.vertices.size());
                    bool inserted = true;
                    if (ni >= 0) {
                        if (corners.empty()) corners.reserve(positions.size());
                        auto result = corners.try_emplace(std::array<int, 3>{vi, ti, ni}, index);
                        index = result.first->second;
                        inserted = result.second;
                    }
                    if (inserted) {
                        mesh.vertices.push_back(positions[static_cast<std::size_t>(vi)]);
                        mesh.uvs.push_back(ti >= 0 ? texcoords[static_cast<std::size_t>(ti)] : Vec2{});
                        mesh.normals.push_back(ni >= 0 ? normals[static_cast<std::size_t>(ni)] : Vec3{});
                    }
                    have_normals = have_normals && ni >= 0;
                    have_uvs = have_uvs || ti >= 0;
                    face[c] = index;
                }
                mesh.faces.push_back(face);
            }
        }
    }
    if (!have_uvs) mesh.uvs.clear();
    if (!have_normals) mesh.compute_normals();
    mesh.compute_bounds();
    return mesh;
}

}  // namespace

MeshLoad load_preview_textured_mesh(std::filesystem::path stem) {
    MeshLoad loaded;
    try {
        auto obj = stem;
        obj += ".obj";
        auto albedo = stem;
        albedo += "_albedo.png";
        std::error_code error;
        if (stem.empty() || !std::filesystem::exists(obj, error)) {
            loaded.error = "No textured OBJ to load";
            return loaded;
        }
        loaded.mesh = preview_mesh_from_obj(obj);
        if (std::filesystem::exists(albedo, error))
            loaded.mesh.albedo_path = albedo;
        loaded.ok = loaded.mesh.has();
        if (!loaded.ok) loaded.error = "Textured mesh has no faces";
    } catch (const std::exception& failure) {
        loaded.error = failure.what();
    }
    return loaded;
}

MeshLoad load_preview_mesh(
    std::filesystem::path mesh_ply, std::filesystem::path ascan) {
    MeshLoad loaded;
    try {
        std::error_code error;
        if (!mesh_ply.empty() && std::filesystem::exists(mesh_ply, error)) {
            auto extension = mesh_ply.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (extension == ".obj") {
                auto stem = mesh_ply;
                stem.replace_extension();
                loaded.mesh = preview_mesh_from_obj(mesh_ply);
                auto albedo = stem;
                albedo += "_albedo.png";
                if (std::filesystem::exists(albedo, error))
                    loaded.mesh.albedo_path = albedo;
            } else {
                loaded.mesh = preview_mesh_from_mvs(photara::mvs::load_mesh_ply(mesh_ply));
            }
        } else if (!ascan.empty() && std::filesystem::exists(ascan, error)) {
            const auto archive = photara::project::Archive::open(ascan);
            if (!archive.has(photara::project::ChunkType::mesh)) {
                loaded.error = "Project has no mesh chunk";
                return loaded;
            }
            loaded.mesh = preview_mesh_from_mvs(
                photara::mvs::decode_mesh(
                    archive.chunk(photara::project::ChunkType::mesh)));
        } else {
            loaded.error = "No mesh file to load";
            return loaded;
        }
        loaded.ok = loaded.mesh.has();
        if (!loaded.ok) loaded.error = "Mesh has no faces";
    } catch (const std::exception& failure) {
        loaded.error = failure.what();
    }
    return loaded;
}

MeshLoad load_preview_mesh_handoff(const std::filesystem::path& ready) {
    MeshLoad loaded;
    try {
        loaded.mesh = preview_mesh_from_mvs(photara::mvs::receive_mesh_handoff(ready));
        loaded.ok = loaded.mesh.has();
        if (!loaded.ok) loaded.error = "Shared mesh has no faces";
    } catch (const std::exception& error) {
        loaded.error = error.what();
    }
    return loaded;
}

MeshLoad prepare_mesh_preview(MeshLoad loaded, const bool fit_box) {
    if (!loaded.ok) return loaded;
    try {
        const auto& mesh = loaded.mesh;
        auto data = std::make_shared<gpu::MeshGpuData>();
        data->vertices.resize(mesh.vertices.size());
        const bool normals = mesh.normals.size() == mesh.vertices.size();
        const bool colors = mesh.colours.size() == mesh.vertices.size();
        const bool uvs = mesh.uvs.size() == mesh.vertices.size();
        for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
            auto& v = data->vertices[i];
            v.px = mesh.vertices[i].x; v.py = mesh.vertices[i].y; v.pz = mesh.vertices[i].z;
            if (normals) {
                v.nx = mesh.normals[i].x; v.ny = mesh.normals[i].y; v.nz = mesh.normals[i].z;
            }
            if (colors) {
                const auto c = mesh.colours[i];
                v.r = float((c >> IM_COL32_R_SHIFT) & 255U) / 255.F;
                v.g = float((c >> IM_COL32_G_SHIFT) & 255U) / 255.F;
                v.b = float((c >> IM_COL32_B_SHIFT) & 255U) / 255.F;
            }
            if (uvs) { v.u = mesh.uvs[i].x; v.v = mesh.uvs[i].y; }
        }
        data->indices.reserve(mesh.faces.size() * 3U);
        for (const auto& face : mesh.faces)
            data->indices.insert(data->indices.end(), face.begin(), face.end());
        loaded.gpu_data = std::move(data);
        if (!mesh.albedo_path.empty()) {
            // Match the existing preview's fallback when an optional texture
            // is missing or unreadable, but perform decoding off the UI thread.
            try {
                const auto rgb = photara::io::load_rgb(mesh.albedo_path);
                loaded.albedo.width = rgb.width;
                loaded.albedo.height = rgb.height;
                const auto count = static_cast<std::size_t>(rgb.width) * rgb.height;
                loaded.albedo.pixels.resize(count * 4U);
                for (std::size_t i = 0; i < count; ++i) {
                    loaded.albedo.pixels[4U * i] = rgb.pixels[3U * i];
                    loaded.albedo.pixels[4U * i + 1U] = rgb.pixels[3U * i + 1U];
                    loaded.albedo.pixels[4U * i + 2U] = rgb.pixels[3U * i + 2U];
                    loaded.albedo.pixels[4U * i + 3U] = 255;
                }
            }
            catch (const std::exception&) { loaded.albedo = {}; }
        }
        if (fit_box) {
            ReconstructionBox box;
            fit_reconstruction_box(box, mesh.vertices);
            loaded.reconstruction_bounds = {box.min, box.max};
            loaded.reconstruction_bounds_valid = box.valid;
        }
    } catch (const std::exception& failure) {
        loaded.error = failure.what();
        loaded.ok = false;
    }
    return loaded;
}

SceneLoad sparse_scene_from_sfm(const photara::sfm::Scene& scene, bool colour_from_photos) {
    SceneLoad loaded;
    loaded.ok = true;
    std::size_t point_count = 0;
    for (const auto& track : scene.tracks)
        if (track.is_triangulated()) ++point_count;
    loaded.scene.points.reserve(point_count);
    std::vector<std::size_t> track_ids;
    track_ids.reserve(point_count);
    for (std::size_t index = 0; index < scene.tracks.size(); ++index) {
        const auto& track = scene.tracks[index];
        if (!track.is_triangulated()) continue;
        loaded.scene.points.push_back({
            static_cast<float>(track.position.x()),
            static_cast<float>(track.position.y()),
            static_cast<float>(track.position.z())});
        track_ids.push_back(index);
    }
    std::vector<std::vector<char>> triangulated(scene.images.size());
    for (const auto& track : scene.tracks) {
        if (!track.is_triangulated()) continue;
        const std::size_t inliers = std::min<std::size_t>(
            track.num_inliers, track.observations.size());
        for (std::size_t i = 0; i < inliers; ++i) {
            const auto& observation = track.observations[i];
            if (observation.image_id >= scene.images.size()) continue;
            const auto& image = scene.images[observation.image_id];
            auto& flags = triangulated[observation.image_id];
            if (flags.empty())
                flags.assign(image.features.keypoints.size(), 0);
            if (observation.feature_id < flags.size())
                flags[observation.feature_id] = 1;
        }
    }

    loaded.scene.views.reserve(scene.images.size());
    for (const auto& image : scene.images) {
        ViewPose pose;
        const auto name = image.path.filename().u8string();
        pose.name.assign(name.begin(), name.end());
        pose.centre = {
            static_cast<float>(image.pose.C.x()),
            static_cast<float>(image.pose.C.y()),
            static_cast<float>(image.pose.C.z())};
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                pose.rotation[static_cast<std::size_t>(row * 3 + column)] =
                    static_cast<float>(image.pose.R(row, column));
        if (image.camera_id < scene.cameras.size()) {
            const auto& camera = scene.cameras[image.camera_id];
            apply_camera_model(pose, camera.model);
            pose.fx = static_cast<float>(camera.fx);
            pose.fy = static_cast<float>(camera.fy);
            pose.cx = static_cast<float>(camera.cx);
            pose.cy = static_cast<float>(camera.cy);
            pose.k1 = static_cast<float>(camera.k1);
            pose.k2 = static_cast<float>(camera.k2);
            pose.k3 = static_cast<float>(camera.p1);
            pose.k4 = static_cast<float>(camera.p2);
            pose.width = camera.width;
            pose.height = camera.height;
        }
        pose.observations = image.id < scene.image_tracks.size()
            ? scene.image_tracks[image.id].size()
            : 0;
        pose.image_path = image.path;
        pose.registered = image.registered;
        pose.position_audit = image.position_audit;
        if (pose.registered) ++loaded.scene.registered_views;

        const auto& keypoints = image.features.keypoints;
        const float image_w = image.features.image_width > 0
            ? static_cast<float>(image.features.image_width)
            : (pose.width > 0 ? static_cast<float>(pose.width) : 1.F);
        const float image_h = image.features.image_height > 0
            ? static_cast<float>(image.features.image_height)
            : (pose.height > 0 ? static_cast<float>(pose.height) : 1.F);
        const std::vector<char>* flags =
            image.id < triangulated.size() ? &triangulated[image.id] : nullptr;
        constexpr std::size_t k_max_stored_features = 24'000;
        pose.features.reserve(std::min(keypoints.size(), k_max_stored_features));
        std::vector<std::size_t> order;
        if (keypoints.size() > k_max_stored_features) {
            order.resize(keypoints.size());
            for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::stable_sort(
                order.begin(), order.end(),
                [&](const std::size_t a, const std::size_t b) {
                    const bool ta = flags && a < flags->size() && (*flags)[a];
                    const bool tb = flags && b < flags->size() && (*flags)[b];
                    if (ta != tb) return ta;
                    return keypoints[a].response > keypoints[b].response;
                });
            order.resize(k_max_stored_features);
        }
        const std::size_t count = order.empty() ? keypoints.size() : order.size();
        for (std::size_t n = 0; n < count; ++n) {
            const std::size_t i = order.empty() ? n : order[n];
            const auto& keypoint = keypoints[i];
            ImageFeature feature;
            feature.u = keypoint.x / image_w;
            feature.v = keypoint.y / image_h;
            feature.scale = std::max(0.002F, keypoint.scale / image_w);
            feature.response = keypoint.response;
            feature.triangulated =
                flags && i < flags->size() && (*flags)[i] != 0;
            if (feature.triangulated) ++pose.triangulated_features;
            pose.features.push_back(feature);
        }
        loaded.scene.views.push_back(std::move(pose));
    }
    loaded.scene.total_views = scene.images.size();
    // Landmarks normally carry their colour in the working copy, so this walks
    // the tracks once and only pays for photo sampling where colour is missing.
    // The previous all-or-nothing rule threw the whole palette away as soon as a
    // single track lacked colour, which on a 739-photo set meant ~40 s of
    // serial full-resolution decodes before the cloud could be shown.
    if (!track_ids.empty()) {
        std::size_t coloured_tracks = 0;
        for (const std::size_t id : track_ids)
            if (scene.tracks[id].has_color) ++coloured_tracks;
        const bool use_default_colour =
            coloured_tracks == 0 && !colour_from_photos;
        if (!use_default_colour) {
            loaded.scene.colours.resize(
                track_ids.size(), IM_COL32(180, 180, 180, 255));
            std::vector<std::size_t> missing;
            missing.reserve(track_ids.size() - coloured_tracks);
            for (std::size_t i = 0; i < track_ids.size(); ++i) {
                const auto& track = scene.tracks[track_ids[i]];
                if (!track.has_color) {
                    missing.push_back(i);
                    continue;
                }
                loaded.scene.colours[i] = IM_COL32(
                    track.color_r, track.color_g, track.color_b, 255);
            }
            const bool worth_sampling =
                colour_from_photos &&
                static_cast<double>(missing.size()) >=
                    k_min_missing_share *
                        static_cast<double>(track_ids.size());
            if (worth_sampling)
                colour_missing_points(scene, loaded.scene, track_ids, missing);
        }
    }
    loaded.scene.compute_bounds();
    return loaded;
}

SceneLoad sparse_scene_from_dataset(
    const std::filesystem::path& source,
    const std::filesystem::path& initial_point_cloud,
    const std::filesystem::path& image_directory) {
    SceneLoad loaded;
    try {
        photara::splat::DatasetLoadRequest request;
        request.source = source;
        request.image_directory = image_directory;
        request.initial_point_cloud = initial_point_cloud;
        const photara::splat::DatasetLoadResult dataset =
            photara::splat::load_splat_dataset(request);
        const photara::mvs::MvsScene& scene = dataset.scene;

        loaded.scene.views.reserve(scene.views.size());
        for (const auto& view : scene.views) {
            ViewPose pose;
            const auto name = view.path.filename().u8string();
            pose.name.assign(name.begin(), name.end());
            pose.image_path = view.path;
            pose.centre = {
                static_cast<float>(view.pose.C.x()),
                static_cast<float>(view.pose.C.y()),
                static_cast<float>(view.pose.C.z())};
            for (int row = 0; row < 3; ++row)
                for (int column = 0; column < 3; ++column)
                    pose.rotation[static_cast<std::size_t>(row * 3 + column)] =
                        static_cast<float>(view.pose.R(row, column));
            apply_camera_model(pose, view.source_model);
            pose.k1 = view.k1;
            pose.k2 = view.k2;
            pose.k3 = view.p1;
            pose.k4 = view.p2;
            pose.src_fx = view.src_fx;
            pose.src_fy = view.src_fy;
            pose.src_cx = view.src_cx;
            pose.src_cy = view.src_cy;
            pose.src_width = view.src_width;
            pose.src_height = view.src_height;
            if (photara::uses_native_splat_projection(view.source_model) &&
                view.src_width != 0 && view.src_height != 0) {
                pose.fx = view.src_fx;
                pose.fy = view.src_fy;
                pose.cx = view.src_cx;
                pose.cy = view.src_cy;
                pose.width = view.src_width;
                pose.height = view.src_height;
            } else {
                pose.fx = view.fx;
                pose.fy = view.fy;
                pose.cx = view.cx;
                pose.cy = view.cy;
                pose.width = view.width;
                pose.height = view.height;
            }
            pose.registered = true;
            ++loaded.scene.registered_views;
            ++loaded.scene.total_views;
            loaded.scene.views.push_back(std::move(pose));
        }

        const auto append_scene_point = [&](const Eigen::Vector3f& position,
                                            const Eigen::Vector3f& color) {
            loaded.scene.points.push_back(
                {position.x(), position.y(), position.z()});
            const auto channel = [](const float value) {
                return static_cast<int>(
                    std::lround(std::clamp(value, 0.F, 1.F) * 255.F));
            };
            loaded.scene.colours.push_back(IM_COL32(
                channel(color.x()), channel(color.y()), channel(color.z()),
                255));
        };
        loaded.scene.points.reserve(scene.sparse_points.size());
        loaded.scene.colours.reserve(scene.sparse_points.size());
        for (const auto& point : scene.sparse_points)
            append_scene_point(point.position, point.color);
        // RealityCapture (and camera-only COLMAP) may have no sparse tracks.
        // Fall back to the trainer's initial cloud so the 2D overlay still
        // has landmarks to project.
        if (loaded.scene.points.empty()) {
            constexpr std::size_t k_max_cloud_points = 40'000;
            const std::size_t stride = std::max<std::size_t>(
                1, (scene.dense_cloud.points.size() + k_max_cloud_points - 1) /
                       k_max_cloud_points);
            loaded.scene.points.reserve(std::min(
                scene.dense_cloud.points.size(), k_max_cloud_points));
            loaded.scene.colours.reserve(loaded.scene.points.capacity());
            for (std::size_t i = 0; i < scene.dense_cloud.points.size();
                 i += stride)
                append_scene_point(
                    scene.dense_cloud.points[i].position,
                    scene.dense_cloud.points[i].color);
        }

        // Imported alignments carry no per-image keypoint table. Projecting
        // the landmarks each view observes gives the same visual overlay and
        // marks every plotted feature as triangulated.
        std::vector<std::vector<std::size_t>> observations(scene.views.size());
        for (std::size_t index = 0; index < scene.sparse_points.size();
             ++index)
            for (const auto view_id : scene.sparse_points[index].view_ids)
                if (view_id < observations.size())
                    observations[view_id].push_back(index);
        constexpr std::size_t k_max_dataset_features = 24'000;
        const auto project_point = [](ViewPose& pose,
                                      const photara::mvs::MvsView& view,
                                      const Eigen::Vector3f& position) {
            if (view.width == 0 || view.height == 0) return;
            if (pose.features.size() >= k_max_dataset_features) return;
            const Eigen::Matrix3f rotation = view.pose.R.cast<float>();
            const Eigen::Vector3f centre = view.pose.C.cast<float>();
            const Eigen::Vector3f camera_point =
                rotation * (position - centre);
            if (!(camera_point.z() > 1e-3F)) return;
            const float inverse_z = 1.F / camera_point.z();
            const float pixel_x =
                view.fx * camera_point.x() * inverse_z + view.cx;
            const float pixel_y =
                view.fy * camera_point.y() * inverse_z + view.cy;
            if (pixel_x < 0.F || pixel_y < 0.F ||
                pixel_x >= static_cast<float>(view.width) ||
                pixel_y >= static_cast<float>(view.height))
                return;
            ImageFeature feature;
            feature.u = pixel_x / static_cast<float>(view.width);
            feature.v = pixel_y / static_cast<float>(view.height);
            feature.scale = 0.006F;
            feature.triangulated = true;
            pose.features.push_back(feature);
            ++pose.triangulated_features;
        };
        for (std::size_t index = 0;
             index < scene.views.size() && index < loaded.scene.views.size();
             ++index) {
            const auto& view = scene.views[index];
            ViewPose& pose = loaded.scene.views[index];
            pose.observations = observations[index].size();
            pose.features.reserve(std::min(
                std::max(observations[index].size(), loaded.scene.points.size()),
                k_max_dataset_features));
            if (!observations[index].empty()) {
                const std::size_t stride = std::max<std::size_t>(
                    1,
                    (observations[index].size() + k_max_dataset_features - 1) /
                        k_max_dataset_features);
                for (std::size_t i = 0; i < observations[index].size();
                     i += stride)
                    project_point(
                        pose, view,
                        scene.sparse_points[observations[index][i]].position);
            } else {
                const std::size_t stride = std::max<std::size_t>(
                    1, (loaded.scene.points.size() + k_max_dataset_features - 1) /
                           k_max_dataset_features);
                for (std::size_t i = 0; i < loaded.scene.points.size();
                     i += stride)
                    project_point(
                        pose, view,
                        {loaded.scene.points[i].x, loaded.scene.points[i].y,
                         loaded.scene.points[i].z});
            }
        }

        loaded.scene.compute_bounds();
        loaded.ok = true;
    } catch (const std::exception& failure) {
        loaded = {};
        loaded.error = failure.what();
    }
    return loaded;
}

Vec3 SparseScene::world_point(const std::size_t index) const {
    const auto& point = points[index];
    if (placements.empty()) return point;
    const auto found = std::upper_bound(placements.begin(), placements.end(), index,
        [](std::size_t value, const Placement& p) { return value < p.transform.end_index; });
    if (found == placements.end()) return point;
    const auto world = splat_render::transform_point(found->transform, {point.x, point.y, point.z});
    return {world[0], world[1], world[2]};
}

GaussianPrimitive SparseScene::world_gaussian(const std::size_t index) const {
    auto result = gaussians[index];
    if (placements.empty()) return result;
    const auto found = std::upper_bound(placements.begin(), placements.end(), index,
        [](std::size_t value, const Placement& p) { return value < p.transform.end_index; });
    if (found == placements.end()) return result;
    result.scale = result.scale * found->transform.scale;
    const auto p = found->transform.rotation, q = result.rotation;
    result.rotation = {p[0]*q[0]-p[1]*q[1]-p[2]*q[2]-p[3]*q[3],
        p[0]*q[1]+p[1]*q[0]+p[2]*q[3]-p[3]*q[2],
        p[0]*q[2]-p[1]*q[3]+p[2]*q[0]+p[3]*q[1],
        p[0]*q[3]+p[1]*q[2]-p[2]*q[1]+p[3]*q[0]};
    return result;
}

void SparseScene::set_model_transforms(std::span<const splat_render::ModelTransform> transforms) {
    std::vector<Placement> next;
    std::size_t begin = 0;
    for (const auto& transform : transforms) {
        auto sampled = transform;
        sampled.end_index = static_cast<std::uint32_t>(std::min(points.size(),
            (std::size_t(transform.end_index) + gaussian_sample_stride - 1) / gaussian_sample_stride));
        if (sampled.end_index <= begin) continue;
        Placement placement;
        placement.transform = sampled;
        const std::size_t slot = next.size();
        if (slot < placements.size() && placements[slot].transform.end_index == sampled.end_index) {
            placement.local_center = placements[slot].local_center;
            placement.local_radius = placements[slot].local_radius;
        } else {
            for (std::size_t i = begin; i < sampled.end_index; ++i)
                placement.local_center = placement.local_center + points[i];
            placement.local_center = placement.local_center * (1.F / float(sampled.end_index - begin));
            std::vector<float> distances;
            distances.reserve(sampled.end_index - begin);
            for (std::size_t i = begin; i < sampled.end_index; ++i) {
                const auto d = points[i] - placement.local_center;
                distances.push_back(std::sqrt(dot(d, d)));
            }
            const auto rank = std::min(distances.size() - 1, std::size_t(distances.size() * .95));
            std::nth_element(distances.begin(), distances.begin() + rank, distances.end());
            placement.local_radius = std::max(1e-3F, distances[rank]);
        }
        next.push_back(placement);
        begin = sampled.end_index;
    }
    placements = std::move(next);
    if (placements.empty()) return;
    centroid = {};
    for (const auto& p : placements) {
        const auto c = splat_render::transform_point(p.transform, {p.local_center.x, p.local_center.y, p.local_center.z});
        centroid = centroid + Vec3{c[0], c[1], c[2]};
    }
    centroid = centroid * (1.F / float(placements.size()));
    radius = 1e-3F;
    for (const auto& p : placements) {
        const auto c = splat_render::transform_point(p.transform, {p.local_center.x, p.local_center.y, p.local_center.z});
        const auto d = Vec3{c[0], c[1], c[2]} - centroid;
        radius = std::max(radius, std::sqrt(dot(d, d)) + p.local_radius * p.transform.scale);
    }
}

void SparseScene::compute_bounds() {
    if (points.empty()) {
        Vec3 sum;
        std::size_t used = 0;
        for (const ViewPose& pose : views) {
            if (!pose.registered) continue;
            sum = sum + pose.centre;
            ++used;
        }
        if (used == 0) {
            centroid = {};
            radius = 1.F;
            return;
        }
        centroid = sum * (1.F / static_cast<float>(used));
        float farthest = 0.F;
        for (const ViewPose& pose : views) {
            if (!pose.registered) continue;
            const Vec3 offset = pose.centre - centroid;
            farthest = std::max(farthest, std::sqrt(dot(offset, offset)));
        }
        radius = std::max(1e-3F, farthest * 1.25F);
        return;
    }
    // Median-ish centre via the mean, then a robust radius from the 95th
    // percentile distance so a few stray tracks cannot zoom the view out.
    Vec3 sum;
    for (const Vec3& point : points) sum = sum + point;
    const float inverse = 1.F / static_cast<float>(points.size());
    centroid = sum * inverse;

    std::vector<float> distances;
    distances.reserve(points.size());
    for (const Vec3& point : points) {
        const Vec3 offset = point - centroid;
        distances.push_back(std::sqrt(dot(offset, offset)));
    }
    const std::size_t index = static_cast<std::size_t>(
        std::min<double>(distances.size() - 1, distances.size() * 0.95));
    std::nth_element(
        distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(index),
        distances.end());
    radius = std::max(1e-3F, distances[index]);
}

SceneLoad load_sparse_scene(
    std::filesystem::path cloud_ply, std::filesystem::path poses_csv) {
    SceneLoad result;
    result.error = load_ply(cloud_ply, result.scene);
    if (!result.error.empty()) return result;
    // Poses are a bonus: report the cloud even when the CSV is absent.
    load_poses(poses_csv, result.scene);
    result.scene.compute_bounds();
    result.ok = true;
    return result;
}

SceneLoad gaussian_scene_from_model(
    const photara::splat::GaussianModel& model,
    std::filesystem::path poses_csv) {
    SceneLoad result;
    try {
        if (model.size() == 0) {
            result.error = "Gaussian model contains no points";
            return result;
        }
        const auto means = model.means.to_vector();
        const auto sh = model.sh.to_vector();
        const auto log_scales = model.log_scales.is_valid()
            ? model.log_scales.to_vector()
            : std::vector<float>{};
        const auto rotations = model.quaternions.is_valid()
            ? model.quaternions.to_vector()
            : std::vector<float>{};
        const std::size_t count = model.size();
        const std::size_t stride = std::max<std::size_t>(
            1, (count + k_max_loaded_points - 1U) / k_max_loaded_points);
        const std::size_t visible_count = (count + stride - 1U) / stride;
        result.scene.gaussian_sample_stride = stride;
        result.scene.points.reserve(visible_count);
        result.scene.colours.reserve(visible_count);
        result.scene.gaussians.reserve(visible_count);
        const std::size_t bases = model.sh.shape()[1];
        constexpr float sh_dc = 0.28209479177387814F;
        for (std::size_t index = 0; index < count; index += stride) {
            const std::size_t mean_offset = index * 3U;
            result.scene.points.push_back({
                means[mean_offset], means[mean_offset + 1U],
                means[mean_offset + 2U]});
            const std::size_t sh_offset = index * bases * 3U;
            const auto channel = [&](const std::size_t component) {
                return static_cast<int>(std::lround(std::clamp(
                    0.5F + sh_dc * sh[sh_offset + component], 0.F, 1.F) *
                    255.F));
            };
            result.scene.colours.push_back(IM_COL32(
                channel(0), channel(1), channel(2), 255));
            const std::size_t scale_offset = index * 3U;
            const std::size_t quat_offset = index * 4U;
            if (scale_offset + 2U < log_scales.size() &&
                quat_offset + 3U < rotations.size()) {
                GaussianPrimitive primitive;
                primitive.scale = {
                    std::exp(log_scales[scale_offset]),
                    std::exp(log_scales[scale_offset + 1U]),
                    std::exp(log_scales[scale_offset + 2U])};
                primitive.rotation = {
                    rotations[quat_offset], rotations[quat_offset + 1U],
                    rotations[quat_offset + 2U], rotations[quat_offset + 3U]};
                result.scene.gaussians.push_back(primitive);
            }
        }
        if (result.scene.gaussians.size() != result.scene.points.size())
            result.scene.gaussians.clear();
        load_poses(poses_csv, result.scene);
        result.scene.compute_bounds();
        result.ok = true;
    } catch (const std::exception& failure) {
        result.error = failure.what();
    }
    return result;
}

SceneLoad load_gaussian_scene(
    std::filesystem::path model_path, std::filesystem::path poses_csv) {
    try {
        const auto model = photara::splat::load_gaussians(model_path);
        return gaussian_scene_from_model(model, std::move(poses_csv));
    } catch (const std::exception& failure) {
        SceneLoad result;
        result.error = failure.what();
        return result;
    }
}

bool load_view_poses(
    const std::filesystem::path& poses_csv, SparseScene& scene) {
    if (!scene.views.empty()) return true;
    const std::string error = load_poses(poses_csv, scene);
    if (error.empty() && !scene.views.empty()) {
        if (!scene.has_points()) scene.compute_bounds();
        return true;
    }
    return false;
}

void attach_view_image_paths(
    SparseScene& scene, const std::filesystem::path& images_dir) {
    // The viewport calls this every frame, and probing one capture file per
    // view (739 of them for the indoor iPhone dataset) dwarfs every other
    // per-frame cost in the editor. The answer only changes when the views or
    // the image directory do, so remember what was already resolved.
    if (scene.image_paths_dir == images_dir &&
        scene.image_paths_view_count == scene.views.size())
        return;
    scene.image_paths_dir = images_dir;
    scene.image_paths_view_count = scene.views.size();
    std::error_code error;
    for (ViewPose& pose : scene.views) {
        if (!pose.image_path.empty()) {
            if (std::filesystem::is_regular_file(pose.image_path, error))
                continue;
            if (pose.image_path.is_relative() && !images_dir.empty()) {
                const auto joined = images_dir / pose.image_path;
                if (std::filesystem::is_regular_file(joined, error)) {
                    pose.image_path = joined;
                    continue;
                }
            }
        }
        if (images_dir.empty() || pose.name.empty()) continue;
        const auto candidate = images_dir / pose.name;
        if (std::filesystem::is_regular_file(candidate, error))
            pose.image_path = candidate;
    }
}

CameraMarkerGeometry make_camera_marker_geometry(const ViewPose& pose, const bool detailed) {
    CameraMarkerGeometry mesh;
    constexpr std::uint32_t invalid = ~std::uint32_t{0};
    const auto triangle = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        if (a == invalid || b == invalid || c == invalid) return;
        const Vec3 area = cross(mesh.vertices[b].position - mesh.vertices[a].position,
                                mesh.vertices[c].position - mesh.vertices[a].position);
        if (dot(area, area) > 1e-14F) mesh.triangles.push_back({a, b, c});
    };
    const auto line = [&](std::uint32_t a, std::uint32_t b) {
        if (a == invalid || b == invalid) return;
        const Vec3 delta = mesh.vertices[a].position - mesh.vertices[b].position;
        if (dot(delta, delta) > 1e-12F) mesh.lines.push_back({a, b});
    };
    if (pose.model == photara::CameraModel::equirectangular) {
        const unsigned columns = detailed ? 64 : 16, rows = detailed ? 32 : 8;
        for (unsigned y = 0; y <= rows; ++y) for (unsigned x = 0; x <= columns; ++x) {
            const float u = float(x) / columns, v = float(y) / rows;
            const auto ray = photara::unproject_equirectangular_camera(u * 2., v, 2, 1);
            mesh.vertices.push_back({{float(ray.x), float(ray.y), float(ray.z)}, {u, v}});
        }
        const auto at = [&](unsigned x, unsigned y) { return y * (columns + 1) + x; };
        for (unsigned y = 0; y < rows; ++y) for (unsigned x = 0; x < columns; ++x) {
            triangle(at(x,y), at(x+1,y), at(x+1,y+1));
            triangle(at(x,y), at(x+1,y+1), at(x,y+1));
            if (x % (columns / (detailed ? 8 : 4)) == 0) line(at(x,y), at(x,y+1));
            if (y != 0 && y % (rows / 4) == 0) line(at(x,y), at(x+1,y));
        }
        return mesh;
    }
    if (pose.model != photara::CameraModel::opencv_fisheye) {
        auto local = pose;
        local.centre = {};
        local.rotation = {1,0,0,0,1,0,0,0,1};
        Vec3 corners[4];
        image_plane_corners(local, 1.F, corners);
        const Vec2 uv[] = {{0,0},{1,0},{1,1},{0,1}};
        for (unsigned i = 0; i < 4; ++i) {
            mesh.vertices.push_back({corners[i], uv[i]});
            mesh.lines.push_back({i, (i+1)%4});
            mesh.rays.push_back(i);
        }
        triangle(0,1,2); triangle(0,2,3);
        return mesh;
    }

    const auto k = marker_intrinsics(pose);
    if (!k.width || !k.height || !std::isfinite(k.fx) || !std::isfinite(k.fy) ||
        !(k.fx > 1e-3F) || !(k.fy > 1e-3F)) return mesh;
    // Polar sampling clips the rectangular sensor against the calibrated
    // forward hemisphere, including circular fisheyes whose corners are invalid.
    const double theta = 0.5 * photara::k_pi - 1e-4, t2 = theta * theta;
    const double radius = theta * (1 + t2 * (pose.k1 + t2 * (pose.k2 + t2 * (pose.k3 + t2 * pose.k4))));
    if (!(radius > 0) || !std::isfinite(radius)) return mesh;
    const double xmin = -k.cx / k.fx, xmax = (k.width - double(k.cx)) / k.fx;
    const double ymin = -k.cy / k.fy, ymax = (k.height - double(k.cy)) / k.fy;
    const unsigned segments = detailed ? 64 : 16, rings = detailed ? 12 : 4;
    std::vector<double> angles;
    for (unsigned i = 0; i < segments; ++i) angles.push_back(2 * photara::k_pi * i / segments);
    for (double x : {xmin, xmax}) for (double y : {ymin, ymax}) {
        double angle = std::atan2(y, x);
        if (angle < 0) angle += 2 * photara::k_pi;
        angles.push_back(angle);
    }
    std::sort(angles.begin(), angles.end());
    angles.erase(std::unique(angles.begin(), angles.end(),
        [](double a, double b) { return std::abs(a-b) < 1e-8; }), angles.end());
    const unsigned columns = static_cast<unsigned>(angles.size());
    std::vector<std::uint32_t> grid(columns * (rings + 1), invalid);
    const auto at = [&](unsigned x, unsigned y) { return grid[(x % columns) * (rings + 1) + y]; };
    for (unsigned x = 0; x < columns; ++x) {
        const double dx = std::cos(angles[x]), dy = std::sin(angles[x]);
        double lo = 0, hi = radius;
        const auto clip = [&](double d, double mn, double mx) {
            if (std::abs(d) < 1e-12) return mn <= 0 && mx >= 0;
            const double a = mn/d, b = mx/d;
            lo = std::max(lo, std::min(a,b)); hi = std::min(hi, std::max(a,b));
            return lo < hi;
        };
        if (!clip(dx, xmin, xmax) || !clip(dy, ymin, ymax)) continue;
        for (unsigned y = 0; y <= rings; ++y) {
            const double r = lo + (hi-lo)*y/rings;
            const double u = std::clamp(k.cx + k.fx*r*dx, 0., double(k.width));
            const double v = std::clamp(k.cy + k.fy*r*dy, 0., double(k.height));
            const auto ray = photara::unproject_fisheye_camera(u, v,
                k.fx,k.fy,k.cx,k.cy,pose.k1,pose.k2,pose.k3,pose.k4);
            if (!ray.valid || !std::isfinite(ray.x) || !std::isfinite(ray.y) || !std::isfinite(ray.z)) continue;
            grid[x*(rings+1)+y] = static_cast<std::uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back({{float(ray.x),float(ray.y),float(ray.z)},
                                     {float(u/k.width),float(v/k.height)}});
        }
        if (x % std::max(1U, columns / 4) == 0 && at(x,rings) != invalid)
            mesh.rays.push_back(at(x,rings));
    }
    for (unsigned x = 0; x < columns; ++x) {
        line(at(x,rings), at(x+1,rings));
        line(at(x,0), at(x+1,0));
        for (unsigned y = 0; y < rings; ++y) {
            triangle(at(x,y),at(x+1,y),at(x+1,y+1));
            triangle(at(x,y),at(x+1,y+1),at(x,y+1));
            if (x % std::max(1U, columns/8) == 0) line(at(x,y), at(x,y+1));
            if (y == rings/2) line(at(x,y), at(x+1,y));
        }
    }
    return mesh;
}

float camera_marker_hit_distance(
    const ViewPose& pose, const float size, const Vec3 origin, const Vec3 direction) {
    const float miss = std::numeric_limits<float>::infinity();
    if (!(size > 0.F)) return miss;
    const auto& r = pose.rotation;
    const auto local = [&](Vec3 v) {
        return Vec3{r[0]*v.x+r[1]*v.y+r[2]*v.z,
                    r[3]*v.x+r[4]*v.y+r[5]*v.z,
                    r[6]*v.x+r[7]*v.y+r[8]*v.z};
    };
    const Vec3 o = local(origin - pose.centre), d = local(direction);
    const auto k = marker_intrinsics(pose);
    const auto in_sensor = [&](double u, double v) {
        return std::isfinite(u) && std::isfinite(v) && u >= 0 && u <= k.width && v >= 0 && v <= k.height;
    };
    if (pose.model != photara::CameraModel::opencv_fisheye &&
        pose.model != photara::CameraModel::equirectangular) {
        if (std::abs(d.z) < 1e-8F) return miss;
        const float t = (size-o.z)/d.z;
        const Vec3 p = o+d*t;
        return t > 0 && in_sensor(k.fx*p.x/size+k.cx, k.fy*p.y/size+k.cy) ? t : miss;
    }
    const float a = dot(d,d), b = dot(o,d), c = dot(o,o)-size*size;
    if (!(a > 0)) return miss;
    const float discriminant = b*b-a*c;
    if (!(discriminant >= 0)) return miss;
    const float root = std::sqrt(discriminant);
    for (float t : {(-b-root)/a, (-b+root)/a}) {
        if (!(t > 0)) continue;
        if (pose.model == photara::CameraModel::equirectangular) return t;
        const Vec3 p = o+d*t;
        const auto uv = photara::project_fisheye_camera(p.x,p.y,p.z,
            k.fx,k.fy,k.cx,k.cy,pose.k1,pose.k2,pose.k3,pose.k4);
        if (uv.valid && in_sensor(uv.u,uv.v)) return t;
    }
    return miss;
}

void sampled_view_indices(
    const SparseScene& scene, std::vector<std::size_t>& indices,
    const std::size_t max_markers) {
    indices.clear();
    const std::size_t registered = std::max<std::size_t>(
        1, std::count_if(
               scene.views.begin(), scene.views.end(),
               [](const ViewPose& pose) { return pose.registered; }));
    const std::size_t stride = std::max<std::size_t>(
        1, (registered + std::max<std::size_t>(1, max_markers) - 1) /
               std::max<std::size_t>(1, max_markers));
    std::size_t ordinal = 0;
    for (std::size_t index = 0; index < scene.views.size(); ++index) {
        if (!scene.views[index].registered) continue;
        const bool sampled =
            ordinal % stride == 0 || ordinal + 1 == registered;
        ++ordinal;
        if (sampled) indices.push_back(index);
    }
}

void OrbitCamera::frame(const Vec3& centroid, const float radius) {
    yaw = 0.785398F;
    pitch = 0.61548F;
    target = centroid;
    const float span = std::max(1e-3F, radius);
    const float half_fov = fov_degrees * 0.5F * k_pi / 180.F;
    if (projection == EditorProjection::orthographic) {
        distance = span * 2.5F;
        ortho_height = span * 2.7F;
    } else {
        distance = span / std::max(0.05F, std::tan(half_fov)) * 1.35F;
    }
}

void OrbitCamera::frame(const SparseScene& scene) {
    if (!scene.has_points()) {
        yaw = 0.785398F;
        pitch = 0.61548F;
        target = {};
        distance = 6.F;
        return;
    }
    frame(scene.centroid, scene.radius);
}

void OrbitCamera::focus_on(const Vec3& point) {
    const float clamped_pitch = std::clamp(pitch, -k_pi * 0.5F, k_pi * 0.5F);
    const Vec3 offset{
        std::cos(clamped_pitch) * std::sin(yaw), -std::sin(clamped_pitch),
        std::cos(clamped_pitch) * std::cos(yaw)};
    const Vec3 eye = target + offset * distance;
    const Vec3 delta = eye - point;
    const float length = std::sqrt(dot(delta, delta));
    target = point;
    distance = std::clamp(length, 1e-3F, 1e7F);
    if (length < 1e-8F) return;
    const Vec3 direction = delta * (1.F / length);
    pitch = std::clamp(
        std::asin(std::clamp(-direction.y, -1.F, 1.F)),
        -k_pi * 0.5F, k_pi * 0.5F);
    yaw = std::atan2(direction.x, direction.z);
}

bool pick_orbit_focus_point(
    const SparseScene& scene, const OrbitCamera& camera, const ImVec2 min,
    const ImVec2 max, const ImVec2 mouse, Vec3& out_point,
    const PreviewMesh* mesh) {
    const ViewFrame frame = build_frame(camera, min, max);
    const auto pick_points = [&](const std::vector<Vec3>& points, bool placed) {
        const std::size_t count = points.size();
        constexpr std::size_t k_pick_budget = 500'000;
        const std::size_t stride =
            std::max<std::size_t>(1, (count + k_pick_budget - 1) / k_pick_budget);
        constexpr float k_radius = 12.F;
        const float radius2 = k_radius * k_radius;
        bool found = false;
        float best_depth = std::numeric_limits<float>::max();
        Vec3 best{};
        for (std::size_t i = 0; i < count; i += stride) {
            ImVec2 screen;
            float depth{};
            const auto point = placed ? scene.world_point(i) : points[i];
            if (!project(frame, point, screen, depth)) continue;
            if (screen.x < min.x || screen.x > max.x || screen.y < min.y ||
                screen.y > max.y)
                continue;
            const float dx = screen.x - mouse.x;
            const float dy = screen.y - mouse.y;
            if (dx * dx + dy * dy > radius2) continue;
            if (!found || depth < best_depth) {
                found = true;
                best_depth = depth;
                best = point;
            }
        }
        if (found) out_point = best;
        return found;
    };
    if (mesh && mesh->has() && pick_points(mesh->vertices, false)) return true;
    if (scene.has_points() && pick_points(scene.points, true)) return true;

    // No reconstructed point under the cursor: pivot on the current look-at
    // plane so a live splat pixel still focuses the orbit without a CPU hit.
    Vec3 origin;
    Vec3 dir;
    if (!camera_world_ray(camera, min, max, mouse, origin, dir)) return false;
    const float denom = dot(dir, frame.forward);
    if (std::abs(denom) < 1e-6F) return false;
    const float t = dot(camera.target - origin, frame.forward) / denom;
    if (t <= k_near_plane) return false;
    out_point = origin + dir * t;
    return true;
}

void update_orbit_camera(
    OrbitCamera& camera, const bool accepts_input, const float scene_radius,
    const bool left_button_orbits) {
    const ImGuiIO& io = ImGui::GetIO();
    if (accepts_input && io.MouseWheel != 0.F) {
        const float zoom = std::exp(-io.MouseWheel * 0.16F);
        if (camera.projection == EditorProjection::orthographic) {
            camera.ortho_height = std::clamp(
                camera.ortho_height * zoom, 1e-3F, 1e7F);
        } else if (camera.navigation == CameraNavigation::first_person || camera.navigation == CameraNavigation::fly) {
            const ViewFrame frame = build_frame(camera, {0.F, 0.F}, {1.F, 1.F});
            camera.target = camera.target + frame.forward * (io.MouseWheel * std::max(scene_radius, .01F) * .12F * camera.move_speed);
        } else {
            camera.distance = std::clamp(camera.distance * zoom, 1e-3F, 1e7F);
        }
    }

    const bool any_down = ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
                          ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                          ImGui::IsMouseDown(ImGuiMouseButton_Middle);
    if (!any_down) camera.interacting = false;
    if (accepts_input &&
        ((left_button_orbits && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) ||
         ImGui::IsMouseClicked(ImGuiMouseButton_Right) ||
         ImGui::IsMouseClicked(ImGuiMouseButton_Middle)))
        camera.interacting = true;

    const float pitch = std::clamp(camera.pitch, -k_pi * 0.5F, k_pi * 0.5F);
    const Vec3 offset{
        std::cos(pitch) * std::sin(camera.yaw), -std::sin(pitch),
        std::cos(pitch) * std::cos(camera.yaw)};
    const Vec3 forward = normalize(offset * -1.F);
    const Vec3 right = camera_right(camera, forward);
    const Vec3 up = cross(right, forward);

    if (camera.interacting) {
        const bool orbiting = left_button_orbits &&
                              ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
                              !io.KeyShift;
        const bool looking = ImGui::IsMouseDown(ImGuiMouseButton_Right);
        const ImVec2 delta = io.MouseDelta;
        if (orbiting || looking) {
            const bool eye_mode = camera.navigation == CameraNavigation::first_person || camera.navigation == CameraNavigation::fly;
            const Vec3 eye = camera.target - forward * camera.distance;
            if (camera.navigation == CameraNavigation::trackball || camera.navigation == CameraNavigation::fly) {
                Vec3 next_forward = rotate_axis(forward, up, -delta.x * .008F);
                Vec3 next_right = rotate_axis(right, up, -delta.x * .008F);
                next_forward = rotate_axis(next_forward, next_right, -delta.y * .008F);
                set_camera_basis(camera, next_forward, next_right);
            } else {
                camera.yaw -= delta.x * .008F;
                camera.pitch = std::clamp(camera.pitch + delta.y * .008F, -k_pi * .5F + .001F, k_pi * .5F - .001F);
                camera.roll = 0.F;
            }
            if (eye_mode) {
                const Vec3 next_forward{ -std::cos(camera.pitch) * std::sin(camera.yaw),
                    std::sin(camera.pitch), -std::cos(camera.pitch) * std::cos(camera.yaw)};
                camera.target = eye + next_forward * camera.distance;
            }
        } else {
            const float pan_ref =
                camera.projection == EditorProjection::orthographic
                    ? camera.ortho_height
                    : camera.distance;
            const float scale = pan_ref * 0.0018F;
            camera.target = camera.target + right * (-delta.x * scale) +
                            up * (delta.y * scale);
        }
    }

    // Fly keys deliberately require RMB so ordinary LMB orbit is not stolen.
    if (!accepts_input || !ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
        io.WantTextInput)
        return;
    const float boost = io.KeyShift ? 4.F : 1.F;
    const float base = std::max(scene_radius, camera.distance * 0.2F);
    const float step = base * camera.move_speed * boost * io.DeltaTime;
    const ViewFrame travel_frame = build_frame(camera, {0.F, 0.F}, {1.F, 1.F});
    const Vec3 travel_forward = camera.navigation == CameraNavigation::first_person
        ? Vec3{-std::sin(camera.yaw), 0.F, -std::cos(camera.yaw)} : travel_frame.forward;
    const Vec3 travel_right = camera.navigation == CameraNavigation::first_person ? orbit_right(travel_forward, camera.yaw) : travel_frame.right;
    const Vec3 travel_up = camera.navigation == CameraNavigation::first_person ? k_world_up : travel_frame.up;
    if (camera.navigation == CameraNavigation::fly) {
        const float roll_step = io.DeltaTime * (io.KeyShift ? 2.F : 1.F);
        if (ImGui::IsKeyDown(ImGuiKey_Z)) camera.roll -= roll_step;
        if (ImGui::IsKeyDown(ImGuiKey_C)) camera.roll += roll_step;
    }
    Vec3 movement;
    if (ImGui::IsKeyDown(ImGuiKey_W)) movement = movement + travel_forward;
    if (ImGui::IsKeyDown(ImGuiKey_S)) movement = movement - travel_forward;
    if (ImGui::IsKeyDown(ImGuiKey_D)) movement = movement + travel_right;
    if (ImGui::IsKeyDown(ImGuiKey_A)) movement = movement - travel_right;
    if (ImGui::IsKeyDown(ImGuiKey_E)) movement = movement + travel_up;
    if (ImGui::IsKeyDown(ImGuiKey_Q)) movement = movement - travel_up;
    if (dot(movement, movement) > 1e-8F)
        camera.target = camera.target + normalize(movement) * step;
}

void camera_view_matrix(
    const OrbitCamera& camera, const ImVec2 min, const ImVec2 max,
    std::array<float, 16>& view) {
    const ViewFrame frame = build_frame(camera, min, max);
    view = {
        frame.right.x, frame.up.x, -frame.forward.x, 0.F,
        frame.right.y, frame.up.y, -frame.forward.y, 0.F,
        frame.right.z, frame.up.z, -frame.forward.z, 0.F,
        -dot(frame.right, frame.eye), -dot(frame.up, frame.eye),
        dot(frame.forward, frame.eye), 1.F};

}

bool project_world_to_screen(
    const OrbitCamera& camera, const ImVec2 min, const ImVec2 max,
    const Vec3& world, ImVec2& screen, float& depth) {
    return project(build_frame(camera, min, max), world, screen, depth);
}

photara::io::RgbImage render_point_cloud_image(
    const SparseScene& scene, const OrbitCamera& camera,
    const ViewOptions& options, const unsigned width, const unsigned height) {
    if (width == 0 || height == 0 || width > 8192 || height > 8192 ||
        !std::isfinite(options.point_size))
        throw std::invalid_argument("Invalid point-cloud frame dimensions or point size");
    photara::io::RgbImage image;
    image.width = width; image.height = height;
    image.pixels.resize(static_cast<std::size_t>(width) * height * 3U);
    // Match the sparse viewport's vertical background gradient.
    for (unsigned y = 0; y < height; ++y) {
        const float t = (static_cast<float>(y) + .5F) / height;
        const std::array<std::uint8_t, 3> colour{
            static_cast<std::uint8_t>(std::lround(9.F + 9.F * t)),
            static_cast<std::uint8_t>(std::lround(11.F + 11.F * t)),
            static_cast<std::uint8_t>(std::lround(16.F + 16.F * t))};
        for (unsigned x = 0; x < width; ++x)
            std::copy(colour.begin(), colour.end(),
                image.pixels.begin() + (static_cast<std::size_t>(y) * width + x) * 3U);
    }
    std::vector<ProjectedCloudPoint> points;
    collect_cloud_points(scene, build_frame(camera, {0.F, 0.F},
        {static_cast<float>(width), static_cast<float>(height)}), options, points);
    const float half = std::max(.5F, options.point_size * .5F);
    const auto bound = [](float edge, unsigned size) {
        return static_cast<unsigned>(std::clamp(std::ceil(edge - .5F), 0.F, static_cast<float>(size)));
    };
    for (const auto& point : points) {
        const unsigned x0 = bound(point.x - half, width), x1 = bound(point.x + half, width);
        const unsigned y0 = bound(point.y - half, height), y1 = bound(point.y + half, height);
        const unsigned alpha = (point.colour >> IM_COL32_A_SHIFT) & 255U;
        const std::array<unsigned, 3> colour{
            (point.colour >> IM_COL32_R_SHIFT) & 255U,
            (point.colour >> IM_COL32_G_SHIFT) & 255U,
            (point.colour >> IM_COL32_B_SHIFT) & 255U};
        for (unsigned y = y0; y < y1; ++y)
            for (unsigned x = x0; x < x1; ++x)
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    auto& pixel = image.pixels[(static_cast<std::size_t>(y) * width + x) * 3U + channel];
                    pixel = static_cast<std::uint8_t>(
                        (colour[channel] * alpha + pixel * (255U - alpha) + 127U) / 255U);
                }
    }
    return image;
}

bool camera_world_ray(
    const OrbitCamera& camera, const ImVec2 min, const ImVec2 max,
    const ImVec2 mouse, Vec3& origin, Vec3& direction) {
    const ViewFrame frame = build_frame(camera, min, max);
    origin = frame.eye;
    const float u = mouse.x - frame.min.x;
    const float v = mouse.y - frame.min.y;
    const auto to_world = [&](const double cx, const double cy,
                              const double cz) {
        const float x = static_cast<float>(cx);
        const float y = static_cast<float>(cy);
        const float z = static_cast<float>(cz);
        return normalize(
            frame.right * x + frame.up * (-y) + frame.forward * z);
    };
    if (frame.projection == EditorProjection::orthographic) {
        const float scale = frame.height / frame.ortho_height;
        const float x = (u - frame.cx) / scale;
        const float y = (v - frame.cy) / scale;
        origin = frame.eye + frame.right * x + frame.up * (-y);
        direction = frame.forward;
        return true;
    }
    if (frame.projection == EditorProjection::panorama) {
        const auto ray = photara::unproject_equirectangular_camera(
            u, v, static_cast<int>(std::lround(frame.width)),
            static_cast<int>(std::lround(frame.height)));
        if (!ray.valid) return false;
        direction = to_world(ray.x, ray.y, ray.z);
        return true;
    }
    if (frame.projection == EditorProjection::fisheye) {
        const auto ray = photara::unproject_fisheye_camera(
            u, v, frame.fx, frame.fy, frame.cx, frame.cy, frame.k1, frame.k2,
            frame.k3, frame.k4);
        if (!ray.valid) return false;
        direction = to_world(ray.x, ray.y, ray.z);
        return true;
    }
    const float sx = mouse.x - frame.centre.x;
    const float sy = frame.centre.y - mouse.y;
    direction = normalize(
        frame.forward * frame.focal + frame.right * sx + frame.up * sy);
    return true;
}

void fit_reconstruction_box(
    ReconstructionBox& box, const std::vector<Vec3>& points) {
    if (box.user_set) return;
    if (points.empty()) {
        box.valid = false;
        return;
    }
    std::vector<photara::mvs::Vec3f> world;
    world.reserve(points.size());
    for (const Vec3& point : points)
        world.emplace_back(point.x, point.y, point.z);
    photara::mvs::OrientedBoundingBox bounds;
    // Same padding splat object-mode uses for SubjectBounds / focus region.
    if (!photara::mvs::detail::estimate_subject_bounds(
            world, bounds, 0, 1.15F) ||
        !bounds.valid) {
        box.valid = false;
        return;
    }
    box.min = {
        bounds.center.x() - bounds.half_extent.x(),
        bounds.center.y() - bounds.half_extent.y(),
        bounds.center.z() - bounds.half_extent.z()};
    box.max = {
        bounds.center.x() + bounds.half_extent.x(),
        bounds.center.y() + bounds.half_extent.y(),
        bounds.center.z() + bounds.half_extent.z()};
    box.valid = true;
}

bool write_reconstruction_box(
    const ReconstructionBox& box, const std::filesystem::path& path) {
    if (!box.valid || path.empty()) return false;
    photara::mvs::OrientedBoundingBox bounds;
    bounds.center = photara::mvs::Vec3f{
        (box.min.x + box.max.x) * 0.5F, (box.min.y + box.max.y) * 0.5F,
        (box.min.z + box.max.z) * 0.5F};
    bounds.half_extent = photara::mvs::Vec3f{
        std::max(1e-4F, (box.max.x - box.min.x) * 0.5F),
        std::max(1e-4F, (box.max.y - box.min.y) * 0.5F),
        std::max(1e-4F, (box.max.z - box.min.z) * 0.5F)};
    bounds.axes = photara::mvs::Mat3f::Identity();
    bounds.valid = true;
    try {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        photara::mvs::save_subject_bounds(bounds, path);
        return true;
    } catch (...) {
        return false;
    }
}

ImU32 shade_u32(const ImU32 base, const float lambert) {
    const float t = 0.28F + 0.72F * std::clamp(lambert, 0.F, 1.F);
    const ImVec4 c = ImGui::ColorConvertU32ToFloat4(base);
    return ImGui::ColorConvertFloat4ToU32(
        {c.x * t, c.y * t, c.z * t, c.w});
}

void draw_reconstruction_box(
    ImDrawList* draw, const ViewFrame& frame, const Vec3 corners[8]) {
    constexpr int faces[6][4] = {
        {0, 1, 3, 2}, {4, 6, 7, 5}, {0, 4, 5, 1},
        {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 5, 7, 3}};
    constexpr int edges[12][2] = {
        {0, 1}, {1, 3}, {3, 2}, {2, 0}, {4, 5}, {5, 7},
        {7, 6}, {6, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    const ImU32 fill = theme::u32(theme::warning, 0.07F);
    for (const auto& face : faces) {
        ImVec2 screen[4];
        float depth{};
        bool visible = true;
        for (int i = 0; i < 4; ++i) {
            visible = visible &&
                      project(frame, corners[face[i]], screen[i], depth);
        }
        if (!visible) continue;
        draw->AddConvexPolyFilled(screen, 4, fill);
    }
    const ImU32 wire = theme::u32(theme::warning, 0.82F);
    for (const auto& edge : edges)
        draw_segment(draw, frame, corners[edge[0]], corners[edge[1]], wire, 1.6F);
}

SceneDrawStats SceneRenderer::draw(
    ImDrawList* draw, const ImVec2 min, const ImVec2 max,
    const SparseScene& scene, const OrbitCamera& camera,
    const ViewOptions& options, const bool hovered,
    const ImTextureID* view_photos, const std::size_t view_photo_count,
    const PreviewMesh* mesh, const ReconstructionBox* region, const int selected_view,
    const std::span<const photara::project::CameraKeyframe> path, const int selected_key) {
    SceneDrawStats stats;
    const ViewFrame frame = build_frame(camera, min, max);
    draw->PushClipRect(min, max, true);
    const bool have_mesh = options.draw_mesh && mesh != nullptr && mesh->has();

    if (options.show_grid) {
        // Empty stage sits on world Y=0. A loaded cloud gets a floor just
        // below it so the grid does not cut through the reconstruction.
        const float plane_y = scene.has_points()
            ? scene.centroid.y + scene.radius * 1.05F
            : (have_mesh ? mesh->centroid.y + mesh->radius * 1.05F : 0.F);
        draw_ground_grid(draw, frame, camera, plane_y);
    }

    // Rings replace the centre dots when a trained Gaussian model is loaded.
    // Sparse SfM clouds have no covariance, so they always stay as points.
    const std::size_t count = scene.points.size();
    if (!options.show_cloud) {
        // Overlay-only draw (live splat / training): cameras and grid stay.
    } else if (options.draw_rings && scene.has_gaussians()) {
        stats.drawn_points =
            draw_gaussian_rings(draw, frame, scene, options, min, max);
    } else if (count > 0) {
        collect_cloud_points(scene, frame, options, scratch_);
        const float half = std::max(0.5F, options.point_size * 0.5F);
        constexpr std::size_t chunk = 4'096;
        for (std::size_t begin = 0; begin < scratch_.size(); begin += chunk) {
            const std::size_t end =
                std::min(scratch_.size(), begin + chunk);
            const int batch = static_cast<int>(end - begin);
            draw->PrimReserve(batch * 6, batch * 4);
            for (std::size_t i = begin; i < end; ++i) {
                const ProjectedCloudPoint& point = scratch_[i];
                draw->PrimRect(
                    {point.x - half, point.y - half},
                    {point.x + half, point.y + half}, point.colour);
            }
        }
        stats.drawn_points = scratch_.size();
    }

    if (have_mesh) {
        constexpr ImU32 k_clay = IM_COL32(196, 186, 174, 255);
        const bool use_vertex_colour =
            options.mesh_vertex_colour && !options.colour_by_depth &&
            mesh->colours.size() == mesh->vertices.size();
        const bool use_normals =
            mesh->normals.size() == mesh->vertices.size();
        const std::size_t face_count = mesh->faces.size();
        const std::size_t stride = 1;
        (void)options.mesh_face_budget;
        mesh_scratch_.clear();
        mesh_scratch_.reserve(face_count / stride + 1);
        float near_depth = std::numeric_limits<float>::max();
        float far_depth = 0.F;
        for (std::size_t i = 0; i < face_count; i += stride) {
            const auto& face = mesh->faces[i];
            const Vec3& v0 = mesh->vertices[face[0]];
            const Vec3& v1 = mesh->vertices[face[1]];
            const Vec3& v2 = mesh->vertices[face[2]];
            ImVec2 s0, s1, s2;
            float d0{}, d1{}, d2{};
            if (!project(frame, v0, s0, d0) || !project(frame, v1, s1, d1) ||
                !project(frame, v2, s2, d2))
                continue;
            const float min_x = std::min(s0.x, std::min(s1.x, s2.x));
            const float max_x = std::max(s0.x, std::max(s1.x, s2.x));
            const float min_y = std::min(s0.y, std::min(s1.y, s2.y));
            const float max_y = std::max(s0.y, std::max(s1.y, s2.y));
            if (max_x < min.x || min_x > max.x || max_y < min.y || min_y > max.y)
                continue;
            Vec3 n;
            if (use_normals) {
                n = mesh->normals[face[0]] + mesh->normals[face[1]] +
                    mesh->normals[face[2]];
                n = normalize(n);
            } else {
                n = normalize(cross(v1 - v0, v2 - v0));
            }
            const Vec3 centre = (v0 + v1 + v2) * (1.F / 3.F);
            const float lambert = std::abs(dot(n, normalize(frame.eye - centre)));
            ImU32 base = k_clay;
            if (use_vertex_colour) {
                const ImVec4 c0 = ImGui::ColorConvertU32ToFloat4(
                    mesh->colours[face[0]]);
                const ImVec4 c1 = ImGui::ColorConvertU32ToFloat4(
                    mesh->colours[face[1]]);
                const ImVec4 c2 = ImGui::ColorConvertU32ToFloat4(
                    mesh->colours[face[2]]);
                base = ImGui::ColorConvertFloat4ToU32(
                    {(c0.x + c1.x + c2.x) / 3.F, (c0.y + c1.y + c2.y) / 3.F,
                     (c0.z + c1.z + c2.z) / 3.F, 1.F});
            }
            const float depth = (d0 + d1 + d2) * (1.F / 3.F);
            near_depth = std::min(near_depth, depth);
            far_depth = std::max(far_depth, depth);
            mesh_scratch_.push_back({s0, s1, s2, depth, shade_u32(base, lambert)});
        }
        if (options.colour_by_depth && !mesh_scratch_.empty()) {
            const float span = std::max(1e-6F, far_depth - near_depth);
            for (MeshTri& tri : mesh_scratch_) {
                const float t = 1.F - (tri.depth - near_depth) / span;
                tri.colour = shade_u32(depth_ramp(t), 1.F);
            }
        }
        std::sort(
            mesh_scratch_.begin(), mesh_scratch_.end(),
            [](const MeshTri& a, const MeshTri& b) {
                return a.depth > b.depth;
            });
        draw->Flags |= ImDrawListFlags_AllowVtxOffset;
        constexpr std::size_t chunk = 2'048;
        const ImVec2 uv = draw->_Data->TexUvWhitePixel;
        for (std::size_t begin = 0; begin < mesh_scratch_.size();
             begin += chunk) {
            const std::size_t end =
                std::min(mesh_scratch_.size(), begin + chunk);
            draw->PrimReserve(
                static_cast<int>((end - begin) * 3),
                static_cast<int>((end - begin) * 3));
            for (std::size_t i = begin; i < end; ++i) {
                const MeshTri& tri = mesh_scratch_[i];
                draw->PrimVtx(tri.a, uv, tri.colour);
                draw->PrimVtx(tri.b, uv, tri.colour);
                draw->PrimVtx(tri.c, uv, tri.colour);
            }
        }
        if (options.mesh_wireframe) {
            const ImU32 wire = IM_COL32(18, 20, 24, 140);
            for (const MeshTri& tri : mesh_scratch_)
                draw->AddTriangle(tri.a, tri.b, tri.c, wire, 1.F);
        }
        stats.drawn_faces = mesh_scratch_.size();
    }

    if (options.show_trajectory && options.camera_overlay_source != 1 && scene.views.size() > 1) {
        const ImU32 colour = theme::u32(theme::accent, 0.18F);
        const ViewPose* previous = nullptr;
        for (const ViewPose& pose : scene.views) {
            if (!pose.registered) continue;
            if (previous)
                draw_segment(
                    draw, frame, previous->centre, pose.centre, colour, 1.F);
            previous = &pose;
        }
    }

    if (options.show_views && options.camera_overlay_source != 1) {
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        float best_distance = 18.F;
        std::vector<std::size_t> markers;
        sampled_view_indices(scene, markers);

        // Sample sparse markers; only the focused cameras expand into photo
        // surfaces. Curved geometry is cached by calibration, in local space.
        const float length = std::max(1e-4F, scene.radius * options.view_scale);
        const auto geometry = [&](const ViewPose& pose, const bool detailed) -> const CameraMarkerGeometry& {
            const auto key = marker_calibration(pose);
            auto found = std::find_if(marker_cache_.begin(), marker_cache_.end(),
                [&](const MarkerCache& c) { return c.calibration == key; });
            if (found == marker_cache_.end()) {
                if (marker_cache_.size() >= 64) marker_cache_.erase(marker_cache_.begin());
                marker_cache_.emplace_back();
                found = marker_cache_.end()-1;
                found->calibration = key;
            }
            if (!found->valid) {
                found->coarse = make_camera_marker_geometry(pose,false);
                found->valid = true;
            }
            if (detailed && !found->detailed_ready) {
                found->detailed = make_camera_marker_geometry(pose,true);
                found->detailed_ready = true;
            }
            return detailed ? found->detailed : found->coarse;
        };

        std::vector<char> drawn(scene.views.size(), 0);
        for (const std::size_t index : markers) {
            if (index < drawn.size()) drawn[index] = 1;
        }

        // Pick before drawing so exactly one camera gets the bright treatment.
        // Sampled frustums are hittable on the far-plane quad; every registered
        // camera remains hittable at its apex so a trajectory vertex can be
        // chosen even when that frustum is not in the drawn subset.
        if (hovered) {
            Vec3 ray_origin, ray_direction;
            const bool have_ray = camera_world_ray(camera,min,max,mouse,ray_origin,ray_direction);
            float best_hit = std::numeric_limits<float>::infinity();
            for (std::size_t index = 0; index < scene.views.size(); ++index) {
                const ViewPose& pose = scene.views[index];
                if (!pose.registered) continue;
                Vec3 corners[4];
                image_plane_corners(pose, length, corners);
                const bool curved = photara::uses_native_splat_projection(pose.model);
                float score = view_pick_score(frame,pose.centre,corners,mouse,drawn[index] && !curved);
                float hit = std::numeric_limits<float>::infinity();
                if (curved && have_ray && (drawn[index] || static_cast<int>(index) == selected_view)) {
                    hit = camera_marker_hit_distance(pose,length,ray_origin,ray_direction);
                    if (std::isfinite(hit)) score = std::min(score,4.F);
                }
                if (score<18.F) {
                    ImVec2 screen; float depth=std::numeric_limits<float>::infinity();
                    project(frame,pose.centre,screen,depth);
                    stats.camera_hits.push_back({static_cast<int>(index),false,score,std::isfinite(hit) ? hit : depth});
                }
                if (score < best_distance || (score == best_distance && hit < best_hit)) {
                    best_distance = score;
                    best_hit = hit;
                    stats.hovered_view = static_cast<int>(index);
                }
            }
        }
        if (selected_view >= 0 && static_cast<std::size_t>(selected_view) < scene.views.size() &&
            scene.views[selected_view].registered && drawn[selected_view] == 0) {
            markers.push_back(static_cast<std::size_t>(selected_view));
            drawn[selected_view] = 1;
        }
        if (stats.hovered_view >= 0) {
            const auto hovered_index =
                static_cast<std::size_t>(stats.hovered_view);
            if (hovered_index < drawn.size() && drawn[hovered_index] == 0)
                markers.push_back(hovered_index);
        }

        for (const std::size_t index : markers) {
            const ViewPose& pose = scene.views[index];
            const bool is_hovered = stats.hovered_view == static_cast<int>(index);
            const bool active = is_hovered || selected_view == static_cast<int>(index);
            if (photara::uses_native_splat_projection(pose.model)) {
                const ImTextureID photo = active && options.show_camera_photos && view_photos && index < view_photo_count
                    ? view_photos[index] : ImTextureID{};
                draw_curved_camera_marker(draw,frame,pose,length,geometry(pose,active),photo,
                    theme::u32(is_hovered ? theme::warning : theme::accent,active ? .9F : .26F),active,is_hovered);
                ++stats.drawn_views;
                continue;
            }
            Vec3 corners[4];
            image_plane_corners(pose, length, corners);
            const Vec3 apex = pose.centre;

            ImVec2 apex_screen;
            float apex_depth{};
            const bool apex_visible =
                project(frame, apex, apex_screen, apex_depth);

            ImVec2 corner_screen[4];
            bool plane_visible = true;
            for (int i = 0; i < 4; ++i) {
                float corner_depth{};
                plane_visible &= project(
                    frame, corners[i], corner_screen[i], corner_depth);
            }

            const ImU32 body = theme::u32(
                is_hovered ? theme::warning : theme::accent,
                active ? 0.98F : 0.24F);
            if (plane_visible) {
                ImTextureID photo{};
                if (active && options.show_camera_photos && view_photos != nullptr &&
                    index < view_photo_count)
                    photo = view_photos[index];
                if (photo) {
                    // Flip U when the plane is seen from behind so the photo
                    // stays readable from either side of the capture ring.
                    const float area =
                        (corner_screen[1].x - corner_screen[0].x) *
                            (corner_screen[3].y - corner_screen[0].y) -
                        (corner_screen[1].y - corner_screen[0].y) *
                            (corner_screen[3].x - corner_screen[0].x);
                    const ImU32 tint =
                        IM_COL32(255, 255, 255, is_hovered ? 255 : 230);
                    if (area >= 0.F) {
                        draw->AddImageQuad(
                            photo, corner_screen[0], corner_screen[1],
                            corner_screen[2], corner_screen[3], {0.F, 0.F},
                            {1.F, 0.F}, {1.F, 1.F}, {0.F, 1.F}, tint);
                    } else {
                        draw->AddImageQuad(
                            photo, corner_screen[0], corner_screen[1],
                            corner_screen[2], corner_screen[3], {1.F, 0.F},
                            {0.F, 0.F}, {0.F, 1.F}, {1.F, 1.F}, tint);
                    }
                } else {
                    draw->AddConvexPolyFilled(
                        corner_screen, 4,
                        theme::u32(
                            is_hovered ? theme::warning : theme::accent,
                            is_hovered ? 0.10F : 0.025F));
                }
            }
            const float line_width = active ? 1.6F : 0.75F;
            for (const Vec3& corner : corners)
                draw_segment(draw, frame, apex, corner, body, line_width);
            for (int i = 0; i < 4; ++i)
                draw_segment(
                    draw, frame, corners[i], corners[(i + 1) % 4], body,
                    line_width);
            // Only the hovered camera needs an up marker; repeating it for
            // every frame is the main source of the former fence-like look.
            if (active) {
                const Vec3 top_mid = (corners[0] + corners[1]) * 0.5F;
                const Vec3 plane_mid =
                    (corners[0] + corners[1] + corners[2] + corners[3]) *
                    0.25F;
                const Vec3 up =
                    plane_mid + (top_mid - plane_mid) * 1.75F;
                draw_segment(draw, frame, corners[0], up, body, line_width);
                draw_segment(draw, frame, corners[1], up, body, line_width);
            }
            if (apex_visible)
                draw->AddCircleFilled(
                    apex_screen, is_hovered ? 3.5F : 1.5F,
                    theme::u32(
                        is_hovered ? theme::warning : theme::accent,
                        is_hovered ? 0.95F : 0.38F));
            ++stats.drawn_views;
        }
    }

    if (!path.empty() && options.camera_overlay_source != 2) {
        const auto as_vec=[](const camera_recording::V3& p){return Vec3{p[0],p[1],p[2]};};
        const auto eye_for=[&](const auto& key){return Vec3{key.target[0],key.target[1],key.target[2]}-as_vec(camera_recording::forward(key))*key.distance;};
        if (options.show_trajectory && path.size()>1) {
            for (std::size_t i=1;i<path.size();++i)
                stats.drawn_path_segments+=camera_path_overlay::tessellate(path[i-1],path[i],scene.radius*.0005F,
                    [&](const auto& point,camera_path_overlay::V2& pixel) {
                        ImVec2 screen; float depth;
                        if (!project(frame,as_vec(point),screen,depth)) return false;
                        pixel={screen.x,screen.y}; return true;
                    },
                    [&](const auto& a,const auto& b) {draw_segment(draw,frame,as_vec(a),as_vec(b),theme::u32(theme::warning,.85F),2.F);});
        }
        if (options.show_views) {
            const std::size_t stride=std::max<std::size_t>(1,(path.size()+255)/256);
            const float length=std::max(1e-4F,scene.radius*options.view_scale);
            const auto shown=[&](std::size_t i){return i%stride==0 || i+1==path.size() || static_cast<int>(i)==selected_key;};
            float best=11.F;
            if (hovered) for (std::size_t i=0;i<path.size();++i) {
                ImVec2 screen; float depth;
                if (!project(frame,eye_for(path[i]),screen,depth)) continue;
                const auto mouse=ImGui::GetIO().MousePos;
                const float distance=std::hypot(screen.x-mouse.x,screen.y-mouse.y);
                if (distance<11.F) stats.camera_hits.push_back({static_cast<int>(i),true,distance,depth});
                if (distance<best) {best=distance;stats.hovered_keyframe=static_cast<int>(i);}
            }
            for (std::size_t i=0;i<path.size();++i) {
                const auto& key=path[i]; const Vec3 eye=eye_for(key);
                const bool selected=static_cast<int>(i)==selected_key, hot=static_cast<int>(i)==stats.hovered_keyframe;
                const ImU32 colour=theme::u32(selected || hot ? theme::accent : theme::warning,selected || hot ? 1.F : .7F);
                // Every key has a visible, pickable point. Only frustum details
                // are thinned in dense paths, with selected/hovered keys expanded.
                if (shown(i) || hot) {
                    const Vec3 forward=as_vec(camera_recording::forward(key)), right=as_vec(camera_recording::right(key));
                    const Vec3 up=cross(right,forward), centre=eye+forward*length;
                    const float half=length*.4F;
                    const Vec3 corners[]{centre-right*half+up*half*.65F,centre+right*half+up*half*.65F,
                                         centre+right*half-up*half*.65F,centre-right*half-up*half*.65F};
                    for (int j=0;j<4;++j) {
                        draw_segment(draw,frame,eye,corners[j],colour,selected || hot ? 2.F : 1.F);
                        draw_segment(draw,frame,corners[j],corners[(j+1)%4],colour,selected || hot ? 2.F : 1.F);
                    }
                    draw_segment(draw,frame,corners[0],centre+up*half*1.1F,colour,1.F);
                    draw_segment(draw,frame,corners[1],centre+up*half*1.1F,colour,1.F);
                }
                ImVec2 screen; float depth;
                if (project(frame,eye,screen,depth)) {
                    ++stats.drawn_keyframes;
                    draw->AddCircleFilled(screen,selected || hot ? 4.F : 2.5F,colour);
                    if (selected || hot) {
                        char label[32]; std::snprintf(label,sizeof(label),"K%zu",i+1);
                        draw->AddText({screen.x+7.F,screen.y-8.F},colour,label);
                    }
                }
            }
        }
    }

    if (options.show_region && region != nullptr && region->valid) {
        const Vec3& mn = region->min;
        const Vec3& mx = region->max;
        const Vec3 local[8] = {
            {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mn.x, mx.y, mn.z},
            {mx.x, mx.y, mn.z}, {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z},
            {mn.x, mx.y, mx.z}, {mx.x, mx.y, mx.z}};
        draw_reconstruction_box(draw, frame, local);
    }

    // World origin axes, drawn last so they stay readable. Green follows the
    // gizmo convention: -Y is up in this Y-down reconstruction world.
    if (options.show_axes) {
        const float axis =
            std::max(scene.radius * 0.25F, camera.distance * 0.08F);
        draw_segment(
            draw, frame, {0, 0, 0}, {axis, 0, 0}, IM_COL32(226, 82, 82, 200),
            1.6F);
        draw_segment(
            draw, frame, {0, 0, 0}, {0, -axis, 0}, IM_COL32(86, 202, 121, 200),
            1.6F);
        draw_segment(
            draw, frame, {0, 0, 0}, {0, 0, axis}, IM_COL32(79, 154, 235, 200),
            1.6F);
    }

    draw->PopClipRect();
    std::stable_sort(stats.camera_hits.begin(),stats.camera_hits.end(),[](const auto& a,const auto& b) {
        if (a.score!=b.score) return a.score<b.score;
        return a.depth<b.depth;
    });
    stats.hovered_view=stats.hovered_keyframe=-1;
    if (!stats.camera_hits.empty()) {
        const auto& hit=stats.camera_hits.front();
        if (hit.keyframe) stats.hovered_keyframe=hit.index; else stats.hovered_view=hit.index;
    }
    hovered_view_ = stats.hovered_view;
    return stats;
}

SplatPreviewCamera make_preview_camera_from_view(
    const ViewPose& pose, const std::uint32_t width,
    const std::uint32_t height) {
    const auto& r = pose.rotation;
    const float tx =
        -(r[0] * pose.centre.x + r[1] * pose.centre.y + r[2] * pose.centre.z);
    const float ty =
        -(r[3] * pose.centre.x + r[4] * pose.centre.y + r[5] * pose.centre.z);
    const float tz =
        -(r[6] * pose.centre.x + r[7] * pose.centre.y + r[8] * pose.centre.z);

    SplatPreviewCamera preview;
    preview.world_to_camera = {
        r[0], r[3], r[6], 0.F,
        r[1], r[4], r[7], 0.F,
        r[2], r[5], r[8], 0.F,
        tx, ty, tz, 1.F};
    preview.position = {pose.centre.x, pose.centre.y, pose.centre.z};
    const float src_w = pose.width > 0
        ? static_cast<float>(pose.width)
        : static_cast<float>(std::max<std::uint32_t>(1, width));
    const float src_h = pose.height > 0
        ? static_cast<float>(pose.height)
        : static_cast<float>(std::max<std::uint32_t>(1, height));
    const float dst_w = static_cast<float>(std::max<std::uint32_t>(1, width));
    const float dst_h = static_cast<float>(std::max<std::uint32_t>(1, height));
    preview.fx = pose.fx * (dst_w / src_w);
    preview.fy = pose.fy * (dst_h / src_h);
    preview.cx = pose.cx >= 0.F ? pose.cx * (dst_w / src_w)
                                : src_w * 0.5F * (dst_w / src_w);
    preview.cy = pose.cy >= 0.F ? pose.cy * (dst_h / src_h)
                                : src_h * 0.5F * (dst_h / src_h);
    preview.width = std::max<std::uint32_t>(1, width);
    preview.height = std::max<std::uint32_t>(1, height);
    preview.model = pose.model;
    preview.k1 = pose.k1;
    preview.k2 = pose.k2;
    preview.k3 = pose.k3;
    preview.k4 = pose.k4;
    if (preview.model == photara::CameraModel::equirectangular) {
        preview.fx = preview.fy =
            static_cast<float>(preview.width) / (2.F * k_pi);
        preview.cx = static_cast<float>(preview.width) * 0.5F;
        preview.cy = static_cast<float>(preview.height) * 0.5F;
        preview.k1 = preview.k2 = preview.k3 = preview.k4 = 0.F;
    }
    return preview;
}

SplatPreviewCamera make_preview_camera(
    const OrbitCamera& camera, const std::uint32_t width,
    const std::uint32_t height) {
    const float pitch = std::clamp(camera.pitch, -k_pi * 0.5F, k_pi * 0.5F);
    const Vec3 offset{
        std::cos(pitch) * std::sin(camera.yaw), -std::sin(pitch),
        std::cos(pitch) * std::cos(camera.yaw)};
    const Vec3 eye = camera.target + offset * camera.distance;
    const Vec3 forward = normalize(camera.target - eye);
    const Vec3 right = camera_right(camera, forward);
    const Vec3 up = cross(right, forward);
    // OpenCV / COLMAP: X right, Y down, Z forward.
    const Vec3 x = right;
    const Vec3 y{-up.x, -up.y, -up.z};
    const Vec3 z = forward;
    const float r00 = x.x, r01 = x.y, r02 = x.z;
    const float r10 = y.x, r11 = y.y, r12 = y.z;
    const float r20 = z.x, r21 = z.y, r22 = z.z;
    const float tx = -(r00 * eye.x + r01 * eye.y + r02 * eye.z);
    const float ty = -(r10 * eye.x + r11 * eye.y + r12 * eye.z);
    const float tz = -(r20 * eye.x + r21 * eye.y + r22 * eye.z);

    SplatPreviewCamera preview;
    preview.world_to_camera = {
        r00, r10, r20, 0.F,
        r01, r11, r21, 0.F,
        r02, r12, r22, 0.F,
        tx, ty, tz, 1.F};
    preview.position = {eye.x, eye.y, eye.z};
    preview.orthographic = camera.projection == EditorProjection::orthographic;
    preview.width = std::max<std::uint32_t>(1, width);
    preview.height = std::max<std::uint32_t>(1, height);
    orbit_intrinsics(
        camera, preview.width, preview.height, preview.fx, preview.fy,
        preview.cx, preview.cy, preview.model, preview.k1, preview.k2,
        preview.k3, preview.k4);
    return preview;
}

void snap_orbit_to_view(OrbitCamera& camera, const ViewPose& pose) {
    const auto& rotation = pose.rotation;
    Vec3 look{rotation[6], rotation[7], rotation[8]};
    const float look_length = std::sqrt(dot(look, look));
    if (look_length < 1e-8F) return;
    look = look * (1.F / look_length);
    camera.target = pose.centre + look * camera.distance;
    const Vec3 offset = pose.centre - camera.target;
    const float horizontal =
        std::sqrt(offset.x * offset.x + offset.z * offset.z);
    camera.pitch = std::clamp(
        std::atan2(-offset.y, std::max(horizontal, 1e-8F)),
        -k_pi * 0.5F, k_pi * 0.5F);
    camera.yaw = std::atan2(offset.x, offset.z);
    if (pose.fy > 1e-3F && pose.height > 0) {
        const float half = static_cast<float>(pose.height) * 0.5F / pose.fy;
        camera.fov_degrees = std::clamp(
            2.F * std::atan(half) * 180.F / 3.14159265F, 10.F, 120.F);
    }
}

bool write_preview_camera_file(
    const std::filesystem::path& path, const SplatPreviewCamera& camera,
    const std::uint64_t revision, const char* vis_mode,
    const float point_size_px, const float ring_scale) {
    return photara::splat::write_preview_camera_sidecar(
        path, camera, revision, vis_mode, point_size_px, ring_scale);
}

const char* editor_projection_name(const EditorProjection projection) {
    switch (projection) {
        case EditorProjection::orthographic:
            return "orthographic";
        case EditorProjection::fisheye:
            return "fisheye";
        case EditorProjection::panorama:
            return "panorama";
        case EditorProjection::perspective:
        default:
            return "perspective";
    }
}

void set_editor_projection(
    OrbitCamera& camera, const EditorProjection projection) {
    if (camera.projection == projection) return;
    const float half = std::clamp(camera.fov_degrees, 10.F, 170.F) * 0.5F *
                       k_pi / 180.F;
    const float tan_half = std::max(1e-4F, std::tan(half));
    if (camera.projection == EditorProjection::orthographic &&
        projection != EditorProjection::orthographic) {
        camera.distance = std::max(1e-3F, camera.ortho_height * 0.5F / tan_half);
    } else if (camera.projection != EditorProjection::orthographic &&
               projection == EditorProjection::orthographic) {
        camera.ortho_height =
            std::max(1e-3F, 2.F * camera.distance * tan_half);
    }
    if (projection == EditorProjection::fisheye && camera.fov_degrees < 80.F)
        camera.fov_degrees = 140.F;
    if (projection == EditorProjection::perspective && camera.fov_degrees > 120.F)
        camera.fov_degrees = 90.F;
    camera.projection = projection;
}

void set_orthographic_view(OrbitCamera& camera, const OrthographicView view) {
    camera.roll = 0.F;
    set_editor_projection(camera, EditorProjection::orthographic);
    constexpr float half_pi = k_pi * 0.5F;
    switch (view) {
        case OrthographicView::front:
            camera.yaw = 0.F;
            camera.pitch = 0.F;
            break;
        case OrthographicView::back:
            camera.yaw = k_pi;
            camera.pitch = 0.F;
            break;
        case OrthographicView::left:
            camera.yaw = -half_pi;
            camera.pitch = 0.F;
            break;
        case OrthographicView::right:
            camera.yaw = half_pi;
            camera.pitch = 0.F;
            break;
        case OrthographicView::top:
            // Positive world Y is up; the orbit pitch uses negative Y for top.
            camera.yaw = k_pi;
            camera.pitch = -half_pi;
            break;
        case OrthographicView::bottom:
            camera.yaw = k_pi;
            camera.pitch = half_pi;
            break;
    }
}

}  // namespace editor
