#include "sparse_view.hpp"

#include "theme.hpp"

#include "io/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

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

// A camera-space basis plus the pixel focal length for the current viewport.
struct ViewFrame {
    Vec3 eye;
    Vec3 right;
    Vec3 up;
    Vec3 forward;
    float focal{};
    float half_x{};
    float half_y{};
    ImVec2 centre;
};

ViewFrame build_frame(
    const OrbitCamera& camera, const ImVec2 min, const ImVec2 max) {
    ViewFrame frame;
    const float pitch = std::clamp(camera.pitch, -1.53F, 1.53F);
    const Vec3 offset{
        std::cos(pitch) * std::sin(camera.yaw), -std::sin(pitch),
        std::cos(pitch) * std::cos(camera.yaw)};
    frame.eye = camera.target + offset * camera.distance;
    frame.forward = normalize(camera.target - frame.eye);
    frame.right = normalize(cross(frame.forward, k_world_up));
    frame.up = cross(frame.right, frame.forward);
    const float height = std::max(1.F, max.y - min.y);
    const float width = std::max(1.F, max.x - min.x);
    const float half_fov = camera.fov_degrees * 0.5F * 3.14159265F / 180.F;
    frame.focal = height * 0.5F / std::max(1e-4F, std::tan(half_fov));
    frame.half_x = width * 0.5F;
    frame.half_y = height * 0.5F;
    frame.centre = {(min.x + max.x) * 0.5F, (min.y + max.y) * 0.5F};
    return frame;
}

// Returns false when the point sits at or behind the near plane.
bool project(
    const ViewFrame& frame, const Vec3 world, ImVec2& screen, float& depth) {
    const Vec3 relative = world - frame.eye;
    depth = dot(relative, frame.forward);
    if (depth <= k_near_plane) return false;
    const float inverse = frame.focal / depth;
    screen.x = frame.centre.x + dot(relative, frame.right) * inverse;
    screen.y = frame.centre.y - dot(relative, frame.up) * inverse;
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
    float extent = camera.distance * 4.F;
    const float xs[3] = {-frame.half_x, 0.F, frame.half_x};
    const float ys[3] = {-frame.half_y, 0.F, frame.half_y};
    for (const float sx : xs) {
        for (const float sy : ys) {
            const Vec3 dir = normalize(
                frame.forward * frame.focal + frame.right * sx + frame.up * sy);
            if (std::abs(dir.y) < 1e-6F) continue;
            const float t = (plane_y - frame.eye.y) / dir.y;
            if (t <= k_near_plane) continue;
            const Vec3 hit = frame.eye + dir * t;
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
    if (!input) return "Cannot open " + path.filename().string();

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

    double reprojection_sum = 0.0;
    std::size_t reprojection_count = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv_row(line);
        if (fields.size() < 28) continue;
        ViewPose pose;
        pose.name = fields[1];
        pose.registered = field_as_double(fields, 2) != 0.0;
        pose.width = static_cast<std::uint32_t>(field_as_double(fields, 4));
        pose.height = static_cast<std::uint32_t>(field_as_double(fields, 5));
        pose.fx = static_cast<float>(field_as_double(fields, 6));
        pose.fy = static_cast<float>(field_as_double(fields, 7));
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
    const aetherscan::io::RgbImage& image, const float x, const float y,
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

void colour_points_from_photos(
    const aetherscan::sfm::Scene& scene, SparseScene& loaded,
    const std::vector<std::size_t>& track_ids) {
    if (loaded.points.empty() || track_ids.size() != loaded.points.size())
        return;

    std::vector<std::vector<std::pair<std::size_t, aetherscan::sfm::Index>>>
        observations_by_image(scene.images.size());
    for (std::size_t point = 0; point < track_ids.size(); ++point) {
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

    std::vector<double> sum_r(loaded.points.size());
    std::vector<double> sum_g(loaded.points.size());
    std::vector<double> sum_b(loaded.points.size());
    std::vector<std::uint32_t> samples(loaded.points.size());

    for (std::size_t image_id = 0; image_id < scene.images.size(); ++image_id) {
        if (observations_by_image[image_id].empty()) continue;
        const auto& image = scene.images[image_id];
        if (image.path.empty()) continue;
        aetherscan::io::RgbImage rgb;
        try {
            rgb = aetherscan::io::load_rgb(image.path);
        } catch (...) {
            continue;
        }
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
        for (const auto& [point, feature_id] :
             observations_by_image[image_id]) {
            if (feature_id >= image.features.keypoints.size()) continue;
            const auto& keypoint = image.features.keypoints[feature_id];
            double r = 0.0;
            double g = 0.0;
            double b = 0.0;
            if (!sample_photo_rgb(
                    rgb, keypoint.x * scale_x, keypoint.y * scale_y, r, g, b))
                continue;
            sum_r[point] += r;
            sum_g[point] += g;
            sum_b[point] += b;
            ++samples[point];
        }
    }

    loaded.colours.clear();
    loaded.colours.reserve(loaded.points.size());
    bool any = false;
    for (std::size_t i = 0; i < loaded.points.size(); ++i) {
        if (samples[i] == 0) {
            loaded.colours.push_back(IM_COL32(180, 180, 180, 255));
            continue;
        }
        any = true;
        const auto channel = [count = samples[i]](const double sum) {
            return static_cast<int>(
                std::clamp(std::lround(sum / count), 0L, 255L));
        };
        loaded.colours.push_back(
            IM_COL32(
                channel(sum_r[i]), channel(sum_g[i]), channel(sum_b[i]), 255));
    }
    if (!any) loaded.colours.clear();
}

}  // namespace

void SparseScene::clear() { *this = {}; }

SceneLoad sparse_scene_from_sfm(const aetherscan::sfm::Scene& scene) {
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
    loaded.scene.views.reserve(scene.images.size());
    for (const auto& image : scene.images) {
        ViewPose pose;
        pose.name = image.path.filename().string();
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
            pose.fx = static_cast<float>(camera.fx);
            pose.fy = static_cast<float>(camera.fy);
            pose.width = camera.width;
            pose.height = camera.height;
        }
        pose.observations = image.id < scene.image_tracks.size()
            ? scene.image_tracks[image.id].size()
            : 0;
        pose.registered = image.registered;
        if (pose.registered) ++loaded.scene.registered_views;
        loaded.scene.views.push_back(std::move(pose));
    }
    loaded.scene.total_views = scene.images.size();
    colour_points_from_photos(scene, loaded.scene, track_ids);
    loaded.scene.compute_bounds();
    return loaded;
}

void SparseScene::compute_bounds() {
    if (points.empty()) {
        centroid = {};
        radius = 1.F;
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

bool load_view_poses(
    const std::filesystem::path& poses_csv, SparseScene& scene) {
    if (!scene.views.empty()) return true;
    const std::string error = load_poses(poses_csv, scene);
    return error.empty() && !scene.views.empty();
}

void OrbitCamera::frame(const SparseScene& scene) {
    yaw = 0.785398F;
    pitch = 0.61548F;
    if (!scene.has_points()) {
        target = {};
        distance = 6.F;
        return;
    }
    target = scene.centroid;
    const float half_fov = fov_degrees * 0.5F * 3.14159265F / 180.F;
    distance = scene.radius / std::max(0.05F, std::tan(half_fov)) * 1.35F;
}

void update_orbit_camera(
    OrbitCamera& camera, const bool accepts_input, const float scene_radius) {
    const ImGuiIO& io = ImGui::GetIO();
    if (accepts_input && io.MouseWheel != 0.F)
        camera.distance = std::clamp(
            camera.distance * std::exp(-io.MouseWheel * 0.16F), 1e-3F, 1e7F);

    const bool any_down = ImGui::IsMouseDown(ImGuiMouseButton_Left) ||
                          ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
                          ImGui::IsMouseDown(ImGuiMouseButton_Middle);
    if (!any_down) camera.interacting = false;
    if (accepts_input && (ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                    ImGui::IsMouseClicked(ImGuiMouseButton_Right) ||
                    ImGui::IsMouseClicked(ImGuiMouseButton_Middle)))
        camera.interacting = true;

    const float pitch = std::clamp(camera.pitch, -1.53F, 1.53F);
    const Vec3 offset{
        std::cos(pitch) * std::sin(camera.yaw), -std::sin(pitch),
        std::cos(pitch) * std::cos(camera.yaw)};
    const Vec3 forward = normalize(offset * -1.F);
    const Vec3 right = normalize(cross(forward, k_world_up));
    const Vec3 up = cross(right, forward);

    if (camera.interacting) {
        const bool orbiting = ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
                              !io.KeyShift;
        const bool looking = ImGui::IsMouseDown(ImGuiMouseButton_Right);
        const ImVec2 delta = io.MouseDelta;
        if (orbiting || looking) {
            camera.yaw -= delta.x * 0.008F;
            camera.pitch = std::clamp(
                camera.pitch + delta.y * 0.008F, -1.53F, 1.53F);
        } else {
            const float scale = camera.distance * 0.0018F;
            camera.target = camera.target + right * (-delta.x * scale) +
                            up * (delta.y * scale);
        }
    }

    // Fly keys deliberately require RMB. This keeps W/E/R available for
    // transform-gizmo shortcuts during ordinary viewport use.
    if (!accepts_input || !ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
        io.WantTextInput)
        return;
    const float boost = io.KeyShift ? 4.F : 1.F;
    const float base = std::max(scene_radius, camera.distance * 0.2F);
    const float step = base * camera.move_speed * boost * io.DeltaTime;
    Vec3 movement;
    if (ImGui::IsKeyDown(ImGuiKey_W)) movement = movement + forward;
    if (ImGui::IsKeyDown(ImGuiKey_S)) movement = movement - forward;
    if (ImGui::IsKeyDown(ImGuiKey_D)) movement = movement + right;
    if (ImGui::IsKeyDown(ImGuiKey_A)) movement = movement - right;
    if (ImGui::IsKeyDown(ImGuiKey_E)) movement = movement + up;
    if (ImGui::IsKeyDown(ImGuiKey_Q)) movement = movement - up;
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

SceneDrawStats SceneRenderer::draw(
    ImDrawList* draw, const ImVec2 min, const ImVec2 max,
    const SparseScene& scene, const OrbitCamera& camera,
    const ViewOptions& options, const bool hovered) {
    SceneDrawStats stats;
    const ViewFrame frame = build_frame(camera, min, max);
    draw->PushClipRect(min, max, true);

    if (options.show_grid) {
        // Empty stage sits on world Y=0. A loaded cloud gets a floor just
        // below it so the grid does not cut through the reconstruction.
        const float plane_y = scene.has_points()
            ? scene.centroid.y + scene.radius * 1.05F
            : 0.F;
        draw_ground_grid(draw, frame, camera, plane_y);
    }

    // Points: project once into the scratch buffer so the depth ramp can be
    // normalised against the visible range, then emit quads in chunks.
    const std::size_t count = scene.points.size();
    if (count > 0) {
        const std::size_t stride = std::max<std::size_t>(
            1, count / std::max(1, options.point_budget) + 1);
        scratch_.clear();
        scratch_.reserve(count / stride + 1);
        float near_depth = std::numeric_limits<float>::max();
        float far_depth = 0.F;
        const bool source_colours =
            !options.colour_by_depth && !scene.colours.empty();
        colour_scratch_.clear();
        if (source_colours) colour_scratch_.reserve(count / stride + 1);

        for (std::size_t i = 0; i < count; i += stride) {
            ImVec2 screen;
            float depth{};
            if (!project(frame, scene.points[i], screen, depth)) continue;
            if (screen.x < min.x || screen.x > max.x || screen.y < min.y ||
                screen.y > max.y)
                continue;
            scratch_.push_back({screen.x, screen.y, depth});
            if (source_colours) colour_scratch_.push_back(scene.colours[i]);
            near_depth = std::min(near_depth, depth);
            far_depth = std::max(far_depth, depth);
        }

        const float span = std::max(1e-6F, far_depth - near_depth);
        const float half = std::max(0.5F, options.point_size * 0.5F);
        constexpr std::size_t chunk = 4'096;
        for (std::size_t begin = 0; begin < scratch_.size(); begin += chunk) {
            const std::size_t end =
                std::min(scratch_.size(), begin + chunk);
            const int batch = static_cast<int>(end - begin);
            draw->PrimReserve(batch * 6, batch * 4);
            for (std::size_t i = begin; i < end; ++i) {
                const Projected& point = scratch_[i];
                const ImU32 colour = source_colours
                    ? colour_scratch_[i]
                    : depth_ramp(1.F - (point.depth - near_depth) / span);
                draw->PrimRect(
                    {point.x - half, point.y - half},
                    {point.x + half, point.y + half}, colour);
            }
        }
        stats.drawn_points = scratch_.size();
    }

    if (options.show_trajectory && scene.views.size() > 1) {
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

    if (options.show_views) {
        const ImVec2 mouse = ImGui::GetIO().MousePos;
        float best_distance = 18.F;
        constexpr std::size_t max_markers = 32;
        const std::size_t registered = std::max<std::size_t>(
            1, std::count_if(
                   scene.views.begin(), scene.views.end(),
                   [](const ViewPose& pose) { return pose.registered; }));
        const std::size_t marker_stride = std::max<std::size_t>(
            1, (registered + max_markers - 1) / max_markers);

        // Pick the hovered marker before drawing so exactly one camera gets
        // the bright treatment. Only uniformly sampled cameras participate;
        // dense capture rings otherwise turn into an unreadable wire cage.
        std::size_t ordinal = 0;
        for (std::size_t index = 0; index < scene.views.size(); ++index) {
            const ViewPose& pose = scene.views[index];
            if (!pose.registered) continue;
            const bool sampled = ordinal % marker_stride == 0 ||
                                 ordinal + 1 == registered;
            ++ordinal;
            if (!sampled) continue;
            ImVec2 apex_screen;
            float apex_depth{};
            if (!hovered ||
                !project(frame, pose.centre, apex_screen, apex_depth))
                continue;
            const float dx = apex_screen.x - mouse.x;
            const float dy = apex_screen.y - mouse.y;
            const float distance = std::sqrt(dx * dx + dy * dy);
            if (distance < best_distance) {
                best_distance = distance;
                stats.hovered_view = static_cast<int>(index);
            }
        }

        // A compact frustum plus a faint image plane reads as a camera without
        // overwhelming the sparse cloud.
        const float length = std::max(1e-4F, scene.radius * options.view_scale);
        ordinal = 0;
        for (std::size_t index = 0; index < scene.views.size(); ++index) {
            const ViewPose& pose = scene.views[index];
            if (!pose.registered) continue;
            const bool sampled = ordinal % marker_stride == 0 ||
                                 ordinal + 1 == registered;
            ++ordinal;
            if (!sampled) continue;
            const auto& r = pose.rotation;
            // Camera-to-world is the transpose of the stored world-to-camera.
            const auto to_world = [&](const float x, const float y,
                                      const float z) {
                return Vec3{
                    pose.centre.x + r[0] * x + r[3] * y + r[6] * z,
                    pose.centre.y + r[1] * x + r[4] * y + r[7] * z,
                    pose.centre.z + r[2] * x + r[5] * y + r[8] * z};
            };
            const float half_x = pose.fx > 1e-3F && pose.width > 0
                ? length * (pose.width * 0.5F) / pose.fx
                : length * 0.5F;
            const float half_y = pose.fy > 1e-3F && pose.height > 0
                ? length * (pose.height * 0.5F) / pose.fy
                : length * 0.35F;
            const Vec3 apex = pose.centre;
            const Vec3 corners[4] = {
                to_world(-half_x, -half_y, length),
                to_world(half_x, -half_y, length),
                to_world(half_x, half_y, length),
                to_world(-half_x, half_y, length)};

            ImVec2 apex_screen;
            float apex_depth{};
            const bool apex_visible =
                project(frame, apex, apex_screen, apex_depth);
            const bool is_hovered =
                stats.hovered_view == static_cast<int>(index);

            ImVec2 corner_screen[4];
            bool plane_visible = true;
            for (int i = 0; i < 4; ++i) {
                float corner_depth{};
                plane_visible &= project(
                    frame, corners[i], corner_screen[i], corner_depth);
            }

            const ImU32 body = theme::u32(
                is_hovered ? theme::warning : theme::accent,
                is_hovered ? 0.98F : 0.24F);
            if (plane_visible) {
                draw->AddConvexPolyFilled(
                    corner_screen, 4,
                    theme::u32(
                        is_hovered ? theme::warning : theme::accent,
                        is_hovered ? 0.10F : 0.025F));
            }
            const float line_width = is_hovered ? 1.6F : 0.75F;
            for (const Vec3& corner : corners)
                draw_segment(draw, frame, apex, corner, body, line_width);
            for (int i = 0; i < 4; ++i)
                draw_segment(
                    draw, frame, corners[i], corners[(i + 1) % 4], body,
                    line_width);
            // Only the hovered camera needs an up marker; repeating it for
            // every frame is the main source of the former fence-like look.
            if (is_hovered) {
                const Vec3 up = to_world(0.F, -half_y * 1.75F, length);
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

    // World origin axes, drawn last so they stay readable. Green follows the
    // gizmo convention: -Y is up in this Y-down reconstruction world.
    const float axis = std::max(scene.radius * 0.25F, camera.distance * 0.08F);
    draw_segment(
        draw, frame, {0, 0, 0}, {axis, 0, 0}, IM_COL32(226, 82, 82, 200), 1.6F);
    draw_segment(
        draw, frame, {0, 0, 0}, {0, -axis, 0}, IM_COL32(86, 202, 121, 200), 1.6F);
    draw_segment(
        draw, frame, {0, 0, 0}, {0, 0, axis}, IM_COL32(79, 154, 235, 200), 1.6F);

    draw->PopClipRect();
    return stats;
}

SplatPreviewCamera make_preview_camera(
    const OrbitCamera& camera, const std::uint32_t width,
    const std::uint32_t height) {
    const float pitch = std::clamp(camera.pitch, -1.53F, 1.53F);
    const Vec3 offset{
        std::cos(pitch) * std::sin(camera.yaw), -std::sin(pitch),
        std::cos(pitch) * std::cos(camera.yaw)};
    const Vec3 eye = camera.target + offset * camera.distance;
    const Vec3 forward = normalize(camera.target - eye);
    const Vec3 right = normalize(cross(forward, k_world_up));
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
    const float half_fov = camera.fov_degrees * 0.5F * 3.14159265F / 180.F;
    const float image_height =
        static_cast<float>(std::max<std::uint32_t>(1, height));
    preview.fy = image_height * 0.5F / std::max(1e-4F, std::tan(half_fov));
    preview.fx = preview.fy;
    preview.cx = static_cast<float>(std::max<std::uint32_t>(1, width)) * 0.5F;
    preview.cy = image_height * 0.5F;
    preview.width = std::max<std::uint32_t>(1, width);
    preview.height = std::max<std::uint32_t>(1, height);
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
        std::atan2(-offset.y, std::max(horizontal, 1e-8F)), -1.53F, 1.53F);
    camera.yaw = std::atan2(offset.x, offset.z);
    if (pose.fy > 1e-3F && pose.height > 0) {
        const float half = static_cast<float>(pose.height) * 0.5F / pose.fy;
        camera.fov_degrees = std::clamp(
            2.F * std::atan(half) * 180.F / 3.14159265F, 10.F, 120.F);
    }
}

bool write_preview_camera_file(
    const std::filesystem::path& path, const SplatPreviewCamera& camera,
    const std::uint64_t revision) {
    if (path.empty()) return false;
    std::ofstream output(path, std::ios::trunc);
    if (!output) return false;
    output << std::setprecision(9);
    output << revision << '\n';
    for (std::size_t i = 0; i < camera.world_to_camera.size(); ++i) {
        if (i) output << ' ';
        output << camera.world_to_camera[i];
    }
    output << '\n'
           << camera.position[0] << ' ' << camera.position[1] << ' '
           << camera.position[2] << '\n'
           << camera.fx << ' ' << camera.fy << ' ' << camera.cx << ' '
           << camera.cy << ' ' << camera.width << ' ' << camera.height << '\n';
    return static_cast<bool>(output);
}

}  // namespace editor
