#include "sparse_view.hpp"

#include "theme.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace editor {
namespace {

// SfM reconstructions use a Y-down world, so world "up" is -Y.
constexpr Vec3 k_world_up{0.F, -1.F, 0.F};
constexpr float k_near_plane = 1e-4F;
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
    const float half_fov = camera.fov_degrees * 0.5F * 3.14159265F / 180.F;
    frame.focal = height * 0.5F / std::max(1e-4F, std::tan(half_fov));
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

// Draws a world-space segment, dropping the part behind the near plane.
void draw_segment(
    ImDrawList* draw, const ViewFrame& frame, const Vec3 a, const Vec3 b,
    const ImU32 colour, const float thickness = 1.F) {
    ImVec2 screen_a;
    ImVec2 screen_b;
    float depth_a{};
    float depth_b{};
    const bool visible_a = project(frame, a, screen_a, depth_a);
    const bool visible_b = project(frame, b, screen_b, depth_b);
    if (!visible_a && !visible_b) return;
    if (visible_a && visible_b) {
        draw->AddLine(screen_a, screen_b, colour, thickness);
        return;
    }
    // Clip against the near plane so the segment does not wrap around.
    const Vec3 inside = visible_a ? a : b;
    const Vec3 outside = visible_a ? b : a;
    const float inside_depth = visible_a ? depth_a : depth_b;
    const float outside_depth = visible_a ? depth_b : depth_a;
    const float span = inside_depth - outside_depth;
    if (std::abs(span) < 1e-9F) return;
    const float t = (inside_depth - k_near_plane * 1.01F) / span;
    const Vec3 clipped = inside + (outside - inside) * std::clamp(t, 0.F, 1.F);
    ImVec2 screen_clipped;
    float clipped_depth{};
    if (!project(frame, clipped, screen_clipped, clipped_depth)) return;
    draw->AddLine(
        visible_a ? screen_a : screen_clipped,
        visible_a ? screen_clipped : screen_b, colour, thickness);
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

}  // namespace

void SparseScene::clear() { *this = {}; }

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

void OrbitCamera::frame(const SparseScene& scene) {
    target = scene.centroid;
    const float half_fov = fov_degrees * 0.5F * 3.14159265F / 180.F;
    distance = scene.radius / std::max(0.05F, std::tan(half_fov)) * 1.35F;
    yaw = 0.7F;
    pitch = 0.35F;
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
        // Ground plane one radius below the cloud, in the Y-down world. A
        // straight 3D line stays straight under projection, so each grid line
        // is a single near-plane-clipped segment.
        const float extent = scene.radius * 3.F;
        const float y = scene.centroid.y + scene.radius * 1.05F;
        constexpr int lines = 14;
        for (int i = -lines; i <= lines; ++i) {
            const float t = static_cast<float>(i) / lines * extent;
            const float fade =
                1.F - std::abs(static_cast<float>(i)) / (lines + 2.F);
            const ImU32 colour = theme::u32(
                i == 0 ? theme::accent : ImVec4(0.35F, 0.38F, 0.44F, 1.F),
                (i == 0 ? 0.32F : 0.15F) * fade);
            draw_segment(
                draw, frame, {scene.centroid.x + t, y, scene.centroid.z - extent},
                {scene.centroid.x + t, y, scene.centroid.z + extent}, colour);
            draw_segment(
                draw, frame, {scene.centroid.x - extent, y, scene.centroid.z + t},
                {scene.centroid.x + extent, y, scene.centroid.z + t}, colour);
        }
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

    // World origin axes, drawn last so they stay readable.
    const float axis = scene.radius * 0.25F;
    draw_segment(
        draw, frame, {0, 0, 0}, {axis, 0, 0}, IM_COL32(226, 82, 82, 200), 1.6F);
    draw_segment(
        draw, frame, {0, 0, 0}, {0, axis, 0}, IM_COL32(86, 202, 121, 200), 1.6F);
    draw_segment(
        draw, frame, {0, 0, 0}, {0, 0, axis}, IM_COL32(79, 154, 235, 200), 1.6F);

    draw->PopClipRect();
    return stats;
}

}  // namespace editor
