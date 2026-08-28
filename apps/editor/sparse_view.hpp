#pragma once

#include "imgui.h"

#include "sfm/scene.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace editor {

struct Vec3 {
    float x{};
    float y{};
    float z{};
};

// One registered view from the SfM diagnostics CSV. `rotation` is the
// world-to-camera matrix in row-major order, matching sfm::Pose3D::R.
struct ViewPose {
    std::string name;
    Vec3 centre;
    std::array<float, 9> rotation{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    float fx{1.F};
    float fy{1.F};
    std::uint32_t width{};
    std::uint32_t height{};
    float reprojection_p95{};
    std::size_t observations{};
    bool registered{};
};

struct SparseScene {
    std::vector<Vec3> points;
    // Empty when the PLY has no RGB and photo sampling found no colours.
    std::vector<std::uint32_t> colours;
    std::vector<ViewPose> views;
    Vec3 centroid;
    float radius{1.F};
    std::size_t registered_views{};
    std::size_t total_views{};
    float mean_reprojection{};

    [[nodiscard]] bool has_points() const { return !points.empty(); }
    void compute_bounds();
    void clear();
};

struct SceneLoad {
    SparseScene scene;
    std::string error;
    bool ok{};
};

// Reads an ASCII or little-endian binary PLY. Poses are optional; a missing or
// unreadable CSV still yields a usable point cloud.
SceneLoad load_sparse_scene(
    std::filesystem::path cloud_ply, std::filesystem::path poses_csv);
SceneLoad sparse_scene_from_sfm(const aetherscan::sfm::Scene& scene);

struct OrbitCamera {
    float yaw{0.785398F};
    float pitch{0.61548F};
    float distance{6.F};
    Vec3 target;
    float fov_degrees{50.F};
    float move_speed{1.F};
    // Latched on press inside the viewport so a drag that wanders over the
    // side panels keeps rotating, and a drag started on a panel does not.
    bool interacting{};

    void frame(const SparseScene& scene);
};

// Rasterizer-facing camera matching splat::Camera (OpenCV +Z, Y-down, W2C
// stored column-major).
struct SplatPreviewCamera {
    std::array<float, 16> world_to_camera{};
    std::array<float, 3> position{};
    float fx{1.F};
    float fy{1.F};
    float cx{};
    float cy{};
    std::uint32_t width{};
    std::uint32_t height{};
};

SplatPreviewCamera make_preview_camera(
    const OrbitCamera& camera, std::uint32_t width, std::uint32_t height);

// Places the orbit eye at the capture pose looking along camera +Z.
void snap_orbit_to_view(OrbitCamera& camera, const ViewPose& pose);

bool write_preview_camera_file(
    const std::filesystem::path& path, const SplatPreviewCamera& camera,
    std::uint64_t revision);

// Loads registered cameras from the SfM diagnostics CSV without touching
// points. Returns false when the file is missing or has no rows.
bool load_view_poses(
    const std::filesystem::path& poses_csv, SparseScene& scene);

// Applies orbit/pan/fly mouse navigation plus WASD/QE movement while the
// viewport owns input. Holding RMB switches mouse motion to fly-look; Shift
// accelerates keyboard movement.
void update_orbit_camera(
    OrbitCamera& camera, bool accepts_input, float scene_radius);

// Column-major view matrix used to project the top-right XYZ navigation axes.
void camera_view_matrix(
    const OrbitCamera& camera, ImVec2 min, ImVec2 max,
    std::array<float, 16>& view);

struct ViewOptions {
    float point_size = 1.7F;
    int point_budget = 160'000;
    // Depth ramp is an overlay. Vertex RGB from the PLY is the default.
    bool colour_by_depth = false;
    bool show_views = true;
    bool show_trajectory = true;
    bool show_grid = true;
    float view_scale = 0.045F;
};

struct SceneDrawStats {
    std::size_t drawn_points{};
    std::size_t drawn_views{};
    // Index into SparseScene::views, or -1 when nothing is under the cursor.
    int hovered_view{-1};
};

class SceneRenderer {
public:
    SceneDrawStats draw(
        ImDrawList* draw, ImVec2 min, ImVec2 max, const SparseScene& scene,
        const OrbitCamera& camera, const ViewOptions& options,
        bool hovered);

private:
    struct Projected {
        float x{};
        float y{};
        float depth{};
    };

    std::vector<Projected> scratch_;
    std::vector<std::uint32_t> colour_scratch_;
};

}  // namespace editor
