#pragma once

#include "imgui.h"

#include "sfm/scene.hpp"
#include "splat/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace editor {

struct Vec2 {
    float x{};
    float y{};
};

struct Vec3 {
    float x{};
    float y{};
    float z{};
};

// Detected keypoint in normalized image coordinates (origin top-left).
struct ImageFeature {
    float u{};
    float v{};
    float scale{0.01F};
    float response{};
    bool triangulated{};
};

// One registered view from the SfM diagnostics CSV. `rotation` is the
// world-to-camera matrix in row-major order, matching sfm::Pose3D::R.
struct ViewPose {
    std::string name;
    std::string camera_model{"Unknown"};
    std::filesystem::path image_path;
    Vec3 centre;
    std::array<float, 9> rotation{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
    float fx{1.F};
    float fy{1.F};
    // Principal point in pixels. Negative means unknown; consumers fall back
    // to the image centre.
    float cx{-1.F};
    float cy{-1.F};
    std::uint32_t width{};
    std::uint32_t height{};
    float reprojection_p95{};
    std::size_t observations{};
    std::size_t triangulated_features{};
    bool registered{};
    std::vector<ImageFeature> features;
};

struct GaussianPrimitive {
    Vec3 scale;
    // Scalar-first quaternion matching splat::GaussianModel::quaternions.
    std::array<float, 4> rotation{{1.F, 0.F, 0.F, 0.F}};
};

struct SparseScene {
    std::vector<Vec3> points;
    // Empty when the PLY has no RGB and photo sampling found no colours.
    std::vector<std::uint32_t> colours;
    // Populated only when the cloud came from a trained Gaussian model.
    std::vector<GaussianPrimitive> gaussians;
    std::vector<ViewPose> views;
    Vec3 centroid;
    float radius{1.F};
    std::size_t registered_views{};
    std::size_t total_views{};
    float mean_reprojection{};

    [[nodiscard]] bool has_points() const { return !points.empty(); }
    [[nodiscard]] bool has_gaussians() const {
        return !gaussians.empty() && gaussians.size() == points.size();
    }
    void compute_bounds();
    void clear();
};

struct PreviewMesh {
    std::vector<Vec3> vertices;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;
    std::vector<std::uint32_t> colours;
    std::vector<std::array<std::uint32_t, 3>> faces;
    std::filesystem::path albedo_path;
    Vec3 centroid;
    float radius{1.F};

    [[nodiscard]] bool has() const { return !faces.empty(); }
    [[nodiscard]] bool has_texture() const {
        return uvs.size() == vertices.size() && !vertices.empty();
    }
    void compute_normals();
    void compute_bounds();
    void clear() { *this = {}; }
};

struct SceneLoad {
    SparseScene scene;
    std::string error;
    bool ok{};
};

struct MeshLoad {
    PreviewMesh mesh;
    std::string error;
    bool ok{};
};

// Reads an ASCII or little-endian binary PLY. Poses are optional; a missing or
// unreadable CSV still yields a usable point cloud.
SceneLoad load_sparse_scene(
    std::filesystem::path cloud_ply, std::filesystem::path poses_csv);
SceneLoad load_gaussian_scene(
    std::filesystem::path model, std::filesystem::path poses_csv);
SceneLoad gaussian_scene_from_model(
    const aetherscan::splat::GaussianModel& model,
    std::filesystem::path poses_csv);
SceneLoad sparse_scene_from_sfm(const aetherscan::sfm::Scene& scene, bool colour_from_photos = true);

// Loads an external camera alignment dataset (COLMAP / RealityCapture /
// OpenMVS) into editor view poses. The same reader order is used by the
// trainer, so view indices line up with the training dataset. Triangulated
// landmarks are projected into each view so the 2D QA feature overlay works
// for imported alignments, which carry no per-image keypoints.
SceneLoad sparse_scene_from_dataset(
    const std::filesystem::path& source, const std::string& dataset_format,
    const std::filesystem::path& initial_point_cloud = {},
    const std::filesystem::path& image_directory = {});

// Prefers `mesh_ply` when it exists; otherwise reads the mesh chunk of `.ascan`.
MeshLoad load_preview_mesh(
    std::filesystem::path mesh_ply, std::filesystem::path ascan = {});

// Loads a baked OBJ (with UVs) and records the sibling albedo PNG path.
MeshLoad load_preview_textured_mesh(std::filesystem::path stem);

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
    void frame(const Vec3& centroid, float radius);
    // SuperSplat-style pivot: keep the current eye, orbit around `point`.
    void focus_on(const Vec3& point);
};

// Viewing / scene-alignment transform for the single reconstruction. Sparse,
// Gaussians, mesh, and cameras share it so they stay registered. It does not
// rewrite files on disk. Rotation is XYZ Euler in degrees; scale is uniform
// (RealityScan-style). Applied as p' = R S (p - pivot) + pivot + translation.
struct ReconstructionTransform {
    Vec3 translation{};
    Vec3 rotation_deg{};
    float scale{1.F};
    Vec3 pivot{};
    bool has_pivot{};

    [[nodiscard]] bool is_identity() const;
    void reset_pose();
    void ensure_pivot(const Vec3& centroid);
};

Vec3 transform_point(const ReconstructionTransform& xf, const Vec3& point);
Vec3 transform_vector(const ReconstructionTransform& xf, const Vec3& vector);
Vec3 transform_direction(const ReconstructionTransform& xf, const Vec3& vector);
Vec3 inverse_transform_point(
    const ReconstructionTransform& xf, const Vec3& world);
void reconstruction_model_matrix(
    const ReconstructionTransform& xf, std::array<float, 16>& matrix);
// Column-major TRS with origin at the pivot — the matrix ImGuizmo edits.
void reconstruction_gizmo_matrix(
    const ReconstructionTransform& xf, std::array<float, 16>& matrix);
void reconstruction_from_gizmo_matrix(
    ReconstructionTransform& xf, const std::array<float, 16>& matrix);
void multiply_mat4(
    const std::array<float, 16>& a, const std::array<float, 16>& b,
    std::array<float, 16>& out);
void reconstruction_set_pivot(
    ReconstructionTransform& xf, const Vec3& local_pivot);
bool reconstruction_pose_equal(
    const ReconstructionTransform& a, const ReconstructionTransform& b);

// Axis-aligned reconstruction volume in reconstruction space. Defaults to the
// sparse-cloud AABB; later edits will clip densify / mesh / splat work.
struct ReconstructionBox {
    Vec3 min;
    Vec3 max;
    bool valid{};
    bool user_set{};

    [[nodiscard]] Vec3 centre() const {
        return {
            (min.x + max.x) * 0.5F, (min.y + max.y) * 0.5F,
            (min.z + max.z) * 0.5F};
    }
    [[nodiscard]] Vec3 size() const {
        return {max.x - min.x, max.y - min.y, max.z - min.z};
    }
    void clear() { *this = {}; }
};

void fit_reconstruction_box(
    ReconstructionBox& box, const std::vector<Vec3>& points);
bool write_reconstruction_box(
    const ReconstructionBox& box, const std::filesystem::path& path);

bool project_world_to_screen(
    const OrbitCamera& camera, ImVec2 min, ImVec2 max, const Vec3& world,
    ImVec2& screen, float& depth);
bool camera_world_ray(
    const OrbitCamera& camera, ImVec2 min, ImVec2 max, ImVec2 mouse,
    Vec3& origin, Vec3& direction);

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

// Exact capture pose, with intrinsics scaled to the requested raster.
SplatPreviewCamera make_preview_camera_from_view(
    const ViewPose& pose, std::uint32_t width, std::uint32_t height);

// Places the orbit eye at the capture pose looking along camera +Z.
void snap_orbit_to_view(OrbitCamera& camera, const ViewPose& pose);

bool write_preview_camera_file(
    const std::filesystem::path& path, const SplatPreviewCamera& camera,
    std::uint64_t revision, const char* vis_mode = nullptr,
    float point_size_px = 2.5F, float ring_scale = 2.5F);

// Loads registered cameras from the SfM diagnostics CSV without touching
// points. Returns false when the file is missing or has no rows.
bool load_view_poses(
    const std::filesystem::path& poses_csv, SparseScene& scene);

// Fills missing ViewPose::image_path entries from `images_dir / name`.
void attach_view_image_paths(
    SparseScene& scene, const std::filesystem::path& images_dir);

constexpr std::size_t k_max_camera_markers = 32;

// Uniform subset of registered cameras drawn as frustum markers.
void sampled_view_indices(
    const SparseScene& scene, std::vector<std::size_t>& indices,
    std::size_t max_markers = k_max_camera_markers);

// Applies orbit/pan/fly mouse navigation plus WASD/QE movement while the
// viewport owns input. Holding RMB switches mouse motion to fly-look; Shift
// accelerates keyboard movement.
void update_orbit_camera(
    OrbitCamera& camera, bool accepts_input, float scene_radius);

// World point under the cursor for double-click orbit focus. Prefers a
// reconstructed / Gaussian centre near the mouse; otherwise the current
// look-at plane so a live splat pixel still has a pivot.
bool pick_orbit_focus_point(
    const SparseScene& scene, const OrbitCamera& camera, ImVec2 min,
    ImVec2 max, ImVec2 mouse, Vec3& out_point,
    const PreviewMesh* mesh = nullptr,
    const ReconstructionTransform* transform = nullptr);

// Column-major view matrix used to project the top-right XYZ navigation axes.
void camera_view_matrix(
    const OrbitCamera& camera, ImVec2 min, ImVec2 max,
    std::array<float, 16>& view);
// OpenGL perspective matching the orbit camera; used by ImGuizmo.
void camera_projection_matrix(
    const OrbitCamera& camera, ImVec2 min, ImVec2 max,
    std::array<float, 16>& projection);

struct ViewOptions {
    float point_size = 1.7F;
    int point_budget = 160'000;
    int ring_budget = 8'000;
    // SuperSplat-style contour: multiples of the projected 2D Gaussian sigma.
    float ring_scale = 2.5F;
    // Depth ramp is an overlay. Vertex RGB from the PLY is the default.
    bool colour_by_depth = false;
    bool show_cloud = true;
    bool show_views = true;
    bool show_camera_photos = true;
    bool show_trajectory = true;
    bool show_grid = true;
    bool show_axes = true;
    bool show_region = true;
    bool draw_rings = false;
    bool draw_mesh = false;
    bool mesh_wireframe = false;
    bool mesh_vertex_colour = false;
    bool mesh_texture = true;
    int mesh_face_budget = 220'000;
    float view_scale = 0.045F;
};

struct SceneDrawStats {
    std::size_t drawn_points{};
    std::size_t drawn_faces{};
    std::size_t drawn_views{};
    // Index into SparseScene::views, or -1 when nothing is under the cursor.
    int hovered_view{-1};
};

class SceneRenderer {
public:
    SceneDrawStats draw(
        ImDrawList* draw, ImVec2 min, ImVec2 max, const SparseScene& scene,
        const OrbitCamera& camera, const ViewOptions& options,
        bool hovered, const ImTextureID* view_photos = nullptr,
        std::size_t view_photo_count = 0, const PreviewMesh* mesh = nullptr,
        const ReconstructionTransform* transform = nullptr,
        const ReconstructionBox* region = nullptr);

private:
    struct Projected {
        float x{};
        float y{};
        float depth{};
    };
    struct MeshTri {
        ImVec2 a{};
        ImVec2 b{};
        ImVec2 c{};
        float depth{};
        ImU32 colour{};
    };

    std::vector<Projected> scratch_;
    std::vector<std::uint32_t> colour_scratch_;
    std::vector<MeshTri> mesh_scratch_;
};

}  // namespace editor
