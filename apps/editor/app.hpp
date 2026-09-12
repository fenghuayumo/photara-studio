#pragma once

#include "console_view.hpp"
#include "image_qa_view.hpp"
#include "pipeline.hpp"
#include "sparse_view.hpp"
#include "theme.hpp"
#include "viewport_gizmo.hpp"
#include "vulkan_backend.hpp"

#include "splat/visualize.hpp"

#include "imgui.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <string>
#include <vector>

namespace editor {

inline constexpr std::uint32_t k_preview_extent = 1920;
inline constexpr float k_toolbar_height = 52.F;
inline constexpr float k_status_height = 34.F;

inline constexpr const char* k_stop_job_tooltip =
    "Abort the running job so you can change images, video, or parameters "
    "and start again.\nProgress since the last saved artifact is discarded.\n"
    "Shift+Esc";

inline constexpr const char* k_pause_job_tooltip =
    "Freeze the running job without discarding progress. Resume to continue.";

inline constexpr const char* k_busy_change_capture_tooltip =
    "Stop the running job first to choose a different image folder or video.";

enum class VisualizationMode { points, splat, rings, mesh };
enum class ViewportWorkspace { scene_3d, image_2d };
enum class StepState { pending, active, done, skipped, failed };
enum class ClearResultsAction { none, clear_view, delete_generated };
enum class AlignmentExportFormat { asfm, colmap, nerfstudio, openmvs };

enum class Action {
    none,
    align,
    train,
    dense,
    texture,
    export_sfm,
    pause,
    resume,
    stop,
    reveal
};

struct App {
    ProjectSettings settings;
    ProjectLayout layout;

    ProcessJob job;
    ProcessJob viewer;
    JobKind active_job{JobKind::none};
    RunMonitor monitor;
    LogStream log;
    ConsoleView console;
    std::vector<std::string> fresh_lines;

    gpu::ExternalPreview preview;
    gpu::CameraPhotoCache photos;

    SparseScene scene;
    PreviewMesh mesh;
    std::future<MeshLoad> pending_mesh_load;
    bool frame_mesh_on_load{};
    bool mesh_load_failed{};
    gpu::MeshPreviewRenderer mesh_renderer;
    std::vector<float> mesh_gpu_positions;
    std::vector<float> mesh_gpu_normals;
    std::vector<float> mesh_gpu_colours;
    std::vector<float> mesh_gpu_uvs;
    std::vector<std::uint32_t> mesh_gpu_indices;
    bool mesh_gpu_uploaded{};
    bool mesh_gpu_failed{};
    OrbitCamera camera;
    ViewOptions view_options;
    ViewportGizmoState gizmo;
    ReconstructionBox reconstruction_box;
    SceneRenderer renderer;
    std::future<SceneLoad> pending_load;
    std::future<SceneLoad> alignment_preview_load;
    std::filesystem::file_time_type alignment_preview_stamp{};
    std::chrono::steady_clock::time_point alignment_preview_poll{};
    bool alignment_preview_seen{};
    unsigned alignment_preview_generation{};
    unsigned alignment_preview_load_generation{};
    unsigned scene_load_generation{};
    unsigned pending_scene_load_generation{};
    bool loading_scene{};
    std::string scene_source;
    std::string dataset_scene_key;

    bool has_sparse{};
    bool has_asfm{};
    bool has_mvs{};
    bool has_model{};
    bool has_mesh{};
    bool has_texture{};
    gpu::PreviewTexture atlas_preview;
    std::filesystem::path atlas_preview_path;
    std::uint32_t project_writer_version{};
    std::uint32_t project_min_reader_version{};
    VisualizationMode view_mode{VisualizationMode::points};
    ViewportWorkspace workspace{ViewportWorkspace::scene_3d};
    ImageQaState image_qa;
    ImageQaSession image_qa_session;
    unsigned qa_preview_view{~0U};
    std::chrono::steady_clock::time_point qa_metrics_after{};
    std::uint64_t qa_camera_timeline{};
    bool qa_camera_valid{};
    unsigned preview_view{};
    std::uint64_t preview_camera_revision{};
    bool preview_follow_view{true};
    OrbitCamera last_preview_orbit{};
    bool has_last_preview_orbit{};
    std::uint32_t preview_raster_width{k_preview_extent};
    std::uint32_t preview_raster_height{k_preview_extent};

    std::string message;
    ImVec4 message_colour{theme::text_muted};

    bool smoke_mode{};
    bool smoke_started{};
    bool smoke_success{};
    bool close_requested{};
    bool show_controls{};
    bool show_clear_results{};
    bool project_folder_automatic{};
    bool suppress_scene_auto_load{};
    bool pending_align_viewport_clear{};

    bool show_scene{true};
    bool show_viewport{true};
    bool show_console{true};
    bool show_inspector{true};
    bool show_status_bar{true};
    bool reset_dock_layout{};

    struct MeshExportState {
        bool show{};
        bool initialized{};
        int format{0};
        bool include_texture{true};
    } mesh_export;

    struct AlignmentExportState {
        bool show{};
        bool initialized{};
        int format{0};
        int path_format{-1};
        bool write_ply{true};
        std::array<char, 1024> path{};
    } alignment_export;

    struct SplatExportState {
        bool show{};
        bool initialized{};
        int format{0};
        int path_format{-1};
        int sh_degree{3};
        std::array<char, 1024> path{};
    } splat_export;

    std::vector<std::string> dropped_paths;
    ImVec2 viewport_min{};
    ImVec2 viewport_max{};
    bool viewport_bounds_valid{};
};

void set_message(App& app, std::string text, const ImVec4& colour);

bool has_external_dataset(const App& app);
bool alignment_ready(const App& app);
bool alignment_job_running(const App& app);
bool waiting_for_train_preview(const App& app);
bool reconstruction_available(const App& app);
bool has_reconstruction_result(const App& app);
bool live_preview_active(const App& app);
bool mouse_over_viewport(const App& app);
bool can_export_model(const App& app);
bool can_export_mesh_file(const App& app);
bool can_export_textured_mesh(const App& app);
bool can_export_alignment(const App& app);
bool can_export_sfm(const App& app);
bool can_export_sparse(const App& app);
bool alignment_mvs_supported(const App& app);
bool mesh_export_wants_texture(const App& app);

Vec3 reconstruction_local_centroid(const App& app);
float reconstruction_local_radius(const App& app);
void ensure_reconstruction_box(App& app);
void write_working_subject_bounds(App& app);
void frame_reconstruction(App& app);

std::filesystem::path reconstruction_images_path(const App& app);
std::filesystem::path existing_splat_model(const App& app);
std::filesystem::path resolve_editor_ini();

void store_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path);
void store_utf8_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path);
void refresh_artifacts(App& app);
void assign_default_project_folder(App& app);
void apply_external_dataset_selection(App& app);
void clear_loaded_result(App& app);
void clear_viewport_scene(App& app);
void delete_reconstruction_results(App& app);

void new_project(App& app);
void select_image_folder(App& app);
void select_video_file(App& app);
void select_project_folder(App& app);
void save_project(App& app);
void save_project_as(App& app);
void request_ascan_scene_load(App& app);
void consume_dropped_paths(App& app);

void ensure_sparse_loaded(App& app);
void ensure_gaussian_scene(App& app);
void ensure_mesh_loaded(App& app);
void ensure_dataset_scene_loaded(App& app);
void poll_scene_load(App& app);
void poll_mesh_load(App& app);
void poll_alignment_preview(App& app);
void poll_camera_photos(App& app);

void stop_splat_view(App& app);
void start_splat_view(App& app);
void start_align(App& app);
void start_train(App& app, bool smoke);
void start_dense(App& app);
void start_texture(App& app);
void start_export_sfm(App& app);
void on_job_finished(App& app);

void set_visualization_mode(App& app, VisualizationMode mode);
void set_viewport_workspace(App& app, ViewportWorkspace workspace);
void set_camera_overlays(ViewOptions& options, bool visible);
void show_mesh_view(App& app, bool frame_when_ready);
void publish_preview_vis(App& app);
void write_preview_vis(App& app);
aetherscan::splat::VisualizeOptions editor_visualize_options(const App& app);
void sync_live_preview_camera(
    App& app, bool force, std::uint32_t width, std::uint32_t height);
bool update_gpu_mesh_preview(App& app, ImVec2 min, ImVec2 max);

void open_mesh_export_panel(App& app);
void open_alignment_export_panel(App& app);
void open_splat_export_panel(App& app);
void export_mesh_file(App& app);
void export_alignment(App& app);
void export_trained_model(App& app);
void sync_splat_export_path(App& app, bool force);
void sync_alignment_export_path(App& app, bool force);
bool browse_splat_export_path(App& app);
bool browse_alignment_export_path(App& app);
AlignmentExportFormat alignment_export_format(const App& app);
std::filesystem::path alignment_export_stem_name(const App& app);

const char* running_job_caption(JobKind kind);
const char* job_state_caption(const App& app);
const char* pause_job_label(JobKind kind);
const char* resume_job_label(JobKind kind);
const char* stop_job_label(JobKind kind);

}  // namespace editor
