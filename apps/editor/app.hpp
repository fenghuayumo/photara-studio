#pragma once

#include "console_view.hpp"
#include "image_qa_view.hpp"
#include "pipeline.hpp"
#include "sam_model.hpp"
#include "moge_model.hpp"
#include "prior_preview.hpp"
#include "cache_maintenance.hpp"
#include "sparse_view.hpp"
#include "theme.hpp"
#include "viewport_gizmo.hpp"
#include "splat_edit.hpp"
#include "video_window.hpp"
#include "mask_editor.hpp"
#include "splat/deployment.hpp"
#include "vulkan_backend.hpp"
#include "splat_render/renderer.hpp"

#include "sfm/align_live.hpp"
#include "splat/visualize.hpp"
#include "splat/types.hpp"
#include "splat/checkpoint.hpp"

#include "imgui.h"
#include "shortcuts.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
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
enum class DisplayChannel { color, depth, normal };
enum class ViewportWorkspace { scene_3d, image_2d };
enum class StepState { pending, active, done, skipped, failed };
enum class InspectorSection { automatic, project, alignment, training, mesh, texture };
enum class ClearResultsAction { none, clear_view, delete_generated };
struct TrainPreparation {
    std::atomic<TrainPrepareStage> stage{TrainPrepareStage::checking};
    std::atomic<bool> cancelled{false};
    std::chrono::steady_clock::time_point started{std::chrono::steady_clock::now()};
};
enum class AlignmentExportFormat { asfm, colmap, nerfstudio, openmvs };

enum class Action {
    none,
    align,
    train,
    continue_training,
    dense,
    texture,
    export_sfm,
    pause,
    resume,
    stop,
    reveal
};

// One imported 3DGS file in the editor's multi-model list. Placement is a
// uniform scale plus translation applied when the composed preview is built
// and when models are merged to disk.
struct GaussianModelEntry {
    std::filesystem::path path;
    std::string name;
    std::uint64_t splat_count{};
    bool visible{true};
    float offset[3]{};
    float scale{1.F};
    std::shared_ptr<const std::vector<std::uint8_t>> embedded;
    std::shared_ptr<const photara::splat::GaussianModel> cached_model;
    std::string cached_key;
};

// Result of the background merge job. An empty error means success.
struct GaussianMergeJob {
    std::uint64_t splat_count{};
    std::string error;
};

struct PreparedSplat {
    std::unique_ptr<splat_render::Renderer> renderer;
    SplatEdit edit;
    std::string key;
    std::filesystem::path source;
    bool handoff{};
};

struct App {
    ProjectSettings settings;
    ProjectLayout layout;

    ProcessJob job;
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
    std::uint64_t mesh_load_generation{};
    std::uint64_t pending_mesh_load_generation{};
    bool pending_mesh_handoff{};
    bool final_mesh_handoff_seen{};
    bool final_mesh_ready{};
    gpu::MeshPreviewRenderer mesh_renderer;
    std::unique_ptr<splat_render::Renderer> splat_renderer{
        std::make_unique<splat_render::Renderer>()};
    std::future<PreparedSplat> pending_splat_model;
    std::shared_ptr<std::atomic<bool>> splat_model_cancel;
    std::vector<std::future<void>> retired_splat_models;
    std::string pending_splat_key;
    bool pending_splat_handoff{};
    bool final_handoff_seen{};
    bool final_splat_ready{};
    std::filesystem::path final_splat_source;
    SplatEdit splat_edit;
    std::vector<GaussianModelEntry> gaussian_models;
    std::shared_ptr<const photara::splat::GaussianModel> composed_model;
    std::string composed_model_key;
    struct GaussianMergeState {
        bool running{};
        bool discard{};
        bool apply_pending{};
        float progress{};
        std::uint32_t input_count{};
        std::filesystem::path output;
        GaussianMergeJob pending;
        std::shared_ptr<std::atomic<float>> progress_atom;
        std::future<GaussianMergeJob> job;
        std::string error;
    } gaussian_merge;
    std::array<VkDescriptorSet, 3> splat_preview_sets{};
    std::array<VkImageView, 3> splat_preview_views{};
    std::uint64_t splat_frames_epoch{};
    std::string splat_load_failed_key;
    std::shared_ptr<const gpu::MeshGpuData> mesh_gpu_data;
    bool mesh_gpu_uploaded{};
    bool mesh_gpu_failed{};
    OrbitCamera camera;
    ViewOptions view_options;
    ViewportGizmoState gizmo;
    ReconstructionBox reconstruction_box;
    ReconstructionBox published_reconstruction_box;
    std::filesystem::path published_subject_bounds_path;
    SceneRenderer renderer;
    std::future<SceneLoad> pending_load;
    std::future<std::string> pending_train_prepare;
    std::shared_ptr<TrainPreparation> train_preparation;
    bool train_prepare_resume{};
    bool training_inputs_ready{};
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
    // A visibility/placement change arrived while a scene load was in
    // flight; reload the composed model once that load is discarded.
    bool gaussian_composition_reload{};

    bool has_sparse{};
    bool has_asfm{};
    bool has_mvs{};
    bool has_model{};
    bool has_mesh{};
    std::filesystem::path imported_mesh_source;
    bool has_texture{};
    gpu::PreviewTexture atlas_preview;
    ImageQaPriors image_qa_priors;
    std::filesystem::path atlas_preview_path;
    std::uint32_t project_writer_version{};
    std::uint32_t project_min_reader_version{};
    // Last cache scan, refreshed at startup and whenever the Inspector asks for
    // it. Sizes come from file metadata only.
    CacheUsage cache_usage;
    VisualizationMode view_mode{VisualizationMode::points};
    DisplayChannel display_channel{DisplayChannel::color};
    splat_render::DisplayOptions display_options;
    bool show_display_legend{true};
    ViewportWorkspace workspace{ViewportWorkspace::scene_3d};
    ImageQaState image_qa;
    ImageQaSession image_qa_session;
    ImageQaSession align_match_session;
    photara::sfm::AlignLiveFrame align_live;
    bool alignment_workspace_user_override{};
    unsigned qa_preview_view{~0U};
    std::chrono::steady_clock::time_point qa_metrics_after{};
    std::uint64_t qa_camera_timeline{};
    bool qa_camera_valid{};
    std::string qa_camera_key;
    ImageQaMode qa_camera_mode{ImageQaMode::photo};
    ImageQaMode qa_raster_mode{ImageQaMode::photo};
    int qa_raster_view{-1};
    unsigned preview_view{};
    std::uint64_t preview_camera_revision{};
    // Last preview frame count published to the trainer's acknowledgement
    // sidecar, so the editor only writes it when the value actually changes.
    std::uint64_t preview_ack_frames{};
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
    bool close_approved{};
    bool show_close_confirmation{};
    bool project_results_dirty{};
    bool saved_results_discarded{};
    std::vector<std::uint8_t> saved_project_settings;
    std::uint64_t saved_edit_revision{};
    std::optional<std::filesystem::file_time_type> saved_sfm_write_time;
    std::filesystem::path pending_project_save_path;
    std::function<void()> pending_project_action;
    bool show_project_switch_confirmation{};
    bool project_transition_approved{};
    bool project_transition_ready{};
    std::optional<photara::splat::TrainingCheckpointInfo> training_checkpoint_info;
    bool saving_training{};
    bool exit_after_save{};
    bool paused_before_save{};
    std::string checkpoint_request;
    std::string checkpoint_save_error;
    std::chrono::steady_clock::time_point checkpoint_save_started{};
    bool show_controls{};
    bool show_about{};
    bool show_preferences{};
    VideoWindowState video_window;
    MaskEditorState mask_editor;
    bool preferences_raise{};
    int preferences_page{};
    bool show_sam_license{};
    bool sam_license_download{};
    std::vector<std::string> sam_license_families;
    std::vector<bool> sam_license_checks;
    SamModelDownload sam_download;
    ModelDownload moge_download;
    bool show_moge_download{};
    bool moge_license_tick{};
    bool moge_train_pending{};
    bool moge_download_train_requested{};
    bool moge_resume_pending{};
    gpu::PreviewTexture about_icon;
    bool about_icon_ready{};
    bool show_clear_results{};
    bool suppress_scene_auto_load{};
    bool pending_align_viewport_clear{};

    bool show_scene{true};
    // -1 follows the capture/project context, 0 reconstruction, 1 viewer.
    int model_workspace{-1};
    int viewer_model{-1}; // Gaussian index; -1 selects the mesh.
    bool viewer_mesh_visible{true};
    bool show_viewport{true};
    bool show_console{true};
    bool show_inspector{true};
    InspectorSection inspector_section{InspectorSection::automatic};
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
        photara::splat::LodOptions lod;
        photara::splat::CollisionOptions collision;
        int fill_mode{};
        std::future<void> job;
        std::shared_ptr<std::atomic<bool>> cancel;
        std::shared_ptr<std::atomic<float>> progress;
        std::filesystem::path output;
        std::string error;
    } splat_export;

    struct SplatSimplifyState {
        bool show{};
        float keep_ratio{0.8F};
        float min_opacity{0.10F};
        bool running{};
        float progress{};
        std::uint32_t source_count{};
        std::uint32_t result_count{};
        bool has_baseline{};
        bool applied{};
        photara::splat::GaussianModel baseline;
        std::shared_ptr<std::atomic<float>> progress_atom;
        std::shared_ptr<std::atomic<bool>> cancel;
        std::future<photara::splat::GaussianModel> job;
        std::string error;
    } splat_simplify;

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
// Editor-level cache root override, persisted next to editor.ini.
void load_editor_cache_dir(App& app);
void store_editor_cache_dir(const App& app);
// Compute choices in editor.prefs, beside editor.ini. Language stays in
// editor.language and the cache root stays in editor.cache.
void load_editor_preferences(App& app);
void store_editor_preferences(const App& app);
void select_cache_folder(App& app);
// Handshake-file folder for this process: created at startup, removed on exit.
void prepare_cache_session(const App& app);
void cleanup_cache_session(const App& app);

void store_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path);
void store_utf8_path_field(
    std::array<char, 1024>& field, const std::filesystem::path& path);
void refresh_artifacts(App& app);
void ensure_task_cache(App& app);
void apply_external_dataset_selection(App& app);
void clear_loaded_result(App& app);
void clear_viewport_scene(App& app);
void delete_reconstruction_results(App& app);

void new_project(App& app);
void select_image_folder(App& app);
void select_video_file(App& app);
void select_project_folder(App& app);
void open_project_from_path(App& app, const std::filesystem::path& path);
void save_project(App& app);
bool save_project_to_path(App& app, const std::filesystem::path& path);
void request_project_save(App& app, bool exit_after_save);
void poll_project_save(App& app);
void cancel_project_save(App& app);
void save_project_as(App& app);
bool has_unsaved_project_changes(const App& app);
void mark_project_saved(App& app);
bool defer_project_transition(App& app, std::function<void()> action, bool confirm_unsaved = true);
void complete_project_transition(App& app);
void process_project_transition(App& app);
void change_image_source(App& app, const std::filesystem::path& path);
void change_external_dataset(App& app, const std::filesystem::path& path);
void request_ascan_scene_load(App& app);
void consume_dropped_paths(App& app);

bool gaussian_models_managed(const App& app);
std::size_t visible_gaussian_model_count(const App& app);
void import_gaussian_models(
    App& app, const std::vector<std::filesystem::path>& paths,
    bool announce = true);
void remove_gaussian_model(App& app, std::size_t index);
void refresh_gaussian_composition(App& app);
void start_gaussian_merge(App& app, const std::filesystem::path& output);
void poll_gaussian_merge(App& app);

std::optional<photara::sfm::Scene> load_alignment_scene(const App& app);
void ensure_sparse_loaded(App& app);
void ensure_gaussian_scene(App& app);
void ensure_mesh_loaded(App& app);
void ensure_dataset_scene_loaded(App& app);
void poll_scene_load(App& app);
void poll_mesh_load(App& app);
void poll_alignment_preview(App& app);
void poll_align_live(App& app);
void poll_camera_photos(App& app);
void poll_splat_simplify(App& app);
void start_splat_simplify(App& app);

bool ensure_splat_renderer(App& app);
void release_splat_preview(App& app);
void release_splat_preview_sets(App& app);
void poll_splat_model_load(App& app);
void cancel_splat_model_load(App& app);
void wait_splat_model_loads(App& app);
void start_align(App& app);
void start_train(App& app, bool smoke, bool resume_training = false);
void poll_train_start(App& app);
void start_dense(App& app);
void start_texture(App& app);
void start_export_sfm(App& app);
void on_job_finished(App& app);

void set_visualization_mode(App& app, VisualizationMode mode);
void set_viewport_workspace(
    App& app, ViewportWorkspace workspace, bool user_driven = true);
void set_camera_overlays(ViewOptions& options, bool visible);
void show_mesh_view(App& app, bool frame_when_ready);
void publish_preview_vis(App& app);
void write_preview_vis(App& app);
void publish_preview_ack(App& app);
photara::splat::VisualizeOptions editor_visualize_options(const App& app);
bool image_qa_prior_matches(const App& app);
splat_render::DisplayOptions image_qa_display_options(const App& app);
void sync_live_preview_camera(
    App& app, bool force, std::uint32_t width, std::uint32_t height);
bool update_gpu_mesh_preview(App& app, ImVec2 min, ImVec2 max);

void open_mesh_export_panel(App& app);
void open_alignment_export_panel(App& app);
void open_splat_export_panel(App& app);
void export_mesh_file(App& app);
bool export_mesh_to_path(App& app, const std::filesystem::path& path);
void export_alignment(App& app);
void export_trained_model(App& app);
bool export_trained_model_to_path(App& app, const std::filesystem::path& path);
void poll_splat_export(App& app);
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
