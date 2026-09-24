#pragma once

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace editor {

// Alignment is the shared product of internal SfM or an imported camera
// dataset. Downstream jobs (Train 3DGS, dense MVS) consume that product and
// must not rebuild it.
enum class JobKind { none, align, train, dense, texture, export_sfm };

const char* job_name(JobKind kind);

enum class Stage {
    idle,
    features,
    matching,
    tracks,
    mapping,
    exporting,
    preparing,
    masking,
    dense,
    training,
    meshing,
    texturing,
    complete,
    failed,
};

const char* stage_name(Stage stage);

// A single ProgressReporter inside the CLI. `total == 0` means the reporter did
// not know its item count, so only the elapsed time is meaningful.
struct SubTask {
    std::string label;
    std::uint64_t completed{};
    std::uint64_t total{};
    float percent{};
    double items_per_second{};
    double eta_seconds{-1.0};
    bool active{};
};

struct TrainingStats {
    unsigned iteration{};
    unsigned total_iterations{};
    std::size_t gaussians{};
    float loss{};
    float rgb_loss{};
    float depth_loss{};
    float normal_loss{};
    float multi_view_geometry_loss{};
    float multi_view_ncc_loss{};
    float resolution_scale{1.F};
    unsigned sh_degree{};
    double step_milliseconds{};
    bool valid{};
};

// Presence flags for the artifact keys the CLI logs. Actual paths are derived
// from the project layout rather than parsed, because the logger writes paths
// through std::quoted.
struct ArtifactFlags {
    bool sparse{};
    bool mvs{};
    bool dense{};
    bool splat{};
    bool mesh{};
    bool texture{};
};

// Parsed child log line. `time` is the HH:MM:SS.mmm fragment when the logger
// prefix is present; otherwise the whole line lives in `message`.
enum class ConsoleSeverity : std::uint8_t {
    debug,
    info,
    warning,
    error,
    other
};

struct ConsoleLine {
    ConsoleSeverity severity{ConsoleSeverity::other};
    std::string time;
    std::string message;
};

struct ConsoleCounts {
    int total{};
    int info{};
    int warning{};
    int error{};
};

// Incremental reader over the redirected child stdout/stderr. Keeps a bounded
// line buffer and hands complete lines to the parser.
class LogStream {
public:
    void open(const std::filesystem::path& path);
    void close();
    void clear();
    // Drop displayed lines without rewinding the file so a live job continues.
    void clear_display();
    void poll(std::vector<std::string>& fresh_lines);

    [[nodiscard]] const std::vector<ConsoleLine>& lines() const { return lines_; }
    [[nodiscard]] const ConsoleCounts& counts() const { return counts_; }
    [[nodiscard]] std::uint64_t generation() const { return generation_; }

private:
    static constexpr std::size_t k_max_lines = 6'000;

    void push_line(std::string line);
    void trim_if_needed();
    void recount();

    std::filesystem::path path_;
    std::uintmax_t offset_{};
    std::string partial_;
    std::vector<ConsoleLine> lines_;
    ConsoleCounts counts_;
    std::uint64_t generation_{};
};

// Parses the CLI log into a stage, the active sub-task and an overall fraction.
class RunMonitor {
public:
    void begin(JobKind kind, Stage opening = Stage::idle);
    void consume(const std::string& line);
    void mark_finished(int exit_code);
    void reset();
    void pause_clock();
    void resume_clock();

    [[nodiscard]] JobKind kind() const { return kind_; }
    [[nodiscard]] Stage stage() const { return stage_; }
    [[nodiscard]] const SubTask& task() const { return task_; }
    [[nodiscard]] const TrainingStats& training() const { return training_; }
    [[nodiscard]] const ArtifactFlags& artifacts() const { return artifacts_; }
    [[nodiscard]] const std::string& last_error() const { return last_error_; }
    [[nodiscard]] bool resumed_from_cache() const { return resumed_; }

    // Overall progress of the running job. Negative means indeterminate.
    [[nodiscard]] float fraction() const;
    [[nodiscard]] double eta_seconds() const;
    // Wall time since begin(); frozen after mark_finished(). Negative if no job.
    [[nodiscard]] double elapsed_seconds() const;
    [[nodiscard]] std::string headline() const;

private:
    void enter_stage(Stage stage, float floor);

    JobKind kind_{JobKind::none};
    Stage stage_{Stage::idle};
    SubTask task_;
    TrainingStats training_;
    ArtifactFlags artifacts_;
    std::string last_error_;
    bool resumed_{};
    // Monotonic floor so a finished sub-task never lets the bar slide back.
    float floor_{};
    float band_begin_{};
    float band_end_{};
    bool has_clock_{};
    bool clock_running_{};
    std::chrono::steady_clock::time_point started_{};
    std::chrono::steady_clock::time_point stopped_{};
};

// Child process with stdout and stderr redirected to a log file. Handles are
// inherited so the trainer can adopt the shared Vulkan allocation.
class ProcessJob {
public:
    ~ProcessJob();
    ProcessJob() = default;
    ProcessJob(const ProcessJob&) = delete;
    ProcessJob& operator=(const ProcessJob&) = delete;

    void start(const std::string& command, const std::filesystem::path& log);
    void poll();
    // Suspend the CLI and every process in its job object (including ffmpeg).
    // Does not discard progress; pair with resume(). stop() still terminates.
    void pause();
    void resume();
    void stop();

    [[nodiscard]] bool running() const { return running_; }
    [[nodiscard]] bool paused() const { return paused_; }
    [[nodiscard]] int exit_code() const { return exit_code_; }
    // True on the frame the process transitions from running to finished.
    [[nodiscard]] bool consume_completion();

private:
    std::atomic_bool running_{};
    std::atomic_bool paused_{};
    std::atomic_int exit_code_{-1};
    bool completion_pending_{};
#if defined(_WIN32)
    HANDLE process_{};
    HANDLE job_{};
#endif
};

// ---------------------------------------------------------------------------

struct ProjectSettings {
    std::array<char, 1024> images_dir{};
    std::array<char, 1024> project_dir{};  // .ascan path, or legacy folder

    // Optional external SfM/camera dataset. When set, Train 3DGS bypasses
    // Photara's own image alignment and uses this dataset directly.
    std::array<char, 1024> dataset_source{};
    std::array<char, 1024> dataset_initial_cloud{};
    // Optional existing trained Gaussian file to preview/import. Empty uses
    // the working copy in cache, or a leftover sidecar from an older run.
    std::array<char, 1024> splat_model_source{};
    int splat_format = 0;  // export format: auto/PLY, PLY, SOG, SPZ, GLB

    // 0 = pinhole, 1 = OpenCV fisheye, 2 = automatic, 3 = equirectangular
    // (the values mirror photara::CameraModel so the CLI flag and the
    // persisted project setting stay in sync).
    int camera_model = 2;
    int sfm_mode = 0;  // global, incremental, hierarchical
    bool reuse_cache = false;
    // Optional cache root for the runtime working copies (sfm.bin, splat.ply,
    // mesh/dense, preview sidecars). Empty keeps them next to the project
    // (<project folder>/<project name>.cache); a set folder collects every
    // dataset's working copy there instead, which keeps the system drive free
    // and lets a dataset on a slow or read-only volume keep its cache local.
    // Editor-level only: never persisted into .ascan.
    std::array<char, 1024> cache_dir{};
    // 0 disables the automatic sweep of cache folders that no project has
    // touched for this many days.
    int cache_retention_days = 0;
    int max_features = 27'000;

    // Video capture. images_dir may be a video file; these knobs control the
    // ffmpeg extract + sharpness selection that runs before SfM. ffmpeg is
    // located automatically (PATH, next to the app, well-known installs).
    std::array<char, 1024> video_frames_dir{};
    float video_fps = 2.0F;
    int video_sharp_window = 3;
    int video_max_frames = 0;
    int video_quality = 95;
    float video_scale = 1.0F;
    int video_rotate = 0;

    bool scene_mode = false;
    int iterations = 30'000;
    int densification_cap = 1'000'000;
    int sh_degree = 3;
    int preview_interval = 50;
    int strategy = 0;  // 0=adc_igs, 1=adc_plus, 2=dense_adaptive
    int max_resolution = 1'920;
    bool progressive_resolution = true;
    bool use_mask = false;
    // 0 = masked (invalid rays are missing observations),
    // 1 = transparent (the mask is the target output alpha).
    int mask_mode = 0;
    bool sam_masks = false;
    std::array<char, 1024> sam_model{};
    std::array<char, 512> sam_text{};
    std::array<char, 512> sam_negative_text{};
    bool sam_keep_prompted = true;
    bool sam_video = true;
    int sam_max_size = 0;

    // Mesh extraction. `mesh_source` selects the product path:
    // 0 = geometry-supervised 3DGS then extract, 1 = photogrammetry (MVS).
    bool build_mesh = false;
    int mesh_source = 0;
    int mesh_method = 0;  // auto, tsdf, delaunay, pam
    float depth_normal_weight = 0.05F;
    float multi_view_geo_weight = 0.02F;
    float multi_view_ncc_weight = 0.6F;
    int geometry_from_iter = 3'000;
    // GaussianWrapping normal-field training. Off is the default GGGS path.
    bool normal_field = false;
    // Training-time colour correction for auto-exposure / auto-white-balance
    // drift. `ppisp_layout`: 0 off, nonzero on (PPISP exposure + white
    // balance). The bilateral grid adds spatially varying affine correction.
    int ppisp_layout = 0;
    bool bilateral_grid = false;

    // Texture projection after mesh extraction (photara_drender).
    // quality: 0 Fast, 1 Standard, 2 High.
    int texture_quality = 1;
    int atlas_resolution = 2048;
    bool texture_delight = false;
    bool texture_optimize = true;
    // false = sRGB blend (matches photographs). true = scene-linear blend.
    bool texture_blend_linear = false;
};

// Everything the editor reads or writes lives under the project directory.
struct ProjectLayout {
    std::filesystem::path root;
    std::filesystem::path project_file;
    std::filesystem::path cache;
    std::filesystem::path sparse_ply;
    std::filesystem::path sparse_asfm;
    std::filesystem::path sparse_mvs;
    std::filesystem::path sparse_poses;
    std::filesystem::path model_output;
    std::filesystem::path splat_ply;
    std::filesystem::path splat_sog;
    std::filesystem::path splat_spz;
    std::filesystem::path splat_glb;
    std::filesystem::path splat_model;
    std::filesystem::path mesh_ply;
    std::filesystem::path mvs_mesh_ply;
    std::filesystem::path mvs_raw_mesh_ply;
    std::filesystem::path dense_ply;
    std::filesystem::path align_log;
    std::filesystem::path train_log;
    std::filesystem::path dense_log;
    std::filesystem::path export_log;
    // Per-process handshake files: orbit camera, visualization options, preview
    // frame index/ack, the alignment live frame and its preview snapshot. They
    // are written every frame (or every 500 ms during Align) and die with the
    // session, so they live in a process-private temp folder instead of next to
    // the working copies, which outlive the run.
    std::filesystem::path session_dir;
    std::filesystem::path preview_view_file;
    std::filesystem::path preview_camera_file;
    std::filesystem::path preview_vis_file;
    // The editor writes the number of preview frames it has copied out of the
    // shared image here, so the trainer can skip a preview instead of waiting.
    std::filesystem::path preview_ack_file;
    std::filesystem::path working_sfm;
    std::filesystem::path align_live;
    std::filesystem::path align_preview;
    std::filesystem::path working_subject_bounds;
    std::filesystem::path working_splat;
    std::filesystem::path working_mesh;
    std::filesystem::path working_dense;
    // Stem for textured.obj / .mtl / _albedo.png working copies.
    std::filesystem::path working_texture;
    std::filesystem::path texture_log;
};

ProjectLayout resolve_layout(const ProjectSettings& settings);

bool is_video_source(const ProjectSettings& settings);

inline bool mesh_from_gaussians(const ProjectSettings& settings) {
    return settings.build_mesh && settings.mesh_source == 0;
}

inline bool mesh_from_mvs(const ProjectSettings& settings) {
    return settings.build_mesh && settings.mesh_source == 1;
}

inline std::filesystem::path textured_obj_path(
    const std::filesystem::path& stem) {
    auto path = stem;
    path += ".obj";
    return path;
}

inline std::filesystem::path textured_mtl_path(
    const std::filesystem::path& stem) {
    auto path = stem;
    path += ".mtl";
    return path;
}

inline std::filesystem::path textured_albedo_path(
    const std::filesystem::path& stem) {
    auto path = stem;
    path += "_albedo.png";
    return path;
}

inline bool textured_mesh_on_disk(const std::filesystem::path& stem) {
    std::error_code error;
    return !stem.empty() &&
           std::filesystem::exists(textured_obj_path(stem), error) &&
           std::filesystem::exists(textured_albedo_path(stem), error);
}

inline void apply_texture_quality_preset(ProjectSettings& settings) {
    switch (settings.texture_quality) {
        case 0:
            settings.atlas_resolution = 1024;
            settings.texture_optimize = false;
            break;
        case 2:
            settings.atlas_resolution = 4096;
            settings.texture_optimize = true;
            break;
        default:
            settings.atlas_resolution = 2048;
            settings.texture_optimize = true;
            break;
    }
}

inline int texture_optimize_steps(const ProjectSettings& settings) {
    if (settings.texture_quality == 0) return 400;
    if (settings.texture_quality == 2) return 2000;
    return 1000;
}

// Folder of stills SfM/training actually reads. Equals images_dir for a photo
// folder; the extract destination when images_dir is a video.
std::filesystem::path reconstruction_images_dir(const ProjectSettings& settings);

// Settings path fields and ImGui buffers are UTF-8. On Windows, constructing
// filesystem::path from char* uses the ACP, so these keep Chinese paths intact.
std::filesystem::path path_from_utf8_field(const char* text);
std::string path_to_utf8(const std::filesystem::path& path);

struct PreviewHandles {
    std::uint64_t memory{};
    std::uint64_t semaphore{};
    std::uint64_t allocation_size{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint64_t device_luid{};
    std::uint32_t device_node_mask{};
};

std::string build_align_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout);

std::string build_train_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout, const PreviewHandles& preview);

std::string build_splat_mesh_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout, const std::filesystem::path& splat_model);

std::string build_export_sfm_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout);

std::string build_dense_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout);

std::string build_texture_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout);

void write_preview_view_index(const ProjectLayout& layout, unsigned index);

std::string format_duration(double seconds);
std::string format_count(std::uint64_t value);

}  // namespace editor
