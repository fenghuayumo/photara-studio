#pragma once

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <string>
#include <vector>

namespace editor {

// The workflow the editor exposes: align cameras first, review the sparse
// result, then optimise Gaussians (optionally with the geometry supervision
// that mesh extraction needs).
enum class JobKind { none, align, train };

enum class Stage {
    idle,
    features,
    matching,
    tracks,
    mapping,
    exporting,
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

// Incremental reader over the redirected child stdout/stderr. Keeps a bounded
// console buffer and hands complete lines to the parser.
class LogStream {
public:
    void open(const std::filesystem::path& path);
    void close();
    void clear();
    void poll(std::vector<std::string>& fresh_lines);

    [[nodiscard]] const std::string& console() const { return console_; }

private:
    static constexpr std::size_t k_console_budget = 192 * 1024;

    std::filesystem::path path_;
    std::uintmax_t offset_{};
    std::string partial_;
    std::string console_;
};

// Parses the CLI log into a stage, the active sub-task and an overall fraction.
class RunMonitor {
public:
    void begin(JobKind kind);
    void consume(const std::string& line);
    void mark_finished(int exit_code);
    void reset();

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
    void stop();

    [[nodiscard]] bool running() const { return running_; }
    [[nodiscard]] int exit_code() const { return exit_code_; }
    // True on the frame the process transitions from running to finished.
    [[nodiscard]] bool consume_completion();

private:
    std::atomic_bool running_{};
    std::atomic_int exit_code_{-1};
    bool completion_pending_{};
#if defined(_WIN32)
    HANDLE process_{};
#endif
};

// ---------------------------------------------------------------------------

struct ProjectSettings {
    std::array<char, 1024> images_dir{};
    std::array<char, 1024> project_dir{};

    int sfm_mode = 0;  // global, incremental, hierarchical
    bool reuse_cache = true;
    int max_features = 27'000;

    bool scene_mode = false;
    int iterations = 30'000;
    int preview_interval = 50;
    int strategy = 1;  // default, adc_plus, adc_igs, dense_adaptive
    int max_resolution = 1'920;
    bool progressive_resolution = true;
    bool use_mask = false;

    // Mesh extraction. Enabling it switches the trainer onto the depth/normal
    // and multi-view geometry objectives that a watertight surface needs.
    bool build_mesh = false;
    int mesh_method = 0;  // auto, tsdf, pam
    float depth_normal_weight = 0.05F;
    float multi_view_geo_weight = 0.02F;
    float multi_view_ncc_weight = 0.6F;
    int geometry_from_iter = 3'000;
    bool normal_field = true;
};

// Everything the editor reads or writes lives under the project directory.
struct ProjectLayout {
    std::filesystem::path root;
    std::filesystem::path cache;
    std::filesystem::path sparse_ply;
    std::filesystem::path sparse_poses;
    std::filesystem::path model_output;
    std::filesystem::path splat_ply;
    std::filesystem::path mesh_ply;
    std::filesystem::path align_log;
    std::filesystem::path train_log;
};

ProjectLayout resolve_layout(const ProjectSettings& settings);

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

std::string format_duration(double seconds);
std::string format_count(std::uint64_t value);

}  // namespace editor
