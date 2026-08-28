#include "pipeline.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace editor {
namespace {

// Maps a ProgressReporter label onto a stage and a slice of the overall bar.
// Longer, more specific prefixes must precede their shorter relatives.
struct Band {
    std::string_view prefix;
    Stage stage;
    float begin;
    float end;
};

constexpr Band k_align_bands[] = {
    {"extract features", Stage::features, 0.03F, 0.34F},
    {"prepare image pairs", Stage::matching, 0.34F, 0.36F},
    {"fast match image pairs", Stage::matching, 0.36F, 0.52F},
    {"lightglue match pairs", Stage::matching, 0.36F, 0.56F},
    {"rescue difficult pairs", Stage::matching, 0.52F, 0.56F},
    {"match image pairs", Stage::matching, 0.36F, 0.56F},
    {"verify pair geometry", Stage::matching, 0.56F, 0.70F},
    {"build tracks", Stage::tracks, 0.70F, 0.77F},
    {"sfm.retrieve_image_pairs", Stage::matching, 0.34F, 0.36F},
    {"register images", Stage::mapping, 0.77F, 0.96F},
};

// On the training run SfM is normally a checkpoint hit, so it only owns a thin
// slice at the front.
constexpr Band k_train_bands[] = {
    {"extract features", Stage::features, 0.01F, 0.04F},
    {"match image pairs", Stage::matching, 0.04F, 0.07F},
    {"fast match image pairs", Stage::matching, 0.04F, 0.07F},
    {"lightglue match pairs", Stage::matching, 0.04F, 0.07F},
    {"verify pair geometry", Stage::matching, 0.07F, 0.08F},
    {"build tracks", Stage::tracks, 0.08F, 0.09F},
    {"register images", Stage::mapping, 0.09F, 0.11F},
    {"mvs.load_images", Stage::meshing, 0.88F, 0.90F},
    {"mvs.estimate_depth", Stage::dense, 0.12F, 0.40F},
    {"mvs.geometric_consistency", Stage::dense, 0.40F, 0.50F},
    {"mvs.filter_depth", Stage::dense, 0.50F, 0.55F},
    {"texture.", Stage::texturing, 0.98F, 1.00F},
};

constexpr float k_training_band_begin = 0.12F;
constexpr float k_training_band_end = 0.88F;
constexpr float k_meshing_band_begin = 0.88F;

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() &&
           text.compare(0, prefix.size(), prefix) == 0;
}

// Reads the token following `key` up to the next space.
bool token_after(
    const std::string& line, const std::string_view key, std::string& out) {
    const std::size_t found = line.find(key);
    if (found == std::string::npos) return false;
    const std::size_t begin = found + key.size();
    const std::size_t end = line.find(' ', begin);
    out = line.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    return !out.empty();
}

template <typename T>
bool number_after(
    const std::string& line, const std::string_view key, T& out) {
    std::string token;
    if (!token_after(line, key, token)) return false;
    std::istringstream stream(token);
    T parsed{};
    stream >> parsed;
    if (stream.fail()) return false;
    out = parsed;
    return true;
}

// Parses "a/b" following `key`.
bool ratio_after(
    const std::string& line, const std::string_view key, std::uint64_t& first,
    std::uint64_t& second) {
    std::string token;
    if (!token_after(line, key, token)) return false;
    const std::size_t slash = token.find('/');
    if (slash == std::string::npos) return false;
    char* end{};
    first = std::strtoull(token.c_str(), &end, 10);
    second = std::strtoull(token.c_str() + slash + 1, nullptr, 10);
    return true;
}

std::string quote(const std::filesystem::path& path) {
    return '"' + path.string() + '"';
}

const char* strategy_flag(const int index) {
    switch (index) {
        case 0: return "default";
        case 2: return "adc_igs";
        case 3: return "dense_adaptive";
        default: return "adc_plus";
    }
}

const char* mesh_method_flag(const int index) {
    switch (index) {
        case 1: return "tsdf";
        case 2: return "delaunay";
        case 3: return "pam";
        default: return "auto";
    }
}

const char* sfm_mode_flag(const int index) {
    switch (index) {
        case 1: return "incremental";
        case 2: return "hierarchical";
        default: return "global";
    }
}

}  // namespace

const char* stage_name(const Stage stage) {
    switch (stage) {
        case Stage::idle: return "Idle";
        case Stage::features: return "Extracting features";
        case Stage::matching: return "Matching views";
        case Stage::tracks: return "Building tracks";
        case Stage::mapping: return "Solving camera poses";
        case Stage::exporting: return "Writing sparse scene";
        case Stage::dense: return "Dense MVS";
        case Stage::training: return "Training Gaussians";
        case Stage::meshing: return "Extracting mesh";
        case Stage::texturing: return "Baking texture";
        case Stage::complete: return "Complete";
        case Stage::failed: return "Failed";
    }
    return "Idle";
}

// ---------------------------------------------------------------------------
// LogStream

void LogStream::open(const std::filesystem::path& path) {
    path_ = path;
    offset_ = 0;
    partial_.clear();
    console_.clear();
}

void LogStream::close() {
    path_.clear();
    offset_ = 0;
    partial_.clear();
}

void LogStream::clear() {
    close();
    console_.clear();
}

void LogStream::poll(std::vector<std::string>& fresh_lines) {
    fresh_lines.clear();
    if (path_.empty()) return;
    std::error_code error;
    const auto size = std::filesystem::file_size(path_, error);
    if (error) return;
    // A restarted job truncates the file; rewind rather than reading garbage.
    if (size < offset_) {
        offset_ = 0;
        partial_.clear();
        console_.clear();
    }
    if (size == offset_) return;

    std::ifstream input(path_, std::ios::binary);
    if (!input) return;
    input.seekg(static_cast<std::streamoff>(offset_));
    std::string chunk;
    chunk.resize(static_cast<std::size_t>(size - offset_));
    input.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    chunk.resize(static_cast<std::size_t>(std::max<std::streamsize>(
        0, input.gcount())));
    offset_ += chunk.size();
    if (chunk.empty()) return;

    console_ += chunk;
    if (console_.size() > k_console_budget) {
        const std::size_t excess = console_.size() - k_console_budget;
        const std::size_t line_break = console_.find('\n', excess);
        console_.erase(
            0, line_break == std::string::npos ? excess : line_break + 1);
    }

    partial_ += chunk;
    std::size_t begin = 0;
    while (true) {
        const std::size_t newline = partial_.find('\n', begin);
        if (newline == std::string::npos) break;
        std::string line = partial_.substr(begin, newline - begin);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) fresh_lines.push_back(std::move(line));
        begin = newline + 1;
    }
    partial_.erase(0, begin);
}

// ---------------------------------------------------------------------------
// RunMonitor

void RunMonitor::reset() {
    kind_ = JobKind::none;
    stage_ = Stage::idle;
    task_ = {};
    training_ = {};
    artifacts_ = {};
    last_error_.clear();
    resumed_ = false;
    floor_ = 0.F;
    band_begin_ = 0.F;
    band_end_ = 0.F;
}

void RunMonitor::begin(const JobKind kind) {
    reset();
    kind_ = kind;
    stage_ = kind == JobKind::align ? Stage::features : Stage::training;
    band_begin_ = 0.F;
    band_end_ = 0.F;
}

void RunMonitor::enter_stage(const Stage stage, const float floor) {
    stage_ = stage;
    floor_ = std::max(floor_, floor);
}

void RunMonitor::consume(const std::string& line) {
    if (line.find("[error]") != std::string::npos) {
        // Strip the timestamp/level prefix for a readable status message.
        const std::size_t level_end = line.find("] ");
        last_error_ = level_end == std::string::npos
            ? line
            : line.substr(level_end + 2);
    }

    if (line.find("checkpoint hit: reconstruction") != std::string::npos ||
        line.find("checkpoint hit: tracks") != std::string::npos)
        resumed_ = true;

    // Artifact keys. Only presence matters; paths come from the layout.
    if (line.find("sfm_diagnostics=") != std::string::npos)
        artifacts_.sparse = true;
    if (line.find("OpenMVS export:") != std::string::npos ||
        line.find(" mvs=") != std::string::npos)
        artifacts_.mvs = true;
    if (line.find("dense_ply=") != std::string::npos) artifacts_.dense = true;
    if (line.find("splat_ply=") != std::string::npos) {
        artifacts_.splat = true;
        enter_stage(Stage::meshing, k_meshing_band_begin);
    }
    if (line.find("mesh_ply=") != std::string::npos ||
        line.find("mvs_mesh_ply=") != std::string::npos)
        artifacts_.mesh = true;
    if (line.find("textured_obj=") != std::string::npos)
        artifacts_.texture = true;

    // Per-iteration training line carries its own totals.
    if (const std::size_t iteration_key = line.find("splat iteration=");
        iteration_key != std::string::npos) {
        std::uint64_t current{};
        std::uint64_t total{};
        if (ratio_after(line, "splat iteration=", current, total)) {
            training_.iteration = static_cast<unsigned>(current);
            training_.total_iterations = static_cast<unsigned>(total);
            training_.valid = true;
            number_after(line, " gaussians=", training_.gaussians);
            number_after(line, " loss=", training_.loss);
            number_after(line, " rgb=", training_.rgb_loss);
            number_after(line, " depth=", training_.depth_loss);
            number_after(line, " normal=", training_.normal_loss);
            number_after(
                line, " mv_geo=", training_.multi_view_geometry_loss);
            number_after(line, " mv_ncc=", training_.multi_view_ncc_loss);
            number_after(
                line, " resolution_scale=", training_.resolution_scale);
            number_after(line, " sh_degree=", training_.sh_degree);
            number_after(line, " step_ms=", training_.step_milliseconds);
            enter_stage(Stage::training, k_training_band_begin);
            task_ = {};
        }
        return;
    }

    if (line.find("stage started: ") != std::string::npos) {
        std::string name;
        if (token_after(line, "stage started: ", name)) {
            if (name == "sfm.frontend")
                enter_stage(Stage::features, 0.02F);
            else if (name.rfind("sfm.", 0) == 0 &&
                     name.find("mapping") != std::string::npos)
                enter_stage(
                    Stage::mapping,
                    kind_ == JobKind::align ? 0.77F : 0.09F);
            else if (
                name == "splat.mesh_render_geometry" || name == "splat.pam" ||
                name.rfind("mvs.mesh", 0) == 0 || name == "mvs.fuse")
                enter_stage(Stage::meshing, k_meshing_band_begin);
            else if (name.rfind("texture.", 0) == 0)
                enter_stage(Stage::texturing, 0.98F);
        }
        return;
    }

    if (line.find("frontend: images=") != std::string::npos ||
        line.find("frontend diagnostics:") != std::string::npos) {
        enter_stage(
            Stage::mapping, kind_ == JobKind::align ? 0.77F : 0.09F);
        task_ = {};
        return;
    }

    if (line.find("sfm_diagnostics=") != std::string::npos &&
        kind_ == JobKind::align) {
        enter_stage(Stage::exporting, 0.97F);
        task_ = {};
        return;
    }

    // ProgressReporter lines. "progress started:" only announces the total.
    const bool finished = line.find("progress finished: ") != std::string::npos;
    const std::size_t running = line.find("progress: ");
    if (!finished && running == std::string::npos) return;

    const std::string_view marker =
        finished ? "progress finished: " : "progress: ";
    const std::size_t marker_at = line.find(marker);
    const std::size_t label_begin = marker_at + marker.size();
    const std::size_t metrics_at = line.find(" completed=", label_begin);
    if (metrics_at == std::string::npos) return;
    const std::string label = line.substr(label_begin, metrics_at - label_begin);

    const Band* bands = kind_ == JobKind::align ? k_align_bands : k_train_bands;
    const std::size_t band_count = kind_ == JobKind::align
        ? std::size(k_align_bands)
        : std::size(k_train_bands);
    const Band* match = nullptr;
    for (std::size_t i = 0; i < band_count; ++i)
        if (starts_with(label, bands[i].prefix)) {
            match = &bands[i];
            break;
        }

    task_.label = label;
    task_.active = !finished;
    ratio_after(line, " completed=", task_.completed, task_.total);
    number_after(line, " percent=", task_.percent);
    number_after(line, " items/s=", task_.items_per_second);
    std::string eta;
    task_.eta_seconds = -1.0;
    if (token_after(line, " eta_s=", eta) && eta != "n/a")
        task_.eta_seconds = std::strtod(eta.c_str(), nullptr);

    if (!match) return;
    enter_stage(match->stage, match->begin);
    band_begin_ = match->begin;
    band_end_ = match->end;
    if (finished) {
        floor_ = std::max(floor_, match->end);
        task_.active = false;
    }
}

void RunMonitor::mark_finished(const int exit_code) {
    task_.active = false;
    stage_ = exit_code == 0 ? Stage::complete : Stage::failed;
    if (exit_code == 0) floor_ = 1.F;
}

float RunMonitor::fraction() const {
    if (stage_ == Stage::complete) return 1.F;
    if (stage_ == Stage::training && training_.valid &&
        training_.total_iterations > 0) {
        const float ratio = static_cast<float>(training_.iteration) /
                            static_cast<float>(training_.total_iterations);
        return std::max(
            floor_, k_training_band_begin +
                        std::clamp(ratio, 0.F, 1.F) *
                            (k_training_band_end - k_training_band_begin));
    }
    if (task_.active && task_.total > 0 && band_end_ > band_begin_) {
        const float ratio = std::clamp(task_.percent / 100.F, 0.F, 1.F);
        return std::max(
            floor_, band_begin_ + ratio * (band_end_ - band_begin_));
    }
    if (floor_ > 0.F) return floor_;
    return -1.F;
}

double RunMonitor::eta_seconds() const {
    if (stage_ == Stage::training && training_.valid &&
        training_.total_iterations > training_.iteration &&
        training_.step_milliseconds > 0.0)
        return (training_.total_iterations - training_.iteration) *
               training_.step_milliseconds / 1000.0;
    if (task_.active) return task_.eta_seconds;
    return -1.0;
}

std::string RunMonitor::headline() const {
    std::string text = stage_name(stage_);
    if (stage_ == Stage::training && training_.valid) {
        text += "  ";
        text += std::to_string(training_.iteration);
        text += " / ";
        text += std::to_string(training_.total_iterations);
    } else if (task_.active && !task_.label.empty()) {
        text += "  ·  ";
        text += task_.label;
    }
    return text;
}

// ---------------------------------------------------------------------------
// ProcessJob

ProcessJob::~ProcessJob() { stop(); }

void ProcessJob::start(
    const std::string& command, const std::filesystem::path& log) {
    if (running_.exchange(true)) return;
    exit_code_ = -1;
    completion_pending_ = false;
#if defined(_WIN32)
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE log_handle = CreateFileW(
        log.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
        &security, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log_handle == INVALID_HANDLE_VALUE) {
        running_ = false;
        throw std::runtime_error("Cannot create reconstruction log");
    }
    const int count =
        MultiByteToWideChar(CP_UTF8, 0, command.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> mutable_command(static_cast<std::size_t>(count));
    MultiByteToWideChar(
        CP_UTF8, 0, command.c_str(), -1, mutable_command.data(), count);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = log_handle;
    startup.hStdError = log_handle;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION info{};
    const BOOL created = CreateProcessW(
        nullptr, mutable_command.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &info);
    CloseHandle(log_handle);
    if (!created) {
        running_ = false;
        throw std::runtime_error(
            "CreateProcess failed: " + std::to_string(GetLastError()));
    }
    process_ = info.hProcess;
    CloseHandle(info.hThread);
#else
    static_cast<void>(command);
    running_ = false;
    throw std::runtime_error(
        "External-memory editor launch is currently Win32-only");
#endif
}

void ProcessJob::poll() {
#if defined(_WIN32)
    if (!running_ || !process_) return;
    DWORD code = STILL_ACTIVE;
    if (GetExitCodeProcess(process_, &code) && code != STILL_ACTIVE) {
        exit_code_ = static_cast<int>(code);
        running_ = false;
        completion_pending_ = true;
        CloseHandle(std::exchange(process_, nullptr));
    }
#endif
}

void ProcessJob::stop() {
#if defined(_WIN32)
    if (process_) {
        TerminateProcess(process_, 2);
        WaitForSingleObject(process_, INFINITE);
        CloseHandle(std::exchange(process_, nullptr));
        exit_code_ = 2;
        completion_pending_ = true;
    }
#endif
    running_ = false;
}

bool ProcessJob::consume_completion() {
    if (!completion_pending_) return false;
    completion_pending_ = false;
    return true;
}

// ---------------------------------------------------------------------------
// Project layout and command construction

ProjectLayout resolve_layout(const ProjectSettings& settings) {
    ProjectLayout layout;
    layout.root = std::filesystem::path(settings.project_dir.data());
    layout.cache = layout.root / "cache";
    layout.sparse_ply = layout.root / "sparse.ply";
    layout.sparse_poses = layout.root / "sparse_sfm_diagnostics.csv";
    layout.model_output = layout.root / "model.ply";
    layout.splat_ply = layout.root / "model_splat.ply";
    layout.mesh_ply = layout.root / "model_splat_mesh.ply";
    layout.align_log = layout.root / "editor_align.log";
    layout.train_log = layout.root / "editor_train.log";
    return layout;
}

std::string build_align_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout) {
    std::ostringstream command;
    // Deliberately no --capture-mode / --splat / --mesh: those switch the CLI
    // into a full rebuild. This invocation stops after the sparse export.
    command << quote(cli_path) << " --images "
            << quote(settings.images_dir.data()) << " --output "
            << quote(layout.sparse_ply) << " --mode "
            << sfm_mode_flag(settings.sfm_mode) << " --max-features "
            << settings.max_features;
    if (settings.reuse_cache)
        command << " --cache-dir " << quote(layout.cache);
    return command.str();
}

std::string build_train_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout, const PreviewHandles& preview) {
    std::ostringstream command;
    command << quote(cli_path) << " --images "
            << quote(settings.images_dir.data()) << " --output "
            << quote(layout.model_output) << " --mode "
            << sfm_mode_flag(settings.sfm_mode) << " --max-features "
            << settings.max_features;
    // The shared cache turns the SfM prefix into a checkpoint hit, so the
    // alignment the user just reviewed is reused verbatim.
    if (settings.reuse_cache)
        command << " --cache-dir " << quote(layout.cache);

    command << " --splat --capture-mode "
            << (settings.scene_mode ? "scene" : "object")
            << " --splat-strategy " << strategy_flag(settings.strategy)
            << " --splat-iterations " << settings.iterations
            << " --splat-preview-interval " << settings.preview_interval
            << " --splat-max-resolution " << settings.max_resolution
            << " --splat-progressive-resolution="
            << (settings.progressive_resolution ? "true" : "false")
            << " --splat-use-mask=" << (settings.use_mask ? "true" : "false")
            << " --splat-normal-field="
            << (settings.normal_field ? "true" : "false");

    // Mesh extraction is what turns on depth/normal and multi-view geometry
    // supervision inside the trainer, so both travel together.
    command << " --mesh=" << (settings.build_mesh ? "true" : "false");
    if (settings.build_mesh) {
        command << " --mesh-method " << mesh_method_flag(settings.mesh_method)
                << " --splat-depth-normal-weight "
                << settings.depth_normal_weight << " --splat-mv-geo-weight "
                << settings.multi_view_geo_weight << " --splat-mv-ncc-weight "
                << settings.multi_view_ncc_weight
                << " --splat-geometry-from-iter "
                << settings.geometry_from_iter;
    }

    if (preview.memory && preview.semaphore) {
        command << " --splat-preview-vk-memory-handle " << preview.memory
                << " --splat-preview-vk-semaphore-handle " << preview.semaphore
                << " --splat-preview-vk-allocation-size "
                << preview.allocation_size << " --splat-preview-vk-width "
                << preview.width << " --splat-preview-vk-height "
                << preview.height << " --splat-preview-vk-device-luid "
                << preview.device_luid
                << " --splat-preview-vk-device-node-mask "
                << preview.device_node_mask;
    }
    return command.str();
}

std::string format_duration(const double seconds) {
    if (!(seconds >= 0.0)) return "--";
    const auto total = static_cast<long long>(seconds + 0.5);
    if (total < 60) return std::to_string(total) + "s";
    const long long minutes = total / 60;
    if (minutes < 60)
        return std::to_string(minutes) + "m " + std::to_string(total % 60) + "s";
    return std::to_string(minutes / 60) + "h " + std::to_string(minutes % 60) +
           "m";
}

std::string format_count(const std::uint64_t value) {
    if (value < 1'000) return std::to_string(value);
    std::array<char, 32> buffer{};
    if (value < 1'000'000) {
        std::snprintf(
            buffer.data(), buffer.size(), "%.1fk",
            static_cast<double>(value) / 1'000.0);
    } else {
        std::snprintf(
            buffer.data(), buffer.size(), "%.2fM",
            static_cast<double>(value) / 1'000'000.0);
    }
    return buffer.data();
}

}  // namespace editor
