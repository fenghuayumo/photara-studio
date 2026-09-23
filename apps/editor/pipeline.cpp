#include "pipeline.hpp"

#include "i18n.hpp"
#include "io/video_frames.hpp"
#include "sam/model_cache.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#if defined(_WIN32)
#include <tlhelp32.h>
#else
#include <unistd.h>
#endif

namespace editor {
namespace {

// Session folder name: one per editor process, so a second instance (and a
// crashed one whose pid was reused later) can never share handshake files.
std::string session_directory_name() {
#if defined(_WIN32)
    return "session-" + std::to_string(GetCurrentProcessId());
#else
    return "session-" + std::to_string(static_cast<long>(::getpid()));
#endif
}

// Project setting -> CLI flag. The stored value mirrors photara::CameraModel:
// 0 pinhole, 1 OpenCV fisheye, 2 automatic, 3 equirectangular panorama.
const char* camera_model_flag(const int model) {
    switch (model) {
        case 0: return "pinhole";
        case 1: return "opencv_fisheye";
        case 3: return "equirectangular";
        default: return "auto";
    }
}

// Maps a ProgressReporter label onto a stage and a slice of the overall bar.
// Longer, more specific prefixes must precede their shorter relatives.
struct Band {
    std::string_view prefix;
    Stage stage;
    float begin;
    float end;
};

constexpr Band k_align_bands[] = {
    {"extract video frames", Stage::preparing, 0.00F, 0.06F},
    {"select sharp frames", Stage::preparing, 0.06F, 0.08F},
    {"generate sam masks", Stage::masking, 0.08F, 0.12F},
    {"extract features", Stage::features, 0.12F, 0.38F},
    {"prepare image pairs", Stage::matching, 0.38F, 0.40F},
    {"fast match image pairs", Stage::matching, 0.40F, 0.54F},
    {"lightglue match pairs", Stage::matching, 0.40F, 0.58F},
    {"rescue difficult pairs", Stage::matching, 0.54F, 0.58F},
    {"match image pairs", Stage::matching, 0.40F, 0.58F},
    {"verify pair geometry", Stage::matching, 0.58F, 0.72F},
    {"build tracks", Stage::tracks, 0.72F, 0.78F},
    {"sfm.retrieve_image_pairs", Stage::matching, 0.38F, 0.40F},
    {"register images", Stage::mapping, 0.78F, 0.96F},
};

constexpr Band k_dense_bands[] = {
    {"mvs.load_images", Stage::dense, 0.05F, 0.18F},
    {"mvs.estimate_depth", Stage::dense, 0.18F, 0.62F},
    {"mvs.geometric_consistency", Stage::dense, 0.62F, 0.78F},
    {"mvs.filter_depth", Stage::dense, 0.78F, 0.86F},
    {"mvs.fuse", Stage::dense, 0.86F, 0.94F},
    {"mvs.mesh", Stage::meshing, 0.94F, 0.99F},
};

// On the training run SfM is skipped when a working alignment exists.
constexpr Band k_texture_bands[] = {
    {"texture.load_views", Stage::texturing, 0.02F, 0.18F},
    {"texture.delight", Stage::texturing, 0.18F, 0.32F},
    {"texture.uv_unwrap", Stage::texturing, 0.32F, 0.50F},
    {"texture.project", Stage::texturing, 0.50F, 0.78F},
    {"texture.optimize", Stage::texturing, 0.78F, 0.98F},
    {"texture.", Stage::texturing, 0.02F, 1.00F},
};

constexpr Band k_train_bands[] = {
    {"extract video frames", Stage::preparing, 0.00F, 0.02F},
    {"select sharp frames", Stage::preparing, 0.00F, 0.02F},
    {"generate sam masks", Stage::masking, 0.02F, 0.04F},
    {"extract features", Stage::features, 0.04F, 0.06F},
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
constexpr float k_meshing_band_begin = 0.88F;

std::string runtime_cache_key(const std::filesystem::path& project_file) {
    std::error_code error;
    std::filesystem::path identity =
        std::filesystem::weakly_canonical(project_file, error);
    if (error) {
        error.clear();
        identity = std::filesystem::absolute(project_file, error);
    }
    if (error) identity = project_file;

    std::u8string bytes = identity.lexically_normal().generic_u8string();
#if defined(_WIN32)
    // Windows paths are case-insensitive. Keep differently-cased spellings of
    // the same project in the same runtime cache namespace.
    for (char8_t& value : bytes) {
        if (value >= u8'A' && value <= u8'Z')
            value = static_cast<char8_t>(value - u8'A' + u8'a');
    }
#endif
    std::uint64_t hash = 14695981039346656037ULL;
    for (const char8_t value : bytes) {
        hash ^= static_cast<std::uint8_t>(value);
        hash *= 1099511628211ULL;
    }

    std::ostringstream key;
    key << project_file.stem().string() << '-' << std::hex << hash;
    return key.str();
}

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
    return '"' + path_to_utf8(path) + '"';
}

std::string quote(const char* utf8_path) {
    return std::string("\"") + (utf8_path == nullptr ? "" : utf8_path) + '"';
}

std::string quote_text(const char* text) {
    std::string out = "\"";
    if (text != nullptr) {
        for (const char value : std::string_view(text)) {
            if (value == '\\' || value == '"') out.push_back('\\');
            out.push_back(value);
        }
    }
    out.push_back('"');
    return out;
}

void append_sam_flags(
    std::ostringstream& command, const ProjectSettings& settings,
    const bool refresh) {
    const bool want_masks = settings.sam_masks || settings.use_mask;
    if (!want_masks) return;
    const auto images = reconstruction_images_dir(settings);
    if (!images.empty())
        command << " --masks " << quote(images.parent_path() / "masks");
    if (!settings.sam_masks) return;
    std::filesystem::path model =
        path_from_utf8_field(settings.sam_model.data());
    if (model.empty()) model = photara::sam::locate_model();
    if (!model.empty()) command << " --sam-model " << quote(model);
    command << " --sam-text " << quote_text(settings.sam_text.data());
    if (settings.sam_negative_text[0] != '\0')
        command << " --sam-neg-text "
                << quote_text(settings.sam_negative_text.data());
    command << " --sam-keep-prompted="
            << (settings.sam_keep_prompted ? "true" : "false")
            << " --sam-video=" << (settings.sam_video ? "true" : "false")
            << " --sam-max-size " << std::max(0, settings.sam_max_size)
            << " --sam-refresh=" << (refresh ? "true" : "false");
}

void append_gui_flags(
    std::ostringstream& command, const ProjectLayout& layout,
    const std::filesystem::path& splat_override = {}) {
    command << " --gui";
    if (!layout.working_sfm.empty())
        command << " --working-sfm " << quote(layout.working_sfm);
    const auto& splat =
        splat_override.empty() ? layout.working_splat : splat_override;
    if (!splat.empty())
        command << " --working-splat " << quote(splat);
    if (!layout.working_mesh.empty())
        command << " --working-mesh " << quote(layout.working_mesh);
    if (!layout.working_dense.empty())
        command << " --working-dense " << quote(layout.working_dense);
    if (!layout.working_texture.empty())
        command << " --working-texture " << quote(layout.working_texture);
    // The alignment preview and live frame are handshake files, not products:
    // point them at the session folder so a Train that has to run SfM does not
    // leave them next to the working copies either.
    if (!layout.align_live.empty())
        command << " --align-live " << quote(layout.align_live);
    if (!layout.align_preview.empty())
        command << " --align-preview " << quote(layout.align_preview);
}

void append_video_extract_flags(
    std::ostringstream& command, const ProjectSettings& settings) {
    if (!is_video_source(settings)) return;
    command << " --video-fps " << settings.video_fps
            << " --video-sharp-window " << settings.video_sharp_window
            << " --video-max-frames " << settings.video_max_frames
            << " --video-quality " << settings.video_quality
            << " --video-scale " << settings.video_scale
            << " --video-rotate " << settings.video_rotate
            << " --video-frames-dir "
            << quote(reconstruction_images_dir(settings));
}

std::string lower_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](const unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    return extension;
}

const char* strategy_flag(const int index) {
    switch (index) {
        case 0: return "adc_igs";
        case 1: return "adc_plus";
        case 2: return "dense_adaptive";
        default:
            throw std::invalid_argument(
                "Editor splat strategy index is invalid");
    }
}

#if defined(_WIN32)
using NtProcessOp = LONG(NTAPI*)(HANDLE);

NtProcessOp load_nt_process_op(const char* name) {
    const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (!ntdll) return nullptr;
    return reinterpret_cast<NtProcessOp>(GetProcAddress(ntdll, name));
}

bool set_threads_suspended(const DWORD pid, const bool suspend) {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return false;
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    bool any = false;
    if (Thread32First(snapshot, &entry)) {
        do {
            if (entry.th32OwnerProcessID != pid) continue;
            HANDLE thread =
                OpenThread(THREAD_SUSPEND_RESUME, FALSE, entry.th32ThreadID);
            if (!thread) continue;
            const DWORD result = suspend ? SuspendThread(thread)
                                         : ResumeThread(thread);
            if (result != static_cast<DWORD>(-1)) any = true;
            CloseHandle(thread);
        } while (Thread32Next(snapshot, &entry));
    }
    CloseHandle(snapshot);
    return any;
}

bool set_handle_suspended(HANDLE process, const bool suspend) {
    static const NtProcessOp nt_suspend = load_nt_process_op("NtSuspendProcess");
    static const NtProcessOp nt_resume = load_nt_process_op("NtResumeProcess");
    const NtProcessOp op = suspend ? nt_suspend : nt_resume;
    if (op != nullptr) return op(process) >= 0;
    const DWORD pid = GetProcessId(process);
    if (pid == 0) return false;
    return set_threads_suspended(pid, suspend);
}

void add_unique_pid(std::vector<DWORD>& pids, const DWORD pid) {
    if (pid == 0) return;
    for (const DWORD existing : pids)
        if (existing == pid) return;
    pids.push_back(pid);
}

std::vector<DWORD> job_process_ids(HANDLE job, HANDLE process) {
    std::vector<DWORD> pids;
    if (process) add_unique_pid(pids, GetProcessId(process));
    if (!job) return pids;

    std::size_t cap = 16;
    for (int attempt = 0; attempt < 5; ++attempt) {
        const std::size_t bytes =
            offsetof(JOBOBJECT_BASIC_PROCESS_ID_LIST, ProcessIdList) +
            sizeof(ULONG_PTR) * cap;
        std::vector<std::uint8_t> buffer(bytes);
        auto* list =
            reinterpret_cast<JOBOBJECT_BASIC_PROCESS_ID_LIST*>(buffer.data());
        if (QueryInformationJobObject(
                job, JobObjectBasicProcessIdList, list,
                static_cast<DWORD>(bytes), nullptr)) {
            for (DWORD i = 0; i < list->NumberOfProcessIdsInList; ++i)
                add_unique_pid(
                    pids, static_cast<DWORD>(list->ProcessIdList[i]));
            break;
        }
        if (GetLastError() != ERROR_MORE_DATA) break;
        if (list->NumberOfAssignedProcesses > cap)
            cap = list->NumberOfAssignedProcesses;
        else
            cap *= 2;
    }
    return pids;
}

HANDLE open_suspend_handle(HANDLE root_process, const DWORD pid) {
    if (root_process && GetProcessId(root_process) == pid) return root_process;
    HANDLE handle = OpenProcess(
        PROCESS_SUSPEND_RESUME | PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
        pid);
    if (!handle) handle = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
    return handle;
}

bool apply_suspend_to_pid(
    HANDLE root_process, const DWORD pid, const bool suspend) {
    HANDLE handle = open_suspend_handle(root_process, pid);
    if (!handle) return false;
    const bool ok = set_handle_suspended(handle, suspend);
    if (handle != root_process) CloseHandle(handle);
    return ok;
}

// Freeze or thaw the CLI and every process in its job object (ffmpeg).
// Suspend the root first so it cannot spawn new children mid-pause.
bool set_job_tree_suspended(HANDLE job, HANDLE process, const bool suspend) {
    if (!process) return false;
    const DWORD root = GetProcessId(process);
    if (root == 0) return false;
    if (suspend) {
        if (!apply_suspend_to_pid(process, root, true)) return false;
        for (const DWORD pid : job_process_ids(job, process)) {
            if (pid != root) apply_suspend_to_pid(process, pid, true);
        }
        return true;
    }
    for (const DWORD pid : job_process_ids(job, process)) {
        if (pid != root) apply_suspend_to_pid(process, pid, false);
    }
    return apply_suspend_to_pid(process, root, false);
}
#endif

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

const char* job_name(const JobKind kind) {
    switch (kind) {
        case JobKind::align: return i18n::tr("Alignment");
        case JobKind::train: return i18n::tr("Training");
        case JobKind::dense: return i18n::tr("Extract Mesh");
        case JobKind::texture: return i18n::tr("Bake Texture");
        case JobKind::export_sfm: return i18n::tr("SfM export");
        case JobKind::none: return i18n::tr("Job");
    }
    return i18n::tr("Job");
}

const char* stage_name(const Stage stage) {
    switch (stage) {
        case Stage::idle: return i18n::tr("Idle");
        case Stage::features: return i18n::tr("Extracting features");
        case Stage::matching: return i18n::tr("Matching views");
        case Stage::tracks: return i18n::tr("Building tracks");
        case Stage::mapping: return i18n::tr("Solving camera poses");
        case Stage::exporting: return i18n::tr("Writing sparse scene");
        case Stage::preparing: return i18n::tr("Extracting frames");
        case Stage::masking: return i18n::tr("Generating masks");
        case Stage::dense: return i18n::tr("MVS stereo");
        case Stage::training: return i18n::tr("Training Gaussians");
        case Stage::meshing: return i18n::tr("Extracting mesh");
        case Stage::texturing: return i18n::tr("Baking texture");
        case Stage::complete: return i18n::tr("Complete");
        case Stage::failed: return i18n::tr("Failed");
    }
    return i18n::tr("Idle");
}

// ---------------------------------------------------------------------------
// LogStream

namespace {

bool is_digit(const char value) {
    return value >= '0' && value <= '9';
}

ConsoleSeverity severity_from_tag(const std::string_view tag) {
    if (tag == "error") return ConsoleSeverity::error;
    if (tag == "warning" || tag == "warn") return ConsoleSeverity::warning;
    if (tag == "debug" || tag == "trace") return ConsoleSeverity::debug;
    if (tag == "info") return ConsoleSeverity::info;
    return ConsoleSeverity::other;
}

void count_severity(ConsoleCounts& counts, const ConsoleSeverity severity, const int delta) {
    counts.total += delta;
    switch (severity) {
        case ConsoleSeverity::warning: counts.warning += delta; break;
        case ConsoleSeverity::error: counts.error += delta; break;
        default: counts.info += delta; break;
    }
}

ConsoleLine parse_console_line(std::string raw) {
    ConsoleLine line;
    // Logger prefix: "YYYY-MM-DD HH:MM:SS.mmm [level] message"
    constexpr std::size_t k_timestamp = 23;
    if (raw.size() > k_timestamp + 3 && is_digit(raw[0]) && raw[4] == '-' &&
        raw[10] == ' ' && raw[13] == ':' && raw[16] == ':' && raw[19] == '.') {
        line.time.assign(raw, 11, 12);
        const std::size_t open = raw.find('[', k_timestamp);
        const std::size_t close =
            open == std::string::npos ? std::string::npos : raw.find(']', open);
        if (open != std::string::npos && close != std::string::npos) {
            line.severity = severity_from_tag(std::string_view(
                raw.data() + open + 1, close - open - 1));
            std::size_t message = close + 1;
            while (message < raw.size() && raw[message] == ' ') ++message;
            line.message = raw.substr(message);
            return line;
        }
    }
    line.severity = ConsoleSeverity::other;
    line.message = std::move(raw);
    return line;
}

}  // namespace

void LogStream::open(const std::filesystem::path& path) {
    path_ = path;
    offset_ = 0;
    partial_.clear();
    lines_.clear();
    counts_ = {};
    ++generation_;
}

void LogStream::close() {
    path_.clear();
    offset_ = 0;
    partial_.clear();
}

void LogStream::clear() {
    close();
    clear_display();
}

void LogStream::clear_display() {
    lines_.clear();
    counts_ = {};
    ++generation_;
}

void LogStream::push_line(std::string line) {
    ConsoleLine parsed = parse_console_line(std::move(line));
    count_severity(counts_, parsed.severity, 1);
    lines_.push_back(std::move(parsed));
    ++generation_;
}

void LogStream::recount() {
    counts_ = {};
    for (const ConsoleLine& line : lines_)
        count_severity(counts_, line.severity, 1);
}

void LogStream::trim_if_needed() {
    if (lines_.size() <= k_max_lines) return;
    const std::size_t drop = lines_.size() - (k_max_lines * 3) / 4;
    lines_.erase(lines_.begin(), lines_.begin() + static_cast<std::ptrdiff_t>(drop));
    recount();
    ++generation_;
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
        lines_.clear();
        counts_ = {};
        ++generation_;
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

    partial_ += chunk;
    std::size_t begin = 0;
    while (true) {
        const std::size_t newline = partial_.find('\n', begin);
        if (newline == std::string::npos) break;
        std::string line = partial_.substr(begin, newline - begin);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) {
            fresh_lines.push_back(line);
            push_line(std::move(line));
        }
        begin = newline + 1;
    }
    partial_.erase(0, begin);
    trim_if_needed();
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
    has_clock_ = false;
    clock_running_ = false;
    started_ = {};
    stopped_ = {};
}

void RunMonitor::begin(const JobKind kind, const Stage opening) {
    reset();
    kind_ = kind;
    if (opening != Stage::idle) stage_ = opening;
    else if (kind == JobKind::train) stage_ = Stage::preparing;
    else if (kind == JobKind::dense) stage_ = Stage::dense;
    else if (kind == JobKind::texture) stage_ = Stage::texturing;
    else stage_ = Stage::features;
    band_begin_ = 0.F;
    band_end_ = 0.F;
    started_ = std::chrono::steady_clock::now();
    stopped_ = started_;
    has_clock_ = true;
    clock_running_ = true;
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

    if (line.find("pipeline_handoff=") != std::string::npos) {
        // Gaussian mesh extraction opens already in meshing. Later handoff
        // lines must not relabel that job as MVS stereo.
        if (stage_ == Stage::meshing) {
            task_ = {};
            return;
        }
        if (line.find("pipeline_handoff=reconstruct") != std::string::npos) {
            if (kind_ == JobKind::train || kind_ == JobKind::dense)
                enter_stage(Stage::features, 0.02F);
            else if (kind_ == JobKind::texture)
                enter_stage(Stage::texturing, 0.02F);
        } else if (kind_ == JobKind::train) {
            enter_stage(Stage::preparing, 0.04F);
        } else if (kind_ == JobKind::dense) {
            enter_stage(Stage::dense, 0.04F);
        } else if (kind_ == JobKind::texture) {
            enter_stage(Stage::texturing, 0.04F);
        }
        task_ = {};
        return;
    }
    if (stage_ != Stage::meshing &&
        (line.find("working_sfm_loaded=") != std::string::npos ||
         line.find("ascan_sfm_loaded") != std::string::npos ||
         line.find("splat_dataset=") != std::string::npos)) {
        if (kind_ == JobKind::train) enter_stage(Stage::preparing, 0.05F);
        else if (kind_ == JobKind::dense) enter_stage(Stage::dense, 0.05F);
        else if (kind_ == JobKind::texture) enter_stage(Stage::texturing, 0.06F);
    }
    if (line.find("splat_input=") != std::string::npos ||
        line.find("mvs sparse colors:") != std::string::npos ||
        line.find("mvs scene:") != std::string::npos) {
        if (kind_ == JobKind::train) enter_stage(Stage::preparing, 0.09F);
    }

    // Artifact keys. Only presence matters; paths come from the layout.
    if (line.find("sfm_diagnostics=") != std::string::npos)
        artifacts_.sparse = true;
    if (line.find("OpenMVS export:") != std::string::npos ||
        line.find(" mvs=") != std::string::npos)
        artifacts_.mvs = true;
    if (line.find("dense_ply=") != std::string::npos) artifacts_.dense = true;
    if (line.find("splat_mesh_only=") != std::string::npos ||
        line.find("splat_mesh_model=") != std::string::npos)
        enter_stage(Stage::meshing, k_meshing_band_begin);
    if (line.find("splat_model=") != std::string::npos ||
        line.find("splat_ply=") != std::string::npos) {
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
            const float iteration_ratio = std::clamp(
                static_cast<float>(training_.iteration) /
                    static_cast<float>(
                        std::max(training_.total_iterations, 1U)),
                0.F, 1.F);
            // Training progress is presented as iteration / total_iterations.
            // Keep the same value as the monotonic floor so saving the model
            // after the final iteration cannot make the status bar regress.
            floor_ = std::max(floor_, iteration_ratio);
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
                    kind_ == JobKind::train ? 0.09F : 0.77F);
            else if (
                name == "splat.mesh_render_geometry" || name == "splat.pam" ||
                name.rfind("mvs.mesh", 0) == 0 || name == "mvs.fuse")
                enter_stage(Stage::meshing, k_meshing_band_begin);
            else if (name.rfind("texture.", 0) == 0)
                enter_stage(
                    Stage::texturing,
                    kind_ == JobKind::texture ? 0.08F : 0.98F);
        }
        return;
    }

    if (line.find("frontend: images=") != std::string::npos ||
        line.find("frontend diagnostics:") != std::string::npos) {
        enter_stage(
            Stage::mapping, kind_ == JobKind::train ? 0.09F : 0.77F);
        task_ = {};
        return;
    }

    if (line.find("sfm_diagnostics=") != std::string::npos &&
        kind_ != JobKind::train) {
        enter_stage(Stage::exporting, 0.97F);
        task_ = {};
        return;
    }

    // ProgressReporter lines. "progress started:" only announces the total.
    const bool finished = line.find("progress finished: ") != std::string::npos;
    const bool started = line.find("progress started: ") != std::string::npos;
    const std::size_t running = line.find("progress: ");
    if (!finished && !started && running == std::string::npos) return;

    const std::string_view marker = finished
        ? "progress finished: "
        : (started ? "progress started: " : "progress: ");
    const std::size_t marker_at = line.find(marker);
    const std::size_t label_begin = marker_at + marker.size();
    std::size_t metrics_at = line.find(" completed=", label_begin);
    if (metrics_at == std::string::npos && started)
        metrics_at = line.find(" total=", label_begin);
    if (metrics_at == std::string::npos) return;
    const std::string label = line.substr(label_begin, metrics_at - label_begin);

    const Band* bands = k_align_bands;
    std::size_t band_count = std::size(k_align_bands);
    if (kind_ == JobKind::train) {
        bands = k_train_bands;
        band_count = std::size(k_train_bands);
    } else if (kind_ == JobKind::dense) {
        bands = k_dense_bands;
        band_count = std::size(k_dense_bands);
    } else if (kind_ == JobKind::texture) {
        bands = k_texture_bands;
        band_count = std::size(k_texture_bands);
    }
    const Band* match = nullptr;
    for (std::size_t i = 0; i < band_count; ++i)
        if (starts_with(label, bands[i].prefix)) {
            match = &bands[i];
            break;
        }

    task_.label = label;
    task_.active = !finished;
    if (started) {
        task_.completed = 0;
        number_after(line, " total=", task_.total);
    } else {
        ratio_after(line, " completed=", task_.completed, task_.total);
    }
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
    if (exit_code == 0) {
        stage_ = Stage::complete;
        floor_ = 1.F;
    } else if (exit_code == 2) {
        // User abort, not a reconstruction failure.
        stage_ = Stage::idle;
    } else {
        stage_ = Stage::failed;
    }
    if (clock_running_) {
        stopped_ = std::chrono::steady_clock::now();
        clock_running_ = false;
    }
}

void RunMonitor::pause_clock() {
    if (!has_clock_ || !clock_running_) return;
    stopped_ = std::chrono::steady_clock::now();
    clock_running_ = false;
}

void RunMonitor::resume_clock() {
    if (!has_clock_ || clock_running_) return;
    const auto now = std::chrono::steady_clock::now();
    started_ += now - stopped_;
    clock_running_ = true;
}

double RunMonitor::elapsed_seconds() const {
    if (!has_clock_) return -1.0;
    const auto end = clock_running_ ? std::chrono::steady_clock::now()
                                    : stopped_;
    return std::chrono::duration<double>(end - started_).count();
}

float RunMonitor::fraction() const {
    if (stage_ == Stage::complete) return 1.F;
    if (stage_ == Stage::training && training_.valid &&
        training_.total_iterations > 0) {
        return std::clamp(
            static_cast<float>(training_.iteration) /
                static_cast<float>(training_.total_iterations),
            0.F, 1.F);
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
        text += "  |  ";
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
    paused_ = false;
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
        CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &startup,
        &info);
    CloseHandle(log_handle);
    if (!created) {
        running_ = false;
        throw std::runtime_error(
            "CreateProcess failed: " + std::to_string(GetLastError()));
    }
    job_ = CreateJobObjectW(nullptr, nullptr);
    if (job_) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(
                job_, JobObjectExtendedLimitInformation, &limits,
                sizeof(limits)) ||
            !AssignProcessToJobObject(job_, info.hProcess)) {
            CloseHandle(job_);
            job_ = nullptr;
        }
    }
    ResumeThread(info.hThread);
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
        paused_ = false;
        completion_pending_ = true;
        CloseHandle(std::exchange(process_, nullptr));
        if (job_) CloseHandle(std::exchange(job_, nullptr));
    }
#endif
}

void ProcessJob::pause() {
#if defined(_WIN32)
    if (!running_ || paused_ || !process_) return;
    if (!set_job_tree_suspended(job_, process_, true)) return;
    paused_ = true;
#endif
}

void ProcessJob::resume() {
#if defined(_WIN32)
    if (!running_ || !paused_ || !process_) return;
    if (!set_job_tree_suspended(job_, process_, false)) return;
    paused_ = false;
#endif
}

void ProcessJob::stop() {
#if defined(_WIN32)
    if (job_) {
        TerminateJobObject(job_, 2);
        CloseHandle(std::exchange(job_, nullptr));
    }
    if (process_) {
        TerminateProcess(process_, 2);
        WaitForSingleObject(process_, INFINITE);
        CloseHandle(std::exchange(process_, nullptr));
        exit_code_ = 2;
        completion_pending_ = true;
    }
#endif
    paused_ = false;
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
    const std::filesystem::path stored =
        path_from_utf8_field(settings.project_dir.data());
    if (lower_extension(stored) == ".ascan") {
        layout.project_file = stored;
        layout.root = stored.parent_path();
    } else {
        layout.root = stored;
        layout.project_file = stored.empty()
            ? std::filesystem::path{}
            : stored / "project.ascan";
    }
    if (layout.root.empty() && !layout.project_file.empty())
        layout.root = std::filesystem::current_path();
    const std::filesystem::path stem = layout.project_file.empty()
        ? std::filesystem::path("project")
        : layout.project_file.stem();
    const auto with_suffix = [&](const char* suffix) {
        std::filesystem::path named = layout.root / stem;
        named += suffix;
        return named;
    };
    layout.cache = with_suffix(".cache");
    layout.sparse_ply = with_suffix("_sparse.ply");
    layout.sparse_asfm = with_suffix(".asfm");
    layout.sparse_mvs = with_suffix(".mvs");
    layout.sparse_poses = with_suffix("_sfm_diagnostics.csv");
    layout.model_output = layout.project_file.empty()
        ? with_suffix(".ply")
        : layout.project_file;
    layout.splat_ply = with_suffix("_splat.ply");
    layout.splat_sog = with_suffix("_splat.sog");
    layout.splat_spz = with_suffix("_splat.spz");
    layout.splat_glb = with_suffix("_splat.glb");
    layout.splat_model = settings.splat_format == 2
        ? layout.splat_sog
        : settings.splat_format == 3 ? layout.splat_spz
        : settings.splat_format == 4 ? layout.splat_glb : layout.splat_ply;
    layout.mesh_ply = with_suffix("_splat_mesh.ply");
    layout.mvs_mesh_ply = with_suffix("_mesh.ply");
    layout.mvs_raw_mesh_ply = with_suffix("_mvs_mesh.ply");
    layout.dense_ply = with_suffix("_dense.ply");
    layout.align_log = with_suffix("_align.log");
    layout.train_log = with_suffix("_train.log");
    layout.dense_log = with_suffix("_dense.log");
    layout.texture_log = with_suffix("_texture.log");
    layout.export_log = with_suffix("_export.log");
    layout.view_log = with_suffix("_view.log");
    // Working copies (sfm.bin, splat.ply, mesh/dense, preview sidecars) sit
    // next to the project by default so a dataset on a data drive cannot fill
    // the system drive with multi-GB caches. A configured cache folder
    // overrides that, and a session without any project path (no usable
    // <root>) falls back to the per-project temp namespace.
    const std::filesystem::path cache_root =
        path_from_utf8_field(settings.cache_dir.data());
    std::filesystem::path runtime_dir = layout.cache;
    if (!cache_root.empty()) {
        runtime_dir = cache_root / runtime_cache_key(layout.project_file);
    } else if (layout.root.empty()) {
        std::error_code temp_error;
        const auto temp = std::filesystem::temp_directory_path(temp_error);
        if (temp_error)
            runtime_dir = std::filesystem::path("photara_cache") /
                          runtime_cache_key(layout.project_file);
        else
            runtime_dir =
                temp / "Photara" / runtime_cache_key(layout.project_file);
    }
    layout.working_sfm = runtime_dir / "sfm.bin";
    layout.working_splat = runtime_dir / "splat.ply";
    layout.working_mesh = runtime_dir / "mesh.ply";
    layout.working_dense = runtime_dir / "dense.ply";
    layout.working_texture = runtime_dir / "textured";
    // Handshake files are per process, not per project: the editor, the CLI
    // child, the trainer and the splat viewer exchange them for one session, and
    // nothing may read them after that. Keeping them in a session folder lets the
    // cache folder hold products only, and stops their per-frame writes (and
    // always-fresh mtimes) from landing on a slow cache drive or from hiding an
    // otherwise unused cache from maintenance.
    std::error_code session_error;
    const auto session_temp = std::filesystem::temp_directory_path(session_error);
    layout.session_dir = session_error
        ? std::filesystem::path("photara_session")
        : session_temp / "Photara" / session_directory_name();
    layout.preview_view_file = layout.session_dir / "preview_view";
    layout.preview_camera_file = layout.session_dir / "preview_camera";
    layout.preview_vis_file = layout.session_dir / "preview_vis";
    layout.preview_ack_file = layout.session_dir / "preview_ack";
    layout.align_live = layout.session_dir / "align.live";
    layout.align_preview = layout.session_dir / "align.preview.asfm";
    layout.working_subject_bounds = layout.session_dir / "subject_bounds.txt";
    return layout;
}

std::filesystem::path path_from_utf8_field(const char* text) {
    if (text == nullptr || text[0] == '\0') return {};
#if defined(_WIN32)
    const auto* begin = reinterpret_cast<const char8_t*>(text);
    return std::filesystem::path(
        std::u8string(begin, begin + std::strlen(text)));
#else
    return std::filesystem::path(text);
#endif
}

std::string path_to_utf8(const std::filesystem::path& path) {
#if defined(_WIN32)
    const std::u8string text = path.u8string();
    return {
        reinterpret_cast<const char*>(text.data()), text.size()};
#else
    return path.string();
#endif
}

bool is_video_source(const ProjectSettings& settings) {
    return photara::io::is_video_path(
        path_from_utf8_field(settings.images_dir.data()));
}

std::filesystem::path reconstruction_images_dir(const ProjectSettings& settings) {
    const std::filesystem::path source =
        path_from_utf8_field(settings.images_dir.data());
    if (!photara::io::is_video_path(source)) return source;
    if (settings.video_frames_dir[0] != '\0')
        return path_from_utf8_field(settings.video_frames_dir.data());
    return photara::io::default_video_frames_dir(source);
}

std::string build_align_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout) {
    std::ostringstream command;
    // Align keeps SfM in a working copy (project .cache by default, or the
    // configured cache folder). .ascan is only written on Save Project.
    command << quote(cli_path) << " --images "
            << quote(settings.images_dir.data()) << " --output "
            << quote(layout.project_file.empty() ? layout.sparse_ply
                                                 : layout.project_file) << " --mode "
            << sfm_mode_flag(settings.sfm_mode)
            << " --camera-model " << camera_model_flag(settings.camera_model) << " --max-features "
            << settings.max_features;
    if (settings.reuse_cache)
        command << " --cache-dir " << quote(layout.cache);
    append_video_extract_flags(command, settings);
    append_sam_flags(command, settings, true);
    append_gui_flags(command, layout);
    return command.str();
}

std::string build_train_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout, const PreviewHandles& preview) {
    std::ostringstream command;
    command << quote(cli_path) << " --images "
            << quote(settings.images_dir.data()) << " --output "
            << quote(layout.model_output);

    // Training reloads SfM from the working copy (or a saved .ascan),
    // unless an external dataset is selected.
    command << " --mode " << sfm_mode_flag(settings.sfm_mode)
            << " --camera-model " << camera_model_flag(settings.camera_model)
            << " --max-features " << settings.max_features;
    if (settings.reuse_cache)
        command << " --cache-dir " << quote(layout.cache);

    if (settings.dataset_source[0] != '\0') {
        command << " --splat-dataset "
                << quote(settings.dataset_source.data());
        if (settings.dataset_initial_cloud[0] != '\0')
            command << " --dense-ply "
                    << quote(settings.dataset_initial_cloud.data());
    }

    command << " --splat --capture-mode "
            << (settings.scene_mode ? "scene" : "object");
    if (!settings.scene_mode) {
        std::error_code bounds_error;
        if (!layout.working_subject_bounds.empty() &&
            std::filesystem::exists(
                layout.working_subject_bounds, bounds_error))
            command << " --subject-bounds "
                    << quote(layout.working_subject_bounds);
    }
    command << " --splat-strategy " << strategy_flag(settings.strategy)
            << " --splat-iterations " << settings.iterations
            << " --splat-densification-cap "
            << std::max(1, settings.densification_cap)
            << " --splat-sh-degree "
            << std::clamp(settings.sh_degree, 0, 3)
            << " --splat-preview-interval " << settings.preview_interval
            << " --splat-preview-view 0"
            << " --splat-preview-view-file "
            << quote(layout.preview_view_file)
            << " --splat-preview-camera-file "
            << quote(layout.preview_camera_file)
            << " --splat-preview-vis-file "
            << quote(layout.preview_vis_file)
            << " --splat-preview-ack-file "
            << quote(layout.preview_ack_file)
            << " --splat-max-resolution " << settings.max_resolution
            << " --splat-progressive-resolution="
            << (settings.progressive_resolution ? "true" : "false")
            << " --splat-use-mask="
            << ((settings.use_mask || settings.sam_masks) ? "true" : "false")
            << " --splat-alpha-mode "
            << (settings.mask_mode == 1 ? "transparent" : "masked")
            << " --splat-normal-field="
            << (settings.normal_field ? "true" : "false");
    if (settings.ppisp_layout != 0)
        command << " --splat-ppisp=true";
    if (settings.bilateral_grid)
        command << " --splat-bilateral-grid=true";
    append_video_extract_flags(command, settings);
    append_sam_flags(command, settings, false);
    append_gui_flags(command, layout);

    // Geometry-supervised 3DGS mesh: depth/normal + multi-view losses, then
    // extract. Extract Mesh (MVS) is a separate photogrammetry job.
    const bool gaussian_mesh = mesh_from_gaussians(settings);
    command << " --mesh=" << (gaussian_mesh ? "true" : "false");
    if (gaussian_mesh) {
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

std::string build_splat_mesh_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout, const std::filesystem::path& splat_model) {
    std::ostringstream command;
    command << quote(cli_path) << " --images "
            << quote(settings.images_dir.data()) << " --output "
            << quote(layout.model_output);
    command << " --mode " << sfm_mode_flag(settings.sfm_mode)
            << " --camera-model " << camera_model_flag(settings.camera_model)
            << " --max-features " << settings.max_features;
    if (settings.reuse_cache)
        command << " --cache-dir " << quote(layout.cache);
    if (settings.dataset_source[0] != '\0') {
        command << " --splat-dataset "
                << quote(settings.dataset_source.data());
        if (settings.dataset_initial_cloud[0] != '\0')
            command << " --dense-ply "
                    << quote(settings.dataset_initial_cloud.data());
    }
    command << " --splat --mesh=true --splat-mesh-only"
            << " --capture-mode "
            << (settings.scene_mode ? "scene" : "object");
    if (!settings.scene_mode) {
        std::error_code bounds_error;
        if (!layout.working_subject_bounds.empty() &&
            std::filesystem::exists(
                layout.working_subject_bounds, bounds_error))
            command << " --subject-bounds "
                    << quote(layout.working_subject_bounds);
    }
    command << " --mesh-method " << mesh_method_flag(settings.mesh_method)
            << " --splat-preview-interval 0"
            << " --splat-max-resolution " << settings.max_resolution
            << " --splat-use-mask="
            << ((settings.use_mask || settings.sam_masks) ? "true" : "false");
    append_video_extract_flags(command, settings);
    append_sam_flags(command, settings, false);
    append_gui_flags(command, layout, splat_model);
    return command.str();
}

std::string build_view_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout, const PreviewHandles& preview) {
    std::ostringstream command;
    command << quote(cli_path) << " --images "
            << quote(settings.images_dir.data()) << " --output "
            << quote(layout.model_output) << " --splat-view"
            << " --splat-preview-camera-file "
            << quote(layout.preview_camera_file)
            << " --splat-preview-vis-file "
            << quote(layout.preview_vis_file);
    std::error_code exists_error;
    std::filesystem::path view_splat;
    const std::filesystem::path imported_model =
        path_from_utf8_field(settings.splat_model_source.data());
    if (!imported_model.empty() &&
        std::filesystem::exists(imported_model, exists_error)) {
        view_splat = imported_model;
    } else {
        const std::array<std::filesystem::path, 6> candidates = {
            layout.working_splat, layout.splat_model, layout.splat_ply,
            layout.splat_sog, layout.splat_spz, layout.splat_glb};
        for (const auto& candidate : candidates) {
            if (!std::filesystem::exists(candidate, exists_error)) continue;
            view_splat = candidate;
            break;
        }
    }
    append_gui_flags(command, layout, view_splat);
    append_video_extract_flags(command, settings);
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

std::string build_dense_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout) {
    std::ostringstream command;
    command << quote(cli_path) << " --images "
            << quote(settings.images_dir.data()) << " --output "
            << quote(layout.project_file.empty() ? layout.sparse_ply
                                                 : layout.project_file);
    if (settings.dataset_source[0] != '\0') {
        command << " --splat-dataset "
                << quote(settings.dataset_source.data());
        if (settings.dataset_initial_cloud[0] != '\0')
            command << " --dense-ply "
                    << quote(settings.dataset_initial_cloud.data());
    } else {
        command << " --mode " << sfm_mode_flag(settings.sfm_mode)
                << " --camera-model " << camera_model_flag(settings.camera_model)
                << " --max-features " << settings.max_features;
        if (settings.reuse_cache)
            command << " --cache-dir " << quote(layout.cache);
    }
    command << " --capture-mode "
            << (settings.scene_mode ? "scene" : "object");
    if (!settings.scene_mode) {
        std::error_code bounds_error;
        if (!layout.working_subject_bounds.empty() &&
            std::filesystem::exists(
                layout.working_subject_bounds, bounds_error))
            command << " --subject-bounds "
                    << quote(layout.working_subject_bounds);
    }
    const bool mvs_mesh = mesh_from_mvs(settings);
    // Capture mode only selects object bounds versus an unbounded scene.
    // --splat=false keeps this job on the MVS mesh path.
    command << " --dense --splat=false --mesh="
            << (mvs_mesh ? "true" : "false");
    if (mvs_mesh) {
        const int method =
            settings.mesh_method == 3 ? 0 : settings.mesh_method;
        command << " --mesh-method " << mesh_method_flag(method);
    }
    append_video_extract_flags(command, settings);
    append_sam_flags(command, settings, false);
    append_gui_flags(command, layout);
    return command.str();
}

std::string build_texture_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout) {
    std::ostringstream command;
    command << quote(cli_path) << " --images "
            << quote(settings.images_dir.data()) << " --output "
            << quote(layout.project_file.empty() ? layout.sparse_ply
                                                 : layout.project_file);
    if (settings.dataset_source[0] != '\0') {
        command << " --splat-dataset "
                << quote(settings.dataset_source.data());
        if (settings.dataset_initial_cloud[0] != '\0')
            command << " --dense-ply "
                    << quote(settings.dataset_initial_cloud.data());
    } else {
        command << " --mode " << sfm_mode_flag(settings.sfm_mode)
                << " --camera-model " << camera_model_flag(settings.camera_model)
                << " --max-features " << settings.max_features;
        if (settings.reuse_cache)
            command << " --cache-dir " << quote(layout.cache);
    }
    command << " --texture"
            << " --atlas-resolution "
            << std::max(64, settings.atlas_resolution)
            << " --texture-optimize="
            << (settings.texture_optimize ? "true" : "false")
            << " --texture-optimize-steps "
            << texture_optimize_steps(settings);
    if (settings.texture_delight) command << " --delight";
    append_video_extract_flags(command, settings);
    append_sam_flags(command, settings, false);
    append_gui_flags(command, layout);
    return command.str();
}

std::string build_export_sfm_command(
    const char* cli_path, const ProjectSettings& settings,
    const ProjectLayout& layout) {
    std::ostringstream command;
    command << quote(cli_path) << " --images "
            << quote(settings.images_dir.data()) << " --output "
            << quote(layout.sparse_asfm) << " --mode "
            << sfm_mode_flag(settings.sfm_mode)
            << " --camera-model " << camera_model_flag(settings.camera_model) << " --max-features "
            << settings.max_features;
    std::error_code exists_error;
    if (settings.reuse_cache &&
        std::filesystem::exists(layout.cache, exists_error))
        command << " --cache-dir " << quote(layout.cache);
    append_video_extract_flags(command, settings);
    append_sam_flags(command, settings, false);
    append_gui_flags(command, layout);
    return command.str();
}

void write_preview_view_index(const ProjectLayout& layout, const unsigned index) {
    if (layout.preview_view_file.empty()) return;
    std::error_code error;
    std::filesystem::create_directories(
        layout.preview_view_file.parent_path(), error);
    std::ofstream output(layout.preview_view_file, std::ios::trunc);
    if (!output) return;
    output << index << '\n';
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
