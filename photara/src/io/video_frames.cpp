#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "io/video_frames.hpp"

#include "core/logging.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace photara::io {

bool operator==(const VideoSegment& a, const VideoSegment& b) {
    return a.start_seconds == b.start_seconds && a.end_seconds == b.end_seconds;
}

bool operator<(const VideoSegment& a, const VideoSegment& b) {
    return a.start_seconds < b.start_seconds;
}

std::vector<VideoSegment> normalize_video_segments(
    const std::vector<VideoSegment>& segments, const double duration_seconds) {
    std::vector<VideoSegment> ranges;
    ranges.reserve(segments.size());
    for (const VideoSegment& segment : segments) {
        if (!std::isfinite(segment.start_seconds) ||
            !std::isfinite(segment.end_seconds))
            continue;
        VideoSegment range = segment;
        range.start_seconds = std::max(0.0, range.start_seconds);
        if (duration_seconds > 0.0) {
            if (range.start_seconds >= duration_seconds) continue;
            if (range.end_seconds > duration_seconds ||
                range.end_seconds <= range.start_seconds)
                range.end_seconds = duration_seconds;
        }
        // end <= start is the open-ended convention. Keep it until the sort
        // so an open range always ends up last after merging.
        if (range.end_seconds > range.start_seconds + 1.0e-6)
            range.end_seconds = std::min(
                range.end_seconds,
                duration_seconds > 0.0 ? duration_seconds
                                       : std::numeric_limits<double>::infinity());
        ranges.push_back(range);
    }
    std::stable_sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
        const double a_end = a.end_seconds > a.start_seconds
            ? a.end_seconds : std::numeric_limits<double>::infinity();
        const double b_end = b.end_seconds > b.start_seconds
            ? b.end_seconds : std::numeric_limits<double>::infinity();
        if (a.start_seconds != b.start_seconds)
            return a.start_seconds < b.start_seconds;
        return a_end < b_end;
    });
    std::vector<VideoSegment> merged;
    for (const VideoSegment& range : ranges) {
        if (merged.empty()) {
            merged.push_back(range);
            continue;
        }
        VideoSegment& previous = merged.back();
        const bool previous_open =
            previous.end_seconds <= previous.start_seconds;
        const bool next_open = range.end_seconds <= range.start_seconds;
        const double previous_end = previous_open
            ? std::numeric_limits<double>::infinity()
            : previous.end_seconds;
        if (range.start_seconds <= previous_end + 1.0e-6) {
            if (previous_open || next_open) {
                previous.end_seconds = 0.0;  // Open end.
            } else {
                previous.end_seconds = std::max(
                    previous.end_seconds, range.end_seconds);
            }
        } else {
            merged.push_back(range);
        }
    }
    return merged;
}

std::string serialize_video_segments(const std::vector<VideoSegment>& segments) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    bool first = true;
    for (const VideoSegment& segment : normalize_video_segments(segments)) {
        if (!first) out << ';';
        first = false;
        out << segment.start_seconds << ':';
        if (segment.end_seconds > segment.start_seconds)
            out << segment.end_seconds;
        else
            out << '-';
    }
    return out.str();
}

bool parse_video_segments(
    const std::string_view text, std::vector<VideoSegment>& segments,
    std::string* error) {
    segments.clear();
    auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    auto trim = [](std::string_view value) {
        while (!value.empty() &&
               std::isspace(static_cast<unsigned char>(value.front())))
            value.remove_prefix(1);
        while (!value.empty() &&
               std::isspace(static_cast<unsigned char>(value.back())))
            value.remove_suffix(1);
        return value;
    };
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find(';', start);
        const std::string_view item = trim(text.substr(
            start, end == std::string_view::npos ? std::string_view::npos
                                                 : end - start));
        if (!item.empty()) {
            const std::size_t colon = item.find(':');
            if (colon == std::string_view::npos)
                return fail("Video segment must be start:end: " +
                            std::string(item));
            const std::string start_text(trim(item.substr(0, colon)));
            const std::string end_text(trim(item.substr(colon + 1)));
            char* parse_end = nullptr;
            const double segment_start =
                std::strtod(start_text.c_str(), &parse_end);
            if (start_text.empty() || !parse_end ||
                *parse_end != '\0' || !std::isfinite(segment_start))
                return fail("Invalid segment start: " + start_text);
            double segment_end = 0.0;
            if (end_text != "-") {
                const double parsed = std::strtod(end_text.c_str(), &parse_end);
                if (end_text.empty() || !parse_end || *parse_end != '\0' ||
                    !std::isfinite(parsed))
                    return fail("Invalid segment end: " + end_text);
                segment_end = parsed;
            }
            if (segment_end > segment_start)
                segments.push_back({segment_start, segment_end});
            else
                segments.push_back({segment_start, 0.0});
        }
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    segments = normalize_video_segments(segments);
    return true;
}

namespace {

constexpr const char* k_video_extensions[] = {
    ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".insv", ".osv", ".avi",
    ".mts", ".m2ts", ".360", ".ts", ".wmv", ".mpeg", ".mpg", ".3gp"};

constexpr const char* k_image_extensions[] = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"};

constexpr const char* k_manifest_name = ".photara_extract";

std::string lower_ascii(std::string text) {
    std::transform(
        text.begin(), text.end(), text.begin(),
        [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

std::string extension_lower(const std::filesystem::path& path) {
    return lower_ascii(path.extension().string());
}

bool is_image_file(const std::filesystem::path& path) {
    const std::string ext = extension_lower(path);
    for (const char* candidate : k_image_extensions)
        if (ext == candidate) return true;
    return false;
}

bool is_extract_still_name(std::string name) {
    name = lower_ascii(std::move(name));
    if (name == k_manifest_name) return true;
    const auto dot = name.rfind('.');
    if (dot == std::string::npos || dot == 0) return false;
    const std::string ext = name.substr(dot);
    if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") return false;
    const std::string stem = name.substr(0, dot);
    auto digits = [](const std::string& text, const std::size_t from) {
        if (from >= text.size()) return false;
        for (std::size_t i = from; i < text.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(text[i]))) return false;
        }
        return true;
    };
    if (stem.size() >= 2 && stem[0] == 'c' && stem[1] == '_')
        return digits(stem, 2);
    return stem.size() == 5 && digits(stem, 0);
}

std::string path_utf8(const std::filesystem::path& path) {
    const std::u8string text = path.generic_u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}

void emit(const VideoExtractOptions& options, const std::string& line) {
    if (options.log) options.log(line);
    else core::Logger::instance().info(line);
}

bool cancelled(const std::atomic<bool>* cancel) {
    return cancel != nullptr && cancel->load(std::memory_order_relaxed);
}

int count_images(const std::filesystem::path& directory) {
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) return 0;
    int count = 0;
    for (const auto& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (error) break;
        std::error_code file_error;
        if (entry.is_regular_file(file_error) && is_image_file(entry.path()))
            ++count;
    }
    return count;
}

std::vector<std::filesystem::path> list_images(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> files;
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) return files;
    for (const auto& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (error) break;
        std::error_code file_error;
        if (entry.is_regular_file(file_error) && is_image_file(entry.path()))
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

bool is_preview_still_name(const std::string& name) {
    const auto dot = name.rfind('.');
    if (dot == std::string::npos || dot <= 2) return false;
    std::string ext = name.substr(dot);
    std::transform(
        ext.begin(), ext.end(), ext.begin(),
        [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext != ".jpg" && ext != ".jpeg") return false;
    if (name[0] != 'p' || name[1] != '_') return false;
    for (std::size_t i = 2; i < dot; ++i)
        if (!std::isdigit(static_cast<unsigned char>(name[i]))) return false;
    return dot > 2;
}

void clear_preview_stills(const std::filesystem::path& directory) {
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) return;
    for (const auto& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (error) break;
        std::error_code file_error;
        if (!entry.is_regular_file(file_error)) continue;
        if (is_preview_still_name(entry.path().filename().string()))
            std::filesystem::remove(entry.path(), file_error);
    }
}

#if defined(_WIN32)
std::wstring utf8_to_wide(const std::string& utf8) {
    if (utf8.empty()) return {};
    const int size = MultiByteToWideChar(
        CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (size <= 1) return {};
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, wide.data(), size);
    wide.pop_back();
    return wide;
}

std::wstring quote_windows_arg(const std::wstring& arg) {
    if (arg.find_first_of(L" \t\"") == std::wstring::npos) return arg;
    std::wstring out = L"\"";
    int backslashes = 0;
    for (const wchar_t c : arg) {
        if (c == L'\\') {
            ++backslashes;
            continue;
        }
        if (c == L'"') {
            out.append(static_cast<std::size_t>(backslashes) * 2 + 1, L'\\');
            out.push_back(L'"');
            backslashes = 0;
            continue;
        }
        if (backslashes > 0) {
            out.append(static_cast<std::size_t>(backslashes), L'\\');
            backslashes = 0;
        }
        out.push_back(c);
    }
    if (backslashes > 0)
        out.append(static_cast<std::size_t>(backslashes) * 2, L'\\');
    out.push_back(L'"');
    return out;
}

std::filesystem::path resolve_ffmpeg(const std::filesystem::path& ffmpeg) {
    const auto found = locate_ffmpeg(ffmpeg);
    if (!found.empty()) return found;
    if (!ffmpeg.empty()) return ffmpeg;
    return std::filesystem::path(L"ffmpeg.exe");
}

int run_process(
    const std::vector<std::string>& args,
    const std::function<void(const std::string&)>& on_line,
    std::string& error,
    const std::atomic<bool>* cancel) {
    if (args.empty()) {
        error = "empty command";
        return -1;
    }
    std::wstring command;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i > 0) command.push_back(L' ');
        command += quote_windows_arg(utf8_to_wide(args[i]));
    }

    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;
    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &security, 0)) {
        error = "failed to create pipe for ffmpeg";
        return -1;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    startup.wShowWindow = SW_HIDE;
    startup.hStdOutput = write_pipe;
    startup.hStdError = write_pipe;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

    PROCESS_INFORMATION info{};
    std::wstring mutable_command = command;
    const BOOL created = CreateProcessW(
        nullptr, mutable_command.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &startup, &info);
    CloseHandle(write_pipe);
    if (!created) {
        CloseHandle(read_pipe);
        error = "failed to launch " + args.front();
        return -1;
    }

    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags =
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(
                job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) ||
            !AssignProcessToJobObject(job, info.hProcess)) {
            CloseHandle(job);
            job = nullptr;
        }
    }
    ResumeThread(info.hThread);

    std::string pending;
    char buffer[4096];
    auto flush_chunk = [&](const bool finish) {
        std::size_t start = 0;
        for (std::size_t i = 0; i < pending.size(); ++i) {
            if (pending[i] != '\n' && pending[i] != '\r') continue;
            if (i > start && on_line)
                on_line(pending.substr(start, i - start));
            start = i + 1;
        }
        if (finish) {
            if (start < pending.size() && on_line) on_line(pending.substr(start));
            pending.clear();
        } else if (start > 0) {
            pending.erase(0, start);
        }
    };
    auto kill_tree = [&] {
        if (job) TerminateJobObject(job, 1);
        else TerminateProcess(info.hProcess, 1);
    };

    for (;;) {
        if (cancelled(cancel)) {
            kill_tree();
            break;
        }
        DWORD available = 0;
        if (PeekNamedPipe(read_pipe, nullptr, 0, nullptr, &available, nullptr) &&
            available > 0) {
            DWORD got = 0;
            if (!ReadFile(read_pipe, buffer, sizeof(buffer), &got, nullptr) ||
                got == 0)
                break;
            pending.append(buffer, buffer + got);
            flush_chunk(false);
            continue;
        }
        const DWORD wait = WaitForSingleObject(info.hProcess, 100);
        if (wait != WAIT_OBJECT_0) continue;
        DWORD got = 0;
        while (ReadFile(read_pipe, buffer, sizeof(buffer), &got, nullptr) &&
               got > 0) {
            pending.append(buffer, buffer + got);
        }
        break;
    }
    flush_chunk(true);
    CloseHandle(read_pipe);
    WaitForSingleObject(info.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(info.hProcess, &code);
    CloseHandle(info.hThread);
    CloseHandle(info.hProcess);
    if (job) CloseHandle(job);
    if (cancelled(cancel)) {
        error = "cancelled";
        return -1;
    }
    return static_cast<int>(code);
}
#else
std::filesystem::path resolve_ffmpeg(const std::filesystem::path& ffmpeg) {
    const auto found = locate_ffmpeg(ffmpeg);
    if (!found.empty()) return found;
    if (!ffmpeg.empty()) return ffmpeg;
    return std::filesystem::path("ffmpeg");
}

int run_process(
    const std::vector<std::string>& args,
    const std::function<void(const std::string&)>& on_line,
    std::string& error,
    const std::atomic<bool>* cancel) {
    if (args.empty()) {
        error = "empty command";
        return -1;
    }
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        error = "failed to create pipe for ffmpeg";
        return -1;
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    posix_spawn_file_actions_addclose(&actions, pipefd[1]);

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    std::vector<std::string> storage = args;
    for (auto& argument : storage) argv.push_back(argument.data());
    argv.push_back(nullptr);

    pid_t pid = 0;
    const int spawned = posix_spawnp(
        &pid, storage.front().c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipefd[1]);
    if (spawned != 0) {
        close(pipefd[0]);
        error = "failed to launch " + args.front();
        return -1;
    }

    std::string pending;
    char buffer[4096];
    auto flush_chunk = [&](const bool finish) {
        std::size_t start = 0;
        for (std::size_t i = 0; i < pending.size(); ++i) {
            if (pending[i] != '\n' && pending[i] != '\r') continue;
            if (i > start && on_line)
                on_line(pending.substr(start, i - start));
            start = i + 1;
        }
        if (finish) {
            if (start < pending.size() && on_line) on_line(pending.substr(start));
            pending.clear();
        } else if (start > 0) {
            pending.erase(0, start);
        }
    };
    ssize_t got = 0;
    while ((got = read(pipefd[0], buffer, sizeof(buffer))) > 0) {
        pending.append(buffer, buffer + got);
        flush_chunk(false);
        if (cancelled(cancel)) {
            kill(pid, SIGTERM);
            break;
        }
    }
    flush_chunk(true);
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (cancelled(cancel)) {
        error = "cancelled";
        return -1;
    }
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    error = "ffmpeg terminated abnormally";
    return -1;
}
#endif

std::int64_t file_size_or_zero(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    return error ? 0 : static_cast<std::int64_t>(size);
}

std::int64_t file_mtime_or_zero(const std::filesystem::path& path) {
    std::error_code error;
    const auto stamp = std::filesystem::last_write_time(path, error);
    if (error) return 0;
    return static_cast<std::int64_t>(stamp.time_since_epoch().count());
}

struct ExtractManifest {
    std::string source;
    std::int64_t size{};
    std::int64_t mtime{};
    float fps{};
    int sharp_window{};
    int max_frames{};
    int quality{};
    float scale{};
    int rotate{};
    std::string segments;
};

ExtractManifest make_manifest(const VideoExtractOptions& options) {
    ExtractManifest manifest;
    std::error_code canonical_error;
    auto canonical = std::filesystem::weakly_canonical(options.video, canonical_error);
    if (canonical_error) canonical = options.video;
    manifest.source = path_utf8(canonical);
    manifest.size = file_size_or_zero(options.video);
    manifest.mtime = file_mtime_or_zero(options.video);
    manifest.fps = options.fps;
    manifest.sharp_window = options.sharp_window;
    manifest.max_frames = options.max_frames;
    manifest.quality = options.quality;
    manifest.scale = options.scale;
    manifest.rotate = options.rotate;
    manifest.segments = serialize_video_segments(options.segments);
    return manifest;
}

bool same_manifest(const ExtractManifest& a, const ExtractManifest& b) {
    auto close = [](const float x, const float y) {
        return std::abs(x - y) <= 1.0e-4F;
    };
    return a.source == b.source && a.size == b.size && a.mtime == b.mtime &&
           close(a.fps, b.fps) && a.sharp_window == b.sharp_window &&
           a.max_frames == b.max_frames && a.quality == b.quality &&
           close(a.scale, b.scale) && a.rotate == b.rotate &&
           a.segments == b.segments;
}

std::filesystem::path manifest_path(const std::filesystem::path& output_dir) {
    return output_dir / k_manifest_name;
}

bool load_manifest(const std::filesystem::path& path, ExtractManifest& out) {
    std::ifstream in(path);
    if (!in) return false;
    ExtractManifest loaded;
    std::string line;
    while (std::getline(in, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const std::string value = line.substr(eq + 1);
        if (key == "source") loaded.source = value;
        else if (key == "size") loaded.size = std::strtoll(value.c_str(), nullptr, 10);
        else if (key == "mtime") loaded.mtime = std::strtoll(value.c_str(), nullptr, 10);
        else if (key == "fps") loaded.fps = std::strtof(value.c_str(), nullptr);
        else if (key == "sharp_window") loaded.sharp_window = std::atoi(value.c_str());
        else if (key == "max_frames") loaded.max_frames = std::atoi(value.c_str());
        else if (key == "quality") loaded.quality = std::atoi(value.c_str());
        else if (key == "scale") loaded.scale = std::strtof(value.c_str(), nullptr);
        else if (key == "rotate") loaded.rotate = std::atoi(value.c_str());
        else if (key == "segments") loaded.segments = value;
    }
    out = loaded;
    return !out.source.empty();
}

void save_manifest(
    const std::filesystem::path& path, const ExtractManifest& manifest) {
    std::ofstream out(path, std::ios::trunc);
    if (!out)
        throw std::runtime_error("Failed to write extract manifest: " + path_utf8(path));
    out << "source=" << manifest.source << '\n'
        << "size=" << manifest.size << '\n'
        << "mtime=" << manifest.mtime << '\n'
        << "fps=" << manifest.fps << '\n'
        << "sharp_window=" << manifest.sharp_window << '\n'
        << "max_frames=" << manifest.max_frames << '\n'
        << "quality=" << manifest.quality << '\n'
        << "scale=" << manifest.scale << '\n'
        << "rotate=" << manifest.rotate << '\n'
        << "segments=" << manifest.segments << '\n';
}

void clear_extracted_stills(const std::filesystem::path& directory) {
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) return;
    for (const auto& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (error) break;
        std::error_code file_error;
        if (!entry.is_regular_file(file_error)) continue;
        if (is_extract_still_name(entry.path().filename().string()))
            std::filesystem::remove(entry.path(), file_error);
    }
}

enum class ExtractFolderKind { empty, matching, stale, occupied };

ExtractFolderKind classify_extract_folder(
    const std::filesystem::path& directory, const ExtractManifest& wanted,
    int& frame_count) {
    frame_count = count_images(directory);
    ExtractManifest previous;
    const bool have_manifest = load_manifest(manifest_path(directory), previous);
    if (have_manifest && same_manifest(previous, wanted) && frame_count > 0)
        return ExtractFolderKind::matching;
    if (have_manifest) return ExtractFolderKind::stale;
    if (frame_count == 0) return ExtractFolderKind::empty;
    for (const auto& file : list_images(directory)) {
        if (!is_extract_still_name(file.filename().string()))
            return ExtractFolderKind::occupied;
    }
    return ExtractFolderKind::stale;
}

std::string ffmpeg_filter(const VideoExtractOptions& options, const float fps) {
    std::ostringstream vf;
    vf << "fps=" << fps;
    if (std::abs(options.scale - 1.0F) > 1.0e-4F)
        vf << ",scale=iw*" << options.scale << ":ih*" << options.scale;
    switch (options.rotate) {
        case 90: vf << ",transpose=1"; break;
        case 180: vf << ",transpose=1,transpose=1"; break;
        case 270: vf << ",transpose=2"; break;
        default: break;
    }
    return vf.str();
}

int jpeg_qscale(const int quality) {
    const int q = std::clamp(quality, 0, 100);
    return std::clamp(2 + (100 - q) / 4, 1, 31);
}

void parse_probe_line(const std::string& line, VideoProbe& out) {
    const auto duration_at = line.find("Duration:");
    if (duration_at != std::string::npos) {
        int hh = 0, mm = 0;
        double ss = 0.0;
        if (std::sscanf(
                line.c_str() + duration_at, "Duration: %d:%d:%lf", &hh, &mm, &ss) ==
            3)
            out.duration_seconds = hh * 3600.0 + mm * 60.0 + ss;
    }
    if (line.find("Video:") == std::string::npos) return;
    for (std::size_t i = 0; i + 3 < line.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(line[i]))) continue;
        if (i > 0 && std::isalnum(static_cast<unsigned char>(line[i - 1])))
            continue;
        int width = 0;
        int height = 0;
        if (std::sscanf(line.c_str() + i, "%dx%d", &width, &height) != 2)
            continue;
        if (width >= 16 && height >= 16) {
            out.width = width;
            out.height = height;
            break;
        }
    }
    const auto fps_at = line.find(" fps");
    if (fps_at != std::string::npos) {
        const auto begin = line.rfind(' ', fps_at - 1);
        if (begin != std::string::npos && begin + 1 < fps_at) {
            const double fps = std::strtod(line.c_str() + begin + 1, nullptr);
            if (fps > 0.1 && fps < 1000.0) out.fps = fps;
        }
    }
}

int run_ffmpeg(
    const std::filesystem::path& ffmpeg,
    const std::vector<std::string>& args,
    const std::function<void(const std::string&)>& on_line,
    std::string& error,
    const std::atomic<bool>* cancel) {
    std::vector<std::string> command;
    command.reserve(args.size() + 1);
    command.push_back(path_utf8(ffmpeg));
    command.insert(command.end(), args.begin(), args.end());
    return run_process(command, on_line, error, cancel);
}

}  // namespace

bool is_video_path(const std::filesystem::path& path) {
    const std::string ext = extension_lower(path);
    for (const char* candidate : k_video_extensions)
        if (ext == candidate) return true;
    return false;
}

std::filesystem::path default_video_frames_dir(const std::filesystem::path& video) {
    const auto parent = video.parent_path();
    const auto folder = parent.empty()
        ? std::filesystem::path(video.stem())
        : parent / video.stem();
    return folder / "images";
}

namespace {

bool ffmpeg_is_runnable(const std::filesystem::path& path) {
    std::error_code error;
    if (path.empty() || !std::filesystem::exists(path, error) || error)
        return false;
#if defined(_WIN32)
    return true;
#else
    return access(path.c_str(), X_OK) == 0;
#endif
}

std::filesystem::path process_directory() {
#if defined(_WIN32)
    wchar_t buffer[32768]{};
    const DWORD n = GetModuleFileNameW(nullptr, buffer, 32768);
    if (n == 0 || n >= 32768) return {};
    return std::filesystem::path(buffer).parent_path();
#else
    char buffer[4096]{};
    const ssize_t n = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (n <= 0) return {};
    return std::filesystem::path(std::string(buffer, buffer + n)).parent_path();
#endif
}

std::filesystem::path env_joined(const char* var, const char* relative) {
    const char* value = std::getenv(var);
    if (value == nullptr || value[0] == '\0') return {};
    return std::filesystem::path(value) / relative;
}

std::filesystem::path search_ffmpeg_on_path(const std::filesystem::path& name) {
#if defined(_WIN32)
    std::wstring wide = name.wstring();
    if (wide.find(L'.') == std::wstring::npos) wide += L".exe";
    wchar_t buffer[MAX_PATH]{};
    const DWORD n = SearchPathW(
        nullptr, wide.c_str(), nullptr, MAX_PATH, buffer, nullptr);
    if (n > 0 && n < MAX_PATH) return std::filesystem::path(buffer);
    return {};
#else
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr) return {};
    const std::string paths(path_env);
    std::size_t begin = 0;
    while (begin <= paths.size()) {
        const std::size_t end = paths.find(':', begin);
        const std::string dir = paths.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        const std::filesystem::path candidate =
            std::filesystem::path(dir) / name;
        if (ffmpeg_is_runnable(candidate)) return candidate;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return {};
#endif
}

}  // namespace

std::filesystem::path locate_ffmpeg(const std::filesystem::path& hint) {
    if (!hint.empty() && hint.has_parent_path() &&
        ffmpeg_is_runnable(hint))
        return hint;

#if defined(_WIN32)
    const std::filesystem::path name = hint.empty()
        ? std::filesystem::path(L"ffmpeg.exe")
        : (hint.extension().empty()
               ? std::filesystem::path(hint.native() + L".exe")
               : hint);
#else
    const std::filesystem::path name =
        hint.empty() ? std::filesystem::path("ffmpeg") : hint;
#endif
    const std::filesystem::path filename =
        name.has_parent_path() ? name.filename() : name;
    if (ffmpeg_is_runnable(name)) return name;

    if (const auto on_path = search_ffmpeg_on_path(filename);
        !on_path.empty() && ffmpeg_is_runnable(on_path))
        return on_path;

    const auto beside = process_directory() / filename;
    if (ffmpeg_is_runnable(beside)) return beside;

    const std::filesystem::path extras[] = {
#if defined(_WIN32)
        std::filesystem::path(L"C:\\ffmpeg\\bin") / filename,
        std::filesystem::path(L"C:\\Program Files\\ffmpeg\\bin") / filename,
        std::filesystem::path(L"C:\\Program Files (x86)\\ffmpeg\\bin") /
            filename,
        env_joined("ProgramFiles", "ffmpeg\\bin") / filename,
        env_joined("LOCALAPPDATA", "Microsoft\\WinGet\\Links") / filename,
        env_joined("ProgramData", "chocolatey\\bin") / filename,
        env_joined("USERPROFILE", "scoop\\shims") / filename,
        env_joined("USERPROFILE", "scoop\\apps\\ffmpeg\\current\\bin") /
            filename,
#else
        std::filesystem::path("/usr/bin") / filename,
        std::filesystem::path("/usr/local/bin") / filename,
        std::filesystem::path("/opt/homebrew/bin") / filename,
        env_joined("HOME", ".local/bin") / filename,
#endif
    };
    for (const auto& candidate : extras) {
        if (ffmpeg_is_runnable(candidate)) return candidate;
    }
    return {};
}

bool ffmpeg_available(const std::filesystem::path& ffmpeg) {
    return !locate_ffmpeg(ffmpeg).empty();
}

bool probe_video(
    const std::filesystem::path& ffmpeg,
    const std::filesystem::path& video,
    VideoProbe& out) {
    out = {};
    const auto resolved = resolve_ffmpeg(ffmpeg);
    std::string error;
    run_ffmpeg(
        resolved,
        {"-nostdin", "-hide_banner", "-i", path_utf8(video)},
        [&](const std::string& line) { parse_probe_line(line, out); },
        error, nullptr);
    return out.duration_seconds > 0.0 || out.fps > 0.0 || out.width > 0;
}

double laplacian_sharpness(const RgbImage& image) {
    if (image.width == 0 || image.height == 0 || image.pixels.empty())
        return -1.0;
    constexpr int k_size = 512;
    std::vector<float> gray(static_cast<std::size_t>(k_size) * k_size);
    double mean = 0.0;
    const int width = static_cast<int>(image.width);
    const int height = static_cast<int>(image.height);
    for (int y = 0; y < k_size; ++y) {
        const int y0 = static_cast<int>((static_cast<std::int64_t>(y) * height) / k_size);
        int y1 = static_cast<int>((static_cast<std::int64_t>(y + 1) * height) / k_size);
        if (y1 <= y0) y1 = y0 + 1;
        for (int x = 0; x < k_size; ++x) {
            const int x0 = static_cast<int>((static_cast<std::int64_t>(x) * width) / k_size);
            int x1 = static_cast<int>((static_cast<std::int64_t>(x + 1) * width) / k_size);
            if (x1 <= x0) x1 = x0 + 1;
            double acc = 0.0;
            for (int yy = y0; yy < y1; ++yy) {
                const std::uint8_t* row =
                    image.pixels.data() + static_cast<std::size_t>(yy) * width * 3;
                for (int xx = x0; xx < x1; ++xx) {
                    const std::uint8_t* p = row + xx * 3;
                    acc += 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
                }
            }
            const float value = static_cast<float>(
                acc / static_cast<double>((y1 - y0) * (x1 - x0)));
            gray[static_cast<std::size_t>(y) * k_size + x] = value;
            mean += value;
        }
    }
    mean /= static_cast<double>(k_size) * k_size;
    for (float& value : gray) value -= static_cast<float>(mean);

    double sum = 0.0;
    double sum2 = 0.0;
    std::int64_t n = 0;
    for (int y = 1; y < k_size - 1; ++y) {
        for (int x = 1; x < k_size - 1; ++x) {
            const float* r = &gray[static_cast<std::size_t>(y) * k_size + x];
            const double lap =
                static_cast<double>(r[-k_size]) + r[k_size] + r[-1] + r[1] -
                4.0 * r[0];
            sum += lap;
            sum2 += lap * lap;
            ++n;
        }
    }
    if (n == 0) return -1.0;
    const double mu = sum / static_cast<double>(n);
    return sum2 / static_cast<double>(n) - mu * mu;
}

double laplacian_sharpness(const std::filesystem::path& path) {
    try {
        return laplacian_sharpness(load_rgb(path));
    } catch (...) {
        return -1.0;
    }
}

namespace {
using SharpnessScores = std::unordered_map<std::filesystem::path, double>;

// Score already closed candidate files while ffmpeg continues decoding. Workers
// are bounded so the scorer leaves CPU capacity for video decoding/encoding.
class SharpnessPipeline {
public:
    SharpnessPipeline(bool enabled, const std::atomic<bool>* cancel) : cancel_(cancel) {
        const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
        const unsigned count = enabled ? std::min(4u, std::max(1u, cores / 2)) : 0;
        for (unsigned i = 0; i < count; ++i) {
            workers_.emplace_back([this](std::stop_token token) {
                for (;;) {
                    std::filesystem::path path;
                    {
                        std::unique_lock lock(mutex_);
                        available_.wait(lock, token, [&] { return closing_ || !pending_.empty(); });
                        if (token.stop_requested() || cancelled(cancel_) ||
                            (closing_ && pending_.empty())) return;
                        if (pending_.empty()) continue;
                        path = std::move(pending_.front());
                        pending_.pop_front();
                    }
                    const double score = laplacian_sharpness(path);
                    std::lock_guard lock(mutex_);
                    scores_.emplace(std::move(path), score);
                }
            });
        }
    }
    ~SharpnessPipeline() {
        for (auto& worker : workers_) worker.request_stop();
        available_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
    void enqueue(std::filesystem::path path) {
        if (workers_.empty() || cancelled(cancel_)) return;
        {
            std::lock_guard lock(mutex_);
            pending_.push_back(std::move(path));
        }
        available_.notify_one();
    }
    SharpnessScores finish() {
        {
            std::lock_guard lock(mutex_);
            closing_ = true;
        }
        available_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
        return std::move(scores_);
    }
private:
    const std::atomic<bool>* cancel_;
    std::mutex mutex_;
    std::condition_variable_any available_;
    std::deque<std::filesystem::path> pending_;
    SharpnessScores scores_;
    bool closing_{};
    std::vector<std::jthread> workers_;
};
} // namespace

static int select_sharpest_frames_with_scores(

    const std::filesystem::path& candidate_dir,
    const std::filesystem::path& output_dir,
    int window,
    int max_frames,
    const std::atomic<bool>* cancel,
    const std::function<void(const std::string&)>& log,
    int first_index,
    const SharpnessScores& prepared_scores) {
    auto files = list_images(candidate_dir);
    if (files.empty()) return 0;
    window = std::max(window, 1);
    if (max_frames > 0)
        window = std::max(
            window,
            static_cast<int>(
                (files.size() + static_cast<std::size_t>(max_frames) - 1) /
                static_cast<std::size_t>(max_frames)));

    std::vector<double> scores(files.size(), -1.0);
    std::vector<std::size_t> unscored;
    if (window > 1) {
        for (std::size_t i = 0; i < files.size(); ++i) {
            const auto prepared = prepared_scores.find(files[i]);
            if (prepared != prepared_scores.end() && prepared->second >= 0.0)
                scores[i] = prepared->second;
            else
                unscored.push_back(i);
        }
    }
    if (!unscored.empty()) {
        const unsigned threads = static_cast<unsigned>(std::min<std::size_t>(
            unscored.size(), std::min(8u, std::max(1u, std::thread::hardware_concurrency()))));
        std::atomic<std::size_t> next{0};
        std::atomic<std::size_t> done{0};
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (unsigned t = 0; t < threads; ++t) {
            pool.emplace_back([&] {
                for (;;) {
                    const std::size_t next_item = next.fetch_add(1);
                    if (next_item >= unscored.size() || cancelled(cancel)) return;
                    const std::size_t i = unscored[next_item];
                    try {
                        scores[i] = laplacian_sharpness(files[i]);
                    } catch (...) {
                        scores[i] = -1.0;
                    }
                    done.fetch_add(1);
                }
            });
        }
        auto last_report = std::chrono::steady_clock::now();
        while (done.load() < unscored.size() && !cancelled(cancel)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            const auto now = std::chrono::steady_clock::now();
            if (log && now - last_report >= std::chrono::seconds(1)) {
                log("select sharp frames scored " + std::to_string(
                        files.size() - unscored.size() + done.load()) +
                    "/" + std::to_string(files.size()));
                last_report = now;
            }
        }
        for (auto& thread : pool) thread.join();
        if (cancelled(cancel)) return -1;
    }

    std::error_code error;
    std::filesystem::create_directories(output_dir, error);
    int kept = 0;
    core::ProgressReporter progress(
        "select sharp frames",
        (files.size() + static_cast<std::size_t>(window) - 1) /
            static_cast<std::size_t>(window));
    for (std::size_t g0 = 0; g0 < files.size(); g0 += static_cast<std::size_t>(window)) {
        if (cancelled(cancel)) return -1;
        const std::size_t g1 = std::min(
            g0 + static_cast<std::size_t>(window), files.size());
        std::size_t best = g0;
        for (std::size_t i = g0 + 1; i < g1; ++i)
            if (scores[i] > scores[best]) best = i;
        char name[64];
        const std::string ext = files[best].extension().string();
        std::snprintf(
            name, sizeof(name), "%05d%s", first_index + kept,
            ext.empty() ? ".jpg" : ext.c_str());
        const auto destination = output_dir / name;
        std::error_code io;
        std::filesystem::rename(files[best], destination, io);
        if (io) {
            io.clear();
            std::filesystem::copy_file(
                files[best], destination,
                std::filesystem::copy_options::overwrite_existing, io);
            if (io) {
                throw std::runtime_error(
                    "Failed to keep frame " + path_utf8(files[best]) + ": " +
                    io.message());
            }
            std::error_code remove_error;
            std::filesystem::remove(files[best], remove_error);
        }
        ++kept;
        progress.advance();
        for (std::size_t i = g0; i < g1; ++i)
            if (i != best) std::filesystem::remove(files[i], error);
    }
    return kept;
}

int select_sharpest_frames(
    const std::filesystem::path& candidate_dir,
    const std::filesystem::path& output_dir,
    int window, int max_frames, const std::atomic<bool>* cancel,
    const std::function<void(const std::string&)>& log, int first_index) {
    return select_sharpest_frames_with_scores(candidate_dir, output_dir, window,
        max_frames, cancel, log, first_index, {});
}

VideoExtractResult extract_video_frames(const VideoExtractOptions& options_in) {
    VideoExtractOptions options = options_in;
    if (options.video.empty())
        throw std::invalid_argument("Video path is empty");
    if (options.fps < 0.01F) options.fps = 0.01F;
    if (options.sharp_window < 1) options.sharp_window = 1;
    if (options.max_frames < 0) options.max_frames = 0;
    if (options.scale <= 0.0F) options.scale = 1.0F;
    if (options.rotate % 90 != 0)
        throw std::invalid_argument("Video rotation must be 0, 90, 180, or 270");
    options.rotate = ((options.rotate % 360) + 360) % 360;

    std::error_code error;
    if (!std::filesystem::is_regular_file(options.video, error))
        throw std::runtime_error("Video file not found: " + path_utf8(options.video));

    if (options.output_dir.empty())
        options.output_dir = default_video_frames_dir(options.video);
    std::filesystem::create_directories(options.output_dir, error);
    if (error)
        throw std::runtime_error(
            "Cannot create frame directory: " + path_utf8(options.output_dir));

    const ExtractManifest wanted = make_manifest(options);
    const auto stored_manifest = manifest_path(options.output_dir);
    int existing_frames = 0;
    const auto folder =
        classify_extract_folder(options.output_dir, wanted, existing_frames);
    if (folder == ExtractFolderKind::occupied) {
        throw std::runtime_error(
            "Frame directory already contains images that are not an "
            "Photara video extract: " +
            path_utf8(options.output_dir) +
            ". Choose an empty folder, or a previous extract.");
    }
    if (folder == ExtractFolderKind::matching && options.resume) {
        emit(
            options,
            "reusing " + std::to_string(existing_frames) +
                " extracted frames in " + path_utf8(options.output_dir));
        return {options.output_dir, existing_frames, true};
    }
    if (folder == ExtractFolderKind::matching ||
        folder == ExtractFolderKind::stale)
        clear_extracted_stills(options.output_dir);

    struct PartialExtractGuard {
        std::filesystem::path output;
        std::filesystem::path temp;
        bool armed{true};
        ~PartialExtractGuard() {
            if (!armed) return;
            std::error_code ignored;
            clear_extracted_stills(output);
            if (!temp.empty()) std::filesystem::remove_all(temp, ignored);
        }
    };
    PartialExtractGuard partial;
    partial.output = options.output_dir;

    const auto ffmpeg = resolve_ffmpeg(options.ffmpeg);
    if (!ffmpeg_available(ffmpeg))
        throw std::runtime_error(
            "ffmpeg was not found (" + path_utf8(ffmpeg) +
            "). Install ffmpeg and add it to PATH, or set --ffmpeg.");

    VideoProbe probe{};
    probe_video(ffmpeg, options.video, probe);
    const int window = options.sharp_window;
    const float candidate_fps = options.fps * static_cast<float>(window);
    const bool jpeg = options.quality >= 0 && options.quality <= 100;
    const std::string pattern = jpeg ? "c_%06d.jpg" : "c_%06d.png";
    const std::vector<VideoSegment> ranges =
        normalize_video_segments(options.segments);
    const bool segmented = !ranges.empty();
    auto report = [&](const float fraction) {
        if (options.progress) options.progress(fraction);
    };

    double expected_seconds = probe.duration_seconds;
    if (segmented) {
        expected_seconds = 0.0;
        for (const VideoSegment& range : ranges) {
            const double start = std::max(0.0, range.start_seconds);
            double end = range.end_seconds > range.start_seconds
                ? range.end_seconds
                : probe.duration_seconds;
            if (probe.duration_seconds > 0.0)
                end = std::min(end, probe.duration_seconds);
            if (end > start) expected_seconds += end - start;
        }
    }
    const int total_limit = options.max_frames > 0
        ? options.max_frames * window
        : 0;
    std::uint64_t expected_candidates = expected_seconds > 0.0
        ? static_cast<std::uint64_t>(
              expected_seconds * static_cast<double>(candidate_fps) + 0.5)
        : 0;
    if (total_limit > 0)
        expected_candidates = std::min(
            expected_candidates,
            static_cast<std::uint64_t>(total_limit));
    if (expected_candidates == 0) report(-1.F);

    std::filesystem::path candidate_root =
        options.output_dir.parent_path() / ".photara_frames_tmp";
    std::filesystem::path write_dir = options.output_dir;
    if (window > 1 || segmented) {
        std::filesystem::remove_all(candidate_root, error);
        std::filesystem::create_directories(candidate_root, error);
        partial.temp = candidate_root;
        if (!segmented) write_dir = candidate_root;
    }

    auto format_seconds = [](const double seconds) {
        char text[32];
        std::snprintf(text, sizeof(text), "%.3f", seconds);
        return std::string(text);
    };
    std::string last_error_line;
    std::string launch_error;
    SharpnessScores prepared_scores;
    const auto extraction_started = std::chrono::steady_clock::now();
    auto run_range = [&](
        const std::filesystem::path& directory, const double start_seconds,
        const double range_seconds, const int frame_limit,
        const std::uint64_t completed_before) {
        std::vector<std::string> args = {
            "-nostdin", "-y", "-hide_banner", "-progress", "pipe:1"};
        if (start_seconds > 0.0) {
            args.push_back("-ss");
            args.push_back(format_seconds(start_seconds));
        }
        args.push_back("-i");
        args.push_back(path_utf8(options.video));
        if (range_seconds > 0.0) {
            args.push_back("-t");
            args.push_back(format_seconds(range_seconds));
        }
        args.insert(args.end(), {"-an", "-sn"});
        args.push_back("-vf");
        args.push_back(ffmpeg_filter(options, candidate_fps));
        if (jpeg) {
            args.emplace_back("-qscale:v");
            args.push_back(std::to_string(jpeg_qscale(options.quality)));
        } else {
            // PNG compression affects file size, not pixel values.
            args.insert(args.end(), {"-compression_level", "1"});
        }
        if (frame_limit > 0) {
            args.emplace_back("-frames:v");
            args.push_back(std::to_string(frame_limit));
        }
        args.push_back(path_utf8(directory / pattern));

        std::uint64_t last_frame = 0;
        std::uint64_t next_to_score = 1;
        SharpnessPipeline scorer(window > 1, options.cancel);
        const auto enqueue_scores = [&](std::uint64_t through) {
            if (window <= 1) return;
            while (next_to_score <= through) {
                char name[40];
                std::snprintf(name, sizeof(name), "c_%06llu.%s",
                    static_cast<unsigned long long>(next_to_score++), jpeg ? "jpg" : "png");
                scorer.enqueue(directory / name);
            }
        };
        core::ProgressReporter progress("extract video frames", expected_candidates);
        const auto decoding_started = std::chrono::steady_clock::now();
        const int rc = run_ffmpeg(
            ffmpeg, args,
            [&](const std::string& line) {
                if (line.rfind("frame=", 0) == 0) {
                    const auto frame = static_cast<std::uint64_t>(
                        std::strtoull(line.c_str() + 6, nullptr, 10));
                    if (frame > last_frame) {
                        progress.advance(frame - last_frame);
                        last_frame = frame;
                        // Leave the newest file alone until ffmpeg closes it.
                        enqueue_scores(frame > 0 ? frame - 1 : 0);
                        char preview_name[40];
                        std::snprintf(preview_name, sizeof(preview_name),
                            "c_%06llu.%s", static_cast<unsigned long long>(
                                frame > 1 ? frame - 1 : frame), jpeg ? "jpg" : "png");
                        core::Logger::instance().info(
                            "capture_frame kind=video ready=1 completed=",
                            completed_before + frame, " total=", expected_candidates,
                            " image=", std::quoted(path_utf8(directory / preview_name)));
                        if (expected_candidates > 0) {
                            const double done = static_cast<double>(
                                completed_before + frame);
                            report(static_cast<float>(
                                0.90 * std::min(1.0, done /
                                    static_cast<double>(expected_candidates))));
                        }
                    }
                } else if (
                    line.find("error") != std::string::npos ||
                    line.find("Error") != std::string::npos ||
                    line.find("Invalid") != std::string::npos) {
                    last_error_line = line;
                }
            },
            launch_error, options.cancel);
        const auto decoding_finished = std::chrono::steady_clock::now();
        if (rc == 0) {
            enqueue_scores(last_frame);
            auto scores = scorer.finish();
            core::Logger::instance().info("video_extract_stage decode_ms=",
                std::chrono::duration<double, std::milli>(
                    decoding_finished - decoding_started).count(),
                " score_wait_ms=", std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - decoding_finished).count(),
                " scored=", scores.size());
            prepared_scores.merge(scores);
        }
        return rc;
    };
    auto throw_ffmpeg_failure = [&]() {
        std::filesystem::remove_all(candidate_root, error);
        if (cancelled(options.cancel)) throw std::runtime_error("cancelled");
        std::string message = "ffmpeg failed to extract frames";
        if (!launch_error.empty()) message += ": " + launch_error;
        if (!last_error_line.empty()) message += " (" + last_error_line + ")";
        throw std::runtime_error(message);
    };

    emit(options, "extract video frames from " + path_utf8(options.video));
    int written = 0;
    if (segmented) {
        std::uint64_t candidates_done = 0;
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            if (options.max_frames > 0 && written >= options.max_frames) break;
            const VideoSegment& range = ranges[i];
            const double start = std::max(0.0, range.start_seconds);
            double end = range.end_seconds > range.start_seconds
                ? range.end_seconds
                : probe.duration_seconds;
            if (probe.duration_seconds > 0.0)
                end = std::min(end, probe.duration_seconds);
            const double duration = end > start ? end - start : 0.0;
            const int remaining = options.max_frames > 0
                ? options.max_frames - written
                : 0;
            char folder[32];
            std::snprintf(folder, sizeof(folder), "seg_%04zu", i);
            const auto segment_dir = candidate_root / folder;
            std::filesystem::create_directories(segment_dir, error);
            if (error)
                throw std::runtime_error(
                    "Cannot create video candidate directory: " +
                    path_utf8(segment_dir));
            const int rc = run_range(
                segment_dir, start, duration,
                remaining > 0 ? remaining * window : 0, candidates_done);
            if (rc != 0) throw_ffmpeg_failure();
            candidates_done += static_cast<std::uint64_t>(
                count_images(segment_dir));
            const int kept = select_sharpest_frames_with_scores(
                segment_dir, options.output_dir, window, remaining,
                options.cancel, options.log, written, prepared_scores);
            prepared_scores.clear();
            std::filesystem::remove_all(segment_dir, error);
            if (kept < 0) throw std::runtime_error("cancelled");
            written += kept;
        }
        std::filesystem::remove_all(candidate_root, error);
    } else {
        const int rc = run_range(
            write_dir, 0.0, 0.0,
            options.max_frames > 0 ? options.max_frames * window : 0, 0);
        if (rc != 0) throw_ffmpeg_failure();
        if (window > 1) {
            written = select_sharpest_frames_with_scores(
                candidate_root, options.output_dir, window, options.max_frames,
                options.cancel, options.log, 0, prepared_scores);
            prepared_scores.clear();
            std::filesystem::remove_all(candidate_root, error);
            if (written < 0) throw std::runtime_error("cancelled");
        } else {
            auto files = list_images(options.output_dir);
            if (options.max_frames > 0 &&
                static_cast<int>(files.size()) > options.max_frames) {
                for (std::size_t i = static_cast<std::size_t>(options.max_frames);
                     i < files.size(); ++i)
                    std::filesystem::remove(files[i], error);
                files.resize(static_cast<std::size_t>(options.max_frames));
            }
            int index = 0;
            for (const auto& file : files) {
                char name[64];
                std::snprintf(
                    name, sizeof(name), "%05d%s", index,
                    file.extension().string().c_str());
                const auto destination = options.output_dir / name;
                if (file.filename() != name) {
                    std::error_code io;
                    std::filesystem::rename(file, destination, io);
                    if (io) {
                        throw std::runtime_error(
                            "Failed to write frame " + path_utf8(destination) +
                            ": " + io.message());
                    }
                }
                ++index;
            }
            written = index;
        }
    }
    report(1.F);

    if (written < 2) {
        throw std::runtime_error(
            "Need at least two frames from " + path_utf8(options.video) +
            " (got " + std::to_string(written) +
            "). Increase --video-fps or the clip length.");
    }

    if (written > 0) {
        char final_name[40];
        std::snprintf(final_name, sizeof(final_name), "%05d.%s", written - 1,
            jpeg ? "jpg" : "png");
        core::Logger::instance().info(
            "capture_frame kind=video ready=1 completed=", written,
            " total=", written, " image=",
            std::quoted(path_utf8(options.output_dir / final_name)));
    }
    core::Logger::instance().info("video_extract_timing total_ms=",
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - extraction_started).count(),
        " frames=", written, " sharpness_pipeline=", window > 1);
    save_manifest(stored_manifest, wanted);
    partial.armed = false;
    emit(
        options,
        "video_frames_dir=" + path_utf8(options.output_dir) +
            " frames=" + std::to_string(written));
    return {options.output_dir, written, false};
}

int generate_video_preview(const VideoPreviewOptions& options_in) {
    VideoPreviewOptions options = options_in;
    if (options.video.empty())
        throw std::invalid_argument("Video path is empty");
    if (options.output_dir.empty())
        throw std::invalid_argument("Video preview needs an output directory");
    if (!(options.fps > 0.0F) || !std::isfinite(options.fps))
        throw std::invalid_argument("Video preview fps must be positive");
    options.fps = std::clamp(options.fps, 0.5F, 30.F);
    if (!std::isfinite(options.start_seconds) || options.start_seconds < 0.0 ||
        !std::isfinite(options.duration_seconds) || options.duration_seconds < 0.0 ||
        options.first_frame < 1)
        throw std::invalid_argument("Invalid video preview range");
    options.max_width = std::max(64, options.max_width);
    options.max_height = std::max(64, options.max_height);
    options.quality = std::clamp(options.quality, 1, 100);
    if (!std::filesystem::is_regular_file(options.video))
        throw std::runtime_error(
            "Video file not found: " + path_utf8(options.video));

    std::error_code error;
    std::filesystem::create_directories(options.output_dir, error);
    if (error)
        throw std::runtime_error(
            "Cannot create video preview directory: " +
            path_utf8(options.output_dir));
    if (!options.keep_existing) clear_preview_stills(options.output_dir);

    const auto ffmpeg = resolve_ffmpeg(options.ffmpeg);
    if (!ffmpeg_available(ffmpeg))
        throw std::runtime_error(
            "ffmpeg was not found (" + path_utf8(ffmpeg) +
            "). Install ffmpeg and add it to PATH, or set --ffmpeg.");

    VideoProbe probe = options.source_probe;
    if (probe.width <= 0 || probe.duration_seconds <= 0.0)
        probe_video(ffmpeg, options.video, probe);
    int target_width = options.max_width;
    if (probe.width > 0 && probe.height > 0) {
        const double scale = std::min(
            1.0, std::min(
                     static_cast<double>(options.max_width) / probe.width,
                     static_cast<double>(options.max_height) / probe.height));
        target_width = std::max(
            2, static_cast<int>(probe.width * scale) & ~1);
    }
    std::ostringstream filter;
    filter << "fps=" << options.fps << ",scale=" << target_width << ":-2:flags=fast_bilinear";
    const auto jpeg_quality = jpeg_qscale(options.quality);
    std::vector<std::string> args = {
        "-nostdin", "-y", "-hide_banner", "-progress", "pipe:1", "-stats_period", "0.1"};
    if (options.start_seconds > 0.0) {
        args.insert(args.end(), {"-ss", std::to_string(options.start_seconds)});
    }
    args.insert(args.end(), {"-i", path_utf8(options.video)});
    if (options.duration_seconds > 0.0)
        args.insert(args.end(), {"-t", std::to_string(options.duration_seconds)});
    args.insert(args.end(), {"-an", "-sn", "-vf", filter.str(), "-qscale:v",
        std::to_string(jpeg_quality), "-start_number", std::to_string(options.first_frame),
        path_utf8(options.output_dir / "p_%06d.jpg")});

    const std::uint64_t expected = probe.duration_seconds > 0.0
        ? static_cast<std::uint64_t>(
              (options.duration_seconds > 0.0 ? std::min(options.duration_seconds,
                  std::max(0.0, probe.duration_seconds - options.start_seconds))
                  : probe.duration_seconds) * static_cast<double>(options.fps) + 0.5)
        : 0;
    if (options.progress && expected == 0) options.progress(-1.F);
    std::string launch_error;
    std::string last_error_line;
    const int rc = run_ffmpeg(
        ffmpeg, args,
        [&](const std::string& line) {
            if (line.rfind("frame=", 0) == 0 && options.progress && expected > 0) {
                const auto frame = static_cast<std::uint64_t>(
                    std::strtoull(line.c_str() + 6, nullptr, 10));
                options.progress(static_cast<float>(std::min(
                    1.0, static_cast<double>(frame) /
                             static_cast<double>(expected))));
            } else if (
                line.find("error") != std::string::npos ||
                line.find("Error") != std::string::npos ||
                line.find("Invalid") != std::string::npos) {
                last_error_line = line;
            }
        },
        launch_error, options.cancel);
    if (rc != 0) {
        if (cancelled(options.cancel)) throw std::runtime_error("cancelled");
        std::string message = "ffmpeg failed to build the video preview";
        if (!launch_error.empty()) message += ": " + launch_error;
        if (!last_error_line.empty()) message += " (" + last_error_line + ")";
        throw std::runtime_error(message);
    }
    int frames = 0;
    for (const auto& entry :
         std::filesystem::directory_iterator(options.output_dir, error)) {
        if (error) break;
        std::error_code file_error;
        if (entry.is_regular_file(file_error) &&
            is_preview_still_name(entry.path().filename().string()) &&
            std::atoi(entry.path().stem().string().c_str() + 2) >= options.first_frame)
            ++frames;
    }
    if (options.progress) options.progress(1.F);
    return frames;
}

bool has_matching_video_extract(const VideoExtractOptions& options_in) {
    VideoExtractOptions options = options_in;
    if (options.video.empty()) return false;
    if (options.output_dir.empty())
        options.output_dir = default_video_frames_dir(options.video);
    int frames = 0;
    return classify_extract_folder(
               options.output_dir, make_manifest(options), frames) ==
           ExtractFolderKind::matching;
}

bool has_video_extract_manifest(const std::filesystem::path& output_dir) {
    ExtractManifest unused;
    return load_manifest(manifest_path(output_dir), unused);
}

}  // namespace photara::io
