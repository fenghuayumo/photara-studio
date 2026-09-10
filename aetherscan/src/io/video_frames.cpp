#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "io/video_frames.hpp"

#include "core/logging.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
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

namespace aetherscan::io {
namespace {

constexpr const char* k_video_extensions[] = {
    ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".insv", ".osv", ".avi",
    ".mts", ".m2ts", ".360", ".ts", ".wmv", ".mpeg", ".mpg", ".3gp"};

constexpr const char* k_image_extensions[] = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"};

constexpr const char* k_manifest_name = ".aetherscan_extract";

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
    std::filesystem::path exe = ffmpeg.empty()
        ? std::filesystem::path(L"ffmpeg.exe")
        : ffmpeg;
    std::error_code error;
    if (exe.has_parent_path() && std::filesystem::exists(exe, error))
        return exe;
    std::wstring name = exe.wstring();
    if (name.find(L'.') == std::wstring::npos) name += L".exe";
    wchar_t buffer[MAX_PATH]{};
    const DWORD n = SearchPathW(
        nullptr, name.c_str(), nullptr, MAX_PATH, buffer, nullptr);
    if (n > 0 && n < MAX_PATH) return std::filesystem::path(buffer);
    return exe;
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
    std::filesystem::path exe = ffmpeg.empty()
        ? std::filesystem::path("ffmpeg")
        : ffmpeg;
    std::error_code error;
    if (exe.has_parent_path() && std::filesystem::exists(exe, error))
        return exe;
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr) return exe;
    std::string paths(path_env);
    std::size_t begin = 0;
    while (begin <= paths.size()) {
        const std::size_t end = paths.find(':', begin);
        const std::string dir = paths.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        const std::filesystem::path candidate = std::filesystem::path(dir) / exe;
        if (std::filesystem::exists(candidate, error) &&
            access(candidate.c_str(), X_OK) == 0)
            return candidate;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return exe;
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
    return manifest;
}

bool same_manifest(const ExtractManifest& a, const ExtractManifest& b) {
    auto close = [](const float x, const float y) {
        return std::abs(x - y) <= 1.0e-4F;
    };
    return a.source == b.source && a.size == b.size && a.mtime == b.mtime &&
           close(a.fps, b.fps) && a.sharp_window == b.sharp_window &&
           a.max_frames == b.max_frames && a.quality == b.quality &&
           close(a.scale, b.scale) && a.rotate == b.rotate;
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
        << "rotate=" << manifest.rotate << '\n';
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
        const auto begin = line.rfind(' ', fps_at);
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

bool ffmpeg_available(const std::filesystem::path& ffmpeg) {
    const auto resolved = resolve_ffmpeg(ffmpeg);
    std::error_code error;
    if (resolved.has_parent_path() && std::filesystem::exists(resolved, error))
        return true;
#if defined(_WIN32)
    wchar_t buffer[MAX_PATH]{};
    std::wstring name = resolved.wstring();
    if (name.find(L'.') == std::wstring::npos) name += L".exe";
    const DWORD n = SearchPathW(
        nullptr, name.c_str(), nullptr, MAX_PATH, buffer, nullptr);
    return n > 0 && n < MAX_PATH;
#else
    return access(resolved.c_str(), X_OK) == 0;
#endif
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

int select_sharpest_frames(
    const std::filesystem::path& candidate_dir,
    const std::filesystem::path& output_dir,
    int window,
    int max_frames,
    const std::atomic<bool>* cancel,
    const std::function<void(const std::string&)>& log) {
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
    if (window > 1) {
        const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
        std::atomic<std::size_t> next{0};
        std::atomic<std::size_t> done{0};
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (unsigned t = 0; t < threads; ++t) {
            pool.emplace_back([&] {
                for (;;) {
                    const std::size_t i = next.fetch_add(1);
                    if (i >= files.size() || cancelled(cancel)) return;
                    try {
                        scores[i] = laplacian_sharpness(files[i]);
                    } catch (...) {
                        scores[i] = -1.0;
                    }
                    done.fetch_add(1);
                }
            });
        }
        while (done.load() < files.size() && !cancelled(cancel)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            if (log)
                log("select sharp frames scored " + std::to_string(done.load()) +
                    "/" + std::to_string(files.size()));
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
            name, sizeof(name), "%05d%s", kept,
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
            "AetherScan video extract: " +
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

    std::filesystem::path write_dir = options.output_dir;
    std::filesystem::path candidate_dir;
    if (window > 1) {
        candidate_dir = options.output_dir.parent_path() / ".aetherscan_frames_tmp";
        std::filesystem::remove_all(candidate_dir, error);
        std::filesystem::create_directories(candidate_dir, error);
        write_dir = candidate_dir;
        partial.temp = candidate_dir;
    }

    std::vector<std::string> args = {
        "-nostdin", "-y", "-hide_banner", "-progress", "pipe:1",
        "-i", path_utf8(options.video),
        "-vf", ffmpeg_filter(options, candidate_fps)};
    if (jpeg) {
        args.emplace_back("-qscale:v");
        args.push_back(std::to_string(jpeg_qscale(options.quality)));
    }
    int limit = 0;
    if (options.max_frames > 0)
        limit = options.max_frames * window;
    if (limit > 0) {
        args.emplace_back("-frames:v");
        args.push_back(std::to_string(limit));
    }
    args.push_back(path_utf8(write_dir / pattern));

    emit(options, "extract video frames from " + path_utf8(options.video));
    const std::uint64_t expected = [&]() -> std::uint64_t {
        if (probe.duration_seconds <= 0.0) return 0;
        const double n = probe.duration_seconds * static_cast<double>(candidate_fps);
        if (n <= 0.0) return 0;
        auto count = static_cast<std::uint64_t>(n + 0.5);
        if (limit > 0)
            count = std::min(count, static_cast<std::uint64_t>(limit));
        return count;
    }();
    core::ProgressReporter progress("extract video frames", expected);
    std::uint64_t last_frame = 0;
    std::string last_error_line;
    std::string launch_error;
    const int rc = run_ffmpeg(
        ffmpeg, args,
        [&](const std::string& line) {
            if (line.rfind("frame=", 0) == 0) {
                const auto frame = static_cast<std::uint64_t>(
                    std::strtoull(line.c_str() + 6, nullptr, 10));
                if (frame > last_frame) {
                    progress.advance(frame - last_frame);
                    last_frame = frame;
                }
            } else if (
                line.find("error") != std::string::npos ||
                line.find("Error") != std::string::npos ||
                line.find("Invalid") != std::string::npos) {
                last_error_line = line;
            }
        },
        launch_error, options.cancel);
    if (rc != 0) {
        if (window > 1) std::filesystem::remove_all(candidate_dir, error);
        std::string message = "ffmpeg failed to extract frames";
        if (!launch_error.empty()) message += ": " + launch_error;
        if (!last_error_line.empty()) message += " (" + last_error_line + ")";
        throw std::runtime_error(message);
    }

    int written = 0;
    if (window > 1) {
        written = select_sharpest_frames(
            candidate_dir, options.output_dir, window, options.max_frames,
            options.cancel, options.log);
        std::filesystem::remove_all(candidate_dir, error);
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

    if (written < 2) {
        throw std::runtime_error(
            "Need at least two frames from " + path_utf8(options.video) +
            " (got " + std::to_string(written) +
            "). Increase --video-fps or the clip length.");
    }

    save_manifest(stored_manifest, wanted);
    partial.armed = false;
    emit(
        options,
        "video_frames_dir=" + path_utf8(options.output_dir) +
            " frames=" + std::to_string(written));
    return {options.output_dir, written, false};
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

}  // namespace aetherscan::io
