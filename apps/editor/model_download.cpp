#include "model_download.hpp"
#include "core/sha256.hpp"
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif
namespace editor {
namespace {
#if defined(_WIN32)
std::wstring utf8_to_wide(const std::string& text) {
    if (text.empty()) return {};
    const int size = MultiByteToWideChar(
        CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(),
        size);
    return wide;
}

std::filesystem::path curl_executable() {
    wchar_t system[MAX_PATH]{};
    const UINT length = GetSystemDirectoryW(system, MAX_PATH);
    if (length > 0 && length < MAX_PATH) {
        const auto candidate =
            std::filesystem::path(system) / L"curl.exe";
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error)) return candidate;
    }
    return L"curl.exe";
}

#endif
std::filesystem::path receipt_path(const ModelDownloadSpec& spec) {
    auto path=spec.destination;path+=".verified";return path;
}
void write_receipt(const ModelDownloadSpec& spec) {
    if (spec.sha256.empty()) return;
    const auto path=receipt_path(spec);
    std::ofstream output(path,std::ios::trunc);
    output<<spec.sha256<<'\n'<<std::filesystem::file_size(spec.destination)<<'\n'
        <<std::filesystem::last_write_time(spec.destination).time_since_epoch().count()<<'\n';
    if (!output) throw std::runtime_error("Cannot save model verification receipt");
}
}
bool verified_download_ready(const ModelDownloadSpec& spec) {
    std::error_code error;
    if (spec.sha256.empty() || !std::filesystem::is_regular_file(spec.destination,error) || error) return false;
    const auto size=std::filesystem::file_size(spec.destination,error);
    if (error || size!=spec.expected_bytes) return false;
    const auto stamp=std::filesystem::last_write_time(spec.destination,error).time_since_epoch().count();
    if (error) return false;
    std::ifstream input(receipt_path(spec));
    std::string hash;std::uint64_t stored_size{};std::int64_t stored_stamp{};
    return bool(input>>hash>>stored_size>>stored_stamp) && hash==spec.sha256 && size==stored_size && stamp==stored_stamp;
}
ModelDownload::~ModelDownload() {
    cancel();
    if (worker_.joinable()) worker_.join();
}

void ModelDownload::start(ModelDownloadSpec spec) {
    if (state() == State::running) return;
    if (worker_.joinable()) worker_.join();
    spec_ = std::move(spec);
    cancel_ = false;
    progress_ = -1.F;
    {
        std::lock_guard lock(mutex_);
        status_ = "Starting download";
    }
    state_ = static_cast<int>(State::running);
    worker_ = std::thread([this] {
        try { run(); } catch (const std::exception& e) {
            terminal(cancel_ ? "Download cancelled" : e.what(), cancel_ ? State::cancelled : State::failed);
        }
    });
}

void ModelDownload::cancel() { cancel_ = true; }

ModelDownload::State ModelDownload::state() const {
    return static_cast<State>(state_.load());
}

float ModelDownload::progress() const { return progress_.load(); }

std::string ModelDownload::status() const {
    std::lock_guard lock(mutex_);
    return status_;
}

void ModelDownload::run() {
    if (cancel_) {terminal("Download cancelled",State::cancelled);return;}
    auto fail = [&](const std::string& why, const State terminal) {
        std::lock_guard lock(mutex_);
        status_ = why;
        state_ = static_cast<int>(terminal);
    };

    const auto destination = spec_.destination;
    if (destination.empty() || spec_.url.empty()) throw std::runtime_error("Model download is not configured");
    if (!spec_.sha256.empty() && std::filesystem::is_regular_file(destination) &&
        std::filesystem::file_size(destination) == spec_.expected_bytes &&
        photara::core::sha256_file(destination,[this] { return cancel_.load(); }) == spec_.sha256) {
        if (spec_.validate) spec_.validate(destination);
        write_receipt(spec_); progress_=1.F; terminal(spec_.ready_message,State::done);return;
    }
    std::error_code error;
    if (!destination.parent_path().empty())
        std::filesystem::create_directories(destination.parent_path(), error);
    if (error) {
        fail("Cannot create the model directory", State::failed);
        return;
    }
    auto partial = destination;
    partial += ".part";
    if (spec_.expected_bytes && std::filesystem::exists(partial)) {
        const auto bytes=std::filesystem::file_size(partial);
        if (bytes == spec_.expected_bytes) {finish(partial);return;}
        if (bytes > spec_.expected_bytes) std::filesystem::remove(partial);
    }

#if !defined(_WIN32)
    int pipes[2]{};
    if (::pipe(pipes) != 0) {
        fail("Cannot capture curl output", State::failed);
        return;
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipes[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, pipes[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipes[0]);
    posix_spawn_file_actions_addclose(&actions, pipes[1]);

    const std::string output_path = partial.string();
    const std::string url = spec_.url;
    const char* argv[] = {
        "curl", "-L", "-f", "--progress-bar", "--proto-redir", "=https", "--retry", "2", "--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "45", "-C", "-", "-o",
        output_path.c_str(), url.c_str(), nullptr};
    pid_t pid = -1;
    const int spawned = ::posix_spawnp(
        &pid, "curl", &actions, nullptr, const_cast<char**>(argv), environ);
    posix_spawn_file_actions_destroy(&actions);
    ::close(pipes[1]);
    if (spawned != 0 || pid <= 0) {
        ::close(pipes[0]);
        fail("curl was not found. Install curl or download the model by hand.",
             State::failed);
        return;
    }
    ::fcntl(pipes[0], F_SETFL, O_NONBLOCK);

    int process_status=0;
    bool reaped=false;
    std::string pending;
    char buffer[256];
    while (!cancel_) {
        const ssize_t read = ::read(pipes[0], buffer, sizeof(buffer));
        if (read > 0) {
            pending.append(buffer, buffer + read);
            for (;;) {
                const auto mark = pending.find_first_of("\r\n");
                if (mark == std::string::npos) break;
                const std::string line = pending.substr(0, mark);
                pending.erase(0, mark + 1);
                const auto percent = line.find('%');
                if (percent == std::string::npos) continue;
                std::size_t start = percent;
                while (start > 0 &&
                       (std::isdigit(
                            static_cast<unsigned char>(line[start - 1])) ||
                        line[start - 1] == '.'))
                    --start;
                if (start >= percent) continue;
                const float value = std::strtof(
                    line.substr(start, percent - start).c_str(), nullptr);
                progress_ = value / 100.F;
                std::lock_guard lock(mutex_);
                status_ =
                    "Downloaded " + std::to_string(static_cast<int>(value)) +
                    "%";
            }
            continue;
        }
        if (read == 0) break;
        if (errno != EAGAIN && errno != EWOULDBLOCK) break;
        if (::waitpid(pid, &process_status, WNOHANG) == pid) {reaped=true;break;}
        ::usleep(50 * 1000);
    }

    if (cancel_) ::kill(pid, SIGTERM);
    if (!reaped) while (::waitpid(pid, &process_status, 0) < 0 && errno == EINTR) {}
    ::close(pipes[0]);
    const int code = WIFEXITED(process_status) ? WEXITSTATUS(process_status) : 1;
    if (cancel_) {
        fail("Download cancelled", State::cancelled);
        return;
    }
    if (code != 0) {
        if (code == 33) std::filesystem::remove(partial, error);
        fail("Download failed (curl exit " + std::to_string(code) + ")",
             State::failed);
        return;
    }
    finish(partial);
    return;
#else
    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;
    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &security, 0)) {
        fail("Cannot capture curl output", State::failed);
        return;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = nullptr;
    startup.hStdOutput = write_pipe;
    startup.hStdError = write_pipe;
    PROCESS_INFORMATION process{};

    const std::wstring command =
        L"\"" + curl_executable().wstring() +
        L"\" -L -f --progress-bar --proto-redir =https --retry 2 --connect-timeout 20 --speed-limit 1024 --speed-time 45 -C - -o \"" + partial.wstring() + L"\" \"" +
        utf8_to_wide(spec_.url) + L"\"";
    std::vector<wchar_t> mutable_command(command.begin(), command.end());
    mutable_command.push_back(L'\0');
    const BOOL started = CreateProcessW(
        nullptr, mutable_command.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    CloseHandle(write_pipe);
    if (!started) {
        CloseHandle(read_pipe);
        fail("curl was not found. Install curl or download the model by hand.",
             State::failed);
        return;
    }

    std::string pending;
    char buffer[256];
    while (!cancel_) {
        DWORD available = 0;
        if (!PeekNamedPipe(read_pipe, nullptr, 0, nullptr, &available, nullptr))
            break;
        if (available == 0) {
            if (WaitForSingleObject(process.hProcess, 50) == WAIT_OBJECT_0)
                break;
            continue;
        }
        DWORD read = 0;
        if (!ReadFile(
                read_pipe, buffer, sizeof(buffer), &read, nullptr) ||
            read == 0)
            break;
        pending.append(buffer, buffer + read);
        for (;;) {
            const auto mark = pending.find_first_of("\r\n");
            if (mark == std::string::npos) break;
            const std::string line = pending.substr(0, mark);
            pending.erase(0, mark + 1);
            const auto percent = line.find('%');
            if (percent == std::string::npos) continue;
            std::size_t start = percent;
            while (start > 0 &&
                   (std::isdigit(static_cast<unsigned char>(line[start - 1])) ||
                    line[start - 1] == '.'))
                --start;
            if (start >= percent) continue;
            const float value = std::strtof(
                line.substr(start, percent - start).c_str(), nullptr);
            progress_ = value / 100.F;
            std::lock_guard lock(mutex_);
            status_ = "Downloaded " + std::to_string(static_cast<int>(value)) + "%";
        }
    }

    if (cancel_) TerminateProcess(process.hProcess, 1);
    WaitForSingleObject(process.hProcess, 15000);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    CloseHandle(read_pipe);

    if (cancel_) {
        fail("Download cancelled", State::cancelled);
        return;
    }
    if (code != 0) {
        if (code == 33) std::filesystem::remove(partial, error);
        fail("Download failed (curl exit " + std::to_string(code) + ")",
             State::failed);
        return;
    }
    finish(partial);
#endif
}


void ModelDownload::terminal(const std::string& message, State state) {
    std::lock_guard lock(mutex_);status_=message;state_=static_cast<int>(state);
}
void ModelDownload::finish(const std::filesystem::path& partial) {
    if (cancel_) {terminal("Download cancelled",State::cancelled);return;}
    {std::lock_guard lock(mutex_);status_="Verifying downloaded model...";}
    const auto bytes=std::filesystem::file_size(partial);
    if ((spec_.expected_bytes && bytes!=spec_.expected_bytes) || bytes<spec_.minimum_bytes) {
        std::filesystem::remove(partial);throw std::runtime_error("Downloaded model has an unexpected file size. Retry the download.");
    }
    if (!spec_.sha256.empty() && photara::core::sha256_file(partial,[this] {return cancel_.load();})!=spec_.sha256) {
        std::filesystem::remove(partial);throw std::runtime_error("Downloaded model failed SHA-256 verification. Retry the download.");
    }
    if (spec_.validate) spec_.validate(partial);
    if (cancel_) {terminal("Download cancelled",State::cancelled);return;}
    if (std::filesystem::exists(spec_.destination)) std::filesystem::remove(spec_.destination);
    std::filesystem::rename(partial,spec_.destination);
    write_receipt(spec_);
    progress_=1.F;terminal(spec_.ready_message,State::done);
}
}
