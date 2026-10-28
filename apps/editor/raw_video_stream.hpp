#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <future>
#include <chrono>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace editor {

// A bounded OS pipe: FFmpeg consumes frames while the next frame is rendered.
// Every pending operation checks cancellation and the child process so a dead
// encoder or a full pipe cannot leave the export worker stuck in a write.
class RawVideoStream {
public:
    explicit RawVideoStream(const std::filesystem::path& unique_path) {
#ifdef _WIN32
        path_ = std::filesystem::path(L"\\\\.\\pipe\\") /
            (unique_path.parent_path().filename().wstring() + L"_" + unique_path.filename().wstring());
        pipe_ = CreateNamedPipeW(path_.c_str(),
            PIPE_ACCESS_OUTBOUND | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
            PIPE_TYPE_BYTE | PIPE_WAIT, 1, 1024 * 1024, 0, 0, nullptr);
        if (pipe_ == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot create video pipe");
        event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!event_) { CloseHandle(pipe_); throw std::runtime_error("Cannot create video pipe event"); }
#else
        path_ = unique_path;
        if (::mkfifo(path_.c_str(), 0600) != 0) throw std::runtime_error("Cannot create video pipe");
        // Holding the read end until close avoids SIGPIPE if FFmpeg exits.
        // Child liveness is checked before every write and while backpressured.
        pipe_ = ::open(path_.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (pipe_ < 0) { ::unlink(path_.c_str()); throw std::runtime_error("Cannot open video pipe"); }
#endif
    }
    RawVideoStream(const RawVideoStream&) = delete;
    RawVideoStream& operator=(const RawVideoStream&) = delete;
    ~RawVideoStream() { close(); }
    const std::filesystem::path& path() const { return path_; }

    // check() throws on cancellation or encoder exit.
    template<class Check> void connect(Check check) {
#ifdef _WIN32
        check();
        ResetEvent(event_);
        OVERLAPPED operation{};
        operation.hEvent = event_;
        if (ConnectNamedPipe(pipe_, &operation)) return;
        const auto error = GetLastError();
        if (error == ERROR_PIPE_CONNECTED) return;
        if (error != ERROR_IO_PENDING) throw std::runtime_error("Cannot connect video pipe");
        complete(operation, check);
#else
        check();
#endif
    }

    template<class Check> void write(const std::vector<std::uint8_t>& bytes, Check check) {
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            check();
            const auto count = std::min<std::size_t>(bytes.size() - offset, 1024 * 1024);
#ifdef _WIN32
            ResetEvent(event_);
            OVERLAPPED operation{};
            operation.hEvent = event_;
            DWORD written = 0;
            if (!WriteFile(pipe_, bytes.data() + offset, static_cast<DWORD>(count), &written, &operation)) {
                if (GetLastError() != ERROR_IO_PENDING) throw std::runtime_error("Cannot write video frame");
                written = complete(operation, check);
            }
            if (!written) throw std::runtime_error("Video pipe closed during frame write");
            offset += written;
#else
            const auto written = ::write(pipe_, bytes.data() + offset, count);
            if (written > 0) { offset += static_cast<std::size_t>(written); continue; }
            if (written < 0 && errno == EINTR) continue;
            if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                pollfd fd{pipe_, POLLOUT, 0};
                ::poll(&fd, 1, 20);
                continue;
            }
            throw std::runtime_error("Cannot write video frame");
#endif
        }
    }

    // Windows discards unread server pipe data on close. Drain on a separate
    // thread so the UI can still cancel. abort() must stop/close the reader,
    // which releases FlushFileBuffers before its worker is joined.
    template<class Check, class Abort> void finish(Check check, Abort abort) {
#ifdef _WIN32
        auto drained = std::async(std::launch::async, [&] { return FlushFileBuffers(pipe_) != FALSE; });
        try {
            while (drained.wait_for(std::chrono::milliseconds(20)) != std::future_status::ready)
                check();
            // A reader that has consumed the requested frame count may close
            // first; encoder exit status is checked by the caller in either case.
            (void)drained.get();
        } catch (...) {
            abort();
            drained.wait();
            throw;
        }
#else
        check();
#endif
        close();
    }

    void close() noexcept {
#ifdef _WIN32
        if (pipe_ != INVALID_HANDLE_VALUE) CloseHandle(pipe_);
        pipe_ = INVALID_HANDLE_VALUE;
        if (event_) CloseHandle(event_);
        event_ = nullptr;
#else
        if (pipe_ >= 0) ::close(pipe_);
        pipe_ = -1;
        if (!path_.empty()) ::unlink(path_.c_str());
#endif
    }

private:
#ifdef _WIN32
    template<class Check> DWORD complete(OVERLAPPED& operation, Check check) {
        try {
            for (;;) {
                const auto status = WaitForSingleObject(event_, 20);
                if (status == WAIT_OBJECT_0) break;
                if (status != WAIT_TIMEOUT) throw std::runtime_error("Cannot wait for video pipe");
                check();
            }
        } catch (...) {
            // The OVERLAPPED structure must remain alive until cancellation finishes.
            CancelIoEx(pipe_, &operation);
            DWORD ignored = 0;
            GetOverlappedResult(pipe_, &operation, &ignored, TRUE);
            throw;
        }
        DWORD transferred = 0;
        if (!GetOverlappedResult(pipe_, &operation, &transferred, FALSE))
            throw std::runtime_error("Video pipe operation failed");
        return transferred;
    }
    HANDLE pipe_{INVALID_HANDLE_VALUE};
    HANDLE event_{};
#else
    int pipe_{-1};
#endif
    std::filesystem::path path_;
};

} // namespace editor
