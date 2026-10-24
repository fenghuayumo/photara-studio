#pragma once
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace editor {
struct ModelDownloadSpec {
    std::string url;
    std::filesystem::path destination;
    std::uint64_t expected_bytes{}; // Zero uses minimum_bytes only (legacy SAM).
    std::uint64_t minimum_bytes{1};
    std::string sha256;
    std::string ready_message{"Model is ready"};
    std::function<void(const std::filesystem::path&)> validate;
    std::function<void()> authorize; // Checked before any network or cache processing.
};
// Receipt is valid only for the exact SHA, file size and last-write timestamp.
bool verified_download_ready(const ModelDownloadSpec& spec);

class ModelDownload {
public:
    enum class State { idle, running, done, failed, cancelled };
    ModelDownload() = default;
    ~ModelDownload();
    ModelDownload(const ModelDownload&) = delete;
    ModelDownload& operator=(const ModelDownload&) = delete;
    void start(ModelDownloadSpec spec);
    void cancel();
    State state() const;
    float progress() const;
    std::string status() const;
private:
    void run();
    void finish(const std::filesystem::path& partial);
    void terminal(const std::string& message, State state);
    ModelDownloadSpec spec_;
    std::thread worker_;
    std::atomic<int> state_{static_cast<int>(State::idle)};
    std::atomic<float> progress_{-1.F};
    std::atomic_bool cancel_{false};
    mutable std::mutex mutex_;
    std::string status_;
};
}
