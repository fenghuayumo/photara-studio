#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

namespace editor {

struct App;

// curl download of the SAM 3 ggml checkpoint into the Photara model cache.
// Starts only after the caller has recorded licence acceptance.
class SamModelDownload {
public:
    enum class State { idle, running, done, failed, cancelled };

    SamModelDownload() = default;
    ~SamModelDownload();
    SamModelDownload(const SamModelDownload&) = delete;
    SamModelDownload& operator=(const SamModelDownload&) = delete;

    void start();
    void cancel();

    [[nodiscard]] State state() const;
    [[nodiscard]] float progress() const;
    [[nodiscard]] std::string status() const;

private:
    void run();

    std::thread worker_;
    std::atomic<int> state_{static_cast<int>(State::idle)};
    std::atomic<float> progress_{-1.F};
    std::atomic_bool cancel_{false};
    mutable std::mutex mutex_;
    std::string status_;
};

// False when a prompt or a checkpoint is still missing. Opens the licence
// dialog and leaves a status message in that case.
bool ensure_sam_ready(App& app);
void draw_sam_license_modal(App& app);

}  // namespace editor
