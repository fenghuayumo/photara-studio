#pragma once

#include "vulkan_backend.hpp"

#include "io/image.hpp"
#include "io/video_frames.hpp"

#include "imgui.h"

#include <atomic>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace editor {

struct App;

// Shared with the background proxy job. Only plain atomics cross the thread
// boundary; `error` is written before the future becomes ready and read only
// after it is ready.
struct VideoPreviewState {
    std::atomic<bool> cancel{false};
    std::atomic<float> progress{-1.F};
    std::atomic<double> duration{0.0};
    std::atomic<double> source_fps{0.0};
    std::atomic<int> width{0};
    std::atomic<int> height{0};
    std::string error;
};

struct VideoWindowFrame {
    gpu::PreviewTexture texture;
    std::future<photara::io::RgbImage> pending;
    std::filesystem::path pending_path;
    int pending_index{-1};
    int loaded_index{-1};
    int failed_index{-1};
    double failed_until{};
};

struct VideoWindowState {
    bool show{};
    bool raise{};
    bool initialized{};
    std::filesystem::path source;
    std::filesystem::path proxy_dir;
    int preview_generation{};

    std::shared_ptr<VideoPreviewState> preview;
    std::future<int> preview_job;
    struct RetiredPreview {
        std::future<int> job;
        std::filesystem::path directory;
    };
    std::vector<RetiredPreview> retired_previews;
    bool preview_done{};
    int preview_frames{};
    double duration{};
    double source_fps{};
    int width{};
    int height{};
    float preview_progress{-1.F};
    std::string preview_error;

    float proxy_fps{10.F};
    double playhead{};
    bool playing{};
    bool dragging_timeline{};
    double last_tick{};
    VideoWindowFrame frame;

    std::vector<photara::io::VideoSegment> segments;
    double range_in{};
    double range_out{};
    std::string segments_error;

    std::shared_ptr<std::atomic<bool>> extract_cancel;
    std::shared_ptr<std::atomic<float>> extract_progress;
    std::future<photara::io::VideoExtractResult> extract_job;
    std::filesystem::path extract_output;
    bool extracting{};
    std::string extract_error;
    std::string extract_note;
};

void open_video_window(App& app);
void draw_video_window(App& app);
void poll_video_window(App& app);
void shutdown_video_window(App& app);

}  // namespace editor
