#include "video_window.hpp"

#include "app.hpp"
#include "file_dialogs.hpp"
#include "i18n.hpp"
#include "theme.hpp"

#include "imgui_internal.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <system_error>
#include <utility>

namespace editor {
namespace {

using i18n::tr;

std::string format_clock(double seconds) {
    if (!std::isfinite(seconds) || seconds < 0.0) seconds = 0.0;
    const auto total = static_cast<long long>(seconds);
    const int millis = static_cast<int>((seconds - total) * 1000.0);
    const int hours = static_cast<int>(total / 3600);
    const int minutes = static_cast<int>((total / 60) % 60);
    const int secs = static_cast<int>(total % 60);
    char text[48];
    if (hours > 0)
        std::snprintf(
            text, sizeof(text), "%d:%02d:%02d.%03d", hours, minutes, secs,
            millis);
    else
        std::snprintf(
            text, sizeof(text), "%02d:%02d.%03d", minutes, secs, millis);
    return text;
}

std::string format_seconds(double seconds) {
    char text[48];
    std::snprintf(text, sizeof(text), "%.2f s", std::max(0.0, seconds));
    return text;
}

std::filesystem::path session_directory(App& app) {
    if (!app.layout.session_dir.empty()) return app.layout.session_dir;
    return std::filesystem::temp_directory_path() / "photara-studio-video";
}

std::filesystem::path preview_directory(App& app, const int generation) {
    return session_directory(app) /
        ("video_preview_" + std::to_string(generation));
}

void sync_segment_text(App& app) {
    app.settings.video_segments =
        photara::io::serialize_video_segments(app.video_window.segments);
}

void load_segments(App& app) {
    auto& window = app.video_window;
    window.segments.clear();
    window.segments_error.clear();
    if (app.settings.video_segments.empty()) return;
    std::string error;
    if (!photara::io::parse_video_segments(
            app.settings.video_segments, window.segments, &error))
        window.segments_error = error;
}

double clamp_time(const double value, const double duration) {
    if (duration <= 0.0) return 0.0;
    return std::clamp(value, 0.0, duration);
}

int current_preview_index(const VideoWindowState& window) {
    const double last_time = std::max(0.0, window.duration - 0.5 / window.proxy_fps);
    return static_cast<int>(std::min(window.playhead, last_time) * window.proxy_fps) + 1;
}

void cancel_preview_job(App& app) {
    auto& window = app.video_window;
    if (window.preview_job.valid()) {
        if (window.preview) window.preview->cancel.store(true);
        window.retired_previews.push_back({std::move(window.preview_job), window.proxy_dir});
    } else if (!window.proxy_dir.empty()) {
        std::promise<int> completed;
        completed.set_value(0);
        window.retired_previews.push_back({completed.get_future(), window.proxy_dir});
    }
    window.preview_job = {};
    window.preview.reset();
    window.proxy_dir.clear();
}

void start_preview_job(App& app, const std::filesystem::path& source,
                       double start = 0.0, bool reset = true, bool append = false) {
    auto& window = app.video_window;
    if (!append) cancel_preview_job(app);
    window.source = source;
    window.preview_done = false;
    window.preview_frames = 0;
    window.preview_progress = -1.F;
    window.preview_error.clear();
    if (reset) {
        window.duration = 0.0;
        window.source_fps = 0.0;
        window.width = 0;
        window.height = 0;
        window.playhead = 0.0;
        window.playing = false;
        window.last_tick = 0.0;
        window.range_in = 0.0;
        window.range_out = 0.0;
        load_segments(app);
    }
    // Only decode a short interval around the requested source time.
    if (!reset) start = std::min(start, std::max(0.0, window.duration - 1.0 / window.proxy_fps));
    window.preview_start = std::floor(start * window.proxy_fps) / window.proxy_fps;
    std::error_code error;
    auto directory = window.proxy_dir;
    if (!append) {
        window.cached_start = window.preview_start;
        window.cached_end = window.preview_start;
        window.decoded_frames.clear();
        if (window.frame.pending.valid()) {
            try { (void)window.frame.pending.get(); } catch (...) {}
        }
        window.frame.pending = {};
        if (reset) window.frame.texture.reset();
        window.frame.loaded_index = -1;
        window.frame.failed_index = -1;
        directory = preview_directory(app, ++window.preview_generation);
        std::filesystem::remove_all(directory, error);
        std::filesystem::create_directories(directory, error);
        if (error) {
            window.preview_error = "Cannot create the video preview folder";
            return;
        }
        window.proxy_dir = directory;
    }
    auto state = std::make_shared<VideoPreviewState>();
    window.preview = state;
    const float proxy_fps = window.proxy_fps;
    const double preview_start = window.preview_start;
    const double preview_length = window.preview_length;
    const photara::io::VideoProbe known_probe{
        window.duration, window.source_fps, window.width, window.height};
    window.preview_job = std::async(
        std::launch::async,
        [source, directory, state, proxy_fps, preview_start, preview_length, known_probe, append]() -> int {
            try {
                auto probe = known_probe;
                if (probe.width <= 0 || probe.duration_seconds <= 0.0)
                    photara::io::probe_video({}, source, probe);
                state->duration.store(probe.duration_seconds);
                state->source_fps.store(probe.fps);
                state->width.store(probe.width);
                state->height.store(probe.height);
                photara::io::VideoPreviewOptions options;
                options.video = source;
                options.output_dir = directory;
                options.fps = proxy_fps;
                options.max_width = 960;
                options.max_height = 540;
                options.start_seconds = preview_start;
                options.duration_seconds = preview_length;
                options.first_frame = static_cast<int>(std::llround(preview_start * proxy_fps)) + 1;
                options.source_probe = probe;
                options.keep_existing = append;
                options.cancel = &state->cancel;
                options.progress = [state](const float value) {
                    state->progress.store(value);
                };
                return photara::io::generate_video_preview(options);
            } catch (const std::exception& error) {
                state->error = error.what();
                throw;
            }
        });
}

void harvest_retired_previews(App& app) {
    auto& retired = app.video_window.retired_previews;
    for (auto it = retired.begin(); it != retired.end();) {
        if (it->job.wait_for(std::chrono::seconds(0)) !=
            std::future_status::ready) {
            ++it;
            continue;
        }
        try {
            (void)it->job.get();
        } catch (...) {
        }
        std::error_code error;
        std::filesystem::remove_all(it->directory, error);
        it = retired.erase(it);
    }
}

void poll_preview_job(App& app) {
    auto& window = app.video_window;
    if (!window.preview_job.valid() || !window.preview) return;
    const bool awaiting_metadata = window.duration <= 0.0;
    if (awaiting_metadata)
        window.duration = window.preview->duration.load();
    if (window.source_fps <= 0.0)
        window.source_fps = window.preview->source_fps.load();
    if (window.width <= 0) window.width = window.preview->width.load();
    if (window.height <= 0) window.height = window.preview->height.load();
    window.preview_progress = window.preview->progress.load();
    if (awaiting_metadata && window.duration > 0.0 && window.range_out <= window.range_in)
        window.range_out = window.duration;
    if (window.preview_job.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
        return;

    const bool cancelled = window.preview->cancel.load();
    try {
        window.preview_frames = window.preview_job.get();
        window.preview_done = true;
        window.preview_progress = 1.F;
        window.cached_end = std::min(window.duration,
            window.preview_start + window.preview_frames / window.proxy_fps);
    } catch (const std::exception& error) {
        window.preview_frames = 0;
        if (!cancelled) window.preview_error = error.what();
    }
    window.preview_job = {};
    window.preview.reset();
    if (window.duration > 0.0 &&
        (window.range_out <= window.range_in ||
         window.range_out > window.duration))
        window.range_out = window.duration;
}

std::filesystem::path preview_frame_path(
    const VideoWindowState& window, const int index) {
    char name[32];
    std::snprintf(name, sizeof(name), "p_%06d.jpg", index);
    return window.proxy_dir / name;
}

void harvest_frame(VideoWindowFrame& frame) {
    if (!frame.pending.valid()) return;
    if (frame.pending.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
        return;
    const int index = frame.pending_index;
    try {
        photara::io::RgbImage image = frame.pending.get();
        frame.texture.upload(image);
        frame.loaded_index = index;
        frame.failed_index = -1;
    } catch (...) {
        frame.failed_index = index;
        frame.failed_until = ImGui::GetTime() + 0.5;
    }
    frame.pending = {};
    frame.pending_index = -1;
}

// Decode ahead in bounded batches. The render thread only uploads the newest
// frame due on the playback clock; it never waits for a disk read or JPEG decode.
void request_frame(VideoWindowState& window, const int index) {
    if (index <= 0) return;
    if (window.decode_batch.valid() &&
        window.decode_batch.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        try {
            auto batch = window.decode_batch.get();
            if (window.decode_directory == window.proxy_dir) {
                for (auto& frame : batch)
                    window.decoded_frames.push_back(std::move(frame));
            }
        } catch (...) {}
    }
    auto& frames = window.decoded_frames;
    if (window.frame.loaded_index > index) {
        window.frame.loaded_index = -1;
        frames.clear();
    }
    int selected = -1;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        if (frames[i].index <= index && frames[i].index > window.frame.loaded_index &&
            (selected < 0 || frames[i].index > frames[selected].index))
            selected = static_cast<int>(i);
    }
    if (selected >= 0) {
        window.frame.texture.upload(frames[selected].image);
        window.frame.loaded_index = frames[selected].index;
    }
    frames.erase(std::remove_if(frames.begin(), frames.end(), [&](const auto& frame) {
        return frame.index <= index;
    }), frames.end());
    if (window.decode_batch.valid()) return;
    const bool missing_current = window.frame.loaded_index != index;
    if (!missing_current && frames.size() >= 12) return;
    if (missing_current) frames.clear();
    int first = std::max(index, window.frame.loaded_index + 1);
    if (!missing_current)
        for (const auto& frame : frames) first = std::max(first, frame.index + 1);
    std::vector<std::pair<int, std::filesystem::path>> paths;
    // Retain roughly half a second in RAM; ffmpeg writes the disk cache ahead.
    for (int next = first; next < first + 12; ++next) {
        const auto path = preview_frame_path(window, next);
        std::error_code error;
        if (!std::filesystem::is_regular_file(path, error)) break;
        paths.emplace_back(next, path);
    }
    if (paths.empty()) return;
    window.decode_directory = window.proxy_dir;
    window.decode_batch = std::async(std::launch::async, [paths = std::move(paths)] {
        std::vector<VideoDecodedFrame> result;
        for (const auto& [number, path] : paths) {
            try {
                const auto size = photara::io::load_image_size(path);
                result.push_back({number, photara::io::load_rgb_with_minimum_size(
                    path, size.width, size.height)});
            } catch (...) { break; } // The last file can still be being written.
        }
        return result;
    });
}

void start_extract_job(App& app) {
    auto& window = app.video_window;
    if (window.extracting || app.job.running()) return;
    const auto source =
        path_from_utf8_field(app.settings.images_dir.data());
    if (!photara::io::is_video_path(source)) {
        set_message(app, "Select a video first", theme::warning);
        return;
    }
    if (!photara::io::ffmpeg_available()) {
        set_message(
            app, "ffmpeg was not found. Install ffmpeg and add it to PATH.",
            theme::danger);
        return;
    }
    auto output = reconstruction_images_dir(app.settings);
    if (output.empty()) {
        set_message(app, "Choose an output folder for extracted frames",
                    theme::warning);
        return;
    }
    sync_segment_text(app);
    photara::io::VideoExtractOptions options;
    options.video = source;
    options.output_dir = output;
    options.fps = app.settings.video_fps;
    options.sharp_window = app.settings.video_sharp_window;
    options.max_frames = app.settings.video_max_frames;
    options.quality = app.settings.video_quality;
    options.scale = app.settings.video_scale;
    options.rotate = app.settings.video_rotate;
    options.segments = window.segments;
    options.resume = true;
    window.extract_cancel = std::make_shared<std::atomic<bool>>(false);
    window.extract_progress = std::make_shared<std::atomic<float>>(-1.F);
    const auto cancel = window.extract_cancel;
    const auto progress = window.extract_progress;
    options.cancel = cancel.get();
    options.progress = [progress](const float value) {
        progress->store(value);
    };
    window.extract_error.clear();
    window.extract_note.clear();
    window.extract_output = output;
    window.extracting = true;
    window.extract_job = std::async(
        std::launch::async,
        [options]() { return photara::io::extract_video_frames(options); });
}

void poll_extract_job(App& app) {
    auto& window = app.video_window;
    if (!window.extracting || !window.extract_job.valid()) return;
    if (window.extract_job.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
        return;
    const bool cancelled =
        window.extract_cancel && window.extract_cancel->load();
    try {
        const auto result = window.extract_job.get();
        store_path_field(app.settings.video_frames_dir, result.image_dir);
        app.settings.alignment_capture_type = 1;
        window.extract_note = result.reused
            ? "Reused " + std::to_string(result.frames_written) +
                  " existing frames in " + path_to_utf8(result.image_dir)
            : "Extracted " + std::to_string(result.frames_written) +
                  " frames to " + path_to_utf8(result.image_dir);
        set_message(app, window.extract_note, theme::success);
        if (!result.reused) {
            // The frames changed under any existing masks. Drop the marker so
            // alignment regenerates them instead of reusing stale edits.
            std::error_code marker_error;
            const auto masks = masks_directory(app.settings);
            const bool had_marker = std::filesystem::is_regular_file(
                masks / k_mask_ready_marker, marker_error);
            std::filesystem::remove(
                masks / k_mask_ready_marker, marker_error);
            if (had_marker)
                window.extract_note +=
                    ". Existing masks are now stale; regenerate them in the "
                    "Mask Editor.";
        }
        if (app.has_sparse)
            window.extract_note +=
                ". Re-run Align Photos to use the new frames.";
    } catch (const std::exception& error) {
        if (cancelled) {
            window.extract_error.clear();
            set_message(app, "Frame extraction cancelled", theme::warning);
        } else {
            window.extract_error = error.what();
            set_message(app, window.extract_error, theme::danger);
        }
    }
    window.extract_job = {};
    window.extract_cancel.reset();
    window.extract_progress.reset();
    window.extracting = false;
}

void draw_preview_canvas(App& app, float height) {
    auto& window = app.video_window;
    const ImVec2 size{0.F, height};
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::viewport_bg);
    ImGui::BeginChild(
        "##video_preview_canvas", size, ImGuiChildFlags_None,
        ImGuiWindowFlags_NoScrollbar);

    const ImVec2 area_min = ImGui::GetCursorScreenPos();
    const ImVec2 area_size = ImGui::GetContentRegionAvail();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(
        area_min, {area_min.x + area_size.x, area_min.y + area_size.y},
        theme::u32(theme::viewport_bg));

    int image_width = window.frame.texture.width > 0
        ? static_cast<int>(window.frame.texture.width)
        : window.width;
    int image_height = window.frame.texture.height > 0
        ? static_cast<int>(window.frame.texture.height)
        : window.height;
    if (image_width <= 0 || image_height <= 0) {
        image_width = 16;
        image_height = 9;
    }
    const float scale = std::min(
        area_size.x / image_width, area_size.y / image_height);
    const ImVec2 image_size{
        image_width * scale, image_height * scale};
    const ImVec2 image_min{
        area_min.x + (area_size.x - image_size.x) * 0.5F,
        area_min.y + (area_size.y - image_size.y) * 0.5F};
    const ImVec2 image_max{
        image_min.x + image_size.x, image_min.y + image_size.y};

    if (!window.follow_alignment) {
        const int desired = current_preview_index(window);
        const double chunk_end = window.preview_start + window.preview_length;
        if (window.duration > 0.0 &&
            (window.playhead < window.cached_start ||
             window.playhead >= std::max(window.cached_end, chunk_end)) &&
            (!window.dragging_timeline || ImGui::IsMouseReleased(ImGuiMouseButton_Left))) {
            start_preview_job(app, window.source, window.playhead, false);
        }
        if (window.playing && window.preview_done && !window.preview_job.valid() &&
            window.preview_error.empty() && window.cached_end < window.duration - 0.5 / window.proxy_fps &&
            window.playhead >= window.cached_end - 4.0) {
            start_preview_job(app, window.source, window.cached_end, false, true);
        }
        request_frame(window, desired);
    }

    if (window.frame.texture.descriptor) {
        draw->AddImage(
            reinterpret_cast<ImTextureID>(window.frame.texture.descriptor),
            image_min, image_max);
        draw->AddRect(
            image_min, image_max, theme::u32(theme::border));
    } else {
        const char* label = window.preview_error.empty()
            ? tr("Preparing video preview...")
            : tr("Video preview is unavailable");
        const ImVec2 text_size = ImGui::CalcTextSize(label);
        draw->AddText(
            {area_min.x + (area_size.x - text_size.x) * 0.5F,
             area_min.y + (area_size.y - text_size.y) * 0.5F},
            theme::u32(theme::text_muted), label);
    }

    if (!window.follow_alignment && window.preview_job.valid() &&
        window.frame.loaded_index != current_preview_index(window)) {
        draw->AddText({image_min.x + 12.F, image_max.y - 24.F},
            theme::u32(theme::text_bright), tr("Loading preview..."));
    }

    ImGui::InvisibleButton(
        "##video_canvas_click", area_size,
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    if (!window.follow_alignment && ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        window.playing = !window.playing;
        window.last_tick = ImGui::GetTime();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void draw_timeline(App& app) {
    auto& window = app.video_window;
    const double duration = window.duration;
    const auto origin = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    constexpr float height = 40.F;
    ImGui::InvisibleButton("##video_timeline", {width, height});
    window.dragging_timeline = ImGui::IsItemActive();
    if (window.dragging_timeline && duration > 0.0) {
        const float x = ImGui::GetIO().MousePos.x - origin.x;
        window.playhead = clamp_time(x / std::max(1.F, width) * duration, duration);
        window.playing = false;
    }
    auto* draw = ImGui::GetWindowDrawList();
    const float y = origin.y + 29.F;
    draw->AddRectFilled({origin.x, y}, {origin.x + width, y + 5.F},
        theme::u32(theme::surface_3), 2.5F);
    if (duration <= 0.0) return;
    ImGui::PushFont(theme::small_font());
    for (int tick = 0; tick <= 4; ++tick) {
        const auto label = format_clock(duration * tick / 4.0);
        const auto size = ImGui::CalcTextSize(label.c_str());
        const float x = std::clamp(width * tick / 4.F - size.x * 0.5F, 0.F, width - size.x);
        draw->AddText({origin.x + x, origin.y + 2.F}, theme::u32(theme::text_faint), label.c_str());
    }
    ImGui::PopFont();
    const double buffered_end = std::max(window.cached_end, window.preview_start +
        window.preview_length * std::clamp(window.preview_progress, 0.F, 1.F));
    const auto xpos = [&](double time) {
        return origin.x + static_cast<float>(clamp_time(time, duration) / duration) * width;
    };
    draw->AddRectFilled({xpos(window.cached_start), y},
        {xpos(buffered_end), y + 5.F}, theme::u32(theme::text_faint), 2.F);
    if (window.range_out > window.range_in) {
        draw->AddRectFilled({xpos(window.range_in), y - 3.F}, {xpos(window.range_out), y + 8.F},
            theme::u32(theme::accent, 0.18F), 3.F);
        for (double time : {window.range_in, window.range_out})
            draw->AddLine({xpos(time), y - 4.F}, {xpos(time), y + 9.F}, theme::u32(theme::accent), 2.F);
    }
    for (const auto& segment : window.segments) {
        const double end = segment.end_seconds > segment.start_seconds ? segment.end_seconds : duration;
        draw->AddRectFilled({xpos(segment.start_seconds), y}, {xpos(end), y + 5.F},
            theme::u32(theme::accent), 2.F);
    }
    draw->AddCircleFilled({xpos(window.playhead), y + 2.5F}, 5.F, theme::u32(theme::text_bright));
    if (ImGui::IsItemHovered()) {
        const double hovered = (ImGui::GetIO().MousePos.x - origin.x) / std::max(1.F, width) * duration;
        ImGui::SetTooltip("%s", format_clock(hovered).c_str());
    }
}

void draw_transport(App& app) {
    auto& window = app.video_window;
    const bool ready = window.duration > 0.0;
    ImGui::BeginDisabled(!ready);
    if (window.playing) {
        if (theme::toolbar_button(tr("Pause"), {76.F, 32.F}, ready, true))
            window.playing = false;
    } else if (theme::primary_button(tr("Play"), {76.F, 32.F}, ready)) {
        if (window.playhead >= window.duration) window.playhead = 0.0;
        window.playing = true;
        window.last_tick = ImGui::GetTime();
    }
    ImGui::SameLine();
    const auto previous_origin = ImGui::GetCursorScreenPos();
    if (theme::toolbar_button("##previous_frame", {32.F, 32.F}, ready)) {
        window.playing = false;
        window.playhead = clamp_time(
            window.playhead - 1.0 / window.proxy_fps, window.duration);
    }
    ImGui::SameLine();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Previous frame"));
    auto* draw = ImGui::GetWindowDrawList();
    const auto icon_colour = theme::u32(theme::text_muted);
    draw->AddTriangleFilled({previous_origin.x + 20.F, previous_origin.y + 10.F},
        {previous_origin.x + 12.F, previous_origin.y + 16.F},
        {previous_origin.x + 20.F, previous_origin.y + 22.F}, icon_colour);
    draw->AddLine({previous_origin.x + 10.F, previous_origin.y + 10.F},
        {previous_origin.x + 10.F, previous_origin.y + 22.F}, icon_colour, 1.5F);
    const auto next_origin = ImGui::GetCursorScreenPos();
    if (theme::toolbar_button("##next_frame", {32.F, 32.F}, ready)) {
        window.playing = false;
        window.playhead = clamp_time(
            window.playhead + 1.0 / window.proxy_fps, window.duration);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Next frame"));
    draw->AddTriangleFilled({next_origin.x + 12.F, next_origin.y + 10.F},
        {next_origin.x + 20.F, next_origin.y + 16.F},
        {next_origin.x + 12.F, next_origin.y + 22.F}, icon_colour);
    draw->AddLine({next_origin.x + 22.F, next_origin.y + 10.F},
        {next_origin.x + 22.F, next_origin.y + 22.F}, icon_colour, 1.5F);
    ImGui::SameLine();
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 8.F);
    ImGui::Text(
        "%s / %s", format_clock(window.playhead).c_str(),
        format_clock(window.duration).c_str());
    ImGui::EndDisabled();
    if (window.playing && ready) {
        const double now = ImGui::GetTime();
        if (window.last_tick <= 0.0) window.last_tick = now;
        const double delta = std::min(0.25, now - window.last_tick);
        window.last_tick = now;
        const int desired = current_preview_index(window);
        // Advance on elapsed time, skipping stale frames when rendering is late.
        // Only stop the clock when the producer has actually run out of frames.
        int available = window.frame.loaded_index;
        for (const auto& frame : window.decoded_frames) available = std::max(available, frame.index);
        if (available >= desired)
            window.playhead = std::min(window.playhead + delta,
                std::min(window.duration, available / static_cast<double>(window.proxy_fps)));
        if (window.playhead >= window.duration - 0.5 / window.proxy_fps) {
            window.playhead = window.duration;
            window.playing = false;
        }
    } else {
        window.last_tick = ImGui::GetTime();
    }
}

void draw_segments(App& app) {
    auto& window = app.video_window;
    theme::caption(tr("Capture segments"));
    ImGui::Spacing();
    ImGui::TextWrapped("%s", tr("Use the playhead to mark the start and end."));
    ImGui::Spacing();
    ImGui::BeginDisabled(window.duration <= 0.0);
    float range_in = static_cast<float>(window.range_in);
    float range_out = static_cast<float>(window.range_out);
    if (ImGui::BeginTable("##range_fields", 2, ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("%s", tr("Start"));
        ImGui::SetNextItemWidth(-1.F);
        if (ImGui::DragFloat("##vw_range_in", &range_in, 0.05F, 0.F,
                static_cast<float>(window.duration), "%.2f s"))
            window.range_in = clamp_time(range_in, window.duration);
        if (ImGui::Button(tr("Set In"), {-1.F, 28.F})) window.range_in = window.playhead;
        ImGui::TableSetColumnIndex(1);
        ImGui::TextDisabled("%s", tr("End"));
        ImGui::SetNextItemWidth(-1.F);
        if (ImGui::DragFloat("##vw_range_out", &range_out, 0.05F, 0.F,
                static_cast<float>(window.duration), "%.2f s"))
            window.range_out = clamp_time(range_out, window.duration);
        if (ImGui::Button(tr("Set Out"), {-1.F, 28.F})) window.range_out = window.playhead;
        ImGui::EndTable();
    }
    ImGui::Spacing();
    const bool valid_range = window.range_out > window.range_in;
    if (theme::primary_button(tr("Add Segment"), {-1.F, 32.F}, valid_range)) {
        window.segments.push_back({window.range_in, window.range_out});
        window.segments = photara::io::normalize_video_segments(window.segments);
        sync_segment_text(app);
    }
    ImGui::EndDisabled();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    if (window.segments.empty()) {
        ImGui::TextColored(
            theme::text_muted, "%s", tr("Whole video will be extracted."));
        return;
    }
    int remove = -1;
    for (std::size_t i = 0; i < window.segments.size(); ++i) {
        const auto& segment = window.segments[i];
        const double end = segment.end_seconds > segment.start_seconds
            ? segment.end_seconds : window.duration;
        ImGui::PushID(static_cast<int>(i));
        ImGui::TextColored(theme::accent, "%02d", static_cast<int>(i + 1));
        ImGui::SameLine();
        ImGui::Text("%s - %s", format_clock(segment.start_seconds).c_str(), format_clock(end).c_str());
        ImGui::TextDisabled("%s", format_seconds(end - segment.start_seconds).c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton(tr("Go"))) {
            window.playhead = segment.start_seconds;
            window.playing = false;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton(tr("Remove"))) remove = static_cast<int>(i);
        ImGui::Spacing();
        ImGui::PopID();
    }
    if (remove >= 0) {
        window.segments.erase(window.segments.begin() + remove);
        sync_segment_text(app);
    }
    if (ImGui::Button(tr("Clear Segments"))) {
        window.segments.clear();
        sync_segment_text(app);
    }
}

void draw_extract_settings(App& app) {
    auto& window = app.video_window;
    auto& settings = app.settings;
    theme::caption(tr("Frame extraction"));
    if (ImGui::BeginTable(
            "##vw_extract_controls", 2,
            ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("%s", tr("Sampling rate"));
        ImGui::SetNextItemWidth(-1.F);
        ImGui::DragFloat(
            "##vw_fps", &settings.video_fps, 0.05F, 0.05F, 60.F, "%.2f fps");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextDisabled("%s", tr("Frame limit"));
        ImGui::SetNextItemWidth(-1.F);
        ImGui::DragInt(
            "##vw_max_frames", &settings.video_max_frames, 1, 0, 100000,
            tr("max %d"));
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("%s", tr("Sharpness"));
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderInt(
            "##vw_sharp", &settings.video_sharp_window, 1, 10,
            tr("sharp %d"));
        ImGui::TableSetColumnIndex(1);
        ImGui::TextDisabled("%s", tr("Image format / quality"));
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderInt(
            "##vw_quality", &settings.video_quality, -1, 100,
            settings.video_quality < 0 ? tr("PNG") : tr("JPEG %d"));
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("%s", tr("Output scale"));
        ImGui::SetNextItemWidth(-1.F);
        ImGui::DragFloat(
            "##vw_scale", &settings.video_scale, 0.01F, 0.05F, 4.F, "%.2fx");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextDisabled("%s", tr("Rotation"));
        ImGui::SetNextItemWidth(-1.F);
        const char* rotations[] = {"0°", "90°", "180°", "270°"};
        int rotation = settings.video_rotate / 90;
        if (ImGui::Combo("##vw_rotate", &rotation, rotations, 4))
            settings.video_rotate = rotation * 90;
        ImGui::EndTable();
    }

    ImGui::Spacing();
    theme::caption(tr("Output folder"));
    const auto resolved = reconstruction_images_dir(settings);
    ImGui::SetNextItemWidth(-54.F);
    ImGui::InputText(
        "##vw_frames_dir", settings.video_frames_dir.data(),
        settings.video_frames_dir.size());
    ImGui::SameLine();
    if (ImGui::Button("...##vw_pick_frames")) {
        pick_folder(
            L"Select the folder for extracted frames",
            settings.video_frames_dir);
    }
    ImGui::TextDisabled("%s", path_to_utf8(resolved).c_str());

    double seconds = window.duration;
    if (!window.segments.empty()) {
        seconds = 0.0;
        for (const auto& segment : window.segments) {
            const double end = segment.end_seconds > segment.start_seconds
                ? segment.end_seconds
                : window.duration;
            seconds += std::max(0.0, end - segment.start_seconds);
        }
    }
    int estimate = static_cast<int>(
        std::llround(seconds * std::max(0.01F, settings.video_fps)));
    if (settings.video_max_frames > 0)
        estimate = std::min(estimate, settings.video_max_frames);
    ImGui::TextColored(theme::accent, "%s %d", tr("Estimated frames:"), estimate);

    ImGui::Spacing();
    const bool can_extract = !window.extracting && !app.job.running() &&
        photara::io::is_video_path(
            path_from_utf8_field(settings.images_dir.data()));
    ImGui::BeginDisabled(!can_extract);
    if (theme::primary_button(tr("Extract Frames"), {150.F, 30.F}))
        start_extract_job(app);
    ImGui::EndDisabled();
    if (window.extracting) {
        ImGui::SameLine();
        if (ImGui::Button(tr("Cancel"), {80.F, 30.F}) &&
            window.extract_cancel)
            window.extract_cancel->store(true);
        const float progress = window.extract_progress
            ? window.extract_progress->load()
            : -1.F;
        theme::progress_track({-1.F, 6.F}, progress, theme::accent);
    }
    if (app.job.running())
        ImGui::TextColored(
            theme::warning, "%s",
            tr("Stop the running job before extracting frames."));
    if (!window.extract_error.empty())
        ImGui::TextColored(
            theme::danger, "%s", window.extract_error.c_str());
    else if (!window.extract_note.empty())
        ImGui::TextWrapped("%s", window.extract_note.c_str());
    if (!window.preview_error.empty())
        ImGui::TextColored(
            theme::warning, "%s", window.preview_error.c_str());
}

void draw_source_header(const App& app, const std::filesystem::path& source) {
    const auto& window = app.video_window;
    const float width = ImGui::GetContentRegionAvail().x;
    const auto filename = path_to_utf8(source.filename());
    const auto origin = ImGui::GetCursorScreenPos();
    ImGui::PushClipRect(origin, {origin.x + width - 200.F, origin.y + 24.F}, true);
    ImGui::TextColored(theme::text_bright, "%s", filename.c_str());
    ImGui::PopClipRect();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", path_to_utf8(source).c_str());
    if (window.width > 0 && window.height > 0) {
        char details[96];
        if (window.source_fps > 0.0)
            std::snprintf(details, sizeof(details), "%d x %d  /  %.2f fps",
                window.width, window.height, window.source_fps);
        else
            std::snprintf(details, sizeof(details), "%d x %d", window.width, window.height);
        ImGui::SameLine(std::max(0.F, width - ImGui::CalcTextSize(details).x));
        ImGui::TextDisabled("%s", details);
    }
    ImGui::PushFont(theme::small_font());
    ImGui::TextDisabled("%s", tr("Preview and select the parts to reconstruct."));
    ImGui::PopFont();
    ImGui::Spacing();
}

}  // namespace

void open_video_window(App& app) {
    app.video_window.show = true;
    app.video_window.raise = true;
}

void poll_video_window(App& app) {
    harvest_retired_previews(app);
    auto& window = app.video_window;
    const bool aligning = app.job.running() && app.active_job == JobKind::align;
    if (window.follow_alignment) {
        cancel_preview_job(app);
        harvest_frame(window.frame);
        const auto& live = app.monitor.video_frame();
        if (window.show && live.kind == "video" && live.revision != window.alignment_revision &&
            !window.frame.pending.valid()) {
            const auto path = path_from_utf8_field(live.image.c_str());
            std::error_code error;
            if (std::filesystem::is_regular_file(path, error)) {
                window.frame.pending_index = static_cast<int>(live.revision);
                window.frame.pending_path = path;
                window.frame.pending = std::async(std::launch::async, [path] {
                    const auto size = photara::io::load_image_size(path);
                    return photara::io::load_rgb_with_minimum_size(path, size.width, size.height);
                });
                window.alignment_revision = live.revision;
            }
        }
        window.preview_done = false;
        window.preview_frames = 0;
        window.preview_progress = -1.F;
        window.playing = false;
        if (!aligning || app.monitor.video_frames_ready() ||
            app.monitor.stage() != Stage::preparing) {
            window.show = false;
            window.raise = false;
            window.follow_alignment = false;
            window.source.clear();
        }
    }
    poll_preview_job(app);
    poll_extract_job(app);
}

void draw_video_window(App& app) {
    auto& window = app.video_window;
    if (!window.show) return;

    const auto source =
        path_from_utf8_field(app.settings.images_dir.data());
    if (photara::io::is_video_path(source) && source != window.source &&
        !window.extracting && !window.follow_alignment)
        start_preview_job(app, source);

    if (window.raise) {
        ImGui::SetNextWindowFocus();
        window.raise = false;
    }
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        viewport->GetCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5F, 0.5F));
    ImGui::SetNextWindowSize({1180.F, 760.F}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(800.F, 640.F), ImVec2(FLT_MAX, FLT_MAX));
    if (!ImGui::Begin(
            i18n::id("Video Preview", "###VideoWindow"),
            &window.show, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    if (!photara::io::is_video_path(source)) {
        ImGui::Spacing();
        ImGui::TextWrapped("%s", tr(
            "Select a video as the capture source, then preview it here "
            "before extracting frames."));
        if (theme::primary_button(tr("Select Video..."), {160.F, 30.F}))
            select_video_file(app);
        ImGui::End();
        return;
    }

    draw_source_header(app, source);
    if (window.follow_alignment) {
        const auto& live = app.monitor.video_frame();
        ImGui::TextUnformatted(tr(app.monitor.stage() == Stage::preparing
            ? "Alignment is extracting frames automatically." : "Frame extraction complete."));
        {
            ImGui::Text("%llu / %llu", static_cast<unsigned long long>(live.completed),
                static_cast<unsigned long long>(live.total));
            theme::progress_track({-1.F, 6.F}, live.total > 0
                ? static_cast<float>(live.completed) / live.total : -1.F, theme::accent);
        }
    }
    ImGui::Spacing();
    if (!window.preview_error.empty())
        ImGui::TextWrapped("%s", window.preview_error.c_str());
    if (!window.segments_error.empty())
        ImGui::TextColored(
            theme::danger, "%s", window.segments_error.c_str());

    const bool wide = ImGui::GetContentRegionAvail().x >= 1000.F;
    const float footer_height = 44.F;
    const float content_height = std::max(240.F, ImGui::GetContentRegionAvail().y - footer_height);
    const float sidebar_width = 304.F;
    const float viewer_width = wide
        ? ImGui::GetContentRegionAvail().x - sidebar_width - 16.F : 0.F;
    const float viewer_height = wide ? content_height : std::max(200.F, content_height - 214.F);
    ImGui::BeginChild("##vw_viewer", {viewer_width, viewer_height}, ImGuiChildFlags_None,
        ImGuiWindowFlags_NoScrollbar);
    const float controls_height = window.follow_alignment ? 0.F : 92.F;
    draw_preview_canvas(app, std::max(100.F, ImGui::GetContentRegionAvail().y - controls_height));
    if (!window.follow_alignment) {
        ImGui::Spacing();
        draw_timeline(app);
        ImGui::Spacing();
        draw_transport(app);
    }
    ImGui::EndChild();
    if (wide) ImGui::SameLine(0.F, 16.F);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::surface_1);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 10.F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {14.F, 14.F});
    ImGui::BeginChild("##vw_options", {0.F, wide ? content_height : 206.F},
        ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::BeginDisabled(app.job.running());
    draw_segments(app);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    if (ImGui::CollapsingHeader(tr("Extraction Settings"))) draw_extract_settings(app);
    ImGui::EndDisabled();
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("%s", tr("Selected segments are used when aligning photos."));
    ImGui::SameLine(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowContentRegionMax().x - 90.F));
    if (theme::toolbar_button(tr("Done"), {90.F, 28.F})) window.show = false;

    ImGui::End();
}

void shutdown_video_window(App& app) {
    auto& window = app.video_window;
    if (window.decode_batch.valid()) {
        try { (void)window.decode_batch.get(); } catch (...) {}
    }
    window.decoded_frames.clear();
    if (window.preview_job.valid()) {
        if (window.preview) window.preview->cancel.store(true);
        try {
            (void)window.preview_job.get();
        } catch (...) {
        }
        window.preview_job = {};
        window.preview.reset();
    }
    if (window.extract_job.valid()) {
        if (window.extract_cancel) window.extract_cancel->store(true);
        try {
            (void)window.extract_job.get();
        } catch (...) {
        }
        window.extract_job = {};
        window.extracting = false;
    }
    for (auto& retired : window.retired_previews) {
        try {
            (void)retired.job.get();
        } catch (...) {
        }
        std::error_code error;
        std::filesystem::remove_all(retired.directory, error);
    }
    window.retired_previews.clear();
    std::error_code error;
    if (!window.proxy_dir.empty())
        std::filesystem::remove_all(window.proxy_dir, error);
    window.proxy_dir.clear();
    if (window.frame.pending.valid()) {
        try {
            (void)window.frame.pending.get();
        } catch (...) {
        }
    }
    window.frame.texture.reset();
}

}  // namespace editor
