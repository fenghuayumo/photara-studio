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

void cancel_preview_job(App& app) {
    auto& window = app.video_window;
    if (!window.preview_job.valid()) return;
    if (window.preview) window.preview->cancel.store(true);
    window.retired_previews.push_back(
        {std::move(window.preview_job), window.proxy_dir});
    window.preview_job = {};
    window.preview.reset();
    window.proxy_dir.clear();
}

void start_preview_job(App& app, const std::filesystem::path& source) {
    auto& window = app.video_window;
    cancel_preview_job(app);
    window.source = source;
    window.preview_done = false;
    window.preview_frames = 0;
    window.preview_progress = -1.F;
    window.preview_error.clear();
    window.duration = 0.0;
    window.source_fps = 0.0;
    window.width = 0;
    window.height = 0;
    window.playhead = 0.0;
    window.playing = false;
    window.last_tick = 0.0;
    if (window.frame.pending.valid()) {
        try {
            (void)window.frame.pending.get();
        } catch (...) {
        }
    }
    window.frame.pending = {};
    window.frame.texture.reset();
    window.frame.loaded_index = -1;
    window.frame.failed_index = -1;
    load_segments(app);

    std::error_code error;
    const int generation = ++window.preview_generation;
    const auto directory = preview_directory(app, generation);
    std::filesystem::remove_all(directory, error);
    std::filesystem::create_directories(directory, error);
    if (error) {
        window.preview_error =
            "Cannot create the video preview folder";
        return;
    }
    window.proxy_dir = directory;
    auto state = std::make_shared<VideoPreviewState>();
    window.preview = state;
    const float proxy_fps = window.proxy_fps;
    window.preview_job = std::async(
        std::launch::async,
        [source, directory, state, proxy_fps]() -> int {
            try {
                photara::io::VideoProbe probe;
                photara::io::probe_video({}, source, probe);
                state->duration.store(probe.duration_seconds);
                state->source_fps.store(probe.fps);
                state->width.store(probe.width);
                state->height.store(probe.height);
                photara::io::VideoPreviewOptions options;
                options.video = source;
                options.output_dir = directory;
                options.fps = proxy_fps;
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
    if (window.duration <= 0.0)
        window.duration = window.preview->duration.load();
    if (window.source_fps <= 0.0)
        window.source_fps = window.preview->source_fps.load();
    if (window.width <= 0) window.width = window.preview->width.load();
    if (window.height <= 0) window.height = window.preview->height.load();
    window.preview_progress = window.preview->progress.load();
    if (window.preview_job.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
        return;

    const bool cancelled = window.preview->cancel.load();
    try {
        window.preview_frames = window.preview_job.get();
        window.preview_done = true;
        window.preview_progress = 1.F;
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

int available_preview_frames(const VideoWindowState& window) {
    if (window.preview_done) return window.preview_frames;
    if (window.duration > 0.0 && window.preview_progress > 0.F)
        return static_cast<int>(
            window.duration * window.proxy_fps *
                std::min(1.F, window.preview_progress));
    return 0;
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

// The proxy writes frames sequentially, so playback can only display an index
// that is already on disk. A single async decode keeps the UI responsive while
// prefetching the next visible frame.
void request_frame(VideoWindowState& window, const int index) {
    auto& frame = window.frame;
    if (index <= 0) return;
    harvest_frame(frame);
    if (frame.pending.valid()) return;
    if (index == frame.loaded_index) return;
    if (index == frame.failed_index && ImGui::GetTime() < frame.failed_until)
        return;
    const auto path = preview_frame_path(window, index);
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        frame.failed_index = index;
        frame.failed_until = ImGui::GetTime() + 0.25;
        return;
    }
    frame.pending_index = index;
    frame.pending_path = path;
    // The caller draws the previous texture while this decode runs.
    frame.pending = std::async(
        std::launch::async, [path]() { return photara::io::load_rgb(path); });
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

    const int available = available_preview_frames(window);
    const int desired = window.duration > 0.0
        ? static_cast<int>(window.playhead * window.proxy_fps) + 1
        : 1;
    if (available > 0) {
        request_frame(window, std::clamp(desired, 1, available));
        const int next = std::min(available, desired + 1);
        if (!window.frame.pending.valid() && next != desired)
            request_frame(window, next);
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

    if (window.preview_job.valid()) {
        char label[96];
        if (window.preview_progress >= 0.F)
            std::snprintf(
                label, sizeof(label), "%s %d%%",
                tr("Building playback proxy..."),
                static_cast<int>(window.preview_progress * 100.F));
        else
            std::snprintf(
                label, sizeof(label), "%s", tr("Probing video..."));
        draw->AddText(
            {image_min.x + 8.F, image_max.y - 22.F},
            theme::u32(theme::text_bright), label);
    } else if (!window.preview_done && window.preview_error.empty()) {
        draw->AddText(
            {image_min.x + 8.F, image_max.y - 22.F},
            theme::u32(theme::warning), tr("Buffering preview frames..."));
    }

    ImGui::InvisibleButton(
        "##video_canvas_click", area_size,
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
        window.playing = !window.playing;
        window.last_tick = ImGui::GetTime();
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void draw_timeline(App& app) {
    auto& window = app.video_window;
    const double duration = window.duration;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    constexpr float height = 24.F;
    ImGui::InvisibleButton(
        "##video_timeline", {width, height},
        ImGuiButtonFlags_MouseButtonLeft);
    if (ImGui::IsItemActive() && duration > 0.0) {
        const float x = ImGui::GetIO().MousePos.x - origin.x;
        window.playhead = clamp_time(
            static_cast<double>(x) / std::max(1.F, width) * duration,
            duration);
        window.playing = false;
    }
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(
        origin, {origin.x + width, origin.y + height},
        theme::u32(theme::surface_2), 4.F);
    if (duration > 0.0) {
        const float buffered = window.preview_done
            ? 1.F
            : std::max(0.F, std::min(1.F, window.preview_progress));
        if (buffered > 0.F)
            draw->AddRectFilled(
                origin,
                {origin.x + width * buffered, origin.y + height},
                theme::u32(theme::surface_3), 4.F);
        for (const auto& segment : window.segments) {
            const float x0 = static_cast<float>(
                segment.start_seconds / duration) * width;
            const double end = segment.end_seconds > segment.start_seconds
                ? segment.end_seconds
                : duration;
            const float x1 = static_cast<float>(end / duration) * width;
            draw->AddRectFilled(
                {origin.x + x0, origin.y + 3.F},
                {origin.x + std::min(width, x1), origin.y + height - 3.F},
                theme::u32(theme::accent_deep, 0.75F), 3.F);
        }
        const float playhead_x = static_cast<float>(
            window.playhead / duration) * width;
        draw->AddLine(
            {origin.x + playhead_x, origin.y},
            {origin.x + playhead_x, origin.y + height},
            theme::u32(theme::accent), 2.F);
        if (ImGui::IsItemHovered()) {
            const float x = ImGui::GetIO().MousePos.x - origin.x;
            const double hovered = clamp_time(
                static_cast<double>(x) / std::max(1.F, width) * duration,
                duration);
            ImGui::SetTooltip("%s", format_clock(hovered).c_str());
        }
    } else {
        draw->AddText(
            {origin.x + 8.F, origin.y + 4.F},
            theme::u32(theme::text_faint), tr("No video loaded"));
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
    if (theme::toolbar_button(tr("<<"), {38.F, 32.F}, ready)) {
        window.playing = false;
        window.playhead = clamp_time(
            window.playhead - 1.0 / window.proxy_fps, window.duration);
    }
    ImGui::SameLine();
    if (theme::toolbar_button(tr(">>"), {38.F, 32.F}, ready)) {
        window.playing = false;
        window.playhead = clamp_time(
            window.playhead + 1.0 / window.proxy_fps, window.duration);
    }
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
        window.playhead += delta;
        if (window.playhead >= window.duration) {
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
    ImGui::TextDisabled("%s", tr(
        "Add one or more source-time ranges. Extraction merges them in time "
        "order into one numbered frame sequence."));
    ImGui::Spacing();
    ImGui::BeginDisabled(window.duration <= 0.0);
    float range_in = static_cast<float>(window.range_in);
    float range_out = static_cast<float>(window.range_out);
    const float field_width = std::max(100.F, (ImGui::GetContentRegionAvail().x - 330.F) * 0.5F);
    ImGui::TextDisabled("%s", tr("Start"));
    ImGui::SameLine(field_width + ImGui::GetStyle().ItemSpacing.x);
    ImGui::TextDisabled("%s", tr("End"));
    ImGui::SetNextItemWidth(field_width);
    if (ImGui::DragFloat(
            "##vw_range_in", &range_in, 0.05F, 0.F,
            static_cast<float>(window.duration), "%.2f s"))
        window.range_in = clamp_time(range_in, window.duration);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(field_width);
    if (ImGui::DragFloat(
            "##vw_range_out", &range_out, 0.05F, 0.F,
            static_cast<float>(window.duration), "%.2f s"))
        window.range_out = clamp_time(range_out, window.duration);
    ImGui::SameLine();
    if (ImGui::Button(tr("Set In")))
        window.range_in = window.playhead;
    ImGui::SameLine();
    if (ImGui::Button(tr("Set Out")))
        window.range_out = window.playhead;
    ImGui::SameLine();
    if (ImGui::Button(tr("Add Segment"), {0.F, 0.F})) {
        double end = window.range_out;
        if (end <= window.range_in) end = window.duration;
        window.segments.push_back({window.range_in, end});
        window.segments =
            photara::io::normalize_video_segments(window.segments);
        sync_segment_text(app);
        window.range_in = window.range_out;
        window.range_out = window.duration;
    }
    ImGui::EndDisabled();
    ImGui::Spacing();
    if (!window.segments_error.empty())
        ImGui::TextColored(
            theme::danger, "%s", window.segments_error.c_str());

    if (window.segments.empty()) {
        ImGui::TextColored(
            theme::text_muted, "%s", tr("Whole video will be extracted."));
        return;
    }
    if (ImGui::BeginTable(
            "##vw_segments", 3,
            ImGuiTableFlags_SizingStretchProp |
                ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH)) {
        ImGui::TableSetupColumn(tr("Range"));
        ImGui::TableSetupColumn(tr("Duration"));
        ImGui::TableSetupColumn("##actions", ImGuiTableColumnFlags_WidthFixed,
                                120.F);
        ImGui::TableHeadersRow();
        int remove = -1;
        for (std::size_t i = 0; i < window.segments.size(); ++i) {
            const auto& segment = window.segments[i];
            const double end = segment.end_seconds > segment.start_seconds
                ? segment.end_seconds
                : window.duration;
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text(
                "%zu  %s - %s", i + 1,
                format_clock(segment.start_seconds).c_str(),
                format_clock(end).c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(
                format_seconds(end - segment.start_seconds).c_str());
            ImGui::TableSetColumnIndex(2);
            if (ImGui::SmallButton(
                    (std::string(tr("Go")) + "##vw_go_" +
                     std::to_string(i)).c_str()))
                window.playhead = segment.start_seconds;
            ImGui::SameLine();
            if (ImGui::SmallButton(
                    (std::string(tr("Remove")) + "##vw_rm_" +
                     std::to_string(i)).c_str()))
                remove = static_cast<int>(i);
        }
        ImGui::EndTable();
        if (remove >= 0) {
            window.segments.erase(window.segments.begin() + remove);
            sync_segment_text(app);
        }
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
            "##vw_extract_controls", 3,
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
        ImGui::TableSetColumnIndex(2);
        ImGui::TextDisabled("%s", tr("Sharpness"));
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderInt(
            "##vw_sharp", &settings.video_sharp_window, 1, 10,
            tr("sharp %d"));
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("%s", tr("Image format / quality"));
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderInt(
            "##vw_quality", &settings.video_quality, -1, 100,
            settings.video_quality < 0 ? tr("PNG") : tr("JPEG %d"));
        ImGui::TableSetColumnIndex(1);
        ImGui::TextDisabled("%s", tr("Output scale"));
        ImGui::SetNextItemWidth(-1.F);
        ImGui::DragFloat(
            "##vw_scale", &settings.video_scale, 0.01F, 0.05F, 4.F, "%.2fx");
        ImGui::TableSetColumnIndex(2);
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
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::surface_1);
    ImGui::BeginChild("##vw_source_header", {0.F, 48.F}, true,
                      ImGuiWindowFlags_NoScrollbar);
    const std::string filename = path_to_utf8(source.filename());
    const float header_width = ImGui::GetWindowWidth();
    ImGui::SetCursorPos({12.F, 7.F});
    const ImVec2 title_min = ImGui::GetCursorScreenPos();
    ImGui::PushClipRect(title_min, {title_min.x + header_width - 180.F,
                                   title_min.y + 20.F}, true);
    ImGui::TextColored(theme::text_bright, "%s", filename.c_str());
    ImGui::PopClipRect();
    const float pill_x = std::min(
        12.F + ImGui::CalcTextSize(filename.c_str()).x + 12.F,
        header_width - 150.F);
    ImGui::SetCursorPos({pill_x, 8.F});
    if (window.preview_job.valid())
        theme::pill(tr("preparing playback"), theme::warning, theme::surface_3);
    else if (window.preview_done)
        theme::pill(
            (std::to_string(window.preview_frames) + " " +
             tr("proxy frames")).c_str(),
            theme::success, theme::surface_3);
    else
        theme::pill(tr("Video"), theme::text_muted, theme::surface_3);

    ImGui::SetCursorPos({12.F, 27.F});
    ImGui::PushFont(theme::small_font());
    const std::string full_path = path_to_utf8(source);
    const ImVec2 path_min = ImGui::GetCursorScreenPos();
    const float path_right = window.width > 0
        ? ImGui::GetWindowPos().x + header_width - 170.F
        : ImGui::GetWindowPos().x + header_width - 12.F;
    ImGui::PushClipRect(path_min, {path_right, path_min.y + 18.F}, true);
    ImGui::TextDisabled("%s", full_path.c_str());
    ImGui::PopClipRect();
    ImGui::PopFont();
    if (window.width > 0 && window.height > 0) {
        char details[80];
        std::snprintf(details, sizeof(details), "%d × %d  ·  %.2f fps",
                      window.width, window.height, window.source_fps);
        const float details_width = ImGui::CalcTextSize(details).x;
        ImGui::SetCursorPos(
            {header_width - details_width - 14.F, 29.F});
        ImGui::TextDisabled("%s", details);
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

}  // namespace

void open_video_window(App& app) {
    app.video_window.show = true;
    app.video_window.raise = true;
}

void poll_video_window(App& app) {
    harvest_retired_previews(app);
    poll_preview_job(app);
    poll_extract_job(app);
}

void draw_video_window(App& app) {
    auto& window = app.video_window;
    if (!window.show) return;

    const auto source =
        path_from_utf8_field(app.settings.images_dir.data());
    if (photara::io::is_video_path(source) && source != window.source &&
        !window.extracting)
        start_preview_job(app, source);

    if (window.raise) {
        ImGui::SetNextWindowFocus();
        window.raise = false;
    }
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        viewport->GetCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5F, 0.5F));
    ImGui::SetNextWindowSize({1120.F, 800.F}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(760.F, 520.F), ImVec2(FLT_MAX, FLT_MAX));
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
    ImGui::Spacing();
    if (!window.segments_error.empty())
        ImGui::TextColored(
            theme::danger, "%s", window.segments_error.c_str());

    const float available_height = ImGui::GetContentRegionAvail().y;
    const float canvas_height =
        std::clamp(available_height - 328.F, 190.F, 520.F);
    draw_preview_canvas(app, canvas_height);
    draw_timeline(app);
    draw_transport(app);
    ImGui::Separator();

    const float list_height =
        std::clamp(ImGui::GetContentRegionAvail().y - 36.F, 120.F, 260.F);
    ImGui::BeginChild(
        "##vw_options", {0.F, list_height}, ImGuiChildFlags_None,
        ImGuiWindowFlags_None);
    if (ImGui::BeginTabBar("##vw_tabs")) {
        if (ImGui::BeginTabItem(tr("Segments"))) {
            draw_segments(app);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem(tr("Extraction Settings"))) {
            draw_extract_settings(app);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::EndChild();

    ImGui::End();
}

void shutdown_video_window(App& app) {
    auto& window = app.video_window;
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
