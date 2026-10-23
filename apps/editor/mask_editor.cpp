#include "mask_editor.hpp"

#include "app.hpp"
#include "i18n.hpp"
#include "sam_model.hpp"
#include "sam_prompt_suggestions.hpp"
#include "theme.hpp"

#include "sam/model_cache.hpp"

#include "imgui_internal.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <system_error>
#include <utility>

namespace editor {
namespace {

using i18n::tr;

constexpr std::size_t k_max_undo = 16;
constexpr std::uint32_t k_max_display_side = 2048;
constexpr std::uint32_t k_max_overlay_side = 1600;

std::filesystem::path session_directory(App& app) {
    if (!app.layout.session_dir.empty()) return app.layout.session_dir;
    return std::filesystem::temp_directory_path() / "photara-studio-masks";
}

bool is_still_image(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](const unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    return extension == ".jpg" || extension == ".jpeg" ||
           extension == ".png" || extension == ".tif" ||
           extension == ".tiff" || extension == ".bmp";
}

std::vector<std::filesystem::path> list_images(
    const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> images;
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) return images;
    for (const auto& entry :
         std::filesystem::directory_iterator(directory, error)) {
        if (error) break;
        std::error_code file_error;
        if (entry.is_regular_file(file_error) && is_still_image(entry.path()))
            images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());
    return images;
}

std::filesystem::path mask_path_for(
    const std::filesystem::path& mask_dir,
    const std::filesystem::path& image) {
    auto png = mask_dir / image.stem();
    png += ".png";
    std::error_code error;
    if (std::filesystem::is_regular_file(png, error)) return png;
    const auto exact = mask_dir / image.filename();
    if (std::filesystem::is_regular_file(exact, error)) return exact;
    return png;
}

bool mask_marker_exists(const std::filesystem::path& mask_dir) {
    if (mask_dir.empty()) return false;
    std::error_code error;
    return std::filesystem::is_regular_file(
        mask_dir / k_mask_ready_marker, error);
}

void write_mask_marker(const std::filesystem::path& mask_dir) {
    std::error_code error;
    std::filesystem::create_directories(mask_dir, error);
    if (error) return;
    std::ofstream out(mask_dir / k_mask_ready_marker, std::ios::trunc);
    if (out) out << "edited=1\n";
}

photara::io::GrayImage resize_mask_nearest(
    const photara::io::GrayImage& source, const std::uint32_t width,
    const std::uint32_t height) {
    if (source.width == width && source.height == height) return source;
    photara::io::GrayImage result;
    if (width == 0 || height == 0 || source.width == 0 || source.height == 0)
        return result;
    result.width = width;
    result.height = height;
    result.pixels.resize(static_cast<std::size_t>(width) * height);
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint32_t source_y = y * source.height / height;
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::uint32_t source_x = x * source.width / width;
            result.pixels[static_cast<std::size_t>(y) * width + x] =
                source.pixels[static_cast<std::size_t>(source_y) *
                                  source.width +
                              source_x];
        }
    }
    return result;
}

photara::io::RgbImage fit_image_for_display(
    const photara::io::RgbImage& source) {
    const std::uint32_t long_side = std::max(source.width, source.height);
    if (long_side <= k_max_display_side || source.width == 0 ||
        source.height == 0)
        return source;
    const double scale = static_cast<double>(k_max_display_side) / long_side;
    photara::io::RgbImage result;
    result.width = std::max(1U, static_cast<std::uint32_t>(source.width * scale));
    result.height = std::max(1U, static_cast<std::uint32_t>(source.height * scale));
    result.pixels.resize(
        static_cast<std::size_t>(result.width) * result.height * 3);
    photara::io::resize_bilinear(
        source.pixels.data(), source.width, source.height, 3,
        result.pixels.data(), result.width, result.height);
    return result;
}

void push_undo(MaskEditorState& state) {
    if (!state.loaded || state.mask.pixels.empty()) return;
    state.undo.push_back(state.mask.pixels);
    while (state.undo.size() > k_max_undo) state.undo.pop_front();
    state.redo.clear();
}

void rebuild_overlay(MaskEditorState& state) {
    state.mask_pixels = state.mask.pixels.size();
    state.kept_pixels = static_cast<std::size_t>(std::count_if(
        state.mask.pixels.begin(), state.mask.pixels.end(),
        [](const std::uint8_t value) { return value >= 128; }));
    if ((!state.has_mask && !state.dirty) ||
        (state.kept_pixels == 0 && state.view_mode != 2)) {
        state.overlay_texture.reset();
        state.texture_dirty = false;
        return;
    }
    if (!state.loaded || state.mask.width == 0 || state.mask.height == 0) {
        state.overlay_texture.reset();
        state.texture_dirty = false;
        return;
    }
    if (state.view_mode == 1) {
        state.texture_dirty = false;
        return;
    }
    const std::uint32_t long_side =
        std::max(state.mask.width, state.mask.height);
    const double scale = long_side > k_max_overlay_side
        ? static_cast<double>(k_max_overlay_side) / long_side
        : 1.0;
    const std::uint32_t width = std::max(
        1U, static_cast<std::uint32_t>(state.mask.width * scale));
    const std::uint32_t height = std::max(
        1U, static_cast<std::uint32_t>(state.mask.height * scale));
    if (state.view_mode == 2) {
        photara::io::RgbImage image;
        image.width = width;
        image.height = height;
        image.pixels.resize(static_cast<std::size_t>(width) * height * 3);
        for (std::uint32_t y = 0; y < height; ++y) {
            const std::uint32_t source_y = y * state.mask.height / height;
            for (std::uint32_t x = 0; x < width; ++x) {
                const std::uint32_t source_x = x * state.mask.width / width;
                const bool keep =
                    state.mask.pixels[static_cast<std::size_t>(source_y) *
                                          state.mask.width +
                                      source_x] >= 128;
                const std::uint8_t value = keep ? 255 : 0;
                const std::size_t index =
                    (static_cast<std::size_t>(y) * width + x) * 3;
                image.pixels[index] = value;
                image.pixels[index + 1] = value;
                image.pixels[index + 2] = value;
            }
        }
        state.overlay_texture.upload(image);
    } else {
        // Keep masks use white for reconstruction and black for exclusion.
        // Tint excluded pixels so the retained subject stays unobscured.
        const ImVec4 colour = theme::danger;
        std::vector<std::uint8_t> rgba(
            static_cast<std::size_t>(width) * height * 4, 0);
        for (std::uint32_t y = 0; y < height; ++y) {
            const std::uint32_t source_y = y * state.mask.height / height;
            for (std::uint32_t x = 0; x < width; ++x) {
                const std::uint32_t source_x = x * state.mask.width / width;
                if (state.mask.pixels[static_cast<std::size_t>(source_y) *
                                          state.mask.width +
                                      source_x] >= 128)
                    continue;
                const std::size_t index =
                    (static_cast<std::size_t>(y) * width + x) * 4;
                rgba[index] = static_cast<std::uint8_t>(
                    std::clamp(colour.x, 0.F, 1.F) * 255.F);
                rgba[index + 1] = static_cast<std::uint8_t>(
                    std::clamp(colour.y, 0.F, 1.F) * 255.F);
                rgba[index + 2] = static_cast<std::uint8_t>(
                    std::clamp(colour.z, 0.F, 1.F) * 255.F);
                rgba[index + 3] = 255;
            }
        }
        state.overlay_texture.upload_rgba(rgba.data(), width, height);
    }
    state.overlay_width = width;
    state.overlay_height = height;
    state.texture_dirty = false;
}

void fill_holes(photara::io::GrayImage& mask) {
    if (mask.width == 0 || mask.height == 0) return;
    const std::size_t width = mask.width;
    const std::size_t height = mask.height;
    std::vector<std::uint8_t> outside(width * height, 0);
    std::vector<std::size_t> stack;
    auto seed = [&](const std::size_t index) {
        if (mask.pixels[index] == 0 && outside[index] == 0) {
            outside[index] = 1;
            stack.push_back(index);
        }
    };
    for (std::size_t x = 0; x < width; ++x) {
        seed(x);
        seed((height - 1) * width + x);
    }
    for (std::size_t y = 0; y < height; ++y) {
        seed(y * width);
        seed(y * width + width - 1);
    }
    while (!stack.empty()) {
        const std::size_t index = stack.back();
        stack.pop_back();
        const std::size_t x = index % width;
        const std::size_t y = index / width;
        if (x > 0) seed(index - 1);
        if (x + 1 < width) seed(index + 1);
        if (y > 0) seed(index - width);
        if (y + 1 < height) seed(index + width);
    }
    for (std::size_t i = 0; i < mask.pixels.size(); ++i) {
        if (mask.pixels[i] == 0 && outside[i] == 0) mask.pixels[i] = 255;
    }
}

void paint_circle(
    photara::io::GrayImage& mask, const float centre_x, const float centre_y,
    const float radius, const std::uint8_t value) {
    if (mask.width == 0 || mask.height == 0) return;
    const int x0 = std::max(
        0, static_cast<int>(std::floor(centre_x - radius)));
    const int x1 = std::min(
        static_cast<int>(mask.width) - 1,
        static_cast<int>(std::ceil(centre_x + radius)));
    const int y0 = std::max(
        0, static_cast<int>(std::floor(centre_y - radius)));
    const int y1 = std::min(
        static_cast<int>(mask.height) - 1,
        static_cast<int>(std::ceil(centre_y + radius)));
    const float radius_squared = radius * radius;
    for (int y = y0; y <= y1; ++y) {
        const float dy = static_cast<float>(y) + 0.5F - centre_y;
        for (int x = x0; x <= x1; ++x) {
            const float dx = static_cast<float>(x) + 0.5F - centre_x;
            if (dx * dx + dy * dy > radius_squared) continue;
            mask.pixels[static_cast<std::size_t>(y) * mask.width + x] = value;
        }
    }
}

void paint_stroke(
    photara::io::GrayImage& mask, const ImVec2 from, const ImVec2 to,
    const float radius, const std::uint8_t value) {
    const float dx = to.x - from.x;
    const float dy = to.y - from.y;
    const float distance = std::sqrt(dx * dx + dy * dy);
    const int steps = std::max(
        1, static_cast<int>(std::ceil(distance / std::max(1.F, radius * 0.5F))));
    for (int i = 0; i <= steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
        paint_circle(
            mask, from.x + dx * t, from.y + dy * t, radius, value);
    }
}

void start_image_load(App& app, const int index) {
    auto& state = app.mask_editor;
    if (index < 0 || index >= static_cast<int>(state.images.size())) return;
    if (state.pending_load.valid())
        state.retired_loads.push_back(std::move(state.pending_load));
    const auto image_path = state.images[static_cast<std::size_t>(index)];
    const auto mask_path = mask_path_for(state.mask_dir, image_path);
    state.pending_index = index;
    state.loaded = false;
    state.load_error.clear();
    state.pending_load = std::async(
        std::launch::async,
        [image_path, mask_path]() {
            MaskImageLoad result;
            result.image_path = image_path;
            result.mask_path = mask_path;
            const auto size = photara::io::load_image_size(image_path);
            result.image = photara::io::load_rgb_with_minimum_size(
                image_path, size.width, size.height);
            std::error_code error;
            if (std::filesystem::is_regular_file(mask_path, error)) {
                try {
                    result.mask = photara::io::load_gray(mask_path);
                    result.has_mask = result.mask.width > 0 &&
                        result.mask.height > 0;
                } catch (...) {
                    result.has_mask = false;
                }
            }
            return result;
        });
}

void apply_select(App& app, const int index) {
    auto& state = app.mask_editor;
    if (index < 0 || index >= static_cast<int>(state.images.size())) return;
    state.selected = index;
    state.dirty = false;
    state.zoom = 1.F;
    state.pan = ImVec2{};
    start_image_load(app, index);
}

void request_select(App& app, const int index) {
    auto& state = app.mask_editor;
    if (index < 0 || index >= static_cast<int>(state.images.size()) ||
        index == state.selected)
        return;
    if (state.generating) return;
    if (state.dirty) {
        state.pending_select = index;
        state.confirm_discard = true;
        return;
    }
    apply_select(app, index);
}

void select_first_available(App& app) {
    auto& state = app.mask_editor;
    if (state.images.empty()) {
        state.selected = -1;
        state.loaded = false;
        state.pending_index = -1;
        state.image_texture.reset();
        state.overlay_texture.reset();
        return;
    }
    const int index = std::clamp(
        state.selected, 0, static_cast<int>(state.images.size()) - 1);
    state.selected = index;
    start_image_load(app, index);
}

void refresh_image_list(App& app, const bool force) {
    auto& state = app.mask_editor;
    const auto images_dir = reconstruction_images_dir(app.settings);
    const auto mask_dir = masks_directory(app.settings);
    std::error_code error;
    const auto stamp = std::filesystem::last_write_time(images_dir, error);
    const bool directory_changed =
        images_dir != state.images_dir || mask_dir != state.mask_dir;
    const bool stamp_changed = !error && stamp != state.images_stamp;
    if (!force && !directory_changed && !stamp_changed && !state.images.empty())
        return;

    const std::filesystem::path previous =
        state.selected >= 0 &&
                state.selected < static_cast<int>(state.images.size())
            ? state.images[static_cast<std::size_t>(state.selected)]
            : std::filesystem::path{};
    const bool reset_content = directory_changed;
    state.images_dir = images_dir;
    state.mask_dir = mask_dir;
    state.images_stamp = error ? std::filesystem::file_time_type{} : stamp;
    state.images = list_images(images_dir);
    state.mask_ready = mask_marker_exists(mask_dir);
    if (reset_content) {
        state.image_texture.reset();
        state.overlay_texture.reset();
        state.mask = {};
        state.saved_mask = {};
        state.has_mask = false;
        state.loaded = false;
        state.dirty = false;
        state.undo.clear();
        state.redo.clear();
        state.selected = -1;
        state.pending_index = -1;
    }
    if (state.images.empty()) {
        select_first_available(app);
        return;
    }
    int selected = 0;
    if (!previous.empty()) {
        const auto it = std::find(state.images.begin(), state.images.end(), previous);
        if (it != state.images.end())
            selected = static_cast<int>(it - state.images.begin());
    }
    if (reset_content || selected != state.selected ||
        !state.pending_load.valid())
        apply_select(app, selected);
}

void poll_image_load(App& app) {
    auto& state = app.mask_editor;
    for (auto it = state.retired_loads.begin();
         it != state.retired_loads.end();) {
        if (it->wait_for(std::chrono::seconds(0)) !=
            std::future_status::ready) {
            ++it;
            continue;
        }
        try {
            (void)it->get();
        } catch (...) {
        }
        it = state.retired_loads.erase(it);
    }
    if (!state.pending_load.valid()) return;
    if (state.pending_load.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
        return;
    try {
        MaskImageLoad load = state.pending_load.get();
        if (load.image.width == 0 || load.image.height == 0)
            throw std::runtime_error("The image could not be decoded");
        state.image_width = load.image.width;
        state.image_height = load.image.height;
        state.image_texture.upload(fit_image_for_display(load.image));
        photara::io::GrayImage mask = std::move(load.mask);
        if (mask.width == 0 || mask.height == 0) {
            mask.width = load.image.width;
            mask.height = load.image.height;
            mask.pixels.assign(
                static_cast<std::size_t>(mask.width) * mask.height, 255);
        } else {
            mask = resize_mask_nearest(
                mask, load.image.width, load.image.height);
        }
        state.mask = std::move(mask);
        state.saved_mask = state.mask;
        state.has_mask = load.has_mask;
        state.loaded = true;
        state.dirty = false;
        state.painting = false;
        state.stroke_active = false;
        state.undo.clear();
        state.redo.clear();
        state.pending_index = -1;
        state.texture_dirty = true;
        state.zoom = 1.F;
        state.pan = ImVec2{};
        state.load_error.clear();
    } catch (const std::exception& error) {
        state.load_error = error.what();
        state.loaded = false;
        state.pending_index = -1;
    }
}

bool save_current_mask(App& app) {
    auto& state = app.mask_editor;
    if (!state.loaded || state.selected < 0 ||
        state.selected >= static_cast<int>(state.images.size()))
        return false;
    try {
        const auto original_size = photara::io::load_image_size(
            state.images[static_cast<std::size_t>(state.selected)]);
        if (state.mask.width != original_size.width ||
            state.mask.height != original_size.height)
            throw std::runtime_error(
                "Mask size does not match the source image; reload before saving");
        std::error_code error;
        std::filesystem::create_directories(state.mask_dir, error);
        auto destination =
            state.mask_dir / state.images[static_cast<std::size_t>(state.selected)]
                                 .stem();
        destination += ".png";
        photara::io::save_gray_png(state.mask, destination);
        state.saved_mask = state.mask;
        state.has_mask = true;
        state.dirty = false;
        write_mask_marker(state.mask_dir);
        state.mask_ready = true;
        app.settings.sam_masks = true;
        app.settings.use_mask = true;
        state.gen_note = tr("Mask saved. Alignment will reuse the edited masks.");
        set_message(app, "Mask saved", theme::success);
        return true;
    } catch (const std::exception& error) {
        state.gen_error = error.what();
        set_message(app, error.what(), theme::danger);
        return false;
    }
}

struct MaskGenerateTemplate {
    std::filesystem::path model;
    std::filesystem::path output_dir;
    std::vector<std::filesystem::path> images;
    std::string text;
    std::string negative_text;
    std::string backend{"auto"};
    bool keep_prompted{true};
    bool video{true};
    int max_size{};
    float central_fraction{0.01F};
    float threshold{0.5F};
    float nms{0.1F};
    float min_area{0.001F};
    float max_area{0.65F};
    int close_kernel{5};
    bool fill_holes{true};
};

#if defined(PHOTARA_HAS_SAM)
void start_generation(App& app, const bool all) {
    auto& state = app.mask_editor;
    if (state.generating || app.job.running()) return;
    if (state.images.empty()) {
        set_message(app, "Extract frames before generating masks", theme::warning);
        return;
    }
    if (!all && (state.selected < 0 ||
                 state.selected >= static_cast<int>(state.images.size())))
        return;
    if (app.settings.sam_text[0] == '\0') {
        set_message(app, "Enter a SAM3 prompt before generating masks",
                    theme::warning);
        return;
    }
    if (!(app.settings.sam_min_area >= 0.F &&
          app.settings.sam_min_area < app.settings.sam_max_area &&
          app.settings.sam_max_area <= 1.F)) {
        set_message(app, "SAM area fractions must satisfy 0 <= min < max <= 1",
                    theme::danger);
        return;
    }
    app.settings.sam_masks = true;
    if (!ensure_sam_ready(app)) return;
    std::filesystem::path model =
        path_from_utf8_field(app.settings.sam_model.data());
    if (model.empty()) model = photara::sam::locate_model();
    if (model.empty()) {
        set_message(app, "The SAM 3 model is not available", theme::danger);
        return;
    }
    std::optional<photara::sfm::Scene> scene;
    try {
        if (all) scene = load_alignment_scene(app);
    } catch (...) {
        // A damaged sidecar must not block a coarse prompt-only pass.
    }
    std::error_code error;
    const auto temp_dir = session_directory(app) /
        (all ? "mask_generation_all" : "mask_generation_one");
    std::filesystem::remove_all(temp_dir, error);
    std::filesystem::create_directories(temp_dir, error);
    if (error) {
        set_message(app, "Cannot create the mask preview folder", theme::danger);
        return;
    }
    state.gen_temp_dir = temp_dir;
    state.generate_all = all;
    state.generating = true;
    state.gen_error.clear();
    state.gen_note.clear();
    state.gen_cancel = std::make_shared<std::atomic<bool>>(false);
    state.gen_completed = std::make_shared<std::atomic<int>>(0);
    state.gen_total = std::make_shared<std::atomic<int>>(
        all ? static_cast<int>(state.images.size()) : 1);

    MaskGenerateTemplate request;
    request.model = model;
    request.output_dir = temp_dir;
    if (all)
        request.images = state.images;
    else
        request.images = {
            state.images[static_cast<std::size_t>(state.selected)]};
    request.text = app.settings.sam_text.data();
    request.negative_text = app.settings.sam_negative_text.data();
    request.backend = sam_backend_name(app.settings);
    request.keep_prompted = app.settings.sam_keep_prompted;
    request.video = all && app.settings.sam_video;
    request.max_size = std::max(0, app.settings.sam_max_size);
    request.threshold = scene ? app.settings.sam_guided_threshold
                              : app.settings.sam_threshold;
    request.nms = app.settings.sam_nms;
    request.min_area = app.settings.sam_min_area;
    request.max_area = app.settings.sam_max_area;
    request.close_kernel = app.settings.sam_close_kernel;
    request.fill_holes = app.settings.sam_fill_holes;
    const auto cancel = state.gen_cancel;
    const auto completed = state.gen_completed;
    state.gen_job = std::async(
        std::launch::async,
        [request = std::move(request), scene = std::move(scene), cancel,
         require_target = !all,
         completed]() mutable {
            photara::sam::GenerateOptions options;
            options.model = request.model;
            options.output_dir = request.output_dir;
            options.images = std::move(request.images);
            options.text = request.text;
            options.negative_text = request.negative_text;
            options.backend = request.backend;
            options.keep_prompted = request.keep_prompted;
            options.video = request.video;
            options.max_size = request.max_size;
            options.sparse_scene = scene ? &*scene : nullptr;
            options.central_fraction = request.central_fraction;
            options.threshold = request.threshold;
            options.nms = request.nms;
            options.min_area_fraction = request.min_area;
            options.max_area_fraction = request.max_area;
            options.close_kernel = request.close_kernel;
            options.fill_holes = request.fill_holes;
            options.require_target = require_target;
            options.cancel = cancel.get();
            options.progress = [completed](
                                   const std::size_t done,
                                   const std::size_t) {
                completed->store(static_cast<int>(done));
            };
            return photara::sam::generate_masks(options);
        });
}

void begin_generate_all(App& app) {
    auto& state = app.mask_editor;
    if (state.generating) return;
    bool has_existing = state.mask_ready || state.has_mask;
    if (!has_existing && !state.mask_dir.empty()) {
        std::error_code error;
        for (const auto& entry :
             std::filesystem::directory_iterator(state.mask_dir, error)) {
            if (error) break;
            std::error_code file_error;
            if (entry.is_regular_file(file_error) &&
                is_still_image(entry.path())) {
                has_existing = true;
                break;
            }
        }
    }
    if (has_existing) {
        state.confirm_generate_all = true;
        return;
    }
    start_generation(app, true);
}
#else
void start_generation(App&, bool) {}
void begin_generate_all(App& app) {
    set_message(app, "This build does not include SAM mask generation",
                theme::danger);
}
#endif

void poll_generation(App& app) {
    auto& state = app.mask_editor;
    if (!state.generating || !state.gen_job.valid()) return;
    if (state.gen_job.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
        return;
    const bool cancelled = state.gen_cancel && state.gen_cancel->load();
    try {
        const auto result = state.gen_job.get();
        state.view_mode = 0;
        state.overlay_opacity = std::max(0.35F, state.overlay_opacity);
        if (state.generate_all) {
            std::error_code error;
            std::filesystem::create_directories(state.mask_dir, error);
            int copied = 0;
            for (const auto& entry :
                 std::filesystem::directory_iterator(state.gen_temp_dir, error)) {
                if (error) break;
                std::error_code file_error;
                if (!entry.is_regular_file(file_error)) continue;
                if (entry.path().extension() != ".png") continue;
                std::error_code copy_error;
                std::filesystem::copy_file(
                    entry.path(), state.mask_dir / entry.path().filename(),
                    std::filesystem::copy_options::overwrite_existing,
                    copy_error);
                if (copy_error) {
                    state.gen_error = copy_error.message();
                    break;
                }
                ++copied;
            }
            write_mask_marker(state.mask_dir);
            state.mask_ready = mask_marker_exists(state.mask_dir);
            state.dirty = false;
            state.undo.clear();
            state.redo.clear();
            state.gen_note = "Generated " + std::to_string(copied) +
                " masks (" + std::to_string(result.empty) + " empty).";
            if (app.has_sparse)
                state.gen_note +=
                    " Training and the next alignment will reuse them.";
            start_image_load(app, state.selected);
        } else if (state.selected >= 0 &&
                   state.selected < static_cast<int>(state.images.size())) {
            auto path = state.gen_temp_dir /
                state.images[static_cast<std::size_t>(state.selected)].stem();
            path += ".png";
            std::error_code error;
            if (!std::filesystem::is_regular_file(path, error))
                throw std::runtime_error(
                    "The re-run produced no mask for this image");
            photara::io::GrayImage mask = photara::io::load_gray(path);
            const bool any_kept = std::any_of(
                mask.pixels.begin(), mask.pixels.end(),
                [](const std::uint8_t value) { return value >= 128; });
            if (!any_kept)
                throw std::runtime_error(tr(
                    "No subject was retained. The current image and mask were left unchanged."));
            mask = resize_mask_nearest(
                mask, state.image_width, state.image_height);
            push_undo(state);
            state.mask = std::move(mask);
            state.has_mask = true;
            state.dirty = true;
            state.texture_dirty = true;
            state.gen_note =
                tr("Re-run result loaded. Save the mask to keep it.");
        }
        if (cancelled)
            state.gen_note = "Mask generation cancelled";
    } catch (const std::exception& error) {
        if (cancelled)
            state.gen_note = "Mask generation cancelled";
        else {
            const std::string message = error.what();
            constexpr const char* no_target =
                "SAM did not identify the requested target: ";
            state.gen_error = message.rfind(no_target, 0) == 0
                ? std::string(tr("SAM did not identify the requested target. No mask was applied:")) +
                    " " + message.substr(std::strlen(no_target))
                : message;
        }
    }
    std::error_code error;
    std::filesystem::remove_all(state.gen_temp_dir, error);
    state.gen_temp_dir.clear();
    state.gen_job = {};
    state.gen_cancel.reset();
    state.gen_completed.reset();
    state.gen_total.reset();
    state.generating = false;
}

void draw_canvas(App& app) {
    auto& state = app.mask_editor;
    ImGui::PushStyleColor(ImGuiCol_ChildBg, theme::viewport_bg);
    ImGui::BeginChild(
        "##mask_canvas", {0.F, 0.F}, ImGuiChildFlags_None,
        ImGuiWindowFlags_NoScrollbar);
    const ImVec2 area_min = ImGui::GetCursorScreenPos();
    const ImVec2 area = ImGui::GetContentRegionAvail();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(
        area_min, {area_min.x + area.x, area_min.y + area.y},
        theme::u32(theme::viewport_bg));

    if (!state.loaded || state.image_width == 0 || state.image_height == 0) {
        const char* label = state.load_error.empty()
            ? (state.pending_load.valid() ? tr("Loading image...")
                                          : tr("No image selected"))
            : state.load_error.c_str();
        const ImVec2 text_size = ImGui::CalcTextSize(label);
        draw->AddText(
            {area_min.x + (area.x - text_size.x) * 0.5F,
             area_min.y + (area.y - text_size.y) * 0.5F},
            state.load_error.empty() ? theme::u32(theme::text_muted)
                                     : theme::u32(theme::danger),
            label);
        ImGui::InvisibleButton("##mask_canvas_input", area);
        ImGui::EndChild();
        ImGui::PopStyleColor();
        return;
    }

    const float fit = std::min(
        area.x / static_cast<float>(state.image_width),
        area.y / static_cast<float>(state.image_height));
    const float scale = std::max(0.0001F, fit * state.zoom);
    const ImVec2 image_size{
        state.image_width * scale, state.image_height * scale};
    const ImVec2 centre{
        area_min.x + area.x * 0.5F + state.pan.x,
        area_min.y + area.y * 0.5F + state.pan.y};
    const ImVec2 image_min{
        centre.x - image_size.x * 0.5F, centre.y - image_size.y * 0.5F};
    const ImVec2 image_max{
        centre.x + image_size.x * 0.5F, centre.y + image_size.y * 0.5F};

    if (state.image_texture.descriptor)
        draw->AddImage(
            reinterpret_cast<ImTextureID>(state.image_texture.descriptor),
            image_min, image_max);
    if (state.view_mode == 2) {
        if (state.overlay_texture.descriptor)
            draw->AddImage(
                reinterpret_cast<ImTextureID>(state.overlay_texture.descriptor),
                image_min, image_max);
    } else if (state.view_mode == 0 && state.overlay_texture.descriptor) {
        draw->AddImage(
            reinterpret_cast<ImTextureID>(state.overlay_texture.descriptor),
            image_min, image_max, ImVec2(0.F, 0.F), ImVec2(1.F, 1.F),
            IM_COL32(255, 255, 255,
                     static_cast<int>(std::clamp(state.overlay_opacity, 0.F, 1.F) *
                                      255.F)));
    }
    draw->AddRect(image_min, image_max, theme::u32(theme::border));

    ImGui::SetCursorScreenPos(area_min);
    ImGui::InvisibleButton(
        "##mask_canvas_input", area,
        ImGuiButtonFlags_MouseButtonLeft |
            ImGuiButtonFlags_MouseButtonRight |
            ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    const ImGuiIO& io = ImGui::GetIO();
    const bool over_image =
        hovered && io.MousePos.x >= image_min.x && io.MousePos.x <= image_max.x &&
        io.MousePos.y >= image_min.y && io.MousePos.y <= image_max.y;
    const bool double_click =
        over_image && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

    if (hovered && io.MouseWheel != 0.F) {
        const ImVec2 centre_without_pan{
            area_min.x + area.x * 0.5F, area_min.y + area.y * 0.5F};
        const ImVec2 relative{
            (io.MousePos.x - centre.x) / scale,
            (io.MousePos.y - centre.y) / scale};
        const float factor = std::pow(1.15F, io.MouseWheel);
        const float new_zoom = std::clamp(state.zoom * factor, 1.F, 12.F);
        const float new_scale = fit * new_zoom;
        state.pan.x = io.MousePos.x - centre_without_pan.x -
            relative.x * new_scale;
        state.pan.y = io.MousePos.y - centre_without_pan.y -
            relative.y * new_scale;
        state.zoom = new_zoom;
    }
    if (double_click) {
        state.zoom = 1.F;
        state.pan = ImVec2{};
    }
    if (ImGui::IsItemActive() &&
        (ImGui::IsMouseDown(ImGuiMouseButton_Right) ||
         ImGui::IsMouseDown(ImGuiMouseButton_Middle))) {
        state.pan.x += io.MouseDelta.x;
        state.pan.y += io.MouseDelta.y;
    }

    const ImVec2 image_point{
        std::clamp((io.MousePos.x - image_min.x) / scale, 0.F,
                   static_cast<float>(state.image_width - 1)),
        std::clamp((io.MousePos.y - image_min.y) / scale, 0.F,
                   static_cast<float>(state.image_height - 1))};
    const float source_radius =
        std::max(1.F, state.brush_radius / std::max(0.0001F, scale));

    if (state.manual_editing && !app.job.running() && over_image && !double_click && !state.generating &&
        ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        push_undo(state);
        state.painting = true;
        state.stroke_active = true;
        state.last_paint = image_point;
        paint_circle(
            state.mask, image_point.x, image_point.y, source_radius,
            state.brush_mode == 0 ? 255 : 0);
        state.texture_dirty = true;
    }
    if (state.painting && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        paint_stroke(
            state.mask, state.last_paint, image_point, source_radius,
            state.brush_mode == 0 ? 255 : 0);
        state.last_paint = image_point;
    }
    if (state.painting && !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        state.painting = false;
        state.stroke_active = false;
        state.dirty = true;
        state.texture_dirty = true;
    }

    if (state.stroke_active || (state.manual_editing && !app.job.running() && over_image && !state.generating)) {
        const ImVec4 colour = state.brush_mode == 0 ? theme::success : theme::danger;
        if (state.stroke_active)
            draw->AddCircleFilled(
                io.MousePos, state.brush_radius,
                theme::u32(colour, 0.35F));
        draw->AddCircle(
            io.MousePos, state.brush_radius, theme::u32(colour), 48, 2.F);
    }

    char status[256];
    std::snprintf(
        status, sizeof(status), "%s%s",
        state.dirty ? tr("Unsaved") : "", "");
    if (status[0] != '\0')
        draw->AddText(
            {image_min.x + 6.F, image_min.y + 6.F},
            state.dirty ? theme::u32(theme::warning)
                        : theme::u32(theme::text_faint),
            status);
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

void draw_toolbar(App& app) {
    auto& state = app.mask_editor;
    const int count = static_cast<int>(state.images.size());
    ImGui::BeginDisabled(state.generating || count == 0);
    if (ImGui::Button(i18n::id("Prev", "##mask_prev")))
        request_select(app, state.selected - 1);
    ImGui::SameLine();
    if (ImGui::Button(i18n::id("Next", "##mask_next")))
        request_select(app, state.selected + 1);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (count > 0)
        ImGui::Text("%d / %d", state.selected + 1, count);
    else
        ImGui::TextDisabled("%s", tr("No frames"));
    ImGui::SameLine();
    if (ImGui::Button(tr("Refresh")))
        refresh_image_list(app, true);
    ImGui::SameLine();
    ImGui::BeginDisabled(!state.loaded || state.undo.empty() || state.generating || app.job.running());
    if (ImGui::Button(tr("Undo")))
        app.mask_editor.undo_requested = true;
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!state.loaded || state.redo.empty() || state.generating || app.job.running());
    if (ImGui::Button(tr("Redo")))
        app.mask_editor.redo_requested = true;
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (theme::primary_button(
            tr("Save Mask"), {120.F, 0.F},
            state.loaded && (state.has_mask || state.dirty) && !state.generating && !app.job.running()))
        save_current_mask(app);
    if (state.dirty) {
        ImGui::SameLine();
        theme::pill(tr("Unsaved"), theme::warning, theme::surface_3);
    }
}

void draw_view_controls(MaskEditorState& state) {
    const int view = state.view_mode;
    if (ImGui::RadioButton(tr("Overlay"), view == 0)) {
        state.view_mode = 0;
        state.texture_dirty = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(tr("Image"), view == 1)) {
        state.view_mode = 1;
        state.texture_dirty = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(tr("Mask"), view == 2)) {
        state.view_mode = 2;
        state.texture_dirty = true;
    }
    if (state.view_mode == 0) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(180.F);
        ImGui::SliderFloat(
            "##mask_opacity", &state.overlay_opacity, 0.05F, 1.F,
            tr("overlay %.2f"));
    }
    if (!state.gen_error.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, theme::danger);
        ImGui::TextWrapped("%s", state.gen_error.c_str());
        ImGui::PopStyleColor();
    }
    if (state.loaded && !state.has_mask && !state.dirty && state.gen_error.empty())
        ImGui::TextDisabled("%s", tr("No mask generated. Showing original image."));
    if (state.loaded && (state.has_mask || state.dirty) && state.mask_pixels > 0) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s %.1f%%", tr("Excluded:"),
            100.0 * (state.mask_pixels - state.kept_pixels) / state.mask_pixels);
        if (state.kept_pixels == 0 || state.kept_pixels == state.mask_pixels) {
            ImGui::PushStyleColor(ImGuiCol_Text, theme::warning);
            ImGui::TextWrapped("%s", tr(state.kept_pixels == 0
                ? "Mask excludes the entire image. Check the prompt or regenerate."
                : "Mask keeps the entire image. No background has been excluded."));
            ImGui::PopStyleColor();
        } else if (state.view_mode == 0) {
            ImGui::TextColored(theme::danger, "%s", tr(
                "Tinted pixels are excluded from reconstruction."));
        }
    }
}

void draw_tools(App& app) {
    auto& state = app.mask_editor;
    auto& settings = app.settings;
    theme::caption(tr("Mask"));
    if (!state.loaded) {
        ImGui::TextDisabled("%s", tr("Select a frame to start"));
    } else {
        ImGui::Text(
            "%u x %u", state.image_width, state.image_height);
        if (state.has_mask)
            theme::metric_coloured(tr("File"), tr("Loaded"), theme::success);
        else
            ImGui::TextDisabled("%s", tr("No mask generated. Showing original image."));
    }
    state.manual_editing = ImGui::CollapsingHeader(tr("Manual touch-up"));
    if (state.manual_editing) {
        ImGui::BeginDisabled(!state.loaded || state.generating);
        if (ImGui::Button(tr("Fill Holes"), {116.F, 0.F})) {
            push_undo(state);
            fill_holes(state.mask);
            state.dirty = true;
            state.texture_dirty = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(tr("Invert"), {116.F, 0.F})) {
            push_undo(state);
            for (auto& value : state.mask.pixels)
                value = static_cast<std::uint8_t>(255 - value);
            state.dirty = true;
            state.texture_dirty = true;
        }
        if (ImGui::Button(tr("Reset to Saved"), {116.F, 0.F})) {
            if (!state.saved_mask.pixels.empty()) {
                push_undo(state);
                state.mask = state.saved_mask;
                state.dirty = true;
                state.texture_dirty = true;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button(tr("Clear"), {116.F, 0.F})) {
            push_undo(state);
            std::fill(state.mask.pixels.begin(), state.mask.pixels.end(), 0);
            state.dirty = true;
            state.texture_dirty = true;
        }
        ImGui::EndDisabled();

        theme::caption(tr("Brush"));
        ImGui::RadioButton(tr("Add"), &state.brush_mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton(tr("Erase"), &state.brush_mode, 1);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderFloat(
        "##mask_brush_radius", &state.brush_radius, 2.F, 160.F,
        tr("radius %.0f px"));
        ImGui::TextDisabled("%s", tr(
        "Left drag paints. Right or middle drag pans. Wheel zooms. "
        "Double-click resets the view."));
    }

    ImGui::Separator();
    theme::caption(tr("Mask generation"));
    ImGui::TextDisabled("%s", tr(
        "Align Photos generates masks automatically. Use these controls to inspect or regenerate them when needed."));
    ImGui::TextDisabled("%s", tr("Subject prompt"));
    ImGui::SetNextItemWidth(-1.F);
    ImGui::InputText(
        "##mask_sam_text", settings.sam_text.data(), settings.sam_text.size());
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", tr("Subject prompt. Semicolons separate phrases."));
    draw_sam_prompt_suggestions(settings.sam_text);
    ImGui::TextDisabled("%s", tr("Exclude from mask"));
    ImGui::SetNextItemWidth(-1.F);
    ImGui::InputText(
        "##mask_sam_negative", settings.sam_negative_text.data(),
        settings.sam_negative_text.size());
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", tr("Phrases to subtract from the mask."));
    ImGui::Checkbox(tr("Keep prompted subject"), &settings.sam_keep_prompted);
    ImGui::BeginDisabled(
        state.generating || !state.loaded || app.job.running());
    if (theme::primary_button(
            tr("Re-run SAM for This Image"), {-1.F, 28.F}))
        start_generation(app, false);
    ImGui::EndDisabled();
    ImGui::BeginDisabled(state.generating || app.job.running());
    if (theme::toolbar_button(
            tr("Generate All Masks..."), {-1.F, 28.F}))
        begin_generate_all(app);
    ImGui::EndDisabled();
    if (state.generating) {
        const int total = state.gen_total ? state.gen_total->load() : 0;
        const int completed = state.gen_completed ? state.gen_completed->load() : 0;
        const float fraction = total > 0
            ? static_cast<float>(completed) / total
            : -1.F;
        theme::progress_track({-1.F, 6.F}, fraction, theme::accent);
        ImGui::Text(
            "%s %d / %d", tr("Generating masks"), completed, total);
        if (ImGui::Button(tr("Cancel")) && state.gen_cancel)
            state.gen_cancel->store(true);
    }
    if (state.mask_ready) {
        theme::metric_coloured(
            tr("Alignment masks"), tr("Reuse edits"), theme::success);
        if (ImGui::Button(tr("Allow Alignment to Regenerate"))) {
            std::error_code error;
            std::filesystem::remove(
                state.mask_dir / k_mask_ready_marker, error);
            state.mask_ready = false;
        }
    } else {
        theme::metric(
            tr("Alignment masks"), tr("Regenerate from prompt"));
    }
    if (!state.gen_error.empty())
        ImGui::TextColored(
            theme::danger, "%s", state.gen_error.c_str());
    if (!state.gen_note.empty())
        ImGui::TextWrapped("%s", state.gen_note.c_str());

    if (ImGui::CollapsingHeader(tr("Advanced SAM settings"))) {
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderFloat(
            "##mask_sam_threshold", &settings.sam_threshold, 0.F, 1.F,
            tr("score %.2f"));
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderFloat(
            "##mask_sam_guided", &settings.sam_guided_threshold, 0.F, 1.F,
            tr("guided score %.2f"));
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderFloat(
            "##mask_sam_nms", &settings.sam_nms, 0.F, 1.F, tr("NMS %.2f"));
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderFloat(
            "##mask_sam_min_area", &settings.sam_min_area, 0.F, 0.5F,
            tr("min area %.3f"));
        ImGui::SetNextItemWidth(-1.F);
        ImGui::SliderFloat(
            "##mask_sam_max_area", &settings.sam_max_area, 0.01F, 1.F,
            tr("max area %.3f"));
        if (settings.sam_min_area >= settings.sam_max_area)
            ImGui::TextColored(
                theme::danger, "%s",
                tr("Minimum area must be below maximum area."));
        const int kernels[] = {1, 3, 5, 7, 9, 11, 15};
        int kernel_index = 2;
        for (int i = 0; i < 7; ++i)
            if (kernels[i] == settings.sam_close_kernel) kernel_index = i;
        ImGui::SetNextItemWidth(-1.F);
        if (ImGui::Combo(
                "##mask_sam_close", &kernel_index,
                [](void*, int index, const char** label) {
                    static const char* names[] = {
                        "close 1", "close 3", "close 5", "close 7",
                        "close 9", "close 11", "close 15"};
                    *label = names[std::clamp(index, 0, 6)];
                    return true;
                },
                nullptr, 7))
            settings.sam_close_kernel = kernels[kernel_index];
        ImGui::Checkbox(tr("Fill holes"), &settings.sam_fill_holes);
        ImGui::TextDisabled("%s", tr(
            "Area, closing and fill-holes cleanup apply when an alignment scene "
            "guides the prompt. Otherwise use Fill Holes above."));
    }
    if (!state.load_error.empty())
        ImGui::TextColored(
            theme::danger, "%s", state.load_error.c_str());
}

void draw_file_list(App& app) {
    auto& state = app.mask_editor;
    if (state.images.empty()) {
        if (is_video_source(app.settings)) {
            ImGui::TextWrapped("%s", tr(
                "Align Photos extracts video frames and generates masks automatically. Open Video Preview only when you need to inspect or select a clip."));
            if (ImGui::Button(tr("Open Video Preview...")))
                open_video_window(app);
        } else {
            ImGui::TextWrapped("%s", tr(
                "No frames found. Extract video frames or choose an image folder "
                "first."));
        }
        return;
    }
    theme::caption(tr("Frame list"));
    ImGui::TextDisabled(
        "%zu %s  %s", state.images.size(), tr("frames"),
        path_to_utf8(state.mask_dir).c_str());
    ImGui::BeginChild("##mask_file_list", {0.F, 0.F});
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(state.images.size()));
    while (clipper.Step()) {
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const bool selected = i == state.selected;
            const std::string label = path_to_utf8(
                state.images[static_cast<std::size_t>(i)].filename());
            if (ImGui::Selectable(label.c_str(), selected))
                request_select(app, i);
        }
    }
    ImGui::EndChild();
}

}  // namespace

void open_mask_editor(App& app) {
    app.mask_editor.show = true;
    app.mask_editor.raise = true;
    refresh_image_list(app, true);
}

void poll_mask_editor(App& app) {
    auto& state = app.mask_editor;
    // Finish the current decode before following a newer generated mask.
    poll_image_load(app);
    const bool aligning = app.job.running() && app.active_job == JobKind::align;
    if (aligning && app.settings.sam_masks && !state.follow_alignment &&
        (app.monitor.stage() == Stage::masking || app.monitor.capture_frame().kind == "mask")) {
        state.follow_alignment = true;
        state.show = true;
        state.raise = true;
        refresh_image_list(app, false);
        if (!state.dirty) state.view_mode = 0;
    }
    const auto& live = app.monitor.mask_frame();
    if (state.show && state.follow_alignment && live.kind == "mask" &&
        live.revision != state.alignment_revision && !state.dirty && !state.generating &&
        !state.pending_load.valid()) {
        refresh_image_list(app, false);
        const auto path = path_from_utf8_field(live.image.c_str()).lexically_normal();
        const auto found = std::find_if(state.images.begin(), state.images.end(),
            [&](const auto& image) { return image.lexically_normal() == path; });
        if (found != state.images.end()) {
            apply_select(app, static_cast<int>(found - state.images.begin()));
            state.alignment_revision = live.revision;
        }
    }
    if (!aligning) state.follow_alignment = false;
    poll_generation(app);
    if (state.undo_requested) {
        state.undo_requested = false;
        if (!state.undo.empty() && state.loaded) {
            state.redo.push_back(state.mask.pixels);
            state.mask.pixels = std::move(state.undo.back());
            state.undo.pop_back();
            state.dirty = true;
            state.texture_dirty = true;
        }
    }
    if (state.redo_requested) {
        state.redo_requested = false;
        if (!state.redo.empty() && state.loaded) {
            state.undo.push_back(state.mask.pixels);
            state.mask.pixels = std::move(state.redo.back());
            state.redo.pop_back();
            state.dirty = true;
            state.texture_dirty = true;
        }
    }
}

void draw_mask_editor(App& app) {
    auto& state = app.mask_editor;
    if (!state.show) return;

    refresh_image_list(app, false);

    if (state.raise) {
        ImGui::SetNextWindowFocus();
        state.raise = false;
    }
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        viewport->GetCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5F, 0.5F));
    ImGui::SetNextWindowSize({1220.F, 820.F}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(860.F, 560.F), ImVec2(FLT_MAX, FLT_MAX));
    if (!ImGui::Begin(
            i18n::id("Mask Editor", "###MaskEditor"), &state.show,
            ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }

    if (!app.job.running() && ImGui::GetIO().WantTextInput == false &&
        ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        const ImGuiIO& io = ImGui::GetIO();
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z))
            state.undo_requested = !io.KeyShift;
        if (io.KeyCtrl && (ImGui::IsKeyPressed(ImGuiKey_Y) ||
                           (io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_Z))))
            state.redo_requested = true;
        if (!io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_LeftArrow))
            request_select(app, state.selected - 1);
        if (!io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_RightArrow))
            request_select(app, state.selected + 1);
    }

    if (state.texture_dirty && state.loaded) rebuild_overlay(state);
    draw_toolbar(app);
    draw_view_controls(state);
    if (app.job.running() && app.active_job == JobKind::align &&
        state.follow_alignment && app.monitor.stage() == Stage::masking) {
        const auto& live = app.monitor.mask_frame();
        {
            const auto completed = live.kind == "mask" ? live.completed : app.monitor.task().completed;
            const auto total = live.kind == "mask" ? live.total : app.monitor.task().total;
            ImGui::Text("%s %llu / %llu", tr("Generating masks"),
                static_cast<unsigned long long>(completed),
                static_cast<unsigned long long>(total));
            theme::progress_track({-1.F, 6.F}, total > 0
                ? static_cast<float>(completed) / total : -1.F, theme::accent);
        }
    }
    ImGui::Separator();

    const float list_height = 132.F;
    const float tools_width = 350.F;
    ImGui::BeginChild(
        "##mask_workspace", {0.F, -list_height}, ImGuiChildFlags_None,
        ImGuiWindowFlags_NoScrollbar);
    const float canvas_width = std::max(
        240.F, ImGui::GetContentRegionAvail().x - tools_width - 8.F);
    ImGui::BeginChild(
        "##mask_canvas_wrap", {canvas_width, 0.F}, ImGuiChildFlags_None,
        ImGuiWindowFlags_NoScrollbar);
    draw_canvas(app);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild(
        "##mask_tools", {0.F, 0.F}, ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::BeginDisabled(app.job.running());
    draw_tools(app);
    ImGui::EndDisabled();
    ImGui::EndChild();
    ImGui::EndChild();

    ImGui::BeginChild("##mask_files_wrap", {0.F, 0.F});
    draw_file_list(app);
    ImGui::EndChild();

    const std::string discard_id =
        i18n::id("Discard mask changes?", "###MaskDiscardChanges");
    if (state.confirm_discard) {
        ImGui::OpenPopup(discard_id.c_str());
        state.confirm_discard = false;
    }
    if (ImGui::BeginPopupModal(
            discard_id.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("%s", tr(
            "The current mask has unsaved changes. Save before switching?"));
        if (ImGui::Button(tr("Save and Continue"), {140.F, 0.F})) {
            const int target = state.pending_select;
            if (save_current_mask(app)) {
                ImGui::CloseCurrentPopup();
                state.pending_select = -1;
                apply_select(app, target);
            }
        }
        ImGui::SameLine();
        if (ImGui::Button(tr("Discard"), {100.F, 0.F})) {
            const int target = state.pending_select;
            ImGui::CloseCurrentPopup();
            state.pending_select = -1;
            state.dirty = false;
            apply_select(app, target);
        }
        ImGui::SameLine();
        if (ImGui::Button(tr("Cancel"), {80.F, 0.F})) {
            state.pending_select = -1;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    const std::string generate_id =
        i18n::id("Regenerate all masks?", "###MaskGenerateAll");
    if (state.confirm_generate_all) {
        ImGui::OpenPopup(generate_id.c_str());
        state.confirm_generate_all = false;
    }
    if (ImGui::BeginPopupModal(
            generate_id.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextWrapped("%s", tr(
            "Generating all masks replaces every mask in the mask folder, "
            "including manual edits. Continue?"));
        ImGui::TextDisabled("%s", path_to_utf8(state.mask_dir).c_str());
        if (theme::primary_button(tr("Regenerate All"), {140.F, 0.F})) {
            ImGui::CloseCurrentPopup();
            start_generation(app, true);
        }
        ImGui::SameLine();
        if (ImGui::Button(tr("Cancel"), {80.F, 0.F}))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    ImGui::End();
}

void shutdown_mask_editor(App& app) {
    auto& state = app.mask_editor;
    if (state.gen_job.valid()) {
        if (state.gen_cancel) state.gen_cancel->store(true);
        try {
            (void)state.gen_job.get();
        } catch (...) {
        }
        state.gen_job = {};
    }
    std::error_code error;
    if (!state.gen_temp_dir.empty())
        std::filesystem::remove_all(state.gen_temp_dir, error);
    state.gen_temp_dir.clear();
    if (state.pending_load.valid()) {
        try {
            (void)state.pending_load.get();
        } catch (...) {
        }
    }
    for (auto& pending : state.retired_loads) {
        try {
            (void)pending.get();
        } catch (...) {
        }
    }
    state.retired_loads.clear();
    state.image_texture.reset();
    state.overlay_texture.reset();
}

}  // namespace editor
