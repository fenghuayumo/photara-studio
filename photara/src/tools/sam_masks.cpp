#include "sam/masks.hpp"

#include "core/logging.hpp"
#include "io/image.hpp"
#include "sam/model_cache.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(PHOTARA_HAS_SAM)
#include "sam3.h"
#endif

namespace photara::sam {
namespace {

std::vector<std::string> split_phrases(const std::string& text) {
    std::vector<std::string> phrases;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = text.find(';', start);
        auto phrase = text.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        const auto first = phrase.find_first_not_of(" \t");
        const auto last = phrase.find_last_not_of(" \t");
        if (first != std::string::npos)
            phrases.push_back(phrase.substr(first, last - first + 1));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return phrases;
}

#if defined(PHOTARA_HAS_SAM)
void paint_phrase(
    std::vector<std::uint8_t>& selected, const int width, const int height,
    const sam3_mask& mask, const bool on) {
    if (mask.width <= 0 || mask.height <= 0 || mask.data.empty()) return;
    for (int y = 0; y < height; ++y) {
        const int source_y = std::min(
            mask.height - 1,
            static_cast<int>(static_cast<std::int64_t>(y) * mask.height / height));
        for (int x = 0; x < width; ++x) {
            const int source_x = std::min(
                mask.width - 1,
                static_cast<int>(
                    static_cast<std::int64_t>(x) * mask.width / width));
            if (mask.data[static_cast<std::size_t>(source_y) * mask.width +
                          source_x] == 0)
                continue;
            selected[static_cast<std::size_t>(y) * width + x] = on ? 1 : 0;
        }
    }
}

void paint_result(
    std::vector<std::uint8_t>& selected, const int width, const int height,
    const sam3_result& result, const bool on) {
    for (const sam3_detection& detection : result.detections)
        paint_phrase(selected, width, height, detection.mask, on);
}

sam3_image image_from_rgb(io::RgbImage rgb) {
    sam3_image image;
    image.width = static_cast<int>(rgb.width);
    image.height = static_cast<int>(rgb.height);
    image.channels = 3;
    image.data = std::move(rgb.pixels);
    return image;
}
#endif

}  // namespace

bool built_with_sam() {
#if defined(PHOTARA_HAS_SAM)
    return true;
#else
    return false;
#endif
}

GenerateResult generate_masks(const GenerateOptions& options) {
#if !defined(PHOTARA_HAS_SAM)
    (void)options;
    throw std::runtime_error(
        "SAM mask generation was not built. Enable PHOTARA_ENABLE_SAM");
#else
    if (options.images.empty())
        throw std::invalid_argument("SAM mask generation needs images");
    if (options.text.empty())
        throw std::invalid_argument(
            "SAM mask generation needs a text prompt (--sam-text)");
    if (options.output_dir.empty())
        throw std::invalid_argument("SAM mask generation needs an output directory");
    const auto positive = split_phrases(options.text);
    const auto negative = split_phrases(options.negative_text);
    if (positive.empty())
        throw std::invalid_argument("SAM mask generation needs a text prompt");
    const auto model_path = options.model.empty() ? locate_model() : options.model;
    if (!model_file_ready(model_path))
        throw std::runtime_error(
            "SAM 3 checkpoint not found. Accept the SAM 3 licence in Photara "
            "Studio and download the model, or pass --sam-model");

    sam3_params params;
    params.model_path = model_path.string();
    params.use_gpu = true;
    params.n_threads = 4;
    params.encode_img_size = std::max(0, options.max_size);
    std::shared_ptr<sam3_model> model = sam3_load_model(params);
    if (!model)
        throw std::runtime_error("Failed to load the SAM 3 checkpoint");
    sam3_state_ptr state = sam3_create_state(*model, params);
    if (!state)
        throw std::runtime_error("Failed to create the SAM 3 inference state");

    std::vector<sam3_tracker_ptr> trackers;
    if (options.video) {
        trackers.reserve(positive.size());
        for (const std::string& phrase : positive) {
            sam3_video_params video;
            video.text_prompt = phrase;
            video.score_threshold = options.threshold;
            video.nms_threshold = options.nms;
            sam3_tracker_ptr tracker = sam3_create_tracker(*model, video);
            if (!tracker)
                throw std::runtime_error(
                    "Failed to start SAM 3 tracking for \"" + phrase + "\"");
            trackers.push_back(std::move(tracker));
        }
    }

    std::error_code filesystem_error;
    std::filesystem::create_directories(options.output_dir, filesystem_error);
    if (filesystem_error)
        throw std::runtime_error(
            "Cannot create mask directory: " + options.output_dir.string());

    core::ProgressReporter progress("generate sam masks", options.images.size());
    GenerateResult result;
    for (const auto& image_path : options.images) {
        io::RgbImage rgb = io::load_rgb(image_path);
        const int width = static_cast<int>(rgb.width);
        const int height = static_cast<int>(rgb.height);
        sam3_image frame = image_from_rgb(std::move(rgb));
        std::vector<std::uint8_t> selected(
            static_cast<std::size_t>(width) * height, 0);
        if (options.video) {
            for (sam3_tracker_ptr& tracker : trackers) {
                const sam3_result tracked =
                    sam3_track_frame(*tracker, *state, *model, frame);
                paint_result(selected, width, height, tracked, true);
            }
        } else if (!sam3_encode_image(*state, *model, frame)) {
            throw std::runtime_error(
                "SAM 3 failed to encode " + image_path.filename().string());
        } else {
            for (const std::string& phrase : positive) {
                sam3_pcs_params prompt;
                prompt.text_prompt = phrase;
                prompt.score_threshold = options.threshold;
                prompt.nms_threshold = options.nms;
                paint_result(
                    selected, width, height,
                    sam3_segment_pcs(*state, *model, prompt), true);
            }
        }
        for (const std::string& phrase : negative) {
            sam3_pcs_params prompt;
            prompt.text_prompt = phrase;
            prompt.score_threshold = options.threshold;
            prompt.nms_threshold = options.nms;
            paint_result(
                selected, width, height,
                sam3_segment_pcs(*state, *model, prompt), false);
        }

        io::GrayImage gray;
        gray.width = static_cast<std::uint32_t>(width);
        gray.height = static_cast<std::uint32_t>(height);
        gray.pixels.resize(selected.size());
        for (std::size_t index = 0; index < selected.size(); ++index) {
            const bool hit = selected[index] != 0;
            gray.pixels[index] = options.keep_prompted
                ? static_cast<std::uint8_t>(hit ? 255 : 0)
                : static_cast<std::uint8_t>(hit ? 0 : 255);
        }
        io::save_gray_png(
            gray, options.output_dir / (image_path.stem().string() + ".png"));
        ++result.written;
        progress.advance();
    }
    core::Logger::instance().info(
        "sam_masks=", result.written, " dir=", options.output_dir,
        " model=", model_path, " keep_prompted=", options.keep_prompted,
        " video=", options.video, " gpu=", params.use_gpu);
    trackers.clear();
    state.reset();
    model.reset();
    return result;
#endif
}

}  // namespace photara::sam
