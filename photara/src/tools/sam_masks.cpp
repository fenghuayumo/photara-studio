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

// The checkpoint runs at 1008. Passing the full photo makes SAM resize every
// detection mask back to that full size on the CPU.
io::RgbImage fit_long_side(io::RgbImage image, const int max_side) {
    const int width = static_cast<int>(image.width);
    const int height = static_cast<int>(image.height);
    const int long_side = std::max(width, height);
    if (long_side <= max_side || width <= 0 || height <= 0) return image;
    const int fitted_width = std::max(
        1, static_cast<int>(
            (static_cast<std::int64_t>(width) * max_side + long_side / 2) /
            long_side));
    const int fitted_height = std::max(
        1, static_cast<int>(
            (static_cast<std::int64_t>(height) * max_side + long_side / 2) /
            long_side));
    io::RgbImage fitted;
    fitted.width = static_cast<std::uint32_t>(fitted_width);
    fitted.height = static_cast<std::uint32_t>(fitted_height);
    fitted.pixels.resize(
        static_cast<std::size_t>(fitted_width) * fitted_height * 3U);
    io::resize_bilinear(
        image.pixels.data(), image.width, image.height, 3,
        fitted.pixels.data(), fitted.width, fitted.height);
    return fitted;
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
    if (options.backend == "auto")
        params.backend = SAM3_BACKEND_AUTO;
    else if (options.backend == "cpu")
        params.backend = SAM3_BACKEND_CPU;
    else if (options.backend == "cuda")
        params.backend = SAM3_BACKEND_CUDA;
    else if (options.backend == "vulkan")
        params.backend = SAM3_BACKEND_VULKAN;
    else if (options.backend == "metal")
        params.backend = SAM3_BACKEND_METAL;
    else
        throw std::invalid_argument(
            "SAM backend must be auto, cpu, cuda, vulkan, or metal");
    params.n_threads = 4;
    params.encode_img_size = std::max(0, options.max_size);
    std::shared_ptr<sam3_model> model = sam3_load_model(params);
    if (!model)
        throw std::runtime_error(
            "Failed to load the SAM 3 checkpoint with backend \"" +
            options.backend + "\"");
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
        // JPEG reads used by the rest of the pipeline may be reduced. The
        // mask has to match the file's stored pixel size.
        const io::ImageSize file_size = io::load_image_size(image_path);
        const std::uint32_t file_long_side =
            std::max(file_size.width, file_size.height);
        // Decode at sufficient resolution for the checkpoint's native input.
        // In particular, do not select libjpeg's 1/2 DCT path for 1920-wide
        // inputs: its small speed gain measurably moves mask boundaries.
        constexpr std::uint32_t jpeg_decode_side = 1008U;
        const std::uint32_t decode_width = file_long_side <= jpeg_decode_side
            ? file_size.width
            : std::max(
                  1U, static_cast<std::uint32_t>(
                          (static_cast<std::uint64_t>(file_size.width) *
                               jpeg_decode_side +
                           file_long_side - 1U) /
                          file_long_side));
        const std::uint32_t decode_height = file_long_side <= jpeg_decode_side
            ? file_size.height
            : std::max(
                  1U, static_cast<std::uint32_t>(
                          (static_cast<std::uint64_t>(file_size.height) *
                               jpeg_decode_side +
                           file_long_side - 1U) /
                          file_long_side));
        io::RgbImage rgb = fit_long_side(
            io::load_rgb_with_minimum_size(
                image_path, decode_width, decode_height),
            1008);
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

        const int output_width = static_cast<int>(file_size.width);
        const int output_height = static_cast<int>(file_size.height);
        io::GrayImage gray;
        gray.width = file_size.width;
        gray.height = file_size.height;
        gray.pixels.resize(
            static_cast<std::size_t>(output_width) * output_height);
        for (int y = 0; y < output_height; ++y) {
            const int source_y = y * height / output_height;
            const auto* row =
                selected.data() + static_cast<std::size_t>(source_y) * width;
            auto* destination = gray.pixels.data() +
                static_cast<std::size_t>(y) * output_width;
            for (int x = 0; x < output_width; ++x) {
                const bool hit = row[x * width / output_width] != 0;
                destination[x] = options.keep_prompted
                    ? static_cast<std::uint8_t>(hit ? 255 : 0)
                    : static_cast<std::uint8_t>(hit ? 0 : 255);
            }
        }
        io::save_gray_png(
            gray, options.output_dir / (image_path.stem().string() + ".png"));
        ++result.written;
        progress.advance();
    }
    core::Logger::instance().info(
        "sam_masks=", result.written, " dir=", options.output_dir,
        " model=", model_path, " keep_prompted=", options.keep_prompted,
        " video=", options.video,
        " backend=", sam3_model_backend_name(*model));
    trackers.clear();
    state.reset();
    model.reset();
    return result;
#endif
}

}  // namespace photara::sam
