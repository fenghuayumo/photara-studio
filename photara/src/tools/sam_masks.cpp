#include "sam/masks.hpp"

#include "core/logging.hpp"
#include "io/image.hpp"
#include "sam/model_cache.hpp"
#include "sfm/scene.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
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
struct GeometryGuide {
    std::vector<std::pair<float, float>> points;
    float center_x{};
    float center_y{};
    sam3_box box{};
};

std::string image_key(const std::filesystem::path& path) {
    std::string key = path.filename().string();
    std::transform(key.begin(), key.end(), key.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return key;
}

float median(std::vector<float> values) {
    if (values.empty()) return 0.F;
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    float result = values[middle];
    if (values.size() % 2 == 0) {
        const auto lower = std::max_element(values.begin(), values.begin() + middle);
        result = 0.5F * (result + *lower);
    }
    return result;
}

float quantile(std::vector<float> values, const float q) {
    if (values.empty()) return 0.F;
    const std::size_t index = std::min(
        values.size() - 1,
        static_cast<std::size_t>(q * static_cast<float>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

std::unordered_map<std::string, GeometryGuide> make_geometry_guides(
    const sfm::Scene& scene, const float central_fraction) {
    std::unordered_map<std::string, GeometryGuide> guides;
    sfm::Vec3 camera_center = sfm::Vec3::Zero();
    std::size_t registered = 0;
    for (const sfm::Image& image : scene.images) {
        if (!image.registered) continue;
        camera_center += image.pose.C;
        ++registered;
    }
    if (registered < 2) return guides;
    camera_center /= static_cast<double>(registered);

    struct RankedPoint {
        double distance{};
        sfm::Vec3 position{sfm::Vec3::Zero()};
    };
    std::vector<RankedPoint> ranked;
    ranked.reserve(scene.tracks.size());
    for (const sfm::Track& track : scene.tracks) {
        if (!track.is_triangulated() || !track.position.allFinite()) continue;
        ranked.push_back({(track.position - camera_center).norm(), track.position});
    }
    if (ranked.size() < 8) return guides;
    const float fraction = std::clamp(central_fraction, 0.0001F, 1.F);
    const std::size_t requested = static_cast<std::size_t>(
        std::ceil(static_cast<double>(ranked.size()) * fraction));
    const std::size_t selected_count = std::min(
        ranked.size(), std::max<std::size_t>(8, requested));
    std::nth_element(
        ranked.begin(), ranked.begin() + selected_count - 1, ranked.end(),
        [](const RankedPoint& left, const RankedPoint& right) {
            return left.distance < right.distance;
        });
    ranked.resize(selected_count);

    for (const sfm::Image& image : scene.images) {
        if (!image.registered || image.camera_id >= scene.cameras.size()) continue;
        const sfm::PinholeCamera& camera = scene.camera_of(image);
        // A rectangular quantile box is ambiguous across a panorama seam.
        // Fall back to the existing text/video path for those views.
        if (camera.width == 0 || camera.height == 0 ||
            camera.is_equirectangular())
            continue;
        GeometryGuide guide;
        guide.points.reserve(ranked.size());
        std::vector<float> xs;
        std::vector<float> ys;
        for (const RankedPoint& point : ranked) {
            sfm::Vec2 pixel;
            if (!camera.project_checked(
                    image.pose.transform_world_to_camera(point.position), pixel))
                continue;
            const float x = static_cast<float>(pixel.x() / camera.width);
            const float y = static_cast<float>(pixel.y() / camera.height);
            if (!(x >= 0.F && x < 1.F && y >= 0.F && y < 1.F)) continue;
            guide.points.emplace_back(x, y);
            xs.push_back(x);
            ys.push_back(y);
        }
        if (guide.points.size() < 3) continue;
        guide.center_x = median(xs);
        guide.center_y = median(ys);
        float x0 = quantile(xs, 0.05F);
        float y0 = quantile(ys, 0.05F);
        float x1 = quantile(xs, 0.95F);
        float y1 = quantile(ys, 0.95F);
        const float pad_x = std::max(0.02F, 0.15F * (x1 - x0));
        const float pad_y = std::max(0.02F, 0.15F * (y1 - y0));
        guide.box = {
            std::max(0.F, x0 - pad_x), std::max(0.F, y0 - pad_y),
            std::min(1.F, x1 + pad_x), std::min(1.F, y1 + pad_y)};
        guides.emplace(image_key(image.path), std::move(guide));
    }
    return guides;
}

void morphology_pass(
    const std::vector<std::uint8_t>& source, std::vector<std::uint8_t>& output,
    const int width, const int height, const int radius, const bool horizontal,
    const bool dilate) {
    output.assign(source.size(), dilate ? 0 : 1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::uint8_t value = dilate ? 0 : 1;
            for (int offset = -radius; offset <= radius; ++offset) {
                const int sx = horizontal ? x + offset : x;
                const int sy = horizontal ? y : y + offset;
                const std::uint8_t sample =
                    sx >= 0 && sx < width && sy >= 0 && sy < height
                    ? source[static_cast<std::size_t>(sy) * width + sx]
                    : static_cast<std::uint8_t>(dilate ? 0 : 1);
                value = dilate ? static_cast<std::uint8_t>(value | sample)
                               : static_cast<std::uint8_t>(value & sample);
                if ((dilate && value) || (!dilate && !value)) break;
            }
            output[static_cast<std::size_t>(y) * width + x] = value;
        }
    }
}

std::vector<std::uint8_t> clean_mask(
    const sam3_mask& mask, const int close_kernel, const bool fill_holes) {
    if (mask.width <= 0 || mask.height <= 0 || mask.data.empty()) return {};
    const std::size_t pixel_count =
        static_cast<std::size_t>(mask.width) * mask.height;
    if (mask.data.size() < pixel_count) return {};
    std::vector<std::uint8_t> binary(pixel_count);
    std::transform(
        mask.data.begin(), mask.data.begin() + pixel_count, binary.begin(),
        [](const std::uint8_t value) { return value ? 1 : 0; });
    if (close_kernel > 1) {
        const int radius = close_kernel / 2;
        std::vector<std::uint8_t> a, b;
        morphology_pass(binary, a, mask.width, mask.height, radius, true, true);
        morphology_pass(a, b, mask.width, mask.height, radius, false, true);
        morphology_pass(b, a, mask.width, mask.height, radius, true, false);
        morphology_pass(a, binary, mask.width, mask.height, radius, false, false);
    }

    const std::size_t count = binary.size();
    std::vector<int> labels(count, -1);
    std::vector<std::size_t> component_sizes;
    std::deque<int> queue;
    const int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    for (int start = 0; start < static_cast<int>(count); ++start) {
        if (!binary[start] || labels[start] >= 0) continue;
        const int label = static_cast<int>(component_sizes.size());
        std::size_t size = 0;
        labels[start] = label;
        queue.push_back(start);
        while (!queue.empty()) {
            const int current = queue.front();
            queue.pop_front();
            ++size;
            const int x = current % mask.width;
            const int y = current / mask.width;
            for (int direction = 0; direction < 8; ++direction) {
                const int nx = x + dx[direction];
                const int ny = y + dy[direction];
                if (nx < 0 || nx >= mask.width || ny < 0 || ny >= mask.height)
                    continue;
                const int next = ny * mask.width + nx;
                if (!binary[next] || labels[next] >= 0) continue;
                labels[next] = label;
                queue.push_back(next);
            }
        }
        component_sizes.push_back(size);
    }
    if (!component_sizes.empty()) {
        const int largest = static_cast<int>(std::distance(
            component_sizes.begin(),
            std::max_element(component_sizes.begin(), component_sizes.end())));
        for (std::size_t i = 0; i < count; ++i)
            binary[i] = labels[i] == largest ? 1 : 0;
    }
    if (fill_holes) {
        std::vector<std::uint8_t> exterior(count, 0);
        const auto seed = [&](const int x, const int y) {
            const int index = y * mask.width + x;
            if (!binary[index] && !exterior[index]) {
                exterior[index] = 1;
                queue.push_back(index);
            }
        };
        for (int x = 0; x < mask.width; ++x) {
            seed(x, 0);
            seed(x, mask.height - 1);
        }
        for (int y = 0; y < mask.height; ++y) {
            seed(0, y);
            seed(mask.width - 1, y);
        }
        while (!queue.empty()) {
            const int current = queue.front();
            queue.pop_front();
            const int x = current % mask.width;
            const int y = current / mask.width;
            for (int direction = 0; direction < 4; ++direction) {
                const int nx = x + dx[direction];
                const int ny = y + dy[direction];
                if (nx < 0 || nx >= mask.width || ny < 0 || ny >= mask.height)
                    continue;
                const int next = ny * mask.width + nx;
                if (binary[next] || exterior[next]) continue;
                exterior[next] = 1;
                queue.push_back(next);
            }
        }
        for (std::size_t i = 0; i < count; ++i)
            if (!binary[i] && !exterior[i]) binary[i] = 1;
    }
    return binary;
}

struct SelectedDetection {
    const sam3_detection* detection{};
    std::vector<std::uint8_t> mask;
};

std::optional<SelectedDetection> select_detection(
    const sam3_result& result, const GeometryGuide& guide,
    const GenerateOptions& options) {
    std::optional<SelectedDetection> best;
    float best_rank = -std::numeric_limits<float>::infinity();
    for (const sam3_detection& detection : result.detections) {
        auto mask = clean_mask(
            detection.mask, options.close_kernel, options.fill_holes);
        if (mask.empty()) continue;
        const int width = detection.mask.width;
        const int height = detection.mask.height;
        std::vector<float> xs;
        std::vector<float> ys;
        std::size_t area = 0;
        std::size_t edge = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (!mask[static_cast<std::size_t>(y) * width + x]) continue;
                ++area;
                xs.push_back(static_cast<float>(x));
                ys.push_back(static_cast<float>(y));
                if (x == 0 || y == 0 || x + 1 == width || y + 1 == height)
                    ++edge;
            }
        }
        const float area_fraction = static_cast<float>(area) /
            static_cast<float>(static_cast<std::size_t>(width) * height);
        if (area_fraction < options.min_area_fraction ||
            area_fraction > options.max_area_fraction || area == 0)
            continue;
        const float center_x = median(std::move(xs)) / width;
        const float center_y = median(std::move(ys)) / height;
        const float distance = std::hypot(
            center_x - guide.center_x, center_y - guide.center_y);
        if (distance > 0.35F * std::sqrt(2.F)) continue;
        std::size_t supported = 0;
        for (const auto& [x, y] : guide.points) {
            const int px = std::clamp(static_cast<int>(x * width), 0, width - 1);
            const int py = std::clamp(static_cast<int>(y * height), 0, height - 1);
            supported += mask[static_cast<std::size_t>(py) * width + px] != 0;
        }
        const float support = guide.points.empty() ? 0.F
            : static_cast<float>(supported) / guide.points.size();
        const float rank = detection.score + 0.8F * support -
            0.6F * distance / std::sqrt(2.F) - (edge > 10 ? 0.3F : 0.F);
        if (!best || rank > best_rank) {
            best_rank = rank;
            best = SelectedDetection{&detection, std::move(mask)};
        }
    }
    return best;
}

void paint_binary(
    std::vector<std::uint8_t>& selected, const int width, const int height,
    const std::vector<std::uint8_t>& mask, const int mask_width,
    const int mask_height, const bool on) {
    if (mask.empty() || mask_width <= 0 || mask_height <= 0) return;
    for (int y = 0; y < height; ++y) {
        const int source_y = std::min(mask_height - 1, y * mask_height / height);
        for (int x = 0; x < width; ++x) {
            const int source_x = std::min(mask_width - 1, x * mask_width / width);
            if (!mask[static_cast<std::size_t>(source_y) * mask_width + source_x])
                continue;
            selected[static_cast<std::size_t>(y) * width + x] = on ? 1 : 0;
        }
    }
}

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
    if (!(options.central_fraction > 0.F && options.central_fraction <= 1.F))
        throw std::invalid_argument("SAM central point fraction must be in (0, 1]");
    if (!(options.min_area_fraction >= 0.F &&
          options.min_area_fraction < options.max_area_fraction &&
          options.max_area_fraction <= 1.F))
        throw std::invalid_argument("SAM area fractions must satisfy 0 <= min < max <= 1");
    if (options.close_kernel < 1 || options.close_kernel % 2 == 0)
        throw std::invalid_argument("SAM close kernel must be a positive odd number");
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

    const auto geometry_guides = options.sparse_scene
        ? make_geometry_guides(*options.sparse_scene, options.central_fraction)
        : std::unordered_map<std::string, GeometryGuide>{};
    if (options.sparse_scene) {
        core::Logger::instance().info(
            "sam_sfm_guides=", geometry_guides.size(),
            " registered=", options.sparse_scene->registered_count(),
            " tracks=", options.sparse_scene->tracks.size(),
            " central_fraction=", options.central_fraction);
    }

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

    core::ProgressReporter progress(
        options.sparse_scene ? "refine sam masks" : "generate sam masks",
        options.images.size());
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
        const auto guide_it = geometry_guides.find(image_key(image_path));
        const GeometryGuide* guide =
            guide_it == geometry_guides.end() ? nullptr : &guide_it->second;
        if (guide) ++result.geometry_guided;
        if (options.video) {
            for (sam3_tracker_ptr& tracker : trackers) {
                const sam3_result tracked =
                    sam3_track_frame(*tracker, *state, *model, frame);
                if (guide) {
                    const auto chosen = select_detection(tracked, *guide, options);
                    if (chosen)
                        paint_binary(
                            selected, width, height, chosen->mask,
                            chosen->detection->mask.width,
                            chosen->detection->mask.height, true);
                    else
                        paint_result(selected, width, height, tracked, true);
                } else {
                    paint_result(selected, width, height, tracked, true);
                }
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
                if (guide) prompt.pos_exemplars.push_back(guide->box);
                const sam3_result segmented =
                    sam3_segment_pcs(*state, *model, prompt);
                if (guide) {
                    const auto chosen = select_detection(segmented, *guide, options);
                    if (chosen)
                        paint_binary(
                            selected, width, height, chosen->mask,
                            chosen->detection->mask.width,
                            chosen->detection->mask.height, true);
                    else
                        paint_result(selected, width, height, segmented, true);
                } else {
                    paint_result(selected, width, height, segmented, true);
                }
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

        if (options.keep_prompted &&
            std::none_of(selected.begin(), selected.end(),
                         [](const std::uint8_t value) { return value != 0; })) {
            ++result.empty;
            core::Logger::instance().warning(
                "sam_mask_empty image=", image_path.filename(),
                " geometry_guided=", guide != nullptr,
                " threshold=", options.threshold);
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
        " geometry_guided=", result.geometry_guided,
        " empty=", result.empty,
        " threshold=", options.threshold,
        " nms=", options.nms,
        " backend=", sam3_model_backend_name(*model));
    trackers.clear();
    state.reset();
    model.reset();
    return result;
#endif
}

}  // namespace photara::sam
