#include "prior_preview.hpp"
#include <chrono>

namespace editor {
void ImageQaPriors::reset_textures() {
    depth.reset(); normal.reset(); file.clear(); stamp = {}; source_stamp = {};
    data = {}; displayed_near = displayed_far = 0.F;
}
void ImageQaPriors::update(const std::filesystem::path& folder, const std::filesystem::path& image) {
    const auto now = std::chrono::steady_clock::now();
    if (directory != folder || source != image) {
        directory = folder; source = image; next_check = {};
        error.clear();
        reset_textures();
    }
    if (pending.valid() && pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        auto result = pending.get();
        if (result.directory == directory && result.source == source) {
            error = result.error;
            if (!result.unchanged) {
                reset_textures();
                if (result.data) {
                    data = std::move(*result.data);
                    depth.upload(data.depth); normal.upload(data.normal);
                    displayed_near = data.depth_near; displayed_far = data.depth_far;
                    displayed_grayscale = false;
                    file = result.file; stamp = result.stamp; source_stamp = result.source_stamp;
                }
            }
            next_check = now + std::chrono::seconds(1);
        }
    }
    if (folder.empty() || image.empty() || pending.valid() || (now < next_check && !refresh_requested)) return;
    const bool force = refresh_requested;
    refresh_requested = false;
    pending = std::async(std::launch::async, [folder, image, previous = file, previous_stamp = stamp, force,
                                           previous_source_stamp = source_stamp] {
        Result result; result.directory = folder; result.source = image;
        try {
            result.file = find_prior_preview(folder, image);
            if (!result.file.empty()) {
                result.stamp = std::filesystem::last_write_time(result.file);
                result.source_stamp = std::filesystem::last_write_time(image);
                result.unchanged = !force && result.file == previous && result.stamp == previous_stamp &&
                    result.source_stamp == previous_source_stamp;
                if (!result.unchanged) result.data = load_prior_preview(result.file);
            }
        } catch (const std::exception& failure) {
            result.data.reset(); result.unchanged = false; result.error = failure.what();
        }
        return result;
    });
}
void ImageQaPriors::set_depth_display(bool automatic, bool grayscale, float near_depth, float far_depth) {
    if (data.prior.depth.empty()) return;
    if (automatic) { near_depth = data.depth_near; far_depth = data.depth_far; }
    if (near_depth == displayed_near && far_depth == displayed_far && grayscale == displayed_grayscale) return;
    depth.upload(colorize_prior_depth(data.prior, near_depth, far_depth, grayscale));
    displayed_near = near_depth; displayed_far = far_depth; displayed_grayscale = grayscale;
}
} // namespace editor
