#pragma once
#include "prior_preview_data.hpp"
#include "vulkan_backend.hpp"
#include <future>
#include <chrono>
#include <string>
#include <optional>

namespace editor {
// Cached AI maps for the selected photo in the 2D viewport.
struct ImageQaPriors {
    struct Result {
        std::filesystem::path directory, source, file;
        std::filesystem::file_time_type stamp{};
        std::filesystem::file_time_type source_stamp{};
        std::optional<PriorPreviewData> data;
        bool unchanged{};
        std::string error;
    };
    std::filesystem::path directory, source, file;
    std::filesystem::file_time_type stamp{};
    std::filesystem::file_time_type source_stamp{};
    std::future<Result> pending;
    std::chrono::steady_clock::time_point next_check{};
    gpu::PreviewTexture depth, normal;
    PriorPreviewData data;
    float displayed_near{}, displayed_far{};
    bool displayed_grayscale{};
    std::string error;
    bool refresh_requested{};
    void update(const std::filesystem::path& folder, const std::filesystem::path& image);
    void reset_textures();
    void set_depth_display(bool automatic, bool grayscale, float near_depth, float far_depth);
    void request_refresh() { refresh_requested = true; }
};
} // namespace editor
