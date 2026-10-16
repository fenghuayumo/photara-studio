#pragma once
#include "prior_preview_data.hpp"
#include "vulkan_backend.hpp"
#include <future>
#include <chrono>
#include <string>
#include <optional>

namespace editor {
// Independent of the floating preview: selecting a photo in either viewer
// must not change the other viewer's textures or selected cache.
struct ImageQaPriors {
    struct Result {
        std::filesystem::path directory, source, file;
        std::filesystem::file_time_type stamp{};
        std::filesystem::file_time_type source_stamp{};
        std::optional<PriorPreviewData> data;
        bool unchanged{};
    };
    std::filesystem::path directory, source, file;
    std::filesystem::file_time_type stamp{};
    std::filesystem::file_time_type source_stamp{};
    std::future<Result> pending;
    std::chrono::steady_clock::time_point next_check{};
    gpu::PreviewTexture depth, normal;
    void update(const std::filesystem::path& folder, const std::filesystem::path& image);
    void reset_textures();
};
struct PriorPreview {
    bool open{};
    int selected{-1};
    std::filesystem::path directory;
    std::filesystem::path requested_directory;
    std::vector<std::filesystem::path> files;
    std::future<std::vector<std::filesystem::path>> pending_files;
    std::future<PriorPreviewData> pending;
    PriorPreviewData data;
    gpu::PreviewTexture photo,depth,normal;
    std::string error;
    float depth_range[2]{};
    float displayed_range[2]{};
    bool manual_range{};
    std::filesystem::path availability_directory,checked_directory;
    std::future<bool> pending_availability;
    std::chrono::steady_clock::time_point next_availability_check{};
    bool cache_available{};
    bool available(const std::filesystem::path& folder);
    void show(const std::filesystem::path& folder);
    void draw();
    void reset_textures();
};
} // namespace editor
