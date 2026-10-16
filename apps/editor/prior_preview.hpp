#pragma once
#include "prior_preview_data.hpp"
#include "vulkan_backend.hpp"
#include <future>
#include <string>

namespace editor {
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
    void show(const std::filesystem::path& folder);
    void draw();
    void reset_textures();
};
} // namespace editor
