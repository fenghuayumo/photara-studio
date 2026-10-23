#pragma once

#include "vulkan_backend.hpp"

#include "io/image.hpp"
#include "sam/masks.hpp"

#include "imgui.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace editor {

struct App;

struct MaskImageLoad {
    photara::io::RgbImage image;
    photara::io::GrayImage mask;
    bool has_mask{};
    std::filesystem::path image_path;
    std::filesystem::path mask_path;
};

struct MaskEditorState {
    bool show{};
    bool raise{};
    bool initialized{};

    std::filesystem::path images_dir;
    std::filesystem::path mask_dir;
    std::vector<std::filesystem::path> images;
    std::filesystem::file_time_type images_stamp{};
    int selected{-1};
    int pending_select{-1};
    bool confirm_discard{};
    int pending_index{-1};
    std::future<MaskImageLoad> pending_load;
    std::vector<std::future<MaskImageLoad>> retired_loads;
    std::string load_error;

    std::uint32_t image_width{};
    std::uint32_t image_height{};
    photara::io::GrayImage mask;
    photara::io::GrayImage saved_mask;
    bool has_mask{};
    bool loaded{};
    bool dirty{};

    gpu::PreviewTexture image_texture;
    gpu::PreviewTexture overlay_texture;
    std::uint32_t overlay_width{};
    std::uint32_t overlay_height{};
    bool texture_dirty{true};
    std::size_t kept_pixels{};
    std::size_t mask_pixels{};
    int view_mode{0};  // 0 overlay, 1 image, 2 mask
    float overlay_opacity{0.55F};

    float zoom{1.F};
    ImVec2 pan{};
    bool painting{};
    bool manual_editing{};
    bool stroke_active{};
    ImVec2 last_paint{};
    int brush_mode{0};  // 0 add, 1 erase
    float brush_radius{28.F};
    std::deque<std::vector<std::uint8_t>> undo;
    std::deque<std::vector<std::uint8_t>> redo;
    bool undo_requested{};
    bool redo_requested{};

    bool generating{};
    bool generate_all{};
    std::shared_ptr<std::atomic<bool>> gen_cancel;
    std::shared_ptr<std::atomic<int>> gen_completed;
    std::shared_ptr<std::atomic<int>> gen_total;
    std::future<photara::sam::GenerateResult> gen_job;
    std::filesystem::path gen_temp_dir;
    std::string gen_error;
    std::string gen_note;
    bool confirm_generate_all{};

    bool mask_ready{};
};

void open_mask_editor(App& app);
void draw_mask_editor(App& app);
void poll_mask_editor(App& app);
void shutdown_mask_editor(App& app);

}  // namespace editor
