#pragma once

#include "sparse_view.hpp"
#include "vulkan_backend.hpp"

#include "imgui.h"

#include <cstdint>
#include <filesystem>
#include <future>
#include <string>
#include <vector>

namespace editor {

enum class ImageQaMode { photo, features, compare, error };

struct ImageQaMetrics {
    bool valid{};
    float psnr{};
    float ssim{};
    float mae{};
    float rmse{};
};

struct ImageQaState {
    ImageQaMode mode{ImageQaMode::photo};
    int selected{-1};
    float wipe{0.5F};
    bool dragging_wipe{};
    float zoom{1.F};
    ImVec2 pan{};
    bool show_untracked{true};
    bool show_triangulated{true};
    int filmstrip_first{};
    int filmstrip_last{};
    bool filmstrip_reselect{};
    bool metrics_dirty{true};
    std::string folder_dir;
    std::vector<std::filesystem::path> folder_images;
};

class ImageQaSession {
public:
    ImageQaSession() = default;
    ImageQaSession(const ImageQaSession&) = delete;
    ImageQaSession& operator=(const ImageQaSession&) = delete;
    ~ImageQaSession();

    void clear();
    void poll();
    void request_gt(int view_index, const std::filesystem::path& path);
    void set_render(
        aetherscan::io::RgbImage render, int view, std::uint64_t revision);

    [[nodiscard]] bool loading() const { return loading_; }
    [[nodiscard]] bool metrics_busy() const { return metrics_busy_; }
    [[nodiscard]] bool has_gt() const;
    [[nodiscard]] bool has_error() const;
    [[nodiscard]] bool has_render_pixels() const;
    [[nodiscard]] int loaded_view() const { return loaded_view_; }
    [[nodiscard]] int render_view() const { return render_view_; }
    [[nodiscard]] std::uint64_t render_revision() const {
        return render_revision_;
    }
    [[nodiscard]] ImTextureID gt_id() const;
    [[nodiscard]] ImTextureID error_id() const;
    [[nodiscard]] const ImageQaMetrics& metrics() const { return metrics_; }
    [[nodiscard]] std::uint32_t gt_width() const { return gt_cpu_.width; }
    [[nodiscard]] std::uint32_t gt_height() const { return gt_cpu_.height; }

private:
    void queue_metrics();

    struct MetricsJob {
        ImageQaMetrics metrics;
        aetherscan::io::RgbImage error;
        int view{-1};
        std::uint64_t revision{};
    };

    gpu::PreviewTexture gt_;
    gpu::PreviewTexture error_;
    aetherscan::io::RgbImage gt_cpu_;
    aetherscan::io::RgbImage render_cpu_;
    std::future<aetherscan::io::RgbImage> pending_;
    std::future<MetricsJob> metrics_pending_;
    std::filesystem::path pending_path_;
    std::filesystem::path loaded_path_;
    int pending_view_{-1};
    int loaded_view_{-1};
    int render_view_{-1};
    std::uint64_t render_revision_{};
    bool loading_{};
    bool metrics_busy_{};
    bool failed_{};
    ImageQaMetrics metrics_;
};

[[nodiscard]] bool image_qa_needs_render(ImageQaMode mode);

void refresh_image_qa_folder(
    ImageQaState& state, const std::filesystem::path& directory);

[[nodiscard]] int image_qa_count(
    const ImageQaState& state, const SparseScene& scene);

void select_image_qa_view(ImageQaState& state, int index, int count);

struct ImageQaDrawInput {
    const SparseScene* scene{};
    gpu::CameraPhotoCache* photos{};
    ImTextureID render{};
    bool has_render{};
    bool render_live{};
    bool has_model{};
};

void draw_image_qa(
    ImageQaState& state, ImageQaSession& session, const ImageQaDrawInput& input,
    ImVec2 min, ImVec2 max);

}  // namespace editor
