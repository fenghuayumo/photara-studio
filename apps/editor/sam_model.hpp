#pragma once

#include "model_download.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace editor {

struct App;

// curl download of the SAM 3 ggml checkpoint into the Photara model cache.
// Starts only after the caller has recorded licence acceptance.
class SamModelDownload : public ModelDownload {
public:
    using ModelDownload::start;
    void start(); // Caller must first obtain SAM's license acceptance.
    std::vector<ModelDownloadSpec> pending;
};

// Starts every missing file needed by the selected engine, one at a time.
void start_sam_model_downloads(App& app);
// Moves a finished download on to the next missing model file.
void update_sam_model_downloads(App& app);

// False when a prompt or a checkpoint is still missing. Opens the licence
// dialog and leaves a status message in that case.
bool ensure_sam_ready(App& app);
void draw_mask_model_controls(App& app);
std::filesystem::path selected_mask_model(const App& app);
std::filesystem::path selected_mask_detector(const App& app);
void draw_sam_license_modal(App& app);
bool open_model_license_url(const char* url);

}  // namespace editor
