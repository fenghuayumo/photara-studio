#pragma once

#include "model_download.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

namespace editor {

struct App;

// curl download of the SAM 3 ggml checkpoint into the Photara model cache.
// Starts only after the caller has recorded licence acceptance.
class SamModelDownload : public ModelDownload {
public:
    void start(); // Caller must first obtain SAM's license acceptance.
};

// False when a prompt or a checkpoint is still missing. Opens the licence
// dialog and leaves a status message in that case.
bool ensure_sam_ready(App& app);
void draw_sam_license_modal(App& app);

}  // namespace editor
