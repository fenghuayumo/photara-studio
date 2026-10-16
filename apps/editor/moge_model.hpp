#pragma once
#include "model_download.hpp"
#if defined(PHOTARA_HAS_INFERENCE)
#include "inference/model_cache.hpp"
#endif

namespace editor {
struct App;
inline constexpr const char* k_moge_model_url =
    "https://github.com/fenghuayumo/photara-studio/releases/download/v0.5.0/moge-2-vitb-normal-f32.gguf";
inline constexpr std::uint64_t k_moge_model_bytes = 419001600;
inline constexpr const char* k_moge_model_sha256 =
    "b515fd1ac1a187f46871aabd346870eaad22d2f20d2848637758c5c6e674b1c2";
inline std::filesystem::path moge_model_path() {
#if defined(PHOTARA_HAS_INFERENCE)
    return photara::inference::cached_model_path("moge-2-vitb-normal-f32.gguf");
#else
    return {};
#endif
}
ModelDownloadSpec moge_download_spec();
bool moge_model_ready();
bool ensure_moge_ready(App& app, bool resume_training);
void request_moge_download(App& app, bool train_after = false, bool resume_training = false);
void poll_moge_download(App& app);
void draw_moge_download_modal(App& app);
}
