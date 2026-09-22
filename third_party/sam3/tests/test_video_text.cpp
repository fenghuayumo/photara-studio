/**
 * Video tracking test with text prompt on first N frames.
 * Usage: test_video_text <model.ggml> <video.mp4> [prompt] [n_frames]
 */
#include "sam3.h"
#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.ggml> <video.mp4> [prompt] [n_frames]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string video_path = argv[2];
    const std::string prompt = argc > 3 ? argv[3] : "car";
    const int n_frames = argc > 4 ? atoi(argv[4]) : 5;

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = false;
    params.n_threads = 4;

    fprintf(stderr, "Loading model...\n");
    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    auto state = sam3_create_state(*model, params);
    if (!state) { fprintf(stderr, "Failed to create state\n"); return 1; }

    auto vi = sam3_get_video_info(video_path);
    fprintf(stderr, "Video: %dx%d, %d frames, %.1f fps\n",
            vi.width, vi.height, vi.n_frames, vi.fps);

    // Create tracker with text prompt
    sam3_video_params vp;
    vp.text_prompt = prompt;
    vp.score_threshold = 0.3f;
    vp.hotstart_delay = 3;  // Quick confirmation for testing
    auto tracker = sam3_create_tracker(*model, vp);
    if (!tracker) { fprintf(stderr, "Failed to create tracker\n"); return 1; }

    int total_tracked = 0;
    for (int fi = 0; fi < std::min(n_frames, vi.n_frames); ++fi) {
        fprintf(stderr, "\n════════════════════════════════════════\n");
        fprintf(stderr, "  Frame %d/%d\n", fi, n_frames);
        fprintf(stderr, "════════════════════════════════════════\n");

        auto frame = sam3_decode_video_frame(video_path, fi);
        if (frame.data.empty()) {
            fprintf(stderr, "Failed to decode frame %d\n", fi);
            continue;
        }

        auto result = sam3_track_frame(*tracker, *state, *model, frame);
        fprintf(stderr, "  Result: %zu detections\n", result.detections.size());

        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& d = result.detections[i];
            int n_on = 0;
            for (size_t j = 0; j < d.mask.data.size(); ++j)
                if (d.mask.data[j] > 0) n_on++;
            fprintf(stderr, "    det %zu: inst=%d score=%.3f box=[%.0f,%.0f,%.0f,%.0f] mask=%.1f%%\n",
                    i, d.instance_id, d.score,
                    d.box.x0, d.box.y0, d.box.x1, d.box.y1,
                    100.0f * n_on / d.mask.data.size());
        }

        if (!result.detections.empty()) {
            total_tracked++;
            // Save first frame's masks
            if (fi == 0) {
                for (size_t i = 0; i < result.detections.size() && i < 3; ++i) {
                    std::string path = "/tmp/video_text_f0_mask" + std::to_string(i) + ".png";
                    sam3_save_mask(result.detections[i].mask, path);
                    fprintf(stderr, "    saved: %s\n", path.c_str());
                }
            }
        }
    }

    fprintf(stderr, "\n═══ Summary ═══\n");
    fprintf(stderr, "  Prompt: '%s'\n", prompt.c_str());
    fprintf(stderr, "  Frames with detections: %d/%d\n", total_tracked, n_frames);

    sam3_free_model(*model);
    return total_tracked > 0 ? 0 : 1;
}
