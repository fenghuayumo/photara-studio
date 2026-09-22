/**
 * Simple 2-frame video tracking test.
 * Frame 0: encode image, add instance via point prompt
 * Frame 1: track (propagate mask to same image)
 *
 * Usage: test_video_2frame <model.ggml> <image.jpg> [x y]
 */
#include "sam3.h"
#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.ggml> <image.jpg> [x y]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    float px = argc > 4 ? atof(argv[3]) : 315.0f;
    float py = argc > 4 ? atof(argv[4]) : 250.0f;

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = false;
    params.n_threads = 4;

    fprintf(stderr, "Loading model...\n");
    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    // Create tracker (no text prompt → propagation only, no PCS detection)
    sam3_video_params vp;
    vp.hotstart_delay = 0;     // instant confirmation (no warmup)
    vp.max_keep_alive = 100;
    auto tracker = sam3_create_tracker(*model, vp);
    if (!tracker) { fprintf(stderr, "Failed to create tracker\n"); return 1; }

    auto state = sam3_create_state(*model, params);
    if (!state) { fprintf(stderr, "Failed to create state\n"); return 1; }

    // Load image (use same image for both frames for simplicity)
    auto image = sam3_load_image(image_path);
    if (image.data.empty()) { fprintf(stderr, "Failed to load image\n"); return 1; }
    fprintf(stderr, "Image: %dx%d\n", image.width, image.height);

    // ════════════════════════════════════════════════════════════════
    // Frame 0: Encode + add instance via point
    // ════════════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ Frame 0: Encode + Add Instance ═══\n");

    // Encode image
    if (!sam3_encode_image(*state, *model, image)) {
        fprintf(stderr, "Failed to encode frame 0\n"); return 1;
    }

    // Add instance at the clicked point
    sam3_pvs_params pvs;
    pvs.pos_points.push_back({px, py});
    pvs.multimask = false;

    int inst_id = sam3_tracker_add_instance(*tracker, *state, *model, pvs);
    fprintf(stderr, "Added instance %d at (%.1f, %.1f)\n", inst_id, px, py);

    if (inst_id < 0) {
        fprintf(stderr, "Failed to add instance\n");
        return 1;
    }

    // Check tracker state
    fprintf(stderr, "Tracker frame: %d\n", sam3_tracker_frame_index(*tracker));

    // ════════════════════════════════════════════════════════════════
    // Frame 1: Track (propagate to same image)
    // ════════════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ Frame 1: Track ═══\n");

    // Re-encode same image as "frame 1"
    if (!sam3_encode_image(*state, *model, image)) {
        fprintf(stderr, "Failed to encode frame 1\n"); return 1;
    }

    // Track
    auto result = sam3_track_frame(*tracker, *state, *model, image);
    fprintf(stderr, "Track result: %zu detections\n", result.detections.size());

    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& d = result.detections[i];
        fprintf(stderr, "  det %zu: inst=%d score=%.4f iou=%.4f box=[%.1f,%.1f,%.1f,%.1f]\n",
                i, d.instance_id, d.score, d.iou_score,
                d.box.x0, d.box.y0, d.box.x1, d.box.y1);

        // Count mask pixels
        int n_on = 0;
        for (size_t j = 0; j < d.mask.data.size(); ++j)
            if (d.mask.data[j] > 0) n_on++;
        fprintf(stderr, "       mask: %d/%zu pixels (%.1f%%)\n",
                n_on, d.mask.data.size(), 100.0f * n_on / d.mask.data.size());

        std::string mask_path = "/tmp/video_frame1_mask_" + std::to_string(i) + ".png";
        sam3_save_mask(d.mask, mask_path);
        fprintf(stderr, "       saved: %s\n", mask_path.c_str());
    }

    sam3_free_model(*model);
    return result.detections.empty() ? 1 : 0;
}
