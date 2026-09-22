/**
 * Quick PVS test: single positive point on the llama image.
 * Dumps mask to file for visual inspection.
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

    auto state = sam3_create_state(*model, params);
    if (!state) { fprintf(stderr, "Failed to create state\n"); return 1; }

    fprintf(stderr, "Loading image: %s\n", image_path.c_str());
    auto image = sam3_load_image(image_path);
    if (image.data.empty()) { fprintf(stderr, "Failed to load image\n"); return 1; }
    fprintf(stderr, "Image: %dx%d\n", image.width, image.height);

    fprintf(stderr, "Encoding image...\n");
    if (!sam3_encode_image(*state, *model, image)) {
        fprintf(stderr, "Failed to encode image\n"); return 1;
    }

    fprintf(stderr, "\n═══ PVS: point at (%.1f, %.1f) ═══\n", px, py);

    sam3_pvs_params pvs;
    pvs.pos_points.push_back({px, py});
    pvs.multimask = false;

    auto result = sam3_segment_pvs(*state, *model, pvs);
    fprintf(stderr, "Result: %zu detections\n", result.detections.size());

    for (size_t i = 0; i < result.detections.size(); ++i) {
        const auto& d = result.detections[i];
        fprintf(stderr, "  det %zu: score=%.4f iou=%.4f obj=%.4f box=[%.1f,%.1f,%.1f,%.1f] mask=%dx%d\n",
                i, d.score, d.iou_score, d.mask.obj_score,
                d.box.x0, d.box.y0, d.box.x1, d.box.y1,
                d.mask.width, d.mask.height);

        std::string mask_path = "/tmp/pvs_mask_" + std::to_string(i) + ".png";
        sam3_save_mask(d.mask, mask_path);
        fprintf(stderr, "  Saved: %s\n", mask_path.c_str());

        // Count mask pixels
        int n_on = 0;
        for (size_t j = 0; j < d.mask.data.size(); ++j)
            if (d.mask.data[j] > 0) n_on++;
        fprintf(stderr, "  Mask: %d/%zu pixels on (%.1f%%)\n",
                n_on, d.mask.data.size(), 100.0f * n_on / d.mask.data.size());
    }

    sam3_free_model(*model);
    return 0;
}
