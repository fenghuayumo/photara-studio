/**
 * Quick test: run PCS on the llama image with various text prompts.
 * Uses sam3_encode_image (same as main_image example) — NOT preprocessed.
 */
#include "sam3.h"
#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.ggml> <image.jpg> [prompt]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string prompt = argc > 3 ? argv[3] : "llama";

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

    // Test multiple prompts
    const char* prompts[] = {"llama", "guitar", "animal"};
    int n_prompts = 3;
    if (argc > 4) {
        prompts[0] = prompt.c_str();
        n_prompts = 1;
    }

    for (int p = 0; p < n_prompts; ++p) {
        fprintf(stderr, "\n═══ Testing prompt: '%s' ═══\n", prompts[p]);

        sam3_pcs_params pcs;
        pcs.text_prompt = prompts[p];
        pcs.score_threshold = 0.3f;
        pcs.nms_threshold = 0.1f;

        auto result = sam3_segment_pcs(*state, *model, pcs);
        fprintf(stderr, "Result: %zu detections\n", result.detections.size());

        for (size_t i = 0; i < result.detections.size() && i < 5; ++i) {
            const auto& d = result.detections[i];
            fprintf(stderr, "  det %zu: score=%.4f box=[%.1f,%.1f,%.1f,%.1f]\n",
                    i, d.score, d.box.x0, d.box.y0, d.box.x1, d.box.y1);
        }

        // Save first mask
        if (!result.detections.empty()) {
            std::string mask_path = "/tmp/llama_mask_" + std::string(prompts[p]) + ".png";
            sam3_save_mask(result.detections[0].mask, mask_path);
            fprintf(stderr, "  Saved mask: %s\n", mask_path.c_str());
        }
    }

    sam3_free_model(*model);
    return 0;
}
