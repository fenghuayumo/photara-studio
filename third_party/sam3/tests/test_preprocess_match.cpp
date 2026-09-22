/**
 * Test preprocessing match: run sam3_encode_image with C++ preprocessing
 * on the same image, then dump outputs for comparison against Python reference.
 *
 * Usage:
 *   ./test_preprocess_match <model.ggml> <image.jpg> <output_dir>
 */
#include "sam3.h"
#include "test_utils.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.ggml> <image.jpg> <output_dir>\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    const std::string out_dir = argv[3];
    ensure_dir(out_dir);

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = false;
    params.n_threads = 4;

    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    auto state = sam3_create_state(*model, params);
    if (!state) { fprintf(stderr, "Failed to create state\n"); return 1; }

    auto img = sam3_load_image(image_path);
    if (img.data.empty()) { fprintf(stderr, "Failed to load image\n"); return 1; }

    fprintf(stderr, "Encoding with C++ preprocessing...\n");
    if (!sam3_encode_image(*state, *model, img)) {
        fprintf(stderr, "Encoding failed\n");
        return 1;
    }

    // Dump outputs
    sam3_dump_state_tensor(*state, "vit_output", out_dir + "/vit_output");
    for (int i = 0; i < 4; ++i) {
        char name[32];
        snprintf(name, sizeof(name), "neck_det_%d", i);
        sam3_dump_state_tensor(*state, name, out_dir + "/" + name);
    }

    fprintf(stderr, "Done. Outputs in %s\n", out_dir.c_str());
    state.reset();
    sam3_free_model(*model);
    return 0;
}
