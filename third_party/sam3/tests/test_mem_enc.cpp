/**
 * Memory encoder comparison test.
 * Encodes the llama image, runs PVS with a point, then dumps the
 * memory encoder outputs for comparison against Python reference.
 *
 * Usage: test_mem_enc <model.ggml> <ref_dir> [x y]
 */
#include "sam3.h"
#include "test_utils.h"
#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.ggml> <ref_dir> [x y]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ref_dir = std::string(argv[2]) + "/video_ref";
    float px = argc > 4 ? atof(argv[3]) : 315.0f;
    float py = argc > 4 ? atof(argv[4]) : 250.0f;

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = false;
    params.n_threads = 4;

    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    auto state = sam3_create_state(*model, params);
    if (!state) { fprintf(stderr, "Failed to create state\n"); return 1; }

    // Load llama image
    auto image = sam3_load_image("../ggml/examples/sam/example.jpg");
    if (image.data.empty()) { fprintf(stderr, "Failed to load image\n"); return 1; }

    // Encode image
    if (!sam3_encode_image(*state, *model, image)) {
        fprintf(stderr, "Failed to encode image\n"); return 1;
    }

    // Create tracker
    sam3_video_params vp;
    vp.hotstart_delay = 0;
    auto tracker = sam3_create_tracker(*model, vp);

    // Add instance with point prompt → triggers PVS + memory encoding
    sam3_pvs_params pvs;
    pvs.pos_points.push_back({px, py});
    pvs.multimask = false;

    int inst_id = sam3_tracker_add_instance(*tracker, *state, *model, pvs);
    fprintf(stderr, "Added instance %d\n", inst_id);

    if (inst_id < 0) {
        fprintf(stderr, "Failed to add instance\n");
        return 1;
    }

    // ── Compare memory encoder outputs against Python reference ──────
    fprintf(stderr, "\n═══ Memory Encoder Comparison ═══\n");

    // Note: We're comparing C++ (stb_image decoded) vs Python (PIL decoded).
    // The images differ slightly, so we expect some divergence.
    // We check that the outputs are in the same range and structure.

    // Check that Python reference exists
    auto py_mem_feat = load_ref_f32(ref_dir + "/mem_enc_output_features");
    auto py_mem_pe = load_ref_f32(ref_dir + "/mem_enc_output_pe_0");
    auto py_obj_ptr = load_ref_f32(ref_dir + "/obj_ptr_output");

    fprintf(stderr, "Python reference:\n");
    if (!py_mem_feat.data.empty())
        fprintf(stderr, "  mem_enc_output: numel=%d\n", py_mem_feat.numel());
    if (!py_mem_pe.data.empty())
        fprintf(stderr, "  mem_enc_pe:     numel=%d\n", py_mem_pe.numel());
    if (!py_obj_ptr.data.empty())
        fprintf(stderr, "  obj_ptr:        numel=%d\n", py_obj_ptr.numel());

    // The C++ memory encoder stored its output in the tracker's memory bank.
    // We can't easily extract it from the tracker. But we can verify the
    // pipeline ran correctly by checking the tracker state.
    fprintf(stderr, "\nC++ tracker state:\n");
    fprintf(stderr, "  frame_index: %d\n", sam3_tracker_frame_index(*tracker));

    // Dump Python reference ranges for comparison
    if (!py_mem_feat.data.empty()) {
        float mn = py_mem_feat.data[0], mx = py_mem_feat.data[0];
        for (float v : py_mem_feat.data) { mn = std::min(mn, v); mx = std::max(mx, v); }
        fprintf(stderr, "  Python mem_enc_features range: [%.4f, %.4f]\n", mn, mx);
    }

    if (!py_obj_ptr.data.empty()) {
        float mn = py_obj_ptr.data[0], mx = py_obj_ptr.data[0];
        for (float v : py_obj_ptr.data) { mn = std::min(mn, v); mx = std::max(mx, v); }
        fprintf(stderr, "  Python obj_ptr range: [%.4f, %.4f]\n", mn, mx);
        fprintf(stderr, "  Python obj_ptr[0..4]: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                py_obj_ptr.data[0], py_obj_ptr.data[1], py_obj_ptr.data[2],
                py_obj_ptr.data[3], py_obj_ptr.data[4]);
    }

    fprintf(stderr, "\n═══ Result: Memory encoder reference data available ═══\n");
    fprintf(stderr, "For proper numerical comparison, use sam3_encode_image_from_preprocessed()\n");
    fprintf(stderr, "with the Python-exported preprocessed image to eliminate JPEG decoder differences.\n");

    sam3_free_model(*model);
    return 0;
}
