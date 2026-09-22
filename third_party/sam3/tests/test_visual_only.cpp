#include "sam3.h"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <visual-only-model.ggml> [image.jpg]\n", argv[0]);
        return 1;
    }

    sam3_params params;
    params.model_path = argv[1];
    params.n_threads = 4;

    // 1. Load visual-only model
    fprintf(stderr, "=== Test 1: Load visual-only model ===\n");
    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "FAIL: failed to load model\n");
        return 1;
    }
    fprintf(stderr, "PASS: model loaded\n");

    // 2. Verify visual_only flag
    fprintf(stderr, "\n=== Test 2: sam3_is_visual_only ===\n");
    if (!sam3_is_visual_only(*model)) {
        fprintf(stderr, "FAIL: expected visual_only=true\n");
        return 1;
    }
    fprintf(stderr, "PASS: sam3_is_visual_only() = true\n");

    // 3. PCS should return empty result (no crash)
    fprintf(stderr, "\n=== Test 3: PCS guard ===\n");
    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "FAIL: create_state\n");
        return 1;
    }
    sam3_pcs_params pcs;
    pcs.text_prompt = "cat";
    auto r = sam3_segment_pcs(*state, *model, pcs);
    if (!r.detections.empty()) {
        fprintf(stderr, "FAIL: PCS should return empty on visual-only\n");
        return 1;
    }
    fprintf(stderr, "PASS: PCS returns empty on visual-only model\n");

    // 4. If image provided, test PVS + visual tracking
    if (argc >= 3) {
        fprintf(stderr, "\n=== Test 4: PVS on visual-only model ===\n");
        auto img = sam3_load_image(argv[2]);
        if (img.data.empty()) {
            fprintf(stderr, "FAIL: failed to load image '%s'\n", argv[2]);
            return 1;
        }
        if (!sam3_encode_image(*state, *model, img)) {
            fprintf(stderr, "FAIL: encode_image\n");
            return 1;
        }

        // PVS with center point
        sam3_pvs_params pvs;
        pvs.pos_points.push_back({(float)img.width / 2, (float)img.height / 2});
        auto rp = sam3_segment_pvs(*state, *model, pvs);
        if (rp.detections.empty()) {
            fprintf(stderr, "FAIL: PVS returned no detections\n");
            return 1;
        }
        fprintf(stderr, "PASS: PVS produced %zu detection(s)\n", rp.detections.size());

        // 5. Visual tracker: create, add instance, propagate
        fprintf(stderr, "\n=== Test 5: Visual tracker ===\n");
        sam3_visual_track_params vtp;
        auto tracker = sam3_create_visual_tracker(*model, vtp);
        if (!tracker) {
            fprintf(stderr, "FAIL: create_visual_tracker\n");
            return 1;
        }

        int id = sam3_tracker_add_instance(*tracker, *state, *model, pvs);
        if (id < 0) {
            fprintf(stderr, "FAIL: tracker_add_instance returned %d\n", id);
            return 1;
        }
        fprintf(stderr, "PASS: tracker_add_instance -> id=%d\n", id);

        // Propagate same frame as a smoke test
        auto rr = sam3_propagate_frame(*tracker, *state, *model, img);
        fprintf(stderr, "PASS: propagate_frame -> %zu detection(s)\n", rr.detections.size());

        // 6. track_frame should fail on visual-only
        fprintf(stderr, "\n=== Test 6: track_frame guard ===\n");
        auto rt = sam3_track_frame(*tracker, *state, *model, img);
        if (!rt.detections.empty()) {
            fprintf(stderr, "FAIL: track_frame should return empty on visual-only\n");
            return 1;
        }
        fprintf(stderr, "PASS: track_frame returns empty on visual-only model\n");
    }

    fprintf(stderr, "\nAll tests passed.\n");
    return 0;
}
