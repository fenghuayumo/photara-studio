/**
 * Compare visual-only model output vs full model output for PVS (points/boxes)
 * and video propagation. The tracker-path weights are identical, so outputs
 * should match bit-for-bit (same ggml graph, same weights, same inputs).
 *
 * Usage: ./test_visual_only_compare <full.ggml> <visual.ggml> <image.jpg> [video.mp4]
 */
#include "sam3.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static float mask_iou(const sam3_mask& a, const sam3_mask& b) {
    if (a.width != b.width || a.height != b.height) return 0.0f;
    int inter = 0, uni = 0;
    for (int i = 0; i < (int)a.data.size(); ++i) {
        bool fa = a.data[i] > 127;
        bool fb = b.data[i] > 127;
        if (fa && fb) inter++;
        if (fa || fb) uni++;
    }
    return uni > 0 ? (float)inter / uni : 1.0f;
}

static int count_fg(const sam3_mask& m) {
    int n = 0;
    for (auto v : m.data) if (v > 127) n++;
    return n;
}

static bool compare_results(const char* label,
                            const sam3_result& full_r,
                            const sam3_result& vis_r) {
    fprintf(stderr, "\n--- %s ---\n", label);
    if (full_r.detections.size() != vis_r.detections.size()) {
        fprintf(stderr, "  FAIL: detection count mismatch: full=%zu visual=%zu\n",
                full_r.detections.size(), vis_r.detections.size());
        return false;
    }
    if (full_r.detections.empty()) {
        fprintf(stderr, "  WARN: both returned 0 detections\n");
        return true;
    }

    bool ok = true;
    for (size_t i = 0; i < full_r.detections.size(); ++i) {
        const auto& fd = full_r.detections[i];
        const auto& vd = vis_r.detections[i];

        float iou = mask_iou(fd.mask, vd.mask);
        int fg_full = count_fg(fd.mask);
        int fg_vis = count_fg(vd.mask);
        float score_diff = fabsf(fd.score - vd.score);
        float iou_diff = fabsf(fd.iou_score - vd.iou_score);

        fprintf(stderr, "  det[%zu]: mask_iou=%.6f  fg_full=%d fg_vis=%d  "
                "score_diff=%.6f  iou_diff=%.6f\n",
                i, iou, fg_full, fg_vis, score_diff, iou_diff);

        if (iou < 0.999f) {
            fprintf(stderr, "  FAIL: mask IoU %.6f < 0.999 threshold\n", iou);
            ok = false;
        }
        if (score_diff > 1e-4f) {
            fprintf(stderr, "  FAIL: score diff %.6f > 1e-4\n", score_diff);
            ok = false;
        }
    }
    fprintf(stderr, "  %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <full.ggml> <visual.ggml> <image.jpg> [video.mp4]\n", argv[0]);
        return 1;
    }

    const std::string full_path = argv[1];
    const std::string vis_path = argv[2];
    const std::string image_path = argv[3];
    const std::string video_path = (argc >= 5) ? argv[4] : "";

    int n_pass = 0, n_fail = 0;
    auto tally = [&](bool ok) { if (ok) n_pass++; else n_fail++; };

    auto img = sam3_load_image(image_path);
    if (img.data.empty()) {
        fprintf(stderr, "Failed to load image '%s'\n", image_path.c_str());
        return 1;
    }
    fprintf(stderr, "Image: %dx%d\n", img.width, img.height);

    // ── Define test prompts ─────────────────────────────────────────────
    struct pvs_test {
        const char* name;
        sam3_pvs_params params;
    };

    pvs_test tests[] = {
        {"center_point", {}},
        {"box_prompt", {}},
        {"two_points", {}},
        {"point_and_box", {}},
    };

    // Center point
    tests[0].params.pos_points.push_back({(float)img.width / 2, (float)img.height / 2});

    // Box prompt
    tests[1].params.box = {(float)img.width * 0.1f, (float)img.height * 0.1f,
                           (float)img.width * 0.9f, (float)img.height * 0.9f};
    tests[1].params.use_box = true;

    // Two points (positive + negative)
    tests[2].params.pos_points.push_back({(float)img.width / 2, (float)img.height / 2});
    tests[2].params.neg_points.push_back({10.0f, 10.0f});

    // Point + box
    tests[3].params.pos_points.push_back({(float)img.width / 2, (float)img.height / 2});
    tests[3].params.box = {(float)img.width * 0.2f, (float)img.height * 0.2f,
                           (float)img.width * 0.8f, (float)img.height * 0.8f};
    tests[3].params.use_box = true;

    // ═══════════════════════════════════════════════════════════════════════
    //  Run on full model
    // ═══════════════════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ Loading FULL model ═══\n");
    sam3_params fparams;
    fparams.model_path = full_path;
    fparams.n_threads = 4;
    fparams.use_gpu = false;

    auto full_model = sam3_load_model(fparams);
    if (!full_model) { fprintf(stderr, "Failed to load full model\n"); return 1; }
    assert(!sam3_is_visual_only(*full_model));

    auto full_state = sam3_create_state(*full_model, fparams);
    if (!full_state) { fprintf(stderr, "Failed to create state\n"); return 1; }

    if (!sam3_encode_image(*full_state, *full_model, img)) {
        fprintf(stderr, "Failed to encode image (full)\n"); return 1;
    }

    std::vector<sam3_result> full_results;
    for (auto& t : tests) {
        fprintf(stderr, "  Running PVS [%s] on full model...\n", t.name);
        full_results.push_back(sam3_segment_pvs(*full_state, *full_model, t.params));
    }

    // Video propagation on full model
    std::vector<sam3_result> full_video_results;
    if (!video_path.empty()) {
        fprintf(stderr, "\n  Running video propagation on full model...\n");
        sam3_visual_track_params vtp;
        auto tracker = sam3_create_visual_tracker(*full_model, vtp);

        // Frame 0: encode + add instance
        auto f0 = sam3_decode_video_frame(video_path, 0);
        if (!f0.data.empty()) {
            sam3_encode_image(*full_state, *full_model, f0);
            sam3_pvs_params vpvs;
            vpvs.pos_points.push_back({(float)f0.width / 2, (float)f0.height / 2});
            sam3_tracker_add_instance(*tracker, *full_state, *full_model, vpvs);

            // Propagate frames 1-3
            for (int fi = 1; fi <= 3; ++fi) {
                auto frame = sam3_decode_video_frame(video_path, fi);
                if (frame.data.empty()) break;
                full_video_results.push_back(
                    sam3_propagate_frame(*tracker, *full_state, *full_model, frame));
            }
        }
    }

    // Free full model to save memory
    full_state.reset();
    sam3_free_model(*full_model);
    full_model.reset();

    // ═══════════════════════════════════════════════════════════════════════
    //  Run on visual-only model
    // ═══════════════════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ Loading VISUAL-ONLY model ═══\n");
    sam3_params vparams;
    vparams.model_path = vis_path;
    vparams.n_threads = 4;
    vparams.use_gpu = false;

    auto vis_model = sam3_load_model(vparams);
    if (!vis_model) { fprintf(stderr, "Failed to load visual model\n"); return 1; }
    assert(sam3_is_visual_only(*vis_model));

    auto vis_state = sam3_create_state(*vis_model, vparams);
    if (!vis_state) { fprintf(stderr, "Failed to create state\n"); return 1; }

    if (!sam3_encode_image(*vis_state, *vis_model, img)) {
        fprintf(stderr, "Failed to encode image (visual)\n"); return 1;
    }

    std::vector<sam3_result> vis_results;
    for (auto& t : tests) {
        fprintf(stderr, "  Running PVS [%s] on visual-only model...\n", t.name);
        vis_results.push_back(sam3_segment_pvs(*vis_state, *vis_model, t.params));
    }

    // Video propagation on visual-only model
    std::vector<sam3_result> vis_video_results;
    if (!video_path.empty()) {
        fprintf(stderr, "\n  Running video propagation on visual-only model...\n");
        sam3_visual_track_params vtp;
        auto tracker = sam3_create_visual_tracker(*vis_model, vtp);

        auto f0 = sam3_decode_video_frame(video_path, 0);
        if (!f0.data.empty()) {
            sam3_encode_image(*vis_state, *vis_model, f0);
            sam3_pvs_params vpvs;
            vpvs.pos_points.push_back({(float)f0.width / 2, (float)f0.height / 2});
            sam3_tracker_add_instance(*tracker, *vis_state, *vis_model, vpvs);

            for (int fi = 1; fi <= 3; ++fi) {
                auto frame = sam3_decode_video_frame(video_path, fi);
                if (frame.data.empty()) break;
                vis_video_results.push_back(
                    sam3_propagate_frame(*tracker, *vis_state, *vis_model, frame));
            }
        }
    }

    vis_state.reset();
    sam3_free_model(*vis_model);
    vis_model.reset();

    // ═══════════════════════════════════════════════════════════════════════
    //  Compare results
    // ═══════════════════════════════════════════════════════════════════════
    fprintf(stderr, "\n══════════════════════════════════════════════════\n");
    fprintf(stderr, "  COMPARISON: Full model vs Visual-only model\n");
    fprintf(stderr, "══════════════════════════════════════════════════\n");

    for (size_t i = 0; i < full_results.size(); ++i) {
        tally(compare_results(tests[i].name, full_results[i], vis_results[i]));
    }

    for (size_t i = 0; i < full_video_results.size() && i < vis_video_results.size(); ++i) {
        char label[64];
        snprintf(label, sizeof(label), "video_frame_%zu", i + 1);
        tally(compare_results(label, full_video_results[i], vis_video_results[i]));
    }

    fprintf(stderr, "\n══════════════════════════════════════════════════\n");
    fprintf(stderr, "  TOTAL: %d PASS, %d FAIL\n", n_pass, n_fail);
    fprintf(stderr, "══════════════════════════════════════════════════\n\n");

    return n_fail > 0 ? 1 : 0;
}
