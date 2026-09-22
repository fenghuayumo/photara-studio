// Profile the REAL SAM3 ViT graph on Metal.
// Runs encode_image twice to separate first-run compilation from steady-state.
// Also breaks down graph_compute into encoding vs GPU execution.

#include "sam3.h"
#include "ggml-backend.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <chrono>
#include <cstdio>
#include <string>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.ggml> <image.jpg>\n", argv[0]);
        return 1;
    }

    auto img = sam3_load_image(argv[2]);
    fprintf(stderr, "Image: %dx%d\n\n", img.width, img.height);

    // ── Metal: run encode_image TWICE ───────────────────────────────────
    fprintf(stderr, "=== Metal: Loading model ===\n");
    sam3_params params;
    params.model_path = argv[1];
    params.use_gpu = true;
    params.n_threads = 8;

    double t0 = now_ms();
    auto model = sam3_load_model(params);
    double t1 = now_ms();
    fprintf(stderr, "  Model load: %.0f ms\n", t1 - t0);

    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    auto state = sam3_create_state(*model, params);
    if (!state) { fprintf(stderr, "Failed to create state\n"); return 1; }

    // Run 1: includes pipeline compilation
    fprintf(stderr, "\n=== Metal: encode_image RUN 1 (cold — includes pipeline compilation) ===\n");
    t0 = now_ms();
    bool ok = sam3_encode_image(*state, *model, img);
    t1 = now_ms();
    fprintf(stderr, "  Run 1 total: %.0f ms %s\n", t1 - t0, ok ? "" : "FAILED");

    // Read back a tensor to force GPU synchronization
    {
        double ts0 = now_ms();
        sam3_dump_state_tensor(*state, "vit_output", "/tmp/sam3_metal_prof_vit");
        double ts1 = now_ms();
        fprintf(stderr, "  Readback sync: %.0f ms\n", ts1 - ts0);
    }

    // Run 2: pipelines already compiled, GPU warmed up
    fprintf(stderr, "\n=== Metal: encode_image RUN 2 (warm — no compilation) ===\n");
    t0 = now_ms();
    ok = sam3_encode_image(*state, *model, img);
    t1 = now_ms();
    fprintf(stderr, "  Run 2 total: %.0f ms %s\n", t1 - t0, ok ? "" : "FAILED");

    {
        double ts0 = now_ms();
        sam3_dump_state_tensor(*state, "vit_output", "/tmp/sam3_metal_prof_vit2");
        double ts1 = now_ms();
        fprintf(stderr, "  Readback sync: %.0f ms\n", ts1 - ts0);
    }

    // Run 3: one more to confirm steady state
    fprintf(stderr, "\n=== Metal: encode_image RUN 3 (warm) ===\n");
    t0 = now_ms();
    ok = sam3_encode_image(*state, *model, img);
    t1 = now_ms();
    fprintf(stderr, "  Run 3 total: %.0f ms %s\n", t1 - t0, ok ? "" : "FAILED");

    {
        double ts0 = now_ms();
        sam3_dump_state_tensor(*state, "vit_output", "/tmp/sam3_metal_prof_vit3");
        double ts1 = now_ms();
        fprintf(stderr, "  Readback sync: %.0f ms\n", ts1 - ts0);
    }

    // ── CPU: for comparison ─────────────────────────────────────────────
    fprintf(stderr, "\n=== CPU: encode_image (for comparison) ===\n");
    {
        sam3_params cpu_params;
        cpu_params.model_path = argv[1];
        cpu_params.use_gpu = false;
        cpu_params.n_threads = 8;

        auto cpu_model = sam3_load_model(cpu_params);
        auto cpu_state = sam3_create_state(*cpu_model, cpu_params);

        t0 = now_ms();
        ok = sam3_encode_image(*cpu_state, *cpu_model, img);
        t1 = now_ms();
        fprintf(stderr, "  CPU total: %.0f ms %s\n", t1 - t0, ok ? "" : "FAILED");
    }

    return 0;
}
