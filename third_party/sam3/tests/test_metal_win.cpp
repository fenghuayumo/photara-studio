// Test Metal kernels for ggml_win_part and ggml_win_unpart.
// Runs the same graph on CPU and Metal, then compares results element-by-element.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Run a graph: build win_part -> win_unpart on the given backend.
// Returns the final output as a flat float vector.
static std::vector<float> run_win_roundtrip(ggml_backend_t backend,
                                            const std::vector<float>& input,
                                            int C, int W, int H, int WS) {
    // Graph context
    size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    struct ggml_init_params params = {ctx_size, nullptr, true};
    struct ggml_context* ctx = ggml_init(params);

    // Input tensor: [C, W, H, 1]
    auto* inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, W, H);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // win_part: [C, W, H, 1] -> [C, WS, WS, np]
    auto* parted = ggml_win_part(ctx, inp, WS);
    ggml_set_name(parted, "parted");

    // win_unpart: [C, WS, WS, np] -> [C, W, H, 1]
    auto* unparted = ggml_win_unpart(ctx, parted, W, H, WS);
    ggml_set_name(unparted, "unparted");
    ggml_set_output(unparted);

    // Build graph
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, unparted);

    // Allocate
    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "FAIL: gallocr_alloc_graph failed\n");
        exit(1);
    }

    // Set input data
    ggml_backend_tensor_set(inp, input.data(), 0, input.size() * sizeof(float));

    // Compute
    ggml_backend_graph_compute(backend, graph);

    // Read output
    std::vector<float> output(C * W * H);
    ggml_backend_tensor_get(unparted, output.data(), 0, output.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return output;
}

// Run win_part only (no roundtrip) — compare intermediate output.
static std::vector<float> run_win_part_only(ggml_backend_t backend,
                                            const std::vector<float>& input,
                                            int C, int W, int H, int WS,
                                            int64_t* out_ne) {
    size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    struct ggml_init_params params = {ctx_size, nullptr, true};
    struct ggml_context* ctx = ggml_init(params);

    auto* inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, W, H);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    auto* parted = ggml_win_part(ctx, inp, WS);
    ggml_set_name(parted, "parted");
    ggml_set_output(parted);

    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, parted);

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "FAIL: gallocr_alloc_graph failed\n");
        exit(1);
    }

    ggml_backend_tensor_set(inp, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_graph_compute(backend, graph);

    // Output shape
    int64_t total = 1;
    for (int d = 0; d < 4; d++) {
        out_ne[d] = parted->ne[d];
        total *= parted->ne[d];
    }

    std::vector<float> output(total);
    ggml_backend_tensor_get(parted, output.data(), 0, output.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return output;
}

static bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b,
                            const char* label, float tol = 1e-6f) {
    if (a.size() != b.size()) {
        fprintf(stderr, "  FAIL [%s]: size mismatch: %zu vs %zu\n", label, a.size(), b.size());
        return false;
    }
    float max_diff = 0.0f;
    int first_fail = -1;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = std::fabs(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol && first_fail < 0) first_fail = (int)i;
    }
    if (first_fail >= 0) {
        fprintf(stderr, "  FAIL [%s]: max_diff=%.8f at index %d (cpu=%.8f metal=%.8f)\n",
                label, max_diff, first_fail, a[first_fail], b[first_fail]);
        return false;
    }
    fprintf(stderr, "  PASS [%s]: max_diff=%.2e (%zu elements)\n", label, max_diff, a.size());
    return true;
}

int main() {
#ifndef GGML_USE_METAL
    fprintf(stderr, "Metal not available, skipping test\n");
    return 0;
#else
    // Init backends
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    ggml_backend_t metal_backend = ggml_backend_metal_init();
    if (!metal_backend) {
        fprintf(stderr, "FAIL: could not init Metal backend\n");
        return 1;
    }

    bool all_pass = true;

    // ── Test 1: sam3 exact dimensions (C=1024, W=H=72, WS=24) ──────────
    {
        fprintf(stderr, "\n=== Test 1: sam3 dimensions (1024, 72, 72, WS=24) ===\n");
        const int C = 1024, W = 72, H = 72, WS = 24;
        std::vector<float> input(C * W * H);
        for (size_t i = 0; i < input.size(); i++) {
            input[i] = 0.001f * (float)(i % 9973);  // pseudo-random
        }

        // win_part only
        int64_t cpu_ne[4], metal_ne[4];
        auto cpu_part   = run_win_part_only(cpu_backend, input, C, W, H, WS, cpu_ne);
        auto metal_part = run_win_part_only(metal_backend, input, C, W, H, WS, metal_ne);
        fprintf(stderr, "  win_part output shape: [%lld, %lld, %lld, %lld]\n",
                cpu_ne[0], cpu_ne[1], cpu_ne[2], cpu_ne[3]);
        all_pass &= compare_vectors(cpu_part, metal_part, "win_part");

        // roundtrip (win_part -> win_unpart)
        auto cpu_rt   = run_win_roundtrip(cpu_backend, input, C, W, H, WS);
        auto metal_rt = run_win_roundtrip(metal_backend, input, C, W, H, WS);
        all_pass &= compare_vectors(cpu_rt, metal_rt, "roundtrip");

        // roundtrip should recover original input exactly (no padding needed)
        all_pass &= compare_vectors(input, cpu_rt, "cpu_identity");
        all_pass &= compare_vectors(input, metal_rt, "metal_identity");
    }

    // ── Test 2: non-divisible dimensions (needs padding) ────────────────
    {
        fprintf(stderr, "\n=== Test 2: padding case (64, 50, 50, WS=24) ===\n");
        const int C = 64, W = 50, H = 50, WS = 24;
        std::vector<float> input(C * W * H);
        for (size_t i = 0; i < input.size(); i++) {
            input[i] = -1.0f + 2.0f * (float)(i % 7919) / 7919.0f;
        }

        int64_t cpu_ne[4], metal_ne[4];
        auto cpu_part   = run_win_part_only(cpu_backend, input, C, W, H, WS, cpu_ne);
        auto metal_part = run_win_part_only(metal_backend, input, C, W, H, WS, metal_ne);
        fprintf(stderr, "  win_part output shape: [%lld, %lld, %lld, %lld]\n",
                cpu_ne[0], cpu_ne[1], cpu_ne[2], cpu_ne[3]);
        all_pass &= compare_vectors(cpu_part, metal_part, "win_part_padded");

        auto cpu_rt   = run_win_roundtrip(cpu_backend, input, C, W, H, WS);
        auto metal_rt = run_win_roundtrip(metal_backend, input, C, W, H, WS);
        all_pass &= compare_vectors(cpu_rt, metal_rt, "roundtrip_padded");
        all_pass &= compare_vectors(input, cpu_rt, "cpu_identity_padded");
        all_pass &= compare_vectors(input, metal_rt, "metal_identity_padded");
    }

    // ── Test 3: small case for easy debugging ──────────────────────────
    {
        fprintf(stderr, "\n=== Test 3: small (4, 6, 6, WS=3) ===\n");
        const int C = 4, W = 6, H = 6, WS = 3;
        std::vector<float> input(C * W * H);
        for (size_t i = 0; i < input.size(); i++) {
            input[i] = (float)(i + 1);
        }

        int64_t cpu_ne[4], metal_ne[4];
        auto cpu_part   = run_win_part_only(cpu_backend, input, C, W, H, WS, cpu_ne);
        auto metal_part = run_win_part_only(metal_backend, input, C, W, H, WS, metal_ne);
        fprintf(stderr, "  win_part output shape: [%lld, %lld, %lld, %lld]\n",
                cpu_ne[0], cpu_ne[1], cpu_ne[2], cpu_ne[3]);
        all_pass &= compare_vectors(cpu_part, metal_part, "win_part_small");

        auto cpu_rt   = run_win_roundtrip(cpu_backend, input, C, W, H, WS);
        auto metal_rt = run_win_roundtrip(metal_backend, input, C, W, H, WS);
        all_pass &= compare_vectors(cpu_rt, metal_rt, "roundtrip_small");
        all_pass &= compare_vectors(input, cpu_rt, "cpu_identity_small");
        all_pass &= compare_vectors(input, metal_rt, "metal_identity_small");
    }

    // ── Test 4: asymmetric dimensions ──────────────────────────────────
    {
        fprintf(stderr, "\n=== Test 4: asymmetric (128, 48, 72, WS=24) ===\n");
        const int C = 128, W = 48, H = 72, WS = 24;
        std::vector<float> input(C * W * H);
        for (size_t i = 0; i < input.size(); i++) {
            input[i] = sinf((float)i * 0.01f);
        }

        int64_t cpu_ne[4], metal_ne[4];
        auto cpu_part   = run_win_part_only(cpu_backend, input, C, W, H, WS, cpu_ne);
        auto metal_part = run_win_part_only(metal_backend, input, C, W, H, WS, metal_ne);
        fprintf(stderr, "  win_part output shape: [%lld, %lld, %lld, %lld]\n",
                cpu_ne[0], cpu_ne[1], cpu_ne[2], cpu_ne[3]);
        all_pass &= compare_vectors(cpu_part, metal_part, "win_part_asym");

        auto cpu_rt   = run_win_roundtrip(cpu_backend, input, C, W, H, WS);
        auto metal_rt = run_win_roundtrip(metal_backend, input, C, W, H, WS);
        all_pass &= compare_vectors(cpu_rt, metal_rt, "roundtrip_asym");
        all_pass &= compare_vectors(input, cpu_rt, "cpu_identity_asym");
        all_pass &= compare_vectors(input, metal_rt, "metal_identity_asym");
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(metal_backend);

    fprintf(stderr, "\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
#endif
}
