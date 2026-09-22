// Test Metal kernel for ggml_conv_2d_dw_direct (GGML_OP_CONV_2D_DW).
// Runs the same graph on CPU and Metal, compares results element-by-element.

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

static std::vector<float> run_conv_2d_dw(ggml_backend_t backend,
                                          const std::vector<float>& kernel_data,
                                          const std::vector<float>& input_data,
                                          int KW, int KH, int C,
                                          int W, int H, int N,
                                          int s0, int s1, int p0, int p1,
                                          int d0, int d1,
                                          int* out_w, int* out_h) {
    size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    struct ggml_init_params params = {ctx_size, nullptr, true};
    struct ggml_context* ctx = ggml_init(params);

    // Kernel: [KW, KH, 1, C]
    auto* knl = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, KW, KH, 1, C);
    ggml_set_name(knl, "kernel");
    ggml_set_input(knl);

    // Input: [W, H, C, N]
    auto* inp = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, N);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // Depthwise conv
    auto* out = ggml_conv_2d_dw_direct(ctx, knl, inp, s0, s1, p0, p1, d0, d1);
    ggml_set_name(out, "output");
    ggml_set_output(out);

    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "FAIL: gallocr_alloc_graph failed\n");
        exit(1);
    }

    ggml_backend_tensor_set(knl, kernel_data.data(), 0, kernel_data.size() * sizeof(float));
    ggml_backend_tensor_set(inp, input_data.data(), 0, input_data.size() * sizeof(float));

    ggml_backend_graph_compute(backend, graph);

    *out_w = (int)out->ne[0];
    *out_h = (int)out->ne[1];
    int64_t total = ggml_nelements(out);

    std::vector<float> output(total);
    ggml_backend_tensor_get(out, output.data(), 0, total * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return output;
}

static bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b,
                            const char* label, float tol = 1e-5f) {
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
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    ggml_backend_t metal_backend = ggml_backend_metal_init();
    if (!metal_backend) {
        fprintf(stderr, "FAIL: could not init Metal backend\n");
        return 1;
    }

    bool all_pass = true;

    // ── Test 1: sam3 exact dimensions (K=7, C=256, 72x72, s=1, p=3, d=1) ──
    {
        fprintf(stderr, "\n=== Test 1: sam3 dims (K=7, C=256, 72x72, s=1, p=3) ===\n");
        const int KW = 7, KH = 7, C = 256, W = 72, H = 72, N = 1;
        const int s0 = 1, s1 = 1, p0 = 3, p1 = 3, d0 = 1, d1 = 1;

        std::vector<float> kernel_data(KW * KH * 1 * C);
        std::vector<float> input_data(W * H * C * N);
        for (size_t i = 0; i < kernel_data.size(); i++)
            kernel_data[i] = 0.01f * ((float)(i % 97) - 48.0f);
        for (size_t i = 0; i < input_data.size(); i++)
            input_data[i] = 0.001f * ((float)(i % 9973) - 4986.0f);

        int ow_cpu, oh_cpu, ow_metal, oh_metal;
        auto cpu_out = run_conv_2d_dw(cpu_backend, kernel_data, input_data,
                                       KW, KH, C, W, H, N, s0, s1, p0, p1, d0, d1,
                                       &ow_cpu, &oh_cpu);
        auto metal_out = run_conv_2d_dw(metal_backend, kernel_data, input_data,
                                         KW, KH, C, W, H, N, s0, s1, p0, p1, d0, d1,
                                         &ow_metal, &oh_metal);

        fprintf(stderr, "  output shape: [%d, %d, %d, %d]\n", ow_cpu, oh_cpu, C, N);
        all_pass &= compare_vectors(cpu_out, metal_out, "sam3_conv_dw", 1e-4f);
    }

    // ── Test 2: small case (K=3, C=4, 8x8, s=1, p=1) ──────────────────
    {
        fprintf(stderr, "\n=== Test 2: small (K=3, C=4, 8x8, s=1, p=1) ===\n");
        const int KW = 3, KH = 3, C = 4, W = 8, H = 8, N = 1;
        const int s0 = 1, s1 = 1, p0 = 1, p1 = 1, d0 = 1, d1 = 1;

        std::vector<float> kernel_data(KW * KH * 1 * C);
        std::vector<float> input_data(W * H * C * N);
        for (size_t i = 0; i < kernel_data.size(); i++)
            kernel_data[i] = (float)(i + 1) * 0.1f;
        for (size_t i = 0; i < input_data.size(); i++)
            input_data[i] = (float)(i + 1) * 0.01f;

        int ow_cpu, oh_cpu, ow_metal, oh_metal;
        auto cpu_out = run_conv_2d_dw(cpu_backend, kernel_data, input_data,
                                       KW, KH, C, W, H, N, s0, s1, p0, p1, d0, d1,
                                       &ow_cpu, &oh_cpu);
        auto metal_out = run_conv_2d_dw(metal_backend, kernel_data, input_data,
                                         KW, KH, C, W, H, N, s0, s1, p0, p1, d0, d1,
                                         &ow_metal, &oh_metal);

        fprintf(stderr, "  output shape: [%d, %d, %d, %d]\n", ow_cpu, oh_cpu, C, N);
        all_pass &= compare_vectors(cpu_out, metal_out, "small_conv_dw", 1e-5f);
    }

    // ── Test 3: stride=2, dilation=2 ────────────────────────────────────
    {
        fprintf(stderr, "\n=== Test 3: stride=2, dilation=2 (K=3, C=16, 32x32) ===\n");
        const int KW = 3, KH = 3, C = 16, W = 32, H = 32, N = 1;
        const int s0 = 2, s1 = 2, p0 = 2, p1 = 2, d0 = 2, d1 = 2;

        std::vector<float> kernel_data(KW * KH * 1 * C);
        std::vector<float> input_data(W * H * C * N);
        for (size_t i = 0; i < kernel_data.size(); i++)
            kernel_data[i] = sinf((float)i * 0.1f);
        for (size_t i = 0; i < input_data.size(); i++)
            input_data[i] = cosf((float)i * 0.01f);

        int ow_cpu, oh_cpu, ow_metal, oh_metal;
        auto cpu_out = run_conv_2d_dw(cpu_backend, kernel_data, input_data,
                                       KW, KH, C, W, H, N, s0, s1, p0, p1, d0, d1,
                                       &ow_cpu, &oh_cpu);
        auto metal_out = run_conv_2d_dw(metal_backend, kernel_data, input_data,
                                         KW, KH, C, W, H, N, s0, s1, p0, p1, d0, d1,
                                         &ow_metal, &oh_metal);

        fprintf(stderr, "  output shape: [%d, %d, %d, %d]\n", ow_cpu, oh_cpu, C, N);
        all_pass &= compare_vectors(cpu_out, metal_out, "strided_dilated", 1e-5f);
    }

    // ── Test 4: no padding ──────────────────────────────────────────────
    {
        fprintf(stderr, "\n=== Test 4: no padding (K=5, C=8, 16x16) ===\n");
        const int KW = 5, KH = 5, C = 8, W = 16, H = 16, N = 2;
        const int s0 = 1, s1 = 1, p0 = 0, p1 = 0, d0 = 1, d1 = 1;

        std::vector<float> kernel_data(KW * KH * 1 * C);
        std::vector<float> input_data(W * H * C * N);
        for (size_t i = 0; i < kernel_data.size(); i++)
            kernel_data[i] = 0.02f * ((float)(i % 51) - 25.0f);
        for (size_t i = 0; i < input_data.size(); i++)
            input_data[i] = 0.005f * ((float)(i % 199) - 99.0f);

        int ow_cpu, oh_cpu, ow_metal, oh_metal;
        auto cpu_out = run_conv_2d_dw(cpu_backend, kernel_data, input_data,
                                       KW, KH, C, W, H, N, s0, s1, p0, p1, d0, d1,
                                       &ow_cpu, &oh_cpu);
        auto metal_out = run_conv_2d_dw(metal_backend, kernel_data, input_data,
                                         KW, KH, C, W, H, N, s0, s1, p0, p1, d0, d1,
                                         &ow_metal, &oh_metal);

        fprintf(stderr, "  output shape: [%d, %d, %d, %d]\n", ow_cpu, oh_cpu, C, N);
        all_pass &= compare_vectors(cpu_out, metal_out, "no_pad_batch2", 1e-5f);
    }

    ggml_backend_free(cpu_backend);
    ggml_backend_free(metal_backend);

    fprintf(stderr, "\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
#endif
}
