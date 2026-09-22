// Measure Metal dispatch overhead and its impact on SAM3 ViT performance.
// Proves WHY Metal is slower than CPU for the full 2524-node ViT graph.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

static double run_graph(ggml_backend_t backend, ggml_context* ctx, ggml_cgraph* graph,
                        const std::vector<ggml_tensor*>& inputs, int warmup, int repeats) {
    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "FAIL: alloc\n");
        ggml_gallocr_free(galloc);
        return -1;
    }
    for (auto* t : inputs) {
        int64_t nel = ggml_nelements(t);
        if (t->type == GGML_TYPE_F32) {
            std::vector<float> d(nel, 0.1f);
            ggml_backend_tensor_set(t, d.data(), 0, nel * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<uint16_t> d(nel);
            for (int64_t j = 0; j < nel; j++) d[j] = ggml_fp32_to_fp16(0.1f);
            ggml_backend_tensor_set(t, d.data(), 0, nel * sizeof(uint16_t));
        }
    }
    if (ggml_backend_is_cpu(backend))
        ggml_backend_cpu_set_n_threads(backend, 8);

    for (int i = 0; i < warmup; i++)
        ggml_backend_graph_compute(backend, graph);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; i++)
        ggml_backend_graph_compute(backend, graph);
    auto t1 = std::chrono::high_resolution_clock::now();
    ggml_gallocr_free(galloc);
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TEST 1: Pure dispatch overhead — chain of N trivial add ops
// ═══════════════════════════════════════════════════════════════════════════════
static void test_dispatch_overhead(ggml_backend_t cpu, ggml_backend_t metal) {
    fprintf(stderr, "\n=== TEST 1: Dispatch overhead (chain of N add ops on [1024,72,72] tensors) ===\n\n");
    fprintf(stderr, "  %6s  %10s  %10s  %12s\n", "N_ops", "CPU (ms)", "Metal (ms)", "Overhead/op");

    int counts[] = {1, 10, 50, 100, 250, 500, 1000, 2500};
    double metal_1op = 0;

    for (int N : counts) {
        int max_tensors = N + 20;
        int max_graph = N + 20;

        auto build_and_run = [&](ggml_backend_t backend) -> double {
            size_t ctx_size = ggml_tensor_overhead() * max_tensors + ggml_graph_overhead_custom(max_graph, false);
            ggml_init_params params = {ctx_size, nullptr, true};
            auto* ctx = ggml_init(params);

            auto* a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 72, 72); ggml_set_input(a);
            auto* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 72, 72); ggml_set_input(b);
            auto* x = ggml_add(ctx, a, b);
            for (int i = 1; i < N; i++)
                x = ggml_add(ctx, x, b);
            ggml_set_output(x);

            auto* graph = ggml_new_graph_custom(ctx, max_graph, false);
            ggml_build_forward_expand(graph, x);

            std::vector<ggml_tensor*> inputs = {a, b};
            double ms = run_graph(backend, ctx, graph, inputs, 1, 3);
            ggml_free(ctx);
            return ms;
        };

        double cpu_ms = build_and_run(cpu);
        double metal_ms = build_and_run(metal);

        if (N == 1) metal_1op = metal_ms;
        double overhead_per_op = (N > 1) ? (metal_ms - metal_1op) / (N - 1) : 0;

        fprintf(stderr, "  %6d  %8.2f ms  %8.2f ms  %10.3f ms\n",
                N, cpu_ms, metal_ms, overhead_per_op);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TEST 2: Single ViT block as ONE graph vs sum of individual ops
// ═══════════════════════════════════════════════════════════════════════════════
static void test_vit_block_graph(ggml_backend_t cpu, ggml_backend_t metal) {
    fprintf(stderr, "\n=== TEST 2: One ViT block as single graph (windowed, all ops together) ===\n\n");

    const int E = 1024, HD = 64, NH = 16, MLP = 4736;
    const int W = 24, H = 24, NP = 9;  // windowed
    const ggml_type WT = GGML_TYPE_F16;

    auto run_block = [&](ggml_backend_t backend) -> std::pair<double, std::vector<float>> {
        size_t ctx_size = ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false);
        ggml_init_params params = {ctx_size, nullptr, true};
        auto* ctx = ggml_init(params);
        std::vector<ggml_tensor*> inputs;

        auto inp = [&](ggml_type type, int64_t d0, int64_t d1=1, int64_t d2=1, int64_t d3=1) {
            auto* t = ggml_new_tensor_4d(ctx, type, d0, d1, d2, d3);
            ggml_set_input(t);
            inputs.push_back(t);
            return t;
        };

        // Input + residual
        auto* x = inp(GGML_TYPE_F32, E, W, H, NP);

        // --- Norm1 ---
        auto* norm1_w = inp(GGML_TYPE_F32, E);
        auto* norm1_b = inp(GGML_TYPE_F32, E);
        auto* xn = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-6f), norm1_w), norm1_b);

        // --- QKV matmul ---
        auto* qkv_w = inp(WT, E, 3*E);
        auto* qkv_b = inp(GGML_TYPE_F32, 3*E);
        auto* qkv = ggml_add(ctx, ggml_mul_mat(ctx, qkv_w, xn), qkv_b);

        // --- Reshape + split Q/K/V ---
        int N = W * H;
        qkv = ggml_reshape_4d(ctx, qkv, E, 3, N, NP);
        qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 3, 1, 2)); // expensive cont #1
        auto* Q = ggml_view_3d(ctx, qkv, E, N, NP, qkv->nb[1], qkv->nb[2], 0);
        auto* K = ggml_view_3d(ctx, qkv, E, N, NP, qkv->nb[1], qkv->nb[2], qkv->nb[3]);
        auto* V = ggml_view_3d(ctx, qkv, E, N, NP, qkv->nb[1], qkv->nb[2], 2*qkv->nb[3]);

        // --- Multi-head reshape + permute + cont (×3) ---
        Q = ggml_reshape_4d(ctx, Q, HD, NH, N, NP);
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3)); // expensive cont #2
        Q = ggml_reshape_4d(ctx, Q, HD, N, NH, NP);

        K = ggml_reshape_4d(ctx, K, HD, NH, N, NP);
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3)); // expensive cont #3
        K = ggml_reshape_4d(ctx, K, HD, N, NH, NP);

        V = ggml_reshape_4d(ctx, V, HD, NH, N, NP);
        V = ggml_permute(ctx, V, 0, 2, 1, 3); // non-contiguous view — flash_attn uses strides

        // --- Flash attention ---
        auto* attn = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, 1.0f/8.0f, 0, 0);
        auto* attn_r = ggml_reshape_4d(ctx, attn, E, W, H, NP);

        // --- Output proj ---
        auto* proj_w = inp(WT, E, E);
        auto* proj_b = inp(GGML_TYPE_F32, E);
        auto* proj = ggml_add(ctx, ggml_mul_mat(ctx, proj_w, attn_r), proj_b);

        // --- Residual ---
        auto* res1 = ggml_add(ctx, x, proj);

        // --- Norm2 ---
        auto* norm2_w = inp(GGML_TYPE_F32, E);
        auto* norm2_b = inp(GGML_TYPE_F32, E);
        auto* xn2 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, res1, 1e-6f), norm2_w), norm2_b);

        // --- MLP ---
        auto* fc1_w = inp(WT, E, MLP);
        auto* fc1_b = inp(GGML_TYPE_F32, MLP);
        auto* fc2_w = inp(WT, MLP, E);
        auto* fc2_b = inp(GGML_TYPE_F32, E);

        auto* h = ggml_add(ctx, ggml_mul_mat(ctx, fc1_w, xn2), fc1_b);
        h = ggml_gelu_erf(ctx, h);
        h = ggml_add(ctx, ggml_mul_mat(ctx, fc2_w, h), fc2_b);

        // --- Residual ---
        auto* out = ggml_add(ctx, res1, h);
        ggml_set_output(out);

        auto* graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, out);

        int n_nodes = ggml_graph_n_nodes(graph);

        // Allocate and fill inputs
        auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_reserve(galloc, graph);
        ggml_gallocr_alloc_graph(galloc, graph);

        // Fill inputs with deterministic data
        for (auto* t : inputs) {
            int64_t nel = ggml_nelements(t);
            if (t->type == GGML_TYPE_F32) {
                std::vector<float> d(nel);
                for (int64_t j = 0; j < nel; j++) d[j] = 0.01f * ((j % 97) - 48);
                ggml_backend_tensor_set(t, d.data(), 0, nel * sizeof(float));
            } else {
                std::vector<uint16_t> d(nel);
                for (int64_t j = 0; j < nel; j++) d[j] = ggml_fp32_to_fp16(0.01f * ((j % 97) - 48));
                ggml_backend_tensor_set(t, d.data(), 0, nel * sizeof(uint16_t));
            }
        }

        if (ggml_backend_is_cpu(backend))
            ggml_backend_cpu_set_n_threads(backend, 8);

        // Warmup + timed run
        ggml_backend_graph_compute(backend, graph);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 5; i++)
            ggml_backend_graph_compute(backend, graph);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5;

        // Read output
        int64_t nel = ggml_nelements(out);
        std::vector<float> output(nel);
        ggml_backend_tensor_get(out, output.data(), 0, nel * sizeof(float));

        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        fprintf(stderr, "  %-20s  %7.2f ms  (%d nodes)\n",
                ggml_backend_is_cpu(backend) ? "CPU" : "Metal", ms, n_nodes);
        return std::make_pair(ms, output);
    };

    auto cpu_result = run_block(cpu);
    auto metal_result = run_block(metal);
    double cpu_block = cpu_result.first;
    double metal_block = metal_result.first;
    const auto& cpu_out = cpu_result.second;
    const auto& metal_out = metal_result.second;

    // Compare outputs
    float max_diff = 0;
    int max_idx = 0;
    for (size_t i = 0; i < cpu_out.size(); i++) {
        float d = std::fabs(cpu_out[i] - metal_out[i]);
        if (d > max_diff) { max_diff = d; max_idx = (int)i; }
    }
    fprintf(stderr, "  Output comparison: max_diff=%.6f at [%d] (cpu=%.6f metal=%.6f) — %s\n",
            max_diff, max_idx, cpu_out[max_idx], metal_out[max_idx],
            max_diff < 0.01f ? "PASS" : "FAIL");

    fprintf(stderr, "  Speedup: %.2fx\n", cpu_block / metal_block);
    fprintf(stderr, "\n  Extrapolated to 32 layers:\n");
    fprintf(stderr, "    CPU:   %.0f ms\n", cpu_block * 32);
    fprintf(stderr, "    Metal: %.0f ms\n", metal_block * 32);
    fprintf(stderr, "    Speedup: %.2fx\n", (cpu_block * 32) / (metal_block * 32));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TEST 3: Scale from 1 to 32 ViT blocks in ONE graph
// ═══════════════════════════════════════════════════════════════════════════════
static void test_scaling(ggml_backend_t cpu, ggml_backend_t metal) {
    fprintf(stderr, "\n=== TEST 3: Scaling — N chained matmul+add blocks in one graph ===\n");
    fprintf(stderr, "  (Each block: F16 matmul [1024,1024]x[1024,24,24,9] + add + norm + gelu)\n\n");
    fprintf(stderr, "  %6s  %6s  %10s  %10s  %8s\n", "Blocks", "Nodes", "CPU (ms)", "Metal (ms)", "Speedup");

    const int E = 1024;
    const ggml_type WT = GGML_TYPE_F16;
    int block_counts[] = {1, 4, 8, 16, 32};

    for (int nblocks : block_counts) {
        int max_nodes = nblocks * 20 + 20;
        size_t ctx_size = ggml_tensor_overhead() * (max_nodes + 50) +
                          ggml_graph_overhead_custom(max_nodes + 50, false);
        ggml_init_params params = {ctx_size, nullptr, true};

        auto run_on = [&](ggml_backend_t backend) -> std::pair<double, int> {
            auto* ctx = ggml_init(params);
            std::vector<ggml_tensor*> inputs;

            auto inp = [&](ggml_type type, int64_t d0, int64_t d1=1, int64_t d2=1, int64_t d3=1) {
                auto* t = ggml_new_tensor_4d(ctx, type, d0, d1, d2, d3);
                ggml_set_input(t);
                inputs.push_back(t);
                return t;
            };

            auto* x = inp(GGML_TYPE_F32, E, 24, 24, 9);

            for (int i = 0; i < nblocks; i++) {
                auto* w = inp(WT, E, E);
                auto* b = inp(GGML_TYPE_F32, E);
                auto* nw = inp(GGML_TYPE_F32, E);
                auto* nb_ = inp(GGML_TYPE_F32, E);
                x = ggml_add(ctx, ggml_mul_mat(ctx, w, x), b);
                x = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-6f), nw), nb_);
                x = ggml_gelu_erf(ctx, x);
            }
            ggml_set_output(x);

            auto* graph = ggml_new_graph_custom(ctx, max_nodes + 50, false);
            ggml_build_forward_expand(graph, x);
            int n_nodes = ggml_graph_n_nodes(graph);

            double ms = run_graph(backend, ctx, graph, inputs, 2, 5);
            ggml_free(ctx);
            return {ms, n_nodes};
        };

        double cpu_ms; int cpu_nodes;
        { auto r = run_on(cpu); cpu_ms = r.first; cpu_nodes = r.second; }
        double metal_ms; int metal_nodes;
        { auto r = run_on(metal); metal_ms = r.first; metal_nodes = r.second; }
        fprintf(stderr, "  %6d  %6d  %8.2f ms  %8.2f ms  %7.2fx\n",
                nblocks, cpu_nodes, cpu_ms, metal_ms, cpu_ms / metal_ms);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  TEST 4: cont(permute) cost in isolation vs inside a larger graph
// ═══════════════════════════════════════════════════════════════════════════════
static void test_cont_in_graph(ggml_backend_t cpu, ggml_backend_t metal) {
    fprintf(stderr, "\n=== TEST 4: N cont(permute) ops in ONE graph ===\n");
    fprintf(stderr, "  (Measures whether cont overhead compounds in large graphs)\n\n");
    fprintf(stderr, "  %6s  %10s  %10s  %10s\n", "N_cont", "CPU (ms)", "Metal (ms)", "Metal/op");

    int counts[] = {1, 4, 16, 32, 64, 96};

    for (int N : counts) {
        int max_nodes = N * 3 + 10;
        size_t ctx_size = ggml_tensor_overhead() * (max_nodes + 20) +
                          ggml_graph_overhead_custom(max_nodes + 20, false);
        ggml_init_params params = {ctx_size, nullptr, true};

        auto run_on = [&](ggml_backend_t backend) -> double {
            auto* ctx = ggml_init(params);
            std::vector<ggml_tensor*> inputs;

            // Chain: permute → cont → permute → cont → ...
            auto* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 576, 144);
            ggml_set_input(x);
            inputs.push_back(x);

            ggml_tensor* last = x;
            for (int i = 0; i < N; i++) {
                last = ggml_cont(ctx, ggml_permute(ctx, last, 0, 2, 1, 3));
            }
            ggml_set_output(last);

            auto* graph = ggml_new_graph_custom(ctx, max_nodes + 20, false);
            ggml_build_forward_expand(graph, last);
            double ms = run_graph(backend, ctx, graph, inputs, 2, 3);
            ggml_free(ctx);
            return ms;
        };

        double cpu_ms = run_on(cpu);
        double metal_ms = run_on(metal);
        fprintf(stderr, "  %6d  %8.2f ms  %8.2f ms  %8.2f ms\n",
                N, cpu_ms, metal_ms, metal_ms / N);
    }
}

int main() {
#ifndef GGML_USE_METAL
    fprintf(stderr, "Metal not available\n"); return 1;
#else
    auto* cpu = ggml_backend_cpu_init();
    auto* metal = ggml_backend_metal_init();
    if (!metal) { fprintf(stderr, "No Metal\n"); return 1; }

    test_dispatch_overhead(cpu, metal);
    test_vit_block_graph(cpu, metal);
    test_scaling(cpu, metal);
    test_cont_in_graph(cpu, metal);

    ggml_backend_free(cpu);
    ggml_backend_free(metal);
    return 0;
#endif
}
