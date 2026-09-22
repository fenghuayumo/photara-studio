// Profile individual ggml ops at SAM3 ViT shapes on CPU vs Metal.
// Each op gets its own tiny graph — runs in seconds, not minutes.

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
#include <functional>
#include <string>
#include <vector>

using build_fn = std::function<std::pair<ggml_tensor*, std::vector<ggml_tensor*>>(ggml_context*)>;

static double run_bench(ggml_backend_t backend, build_fn build,
                        int warmup = 2, int repeats = 5) {
    size_t ctx_size = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    struct ggml_init_params params = {ctx_size, nullptr, true};
    auto* ctx = ggml_init(params);

    auto result = build(ctx);
    auto* out = result.first;
    auto& inputs = result.second;
    ggml_set_output(out);

    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "FAIL: alloc\n");
        ggml_gallocr_free(galloc); ggml_free(ctx);
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

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeats;
    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return ms;
}

struct op_bench { const char* name; int count; double cpu_ms; double metal_ms; };

// Helper macros for input tensor creation
#define INP_F32(...) [&]{ auto* t = ggml_new_tensor(__VA_ARGS__); ggml_set_name(t,"i"); ggml_set_input(t); return t; }()
#define INP(ctx, type, ...) [&]{ int64_t ne[] = {__VA_ARGS__}; auto* t = ggml_new_tensor(ctx, type, sizeof(ne)/sizeof(ne[0]), ne); ggml_set_input(t); return t; }()

int main() {
#ifndef GGML_USE_METAL
    fprintf(stderr, "Metal not available\n"); return 1;
#else
    auto* cpu = ggml_backend_cpu_init();
    auto* metal = ggml_backend_metal_init();
    if (!metal) { fprintf(stderr, "No Metal\n"); return 1; }

    const int E = 1024, MLP = 4736, HD = 64, NH = 16;
    const int W_win = 24, H_win = 24, NP = 9;
    const int W_glb = 72, H_glb = 72;
    const int N_win = W_win * H_win;
    const int N_glb = W_glb * H_glb;
    const ggml_type WT = GGML_TYPE_F16;

    std::vector<op_bench> results;

    auto bench = [&](const char* name, int count, build_fn build) {
        double cpu_ms = run_bench(cpu, build);
        double metal_ms = run_bench(metal, build);
        results.push_back({name, count, cpu_ms, metal_ms});
        fprintf(stderr, "  %-45s  CPU:%7.2f ms  Metal:%7.2f ms  (x%d → CPU %5.0f / Metal %5.0f)\n",
                name, cpu_ms, metal_ms, count, cpu_ms*count, metal_ms*count);
    };

    fprintf(stderr, "\n=== SAM3 ViT Op Profiling (F16 weights, F32 activations, CPU 8 threads) ===\n\n");

    // ── Matmuls (windowed: 28 layers × 4 = 112) ────────────────────────
    bench("matmul QKV win [1024,3072]x[1024,24,24,9]", 28, [&](ggml_context* c) {
        auto* w = ggml_new_tensor_2d(c, WT, E, 3*E);       ggml_set_input(w);
        auto* x = ggml_new_tensor_4d(c, GGML_TYPE_F32, E, W_win, H_win, NP); ggml_set_input(x);
        return std::make_pair(ggml_mul_mat(c, w, x), std::vector<ggml_tensor*>{w, x});
    });
    bench("matmul proj win [1024,1024]x[1024,24,24,9]", 28, [&](ggml_context* c) {
        auto* w = ggml_new_tensor_2d(c, WT, E, E);         ggml_set_input(w);
        auto* x = ggml_new_tensor_4d(c, GGML_TYPE_F32, E, W_win, H_win, NP); ggml_set_input(x);
        return std::make_pair(ggml_mul_mat(c, w, x), std::vector<ggml_tensor*>{w, x});
    });
    bench("matmul fc1 win [1024,4736]x[1024,24,24,9]", 28, [&](ggml_context* c) {
        auto* w = ggml_new_tensor_2d(c, WT, E, MLP);       ggml_set_input(w);
        auto* x = ggml_new_tensor_4d(c, GGML_TYPE_F32, E, W_win, H_win, NP); ggml_set_input(x);
        return std::make_pair(ggml_mul_mat(c, w, x), std::vector<ggml_tensor*>{w, x});
    });
    bench("matmul fc2 win [4736,1024]x[4736,24,24,9]", 28, [&](ggml_context* c) {
        auto* w = ggml_new_tensor_2d(c, WT, MLP, E);       ggml_set_input(w);
        auto* x = ggml_new_tensor_4d(c, GGML_TYPE_F32, MLP, W_win, H_win, NP); ggml_set_input(x);
        return std::make_pair(ggml_mul_mat(c, w, x), std::vector<ggml_tensor*>{w, x});
    });

    // ── Matmuls (global: 4 layers × 4 = 16) ────────────────────────────
    bench("matmul QKV glb [1024,3072]x[1024,72,72,1]", 4, [&](ggml_context* c) {
        auto* w = ggml_new_tensor_2d(c, WT, E, 3*E);       ggml_set_input(w);
        auto* x = ggml_new_tensor_4d(c, GGML_TYPE_F32, E, W_glb, H_glb, 1); ggml_set_input(x);
        return std::make_pair(ggml_mul_mat(c, w, x), std::vector<ggml_tensor*>{w, x});
    });
    bench("matmul fc1 glb [1024,4736]x[1024,72,72,1]", 4, [&](ggml_context* c) {
        auto* w = ggml_new_tensor_2d(c, WT, E, MLP);       ggml_set_input(w);
        auto* x = ggml_new_tensor_4d(c, GGML_TYPE_F32, E, W_glb, H_glb, 1); ggml_set_input(x);
        return std::make_pair(ggml_mul_mat(c, w, x), std::vector<ggml_tensor*>{w, x});
    });

    // ── Flash attention ─────────────────────────────────────────────────
    bench("flash_attn win HD=64 N=576 NH=16 B=9", 28, [&](ggml_context* c) {
        auto* Q = ggml_new_tensor_4d(c, GGML_TYPE_F32, HD, N_win, NH, NP); ggml_set_input(Q);
        auto* K = ggml_new_tensor_4d(c, GGML_TYPE_F32, HD, N_win, NH, NP); ggml_set_input(K);
        auto* V = ggml_new_tensor_4d(c, GGML_TYPE_F32, HD, N_win, NH, NP); ggml_set_input(V);
        return std::make_pair(ggml_flash_attn_ext(c, Q, K, V, nullptr, 1.0f/8.0f, 0, 0),
                              std::vector<ggml_tensor*>{Q, K, V});
    });
    bench("flash_attn glb HD=64 N=5184 NH=16 B=1", 4, [&](ggml_context* c) {
        auto* Q = ggml_new_tensor_4d(c, GGML_TYPE_F32, HD, N_glb, NH, 1); ggml_set_input(Q);
        auto* K = ggml_new_tensor_4d(c, GGML_TYPE_F32, HD, N_glb, NH, 1); ggml_set_input(K);
        auto* V = ggml_new_tensor_4d(c, GGML_TYPE_F32, HD, N_glb, NH, 1); ggml_set_input(V);
        return std::make_pair(ggml_flash_attn_ext(c, Q, K, V, nullptr, 1.0f/8.0f, 0, 0),
                              std::vector<ggml_tensor*>{Q, K, V});
    });

    // ── Layer norm (norm + mul + add) ───────────────────────────────────
    bench("layer_norm [1024, 72, 72]", 64, [&](ggml_context* c) {
        auto* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, E, W_glb, H_glb); ggml_set_input(x);
        auto* w = ggml_new_tensor_1d(c, GGML_TYPE_F32, E); ggml_set_input(w);
        auto* b = ggml_new_tensor_1d(c, GGML_TYPE_F32, E); ggml_set_input(b);
        auto* n = ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-6f), w), b);
        return std::make_pair(n, std::vector<ggml_tensor*>{x, w, b});
    });

    // ── GELU ────────────────────────────────────────────────────────────
    bench("gelu_erf [4736, 24, 24, 9]", 28, [&](ggml_context* c) {
        auto* x = ggml_new_tensor_4d(c, GGML_TYPE_F32, MLP, W_win, H_win, NP); ggml_set_input(x);
        return std::make_pair(ggml_gelu_erf(c, x), std::vector<ggml_tensor*>{x});
    });

    // ── Add ─────────────────────────────────────────────────────────────
    bench("add [1024, 72, 72]", 128, [&](ggml_context* c) {
        auto* a = ggml_new_tensor_3d(c, GGML_TYPE_F32, E, W_glb, H_glb); ggml_set_input(a);
        auto* b = ggml_new_tensor_3d(c, GGML_TYPE_F32, E, W_glb, H_glb); ggml_set_input(b);
        return std::make_pair(ggml_add(c, a, b), std::vector<ggml_tensor*>{a, b});
    });

    // ── Win part / unpart ───────────────────────────────────────────────
    bench("win_part [1024,72,72] -> [1024,24,24,9]", 28, [&](ggml_context* c) {
        auto* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, E, W_glb, H_glb); ggml_set_input(x);
        return std::make_pair(ggml_win_part(c, x, 24), std::vector<ggml_tensor*>{x});
    });
    bench("win_unpart [1024,24,24,9] -> [1024,72,72]", 28, [&](ggml_context* c) {
        auto* x = ggml_new_tensor_4d(c, GGML_TYPE_F32, E, W_win, H_win, NP); ggml_set_input(x);
        return std::make_pair(ggml_win_unpart(c, x, W_glb, H_glb, 24), std::vector<ggml_tensor*>{x});
    });

    // ── Cont (permute + make contiguous — happens ~96x in ViT) ──────────
    bench("cont(permute) [1024, 576, 144]", 96, [&](ggml_context* c) {
        auto* x = ggml_new_tensor_3d(c, GGML_TYPE_F32, E, N_win, NH*NP); ggml_set_input(x);
        return std::make_pair(ggml_cont(c, ggml_permute(c, x, 0, 2, 1, 3)),
                              std::vector<ggml_tensor*>{x});
    });

    // ── Summary ─────────────────────────────────────────────────────────
    fprintf(stderr, "\n=== Estimated Total ViT Cost ===\n\n");
    fprintf(stderr, "  %-45s  %10s  %10s  %8s\n", "Op", "CPU total", "Metal total", "Speedup");
    fprintf(stderr, "  %-45s  %10s  %10s  %8s\n",
            "---------------------------------------------", "---------", "-----------", "-------");

    double total_cpu = 0, total_metal = 0;
    for (auto& r : results) {
        double ct = r.cpu_ms * r.count;
        double mt = r.metal_ms * r.count;
        total_cpu += ct;
        total_metal += mt;
        fprintf(stderr, "  %-45s  %7.0f ms  %9.0f ms  %7.2fx\n", r.name, ct, mt, ct/mt);
    }
    fprintf(stderr, "  %-45s  %7.0f ms  %9.0f ms  %7.2fx\n",
            "TOTAL (estimated ViT)", total_cpu, total_metal, total_cpu/total_metal);

    ggml_backend_free(cpu);
    ggml_backend_free(metal);
    return 0;
#endif
}
