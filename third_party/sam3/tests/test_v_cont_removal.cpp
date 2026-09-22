// Validate that removing cont(permute) for V produces identical output.
// Builds the same ViT block twice — once with V cont, once without — on CPU.
// Output must be bit-identical.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cstdio>
#include <cmath>
#include <vector>

struct block_result {
    std::vector<float> output;
    int n_nodes;
};

static block_result run_vit_block(ggml_backend_t backend, bool use_v_cont) {
    const int E = 1024, HD = 64, NH = 16, MLP = 4736;
    const int W = 24, H = 24, NP = 9, N = W * H;
    const ggml_type WT = GGML_TYPE_F16;

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

    auto* x = inp(GGML_TYPE_F32, E, W, H, NP);
    auto* norm1_w = inp(GGML_TYPE_F32, E);
    auto* norm1_b = inp(GGML_TYPE_F32, E);
    auto* xn = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-6f), norm1_w), norm1_b);

    auto* qkv_w = inp(WT, E, 3*E);
    auto* qkv_b = inp(GGML_TYPE_F32, 3*E);
    auto* qkv = ggml_add(ctx, ggml_mul_mat(ctx, qkv_w, xn), qkv_b);

    qkv = ggml_reshape_4d(ctx, qkv, E, 3, N, NP);
    qkv = ggml_cont(ctx, ggml_permute(ctx, qkv, 0, 3, 1, 2));
    auto* Q = ggml_view_3d(ctx, qkv, E, N, NP, qkv->nb[1], qkv->nb[2], 0);
    auto* K = ggml_view_3d(ctx, qkv, E, N, NP, qkv->nb[1], qkv->nb[2], qkv->nb[3]);
    auto* V = ggml_view_3d(ctx, qkv, E, N, NP, qkv->nb[1], qkv->nb[2], 2*qkv->nb[3]);

    // Q and K: always use cont (they need it for RoPE)
    Q = ggml_reshape_4d(ctx, Q, HD, NH, N, NP);
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    Q = ggml_reshape_4d(ctx, Q, HD, N, NH, NP);

    K = ggml_reshape_4d(ctx, K, HD, NH, N, NP);
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    K = ggml_reshape_4d(ctx, K, HD, N, NH, NP);

    // V: the part we're testing
    if (use_v_cont) {
        // OLD path: reshape → permute → cont → reshape_3d → reshape_4d
        V = ggml_reshape_4d(ctx, V, HD, NH, N, NP);
        V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));
        V = ggml_reshape_3d(ctx, V, HD, N, NH * NP);
        V = ggml_reshape_4d(ctx, V, HD, N, NH, NP);
    } else {
        // NEW path: reshape → permute (no cont, no reshape)
        V = ggml_reshape_4d(ctx, V, HD, NH, N, NP);
        V = ggml_permute(ctx, V, 0, 2, 1, 3);
    }

    auto* attn = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, 1.0f/8.0f, 0, 0);
    auto* attn_r = ggml_reshape_4d(ctx, attn, E, W, H, NP);

    auto* proj_w = inp(WT, E, E);
    auto* proj_b = inp(GGML_TYPE_F32, E);
    auto* proj = ggml_add(ctx, ggml_mul_mat(ctx, proj_w, attn_r), proj_b);
    auto* res1 = ggml_add(ctx, x, proj);

    auto* norm2_w = inp(GGML_TYPE_F32, E);
    auto* norm2_b = inp(GGML_TYPE_F32, E);
    auto* xn2 = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, res1, 1e-6f), norm2_w), norm2_b);

    auto* fc1_w = inp(WT, E, MLP);
    auto* fc1_b = inp(GGML_TYPE_F32, MLP);
    auto* fc2_w = inp(WT, MLP, E);
    auto* fc2_b = inp(GGML_TYPE_F32, E);
    auto* h = ggml_add(ctx, ggml_mul_mat(ctx, fc1_w, xn2), fc1_b);
    h = ggml_gelu_erf(ctx, h);
    h = ggml_add(ctx, ggml_mul_mat(ctx, fc2_w, h), fc2_b);
    auto* out = ggml_add(ctx, res1, h);
    ggml_set_output(out);

    auto* graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, out);

    auto* galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_reserve(galloc, graph);
    ggml_gallocr_alloc_graph(galloc, graph);

    // Fill with deterministic data
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

    ggml_backend_graph_compute(backend, graph);

    int64_t nel = ggml_nelements(out);
    block_result r;
    r.output.resize(nel);
    r.n_nodes = ggml_graph_n_nodes(graph);
    ggml_backend_tensor_get(out, r.output.data(), 0, nel * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return r;
}

static bool compare(const std::vector<float>& a, const std::vector<float>& b,
                    const char* label, float tol) {
    float max_diff = 0;
    int max_idx = 0;
    for (size_t i = 0; i < a.size(); i++) {
        float d = std::fabs(a[i] - b[i]);
        if (d > max_diff) { max_diff = d; max_idx = (int)i; }
    }
    bool ok = max_diff <= tol;
    fprintf(stderr, "  %s [%s]: max_diff=%.8f at [%d] (a=%.6f b=%.6f) (%zu elements)\n",
            ok ? "PASS" : "FAIL", label, max_diff, max_idx,
            a[max_idx], b[max_idx], a.size());
    return ok;
}

int main() {
    auto* cpu = ggml_backend_cpu_init();
    bool all_pass = true;

    // TEST 1: CPU with V cont vs CPU without V cont — must be IDENTICAL
    fprintf(stderr, "\n=== CPU: with V cont vs without V cont ===\n");
    auto cpu_with = run_vit_block(cpu, true);
    auto cpu_without = run_vit_block(cpu, false);
    fprintf(stderr, "  Nodes: with=%d, without=%d\n", cpu_with.n_nodes, cpu_without.n_nodes);
    all_pass &= compare(cpu_with.output, cpu_without.output, "CPU with_cont vs without_cont", 0.0f);

#ifdef GGML_USE_METAL
    auto* metal = ggml_backend_metal_init();
    if (metal) {
        // TEST 2: Metal with V cont vs Metal without V cont — must be IDENTICAL
        fprintf(stderr, "\n=== Metal: with V cont vs without V cont ===\n");
        auto metal_with = run_vit_block(metal, true);
        auto metal_without = run_vit_block(metal, false);
        fprintf(stderr, "  Nodes: with=%d, without=%d\n", metal_with.n_nodes, metal_without.n_nodes);
        all_pass &= compare(metal_with.output, metal_without.output, "Metal with_cont vs without_cont", 0.0f);

        // TEST 3: CPU without vs Metal without — F16 precision diff (informational)
        fprintf(stderr, "\n=== CPU vs Metal (both without V cont) ===\n");
        compare(cpu_without.output, metal_without.output, "CPU vs Metal (informational)", 50.0f);

        ggml_backend_free(metal);
    }
#endif

    ggml_backend_free(cpu);
    fprintf(stderr, "\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
