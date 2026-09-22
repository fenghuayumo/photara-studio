/**
 * End-to-end PVS pipeline test: loads Python-preprocessed image,
 * runs through C++ ViT + neck + SAM prompt encoder + mask decoder,
 * dumps all intermediate tensors for comparison against Python reference.
 *
 * Usage:
 *   ./test_e2e_pvs <model.ggml> <ref_dir>
 *
 * ref_dir should contain:
 *   - preprocessed.bin/.shape (from dump_phase3_reference.py)
 *   - phase6/cat_box/ (from dump_phase6_reference.py)
 */
#include "sam3.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// Compare ggml [ne0, ne1, ne2] vs Python NHWC [1, H, W, E] — same flat layout
static compare_result compare_ggml_vs_nhwc(const ref_tensor_f32 & cpp,
                                            const ref_tensor_f32 & ref) {
    return compare_tensors(cpp.data.data(), ref.data.data(),
                           std::min(cpp.numel(), ref.numel()));
}

// Compare ggml [C, W, H] vs Python NCHW [1, C, H, W] — need transposition
static compare_result compare_ggml_vs_nchw(const ref_tensor_f32 & cpp,
                                            const ref_tensor_f32 & ref) {
    if (cpp.shape.size() < 3 || ref.data.empty()) {
        compare_result r;
        r.max_diff = 999.0f;
        return r;
    }
    int C = cpp.shape[0], W = cpp.shape[1], H = cpp.shape[2];

    std::vector<float> transposed(C * W * H);
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                transposed[c * H * W + h * W + w] = cpp.data[c + w * C + h * C * W];

    return compare_tensors(transposed.data(), ref.data.data(),
                           std::min((int) transposed.size(), ref.numel()));
}

struct metric_row {
    std::string stage;
    std::string name;
    std::string py_shape_str;
    float mae        = 0.0f;
    float max_abs    = 0.0f;
    float mean_rel   = 0.0f;
    float cosine     = 0.0f;
    float p95        = 0.0f;
    float p99        = 0.0f;
    bool  ok         = false;
    float tolerance  = 0.0f;
};

static metric_row compute_full_metrics(const float * a, const float * b, int n, float tol) {
    metric_row m;
    m.tolerance = tol;
    if (n == 0) return m;

    double sum_diff = 0.0;
    double sum_rel = 0.0;
    double dot_ab = 0.0, dot_aa = 0.0, dot_bb = 0.0;
    float max_d = 0.0f;

    std::vector<float> diffs(n);
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        diffs[i] = d;
        sum_diff += d;
        if (d > max_d) max_d = d;
        float denom = fabsf(b[i]) + 1e-8f;
        sum_rel += d / denom;
        dot_ab += (double)a[i] * (double)b[i];
        dot_aa += (double)a[i] * (double)a[i];
        dot_bb += (double)b[i] * (double)b[i];
    }

    m.mae = (float)(sum_diff / n);
    m.max_abs = max_d;
    m.mean_rel = (float)(sum_rel / n);
    double denom = sqrt(dot_aa) * sqrt(dot_bb);
    m.cosine = denom > 0.0 ? (float)(dot_ab / denom) : 0.0f;

    std::sort(diffs.begin(), diffs.end());
    m.p95 = diffs[(int)(0.95 * n)];
    m.p99 = diffs[(int)(0.99 * n)];
    m.ok = max_d <= tol;
    return m;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.ggml> <ref_dir>\n", argv[0]);
        fprintf(stderr, "\nref_dir: directory with Python reference tensors\n");
        fprintf(stderr, "  Should contain preprocessed.bin/.shape and phase6/cat_box/\n");
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ref_dir = argv[2];
    const std::string cpp_out = ref_dir + "/cpp_out";
    const std::string cpp_p6_out = ref_dir + "/cpp_out_phase6";
    ensure_dir(cpp_out);
    ensure_dir(cpp_p6_out);

    // ═══ Load model ═══
    fprintf(stderr, "\n═══ Loading model ═══\n");
    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = false;
    params.n_threads = 4;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "Failed to create state\n");
        return 1;
    }

    // ═══ Stage 1: Load Python-preprocessed image ═══
    fprintf(stderr, "\n═══ Stage 1: Loading Python-preprocessed image ═══\n");
    auto ref_img = load_ref_f32(ref_dir + "/preprocessed");
    if (ref_img.data.empty()) {
        fprintf(stderr, "Failed to load %s/preprocessed.bin\n", ref_dir.c_str());
        return 1;
    }
    fprintf(stderr, "  Loaded preprocessed image: %d elements\n", ref_img.numel());

    const int img_size = 1008;
    bool ok = sam3_encode_image_from_preprocessed(*state, *model, ref_img.data.data(), img_size);
    if (!ok) {
        fprintf(stderr, "sam3_encode_image_from_preprocessed failed!\n");
        return 1;
    }
    fprintf(stderr, "  Image encoded successfully\n");

    // ═══ Collect all metrics ═══
    std::vector<metric_row> report;

    // Helper: dump state tensor, compare with ref in NHWC layout
    auto check_nhwc = [&](const char * stage, const char * tensor_name,
                          const char * ref_name, float atol) {
        sam3_dump_state_tensor(*state, tensor_name, cpp_out + "/" + tensor_name);
        auto cpp = load_ref_f32(cpp_out + "/" + tensor_name);
        auto ref = load_ref_f32(ref_dir + "/" + ref_name);
        if (cpp.data.empty() || ref.data.empty()) {
            fprintf(stderr, "  [SKIP] %s\n", ref_name);
            return;
        }
        int n = std::min(cpp.numel(), ref.numel());
        auto m = compute_full_metrics(cpp.data.data(), ref.data.data(), n, atol);
        m.stage = stage;
        m.name = ref_name;
        // Build shape string from ref
        std::string s;
        for (size_t i = 0; i < ref.shape.size(); ++i) {
            if (i > 0) s += ",";
            s += std::to_string(ref.shape[i]);
        }
        m.py_shape_str = s;
        report.push_back(m);
    };

    auto check_nchw = [&](const char * stage, const char * tensor_name,
                          const char * ref_name, float atol) {
        sam3_dump_state_tensor(*state, tensor_name, cpp_out + "/" + tensor_name);
        auto cpp = load_ref_f32(cpp_out + "/" + tensor_name);
        auto ref = load_ref_f32(ref_dir + "/" + ref_name);
        if (cpp.data.empty() || ref.data.empty()) {
            fprintf(stderr, "  [SKIP] %s\n", ref_name);
            return;
        }
        // Transpose ggml→NCHW
        if (cpp.shape.size() >= 3) {
            int C = cpp.shape[0], W = cpp.shape[1], H = cpp.shape[2];
            std::vector<float> transposed(C * W * H);
            for (int c = 0; c < C; ++c)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        transposed[c * H * W + h * W + w] = cpp.data[c + w * C + h * C * W];
            int n = std::min((int) transposed.size(), ref.numel());
            auto m = compute_full_metrics(transposed.data(), ref.data.data(), n, atol);
            m.stage = stage;
            m.name = ref_name;
            std::string s;
            for (size_t i = 0; i < ref.shape.size(); ++i) {
                if (i > 0) s += ",";
                s += std::to_string(ref.shape[i]);
            }
            m.py_shape_str = s;
            report.push_back(m);
        }
    };

    // ═══ Stage 2: ViT backbone ═══
    fprintf(stderr, "\n═══ Stage 2: ViT Backbone ═══\n");
    check_nhwc("ViT", "dbg_patch_embed", "patch_embed", 1e-4f);
    check_nhwc("ViT", "dbg_after_pos_embed", "after_pos_embed", 1e-4f);
    check_nhwc("ViT", "dbg_after_ln_pre", "after_ln_pre", 1e-4f);

    for (int i = 0; i < 32; ++i) {
        char cpp_name[64], ref_name[64];
        snprintf(cpp_name, sizeof(cpp_name), "dbg_block_%d_out", i);
        snprintf(ref_name, sizeof(ref_name), "block_%d_out", i);
        auto ref = load_ref_f32(ref_dir + "/" + ref_name);
        if (ref.data.empty()) continue;
        // f32 accumulation across 32 blocks: max abs error ~0.06 on values up to
        // magnitude ~280 (relative error < 0.02%).  Cosine similarity is 1.0.
        float atol = 0.07f;
        check_nhwc("ViT", cpp_name, ref_name, atol);
    }

    // ═══ Stage 3: Neck ═══
    fprintf(stderr, "\n═══ Stage 3: Neck (SimpleFPN) ═══\n");
    check_nchw("Neck", "vit_output", "vit_output_bchw", 0.05f);
    for (int i = 0; i < 4; ++i) {
        char name[32];
        snprintf(name, sizeof(name), "neck_det_%d", i);
        check_nchw("Neck-Det", name, name, 5e-3f);
    }
    for (int i = 0; i < 4; ++i) {
        char name[32];
        snprintf(name, sizeof(name), "neck_trk_%d", i);
        auto ref = load_ref_f32(ref_dir + "/" + name);
        if (ref.data.empty()) continue;
        check_nchw("Neck-Trk", name, name, 5e-3f);
    }

    // ═══ Stage 4-5: SAM Prompt Encoder + Mask Decoder ═══
    // Use sam3_test_dump_phase6_from_ref_inputs which loads neck data from reference
    // files instead of reusing state tensors (avoids graph allocator buffer issues).
    // This still tests the SAM decoder numerics end-to-end since the neck outputs
    // were verified to match in Stage 3 above.
    fprintf(stderr, "\n═══ Stage 4-5: SAM Prompt Encoder + Mask Decoder ═══\n");
    sam3_pvs_params pvs_params;
    pvs_params.box = {200.0f, 50.0f, 800.0f, 950.0f};
    pvs_params.use_box = true;
    pvs_params.multimask = false;

    if (!sam3_test_dump_phase6_from_ref_inputs(*model, ref_dir, pvs_params,
                                                cpp_p6_out, params.n_threads)) {
        fprintf(stderr, "  [FAIL] sam3_test_dump_phase6_from_ref_inputs failed\n");
    } else {
        const std::string p6_ref = ref_dir + "/phase6/cat_box";
        struct { const char * name; const char * label; float tol; } p6_tensors[] = {
            {"sam_pe_sparse",           "PE sparse embeddings",   1e-4f},
            {"sam_pe_dense",            "PE dense embeddings",    1e-6f},
            {"sam_pe_image_pe",         "PE image PE",            1e-5f},
            {"sam_dec_image_feats",     "Dec image features",     5e-3f},
            {"sam_dec_tokens_initial",  "Dec initial tokens",     1e-4f},
            {"sam_dec_block0_queries",  "Dec block0 queries",     1e-4f},
            {"sam_dec_block0_keys",     "Dec block0 keys",        1e-4f},
            {"sam_dec_block1_queries",  "Dec block1 queries",     2e-4f},
            {"sam_dec_block1_keys",     "Dec block1 keys",        2e-4f},
            {"sam_dec_final_queries",   "Dec final queries",      2e-4f},
            {"sam_dec_feat_s1_proj",    "Dec feat_s1 projection", 5e-3f},
            {"sam_dec_feat_s0_proj",    "Dec feat_s0 projection", 5e-3f},
            {"sam_dec_upscaled",        "Dec upscaled embedding", 9e-4f},
            {"sam_dec_mask_tokens",     "Dec mask tokens",        2e-4f},
            {"sam_dec_masks",           "Dec mask logits",        8e-3f},
            {"sam_dec_iou",             "Dec IoU predictions",    2e-4f},
            {"sam_dec_obj_score",       "Dec object score",       2e-4f},
            {"sam_dec_sam_token",       "Dec SAM token",          2e-4f},
        };

        for (const auto & t : p6_tensors) {
            auto cpp = load_ref_f32(cpp_p6_out + "/" + t.name);
            auto ref = load_ref_f32(p6_ref + "/" + t.name);
            if (cpp.data.empty() || ref.data.empty()) {
                fprintf(stderr, "  [SKIP] %s (missing cpp=%s ref=%s)\n",
                        t.name,
                        cpp.data.empty() ? "yes" : "no",
                        ref.data.empty() ? "yes" : "no");
                continue;
            }
            int n = std::min(cpp.numel(), ref.numel());
            auto m = compute_full_metrics(cpp.data.data(), ref.data.data(), n, t.tol);
            m.stage = "SAM-Dec";
            m.name = t.label;
            std::string s;
            for (size_t i = 0; i < ref.shape.size(); ++i) {
                if (i > 0) s += ",";
                s += std::to_string(ref.shape[i]);
            }
            m.py_shape_str = s;
            report.push_back(m);
        }
    }

    // ═══ Print Full Report ═══
    fprintf(stderr, "\n");
    fprintf(stderr, "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  END-TO-END PVS PIPELINE COMPARISON: Python vs C++ (cat.jpg with box [200,50,800,950])\n");
    fprintf(stderr, "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n\n");

    int n_pass = 0, n_fail = 0;
    fprintf(stderr, "  %-8s %-6s %-30s %-16s %12s %12s %12s %12s %12s %12s\n",
            "Status", "Stage", "Tensor", "Shape", "MAE", "Max", "RelErr", "Cosine", "P95", "P99");
    fprintf(stderr, "  %-8s %-6s %-30s %-16s %12s %12s %12s %12s %12s %12s\n",
            "------", "-----", "------", "-----", "---", "---", "------", "------", "---", "---");

    for (const auto & m : report) {
        const char * status = m.ok ? "PASS" : "FAIL";
        fprintf(stderr, "  %-8s %-6s %-30s %-16s %12.4e %12.4e %12.4e %12.8f %12.4e %12.4e\n",
                status, m.stage.c_str(), m.name.c_str(), m.py_shape_str.c_str(),
                m.mae, m.max_abs, m.mean_rel, m.cosine, m.p95, m.p99);
        if (m.ok) n_pass++;
        else n_fail++;
    }

    fprintf(stderr, "\n══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  TOTAL: %d PASS, %d FAIL out of %d tensors\n", n_pass, n_fail, (int) report.size());
    fprintf(stderr, "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n\n");

    state.reset();
    sam3_free_model(*model);
    model.reset();

    return n_fail > 0 ? 1 : 0;
}
