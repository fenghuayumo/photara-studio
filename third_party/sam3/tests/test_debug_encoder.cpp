/**
 * Debug test for image encoder: loads Python-preprocessed image,
 * runs through C++ ViT, dumps all intermediate tensors for comparison.
 *
 * Usage:
 *   ./test_debug_encoder <model.ggml> <ref_dir> <output_dir>
 *
 * ref_dir should contain preprocessed.bin/.shape from dump_phase3_reference.py.
 * The test uses sam3_encode_image_from_preprocessed() to ensure identical input.
 */
#include "sam3.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// Compare C++ ggml tensor (dumped as [ne0, ne1, ne2]) vs Python NHWC [1, H, W, E]
// ggml stores: flat[e + w*E + h*E*W] where ne0=E, ne1=W, ne2=H
// Python NHWC: flat[h*W*E + w*E + e] (with optional batch dim)
// These are actually the same flat layout!
static compare_result compare_ggml_vs_nhwc(const ref_tensor_f32& cpp,
                                            const ref_tensor_f32& ref) {
    return compare_tensors(cpp.data.data(), ref.data.data(),
                           std::min(cpp.numel(), ref.numel()));
}

// Compare C++ ggml tensor (dumped as [C, W, H]) vs Python NCHW [1, C, H, W]
// ggml stores: flat[c + w*C + h*C*W]
// Python NCHW: flat[c*H*W + h*W + w]
// These differ! Need transposition.
static compare_result compare_ggml_vs_nchw(const ref_tensor_f32& cpp,
                                            const ref_tensor_f32& ref) {
    if (cpp.shape.size() < 3 || ref.data.empty()) {
        compare_result r;
        r.max_diff = 999.0f;
        return r;
    }
    int C = cpp.shape[0], W = cpp.shape[1], H = cpp.shape[2];

    // Transpose ggml → NCHW
    std::vector<float> transposed(C * W * H);
    for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                transposed[c * H * W + h * W + w] = cpp.data[c + w * C + h * C * W];

    return compare_tensors(transposed.data(), ref.data.data(),
                           std::min((int)transposed.size(), ref.numel()));
}

static void print_result(const char* name, const compare_result& r, float atol) {
    const char* status = (r.max_diff < atol) ? "[PASS]" : "[FAIL]";
    fprintf(stderr, "  %s %-40s max=%.6e mean=%.6e cos=%.8f\n",
            status, name, r.max_diff, r.mean_diff, r.cosine_sim);
    if (r.max_diff >= atol) {
        fprintf(stderr, "         worst_idx=%d a=%.6e b=%.6e\n",
                r.worst_index, r.worst_a, r.worst_b);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.ggml> <ref_dir> <output_dir>\n", argv[0]);
        fprintf(stderr, "\nref_dir: directory with Python reference tensors\n");
        fprintf(stderr, "output_dir: where to dump C++ tensors\n");
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ref_dir = argv[2];
    const std::string out_dir = argv[3];
    ensure_dir(out_dir);

    // ═══ Load model ═══
    fprintf(stderr, "\n═══ Loading model ═══\n");
    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = false;  // CPU for determinism
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
    fprintf(stderr, "  Loaded preprocessed image: shape=[");
    for (size_t i = 0; i < ref_img.shape.size(); ++i) {
        if (i > 0) fprintf(stderr, ",");
        fprintf(stderr, "%d", ref_img.shape[i]);
    }
    fprintf(stderr, "] (%d elements)\n", ref_img.numel());

    const int img_size = 1008;
    const int expected_elems = 3 * img_size * img_size;
    if (ref_img.numel() != expected_elems) {
        fprintf(stderr, "  ERROR: expected %d elements, got %d\n", expected_elems, ref_img.numel());
        return 1;
    }

    // ═══ Encode image using Python-preprocessed data ═══
    fprintf(stderr, "\n═══ Encoding image (from Python preprocessed data) ═══\n");
    bool ok = sam3_encode_image_from_preprocessed(*state, *model, ref_img.data.data(), img_size);
    if (!ok) {
        fprintf(stderr, "sam3_encode_image_from_preprocessed failed!\n");
        return 1;
    }

    // ═══ Stage 2: Dump and compare ViT intermediate tensors ═══
    fprintf(stderr, "\n═══ Stage 2: Comparing ViT intermediate tensors ═══\n");

    int n_pass = 0, n_fail = 0, n_skip = 0;

    // Helper to dump via sam3_dump_state_tensor and compare with ref in NHWC layout
    auto check_nhwc = [&](const char* tensor_name, const char* ref_name, float atol) {
        // Dump C++ tensor
        sam3_dump_state_tensor(*state, tensor_name, out_dir + "/" + tensor_name);

        // Load both
        auto cpp = load_ref_f32(out_dir + "/" + tensor_name);
        auto ref = load_ref_f32(ref_dir + "/" + ref_name);

        if (cpp.data.empty()) {
            fprintf(stderr, "  [SKIP] %s — not found in C++ state\n", tensor_name);
            n_skip++;
            return;
        }
        if (ref.data.empty()) {
            fprintf(stderr, "  [SKIP] %s — no Python reference '%s'\n", tensor_name, ref_name);
            n_skip++;
            return;
        }

        auto r = compare_ggml_vs_nhwc(cpp, ref);
        print_result(ref_name, r, atol);
        if (r.max_diff < atol) n_pass++; else n_fail++;
    };

    auto check_nchw = [&](const char* tensor_name, const char* ref_name, float atol) {
        sam3_dump_state_tensor(*state, tensor_name, out_dir + "/" + tensor_name);

        auto cpp = load_ref_f32(out_dir + "/" + tensor_name);
        auto ref = load_ref_f32(ref_dir + "/" + ref_name);

        if (cpp.data.empty()) {
            fprintf(stderr, "  [SKIP] %s — not found in C++ state\n", tensor_name);
            n_skip++;
            return;
        }
        if (ref.data.empty()) {
            fprintf(stderr, "  [SKIP] %s — no Python reference '%s'\n", tensor_name, ref_name);
            n_skip++;
            return;
        }

        auto r = compare_ggml_vs_nchw(cpp, ref);
        print_result(ref_name, r, atol);
        if (r.max_diff < atol) n_pass++; else n_fail++;
    };

    // ── Patch Embedding ──
    fprintf(stderr, "\n--- Patch Embedding ---\n");
    check_nhwc("dbg_patch_embed", "patch_embed", 1e-4f);

    // ── After Positional Embedding ──
    fprintf(stderr, "\n--- After Positional Embedding ---\n");
    check_nhwc("dbg_after_pos_embed", "after_pos_embed", 1e-4f);

    // ── After LayerNorm Pre ──
    fprintf(stderr, "\n--- After LayerNorm Pre ---\n");
    check_nhwc("dbg_after_ln_pre", "after_ln_pre", 1e-4f);

    // ── Per-block outputs ──
    // Tolerance is progressive: early blocks have tiny error, later blocks accumulate
    // f32 drift on values of magnitude ~100-270 across 32 blocks.
    // Max absolute error grows but relative error stays < 0.1% and cosine ≈ 1.0.
    fprintf(stderr, "\n--- ViT Block Outputs ---\n");
    for (int i = 0; i < 32; ++i) {
        char cpp_name[64], ref_name[64];
        snprintf(cpp_name, sizeof(cpp_name), "dbg_block_%d_out", i);
        snprintf(ref_name, sizeof(ref_name), "block_%d_out", i);

        // Check if Python reference exists for this block
        auto ref = load_ref_f32(ref_dir + "/" + ref_name);
        if (ref.data.empty()) continue;

        // Realistic tolerance for f32 accumulation over transformer blocks.
        // Values reach magnitude ~270 by block 20, so 0.05 absolute tolerance
        // corresponds to ~0.02% relative error.
        float atol = 1e-3f + i * 2e-3f;
        check_nhwc(cpp_name, ref_name, atol);
    }

    // ── Final ViT output (NCHW comparison) ──
    fprintf(stderr, "\n--- Final ViT Output ---\n");
    check_nchw("vit_output", "vit_output_bchw", 0.05f);

    // ── Neck outputs (NCHW comparison) ──
    fprintf(stderr, "\n--- Neck Outputs (detector) ---\n");
    for (int i = 0; i < 4; ++i) {
        char name[32];
        snprintf(name, sizeof(name), "neck_det_%d", i);
        check_nchw(name, name, 5e-3f);
    }

    fprintf(stderr, "\n--- Neck Outputs (tracker) ---\n");
    for (int i = 0; i < 4; ++i) {
        char name[32];
        snprintf(name, sizeof(name), "neck_trk_%d", i);
        // Check if ref exists for tracker path
        auto ref = load_ref_f32(ref_dir + "/" + name);
        if (ref.data.empty()) continue;
        check_nchw(name, name, 5e-3f);
    }

    // ═══ Summary ═══
    fprintf(stderr, "\n═══════════════════════════════════════════\n");
    fprintf(stderr, "Results: %d passed, %d failed, %d skipped\n", n_pass, n_fail, n_skip);
    fprintf(stderr, "═══════════════════════════════════════════\n");

    state.reset();
    sam3_free_model(*model);
    model.reset();

    return n_fail > 0 ? 1 : 0;
}
