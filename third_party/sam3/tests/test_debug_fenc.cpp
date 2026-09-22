#include "sam3.h"

#include "test_utils.h"

#include <cstdio>
#include <string>
#include <vector>

struct tensor_case {
    std::string name;
    std::string label;
    float atol;
};

static std::vector<tensor_case> build_cases() {
    std::vector<tensor_case> cases;
    for (int i = 0; i < 6; ++i) {
        char name[64], label[64];
        snprintf(name, sizeof(name), "fenc_layer%d_out", i);
        snprintf(label, sizeof(label), "Fusion encoder layer %d", i);
        // Tolerance: early layers should be tight, later layers allow more accumulation
        float atol = (i < 3) ? 1e-4f : 2e-4f;
        cases.push_back({name, label, atol});
    }
    return cases;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <ref_dir> <model_path> [n_threads]\n"
                "\n"
                "  ref_dir    : directory with Python-dumped fenc tensors\n"
                "               (from dump_fenc_from_package.py)\n"
                "  model_path : path to sam3_f32.ggml weights\n"
                "  n_threads  : optional (default 1)\n",
                argv[0]);
        return 1;
    }

    const std::string ref_dir = argv[1];
    const std::string model_path = argv[2];
    const int n_threads = (argc > 3) ? std::atoi(argv[3]) : 1;

    const std::string cpp_dir = ref_dir + "/cpp_out_fenc";
    if (!ensure_dir(cpp_dir)) {
        fprintf(stderr, "Failed to create output dir %s\n", cpp_dir.c_str());
        return 1;
    }

    // ── Load model ──
    sam3_params params;
    params.model_path = model_path;
    params.n_threads = n_threads;
    params.use_gpu = false;

    fprintf(stderr, "Loading model from %s...\n", model_path.c_str());
    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // ── Run fenc-only test ──
    fprintf(stderr, "\nRunning fusion encoder with Python-dumped inputs from %s\n", ref_dir.c_str());
    if (!sam3_test_fenc_only(*model, ref_dir, cpp_dir, n_threads)) {
        fprintf(stderr, "sam3_test_fenc_only failed\n");
        return 1;
    }

    // ── Compare per-layer outputs ──
    fprintf(stderr, "\n══════════════════════════════════════════════════════════════\n");
    fprintf(stderr, "  Fusion Encoder: Python vs C++ Comparison\n");
    fprintf(stderr, "══════════════════════════════════════════════════════════════\n\n");

    // First verify inputs are loaded correctly
    {
        auto ref_img = load_ref_f32(ref_dir + "/fenc_input_tgt");
        auto cpp_img = load_ref_f32(cpp_dir + "/fenc_img_input");
        if (!ref_img.data.empty() && !cpp_img.data.empty() &&
            ref_img.numel() == cpp_img.numel()) {
            auto r = compare_tensors(cpp_img.data.data(), ref_img.data.data(),
                                     ref_img.numel(), 1e-6f);
            fprintf(stderr, "  Input check: img_feats  max_diff=%.2e  (should be ~0)\n",
                    r.max_diff);
        }

        auto ref_pos = load_ref_f32(ref_dir + "/fenc_input_pos");
        auto cpp_pos = load_ref_f32(cpp_dir + "/fenc_pos_input");
        if (!ref_pos.data.empty() && !cpp_pos.data.empty() &&
            ref_pos.numel() == cpp_pos.numel()) {
            auto r = compare_tensors(cpp_pos.data.data(), ref_pos.data.data(),
                                     ref_pos.numel(), 1e-6f);
            fprintf(stderr, "  Input check: pos_embed   max_diff=%.2e  (should be ~0)\n",
                    r.max_diff);
        }

        auto ref_prompt = load_ref_f32(ref_dir + "/fenc_input_prompt");
        auto cpp_prompt = load_ref_f32(cpp_dir + "/fenc_prompt_input");
        if (!ref_prompt.data.empty() && !cpp_prompt.data.empty() &&
            ref_prompt.numel() == cpp_prompt.numel()) {
            auto r = compare_tensors(cpp_prompt.data.data(), ref_prompt.data.data(),
                                     ref_prompt.numel(), 1e-6f);
            fprintf(stderr, "  Input check: prompt      max_diff=%.2e  (should be ~0)\n",
                    r.max_diff);
        }
    }

    fprintf(stderr, "\n");

    // Compare per-layer outputs
    const auto cases = build_cases();
    bool overall_ok = true;
    int n_pass = 0;

    fprintf(stderr, "| %-30s | %10s | %10s | %10s | %10s | %6s |\n",
            "Layer", "max_diff", "mean_diff", "cosine", "tolerance", "status");
    fprintf(stderr, "|%s|%s|%s|%s|%s|%s|\n",
            "--------------------------------", "------------", "------------",
            "------------", "------------", "--------");

    for (const auto & tc : cases) {
        auto ref = load_ref_f32(ref_dir + "/" + tc.name);
        auto cpp = load_ref_f32(cpp_dir + "/" + tc.name);

        if (ref.data.empty()) {
            fprintf(stderr, "| %-30s | %10s | %10s | %10s | %10.1e | %-6s |\n",
                    tc.label.c_str(), "N/A", "N/A", "N/A", tc.atol, "SKIP");
            fprintf(stderr, "  (missing Python reference %s/%s)\n", ref_dir.c_str(), tc.name.c_str());
            continue;
        }
        if (cpp.data.empty()) {
            fprintf(stderr, "| %-30s | %10s | %10s | %10s | %10.1e | %-6s |\n",
                    tc.label.c_str(), "N/A", "N/A", "N/A", tc.atol, "SKIP");
            fprintf(stderr, "  (missing C++ output %s/%s)\n", cpp_dir.c_str(), tc.name.c_str());
            continue;
        }

        if (ref.numel() != cpp.numel()) {
            fprintf(stderr, "| %-30s | %10s | %10s | %10s | %10.1e | %-6s |\n",
                    tc.label.c_str(), "N/A", "N/A", "N/A", tc.atol, "FAIL");
            fprintf(stderr, "  SHAPE MISMATCH: ref=%d cpp=%d\n", ref.numel(), cpp.numel());
            overall_ok = false;
            continue;
        }

        auto r = compare_tensors(cpp.data.data(), ref.data.data(), ref.numel(), tc.atol);
        bool pass = r.max_diff <= tc.atol;
        if (pass) n_pass++;
        if (!pass) overall_ok = false;

        fprintf(stderr, "| %-30s | %10.3e | %10.3e | %10.8f | %10.1e | %-6s |\n",
                tc.label.c_str(),
                r.max_diff, r.mean_diff, r.cosine_sim,
                tc.atol, pass ? "PASS" : "FAIL");

        if (!pass) {
            fprintf(stderr, "  worst idx=%d  ref=%.6g  cpp=%.6g  bad=%d/%d\n",
                    r.worst_index, r.worst_b, r.worst_a, r.n_bad, r.n_total);
        }
    }

    fprintf(stderr, "\n%d/%zu layers passed\n", n_pass, cases.size());
    fprintf(stderr, "Overall: %s\n", overall_ok ? "PASS" : "FAIL");

    if (!overall_ok) {
        fprintf(stderr, "\n>>> DIVERGENCE DETECTED. Next steps:\n");
        fprintf(stderr, "    1. If layer 0 fails: bug is in self-attention or cross-attention\n");
        fprintf(stderr, "    2. Add finer-grained dumps (after SA, after CA, after FFN)\n");
        fprintf(stderr, "    3. Check weight loading in convert_sam3_to_ggml.py\n");
    }

    return overall_ok ? 0 : 1;
}
