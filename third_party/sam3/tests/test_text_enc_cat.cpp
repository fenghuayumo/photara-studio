/**
 * Quick test: run the text encoder in isolation for "cat" token IDs
 * and compare against Python reference.
 */
#include "sam3.h"
#include "test_utils.h"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.ggml> <ref_dir>\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ref_dir = std::string(argv[2]) + "/pcs_ref";
    const std::string out_dir = std::string(argv[2]) + "/pcs_cpp_text_only";
    ensure_dir(out_dir);

    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = false;
    params.n_threads = 4;

    auto model = sam3_load_model(params);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    // Load cat token IDs from Python reference
    auto ref_tokens_f = load_ref_f32(ref_dir + "/token_ids");
    if (ref_tokens_f.data.empty()) {
        fprintf(stderr, "Failed to load token IDs\n");
        return 1;
    }
    std::vector<int32_t> token_ids(ref_tokens_f.numel());
    for (int i = 0; i < ref_tokens_f.numel(); ++i) {
        token_ids[i] = (int32_t)ref_tokens_f.data[i];
    }
    while ((int)token_ids.size() < 32) token_ids.push_back(0);

    fprintf(stderr, "Cat token IDs: [%d, %d, %d, ...]\n",
            token_ids[0], token_ids[1], token_ids[2]);

    // Run text encoder in ISOLATION
    if (!sam3_test_dump_text_encoder(*model, token_ids, out_dir, params.n_threads)) {
        fprintf(stderr, "Failed to dump text encoder\n");
        return 1;
    }

    // Compare against Python reference
    auto cpp = load_ref_f32(out_dir + "/text_features_2d");
    auto py = load_ref_f32(ref_dir + "/text_features");
    if (cpp.data.empty() || py.data.empty()) {
        fprintf(stderr, "Failed to load tensors for comparison\n");
        return 1;
    }

    int n = std::min(cpp.numel(), py.numel());
    auto r = compare_tensors(cpp.data.data(), py.data.data(), n, 1e-4f);

    fprintf(stderr, "\n═══ Text Encoder (Isolated): C++ vs Python for 'cat' ═══\n");
    fprintf(stderr, "  shape: cpp=[%d,%d] py=[%d,%d]\n",
            cpp.shape[0], cpp.shape.size()>1?cpp.shape[1]:0,
            py.shape[0], py.shape.size()>1?py.shape[1]:0);
    fprintf(stderr, "  max_diff: %.6e\n", r.max_diff);
    fprintf(stderr, "  mean_diff: %.6e\n", r.mean_diff);
    fprintf(stderr, "  cosine: %.8f\n", r.cosine_sim);
    fprintf(stderr, "  n_bad (atol=1e-4): %d/%d\n", r.n_bad, r.n_total);

    fprintf(stderr, "\n  Result: %s\n", r.cosine_sim > 0.999f ? "PASS" : "FAIL");

    sam3_free_model(*model);
    return r.cosine_sim > 0.999f ? 0 : 1;
}
