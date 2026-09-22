#include "sam3.h"

#include "test_utils.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

struct tensor_case {
    std::string name;
    std::string label;
};

struct report_row {
    bool measured = false;
    bool ok = false;
    compare_result r;
    std::string note;
};

static std::string load_prompt(const std::string & path) {
    std::ifstream f(path);
    std::string line;
    if (f) {
        std::getline(f, line);
    }
    return line;
}

static std::vector<tensor_case> tensor_cases() {
    std::vector<tensor_case> cases = {
        {"text_valid_mask", "Text valid mask"},
        {"text_features", "Text features"},
        {"fenc_img_input", "Fusion encoder image input"},
        {"fenc_pos_embed", "Fusion encoder position input"},
        {"fenc_prompt", "Fusion encoder prompt input"},
        {"img_pe_72", "Image PE 72x72"},
        {"fenc_output", "Fusion encoder final output"},
        {"ddec_query_embed", "Decoder query embed"},
        {"ddec_ref_pts_raw", "Decoder reference points raw"},
        {"ddec_presence_token", "Decoder presence token"},
        {"ddec_ref_boxes_init", "Decoder initial reference boxes"},
        {"ddec_query_sine_0", "Decoder layer 0 query sine"},
        {"ddec_query_pos_0", "Decoder layer 0 query pos"},
        {"ddec_rpb_mask_0", "Decoder layer 0 RPB mask"},
        {"ddec_layer0_after_sa", "Decoder layer 0 after self-attn"},
        {"ddec_layer0_after_text_ca", "Decoder layer 0 after text CA"},
        {"ddec_layer0_after_img_ca", "Decoder layer 0 after image CA"},
        {"ddec_layer0_full_out", "Decoder layer 0 full output"},
        {"ddec_layer0_presence", "Decoder layer 0 presence"},
        {"ddec_normed_output", "Decoder final normed output"},
        {"ddec_pred_boxes", "Decoder predicted boxes"},
        {"ddec_presence_logit", "Decoder presence logit"},
        {"scoring_prompt_mlp_out", "Scoring prompt MLP"},
        {"scoring_pooled", "Scoring pooled prompt"},
        {"scoring_proj_pooled", "Scoring projected pooled prompt"},
        {"scoring_proj_hs", "Scoring projected queries"},
        {"scoring_class_scores", "Scoring class scores"},
        {"seg_enc_after_ca", "Seg head encoder after prompt CA"},
        {"seg_enc_visual", "Seg head visual encoder map"},
        {"seg_pixel_dec_stage0", "Seg head pixel decoder stage 0"},
        {"seg_pixel_dec_stage1", "Seg head pixel decoder stage 1"},
        {"seg_pixel_decoder_out", "Seg head pixel decoder output"},
        {"seg_instance_embed", "Seg head instance embed"},
        {"seg_mask_embed", "Seg head mask embed"},
        {"seg_mask_logits", "Seg head mask logits"},
    };

    for (int i = 0; i < 6; ++i) {
        char name[64];
        char label[64];

        snprintf(name, sizeof(name), "fenc_layer%d_out", i);
        snprintf(label, sizeof(label), "Fusion encoder layer %d", i);
        cases.push_back({name, label});

        snprintf(name, sizeof(name), "ddec_layer%d_out", i);
        snprintf(label, sizeof(label), "Decoder layer %d output", i);
        cases.push_back({name, label});

        snprintf(name, sizeof(name), "ddec_layer%d_refboxes", i);
        snprintf(label, sizeof(label), "Decoder layer %d ref boxes", i);
        cases.push_back({name, label});
    }

    return cases;
}

static float tolerance_for(const std::string & name) {
    if (name == "ddec_rpb_mask_0") {
        return 2e-4f;
    }
    if (name == "seg_mask_embed") {
        return 2e-4f;
    }
    if (name == "seg_mask_logits") {
        return 3e-3f;
    }
    if (name == "ddec_layer2_out" ||
        name == "ddec_layer3_out" ||
        name == "ddec_layer4_out" ||
        name == "ddec_layer5_out") {
        return 2e-4f;
    }
    return 1e-4f;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <ref_dir> <model_path> <image_path>\n", argv[0]);
        return 1;
    }

    const std::string ref_dir = argv[1];
    const std::string model_path = argv[2];
    const std::string image_path = argv[3];
    const std::string cpp_dir = ref_dir + "/cpp_out_phase5";
    if (!ensure_dir(cpp_dir)) {
        fprintf(stderr, "Failed to create %s\n", cpp_dir.c_str());
        return 1;
    }

    std::string prompt = load_prompt(ref_dir + "/prompt.txt");
    if (prompt.empty()) {
        prompt = "yellow school bus";
    }

    auto ref_token_ids = load_ref_i32(ref_dir + "/token_ids");

    if (!sam3_test_load_tokenizer(model_path)) {
        fprintf(stderr, "Failed to load tokenizer from %s\n", model_path.c_str());
        return 1;
    }

    auto cpp_token_ids = sam3_test_tokenize(prompt);
    std::vector<int32_t> run_token_ids = cpp_token_ids;

    report_row token_row;
    if (!ref_token_ids.data.empty()) {
        const int token_bad = compare_exact_i32(cpp_token_ids, ref_token_ids.data);
        token_row.measured = true;
        token_row.ok = token_bad == 0;
        token_row.r.max_diff = token_bad == 0 ? 0.0f : 1.0f;
        token_row.r.mean_diff = token_bad == 0 ? 0.0f : 1.0f;
        token_row.r.cosine_sim = token_bad == 0 ? 1.0f : 0.0f;
        token_row.note = token_bad == 0 ? "exact int compare" : "tokenizer mismatch";
        run_token_ids = ref_token_ids.data;
    } else {
        token_row.measured = false;
        token_row.ok = false;
        token_row.note = "missing token_ids reference";
    }

    sam3_params params;
    params.model_path = model_path;
    params.n_threads = 1;
    params.use_gpu = false;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", model_path.c_str());
        return 1;
    }

    (void) image_path;

    const std::string prephase_ref_dir = ref_dir + "/../ref";
    if (!sam3_test_dump_phase5_from_ref_inputs(*model, run_token_ids,
                                               prephase_ref_dir, ref_dir,
                                               cpp_dir, params.n_threads)) {
        fprintf(stderr, "Failed to dump C++ phase 5 tensors\n");
        return 1;
    }

    const auto cases = tensor_cases();
    std::vector<report_row> rows(cases.size());
    bool overall_ok = token_row.measured ? token_row.ok : false;

    for (size_t i = 0; i < cases.size(); ++i) {
        const auto & tc = cases[i];
        auto ref = load_ref_f32(ref_dir + "/" + tc.name);
        auto got = load_ref_f32(cpp_dir + "/" + tc.name);

        if (ref.data.empty() || got.data.empty()) {
            rows[i].ok = false;
            rows[i].note = "missing reference or C++ output";
            overall_ok = false;
            fprintf(stderr, "  [FAIL] missing tensor %s\n", tc.name.c_str());
            continue;
        }

        if (ref.numel() != got.numel()) {
            rows[i].ok = false;
            rows[i].note = "numel mismatch";
            overall_ok = false;
            fprintf(stderr, "  [FAIL] numel mismatch for %s (%d vs %d)\n",
                    tc.name.c_str(), got.numel(), ref.numel());
            continue;
        }

        rows[i].measured = true;
        const float atol = tolerance_for(tc.name);
        rows[i].r = compare_tensors(got.data.data(), ref.data.data(), ref.numel(), atol);
        rows[i].ok = rows[i].r.max_diff <= atol;
        if (!rows[i].ok) {
            char note[160];
            snprintf(note, sizeof(note),
                     "worst_idx=%d got=%.6g ref=%.6g bad=%d/%d",
                     rows[i].r.worst_index,
                     rows[i].r.worst_a,
                     rows[i].r.worst_b,
                     rows[i].r.n_bad,
                     rows[i].r.n_total);
            rows[i].note = note;
            overall_ok = false;
            fprintf(stderr,
                    "  [FAIL] %-24s max=%.3e mean=%.3e cosine=%.8f idx=%d got=%.6g ref=%.6g\n",
                    tc.name.c_str(),
                    rows[i].r.max_diff,
                    rows[i].r.mean_diff,
                    rows[i].r.cosine_sim,
                    rows[i].r.worst_index,
                    rows[i].r.worst_a,
                    rows[i].r.worst_b);
        }
    }

    fprintf(stderr, "\n## Numerical Precision Report\n\n");
    fprintf(stderr, "| Component | Operation | max_diff | mean_diff | cosine | Tolerance | Status | Notes |\n");
    fprintf(stderr, "|-----------|-----------|----------|-----------|--------|-----------|--------|-------|\n");
    fprintf(stderr,
            "| Tokenizer | Token IDs | %.3e | %.3e | %.8f | exact | %s | %s |\n",
            token_row.r.max_diff,
            token_row.r.mean_diff,
            token_row.r.cosine_sim,
            token_row.ok ? "PASS" : "FAIL",
            token_row.note.c_str());

    for (size_t i = 0; i < cases.size(); ++i) {
        const auto & tc = cases[i];
        const auto & row = rows[i];
        const char * status = !row.measured ? "UNTESTED" : (row.ok ? "PASS" : "FAIL");
        fprintf(stderr,
                "| Phase 5 | %s | %.3e | %.3e | %.8f | %.1e | %s | %s |\n",
                tc.label.c_str(),
                row.r.max_diff,
                row.r.mean_diff,
                row.r.cosine_sim,
                tolerance_for(tc.name),
                status,
                row.note.empty() ? "measured" : row.note.c_str());
    }

    return overall_ok ? 0 : 1;
}
