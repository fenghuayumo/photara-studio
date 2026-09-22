#include "sam3.h"
#include "test_utils.h"

#include <cstdio>
#include <string>
#include <vector>

struct geom_case {
    std::string id;
    sam3_pcs_params params;
};

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <ref_dir> <prephase_ref_dir> <model_path>\n", argv[0]);
        return 1;
    }

    const std::string ref_dir = argv[1];
    const std::string prephase_ref_dir = argv[2];
    const std::string model_path = argv[3];
    const std::string cpp_root = ref_dir + "/cpp_out_geom";

    // Test cases matching the Python dump script
    std::vector<geom_case> cases;

    // Case 1: dummy prompt (no exemplars, just CLS)
    {
        geom_case c;
        c.id = "dummy_prompt";
        // No exemplars
        cases.push_back(std::move(c));
    }

    // Case 2: single positive box (cx=0.5, cy=0.5, w=0.3, h=0.3)
    // API uses XYXY format — convert from CxCyWH to XYXY
    {
        geom_case c;
        c.id = "single_box";
        float cx = 0.5f, cy = 0.5f, w = 0.3f, h = 0.3f;
        sam3_box box;
        box.x0 = cx - w * 0.5f;  // 0.35
        box.y0 = cy - h * 0.5f;  // 0.35
        box.x1 = cx + w * 0.5f;  // 0.65
        box.y1 = cy + h * 0.5f;  // 0.65
        c.params.pos_exemplars.push_back(box);
        cases.push_back(std::move(c));
    }

    // Case 3: two boxes (one positive, one negative)
    // Box1: cx=0.3, cy=0.4, w=0.2, h=0.25 (positive)
    // Box2: cx=0.7, cy=0.6, w=0.15, h=0.2 (negative)
    {
        geom_case c;
        c.id = "two_boxes";
        {
            float cx = 0.3f, cy = 0.4f, w = 0.2f, h = 0.25f;
            sam3_box box = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f};
            c.params.pos_exemplars.push_back(box);
        }
        {
            float cx = 0.7f, cy = 0.6f, w = 0.15f, h = 0.2f;
            sam3_box box = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f};
            c.params.neg_exemplars.push_back(box);
        }
        cases.push_back(std::move(c));
    }

    sam3_params mparams;
    mparams.model_path = model_path;
    mparams.n_threads = 1;
    mparams.use_gpu = false;

    auto model = sam3_load_model(mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    bool overall_ok = true;
    float atol = 1e-4f;

    for (const auto& tc : cases) {
        fprintf(stderr, "\n=== Geometry Encoder: %s ===\n", tc.id.c_str());

        const std::string case_ref_dir = ref_dir + "/" + tc.id;
        const std::string case_cpp_dir = cpp_root + "/" + tc.id;
        ensure_dir(cpp_root);
        ensure_dir(case_cpp_dir);

        if (!sam3_test_dump_geom_enc(*model, prephase_ref_dir, tc.params,
                                      case_cpp_dir, mparams.n_threads)) {
            fprintf(stderr, "  [FAIL] dump failed for %s\n", tc.id.c_str());
            overall_ok = false;
            continue;
        }

        // Compare tensors
        struct tensor_check {
            std::string name;
            float tol;
        };
        std::vector<tensor_check> checks = {
            // post_final_proj is an input tensor whose buffer gets reused by the graph
            // allocator during computation — comparing it after graph compute gives stale
            // data. Only geom_output (the final transformer output) is meaningful.
            {"geom_output", 5e-4f},
        };

        for (const auto& chk : checks) {
            auto ref = load_ref_f32(case_ref_dir + "/" + chk.name);
            auto got = load_ref_f32(case_cpp_dir + "/" + chk.name);

            if (ref.data.empty() || got.data.empty()) {
                fprintf(stderr, "  [FAIL] %s: missing tensor\n", chk.name.c_str());
                overall_ok = false;
                continue;
            }
            if (ref.numel() != got.numel()) {
                fprintf(stderr, "  [FAIL] %s: numel mismatch ref=%d got=%d\n",
                        chk.name.c_str(), ref.numel(), got.numel());
                overall_ok = false;
                continue;
            }

            auto r = compare_tensors(got.data.data(), ref.data.data(), ref.numel(), chk.tol);
            bool ok = r.max_diff <= chk.tol;
            fprintf(stderr, "  [%s] %-30s max=%.3e mean=%.3e cos=%.8f tol=%.1e",
                    ok ? "PASS" : "FAIL", chk.name.c_str(),
                    r.max_diff, r.mean_diff, r.cosine_sim, chk.tol);
            if (!ok) {
                fprintf(stderr, " worst=%d got=%.6g ref=%.6g",
                        r.worst_index, r.worst_a, r.worst_b);
                overall_ok = false;
            }
            fprintf(stderr, "\n");
        }
    }

    fprintf(stderr, "\n%s\n", overall_ok ? "ALL PASS" : "SOME FAILED");
    return overall_ok ? 0 : 1;
}
