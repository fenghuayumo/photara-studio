#include "sam3.h"

#include "test_utils.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

struct phase6_case {
    std::string id;
    sam3_pvs_params params;
};

struct tensor_case {
    const char * name;
    const char * label;
    float tol;
};

static std::vector<sam3_point> parse_points(const std::string & field) {
    std::vector<sam3_point> pts;
    if (field.empty()) {
        return pts;
    }

    size_t start = 0;
    while (start < field.size()) {
        size_t end = field.find('|', start);
        if (end == std::string::npos) {
            end = field.size();
        }

        const std::string part = field.substr(start, end - start);
        const size_t colon = part.find(':');
        if (colon != std::string::npos) {
            pts.push_back({std::stof(part.substr(0, colon)), std::stof(part.substr(colon + 1))});
        }

        start = end + 1;
    }

    return pts;
}

static bool parse_box(const std::string & field, sam3_box & box) {
    if (field.empty()) {
        return false;
    }

    float vals[4];
    size_t start = 0;
    for (int i = 0; i < 4; ++i) {
        size_t end = field.find(':', start);
        if (end == std::string::npos) {
            end = field.size();
        }
        vals[i] = std::stof(field.substr(start, end - start));
        start = end + 1;
    }

    box = {vals[0], vals[1], vals[2], vals[3]};
    return true;
}

static std::vector<phase6_case> load_cases(const std::string & path) {
    std::vector<phase6_case> cases;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> fields;
        size_t start = 0;
        while (start <= line.size()) {
            size_t end = line.find('\t', start);
            if (end == std::string::npos) {
                end = line.size();
            }
            fields.push_back(line.substr(start, end - start));
            start = end + 1;
            if (end == line.size()) {
                break;
            }
        }
        while (fields.size() < 5) {
            fields.emplace_back();
        }

        phase6_case tc;
        tc.id = fields[0];
        tc.params.multimask = fields[1] == "1";
        tc.params.pos_points = parse_points(fields[2]);
        tc.params.neg_points = parse_points(fields[3]);
        tc.params.use_box = parse_box(fields[4], tc.params.box);
        cases.push_back(std::move(tc));
    }

    return cases;
}

static const std::vector<tensor_case> & tensor_cases() {
    static const std::vector<tensor_case> cases = {
        {"sam_dec_final_queries", "Decoder final queries", 6e-3f},
        {"sam_dec_masks", "Decoder mask logits", 2e-2f},
        {"sam_dec_iou", "Decoder IoU predictions", 2e-3f},
        {"sam_dec_obj_score", "Decoder object score logit", 3e-3f},
        {"sam_dec_sam_token", "Decoder SAM token", 6e-3f},
    };
    return cases;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <ref_dir> <prephase_ref_dir> <model_path> <cases_tsv>\n", argv[0]);
        return 1;
    }

    const std::string ref_dir = argv[1];
    const std::string prephase_ref_dir = argv[2];
    const std::string model_path = argv[3];
    const std::string cases_path = argv[4];
    const std::string cpp_root = ref_dir + "/cpp_out_phase6_metal";

    auto cases = load_cases(cases_path);
    if (cases.empty()) {
        fprintf(stderr, "No phase 6 cases loaded from %s\n", cases_path.c_str());
        return 1;
    }

    if (!ensure_dir(cpp_root)) {
        fprintf(stderr, "Failed to create %s\n", cpp_root.c_str());
        return 1;
    }

    sam3_params params;
    params.model_path = model_path;
    params.n_threads = 1;
    params.use_gpu = true;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "Failed to load model from %s\n", model_path.c_str());
        return 1;
    }

    bool overall_ok = true;
    for (const auto & tc : cases) {
        fprintf(stderr, "\n=== Metal Phase 6 Case %s ===\n", tc.id.c_str());

        const std::string case_ref_dir = ref_dir + "/" + tc.id;
        const std::string case_cpp_dir = cpp_root + "/" + tc.id;
        if (!ensure_dir(case_cpp_dir)) {
            fprintf(stderr, "Failed to create %s\n", case_cpp_dir.c_str());
            overall_ok = false;
            continue;
        }

        if (!sam3_test_dump_phase6_from_ref_inputs(*model, prephase_ref_dir, tc.params,
                                                   case_cpp_dir, params.n_threads)) {
            fprintf(stderr, "  [FAIL] could not run Metal phase 6 dump for %s\n", tc.id.c_str());
            overall_ok = false;
            continue;
        }

        for (const auto & tensor : tensor_cases()) {
            auto ref = load_ref_f32(case_ref_dir + "/" + tensor.name);
            auto got = load_ref_f32(case_cpp_dir + "/" + tensor.name);
            if (ref.data.empty() || got.data.empty() || ref.numel() != got.numel()) {
                fprintf(stderr, "  [FAIL] %s missing or shape mismatch\n", tensor.name);
                overall_ok = false;
                continue;
            }

            auto cmp = compare_tensors(got.data.data(), ref.data.data(), ref.numel(), tensor.tol);
            const bool ok = cmp.max_diff <= tensor.tol;
            fprintf(stderr,
                    "  %-22s max=%.6f mean=%.6f cos=%.8f tol=%.6f %s\n",
                    tensor.name,
                    cmp.max_diff,
                    cmp.mean_diff,
                    cmp.cosine_sim,
                    tensor.tol,
                    ok ? "PASS" : "FAIL");
            if (!ok) {
                overall_ok = false;
            }
        }
    }

    sam3_free_model(*model);
    return overall_ok ? 0 : 1;
}
