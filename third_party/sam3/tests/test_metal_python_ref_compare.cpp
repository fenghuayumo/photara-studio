#include "sam3.h"

#include "test_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct metric_row {
    std::string tensor;
    std::string shape;
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float mean_rel = 0.0f;
    float cosine = 0.0f;
    bool measured = false;
    std::string note;
};

struct backend_run {
    std::string backend;
    double encode_ms = 0.0;
    double final_ms = 0.0;
    std::vector<metric_row> encoder_rows;
    std::vector<metric_row> final_rows;
};

static std::string shape_to_string(const std::vector<int> & shape) {
    std::string s;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) s += "x";
        s += std::to_string(shape[i]);
    }
    return s;
}

static metric_row compute_metrics(const std::string & tensor,
                                  const float * got,
                                  const float * ref,
                                  int n,
                                  const std::vector<int> & ref_shape) {
    metric_row row;
    row.tensor = tensor;
    row.shape = shape_to_string(ref_shape);
    if (n <= 0) {
        row.note = "empty";
        return row;
    }

    double sum_abs = 0.0;
    double sum_rel = 0.0;
    double dot_ab = 0.0;
    double dot_aa = 0.0;
    double dot_bb = 0.0;
    float max_abs = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float a = got[i];
        const float b = ref[i];
        const float d = fabsf(a - b);
        sum_abs += d;
        sum_rel += d / (fabsf(b) + 1e-8f);
        if (d > max_abs) {
            max_abs = d;
        }
        dot_ab += (double) a * (double) b;
        dot_aa += (double) a * (double) a;
        dot_bb += (double) b * (double) b;
    }

    row.max_abs = max_abs;
    row.mean_abs = (float) (sum_abs / n);
    row.mean_rel = (float) (sum_rel / n);
    const double denom = sqrt(dot_aa) * sqrt(dot_bb);
    row.cosine = denom > 0.0 ? (float) (dot_ab / denom) : 0.0f;
    row.measured = true;
    return row;
}

static std::vector<float> ggml_dump_to_nhwc_flat(const ref_tensor_f32 & cpp) {
    std::vector<float> out;
    if (cpp.shape.size() < 3) {
        return out;
    }

    const int E = cpp.shape[0];
    const int W = cpp.shape[1];
    const int H = cpp.shape[2];
    out.resize((size_t) E * W * H);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int e = 0; e < E; ++e) {
                out[(size_t) h * W * E + (size_t) w * E + e] =
                        cpp.data[(size_t) e + (size_t) w * E + (size_t) h * E * W];
            }
        }
    }
    return out;
}

static std::vector<float> ggml_dump_to_nchw_flat(const ref_tensor_f32 & cpp) {
    std::vector<float> out;
    if (cpp.shape.size() < 3) {
        return out;
    }

    const int C = cpp.shape[0];
    const int W = cpp.shape[1];
    const int H = cpp.shape[2];
    out.resize((size_t) C * W * H);
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                out[(size_t) c * H * W + (size_t) h * W + w] =
                        cpp.data[(size_t) c + (size_t) w * C + (size_t) h * C * W];
            }
        }
    }
    return out;
}

static bool parse_phase6_case(const std::string & path,
                              sam3_pvs_params & params,
                              std::string & case_id) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) {
        return false;
    }

    char line[1024];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return false;
    }
    fclose(f);

    std::string row(line);
    while (!row.empty() && (row.back() == '\n' || row.back() == '\r')) {
        row.pop_back();
    }

    std::vector<std::string> fields;
    size_t start = 0;
    while (start <= row.size()) {
        size_t end = row.find('\t', start);
        if (end == std::string::npos) {
            end = row.size();
        }
        fields.emplace_back(row.substr(start, end - start));
        start = end + 1;
        if (end == row.size()) {
            break;
        }
    }
    while (fields.size() < 5) {
        fields.emplace_back();
    }

    case_id = fields[0];
    params.multimask = fields[1] == "1";
    params.use_box = false;

    if (!fields[4].empty()) {
        float vals[4] = {0, 0, 0, 0};
        size_t start = 0;
        for (int i = 0; i < 4; ++i) {
            size_t end = fields[4].find(':', start);
            if (end == std::string::npos) end = fields[4].size();
            vals[i] = std::stof(fields[4].substr(start, end - start));
            start = end + 1;
        }
        params.box = {vals[0], vals[1], vals[2], vals[3]};
        params.use_box = true;
    }

    auto parse_points = [](const std::string & field, std::vector<sam3_point> & dst) {
        size_t start = 0;
        while (start < field.size()) {
            size_t end = field.find('|', start);
            if (end == std::string::npos) end = field.size();
            const std::string part = field.substr(start, end - start);
            size_t sep = part.find(':');
            if (sep != std::string::npos) {
                sam3_point p;
                p.x = std::stof(part.substr(0, sep));
                p.y = std::stof(part.substr(sep + 1));
                dst.push_back(p);
            }
            start = end + 1;
        }
    };

    parse_points(fields[2], params.pos_points);
    parse_points(fields[3], params.neg_points);
    return true;
}

static metric_row compare_state_nhwc(const sam3_state & state,
                                     const std::string & tensor_name,
                                     const std::string & ref_path,
                                     const std::string & dump_dir) {
    metric_row row;
    auto ref = load_ref_f32(ref_path);
    row.tensor = tensor_name;
    row.shape = shape_to_string(ref.shape);

    if (ref.data.empty()) {
        row.note = "missing python ref";
        return row;
    }
    if (!sam3_dump_state_tensor(state, tensor_name, dump_dir + "/" + tensor_name)) {
        row.note = "missing state tensor";
        return row;
    }
    auto cpp = load_ref_f32(dump_dir + "/" + tensor_name);
    if (cpp.data.empty()) {
        row.note = "missing dumped tensor";
        return row;
    }

    auto transposed = ggml_dump_to_nhwc_flat(cpp);
    if ((int) transposed.size() != ref.numel()) {
        row.note = "numel mismatch";
        return row;
    }
    return compute_metrics(tensor_name, transposed.data(), ref.data.data(), ref.numel(), ref.shape);
}

static metric_row compare_state_nchw(const sam3_state & state,
                                     const std::string & tensor_name,
                                     const std::string & ref_path,
                                     const std::string & dump_dir) {
    metric_row row;
    auto ref = load_ref_f32(ref_path);
    row.tensor = tensor_name;
    row.shape = shape_to_string(ref.shape);

    if (ref.data.empty()) {
        row.note = "missing python ref";
        return row;
    }
    if (!sam3_dump_state_tensor(state, tensor_name, dump_dir + "/" + tensor_name)) {
        row.note = "missing state tensor";
        return row;
    }
    auto cpp = load_ref_f32(dump_dir + "/" + tensor_name);
    if (cpp.data.empty()) {
        row.note = "missing dumped tensor";
        return row;
    }

    auto transposed = ggml_dump_to_nchw_flat(cpp);
    if ((int) transposed.size() != ref.numel()) {
        row.note = "numel mismatch";
        return row;
    }
    return compute_metrics(tensor_name, transposed.data(), ref.data.data(), ref.numel(), ref.shape);
}

static bool save_ref_f32(const std::string & path,
                         const std::vector<float> & data,
                         const std::vector<int> & shape) {
    FILE * shape_f = fopen((path + ".shape").c_str(), "w");
    if (!shape_f) {
        return false;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
        fprintf(shape_f, "%s%d", i == 0 ? "" : ",", shape[i]);
    }
    fclose(shape_f);

    FILE * data_f = fopen((path + ".bin").c_str(), "wb");
    if (!data_f) {
        return false;
    }
    const size_t n_written = fwrite(data.data(), sizeof(float), data.size(), data_f);
    fclose(data_f);
    return n_written == data.size();
}

static bool export_dump_as_nchw_ref(const std::string & ggml_dump_path,
                                    const std::string & out_path) {
    auto cpp = load_ref_f32(ggml_dump_path);
    if (cpp.data.empty() || cpp.shape.size() < 3) {
        return false;
    }
    const std::vector<float> transposed = ggml_dump_to_nchw_flat(cpp);
    const std::vector<int> shape = {1, cpp.shape[0], cpp.shape[2], cpp.shape[1]};
    return save_ref_f32(out_path, transposed, shape);
}

static metric_row compare_dump_direct(const std::string & name,
                                      const std::string & got_path,
                                      const std::string & ref_path) {
    metric_row row;
    auto got = load_ref_f32(got_path);
    auto ref = load_ref_f32(ref_path);
    row.tensor = name;
    row.shape = shape_to_string(ref.shape);

    if (got.data.empty() || ref.data.empty()) {
        row.note = "missing dump";
        return row;
    }
    if (got.numel() != ref.numel()) {
        row.note = "numel mismatch";
        return row;
    }
    return compute_metrics(name, got.data.data(), ref.data.data(), ref.numel(), ref.shape);
}

static backend_run run_backend(const std::string & backend_name,
                               bool use_gpu,
                               const std::string & model_path,
                               const std::string & ref_dir,
                               const sam3_pvs_params & pvs_params,
                               const std::string & phase6_case_id) {
    backend_run run;
    run.backend = backend_name;

    sam3_params params;
    params.model_path = model_path;
    params.n_threads = 4;
    params.use_gpu = use_gpu;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "Failed to load model for backend=%s\n", backend_name.c_str());
        return run;
    }
    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "Failed to create state for backend=%s\n", backend_name.c_str());
        return run;
    }

    auto pre = load_ref_f32(ref_dir + "/preprocessed");
    if (pre.data.empty()) {
        fprintf(stderr, "Missing preprocessed tensor in %s\n", ref_dir.c_str());
        return run;
    }

    const std::string dump_root = "/tmp/sam3_pyref_" + backend_name;
    ensure_dir(dump_root);

    auto t0 = std::chrono::high_resolution_clock::now();
    const bool ok = sam3_encode_image_from_preprocessed(*state, *model, pre.data.data(), 1008);
    auto t1 = std::chrono::high_resolution_clock::now();
    run.encode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!ok) {
        fprintf(stderr, "sam3_encode_image_from_preprocessed failed for backend=%s\n", backend_name.c_str());
        return run;
    }

    const char * nhwc_tensors[] = {
        "dbg_patch_embed",
        "dbg_after_pos_embed",
        "dbg_after_ln_pre",
        "dbg_block_0_out",
        "dbg_block_1_out",
        "dbg_block_2_out",
        "dbg_block_10_out",
        "dbg_block_14_out",
        "dbg_block_15_out",
        "dbg_block_16_out",
    };
    for (const char * name : nhwc_tensors) {
        std::string ref_name = name;
        if (strncmp(name, "dbg_", 4) == 0) {
            ref_name = std::string(name + 4);
        }
        run.encoder_rows.push_back(compare_state_nhwc(*state, name, ref_dir + "/" + ref_name, dump_root));
    }

    run.encoder_rows.push_back(compare_state_nchw(*state, "vit_output", ref_dir + "/vit_output_bchw", dump_root));
    run.encoder_rows.push_back(compare_state_nchw(*state, "neck_trk_0", ref_dir + "/neck_trk_0", dump_root));
    run.encoder_rows.push_back(compare_state_nchw(*state, "neck_trk_1", ref_dir + "/neck_trk_1", dump_root));
    run.encoder_rows.push_back(compare_state_nchw(*state, "neck_trk_2", ref_dir + "/neck_trk_2", dump_root));

    const std::string phase6_inputs = dump_root + "/phase6_inputs";
    const std::string phase6_dump = dump_root + "/phase6_out";
    ensure_dir(phase6_inputs);
    ensure_dir(phase6_dump);

    if (!export_dump_as_nchw_ref(dump_root + "/neck_trk_0", phase6_inputs + "/neck_trk_0") ||
        !export_dump_as_nchw_ref(dump_root + "/neck_trk_1", phase6_inputs + "/neck_trk_1") ||
        !export_dump_as_nchw_ref(dump_root + "/neck_trk_2", phase6_inputs + "/neck_trk_2")) {
        metric_row row;
        row.tensor = "phase6_bridge";
        row.note = "failed to export neck_trk_* as NCHW ref inputs";
        run.final_rows.push_back(std::move(row));
        return run;
    }

    state.reset();
    sam3_free_model(*model);
    model.reset();

    sam3_params p6_params;
    p6_params.model_path = model_path;
    p6_params.n_threads = 4;
    p6_params.use_gpu = false;

    auto phase6_model = sam3_load_model(p6_params);
    if (!phase6_model) {
        metric_row row;
        row.tensor = "phase6_model";
        row.note = "failed to load CPU model for phase6 compare";
        run.final_rows.push_back(std::move(row));
        return run;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    const bool phase6_ok = sam3_test_dump_phase6_from_ref_inputs(*phase6_model, phase6_inputs,
                                                                 pvs_params, phase6_dump,
                                                                 p6_params.n_threads);
    auto t3 = std::chrono::high_resolution_clock::now();
    run.final_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    if (!phase6_ok) {
        metric_row row;
        row.tensor = "phase6_compare";
        row.note = "sam3_test_dump_phase6_from_ref_inputs failed";
        run.final_rows.push_back(std::move(row));
        return run;
    }

    const std::string p6_ref_dir = ref_dir + "/phase6/" + phase6_case_id;
    const char * final_tensors[] = {
        "sam_dec_masks",
        "sam_dec_iou",
        "sam_dec_obj_score",
        "sam_dec_sam_token",
    };
    for (const char * name : final_tensors) {
        run.final_rows.push_back(compare_dump_direct(name, phase6_dump + "/" + name, p6_ref_dir + "/" + name));
    }

    return run;
}

static void print_table(const std::string & title, const std::vector<metric_row> & rows) {
    fprintf(stderr, "\n%s\n", title.c_str());
    fprintf(stderr, "| tensor | shape | max abs diff | mean abs diff | mean rel err | cosine | note |\n");
    fprintf(stderr, "|--------|-------|--------------|---------------|--------------|--------|------|\n");
    for (const auto & row : rows) {
        if (!row.measured) {
            fprintf(stderr, "| %s | %s | n/a | n/a | n/a | n/a | %s |\n",
                    row.tensor.c_str(), row.shape.c_str(), row.note.c_str());
            continue;
        }
        fprintf(stderr, "| %s | %s | %.6e | %.6e | %.6e | %.8f | %s |\n",
                row.tensor.c_str(), row.shape.c_str(), row.max_abs,
                row.mean_abs, row.mean_rel, row.cosine, row.note.c_str());
    }
}

static void print_summary(const backend_run & run) {
    fprintf(stderr, "\n=== %s vs Python ===\n", run.backend.c_str());
    fprintf(stderr, "encode_ms=%.3f final_ms=%.3f\n", run.encode_ms, run.final_ms);
    print_table("Encoder / Neck", run.encoder_rows);
    print_table("Final Decoder Outputs", run.final_rows);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <model.ggml> [ref_dir] [backend=metal|cpu|both] [phase6_case_tsv]\n",
                argv[0]);
        fprintf(stderr,
                "Defaults: ref_dir=tests/debug_pipeline/e2e_ref backend=both phase6_case_tsv=tests/debug_pipeline/phase6_cat_box.tsv\n");
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ref_dir = argc >= 3 ? argv[2] : "tests/debug_pipeline/e2e_ref";
    const std::string backend_mode = argc >= 4 ? argv[3] : "both";
    const std::string case_tsv = argc >= 5 ? argv[4] : "tests/debug_pipeline/phase6_cat_box.tsv";

    sam3_pvs_params pvs_params;
    std::string phase6_case_id;
    if (!parse_phase6_case(case_tsv, pvs_params, phase6_case_id)) {
        fprintf(stderr, "Failed to parse phase 6 case file: %s\n", case_tsv.c_str());
        return 1;
    }

    std::vector<backend_run> runs;
    if (backend_mode == "cpu" || backend_mode == "both") {
        runs.push_back(run_backend("cpu", false, model_path, ref_dir, pvs_params, phase6_case_id));
    }
    if (backend_mode == "metal" || backend_mode == "both") {
        runs.push_back(run_backend("metal", true, model_path, ref_dir, pvs_params, phase6_case_id));
    }

    for (const auto & run : runs) {
        print_summary(run);
    }

    return 0;
}
