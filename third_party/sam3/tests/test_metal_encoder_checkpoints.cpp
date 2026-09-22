#include "sam3.h"
#include "test_utils.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct checkpoint_stat {
    std::string name;
    std::string shape;
    compare_result diff;
};

static std::string format_shape(const std::vector<int> & shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

static bool dump_checkpoints(const sam3_state & state,
                             const std::vector<std::string> & checkpoints,
                             const std::string & output_dir) {
    for (const auto & name : checkpoints) {
        if (!sam3_dump_state_tensor(state, name, output_dir + "/" + name)) {
            fprintf(stderr, "failed to dump checkpoint '%s'\n", name.c_str());
            return false;
        }
    }

    return true;
}

static bool run_backend(const std::string & model_path,
                        const std::vector<float> & chw_data,
                        int img_size,
                        bool use_gpu,
                        int n_threads,
                        const std::vector<std::string> & checkpoints,
                        const std::string & dump_dir,
                        double & elapsed_ms) {
    sam3_params params;
    params.model_path = model_path;
    params.use_gpu = use_gpu;
    params.n_threads = n_threads;

    auto model = sam3_load_model(params);
    if (!model) {
        fprintf(stderr, "failed to load model for %s\n", use_gpu ? "Metal" : "CPU");
        return false;
    }

    auto state = sam3_create_state(*model, params);
    if (!state) {
        fprintf(stderr, "failed to create state for %s\n", use_gpu ? "Metal" : "CPU");
        return false;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    const bool ok = sam3_encode_image_from_preprocessed(*state, *model, chw_data.data(), img_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!ok) {
        fprintf(stderr, "%s encode failed\n", use_gpu ? "Metal" : "CPU");
        return false;
    }

    if (!dump_checkpoints(*state, checkpoints, dump_dir)) {
        return false;
    }

    state.reset();
    sam3_free_model(*model);
    model.reset();
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.ggml> [ref_dir]\n", argv[0]);
        fprintf(stderr, "Default ref_dir: tests/ref_phase3\n");
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ref_dir = argc >= 3 ? argv[2] : "tests/ref_phase3";

    auto preprocessed = load_ref_f32(ref_dir + "/preprocessed");
    if (preprocessed.data.empty()) {
        fprintf(stderr, "failed to load %s/preprocessed.bin\n", ref_dir.c_str());
        return 1;
    }
    if (preprocessed.shape.size() != 4) {
        fprintf(stderr, "unexpected preprocessed shape rank: %zu\n", preprocessed.shape.size());
        return 1;
    }

    const int img_size = preprocessed.shape[2];
    std::vector<std::string> checkpoints = {
        "dbg_patch_embed",
        "dbg_after_pos_embed",
        "dbg_ln_pre_norm",
        "dbg_ln_pre_scale",
        "dbg_after_ln_pre",
        "dbg_block_15_norm1",
        "dbg_block_15_attn_out",
        "dbg_block_15_attn_proj",
        "dbg_block_15_resid1",
        "dbg_block_15_norm2",
        "dbg_block_15_mlp",
    };
    for (int i = 0; i < 31; ++i) {
        checkpoints.emplace_back("dbg_block_" + std::to_string(i) + "_out");
    }
    checkpoints.emplace_back("vit_output");
    checkpoints.emplace_back("neck_trk_0");
    checkpoints.emplace_back("neck_trk_1");
    checkpoints.emplace_back("neck_trk_2");

    const std::string cpu_dir = "/tmp/sam3_cpu_checkpoints";
    const std::string metal_dir = "/tmp/sam3_metal_checkpoints";
    ensure_dir(cpu_dir);
    ensure_dir(metal_dir);

    double cpu_ms = 0.0;
    double metal_ms = 0.0;

    fprintf(stderr, "\n=== CPU Run ===\n");
    if (!run_backend(model_path, preprocessed.data, img_size, false, 8, checkpoints, cpu_dir, cpu_ms)) {
        return 1;
    }
    fprintf(stderr, "CPU encoder wall time: %.1f ms\n", cpu_ms);

    fprintf(stderr, "\n=== Metal Run ===\n");
    if (!run_backend(model_path, preprocessed.data, img_size, true, 8, checkpoints, metal_dir, metal_ms)) {
        return 1;
    }
    fprintf(stderr, "Metal encoder wall time: %.1f ms\n", metal_ms);

    fprintf(stderr, "\n=== CPU vs Metal Checkpoints ===\n");
    fprintf(stderr, "%-22s %-18s %14s %14s %12s\n",
            "checkpoint", "shape", "max_abs_diff", "mean_abs_diff", "n_bad");

    std::vector<checkpoint_stat> stats;
    stats.reserve(checkpoints.size());

    for (const auto & name : checkpoints) {
        auto cpu = load_ref_f32(cpu_dir + "/" + name);
        auto metal = load_ref_f32(metal_dir + "/" + name);
        if (cpu.data.empty() || metal.data.empty()) {
            fprintf(stderr, "%-22s %-18s %14s %14s %12s\n",
                    name.c_str(), "-", "load-fail", "load-fail", "-");
            continue;
        }
        if (cpu.shape != metal.shape) {
            fprintf(stderr, "%-22s %-18s %14s %14s %12s\n",
                    name.c_str(), "shape-mismatch", "shape-mismatch", "shape-mismatch", "-");
            continue;
        }

        checkpoint_stat stat;
        stat.name = name;
        stat.shape = format_shape(cpu.shape);
        stat.diff = compare_tensors(cpu.data.data(), metal.data.data(), cpu.numel(), 1e-4f);
        stats.push_back(stat);

        fprintf(stderr, "%-22s %-18s %14.6f %14.6f %12d\n",
                stat.name.c_str(),
                stat.shape.c_str(),
                stat.diff.max_diff,
                stat.diff.mean_diff,
                stat.diff.n_bad);
    }

    const float checkpoint_tol = 1e-2f;
    const checkpoint_stat * first_mismatch = nullptr;
    for (const auto & stat : stats) {
        if (stat.diff.max_diff > checkpoint_tol) {
            first_mismatch = &stat;
            break;
        }
    }

    if (first_mismatch) {
        fprintf(stderr,
                "\nFIRST CLEAR MISMATCH: %s shape=%s max_abs_diff=%.6f mean_abs_diff=%.6f\n",
                first_mismatch->name.c_str(),
                first_mismatch->shape.c_str(),
                first_mismatch->diff.max_diff,
                first_mismatch->diff.mean_diff);
        return 1;
    }

    fprintf(stderr, "\nAll checkpoints stayed within %.3g max abs diff.\n", checkpoint_tol);
    return 0;
}
