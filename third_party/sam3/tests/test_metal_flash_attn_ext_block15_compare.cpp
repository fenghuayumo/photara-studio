#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "sam3.h"
#include "test_utils.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

struct tensor_case {
    ref_tensor_f32 qkv_proj;
    ref_tensor_f32 freqs_cis;
    ref_tensor_f32 q_heads;
    ref_tensor_f32 k_heads;
    ref_tensor_f32 q_rope;
    ref_tensor_f32 k_rope;
    ref_tensor_f32 q_flash;
    ref_tensor_f32 k_flash;
    ref_tensor_f32 v_base;
    ref_tensor_f32 v_flash;
    ref_tensor_f32 attn_out;
};

struct prep_run_result {
    std::map<std::string, ref_tensor_f32> tensors;
};

struct backend_run_result {
    double elapsed_ms = 0.0;
    std::map<std::string, sam3_tensor_info> infos;
};

struct isolated_run_result {
    double elapsed_ms = 0.0;
    std::vector<float> output;
    std::vector<int> shape;
};

static void usage(const char * argv0) {
    fprintf(stderr, "Usage: %s <model.ggml> [ref_dir]\n", argv0);
    fprintf(stderr, "Default ref_dir: tests/ref_phase3\n");
}

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

static const char * ggml_type_name_public(int type) {
    switch ((ggml_type) type) {
        case GGML_TYPE_F32: return "F32";
        case GGML_TYPE_F16: return "F16";
        case GGML_TYPE_BF16: return "BF16";
        case GGML_TYPE_I32: return "I32";
        default: return "other";
    }
}

static bool dump_named_tensors(const sam3_state & state,
                               const std::vector<std::string> & names,
                               const std::string & output_dir,
                               std::map<std::string, sam3_tensor_info> & infos) {
    for (const auto & name : names) {
        sam3_tensor_info info;
        if (!sam3_get_state_tensor_info(state, name, info)) {
            fprintf(stderr, "missing tensor info for %s\n", name.c_str());
            return false;
        }
        infos[name] = info;

        if (!sam3_dump_state_tensor(state, name, output_dir + "/" + name)) {
            fprintf(stderr, "failed to dump %s\n", name.c_str());
            return false;
        }
    }
    return true;
}

static bool run_encoder_backend(const std::string & model_path,
                                const std::vector<float> & chw_data,
                                int img_size,
                                bool use_gpu,
                                int n_threads,
                                const std::vector<std::string> & names,
                                const std::string & dump_dir,
                                backend_run_result & result) {
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
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!ok) {
        fprintf(stderr, "%s encode failed\n", use_gpu ? "Metal" : "CPU");
        return false;
    }

    if (!dump_named_tensors(*state, names, dump_dir, result.infos)) {
        return false;
    }

    state.reset();
    sam3_free_model(*model);
    model.reset();
    return true;
}

static ggml_backend_t create_backend(bool use_gpu, int n_threads) {
#ifdef GGML_USE_METAL
    if (use_gpu) {
        return ggml_backend_metal_init();
    }
#else
    (void) use_gpu;
#endif
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
    return backend;
}

static std::vector<int> normalize_shape_4d(const std::vector<int> & shape) {
    std::vector<int> out = shape;
    while (out.size() < 4) {
        out.push_back(1);
    }
    return out;
}

static ggml_tensor * new_tensor_4d_from_shape(ggml_context * ctx,
                                              ggml_type type,
                                              const std::vector<int> & shape) {
    const std::vector<int> s = normalize_shape_4d(shape);
    if (s.size() != 4) {
        return nullptr;
    }
    return ggml_new_tensor_4d(ctx, type, s[0], s[1], s[2], s[3]);
}

static isolated_run_result run_isolated_attention(ggml_backend_t backend,
                                                  const tensor_case & tc,
                                                  bool force_contiguous_v) {
    isolated_run_result result;

    const size_t ctx_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return result;
    }

    ggml_tensor * q = new_tensor_4d_from_shape(ctx, GGML_TYPE_F32, tc.q_flash.shape);
    ggml_tensor * k = new_tensor_4d_from_shape(ctx, GGML_TYPE_F32, tc.k_flash.shape);
    ggml_tensor * v_src = nullptr;
    ggml_tensor * v = nullptr;

    ggml_set_input(q);
    ggml_set_input(k);
    if (!q || !k) {
        ggml_free(ctx);
        return result;
    }

    if (force_contiguous_v) {
        v = new_tensor_4d_from_shape(ctx, GGML_TYPE_F32, tc.v_flash.shape);
        ggml_set_input(v);
    } else {
        v_src = new_tensor_4d_from_shape(ctx, GGML_TYPE_F32, tc.v_base.shape);
        ggml_set_input(v_src);
        v = ggml_permute(ctx, v_src, 0, 2, 1, 3);
    }
    if (!v) {
        ggml_free(ctx);
        return result;
    }

    const float scale = 1.0f / std::sqrt((float) tc.q_flash.shape[0]);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return result;
    }

    ggml_backend_tensor_set(q, tc.q_flash.data.data(), 0, tc.q_flash.data.size() * sizeof(float));
    ggml_backend_tensor_set(k, tc.k_flash.data.data(), 0, tc.k_flash.data.size() * sizeof(float));
    if (force_contiguous_v) {
        ggml_backend_tensor_set(v, tc.v_flash.data.data(), 0, tc.v_flash.data.size() * sizeof(float));
    } else {
        ggml_backend_tensor_set(v_src, tc.v_base.data.data(), 0, tc.v_base.data.size() * sizeof(float));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (status == GGML_STATUS_SUCCESS) {
        result.shape = {
            (int) out->ne[0],
            (int) out->ne[1],
            (int) out->ne[2],
            (int) out->ne[3],
        };
        result.output.resize((size_t) ggml_nelements(out));
        ggml_backend_tensor_get(out, result.output.data(), 0, result.output.size() * sizeof(float));
    } else {
        fprintf(stderr, "isolated attention failed: %s\n", ggml_status_to_string(status));
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}

static ref_tensor_f32 tensor_to_ref(ggml_tensor * t) {
    ref_tensor_f32 out;
    out.shape = {
        (int) t->ne[0],
        (int) t->ne[1],
        (int) t->ne[2],
        (int) t->ne[3],
    };
    while (!out.shape.empty() && out.shape.back() == 1) {
        out.shape.pop_back();
    }
    if (out.shape.empty()) {
        out.shape.push_back(1);
    }

    out.data.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data.data(), 0, out.data.size() * sizeof(float));
    return out;
}

static prep_run_result run_isolated_qkv_prep(ggml_backend_t backend, const tensor_case & tc) {
    prep_run_result result;

    const int E = 1024;
    const int NH = 16;
    const int HD = 64;
    const int W = 72;
    const int H = 72;
    const int N = W * H;

    const size_t ctx_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return result;
    }

    ggml_tensor * qkv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3 * E, W, H, 1);
    ggml_set_input(qkv);
    ggml_set_name(qkv, "qkv_proj");

    ggml_tensor * cur = ggml_reshape_4d(ctx, qkv, E, 3, N, 1);
    cur = ggml_cont(ctx, ggml_permute(ctx, cur, 0, 3, 1, 2));

    ggml_tensor * q_split = ggml_view_3d(ctx, cur, E, N, 1, cur->nb[1], cur->nb[2], 0);
    ggml_tensor * k_split = ggml_view_3d(ctx, cur, E, N, 1, cur->nb[1], cur->nb[2], 1 * cur->nb[3]);
    ggml_tensor * v_split = ggml_view_3d(ctx, cur, E, N, 1, cur->nb[1], cur->nb[2], 2 * cur->nb[3]);
    ggml_set_name(q_split, "q_split");
    ggml_set_name(k_split, "k_split");
    ggml_set_name(v_split, "v_split");

    ggml_tensor * q_heads_base = ggml_reshape_4d(ctx, q_split, HD, NH, N, 1);
    ggml_tensor * k_heads_base = ggml_reshape_4d(ctx, k_split, HD, NH, N, 1);
    ggml_tensor * v_heads_base = ggml_reshape_4d(ctx, v_split, HD, NH, N, 1);
    ggml_set_name(q_heads_base, "q_heads_base");
    ggml_set_name(k_heads_base, "k_heads_base");
    ggml_set_name(v_heads_base, "v_heads_base");

    ggml_tensor * q_heads = ggml_cont(ctx, ggml_permute(ctx, q_heads_base, 0, 2, 1, 3));
    ggml_tensor * k_heads = ggml_cont(ctx, ggml_permute(ctx, k_heads_base, 0, 2, 1, 3));
    ggml_tensor * v_flash = ggml_permute(ctx, v_heads_base, 0, 2, 1, 3);
    ggml_set_name(q_heads, "q_heads");
    ggml_set_name(k_heads, "k_heads");
    ggml_set_name(v_flash, "v_flash");

    std::vector<ggml_tensor *> outs = {
        q_split,
        k_split,
        v_split,
        q_heads_base,
        k_heads_base,
        v_heads_base,
        q_heads,
        k_heads,
        v_flash,
    };
    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (ggml_tensor * out : outs) {
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return result;
    }

    ggml_backend_tensor_set(qkv, tc.qkv_proj.data.data(), 0, tc.qkv_proj.data.size() * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status == GGML_STATUS_SUCCESS) {
        for (ggml_tensor * out : outs) {
            result.tensors[out->name] = tensor_to_ref(out);
        }
    } else {
        fprintf(stderr, "isolated qkv prep failed: %s\n", ggml_status_to_string(status));
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}

static isolated_run_result run_isolated_rope(ggml_backend_t backend,
                                             const ref_tensor_f32 & x_in,
                                             const ref_tensor_f32 & freqs_in) {
    isolated_run_result result;

    const std::vector<int> x_shape = normalize_shape_4d(x_in.shape);
    const std::vector<int> freqs_shape = normalize_shape_4d(freqs_in.shape);
    if (x_shape[0] != 64 || freqs_shape[0] != 2) {
        return result;
    }

    const int64_t head_dim = x_shape[0];
    const int64_t N = x_shape[1];
    const int64_t nheads_B = x_shape[2] * x_shape[3];
    const int64_t half = head_dim / 2;

    const size_t ctx_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return result;
    }

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, x_shape[0], x_shape[1], x_shape[2] * x_shape[3]);
    ggml_tensor * freqs = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, freqs_shape[0], freqs_shape[1], freqs_shape[2] * freqs_shape[3]);
    ggml_set_input(x);
    ggml_set_input(freqs);

    ggml_tensor * x_pairs = ggml_reshape_4d(ctx, x, 2, half, N, nheads_B);
    ggml_tensor * fc = ggml_reshape_4d(ctx, freqs, 2, half, N, 1);

    ggml_tensor * cos_f = ggml_view_4d(ctx, fc, 1, half, N, 1, fc->nb[1], fc->nb[2], fc->nb[3], 0);
    ggml_tensor * sin_f = ggml_view_4d(ctx, fc, 1, half, N, 1, fc->nb[1], fc->nb[2], fc->nb[3], fc->nb[0]);
    ggml_tensor * x_re = ggml_view_4d(ctx, x_pairs, 1, half, N, nheads_B, x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], 0);
    ggml_tensor * x_im = ggml_view_4d(ctx, x_pairs, 1, half, N, nheads_B, x_pairs->nb[1], x_pairs->nb[2], x_pairs->nb[3], x_pairs->nb[0]);

    ggml_tensor * out_re = ggml_sub(ctx, ggml_mul(ctx, x_re, cos_f), ggml_mul(ctx, x_im, sin_f));
    ggml_tensor * out_im = ggml_add(ctx, ggml_mul(ctx, x_re, sin_f), ggml_mul(ctx, x_im, cos_f));
    ggml_tensor * out = ggml_concat(ctx, out_re, out_im, 0);
    out = ggml_reshape_3d(ctx, ggml_cont(ctx, out), head_dim, N, nheads_B);
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return result;
    }

    ggml_backend_tensor_set(x, x_in.data.data(), 0, x_in.data.size() * sizeof(float));
    ggml_backend_tensor_set(freqs, freqs_in.data.data(), 0, freqs_in.data.size() * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status == GGML_STATUS_SUCCESS) {
        result.shape = { (int) out->ne[0], (int) out->ne[1], (int) out->ne[2] };
        result.output.resize((size_t) ggml_nelements(out));
        ggml_backend_tensor_get(out, result.output.data(), 0, result.output.size() * sizeof(float));
    } else {
        fprintf(stderr, "isolated rope failed: %s\n", ggml_status_to_string(status));
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}

static bool load_tensor_case(const std::string & dir, tensor_case & tc) {
    tc.qkv_proj = load_ref_f32(dir + "/dbg_block_15_qkv_proj");
    tc.q_heads = load_ref_f32(dir + "/dbg_block_15_q_heads");
    tc.k_heads = load_ref_f32(dir + "/dbg_block_15_k_heads");
    tc.q_rope = load_ref_f32(dir + "/dbg_block_15_q_rope");
    tc.k_rope = load_ref_f32(dir + "/dbg_block_15_k_rope");
    tc.q_flash = load_ref_f32(dir + "/dbg_block_15_q_flash");
    tc.k_flash = load_ref_f32(dir + "/dbg_block_15_k_flash");
    tc.v_base = load_ref_f32(dir + "/dbg_block_15_v_heads_base");
    tc.v_flash = load_ref_f32(dir + "/dbg_block_15_v_flash");
    tc.attn_out = load_ref_f32(dir + "/dbg_block_15_attn_out");

    return !tc.qkv_proj.data.empty() &&
           !tc.q_heads.data.empty() &&
           !tc.k_heads.data.empty() &&
           !tc.q_rope.data.empty() &&
           !tc.k_rope.data.empty() &&
           !tc.q_flash.data.empty() &&
           !tc.k_flash.data.empty() &&
           !tc.v_base.data.empty() &&
           !tc.v_flash.data.empty() &&
           !tc.attn_out.data.empty();
}

static void print_tensor_info(const char * label,
                              const std::string & name,
                              const sam3_tensor_info & info) {
    fprintf(stderr,
            "%s %-24s shape=[%lld,%lld,%lld,%lld] strides=[%llu,%llu,%llu,%llu] type=%s contiguous=%d op=%d\n",
            label,
            name.c_str(),
            (long long) info.ne[0],
            (long long) info.ne[1],
            (long long) info.ne[2],
            (long long) info.ne[3],
            (unsigned long long) info.nb[0],
            (unsigned long long) info.nb[1],
            (unsigned long long) info.nb[2],
            (unsigned long long) info.nb[3],
            ggml_type_name_public(info.type),
            info.is_contiguous ? 1 : 0,
            info.op);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ref_dir = argc >= 3 ? argv[2] : "tests/ref_phase3";

    auto preprocessed = load_ref_f32(ref_dir + "/preprocessed");
    if (preprocessed.data.empty() || preprocessed.shape.size() != 4) {
        fprintf(stderr, "failed to load %s/preprocessed\n", ref_dir.c_str());
        return 1;
    }

    const int img_size = preprocessed.shape[2];
    const std::vector<std::string> names = {
        "dbg_block_15_norm1",
        "dbg_block_15_qkv_proj",
        "dbg_block_15_q_split",
        "dbg_block_15_k_split",
        "dbg_block_15_v_split",
        "dbg_block_15_q_heads_base",
        "dbg_block_15_k_heads_base",
        "dbg_block_15_v_heads_base",
        "dbg_block_15_q_heads",
        "dbg_block_15_k_heads",
        "dbg_block_15_v_flash",
        "dbg_block_15_q_rope",
        "dbg_block_15_k_rope",
        "dbg_block_15_q_flash",
        "dbg_block_15_k_flash",
        "dbg_block_15_attn_out",
    };

    const std::string cpu_dir = "/tmp/sam3_block15_flash_cpu";
    const std::string metal_dir = "/tmp/sam3_block15_flash_metal";
    ensure_dir(cpu_dir);
    ensure_dir(metal_dir);

    backend_run_result cpu_run;
    backend_run_result metal_run;

    fprintf(stderr, "\n=== Extract CPU Block-15 Attention Tensors ===\n");
    if (!run_encoder_backend(model_path, preprocessed.data, img_size, false, 8, names, cpu_dir, cpu_run)) {
        return 1;
    }
    fprintf(stderr, "CPU extract wall time: %.1f ms\n", cpu_run.elapsed_ms);

    fprintf(stderr, "\n=== Extract Metal Block-15 Attention Tensors ===\n");
    if (!run_encoder_backend(model_path, preprocessed.data, img_size, true, 8, names, metal_dir, metal_run)) {
        return 1;
    }
    fprintf(stderr, "Metal extract wall time: %.1f ms\n", metal_run.elapsed_ms);

    fprintf(stderr, "\n=== Exact Attention Case ===\n");
    fprintf(stderr, "block=15 variant=global_self_attention heads=16 head_dim=64 tokens=72*72=5184 batch=1 mask=none\n");
    print_tensor_info("CPU  ", "dbg_block_15_q_flash", cpu_run.infos["dbg_block_15_q_flash"]);
    print_tensor_info("CPU  ", "dbg_block_15_k_flash", cpu_run.infos["dbg_block_15_k_flash"]);
    print_tensor_info("CPU  ", "dbg_block_15_v_heads_base", cpu_run.infos["dbg_block_15_v_heads_base"]);
    print_tensor_info("CPU  ", "dbg_block_15_v_flash", cpu_run.infos["dbg_block_15_v_flash"]);
    print_tensor_info("Metal", "dbg_block_15_q_flash", metal_run.infos["dbg_block_15_q_flash"]);
    print_tensor_info("Metal", "dbg_block_15_k_flash", metal_run.infos["dbg_block_15_k_flash"]);
    print_tensor_info("Metal", "dbg_block_15_v_heads_base", metal_run.infos["dbg_block_15_v_heads_base"]);
    print_tensor_info("Metal", "dbg_block_15_v_flash", metal_run.infos["dbg_block_15_v_flash"]);

    fprintf(stderr, "\n=== CPU vs Metal Attention Prep Checkpoints ===\n");
    fprintf(stderr, "%-24s %-18s %14s %14s %12s\n",
            "checkpoint", "shape", "max_abs_diff", "mean_abs_diff", "n_bad");
    for (const auto & name : names) {
        auto cpu = load_ref_f32(cpu_dir + "/" + name);
        auto metal = load_ref_f32(metal_dir + "/" + name);
        if (cpu.data.empty() || metal.data.empty() || cpu.shape != metal.shape) {
            fprintf(stderr, "%-24s %-18s %14s %14s %12s\n",
                    name.c_str(), "-", "load-fail", "load-fail", "-");
            continue;
        }
        compare_result diff = compare_tensors(cpu.data.data(), metal.data.data(), cpu.numel(), 1e-4f);
        fprintf(stderr, "%-24s %-18s %14.6f %14.6f %12d\n",
                name.c_str(), format_shape(cpu.shape).c_str(), diff.max_diff, diff.mean_diff, diff.n_bad);
    }

    tensor_case tc;
    if (!load_tensor_case(cpu_dir, tc)) {
        fprintf(stderr, "failed to load isolated attention tensors from %s\n", cpu_dir.c_str());
        return 1;
    }

    {
        sam3_params params;
        params.model_path = model_path;
        params.use_gpu = false;
        params.n_threads = 8;
        auto model = sam3_load_model(params);
        if (!model) {
            fprintf(stderr, "failed to reload model for freqs_cis dump\n");
            return 1;
        }
        const std::string freqs_path = "/tmp/sam3_block15_flash_cpu/vit.blocks.15.attn.freqs_cis";
        if (!sam3_dump_model_tensor(*model, "vit.blocks.15.attn.freqs_cis", freqs_path)) {
            fprintf(stderr, "failed to dump vit.blocks.15.attn.freqs_cis\n");
            sam3_free_model(*model);
            model.reset();
            return 1;
        }
        tc.freqs_cis = load_ref_f32(freqs_path);
        sam3_free_model(*model);
        model.reset();
        if (tc.freqs_cis.data.empty()) {
            fprintf(stderr, "failed to load dumped freqs_cis\n");
            return 1;
        }
    }

    ggml_backend_t cpu_backend = create_backend(false, 8);
    ggml_backend_t metal_backend = create_backend(true, 8);
    if (!cpu_backend || !metal_backend) {
        fprintf(stderr, "failed to initialize CPU or Metal backend\n");
        if (cpu_backend) ggml_backend_free(cpu_backend);
        if (metal_backend) ggml_backend_free(metal_backend);
        return 1;
    }

    fprintf(stderr, "\n=== Isolated QKV Preparation From Exact qkv_proj ===\n");
    prep_run_result cpu_prep = run_isolated_qkv_prep(cpu_backend, tc);
    prep_run_result metal_prep = run_isolated_qkv_prep(metal_backend, tc);
    const char * prep_names[] = {
        "q_split",
        "k_split",
        "v_split",
        "q_heads_base",
        "k_heads_base",
        "v_heads_base",
        "q_heads",
        "k_heads",
        "v_flash",
    };
    fprintf(stderr, "%-16s %-18s %14s %14s %12s %14s\n",
            "stage", "shape", "cpu_vs_dump", "metal_vs_cpu", "n_bad", "mean_diff");
    for (const char * stage : prep_names) {
        const auto cpu_it = cpu_prep.tensors.find(stage);
        const auto metal_it = metal_prep.tensors.find(stage);
        if (cpu_it == cpu_prep.tensors.end() || metal_it == metal_prep.tensors.end()) {
            fprintf(stderr, "%-16s %-18s %14s %14s %12s %14s\n",
                    stage, "-", "missing", "missing", "-", "-");
            continue;
        }

        const std::string dump_name = std::string("dbg_block_15_") + stage;
        auto cpu_dump = load_ref_f32(cpu_dir + "/" + dump_name);
        compare_result cpu_vs_dump_stage = compare_tensors(cpu_it->second.data.data(), cpu_dump.data.data(), cpu_it->second.numel(), 1e-4f);
        compare_result metal_vs_cpu_stage = compare_tensors(metal_it->second.data.data(), cpu_it->second.data.data(), cpu_it->second.numel(), 1e-4f);
        fprintf(stderr, "%-16s %-18s %14.6f %14.6f %12d %14.6f\n",
                stage,
                format_shape(cpu_it->second.shape).c_str(),
                cpu_vs_dump_stage.max_diff,
                metal_vs_cpu_stage.max_diff,
                metal_vs_cpu_stage.n_bad,
                metal_vs_cpu_stage.mean_diff);
    }

    fprintf(stderr, "\n=== Isolated RoPE On Exact q_heads / k_heads ===\n");
    isolated_run_result cpu_q_rope = run_isolated_rope(cpu_backend, tc.q_heads, tc.freqs_cis);
    isolated_run_result metal_q_rope = run_isolated_rope(metal_backend, tc.q_heads, tc.freqs_cis);
    isolated_run_result cpu_k_rope = run_isolated_rope(cpu_backend, tc.k_heads, tc.freqs_cis);
    isolated_run_result metal_k_rope = run_isolated_rope(metal_backend, tc.k_heads, tc.freqs_cis);
    compare_result q_rope_cpu_vs_dump = compare_tensors(cpu_q_rope.output.data(), tc.q_rope.data.data(), (int) tc.q_rope.data.size(), 1e-4f);
    compare_result q_rope_metal_vs_cpu = compare_tensors(metal_q_rope.output.data(), cpu_q_rope.output.data(), (int) cpu_q_rope.output.size(), 1e-4f);
    compare_result k_rope_cpu_vs_dump = compare_tensors(cpu_k_rope.output.data(), tc.k_rope.data.data(), (int) tc.k_rope.data.size(), 1e-4f);
    compare_result k_rope_metal_vs_cpu = compare_tensors(metal_k_rope.output.data(), cpu_k_rope.output.data(), (int) cpu_k_rope.output.size(), 1e-4f);
    fprintf(stderr,
            "q_rope cpu_vs_dump(max=%.6f mean=%.6f) metal_vs_cpu(max=%.6f mean=%.6f n_bad=%d)\n",
            q_rope_cpu_vs_dump.max_diff,
            q_rope_cpu_vs_dump.mean_diff,
            q_rope_metal_vs_cpu.max_diff,
            q_rope_metal_vs_cpu.mean_diff,
            q_rope_metal_vs_cpu.n_bad);
    fprintf(stderr,
            "k_rope cpu_vs_dump(max=%.6f mean=%.6f) metal_vs_cpu(max=%.6f mean=%.6f n_bad=%d)\n",
            k_rope_cpu_vs_dump.max_diff,
            k_rope_cpu_vs_dump.mean_diff,
            k_rope_metal_vs_cpu.max_diff,
            k_rope_metal_vs_cpu.mean_diff,
            k_rope_metal_vs_cpu.n_bad);

    fprintf(stderr, "\n=== Isolated FLASH_ATTN_EXT (Exact V View) ===\n");
    isolated_run_result cpu_exact = run_isolated_attention(cpu_backend, tc, false);
    isolated_run_result metal_exact = run_isolated_attention(metal_backend, tc, false);
    if (cpu_exact.output.empty() || metal_exact.output.empty()) {
        fprintf(stderr, "isolated exact-view attention did not produce outputs\n");
        ggml_backend_free(cpu_backend);
        ggml_backend_free(metal_backend);
        return 1;
    }
    compare_result cpu_vs_dump = compare_tensors(cpu_exact.output.data(), tc.attn_out.data.data(), (int) tc.attn_out.data.size(), 1e-4f);
    compare_result metal_vs_cpu = compare_tensors(metal_exact.output.data(), cpu_exact.output.data(), (int) cpu_exact.output.size(), 1e-4f);
    fprintf(stderr,
            "exact_view shape=%s cpu_ms=%.3f metal_ms=%.3f cpu_vs_dump(max=%.6f mean=%.6f) metal_vs_cpu(max=%.6f mean=%.6f n_bad=%d)\n",
            format_shape(cpu_exact.shape).c_str(),
            cpu_exact.elapsed_ms,
            metal_exact.elapsed_ms,
            cpu_vs_dump.max_diff,
            cpu_vs_dump.mean_diff,
            metal_vs_cpu.max_diff,
            metal_vs_cpu.mean_diff,
            metal_vs_cpu.n_bad);

    fprintf(stderr, "\n=== Isolated FLASH_ATTN_EXT (Forced Contiguous V) ===\n");
    isolated_run_result cpu_cont = run_isolated_attention(cpu_backend, tc, true);
    isolated_run_result metal_cont = run_isolated_attention(metal_backend, tc, true);
    if (cpu_cont.output.empty() || metal_cont.output.empty()) {
        fprintf(stderr, "isolated contiguous-V attention did not produce outputs\n");
        ggml_backend_free(cpu_backend);
        ggml_backend_free(metal_backend);
        return 1;
    }
    compare_result cont_cpu_vs_exact = compare_tensors(cpu_cont.output.data(), cpu_exact.output.data(), (int) cpu_exact.output.size(), 1e-4f);
    compare_result cont_metal_vs_cpu = compare_tensors(metal_cont.output.data(), cpu_cont.output.data(), (int) cpu_cont.output.size(), 1e-4f);
    fprintf(stderr,
            "contiguous_v shape=%s cpu_ms=%.3f metal_ms=%.3f cpu_cont_vs_exact(max=%.6f mean=%.6f) metal_vs_cpu(max=%.6f mean=%.6f n_bad=%d)\n",
            format_shape(cpu_cont.shape).c_str(),
            cpu_cont.elapsed_ms,
            metal_cont.elapsed_ms,
            cont_cpu_vs_exact.max_diff,
            cont_cpu_vs_exact.mean_diff,
            cont_metal_vs_cpu.max_diff,
            cont_metal_vs_cpu.mean_diff,
            cont_metal_vs_cpu.n_bad);

    ggml_backend_free(cpu_backend);
    ggml_backend_free(metal_backend);
    return 0;
}
