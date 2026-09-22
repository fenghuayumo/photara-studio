#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct options {
    std::string case_name = "all";
    bool dump_mismatch = false;
    int n_threads = 8;
    int max_wall_ms = 30000;
};

struct case_desc {
    const char * name;
    int64_t in_w;
    int64_t in_h;
    int64_t in_c;
    int64_t out_c;
    int64_t kw;
    int64_t kh;
    int stride;
    uint32_t seed;
    float tol;
};

struct run_result {
    std::vector<float> output;
    int64_t out_w = 0;
    int64_t out_h = 0;
    int64_t out_c = 0;
};

static const case_desc CASES[] = {
    { "cpu_ref_stride1", 3,   2,    2,   3, 2, 2, 1, 0x1234u, 1e-5f },
    { "cpu_ref_stride2", 3,   2,    2,   3, 2, 2, 2, 0x2345u, 1e-5f },
    { "cpu_ref_stride3", 3,   2,    2,   3, 2, 2, 3, 0x3456u, 1e-5f },
    { "small_rect",      5,   4,    3,   4, 3, 2, 1, 0x4567u, 1e-5f },
    { "medium_stride2",  8,   7,    5,   6, 3, 3, 2, 0x5678u, 1e-4f },
    { "sam_first",      72,  72, 1024, 512, 2, 2, 2, 0x6789u, 2e-3f },
    { "sam_second",    144, 144,  512, 256, 2, 2, 2, 0x789au, 2e-3f },
};

static void usage(const char * argv0) {
    fprintf(stderr,
            "Usage: %s [--case NAME|all] [--dump-mismatch] [--n-threads N] [--max-wall-ms N] [--list]\n",
            argv0);
}

static bool parse_args(int argc, char ** argv, options & opts) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--case") == 0) {
            if (i + 1 >= argc) return false;
            opts.case_name = argv[++i];
        } else if (strcmp(argv[i], "--dump-mismatch") == 0) {
            opts.dump_mismatch = true;
        } else if (strcmp(argv[i], "--n-threads") == 0) {
            if (i + 1 >= argc) return false;
            opts.n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-wall-ms") == 0) {
            if (i + 1 >= argc) return false;
            opts.max_wall_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--list") == 0) {
            for (const case_desc & tc : CASES) {
                fprintf(stderr, "%s\n", tc.name);
            }
            exit(0);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            return false;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

static float gen_value(uint64_t idx, uint32_t seed, float scale) {
    uint32_t x = (uint32_t) idx ^ seed;
    x = x * 1664525u + 1013904223u;
    x ^= x >> 16;
    const int32_t centered = (int32_t) (x % 511u) - 255;
    return centered * scale;
}

static void fill_input_data(const case_desc & tc, std::vector<float> & input) {
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = gen_value(i, tc.seed ^ 0x13579bdu, 1.0f / 256.0f);
    }
}

static void fill_weight_data(const case_desc & tc, std::vector<uint16_t> & weight_f16, std::vector<float> & weight_f32) {
    for (size_t i = 0; i < weight_f16.size(); ++i) {
        const float v = gen_value(i, tc.seed ^ 0x2468aceu, 1.0f / 128.0f);
        weight_f32[i] = v;
        weight_f16[i] = ggml_fp32_to_fp16(v);
    }
}

static ggml_backend_t create_backend(const std::string & name, int n_threads) {
    if (name == "cpu") {
        ggml_backend_t backend = ggml_backend_cpu_init();
        if (backend) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }
        return backend;
    }

#ifdef GGML_USE_METAL
    if (name == "metal") {
        return ggml_backend_metal_init();
    }
#endif

    return nullptr;
}

static run_result run_case(const case_desc & tc,
                           const std::string & backend_name,
                           int n_threads,
                           const std::vector<uint16_t> & weight_data,
                           const std::vector<float> & input_data) {
    ggml_backend_t backend = create_backend(backend_name, n_threads);
    if (!backend) {
        fprintf(stderr, "Failed to create backend '%s'\n", backend_name.c_str());
        exit(2);
    }

    const size_t ctx_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Failed to create ggml context\n");
        ggml_backend_free(backend);
        exit(3);
    }

    ggml_tensor * weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.kw, tc.kh, tc.out_c, tc.in_c);
    ggml_tensor * input  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.in_w, tc.in_h, tc.in_c, 1);
    ggml_set_name(weight, "weight");
    ggml_set_name(input, "input");
    ggml_set_input(weight);
    ggml_set_input(input);

    ggml_tensor * output = ggml_conv_transpose_2d_p0(ctx, weight, input, tc.stride);
    ggml_set_name(output, "output");
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "Failed to allocate graph for case %s (%s)\n", tc.name, backend_name.c_str());
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        ggml_backend_free(backend);
        exit(4);
    }

    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(input,  input_data.data(), 0, input_data.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Graph compute failed for case %s on %s: %s\n",
                tc.name, backend_name.c_str(), ggml_status_to_string(status));
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        ggml_backend_free(backend);
        exit(5);
    }

    run_result res;
    res.out_w = output->ne[0];
    res.out_h = output->ne[1];
    res.out_c = output->ne[2];
    res.output.resize((size_t) ggml_nelements(output));
    ggml_backend_tensor_get(output, res.output.data(), 0, res.output.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return res;
}

static void dump_reference_mismatch(const case_desc & tc,
                                    const std::vector<float> & input_data,
                                    const std::vector<float> & weight_data,
                                    const run_result & cpu,
                                    const run_result & metal,
                                    int worst_index) {
    if (worst_index < 0) {
        return;
    }

    const int64_t ow = worst_index % cpu.out_w;
    const int64_t rem0 = worst_index / cpu.out_w;
    const int64_t oh = rem0 % cpu.out_h;
    const int64_t oc = rem0 / cpu.out_h;

    fprintf(stderr,
            "  mismatch case=%s coord=(ow=%lld, oh=%lld, oc=%lld) cpu=%.8f metal=%.8f\n",
            tc.name,
            (long long) ow,
            (long long) oh,
            (long long) oc,
            cpu.output[worst_index],
            metal.output[worst_index]);

    int shown = 0;
    double recomputed = 0.0;
    for (int64_t kh = 0; kh < tc.kh; ++kh) {
        const int64_t in_y_nom = oh - kh;
        if (in_y_nom < 0 || in_y_nom % tc.stride != 0) {
            continue;
        }
        const int64_t in_y = in_y_nom / tc.stride;
        if (in_y >= tc.in_h) {
            continue;
        }

        for (int64_t kw = 0; kw < tc.kw; ++kw) {
            const int64_t in_x_nom = ow - kw;
            if (in_x_nom < 0 || in_x_nom % tc.stride != 0) {
                continue;
            }
            const int64_t in_x = in_x_nom / tc.stride;
            if (in_x >= tc.in_w) {
                continue;
            }

            for (int64_t ic = 0; ic < tc.in_c; ++ic) {
                const size_t input_idx = (size_t) ic * tc.in_h * tc.in_w + (size_t) in_y * tc.in_w + (size_t) in_x;
                const size_t weight_idx = ((size_t) ic * tc.out_c * tc.kh * tc.kw) +
                                          ((size_t) oc * tc.kh * tc.kw) +
                                          (size_t) kh * tc.kw +
                                          (size_t) kw;
                const float contrib = input_data[input_idx] * weight_data[weight_idx];
                recomputed += contrib;
                if (shown < 16) {
                    fprintf(stderr,
                            "    term[%2d] in=(x=%lld,y=%lld,c=%lld)=%.8f weight=(kw=%lld,kh=%lld,oc=%lld,ic=%lld)=%.8f contrib=%.8f\n",
                            shown,
                            (long long) in_x,
                            (long long) in_y,
                            (long long) ic,
                            input_data[input_idx],
                            (long long) kw,
                            (long long) kh,
                            (long long) oc,
                            (long long) ic,
                            weight_data[weight_idx],
                            contrib);
                }
                ++shown;
            }
        }
    }

    fprintf(stderr, "  recomputed_reference=%.8f from %d contributing terms\n", recomputed, shown);
}

int main(int argc, char ** argv) {
    options opts;
    if (!parse_args(argc, argv, opts)) {
        usage(argv[0]);
        return 1;
    }

#ifndef GGML_USE_METAL
    fprintf(stderr, "Metal not available, skipping test\n");
    return 0;
#else
    bool all_pass = true;

    fprintf(stderr, "| case | output | max abs diff | mean abs diff | tolerance | status |\n");
    fprintf(stderr, "| --- | --- | ---: | ---: | ---: | --- |\n");

    for (const case_desc & tc : CASES) {
        if (opts.case_name != "all" && opts.case_name != tc.name) {
            continue;
        }

        std::vector<uint16_t> weight_f16((size_t) tc.kw * tc.kh * tc.out_c * tc.in_c);
        std::vector<float>    weight_f32(weight_f16.size());
        std::vector<float>    input_data((size_t) tc.in_w * tc.in_h * tc.in_c);

        fill_weight_data(tc, weight_f16, weight_f32);
        fill_input_data(tc, input_data);

        run_result cpu = run_case(tc, "cpu", opts.n_threads, weight_f16, input_data);
        run_result metal = run_case(tc, "metal", opts.n_threads, weight_f16, input_data);

        compare_result diff = compare_tensors(cpu.output.data(), metal.output.data(), (int) cpu.output.size(), tc.tol);
        const bool pass = diff.max_diff <= tc.tol;
        fprintf(stderr,
                "| %s | [%lld,%lld,%lld] | %.8f | %.8f | %.1e | %s |\n",
                tc.name,
                (long long) cpu.out_w,
                (long long) cpu.out_h,
                (long long) cpu.out_c,
                diff.max_diff,
                diff.mean_diff,
                tc.tol,
                pass ? "PASS" : "FAIL");

        if (!pass) {
            all_pass = false;
            if (opts.dump_mismatch || (tc.in_w <= 8 && tc.in_h <= 8 && tc.in_c <= 8)) {
                dump_reference_mismatch(tc, input_data, weight_f32, cpu, metal, diff.worst_index);
            }
        }
    }

    fprintf(stderr, "\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
#endif
}
