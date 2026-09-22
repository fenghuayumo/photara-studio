#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct options {
    std::string backend = "metal";
    std::string variant = "first";
    int warmup = 1;
    int repeats = 3;
    int n_threads = 8;
    int max_wall_ms = 0;
};

struct variant_desc {
    const char * name;
    int64_t in_w;
    int64_t in_h;
    int64_t in_c;
    int64_t out_c;
};

static const variant_desc VARIANT_FIRST  = { "first",  72,  72, 1024, 512 };
static const variant_desc VARIANT_SECOND = { "second", 144, 144,  512, 256 };

static void usage(const char * argv0) {
    fprintf(stderr,
            "Usage: %s [--backend metal|cpu] [--variant first|second] [--warmup N] [--repeats N] [--max-wall-ms N]\n",
            argv0);
}

static bool parse_args(int argc, char ** argv, options & opts) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--backend") == 0) {
            if (i + 1 >= argc) return false;
            opts.backend = argv[++i];
        } else if (strcmp(argv[i], "--variant") == 0) {
            if (i + 1 >= argc) return false;
            opts.variant = argv[++i];
        } else if (strcmp(argv[i], "--warmup") == 0) {
            if (i + 1 >= argc) return false;
            opts.warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeats") == 0) {
            if (i + 1 >= argc) return false;
            opts.repeats = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-threads") == 0) {
            if (i + 1 >= argc) return false;
            opts.n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-wall-ms") == 0) {
            if (i + 1 >= argc) return false;
            opts.max_wall_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            return false;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return false;
        }
    }

    if (opts.backend != "metal" && opts.backend != "cpu") {
        return false;
    }

    if (opts.variant != "first" && opts.variant != "second") {
        return false;
    }

    return true;
}

static const variant_desc & get_variant(const std::string & name) {
    return name == "second" ? VARIANT_SECOND : VARIANT_FIRST;
}

static ggml_backend_t create_backend(const options & opts) {
#ifdef GGML_USE_METAL
    if (opts.backend == "metal") {
        return ggml_backend_metal_init();
    }
#endif
    return ggml_backend_cpu_init();
}

int main(int argc, char ** argv) {
    options opts;
    if (!parse_args(argc, argv, opts)) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_t backend = create_backend(opts);
    if (!backend) {
        fprintf(stderr, "Failed to create backend '%s'\n", opts.backend.c_str());
        return 2;
    }

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, opts.n_threads);
    }

    const variant_desc & v = get_variant(opts.variant);
    const int64_t kw = 2;
    const int64_t kh = 2;
    const int stride = 2;

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
        return 3;
    }

    ggml_tensor * weight = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kw, kh, v.out_c, v.in_c);
    ggml_tensor * input  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, v.in_w, v.in_h, v.in_c, 1);
    ggml_set_name(weight, "weight");
    ggml_set_name(input,  "input");
    ggml_set_input(weight);
    ggml_set_input(input);

    ggml_tensor * output = ggml_conv_transpose_2d_p0(ctx, weight, input, stride);
    ggml_set_name(output, "output");
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, graph) || !ggml_gallocr_alloc_graph(galloc, graph)) {
        fprintf(stderr, "Failed to allocate graph\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 4;
    }

    std::vector<uint16_t> wdata((size_t) ggml_nelements(weight));
    for (size_t i = 0; i < wdata.size(); ++i) {
        const int32_t centered = (int32_t) (i % 17) - 8;
        wdata[i] = ggml_fp32_to_fp16((float) centered * 0.015625f);
    }
    std::vector<float> xdata((size_t) ggml_nelements(input));
    for (size_t i = 0; i < xdata.size(); ++i) {
        const int32_t centered = (int32_t) (i % 23) - 11;
        xdata[i] = (float) centered * 0.0078125f;
    }

    ggml_backend_tensor_set(weight, wdata.data(), 0, wdata.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(input,  xdata.data(), 0, xdata.size() * sizeof(float));

    for (int i = 0; i < opts.warmup; ++i) {
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "Warmup failed: %s\n", ggml_status_to_string(status));
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 5;
        }
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < opts.repeats; ++i) {
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "Run %d failed: %s\n", i + 1, ggml_status_to_string(status));
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 6;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> out((size_t) std::min<int64_t>(ggml_nelements(output), 64));
    ggml_backend_tensor_get(output, out.data(), 0, out.size() * sizeof(float));
    double checksum = 0.0;
    for (float vout : out) {
        checksum += vout;
    }

    const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / opts.repeats;

    fprintf(stderr,
            "RESULT backend=%s variant=%s in=[%lld,%lld,%lld] out_c=%lld out=[%lld,%lld,%lld] avg_ms=%.3f checksum=%.6f repeats=%d\n",
            opts.backend.c_str(),
            v.name,
            (long long) v.in_w,
            (long long) v.in_h,
            (long long) v.in_c,
            (long long) v.out_c,
            (long long) output->ne[0],
            (long long) output->ne[1],
            (long long) output->ne[2],
            avg_ms,
            checksum,
            opts.repeats);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
