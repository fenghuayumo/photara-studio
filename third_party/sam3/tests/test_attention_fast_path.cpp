#include "../sam3.cpp"

#include <stdexcept>

static void check_attention(ggml_backend_t backend, int dim, bool masked) {
    const int nq = 7, nk = 5, heads = 3, batch = 2;
    ggml_init_params params = {1024 * 1024, nullptr, true};
    ggml_context_ptr ctx(ggml_init(params));
    auto* q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, dim, nq, heads, batch);
    auto* k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, dim, nk, heads, batch);
    auto* vb = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, dim, heads, nk, batch);
    auto* v = ggml_permute(ctx.get(), vb, 0, 2, 1, 3);
    ggml_set_input(q);
    ggml_set_input(k);
    ggml_set_input(vb);
    ggml_tensor* mask = nullptr;
    if (masked) {
        mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, nk, nq);
        ggml_set_input(mask);
    }
    auto* reference = sam3_attention(ctx.get(), q, k, v, mask, 0.125f, 0, 0);
    auto* actual = sam3_attention(ctx.get(), q, k, v, mask, 0.125f, 0, 0, backend);
    bool supported = false;
    if (dim == 64 && !masked) {
        auto* probe = ggml_flash_attn_ext(ctx.get(), q, k, v, nullptr, 0.125f, 0, 0);
        ggml_flash_attn_ext_set_prec(probe, GGML_PREC_F32);
        supported = ggml_backend_supports_op(backend, probe);
    }
    if (supported) {
        if (actual->op != GGML_OP_FLASH_ATTN_EXT)
            throw std::runtime_error("expected Flash Attention");
    } else if (actual->op == GGML_OP_FLASH_ATTN_EXT) {
        throw std::runtime_error("expected fallback");
    }
    ggml_set_output(reference);
    ggml_set_output(actual);
    auto* graph = ggml_new_graph_custom(ctx.get(), 256, false);
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, actual);
    ggml_gallocr_ptr alloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    if (!ggml_gallocr_alloc_graph(alloc.get(), graph))
        throw std::runtime_error("graph allocation failed");
    for (auto* input : {q, k, vb}) {
        std::vector<float> values(ggml_nelements(input));
        for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i) * 0.17f);
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    }
    if (mask) {
        std::vector<ggml_fp16_t> values(nk * nq);
        for (int i = 0; i < nk * nq; ++i)
            values[i] = ggml_fp32_to_fp16(i % nk == 0 ? -1000.f : 0.f);
        ggml_backend_tensor_set(mask, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
    }
    if (!sam3_graph_compute(backend, graph, 2)) throw std::runtime_error("compute failed");
    std::vector<float> expected(ggml_nelements(reference)), got(expected.size());
    ggml_backend_tensor_get(reference, expected.data(), 0, expected.size() * sizeof(float));
    ggml_backend_tensor_get(actual, got.data(), 0, got.size() * sizeof(float));
    float error = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) throw std::runtime_error("non-finite output");
        error = std::max(error, std::abs(got[i] - expected[i]));
    }
    printf("%s dim=%d masked=%d max_error=%.8f\n",
           ggml_backend_name(backend), dim, masked, error);
    // Vulkan cooperative-matrix kernels use fp16 operands with fp32 accumulation.
    const float tolerance = ggml_backend_is_cpu(backend) ? 2e-5f : 2e-3f;
    if (error > tolerance) throw std::runtime_error("attention output mismatch");
    printf("%s dim=%d masked=%d max_error=%.8f PASS\n",
           ggml_backend_name(backend), dim, masked, error);
}

int main(int argc, char** argv) {
    ggml_backend_t backend = nullptr;
#ifdef GGML_USE_VULKAN
    if (argc > 1 && std::string(argv[1]) == "vulkan") backend = ggml_backend_vk_init(0);
    else
#endif
        backend = ggml_backend_cpu_init();
    if (!backend) return 1;
    int status = 0;
    try {
        check_attention(backend, 64, false);
        check_attention(backend, 32, false);
        check_attention(backend, 64, true);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        status = 1;
    }
    ggml_backend_free(backend);
    return status;
}
