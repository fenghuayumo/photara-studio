#include "inference/geometry_predictor.hpp"
#include "inference/model_cache.hpp"
#include "runtime/ggml_runtime.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace photara::inference;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void invalid(F action) {
    bool caught = false;
    try { action(); } catch (const std::invalid_argument&) { caught = true; }
    check(caught, "Expected invalid_argument");
}
void geometry_contract() {
    GeometryPrior p;
    p.width = p.input_width = 2;
    p.height = p.input_height = 1;
    p.depth = {2.F, std::numeric_limits<float>::quiet_NaN()};
    p.valid = {1, 0};
    p.validate(); // Nonfinite invalid pixels are intentionally allowed.
    p.valid[1] = 1;
    invalid([&] { p.validate(); });
    p.valid[1] = 0;
    p.valid[0] = 255;
    invalid([&] { p.validate(); });
    p.valid[0] = 1;
    p.depth[0] = -1;
    invalid([&] { p.validate(); });
    p.depth[0] = 2;
    p.normals = {{0, 0, -1}, {0, 0, 0}};
    invalid([&] { p.validate(); });
    p.normal_source = NormalSource::predicted;
    p.validate();
    p.normals[0] = {0, 0, -2};
    invalid([&] { p.validate(); });
    p.normals.clear();
    p.normal_source = NormalSource::absent;
    p.input_pixel_scale[0] = 0;
    invalid([&] { p.validate(); });
    p.input_pixel_scale[0] = 1;
    p.intrinsics = PinholeIntrinsics{0, 10, 0, 0};
    invalid([&] { p.validate(); });
    p.intrinsics.reset();
    p.confidence = {1.1F, 0};
    invalid([&] { p.validate(); });
    p.confidence.clear();
    p.points = {{0, 0, -1}, {0, 0, 0}};
    invalid([&] { p.validate(); });
    p.points[0][2] = 2;
    p.depth.clear();
    p.validate(); // Point-only geometry is supported.
    p.valid.clear();
    invalid([&] { p.validate(); });
}
void cache_contract() {
    check(cached_model_path("test.gguf").filename() == "test.gguf", "Cache filename");
    invalid([] { (void)cached_model_path("../test.gguf"); });
    invalid([] { (void)cached_model_path(""); });
    invalid([] { (void)cached_model_path(".."); });
    const auto file = std::filesystem::temp_directory_path() /
        ("photara-inference-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code error; std::filesystem::remove(path, error); }
    } cleanup{file};
    { std::ofstream output(file, std::ios::binary); output << "abc"; }
    check(model_file_ready(file, 3), "Complete file rejected");
    check(!model_file_ready(file, 4), "Truncated file accepted");
    check(!model_file_ready(file.parent_path()), "Directory accepted as model");
}
void arithmetic(const RuntimeOptions& options) {
    detail::GgmlRuntime runtime(options);
    if (!options.allow_cpu_fallback)
        check(runtime.device().backend == options.backend, "Unexpected fallback");
    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    auto* context = ggml_init(params);
    check(context != nullptr, "Cannot create test context");
    struct Cleanup {
        ggml_context* context;
        ~Cleanup() { ggml_free(context); }
    } cleanup{context};
    auto* a = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
    auto* b = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
    ggml_set_input(a);
    ggml_set_input(b);
    auto* sum = ggml_add(context, a, b);
    ggml_set_output(sum);
    auto* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, sum);
    runtime.allocate(graph);
    const float av[]{1, -2, 3.5F, 0}, bv[]{2, 5, -1.5F, -4};
    ggml_backend_tensor_set(a, av, 0, sizeof(av));
    ggml_backend_tensor_set(b, bv, 0, sizeof(bv));
    runtime.compute(graph);
    float result[4]{};
    ggml_backend_tensor_get(sum, result, 0, sizeof(result));
    const float expected[]{3, 3, 2, -4};
    for (int i = 0; i < 4; ++i)
        check(std::abs(result[i] - expected[i]) < 1e-6F, "Arithmetic mismatch");
    check(runtime.compute_buffer_bytes() > 0, "Graph memory accounting missing");
    std::cout << "Arithmetic passed on " << backend_name(runtime.device().backend)
        << ": " << runtime.device().name << '\n';
}
}
int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--vulkan") {
            bool found = false;
            for (const auto& device : available_devices())
                found = found || device.backend == Backend::vulkan;
            if (!found) { std::cout << "Vulkan unavailable; skipping\n"; return 77; }
            RuntimeOptions options;
            options.allow_cpu_fallback = false;
            arithmetic(options);
            return 0;
        }
        geometry_contract();
        cache_contract();
        RuntimeOptions cpu;
        cpu.backend = Backend::cpu;
        cpu.allow_cpu_fallback = false;
        cpu.cpu_threads = 1;
        arithmetic(cpu);
        cpu.device_index = 1;
        invalid([&] { detail::GgmlRuntime runtime(cpu); });
        RuntimeOptions fallback;
        fallback.device_index = std::numeric_limits<std::uint32_t>::max();
        detail::GgmlRuntime runtime(fallback);
        check(runtime.device().backend == Backend::cpu, "CPU fallback not reported");
        fallback.allow_cpu_fallback = false;
        bool rejected = false;
        try { detail::GgmlRuntime strict(fallback); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "Strict Vulkan selection silently fell back");
        std::cout << "Inference contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
