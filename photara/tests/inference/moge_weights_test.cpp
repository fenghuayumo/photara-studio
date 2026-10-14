// Manual checkpoint verification. No weights are bundled or downloaded by tests.
#include "models/moge/weights.hpp"
#include "runtime/ggml_runtime.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace photara::inference;
namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: moge_weights_test model.gguf reference.txt [--vulkan]\n";
            return 1;
        }
        const auto model = inspect_moge_model(argv[1]);
        RuntimeOptions options;
        options.backend = argc > 3 && std::string(argv[3]) == "--vulkan"
            ? Backend::vulkan : Backend::cpu;
        options.allow_cpu_fallback = false;
        detail::GgmlRuntime runtime(options);
        bool limited = false;
        try { detail::MogeWeights too_large(argv[1], runtime.backend(), 1); }
        catch (const std::runtime_error&) { limited = true; }
        check(limited, "Weight byte limit was ignored");
        detail::MogeWeights weights(argv[1], runtime.backend());
        check(weights.info().tensor_count == model.tensor_count, "Inspection/load mismatch");
        std::ifstream reference(argv[2]);
        check(reference.is_open(), "Cannot open source checkpoint reference");
        std::string name;
        std::size_t samples = 0;
        while (reference >> name) {
            std::size_t index;
            float expected;
            check(bool(reference >> index >> expected), "Malformed reference sample");
            auto* tensor = weights.tensor(name);
            check(index < static_cast<std::size_t>(ggml_nelements(tensor)), "Sample out of bounds");
            float actual;
            if (tensor->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(tensor, &actual, index * sizeof(float), sizeof(float));
            } else {
                ggml_fp16_t value;
                ggml_backend_tensor_get(tensor, &value, index * sizeof(value), sizeof(value));
                actual = ggml_fp16_to_fp32(value);
            }
            check(std::isfinite(actual) && actual == expected, "Weight roundtrip mismatch");
            ++samples;
        }
        check(reference.eof() && samples >= model.tensor_count,
            "Reference must sample every tensor");
        std::cout << "MoGe " << model.tensor_count << " tensors, " << model.tensor_bytes
            << " bytes, " << samples << " samples verified on "
            << backend_name(runtime.device().backend) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
