#pragma once
#include "runtime/ggml_runtime.hpp"
#include <memory>
#include <vector>
namespace photara::inference::detail {
// One shape per predictor: bounded GPU memory, independent encoder/decoder arenas.
struct MogeGraphCache {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, &ggml_free};
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *input = nullptr, *position = nullptr, *features = nullptr, *class_token = nullptr;
    std::vector<ggml_tensor*> outputs;
    std::uint32_t rows = 0, columns = 0;
    float aspect = 0;
    const void* weights = nullptr;
    ggml_backend_t backend = nullptr;
    ~MogeGraphCache() { reset(); }
    void reset() {
        if (allocator) ggml_gallocr_free(allocator);
        allocator = nullptr;
        context.reset(); graph = nullptr;
        input = position = features = class_token = nullptr;
        outputs.clear(); weights = nullptr; backend = nullptr;
    }
};
}
