#include "models/moge/decoder.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace photara::inference;
namespace {
std::vector<float> load(const std::filesystem::path& path, std::size_t count) {
    std::vector<float> data(count);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    if (!file || file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Invalid input file");
    return data;
}
void save(const std::filesystem::path& directory, const std::string& name, const std::vector<float>& values) {
    std::ofstream output(directory / (name + ".bin"), std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    if (!output) throw std::runtime_error("Cannot write decoder stage " + name);
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 8) {
            std::cerr << "Usage: moge_decoder_test model.gguf rows cols aspect features.bin class.bin output-dir [--vulkan] [--encoder-image normalized-chw.bin]\n";
            return 1;
        }
        const auto rows = std::stoul(argv[2]), columns = std::stoul(argv[3]);
        if (!rows || !columns || rows > 4096 || columns > 4096 || rows * columns > 4096)
            throw std::invalid_argument("Invalid decoder grid");
        detail::EncoderResult input;
        input.rows = static_cast<std::uint32_t>(rows);
        input.columns = static_cast<std::uint32_t>(columns);
        input.features = load(argv[5], rows * columns * 768);
        input.class_token = load(argv[6], 768);
        const std::filesystem::path output_directory(argv[7]);
        std::filesystem::create_directories(output_directory);
        RuntimeOptions options;
        options.backend = Backend::cpu;
        std::filesystem::path encoder_image;
        for (int argument = 8; argument < argc; ++argument) {
            const std::string option(argv[argument]);
            if (option == "--vulkan") options.backend = Backend::vulkan;
            else if (option == "--encoder-image" && argument + 1 < argc) encoder_image = argv[++argument];
            else throw std::invalid_argument("Unknown decoder test option");
        }
        options.allow_cpu_fallback = false;
        options.cpu_threads = 8;
        detail::GgmlRuntime runtime(options);
        detail::MogeWeights weights(argv[1], runtime.backend());
        if (!encoder_image.empty()) {
            const auto normalized = load(encoder_image, rows * columns * 3 * 14 * 14);
            input = detail::encode_moge_vitb(weights, runtime, input.rows, input.columns, normalized);
        }
        auto result = detail::decode_moge_vitb(weights, runtime, input, std::stof(argv[4]), true);
        for (const auto& stage : result.stages) save(output_directory, stage.name, stage.values);
        save(output_directory, "points", result.points);
        save(output_directory, "normals", result.normals);
        save(output_directory, "mask_logits", result.mask_logits);
        save(output_directory, "log_metric_scale", {result.log_metric_scale});
        auto repeated = detail::decode_moge_vitb(weights, runtime, input, std::stof(argv[4]), false);
        if (repeated.points != result.points || repeated.normals != result.normals ||
            repeated.mask_logits != result.mask_logits || repeated.log_metric_scale != result.log_metric_scale)
            throw std::runtime_error("Decoder diagnostic/normal path mismatch");
        std::cout << "Decoder " << rows << 'x' << columns << " patches on "
            << backend_name(runtime.device().backend) << ", " << result.stages.size()
            << " stages, compute buffers " << runtime.compute_buffer_bytes() << " bytes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
