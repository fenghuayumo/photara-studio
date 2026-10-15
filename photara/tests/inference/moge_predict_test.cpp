#include "inference/models/moge.hpp"
#include "models/moge/preprocess.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace photara::inference;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void invalid(F action) {
    try { action(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected invalid_argument");
}
template<class T> void save(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    check(static_cast<bool>(output), "Cannot write prediction output");
}
void contracts() {
    invalid([] { (void)detail::preprocess_moge({}, 1200); });
    photara::io::RgbImage image{35, 20, std::vector<std::uint8_t>(35 * 20 * 3, 255)};
    invalid([&] { (void)detail::preprocess_moge(image, 0); });
    invalid([&] { (void)detail::preprocess_moge(image, 4097); });
    auto result = detail::preprocess_moge(image, 7);
    // sqrt(7*1.75)=3.5 must round to even (4).
    check(result.rows == 2 && result.columns == 4, "Token grid rounding");
    constexpr float mean[]{0.485F, 0.456F, 0.406F}, deviation[]{0.229F, 0.224F, 0.225F};
    const auto plane = result.normalized_rgb.size() / 3;
    for (std::size_t i = 0; i < result.normalized_rgb.size(); ++i)
        check(std::abs(result.normalized_rgb[i] - (1.F - mean[i / plane]) / deviation[i / plane]) < 2e-6F,
            "Antialias resize changed constant image");
    image.pixels.pop_back();
    invalid([&] { (void)detail::preprocess_moge(image, 1200); });
    image = {1, 10000, std::vector<std::uint8_t>(30000)};
    invalid([&] { (void)detail::preprocess_moge(image, 1); });
    MogeOptions options;
    options.num_tokens = 0;
    invalid([&] { (void)create_moge_predictor("missing.gguf", options); });
    options.num_tokens = 1200;
    options.mask_threshold = std::numeric_limits<float>::quiet_NaN();
    invalid([&] { (void)create_moge_predictor("missing.gguf", options); });
    options = {};
    options.runtime.allow_cpu_fallback = true;
    invalid([&] { (void)create_moge_predictor("missing.gguf", options); });
    options = {};
    options.runtime.backend = Backend::cpu;
    invalid([&] { (void)create_moge_predictor("missing.gguf", options); });
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--contracts") {
            contracts();
            std::cout << "MoGe preprocessing contracts passed\n";
            return 0;
        }
        if (argc < 8) {
            std::cerr << "Usage: moge_predict_test model.gguf rgb.bin width height tokens output-dir vulkan [--focal pixels] [--preprocess-only] [--lifecycle]\n";
            return 1;
        }
        photara::io::RgbImage image;
        image.width = static_cast<std::uint32_t>(std::stoul(argv[3]));
        image.height = static_cast<std::uint32_t>(std::stoul(argv[4]));
        const auto count = static_cast<std::uint64_t>(image.width) * image.height;
        check(count > 0 && count <= 100'000'000, "Invalid diagnostic image size");
        image.pixels.resize(static_cast<std::size_t>(count * 3));
        std::ifstream input(argv[2], std::ios::binary);
        input.read(reinterpret_cast<char*>(image.pixels.data()), image.pixels.size());
        check(input && input.peek() == std::char_traits<char>::eof(), "Invalid RGB file");
        const std::filesystem::path output(argv[6]);
        std::filesystem::create_directories(output);
        MogeOptions options;
        options.num_tokens = static_cast<std::uint32_t>(std::stoul(argv[5]));
        const std::string backend(argv[7]);
        check(backend == "cpu" || backend == "vulkan", "Unknown backend");
        options.runtime.backend = backend == "vulkan" ? Backend::vulkan : Backend::cpu;
        options.runtime.allow_cpu_fallback = false;
        options.runtime.cpu_threads = 8;
        bool preprocess_only = false, lifecycle = false;
        GeometryInput geometry{image, {}};
        for (int i = 8; i < argc; ++i) {
            const std::string option(argv[i]);
            if (option == "--preprocess-only") preprocess_only = true;
            else if (option == "--lifecycle") lifecycle = true;
            else if (option == "--focal" && i + 1 < argc) {
                const float focal = std::stof(argv[++i]);
                geometry.intrinsics = PinholeIntrinsics{focal, focal, (image.width - 1) / 2.F, (image.height - 1) / 2.F};
            } else throw std::invalid_argument("Unknown prediction test option");
        }
        const auto preprocessed = detail::preprocess_moge(image, options.num_tokens);
        save(output / "normalized.bin", preprocessed.normalized_rgb);
        { std::ofstream grid(output / "grid.txt"); grid << preprocessed.rows << ' ' << preprocessed.columns; }
        if (preprocess_only) return 0;
        auto predictor = create_moge_predictor(argv[1], options);
        check(predictor->device_info().backend == options.runtime.backend, "Unexpected CPU fallback");
        if (lifecycle) {
            for (const float stop_at : {0.F, 0.1F, 0.5F, 0.9F}) {
                std::stop_source cancellation;
                PredictOptions cancelled;
                cancelled.stop = cancellation.get_token();
                cancelled.progress = [&](float value) { if (value >= stop_at) cancellation.request_stop(); };
                if (stop_at == 0) cancellation.request_stop();
                bool caught = false;
                try { (void)predictor->predict(geometry, cancelled); }
                catch (const std::runtime_error& error) { caught = std::string(error.what()) == "MoGe prediction cancelled"; }
                check(caught, "Cancellation did not stop prediction");
            }
            auto malformed = geometry;
            malformed.intrinsics = PinholeIntrinsics{0, 1, 0, 0};
            invalid([&] { (void)predictor->predict(malformed); });
        }
        std::vector<float> progress;
        PredictOptions prediction;
        prediction.progress = [&](float value) { progress.push_back(value); };
        auto prior = predictor->predict(geometry, prediction);
        if (lifecycle) {
            const auto repeated = predictor->predict(geometry);
            check(prior.depth == repeated.depth && prior.normals == repeated.normals &&
                prior.points == repeated.points && prior.valid == repeated.valid, "Repeated prediction changed outputs");
        }
        check(progress == std::vector<float>({0, 0.1F, 0.5F, 0.9F, 1}), "Prediction progress sequence");
        prior.validate();
        check(prior.width == image.width && prior.height == image.height, "Output resolution changed");
        check(prior.scale == GeometryScale::meters && prior.depth_representation == DepthRepresentation::camera_z &&
            prior.normal_source == NormalSource::predicted && prior.confidence.empty(), "Invalid prediction semantics");
        check(prior.model.id == predictor->model_info().id && prior.device.backend == predictor->device_info().backend,
            "Prediction metadata mismatch");
        save(output / "depth.bin", prior.depth);
        save(output / "points.bin", prior.points);
        save(output / "normals.bin", prior.normals);
        save(output / "valid.bin", prior.valid);
        const auto& k = prior.intrinsics.value();
        std::ofstream metadata(output / "intrinsics.txt");
        metadata << std::setprecision(17) << k.fx << ' ' << k.fy << ' ' << k.cx << ' ' << k.cy;
        check(static_cast<bool>(metadata), "Cannot write intrinsics");
        std::cout << "MoGe prediction " << image.width << 'x' << image.height << " grid "
            << preprocessed.rows << 'x' << preprocessed.columns << " on " << backend_name(prior.device.backend) << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
