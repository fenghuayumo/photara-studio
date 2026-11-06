#include "splat/formats.hpp"
#include "splat/trainer.hpp"
#include "sfm/asfm.hpp"
#include "../photara/src/splat/formats_sog_xt.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
    try {
        if (argc != 6 && argc != 7)
            throw std::runtime_error("Usage: sog_xt_field_audit ORIGINAL_PLY ENCODED_SOG_XT SFM "
                                     "IMAGES OUTPUT [LABELS_U32]");
        using namespace photara;
        using namespace photara::splat;
        const std::filesystem::path output(argv[5]);
        std::filesystem::create_directories(output);
        auto original = load_gaussians(argv[1], tinytensor::Device::CPU);
        auto decoded = load_gaussians(argv[2], tinytensor::Device::CPU);
        if (original.size() != decoded.size())
            throw std::runtime_error("Count mismatch");
        const auto n = original.size();
        const unsigned side = unsigned((std::size_t(std::ceil(std::sqrt(double(n)))) + 7) / 8 * 8);
        auto contracted = original.means.to_vector();
        for (float& v : contracted)
            v = std::copysign(std::log1p(std::abs(v)), v);
        const float lo = *std::min_element(contracted.begin(), contracted.end());
        const float hi = *std::max_element(contracted.begin(), contracted.end());
        std::vector<float> features(std::size_t(side) * side * 6);
        for (std::size_t i = 0; i < n; ++i)
            for (unsigned d = 0; d < 3; ++d) {
                const auto code = unsigned(std::llround(
                    std::clamp(
                        (double(contracted[i * 3 + d]) - lo) / std::max(double(hi) - lo, 1e-20),
                        0.0, 1.0) *
                    65535));
                features[i * 6 + d] = float(code >> 8) / 255;
                features[i * 6 + 3 + d] = float(code & 255) / 255 * .1F;
            }
        const auto order = detail::sog_xt_layout(features, side, 6);
        std::vector<unsigned> active;
        for (auto row : order)
            if (row < n)
                active.push_back(row);
        const auto reorder = [&](tinytensor::Tensor& tensor) {
            if (!tensor.is_valid())
                return;
            const auto values = tensor.to_vector();
            const auto stride = values.size() / n;
            std::vector<float> result(values.size());
            for (std::size_t row = 0; row < n; ++row)
                std::copy_n(
                    values.data() + std::size_t(active[row]) * stride, stride,
                    result.data() + row * stride);
            tensor =
                tinytensor::Tensor::from_vector(result, tensor.shape(), tinytensor::Device::CPU);
        };
        for (auto* t :
             {&original.means, &original.log_scales, &original.quaternions,
              &original.opacity_logits, &original.sh, &original.normal_features,
              &original.filter_3d})
            reorder(*t);
        // Verify the reconstructed permutation against every decoded position.
        const auto positions = decoded.means.to_vector();
        for (std::size_t row = 0; row < n; ++row)
            for (unsigned d = 0; d < 3; ++d) {
                const float v = contracted[std::size_t(active[row]) * 3 + d];
                const auto code = unsigned(std::llround(
                    std::clamp((double(v) - lo) / std::max(double(hi) - lo, 1e-20), 0.0, 1.0) *
                    65535));
                const float q = float(double(lo) + (double(hi) - lo) * code / 65535);
                const float expected = std::copysign(std::expm1(std::abs(q)), q);
                if (std::abs(expected - positions[row * 3 + d]) >
                    1e-5F * std::max(1.F, std::abs(expected)))
                    throw std::runtime_error("Permutation reconstruction failed");
            }
        std::vector<float> cluster_sh;
        double cluster_mse = 0, quantization_mse = 0;
        if (argc == 7) {
            std::vector<unsigned> labels(n);
            std::ifstream label_file(argv[6], std::ios::binary);
            label_file.read(reinterpret_cast<char*>(labels.data()), n * sizeof(unsigned));
            if (!label_file || label_file.peek() != std::char_traits<char>::eof())
                throw std::runtime_error("Invalid labels file");
            const auto k = *std::max_element(labels.begin(), labels.end()) + 1;
            const auto stride = original.sh.shape()[1] * 3;
            const auto dims = stride - 3;
            const auto sh = original.sh.to_vector(), decoded_sh = decoded.sh.to_vector();
            std::vector<double> sums(std::size_t(k) * dims);
            std::vector<unsigned> counts(k);
            for (std::size_t i = 0; i < n; ++i) {
                ++counts[labels[i]];
                for (std::size_t d = 0; d < dims; ++d)
                    sums[std::size_t(labels[i]) * dims + d] += sh[i * stride + 3 + d];
            }
            cluster_sh = sh;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t d = 0; d < dims; ++d) {
                    const float center =
                        float(sums[std::size_t(labels[i]) * dims + d] / counts[labels[i]]);
                    cluster_sh[i * stride + 3 + d] = center;
                    cluster_mse += std::pow(double(sh[i * stride + 3 + d]) - center, 2);
                    quantization_mse +=
                        std::pow(double(decoded_sh[i * stride + 3 + d]) - center, 2);
                }
            cluster_mse /= n * dims;
            quantization_mse /= n * dims;
        }
        const auto sfm_scene = sfm::load_asfm(argv[3]);
        mvs::MvsScene scene;
        for (const auto& image : sfm_scene.images)
            if (image.registered) {
                const auto& intr = sfm_scene.camera_of(image);
                mvs::MvsView v;
                v.id = static_cast<mvs::Index>(scene.views.size());
                v.path = std::filesystem::path(argv[4]) / image.path.filename();
                v.pose = image.pose;
                v.source_model = intr.model;
                v.width = v.src_width = intr.width;
                v.height = v.src_height = intr.height;
                v.fx = v.src_fx = intr.fx;
                v.fy = v.src_fy = intr.fy;
                v.cx = v.src_cx = intr.cx;
                v.cy = v.src_cy = intr.cy;
                v.k1 = intr.k1;
                v.k2 = intr.k2;
                v.p1 = intr.p1;
                v.p2 = intr.p2;
                scene.views.push_back(v);
            }
        TrainingOptions options;
        options.filter_mode = original.filter_mode;
        options.max_image_dimension = 960;
        options.use_mask = false;
        nlohmann::ordered_json report{
            {"count", n},
            {"permutation_verified", true},
            {"filter_mode", filter_mode_name(original.filter_mode)},
            {"variants", nlohmann::ordered_json::object()}};
        report["sh_cluster_coefficient_mse"] = cluster_mse;
        report["sh_quantization_coefficient_mse"] = quantization_mse;
        for (const std::string variant :
             {"original", "all", "means", "shape", "opacity", "dc", "sh_rest", "sh_cluster_only",
              "sh_quant_only"}) {
            if (cluster_sh.empty() && (variant == "sh_cluster_only" || variant == "sh_quant_only"))
                continue;
            auto model = original;
            if (variant == "all")
                model = decoded;
            if (variant == "means")
                model.means = decoded.means;
            if (variant == "shape") {
                model.log_scales = decoded.log_scales;
                model.quaternions = decoded.quaternions;
            }
            if (variant == "opacity")
                model.opacity_logits = decoded.opacity_logits;
            if (variant == "dc" || variant == "sh_rest") {
                auto sh = original.sh.to_vector();
                const auto other = decoded.sh.to_vector();
                const auto stride = original.sh.shape()[1] * 3;
                for (std::size_t row = 0; row < n; ++row)
                    for (std::size_t c = 0; c < stride; ++c)
                        if ((variant == "dc") == (c < 3))
                            sh[row * stride + c] = other[row * stride + c];
                model.sh = tinytensor::Tensor::from_vector(
                    sh, original.sh.shape(), tinytensor::Device::CPU);
            }
            if (variant == "sh_cluster_only" || variant == "sh_quant_only") {
                auto sh = cluster_sh;
                if (variant == "sh_quant_only") {
                    const auto src = original.sh.to_vector(), dst = decoded.sh.to_vector();
                    for (std::size_t i = 0; i < sh.size(); ++i)
                        sh[i] = src[i] + dst[i] - cluster_sh[i];
                    const auto stride = original.sh.shape()[1] * 3;
                    for (std::size_t row = 0; row < n; ++row)
                        for (unsigned d = 0; d < 3; ++d)
                            sh[row * stride + d] = src[row * stride + d];
                }
                model.sh = tinytensor::Tensor::from_vector(
                    sh, original.sh.shape(), tinytensor::Device::CPU);
            }
            for (auto* t :
                 {&model.means, &model.log_scales, &model.quaternions, &model.opacity_logits,
                  &model.sh, &model.normal_features, &model.filter_3d})
                if (t->is_valid())
                    *t = t->to(tinytensor::Device::CUDA);
            auto rows = nlohmann::ordered_json::array();
            for (std::size_t v = 0; v < scene.views.size(); ++v) {
                if (v % 32 && v != 112 && v != 368 && v != 559)
                    continue;
                const auto metric = render_evaluation_png(
                    model, scene.views[v], output / (variant + "-" + std::to_string(v) + ".png"),
                    options);
                rows.push_back({{"view", v}, {"psnr", metric.psnr}, {"ssim", metric.ssim}});
            }
            report["variants"][variant] = rows;
            std::cout << "Finished " << variant << std::endl;
        }
        std::ofstream(output / "result.json") << report.dump(2) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
