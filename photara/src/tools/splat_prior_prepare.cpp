#include "inference/models/moge.hpp"
#include "splat/dataset.hpp"
#include "splat/prior.hpp"
#include "training_data_loader.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
double median(std::vector<double> values) {
    if (values.empty()) throw std::runtime_error("No sparse alignment support");
    const auto middle = values.begin()+values.size()/2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}
struct Alignment { double scale, mad; std::size_t samples; };
Alignment align(const photara::mvs::MvsScene& scene, std::size_t view_index,
    const photara::splat::Camera& camera, const photara::inference::GeometryPrior& prior) {
    std::vector<double> ratios;
    for (const auto& point : scene.sparse_points) {
        if (std::find(point.view_ids.begin(), point.view_ids.end(), view_index) == point.view_ids.end()) continue;
        const auto& m = camera.world_to_camera;
        const auto p = point.position;
        const double x=m[0]*p.x()+m[4]*p.y()+m[8]*p.z()+m[12];
        const double y=m[1]*p.x()+m[5]*p.y()+m[9]*p.z()+m[13];
        const double z=m[2]*p.x()+m[6]*p.y()+m[10]*p.z()+m[14];
        if (!(z > 0) || !std::isfinite(x+y+z)) continue;
        const auto u = std::lround(camera.fx*x/z+camera.cx), v = std::lround(camera.fy*y/z+camera.cy);
        if (u < 0 || v < 0 || u >= camera.width || v >= camera.height) continue;
        const auto pixel = static_cast<std::size_t>(v)*camera.width+u;
        if (!prior.valid[pixel] || !(prior.depth[pixel] > 0)) continue;
        ratios.push_back(std::log(z/prior.depth[pixel]));
    }
    if (ratios.size() < 16) throw std::runtime_error("Need at least 16 observed sparse points to align each prior");
    const double center = median(ratios);
    std::vector<double> deviations;
    for (double value : ratios) deviations.push_back(std::abs(value-center));
    const double mad = median(deviations), threshold = std::max(.1, 3*1.4826*mad);
    std::vector<double> inliers;
    for (double value : ratios) if (std::abs(value-center) <= threshold) inliers.push_back(value);
    if (inliers.size() < 16) throw std::runtime_error("Insufficient robust prior scale support");
    const double scale = std::exp(median(inliers));
    if (!std::isfinite(scale) || scale <= 0) throw std::runtime_error("Invalid prior alignment scale");
    return {scale, mad, inliers.size()};
}
}
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
    try {
        if (argc != 8 && argc != 10) {
            std::cerr << "Usage: photara_splat_prior_prepare dataset images model.gguf output-dir max-resolution tokens eval-split-every [lambda_d lambda_n]\n"
                         "Weights default to zero: no inference or cache creation when both are disabled.\n";
            return 1;
        }
        const float lambda_d = argc == 10 ? std::stof(argv[8]) : 0.F;
        const float lambda_n = argc == 10 ? std::stof(argv[9]) : 0.F;
        if (!std::isfinite(lambda_d) || lambda_d < 0.F || !std::isfinite(lambda_n) || lambda_n < 0.F)
            throw std::invalid_argument("lambda_d and lambda_n must be finite and nonnegative");
        const bool need_depth = lambda_d > 0.F, need_normals = lambda_n > 0.F;
        if (!need_depth && !need_normals) {
            std::cout << "Prior preparation skipped: lambda_d=0 lambda_n=0\n";
            return 0;
        }
        photara::splat::DatasetLoadRequest request;
        request.source = argv[1]; request.image_directory = argv[2];
        auto loaded = photara::splat::load_splat_dataset(request);
        photara::splat::TrainingOptions training;
        training.max_image_dimension = static_cast<unsigned>(std::stoul(argv[5]));
        training.use_source_resolution = true;
        training.undistort_to_pinhole = true;
        const unsigned split = static_cast<unsigned>(std::stoul(argv[7]));
        const std::filesystem::path output(argv[4]);
        std::filesystem::create_directories(output);
        photara::inference::MogeOptions options;
        options.num_tokens = static_cast<unsigned>(std::stoul(argv[6]));
        auto predictor = photara::inference::create_moge_predictor(argv[3], options);
        std::ofstream manifest(output/"alignment.tsv");
        manifest << "# model=" << predictor->model_info().id << " revision=" << predictor->model_info().revision
            << " tokens=" << options.num_tokens << " eval_split_every=" << split
            << " lambda_d=" << lambda_d << " lambda_n=" << lambda_n << '\n';
        manifest << "view\tscale\tlog_ratio_mad\tsparse_inliers\tvalid_fraction\tseconds\n";
        for (std::size_t i = 0; i < loaded.scene.views.size(); ++i) {
            if (split != 0 && i % split == 0) continue; // No held-out image prediction or supervision.
            const auto start = std::chrono::steady_clock::now();
            const auto& view = loaded.scene.views[i];
            const auto camera = photara::splat::training_data::training_camera(view, training);
            if (camera.model != photara::CameraModel::pinhole)
                throw std::invalid_argument("AI prior preparation currently requires pinhole training cameras");
            const auto image = photara::splat::training_data::prior_input_image(view, training);
            photara::inference::GeometryInput input{image, photara::inference::PinholeIntrinsics{
                camera.fx,camera.fy,camera.cx,camera.cy}};
            const auto prediction = predictor->predict(input);
            // Camera-space normals have no scene-scale ambiguity. Sparse depth
            // alignment is only needed when depth supervision is requested.
            const auto alignment = need_depth ? align(loaded.scene, i, camera, prediction) : Alignment{1.,0.,0};
            photara::splat::TrainingPrior prior;
            prior.camera = camera;
            const auto count = prediction.depth.size();
            if (need_depth) prior.depth.resize(count);
            if (need_normals) prior.normals.resize(3*count);
            std::size_t valid = 0;
            for (std::size_t pixel = 0; pixel < count; ++pixel) {
                if (!prediction.valid[pixel]) continue;
                if (need_depth) {
                    const double depth = prediction.depth[pixel]*alignment.scale;
                    if (!(depth > 0) || depth > std::numeric_limits<float>::max()) continue;
                    prior.depth[pixel] = static_cast<float>(depth);
                }
                if (need_normals)
                    for (int c = 0; c < 3; ++c) prior.normals[c*count+pixel] = prediction.normals[pixel][c];
                ++valid;
            }
            photara::splat::save_training_prior(photara::splat::training_prior_path(output, view.id), view.path, prior);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            manifest << std::setprecision(12) << view.id << '\t' << alignment.scale << '\t' << alignment.mad << '\t'
                << alignment.samples << '\t' << static_cast<double>(valid)/count << '\t' << seconds << '\n';
            manifest.flush();
            std::cout << "prior view=" << view.id << " progress=" << i+1 << '/' << loaded.scene.views.size()
                << " scale=" << alignment.scale << " sparse_inliers=" << alignment.samples << " seconds=" << seconds << '\n' << std::flush;
        }
        if (!manifest) throw std::runtime_error("Cannot write prior alignment manifest");
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
