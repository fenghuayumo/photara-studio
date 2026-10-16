#include "splat/prior_prepare.hpp"
#include "inference/models/moge.hpp"
#include "splat/prior.hpp"
#include "training_data_loader.hpp"
#include "core/logging.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <memory>

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

namespace photara::splat {
void prepare_training_priors(const mvs::MvsScene& scene, const TrainingOptions& training,
    const std::filesystem::path& model, unsigned tokens) {
    if (!(training.lambda_d > 0.F || training.lambda_n > 0.F)) return;
    if (training.prior_directory.empty()) throw std::invalid_argument("AI prior cache directory is empty");
    if (tokens < 1 || tokens > 10000) throw std::invalid_argument("AI prior tokens must be between 1 and 10000");
    if (training.use_mvs_depth || training.use_mvs_normals)
        throw std::invalid_argument("AI priors cannot share targets with MVS supervision");
    core::StageScope stage("splat.prepare_ai_priors");
    const bool need_depth = training.lambda_d > 0.F, need_normals = training.lambda_n > 0.F;
    const auto split = training.evaluation_split_every;
    const bool held_out = split > 0;
    std::size_t total = 0;
    for (std::size_t i=0;i<scene.views.size();++i) {
        if (held_out && i % split == 0) continue;
        if (training_data::training_camera(scene.views[i],training).model != CameraModel::pinhole)
            throw std::invalid_argument("AI priors require pinhole training cameras; enable pinhole undistortion for fisheye captures. Panoramas are not supported.");
        ++total;
    }
    if (!total) throw std::invalid_argument("No training views available for AI priors");
    // Include sparse scale support, preprocessing and model generation in reuse
    // identity. The map loader separately checks every camera and image file.
    std::uint64_t sparse_hash = 14695981039346656037ULL;
    const auto hash = [&](const auto& value) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
        for (std::size_t i=0;i<sizeof(value);++i) { sparse_hash ^= bytes[i]; sparse_hash *= 1099511628211ULL; }
    };
    for (const auto& point:scene.sparse_points) {
        for (int c=0;c<3;++c) hash(point.position[c]);
        for (auto id:point.view_ids) hash(id);
        hash(point.view_ids.size());
    }
    const auto model_name = std::filesystem::absolute(model).lexically_normal().generic_u8string();
    std::ostringstream identity;
    identity << "photara-ai-priors-v1\n" << std::string(model_name.begin(),model_name.end())
        << '\n' << std::filesystem::file_size(model) << '\n'
        << std::filesystem::last_write_time(model).time_since_epoch().count() << '\n'
        << tokens << '\n' << sparse_hash << '\n' << training.undistort_to_pinhole
        << '\n' << training.ignore_undistortion_border << '\n';
    const auto& output = training.prior_directory;
    std::filesystem::create_directories(output);
    const auto signature = output / "generation.txt";
    std::ifstream old(signature);
    const std::string previous((std::istreambuf_iterator<char>(old)),{});
    const bool reuse = previous == identity.str();
    old.close();
    // A cancelled generation must not label a mixed old/new cache as reusable.
    std::filesystem::remove(signature);
    core::ProgressReporter progress("Prepare AI priors", total);
    std::unique_ptr<inference::GeometryPredictor> predictor;
    std::ofstream manifest;
    std::size_t reused = 0, generated = 0;
    for (std::size_t i=0;i<scene.views.size();++i) {
        if (held_out && i % split == 0) continue;
        const auto& view=scene.views[i];
        const auto camera=training_data::training_camera(view,training);
        const auto path=training_prior_path(output,view.id);
        if (reuse) {
            try {
                (void)load_training_prior(path,view.path,camera,need_depth,need_normals);
                ++reused;progress.advance();continue;
            } catch (const std::exception&) { /* Regenerate stale or incomplete maps. */ }
        }
        if (!predictor) {
            inference::MogeOptions options;options.num_tokens=tokens;
            predictor=inference::create_moge_predictor(model,options);
            manifest.open(output/"alignment.tsv");
            if (!manifest) throw std::runtime_error("Cannot write AI prior alignment manifest");
            manifest << "# model=" << predictor->model_info().id << " revision=" << predictor->model_info().revision
                << " tokens=" << tokens << " eval_split_every=" << split << '\n'
                << "view\tscale\tlog_ratio_mad\tsparse_inliers\tvalid_fraction\tseconds\n";
        }
        const auto start=std::chrono::steady_clock::now();
        const auto image=training_data::prior_input_image(view,training);
        const auto prediction=predictor->predict({image,inference::PinholeIntrinsics{camera.fx,camera.fy,camera.cx,camera.cy}});
        const auto alignment=need_depth ? align(scene,i,camera,prediction) : Alignment{1.,0.,0};
        TrainingPrior prior;prior.camera=camera;
        const auto count=prediction.depth.size();
        if (need_depth) prior.depth.resize(count);
        if (need_normals) prior.normals.resize(3*count);
        std::size_t valid=0;
        for(std::size_t pixel=0;pixel<count;++pixel) {
            if (!prediction.valid[pixel]) continue;
            if (need_depth) {
                const double depth=prediction.depth[pixel]*alignment.scale;
                if (!(depth>0) || depth>std::numeric_limits<float>::max()) continue;
                prior.depth[pixel]=static_cast<float>(depth);
            }
            if (need_normals)
                for(int c=0;c<3;++c)prior.normals[c*count+pixel]=prediction.normals[pixel][c];
            ++valid;
        }
        save_training_prior(path,view.path,prior);
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        manifest << std::setprecision(12) << view.id << '\t' << alignment.scale << '\t' << alignment.mad << '\t'
            << alignment.samples << '\t' << static_cast<double>(valid)/count << '\t' << seconds << '\n';
        manifest.flush();
        if (!manifest) throw std::runtime_error("Cannot write AI prior alignment manifest");
        ++generated;progress.advance();
        core::Logger::instance().info("AI prior view=",view.id," scale=",alignment.scale," sparse_inliers=",alignment.samples);
    }
    std::ofstream stamp(signature,std::ios::trunc);stamp<<identity.str();stamp.close();
    if (!stamp) throw std::runtime_error("Cannot write AI prior generation identity");
    core::Logger::instance().info("AI priors ready: generated=",generated," reused=",reused," directory=",output);
}
}
