#include "core/logging.hpp"
#include "io/image.hpp"
#include "sfm/asfm.hpp"
#include "splat/formats.hpp"
#include "splat/trainer.hpp"
#include "../photara/src/splat/densification_internal.hpp"
#include "../photara/src/splat/training_data_loader.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

// Evaluate exported canonical RGB, never the training-only appearance transforms.
// A compaction holdout may have been seen by the original reconstruction.
int main(int argc, char** argv) {
    try {
        if (argc != 6 && argc != 7)
            throw std::runtime_error(
                "Usage: popspa_review MODEL SFM IMAGES OUTPUT MODE [STEPS_PER_STAGE]; "
                "MODE=baseline|prune20|popspa20|popspa40|unpruned|codec");
        using namespace photara;
        using namespace photara::splat;
        const std::filesystem::path output(argv[4]);
        std::filesystem::create_directories(output);
        core::Logger::instance().configure(output, "review", core::LogLevel::warning);
        const auto source = sfm::load_asfm(argv[2]);
        mvs::MvsScene scene;
        for (const auto& image : source.images)
            if (image.registered) {
                const auto& intr = source.camera_of(image);
                mvs::MvsView view;
                view.id = static_cast<mvs::Index>(scene.views.size());
                view.path = std::filesystem::path(argv[3]) / image.path.filename();
                if (!std::filesystem::is_regular_file(view.path))
                    throw std::runtime_error("Missing training image");
                view.pose = image.pose;
                view.source_model = intr.model;
                view.width = view.src_width = intr.width;
                view.height = view.src_height = intr.height;
                view.fx = view.src_fx = intr.fx;
                view.fy = view.src_fy = intr.fy;
                view.cx = view.src_cx = intr.cx;
                view.cy = view.src_cy = intr.cy;
                view.k1 = intr.k1;
                view.k2 = intr.k2;
                view.p1 = intr.p1;
                view.p2 = intr.p2;
                scene.views.push_back(std::move(view));
            }
        if (scene.views.size() < 16)
            throw std::runtime_error("Need at least 16 registered views");
        const auto original = load_gaussians(argv[1]);
        const std::string mode(argv[5]);
        if (mode != "baseline" && mode != "prune20" && mode != "popspa20" && mode != "popspa40" &&
            mode != "unpruned" && mode != "codec" && mode != "speckles" && mode != "s3pair20" && mode != "adapt")
            throw std::runtime_error("Unknown MODE");
        TrainingOptions options;
        options.sh_degree = original.sh_degree;
        options.filter_mode = original.filter_mode;
        options.render_method = original.render_method;
        options.popspa = true;
        options.popspa_keep_ratio = mode == "unpruned" ? 1.F : mode == "popspa40" ? 0.4F : 0.2F;
        options.popspa_refine_iterations = options.popspa_finetune_iterations =
            argc == 7 ? unsigned(std::stoul(argv[6])) : 5000U;
        options.use_ppisp = options.use_bilateral_grid = true;
        options.evaluation_split_every = 8;
        options.max_image_dimension = 960;
        options.use_mask = options.use_mvs_depth = options.use_mvs_normals = false;
        options.use_normal_field = options.use_depth_normal_loss = false;
        options.lambda_d = options.lambda_n = options.multi_view_geo_weight =
            options.multi_view_ncc_weight = 0.F;
        options.log_interval = 500;
        options.sog_xt_finetune_iterations = argc == 7 ? unsigned(std::stoul(argv[6])) : 4000U;
        if (mode == "s3pair20")
            options.popspa_geometry_lr_scale = 1.F;
        if (mode == "speckles") {
            nlohmann::ordered_json audit{
                {"geometry_unchanged", true}, {"variants", nlohmann::ordered_json::array()}};
            for (const std::size_t v : {112U, 368U, 559U}) {
                if (v >= scene.views.size())
                    continue;
                const auto target = make_training_view(scene.views[v], options);
                const auto rgb = target.rgb.to_vector();
                const auto pixels = std::size_t(target.camera.width) * target.camera.height;
                io::RgbImage truth{
                    target.camera.width, target.camera.height,
                    std::vector<std::uint8_t>(pixels * 3)};
                for (std::size_t p = 0; p < pixels; ++p)
                    for (unsigned c = 0; c < 3; ++c)
                        truth.pixels[3 * p + c] = static_cast<std::uint8_t>(
                            std::lround(255.F * std::clamp(rgb[c * pixels + p], 0.F, 1.F)));
                io::save_rgb_png(truth, output / ("target-" + std::to_string(v) + ".png"));
                for (const unsigned degree : {original.sh_degree, 0U}) {
                    auto variant = original;
                    variant.sh_degree = degree; // Keep the resident coefficient stride unchanged.
                    const auto metrics = render_evaluation_png(
                        variant, scene.views[v],
                        output / ("sh" + std::to_string(degree) + "-" + std::to_string(v) + ".png"),
                        options);
                    audit["variants"].push_back(
                        {{"view", v},
                         {"sh_degree", degree},
                         {"psnr", metrics.psnr},
                         {"ssim", metrics.ssim}});
                }
            }
            std::ofstream(output / "audit.json") << audit.dump(2) << '\n';
            return 0;
        }
        const auto start = std::chrono::steady_clock::now();
        GaussianModel result = original;
        if (mode == "prune20") {
            training_data::TrainingDataLoader cameras(scene.views, options);
            Rasterizer raster;
            RasterizeOptions raster_options;
            configure_rasterization(raster_options, options);
            raster_options.active_sh_degree = original.sh_degree;
            auto scores = tinytensor::Tensor::zeros({original.size()}, original.means.device());
            for (std::size_t v = 0; v < scene.views.size(); ++v) {
                if (v % options.evaluation_split_every == 0)
                    continue;
                scores = scores.add(
                    raster.pruning_scores(original, cameras.get_camera(v), raster_options, true));
            }
            const auto values = scores.to_vector();
            std::vector<int> rows(original.size());
            std::iota(rows.begin(), rows.end(), 0);
            const auto keep = std::max<std::size_t>(
                1, std::llround(original.size() * double(options.popspa_keep_ratio)));
            std::partial_sort(rows.begin(), rows.begin() + keep, rows.end(), [&](int a, int b) {
                return values[a] != values[b] ? values[a] > values[b] : a < b;
            });
            rows.resize(keep);
            std::sort(rows.begin(), rows.end());
            result = densification::gpu_detail::select_model_rows(
                original,
                tinytensor::Tensor::from_vector(rows, {rows.size()}, original.means.device()));
        } else if (mode == "adapt") {
            result=Trainer(options).finetune_sog_xt(original,scene,output/"model.sog-xt",[&](const TrainingProgress& p){
                std::cout<<"adapt step="<<p.iteration<<"/"<<p.total_iterations<<" loss="<<p.loss<<std::endl;return true;
            });
        } else if (mode == "codec") {
            save_gaussians(result, output / "model.sog-xt");
            result = load_gaussians(output / "model.sog-xt");
        } else if (mode != "baseline") {
            result =
                Trainer(options).compact_popspa(original, scene, [&](const TrainingProgress& p) {
                    if (p.iteration)
                        std::cout << mode << " step=" << p.iteration << "/" << p.total_iterations
                                  << " count=" << p.gaussian_count << " loss=" << p.loss
                                  << std::endl;
                    return true;
                });
        }
        const auto seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        nlohmann::ordered_json report{
            {"mode", mode},
            {"s3_variant", "normalized_share"},
            {"filter_mode", filter_mode_name(options.filter_mode)},
            {"input_count", original.size()},
            {"output_count", result.size()},
            {"registered_views", scene.views.size()},
            {"max_image_dimension", 960},
            {"compaction_holdout_every", options.evaluation_split_every},
            {"use_ppisp", options.use_ppisp},
            {"use_bilateral_grid", options.use_bilateral_grid},
            {"steps_per_stage", mode == "baseline" || mode == "prune20" || mode == "codec"
                                    ? 0
                                    : options.popspa_refine_iterations},
            {"prune_cycles", options.popspa_prune_cycles},
            {"geometry_lr_scale", options.popspa_geometry_lr_scale},
            {"seed", options.seed},
            {"finetune_lr_scale", options.popspa_finetune_lr_scale},
            {"optimization_seconds", seconds},
            {"codec_adaptation_steps", mode == "adapt" ? options.sog_xt_finetune_iterations : 0U},
            {"codec_adaptation_batch", mode == "adapt" ? options.sog_xt_batch_size : 0U},
            {"codec_reference_weight", options.sog_xt_reference_weight},
            {"codec_learning_rate_scale", options.sog_xt_learning_rate_scale},
            {"metrics", nlohmann::ordered_json::array()}};
        // Retain every view so regressions cannot be hidden by a selected subset.
        for (std::size_t v = 0; v < scene.views.size(); ++v) {
            const auto metrics = render_evaluation_png(
                result, scene.views[v], output / ("view-" + std::to_string(v) + ".png"), options);
            report["metrics"].push_back(
                {{"view", v},
                 {"holdout", v % options.evaluation_split_every == 0},
                 {"psnr", metrics.psnr},
                 {"ssim", metrics.ssim}});
            if (v % 50 == 0)
                std::cout << mode << " evaluated=" << v + 1 << "/" << scene.views.size()
                          << std::endl;
        }
        if (mode == "codec" || mode == "adapt")
            report["encoded_bytes"] = std::filesystem::file_size(output / "model.sog-xt");
        else if (mode != "baseline")
            save_gaussians_ply(result, output / "model.ply");
        std::ofstream file(output / "result.json");
        file << report.dump(2) << '\n';
        if (!file)
            throw std::runtime_error("Could not save result.json");
        std::cout << "DONE " << mode << " count=" << result.size()
                  << " optimization_seconds=" << seconds << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
