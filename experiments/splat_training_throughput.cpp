// Matched training-loop benchmark. Never modifies the input model/dataset.
#include "splat/dataset.hpp"
#include "splat/trainer.hpp"
#include "splat/formats.hpp"
#include "../photara/src/splat/checkpoint.hpp"
#include "core/logging.hpp"
#include <cuda_runtime.h>
#include "../photara/src/splat/training_data_loader.hpp"
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    try {
        // CPU-only byte-equivalence and cold image-preparation benchmark. Run
        // with OMP_NUM_THREADS=1 to match a background prefetch job.
        if (argc == 5 && std::string(argv[1]) == "--host-images") {
            using namespace photara::splat;
            DatasetLoadRequest request;
            request.source = argv[2]; request.image_directory = argv[3];
            const auto scene = load_splat_dataset(request).scene;
            const std::filesystem::path output = argv[4];
            if (std::filesystem::exists(output)) throw std::runtime_error("Output exists");
            std::ofstream csv(output);
            csv << "view,scale,width,height,hash,ms\n";
            TrainingOptions options;
            options.use_mask = false;
            options.use_source_resolution = true;
            options.max_image_dimension = 4096;
            for (float scale : {0.25F, 0.5F, 1.F}) {
                for (std::size_t index = 0; index < std::min<std::size_t>(12, scene.views.size()); ++index) {
                    const auto& view = scene.views[index];
                    const auto camera = training_data::training_camera(view, options, scale);
                    auto image_options = options;
                    image_options.max_image_dimension = std::max(camera.width, camera.height);
                    const auto started = std::chrono::steady_clock::now();
                    const auto image = training_data::load_training_image(view, image_options);
                    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started).count();
                    std::uint64_t hash = 14695981039346656037ULL;
                    for (const auto byte : image.rgb.pixels) hash = (hash ^ byte) * 1099511628211ULL;
                    for (const auto byte : image.mask.pixels) hash = (hash ^ byte) * 1099511628211ULL;
                    csv << index << ',' << scale << ',' << image.camera.width << ','
                        << image.camera.height << ',' << hash << ',' << ms << '\n';
                }
            }
            return 0;
        }
        if (argc < 9 || argc > 12) throw std::runtime_error(
            "DATASET IMAGES MODEL OUTPUT STEPS LOG_INTERVAL PROFILE PREFETCH [START_ITERATION [MAX_RESOLUTION [PROGRESSIVE]]]");
        using namespace photara::splat;
        photara::core::Logger::instance().set_levels(
            photara::core::LogLevel::info, photara::core::LogLevel::off);
        DatasetLoadRequest request;
        request.source = argv[1]; request.image_directory = argv[2];
        auto scene = load_splat_dataset(request).scene;
        const std::filesystem::path output = argv[4];
        if (std::filesystem::exists(output)) throw std::runtime_error("Output exists");
        std::filesystem::create_directories(output);
        TrainingOptions options;
        options.input_is_dense = false;
        apply_strategy_defaults(options);
        options.use_mask = false;
        options.use_source_resolution = true;
        options.max_image_dimension = argc > 10 ? std::stoul(argv[10]) : 0;
        options.progressive_resolution = argc > 11 && std::stoi(argv[11]) != 0;
        options.initial_point_budget = 500000;
        options.densification_cap = 1500000;
        options.grow_stop_iter = 15000;
        options.use_mvs_depth = options.use_mvs_normals = false;
        options.use_depth_normal_loss = options.use_normal_field = false;
        options.multi_view_geo_weight = options.multi_view_ncc_weight = 0;
        options.scale_regularization_weight = .01F;
        options.shape_axis_ratio_reg = 0;
        options.preview_interval = 0;
        options.evaluation_split_every = 21;
        options.log_interval = std::stoul(argv[6]);
        options.profile_cuda = std::stoi(argv[7]) != 0;
        options.training_prefetch_views = std::stoul(argv[8]);
        options.checkpoint_path = output / "warm-start.json";
        options.resume_checkpoint = true;
        const unsigned start = argc > 9 ? std::stoul(argv[9]) : 16000;
        const unsigned steps = std::stoul(argv[5]);
        if (start >= options.iterations || steps > options.iterations - start)
            throw std::runtime_error("Benchmark window exceeds training iterations");
        if (!steps || steps > 10000 || options.log_interval == 0 ||
            steps % options.log_interval != 0)
            throw std::runtime_error("Steps must be 1..10000 and divisible by LOG_INTERVAL");
        {
            auto model = load_gaussians(argv[3]);
            const TrainingCheckpointInfo info{start, options.iterations,
                model.size(), detail::training_fingerprint(scene, options), {}, {}};
            detail::save_model_checkpoint(options.checkpoint_path, info, model);
        }
        std::ofstream csv(output / "progress.csv");
        csv << "iteration,count,loss,ms,vram\n";
        auto model = Trainer(options).train(scene, [&](const TrainingProgress& p) {
            csv << p.iteration << ',' << p.gaussian_count << ',' << p.loss
                << ',' << p.milliseconds << ',' << p.vram_used_bytes << '\n';
            if (p.iteration % 100 == 0)
                std::cout << "iteration=" << p.iteration << " loss=" << p.loss << std::endl;
            return p.iteration < start + steps;
        });
        // The same view after the same updates exposes any numerical regression.
        auto metrics = render_evaluation_png(model, scene.views[0], output / "eval.png", options);
        std::ofstream(output / "result.json") << "{\"psnr\":" << metrics.psnr
            << ",\"ssim\":" << metrics.ssim << ",\"count\":" << model.size() << "}\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
