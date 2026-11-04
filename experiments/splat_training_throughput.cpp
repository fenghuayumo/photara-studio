// Matched training-loop benchmark. Never modifies the input model/dataset.
#include "splat/dataset.hpp"
#include "splat/trainer.hpp"
#include "splat/formats.hpp"
#include "../photara/src/splat/checkpoint.hpp"
#include "core/logging.hpp"
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    try {
        if (argc != 9 && argc != 10) throw std::runtime_error(
            "DATASET IMAGES MODEL OUTPUT STEPS LOG_INTERVAL PROFILE PREFETCH [START_ITERATION]");
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
        options.max_image_dimension = 0;
        options.progressive_resolution = false;
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
