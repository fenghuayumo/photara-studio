// Evaluate a saved model through the same native metric path used by training.
// This permits recovering final evaluation without repeating optimization.
int run_rgb_evaluation(int argc, char** argv) {
    using namespace photara::splat;
    require(argc >= 6, "Usage: --eval-rgb MODEL IMAGES OUTPUT SFM [cuda|vulkan] [STRIDE]");
    const std::string backend = argc > 6 ? argv[6] : "cuda";
    require(backend == "cuda" || backend == "vulkan", "Invalid evaluation backend");
    const unsigned stride = argc > 7 ? std::stoul(argv[7]) : 8U;
    require(stride > 0, "Evaluation stride must be positive");
    const auto device = backend == "vulkan" ? tinytensor::Device::Vulkan : tinytensor::Device::CUDA;
    auto model = load_gaussians(argv[2], device);
    const auto views = load_views(argv[3], argv[5]);
    const std::filesystem::path output(argv[4]);
    const auto stem = std::filesystem::path(argv[2]).stem().string();
    std::filesystem::create_directories(output);
    TrainingOptions options;
    options.backend = backend == "vulkan" ? TrainingBackend::vulkan : TrainingBackend::cuda;
    options.use_mask = false;
    options.use_source_resolution = true;
    options.max_image_dimension = 1920;
    options.progressive_resolution = false;
    options.ignore_undistortion_border = true;
    double psnr = 0.0, ssim = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < views.size(); i += stride) {
        const auto path = output / (stem + "_view_" + std::to_string(i) + ".png");
        const auto metrics = render_evaluation_png(model, views[i], path, options);
        std::cout << "splat_render=" << path << " view=" << i
                  << " psnr=" << metrics.psnr << " ssim=" << metrics.ssim
                  << " foreground_psnr=" << metrics.foreground_psnr << std::endl;
        psnr += metrics.psnr; ssim += metrics.ssim; ++count;
    }
    require(count > 0, "No evaluation views");
    std::cout << "splat_saved_evaluation_views=" << count << " stride=" << stride << " average_psnr="
              << psnr/count << " average_ssim=" << ssim/count << std::endl;
    return 0;
}

// Same model, camera and optimizer configuration on CUDA and Vulkan.
// Measures RGB render/loss/fused backward+Adam and the per-step ADC statistics
// and exploration noise. Excludes topology refinement, decay and image loading.
int run_rgb_benchmark(int argc, char** argv) {
    using namespace photara::splat;
    using tinytensor::Device;
    using tinytensor::Tensor;
    require(argc >= 6, "Usage: --bench-rgb MODEL IMAGES OUTPUT SFM [ITERS] [VIEW]");
    const std::filesystem::path output(argv[4]);
    std::filesystem::create_directories(output);
    const auto views = load_views(argv[3], argv[5]);
    const unsigned iterations = argc > 6 ? std::max(1, std::stoi(argv[6])) : 21;
    const std::size_t index = argc > 7 ? std::stoul(argv[7]) : 0;
    require(index < views.size(), "RGB benchmark view outside dataset");
    std::ofstream csv(output / ("rgb_view_" + std::to_string(index) + ".csv"));
    csv << "backend,iteration,forward_ms,loss_ms,backward_adam_ms,adc_stats_ms,noise_ms,total_ms\n";
    for (const auto device : {Device::CUDA, Device::Vulkan}) {
        const bool vk = device == Device::Vulkan;
        const char* name = vk ? "Vulkan" : "CUDA";
        auto model = load_gaussians(argv[2], device);
        TrainingOptions o;
        o.backend = vk ? TrainingBackend::vulkan : TrainingBackend::cuda;
        o.use_mask = false; o.use_source_resolution = true;
        o.max_image_dimension = 1920; o.ignore_undistortion_border = true;
        o.progressive_resolution = false; o.use_depth_normal_loss = false;
        o.use_normal_field = false; o.multi_view_geo_weight = o.multi_view_ncc_weight = 0.F;
        const auto view = make_training_view(views[index], o);
        auto means = detail::make_adam_state(model.means);
        auto scales = detail::make_adam_state(model.log_scales);
        auto rotations = detail::make_adam_state(model.quaternions);
        auto opacity = detail::make_adam_state(model.opacity_logits);
        auto sh = detail::make_sh_adam_quant(model.size(), model.sh.shape()[1] * 3, device);
        auto stats = detail::make_densification_stats(model.size(), device);
        Rasterizer rasterizer;
        RasterizeOptions ro;
        ro.active_sh_degree = model.sh_degree; ro.require_depth = false;
        ro.copy_attachments = !vk; ro.defer_visibility = vk;
        const auto sync = [&] {
            if (vk) (void)Tensor::zeros({1}, Device::Vulkan).to_vector();
            else require(cudaDeviceSynchronize() == cudaSuccess, "RGB benchmark CUDA sync failed");
        };
        std::vector<double> times[6];
        std::cout << name << " RGB benchmark gaussians=" << model.size() << " view=" << index
                  << " size=" << view.camera.width << 'x' << view.camera.height << std::endl;
        for (unsigned step = 1; step <= iterations + 3; ++step) {
            SHAdamUpdate su;
            su.first = sh.first; su.packed = sh.packed; su.bounds = sh.bounds;
            su.lr = o.sh0_lr; su.rest_lr = o.sh_rest_lr;
            su.beta1 = o.beta1; su.beta2 = o.beta2;
            su.correction1 = 1.F - std::pow(o.beta1, float(step));
            su.correction2 = 1.F - std::pow(o.beta2, float(step));
            su.epsilon = o.adam_epsilon; su.regularization_weight = o.sh_regularization_weight;
            StructureAdamUpdate st{means.first, means.second, scales.first, scales.second,
                rotations.first, rotations.second, opacity.first, opacity.second,
                o.means_lr, o.scales_lr, o.quaternions_lr, o.opacities_lr,
                o.beta1, o.beta2, su.correction1, su.correction2, o.adam_epsilon};
            st.minimum_log_scale = -12.F; st.maximum_log_scale = 4.F;
            st.max_log_scale_ratio = o.max_scale_ratio > 1.F ? std::log(o.max_scale_ratio) : 0.F;
            sync();
            const auto start = std::chrono::steady_clock::now();
            auto rendered = rasterizer.forward(model, view.camera, ro);
            sync(); const auto after_forward = std::chrono::steady_clock::now();
            detail::LossGradients loss;
            if (vk) loss.total = rasterizer.photometric_loss(rendered, view.rgb, view.mask,
                false, o.ssim_weight, o.photometric_weight, false);
            else loss = detail::compute_training_loss(rendered, view, o, false, false, false);
            sync(); const auto after_loss = std::chrono::steady_clock::now();
            const auto gradients = rasterizer.backward(model, rendered, loss.color,
                loss.alpha, {}, {}, {}, &su, &st);
            sync(); const auto after_backward = std::chrono::steady_clock::now();
            rasterizer.materialize_visibility(rendered);
            detail::accumulate_densification_stats(
                gradients.refine_weight, rendered.visibility, rendered.radii,
                stats, view.camera.width, view.camera.height, true, true,
                1.F, o.densify_screen_threshold, gradients.refine_weight,
                static_cast<int>(index));
            sync(); const auto after_stats = std::chrono::steady_clock::now();
            detail::inject_adc_noise(model, rendered.visibility,
                o.means_lr * o.mean_noise_weight, 1.F, o.seed + step,
                rendered.radii, false);
            sync(); const auto after_noise = std::chrono::steady_clock::now();
            require(!gradients.means.is_valid() && !gradients.sh.is_valid(),
                "RGB benchmark did not exercise fused optimizers");
            const double row[]{
                std::chrono::duration<double,std::milli>(after_forward-start).count(),
                std::chrono::duration<double,std::milli>(after_loss-after_forward).count(),
                std::chrono::duration<double,std::milli>(after_backward-after_loss).count(),
                std::chrono::duration<double,std::milli>(after_stats-after_backward).count(),
                std::chrono::duration<double,std::milli>(after_noise-after_stats).count(),
                std::chrono::duration<double,std::milli>(after_noise-start).count()};
            if (step > 3) {
                csv << name << ',' << step-4;
                for (unsigned i=0; i<6; ++i) { times[i].push_back(row[i]); csv << ',' << row[i]; }
                csv << '\n';
            }
        }
        const char* labels[]{"forward","photo_loss","backward_adam","adc_stats","noise","total"};
        for (unsigned i=0; i<6; ++i)
            std::cout << name << " phase=" << labels[i] << " median_ms=" << median_ms(times[i]) << std::endl;
        for (const auto* tensor : {&model.means, &model.log_scales, &model.quaternions,
                                   &model.opacity_logits, &model.sh}) {
            const auto values = tensor->to_vector();
            require(std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }),
                "RGB benchmark produced non-finite parameters");
        }
    }
    return 0;
}
