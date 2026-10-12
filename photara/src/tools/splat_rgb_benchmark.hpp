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
// Streamed mode measures the whole loop and waits only at its boundaries, like
// normal training. Phase/iteration waits can trim the CUDA driver memory pool
// and change the workload, so those modes are for diagnosis rather than throughput.
int run_rgb_benchmark(int argc, char** argv) {
    using namespace photara::splat;
    using tinytensor::Device;
    using tinytensor::Tensor;
    require(argc >= 6, "Usage: --bench-rgb MODEL IMAGES OUTPUT SFM [ITERS] [VIEW] [phases|pipeline|streamed|readback] [cuda-first|vulkan-first] [compact|uncompressed] [fused|split-sh|split-all] [default|retain]");
    const std::filesystem::path output(argv[4]);
    std::filesystem::create_directories(output);
    const auto views = load_views(argv[3], argv[5]);
    const unsigned iterations = argc > 6 ? std::max(1, std::stoi(argv[6])) : 21;
    const std::size_t index = argc > 7 ? std::stoul(argv[7]) : 0;
    const std::string timing = argc > 8 ? argv[8] : "streamed";
    const std::string order = argc > 9 ? argv[9] : "cuda-first";
    const std::string snapshots = argc > 10 ? argv[10] : "compact";
    const std::string optimizer = argc > 11 ? argv[11] : "fused";
    const std::string pool_policy = argc > 12 ? argv[12] : "default";
    require(timing == "phases" || timing == "pipeline" || timing == "streamed" || timing == "readback",
            "Invalid RGB benchmark timing mode");
    require(order == "cuda-first" || order == "vulkan-first", "Invalid RGB benchmark order");
    require(snapshots == "compact" || snapshots == "uncompressed", "Invalid snapshot mode");
    require(optimizer == "fused" || optimizer == "split-sh" || optimizer == "split-all",
            "Invalid CUDA optimizer mode");
    require(pool_policy == "default" || pool_policy == "retain", "Invalid CUDA pool policy");
    const bool streamed = timing == "streamed";
    const bool phase_sync = timing == "phases" || timing == "readback";
    const double unmeasured_phase = std::numeric_limits<double>::quiet_NaN();
    require(index < views.size(), "RGB benchmark view outside dataset");
    const char* capture_env = std::getenv("PHOTARA_RGB_BENCH_CAPTURE_SAMPLES");
    const bool capture_samples = capture_env != nullptr && capture_env[0] == '1';
    std::ofstream csv(output / ("rgb_view_" + std::to_string(index) + ".csv"));
    csv << "backend,iteration,forward_ms,loss_ms,backward_adam_ms,adc_stats_ms,noise_ms,total_ms\n";
    std::ofstream summary(output / ("rgb_summary_view_" + std::to_string(index) + ".csv"));
    summary << "backend,timing,order,cuda_snapshots,cuda_optimizer,cuda_pool,iterations,loop_average_ms\n";
    require(bool(csv) && bool(summary), "Cannot open RGB benchmark output files");
    const Device devices[]{order == "cuda-first" ? Device::CUDA : Device::Vulkan,
                           order == "cuda-first" ? Device::Vulkan : Device::CUDA};
    for (const auto device : devices) {
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
        ro.compact_pixel_snapshots = snapshots == "compact";
        const auto handles = vk ? tinytensor::vulkan::device_handles()
                                : tinytensor::vulkan::DeviceHandles{};
        const bool fuse_sh = vk || optimizer == "fused";
        const bool fuse_structure = vk || optimizer != "split-all";
        cudaMemPool_t cuda_pool{};
        std::uint64_t original_pool_threshold = 0;
        if (!vk) {
            int cuda_device = 0;
            require(cudaGetDevice(&cuda_device) == cudaSuccess &&
                    cudaDeviceGetDefaultMemPool(&cuda_pool, cuda_device) == cudaSuccess &&
                    cudaMemPoolGetAttribute(cuda_pool, cudaMemPoolAttrReleaseThreshold,
                                           &original_pool_threshold) == cudaSuccess,
                    "Cannot inspect CUDA pool");
            if (pool_policy == "retain") {
                std::uint64_t threshold = 8ULL << 30;
                require(cudaMemPoolSetAttribute(cuda_pool, cudaMemPoolAttrReleaseThreshold,
                                               &threshold) == cudaSuccess,
                        "Cannot configure diagnostic CUDA pool retention");
            }
        }
        const auto sync = [&] {
            if (vk && timing == "readback") (void)Tensor::zeros({1}, Device::Vulkan).to_vector();
            else if (vk) {
                tinytensor::vulkan::synchronize();
                require(vkDeviceWaitIdle(handles.device) == VK_SUCCESS,
                        "RGB benchmark Vulkan device sync failed");
            }
            else require(cudaDeviceSynchronize() == cudaSuccess, "RGB benchmark CUDA sync failed");
        };
        std::vector<double> times[6];
        std::cout << name << " RGB benchmark gaussians=" << model.size() << " view=" << index
                  << " size=" << view.camera.width << 'x' << view.camera.height
                  << " timing=" << timing << " order=" << order
                  << " cuda_snapshots=" << snapshots << " cuda_optimizer=" << optimizer
                  << " cuda_pool=" << pool_policy << std::endl;
        std::chrono::steady_clock::time_point measured_loop_start;
        const auto& pool_stats = tinytensor::SizeBucketedPool::instance().stats();
        std::uint64_t initial_pool_hits = 0, initial_pool_misses = 0;
        for (unsigned step = 1; step <= iterations + 3; ++step) {
            if (step == 4) {
                if (streamed) sync(); // Drain warmup before measuring sustained throughput.
                initial_pool_hits = pool_stats.cache_hits.load();
                initial_pool_misses = pool_stats.cache_misses.load();
                measured_loop_start = std::chrono::steady_clock::now();
            }
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
            if (!streamed) sync();
            const auto start = std::chrono::steady_clock::now();
            auto rendered = rasterizer.forward(model, view.camera, ro);
            if (phase_sync) sync();
            const auto after_forward = std::chrono::steady_clock::now();
            detail::LossGradients loss;
            if (vk) loss.total = rasterizer.photometric_loss(rendered, view.rgb, view.mask,
                view.has_mask && (o.use_mask || view.mask_is_validity),
                o.ssim_weight, o.photometric_weight, false);
            else loss = detail::compute_training_loss(rendered, view, o, false, false, false);
            if (phase_sync) sync();
            const auto after_loss = std::chrono::steady_clock::now();
            const auto gradients = rasterizer.backward(model, rendered, loss.color,
                loss.alpha, {}, {}, {}, fuse_sh ? &su : nullptr,
                fuse_structure ? &st : nullptr);
            if (!fuse_sh)
                detail::sh_adam_quant_step(model.sh, gradients.sh, sh,
                    (ro.active_sh_degree + 1) * (ro.active_sh_degree + 1) * 3,
                    su.lr, su.rest_lr, su.beta1, su.beta2, su.correction1,
                    su.correction2, su.epsilon,
                    sh.stride > 3 ? 2.F * std::max(su.regularization_weight, 0.F) /
                        float(model.size() * (sh.stride - 3)) : 0.F);
            if (!fuse_structure)
                detail::adam_step_structure(model, gradients, means, scales, rotations,
                    opacity, st.means_lr, step, o, st.minimum_log_scale, st.maximum_log_scale);
            if (phase_sync) sync();
            const auto after_backward = std::chrono::steady_clock::now();
            rasterizer.materialize_visibility(rendered);
            detail::accumulate_densification_stats(
                gradients.refine_weight, rendered.visibility, rendered.radii,
                stats, view.camera.width, view.camera.height, true, true,
                1.F, o.densify_screen_threshold, gradients.refine_weight,
                static_cast<int>(index));
            if (phase_sync) sync();
            const auto after_stats = std::chrono::steady_clock::now();
            detail::inject_adc_noise(model, rendered.visibility,
                o.means_lr * o.mean_noise_weight, 1.F, o.seed + step,
                rendered.radii, false);
            if (!streamed) sync();
            const auto after_noise = std::chrono::steady_clock::now();
            require(gradients.means.is_valid() == !fuse_structure &&
                    gradients.sh.is_valid() == !fuse_sh,
                    "RGB benchmark did not exercise requested optimizers");
            const double row[]{
                phase_sync ? std::chrono::duration<double,std::milli>(after_forward-start).count() : unmeasured_phase,
                phase_sync ? std::chrono::duration<double,std::milli>(after_loss-after_forward).count() : unmeasured_phase,
                phase_sync ? std::chrono::duration<double,std::milli>(after_backward-after_loss).count() : unmeasured_phase,
                phase_sync ? std::chrono::duration<double,std::milli>(after_stats-after_backward).count() : unmeasured_phase,
                phase_sync ? std::chrono::duration<double,std::milli>(after_noise-after_stats).count() : unmeasured_phase,
                streamed ? unmeasured_phase :
                    std::chrono::duration<double,std::milli>(after_noise-start).count()};
            if (step > 3) {
                csv << name << ',' << step-4;
                for (unsigned i=0; i<6; ++i) { times[i].push_back(row[i]); csv << ',' << row[i]; }
                csv << '\n';
            }
        }
        // Include destruction, deferred frees and preparation between iterations.
        // Per-step phase totals intentionally omit these costs.
        sync();
        const double loop_average = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - measured_loop_start).count() / iterations;
        std::cout << name << " loop_average_ms=" << loop_average << std::endl;
        summary << name << ',' << timing << ',' << order << ',' << snapshots << ','
                << optimizer << ',' << pool_policy << ',' << iterations << ',' << loop_average << '\n';
        if (!vk) {
            std::uint64_t reserved = 0, used = 0;
            require(cudaMemPoolGetAttribute(cuda_pool, cudaMemPoolAttrReservedMemCurrent,
                                           &reserved) == cudaSuccess &&
                    cudaMemPoolGetAttribute(cuda_pool, cudaMemPoolAttrUsedMemCurrent,
                                           &used) == cudaSuccess,
                    "Cannot inspect final CUDA pool memory");
            std::cout << "CUDA pool_release_threshold_original_bytes=" << original_pool_threshold
                      << " reserved_bytes=" << reserved << " used_bytes=" << used
                      << " measured_bucket_hits=" << pool_stats.cache_hits.load() - initial_pool_hits
                      << " measured_bucket_misses=" << pool_stats.cache_misses.load() - initial_pool_misses
                      << " cached_bytes=" << pool_stats.bytes_cached.load() << std::endl;
            require(cudaMemPoolSetAttribute(cuda_pool, cudaMemPoolAttrReleaseThreshold,
                                           &original_pool_threshold) == cudaSuccess,
                    "Cannot restore CUDA pool policy");
        }
        const char* labels[]{"forward","photo_loss","backward_adam","adc_stats","noise","total"};
        for (unsigned i=0; i<6; ++i) {
            if (!streamed && (phase_sync || i == 5))
                std::cout << name << " phase=" << labels[i] << " median_ms=" << median_ms(times[i]) << std::endl;
        }
        const Tensor* parameters[]{&model.means, &model.log_scales, &model.quaternions,
                                   &model.opacity_logits, &model.sh};
        const char* parameter_names[]{"means", "log_scales", "quaternions", "opacity_logits", "sh"};
        for (unsigned parameter = 0; parameter < 5; ++parameter) {
            const auto* tensor = parameters[parameter];
            const auto values = tensor->to_vector();
            require(std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }),
                "RGB benchmark produced non-finite parameters");
            if (!vk && capture_samples) {
                // Evenly spaced values cover the full tensor. These diagnostic
                // samples check numerical agreement, not complete model parity.
                std::vector<float> samples;
                const auto stride = std::max<std::size_t>(1, values.size() / 65536);
                for (std::size_t i = 0; i < values.size(); i += stride) samples.push_back(values[i]);
                std::ofstream sample_file(output / (std::string("CUDA_") +
                    parameter_names[parameter] + ".samples.f32"), std::ios::binary);
                sample_file.write(reinterpret_cast<const char*>(samples.data()),
                                  samples.size() * sizeof(float));
                require(bool(sample_file), "Cannot write benchmark parameter samples");
            }
        }
    }
    return 0;
}
