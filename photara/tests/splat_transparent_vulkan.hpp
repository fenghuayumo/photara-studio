// Included by splat_test.cpp to share the CUDA reference and test helpers.
void test_vulkan_transparent_loss() {
#ifdef TINYTENSOR_HAS_VULKAN
    using namespace photara::splat;
    using tinytensor::Tensor;
    using tinytensor::Device;
    constexpr std::size_t count = 513;
    std::vector<float> alpha(count), mask(count);
    const float predictions[] = {0.F, 1.F, 1e-8F, 1.F-1e-7F, .2F, .65F, .8F};
    const float targets[] = {0.F, 1.F, .25F, .75F};
    for (std::size_t i = 0; i < count; ++i) {
        alpha[i] = predictions[i % std::size(predictions)];
        mask[i] = targets[i % std::size(targets)];
    }
    TrainingOptions options;
    options.use_mask = true;
    options.alpha_mode = AlphaMode::transparent;
    options.match_alpha_weight = .37F;
    options.photometric_weight = 0.F;
    const auto run = [&](Device device, const std::vector<float>& values,
                         const TrainingOptions& settings, bool collect = true,
                         bool has_mask = true, bool validity = false) {
        RenderResult rendered;
        rendered.color = Tensor::zeros({3, 1, count}, device);
        rendered.alpha = Tensor::from_vector(values, {1, count}, device);
        rendered.median_depth = Tensor::zeros({1, count}, device);
        rendered.normal = Tensor::zeros({3, 1, count}, device);
        TrainingView target;
        target.camera.width = count; target.camera.height = 1;
        target.rgb = rendered.color;
        target.depth = rendered.median_depth; target.normal = rendered.normal;
        target.mask = Tensor::from_vector(mask, {1, count}, device);
        target.has_mask = has_mask; target.mask_is_validity = validity;
        return detail::compute_training_loss(rendered, target, settings, collect);
    };
    auto cuda = run(Device::CUDA, alpha, options);
    auto vk = run(Device::Vulkan, alpha, options);
    require(std::abs(cuda.alpha_value-vk.alpha_value) < 2e-5F,
        "Vulkan transparent BCE differs from CUDA");
    const auto reference = cuda.alpha.to_vector(), gradient = vk.alpha.to_vector();
    for (std::size_t i = 0; i < count; ++i)
        require(std::isfinite(gradient[i]) && std::abs(reference[i]-gradient[i]) < 2e-5F,
            "Vulkan alpha gradient differs from CUDA, including clamp boundaries");
    auto upper = alpha, lower = alpha;
    constexpr std::size_t pixel = 5;
    constexpr float step = .001F;
    upper[pixel] += step; lower[pixel] -= step;
    const float numerical = (run(Device::Vulkan, upper, options).alpha_value -
        run(Device::Vulkan, lower, options).alpha_value) / (2.F*step);
    require(std::abs(numerical-gradient[pixel]) < 1e-4F,
        "Vulkan transparent alpha gradient failed finite differences");
    const auto uncollected = run(Device::Vulkan, alpha, options, false);
    require(uncollected.alpha.to_vector() == gradient && uncollected.alpha_value == 0.F,
        "Unreported Vulkan alpha loss changed gradients");
    for (unsigned variant = 0; variant < 5; ++variant) {
        auto disabled = options;
        if (variant == 0) disabled.use_mask = false;
        if (variant == 1) disabled.alpha_mode = AlphaMode::masked;
        if (variant == 2) disabled.match_alpha_weight = 0.F;
        const auto loss = run(Device::Vulkan, alpha, disabled, true,
            variant != 3, variant == 4);
        const auto values = loss.alpha.to_vector();
        require(loss.alpha_value == 0.F && std::all_of(values.begin(), values.end(),
            [](float v) { return v == 0.F; }),
            "Disabled/missing/validity mask unexpectedly supervised Vulkan alpha");
    }
    std::cout << "Vulkan transparent BCE CUDA parity and finite differences passed\n";

    // Exercise RGB+alpha together, including the rasterizer backward that
    // transfers image-alpha gradients to Gaussian opacity.
    const auto raster_loss = [&](Device device, float opacity, bool backward) {
        GaussianModel model;
        model.means = Tensor::from_vector(std::vector<float>{0.F,0.F,2.F}, {1,3}, device);
        model.log_scales = Tensor::full({1,3}, std::log(.3F), device);
        model.quaternions = Tensor::from_vector(std::vector<float>{1.F,0.F,0.F,0.F}, {1,4}, device);
        model.opacity_logits = Tensor::full({1,1}, opacity, device);
        model.sh = Tensor::zeros({1,1,3}, device); model.sh_degree = 0;
        Camera camera;
        camera.width = camera.height = 16;
        camera.fx = camera.fy = 20.F; camera.cx = camera.cy = 7.5F;
        camera.world_to_camera = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        Rasterizer rasterizer;
        RasterizeOptions raster_options; raster_options.require_depth = false;
        const auto rendered = rasterizer.forward(model, camera, raster_options);
        TrainingView target; target.camera = camera;
        target.rgb = Tensor::full({3,16,16}, .2F, device);
        std::vector<float> half_mask(256);
        for (unsigned i=0; i<256; ++i) half_mask[i] = i%16 >= 8 ? 1.F : 0.F;
        target.mask = Tensor::from_vector(half_mask, {16,16}, device); target.has_mask = true;
        auto settings = options; settings.photometric_weight = .7F; settings.ssim_weight = 0.F;
        const auto loss = detail::compute_training_loss(rendered, target, settings, true);
        const auto rgb = loss.color.to_vector();
        for (unsigned i=0; i<256; ++i) if (half_mask[i] == 0.F)
            require(rgb[i] == 0.F && rgb[256+i] == 0.F && rgb[512+i] == 0.F,
                "Transparent mode must still mask RGB loss on background rays");
        require(loss.rgb > 0.F && loss.alpha_value > 0.F &&
            std::abs(loss.total-loss.rgb-loss.alpha_value) < 1e-6F,
            "Vulkan transparent total omitted RGB or alpha loss");
        float opacity_gradient = 0.F;
        if (backward) opacity_gradient = rasterizer.backward(model, rendered,
            loss.color, loss.alpha, {}, {}).opacity_logits.to_vector()[0];
        return std::pair{loss.total, opacity_gradient};
    };
    const auto cuda_raster = raster_loss(Device::CUDA, .4F, true);
    const auto vk_raster = raster_loss(Device::Vulkan, .4F, true);
    require(std::abs(cuda_raster.first-vk_raster.first) < 2e-4F &&
        std::abs(cuda_raster.second-vk_raster.second) < 2e-4F,
        "Transparent RGB/alpha raster backward differs between CUDA and Vulkan");
    const float raster_numerical = (raster_loss(Device::Vulkan, .401F, false).first -
        raster_loss(Device::Vulkan, .399F, false).first) / .002F;
    require(std::abs(vk_raster.second-raster_numerical) < 5e-4F,
        "Vulkan transparent opacity gradient failed end-to-end finite differences");
    std::cout << "Vulkan transparent RGB/alpha raster backward parity passed\n";
#endif
}

void test_vulkan_transparent_trainer() {
#ifdef TINYTENSOR_HAS_VULKAN
    using namespace photara::splat;
    constexpr unsigned side = 32;
    const auto root = std::filesystem::temp_directory_path() /
        ("photara_transparent_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    photara::mvs::MvsScene scene;
    for (unsigned i = 0; i < 3; ++i) {
        photara::mvs::MvsView view;
        view.id = i; view.path = root / (std::to_string(i)+".png");
        photara::io::save_rgb_png({side,side,std::vector<std::uint8_t>(3*side*side,128)}, view.path);
        view.width = view.src_width = view.height = view.src_height = side;
        view.fx = view.fy = view.src_fx = view.src_fy = 35.F;
        view.cx = view.cy = view.src_cx = view.src_cy = 15.5F;
        view.pose.C.x() = .1 * i;
        view.foreground_mask.resize(side*side);
        for (unsigned y=0; y<side; ++y) for (unsigned x=0; x<side; ++x)
            view.foreground_mask[y*side+x] = x >= side/2 ? 255 : 0;
        scene.views.push_back(view);
    }
    for (float x : {-.35F, .35F}) {
        photara::mvs::DensePoint p;
        p.position = {x,0.F,2.F}; p.normal = photara::mvs::Vec3f::UnitZ();
        p.color = photara::mvs::Vec3f::Constant(.5F); p.views = {0,1,2};
        scene.dense_cloud.points.push_back(p);
    }
    TrainingOptions options;
    options.backend = TrainingBackend::vulkan;
    options.iterations = 30; options.log_interval = 1;
    options.sh_degree = 0; options.input_is_dense = false;
    options.initial_opacity = .5F; options.initial_scale = .25F;
    options.use_mask = true; options.alpha_mode = AlphaMode::transparent;
    options.photometric_weight = 0.F; options.match_alpha_weight = .25F;
    options.enable_densification = false; options.mean_noise_weight = 0.F;
    options.means_lr = options.scales_lr = options.quaternions_lr = 0.F;
    options.sh0_lr = options.sh_rest_lr = 0.F; options.opacities_lr = .1F;
    options.use_depth_normal_loss = false; options.use_normal_field = false;
    options.multi_view_geo_weight = options.multi_view_ncc_weight = 0.F;
    options.lambda_d = options.lambda_n = 0.F;
    options.use_mvs_depth = options.use_mvs_normals = false;
    options.constrain_scale_range = false;
    float first = -1.F, last = -1.F;
    unsigned completed = 0;
    const auto model = Trainer(options).train(scene, [&](const TrainingProgress& p) {
        require(std::isfinite(p.alpha_loss) && p.alpha_loss > 0.F,
            "Vulkan trainer did not report transparent mask supervision");
        if (first < 0.F) first = p.alpha_loss;
        last = p.alpha_loss; completed = p.iteration;
        return true;
    });
    const auto opacity = model.opacity_logits.to_vector();
    require(completed == options.iterations && last < first,
        "Vulkan transparent training did not reduce alpha loss without geometry");
    require(opacity.size() == 2 && opacity[0] < 0.F && opacity[1] > 0.F,
        "Vulkan alpha backward did not remove background and retain foreground");
    for (const auto& view : scene.views) std::filesystem::remove(view.path);
    std::filesystem::remove(root);
    std::cout << "Vulkan transparent trainer alpha_loss " << first << " -> " << last
              << " background_logit=" << opacity[0] << " foreground_logit=" << opacity[1] << '\n';
#endif
}
