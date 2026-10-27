// Independent single-pixel projection/compositing oracles for raster policy.
inline void test_raster_policy(tinytensor::Device device) {
    using namespace photara::splat;
    using tinytensor::Tensor;
    Camera camera;
    camera.world_to_camera = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    camera.width = camera.height = 32;
    camera.fx = camera.fy = 20.F;
    camera.cx = camera.cy = 16.F;
    constexpr std::size_t center = 16 * 32 + 16;
    const auto make = [device](std::vector<float> means, float scale, float opacity) {
        GaussianModel model;
        const auto n = means.size() / 3;
        model.means = Tensor::from_vector(means, {n,3}, device);
        model.log_scales = Tensor::full({n,3}, std::log(scale), device);
        std::vector<float> quaternions(n * 4, 0.F);
        for (std::size_t i = 0; i < n; ++i) quaternions[4*i] = 1.F;
        model.quaternions = Tensor::from_vector(quaternions, {n,4}, device);
        model.opacity_logits = Tensor::full({n,1}, std::log(opacity / (1-opacity)), device);
        model.sh = Tensor::zeros({n,1,3}, device);
        return model;
    };
    const auto close = [](float actual, float expected, const char* message) {
        require(std::isfinite(actual) && std::abs(actual-expected) < 3e-5F, message);
    };
    Rasterizer rasterizer;
    RasterizeOptions options;
    options.require_depth = false;
    auto model = make({0,0,2}, .04F, .9999F);
    close(rasterizer.forward(model, camera, options).alpha.to_vector()[center], .99F,
          "Legacy alpha ceiling changed");
    options.rasterization.alpha_clip = .999F;
    close(rasterizer.forward(model, camera, options).alpha.to_vector()[center], .999F,
          "Mip Splatting preset alpha ceiling was not applied");
    options.rasterization.pixel_center_offset = .5F;
    const float expected_half = .9999F * std::exp(-.5F * (.25F+.25F) / .16F);
    close(rasterizer.forward(model, camera, options).alpha.to_vector()[center], expected_half,
          "Half-pixel sampling differs from the analytic Gaussian");
    options.rasterization.pixel_center_offset = 0.F;
    model = make({0,0,.05F}, .001F, .6F);
    close(rasterizer.forward(model, camera, options).alpha.to_vector()[center], 0.F,
          "Legacy near plane failed to cull a near Gaussian");
    options.rasterization.near_plane = .01F;
    close(rasterizer.forward(model, camera, options).alpha.to_vector()[center], .6F,
          "Mip Splatting preset near plane still uses the old projection culling threshold");
    model = make({0,0,2}, .04F, .6F);
    options.kernel_size = .3F;
    options.rasterization.compensate_2d = false;
    auto rendered = rasterizer.forward(model, camera, options);
    close(rendered.alpha.to_vector()[center], .6F,
          "Classic dilation incorrectly compensates opacity");
    // Opacity derivative has an independent sigmoid oracle; checks backward
    // receives the same filter policy as forward on both backends.
    auto color_gradient = Tensor::zeros({3,32,32}, device);
    auto alpha_data = std::vector<float>(32*32, 0.F);
    alpha_data[center] = 1.F;
    const auto gradient = rasterizer.backward(model, rendered, color_gradient,
        Tensor::from_vector(alpha_data, {32,32}, device), {}, {});
    close(gradient.opacity_logits.to_vector()[0], .24F,
          "Classic opacity backward uses Mip compensation");
    options.kernel_size = .1F;
    options.rasterization.compensate_2d = true;
    close(rasterizer.forward(model, camera, options).alpha.to_vector()[center], .6F*.16F/.26F,
          "Mip 2D determinant compensation differs from analytic projection");
    options.kernel_size = 0.F;
    model.filter_3d = Tensor::full({1,1}, .03F, device);
    close(rasterizer.forward(model, camera, options).alpha.to_vector()[center], .6F*.512F,
          "Mip 3D determinant compensation differs from analytic activation");

    model = make({1.5F,0,2, 0,0,2.2F}, 2.F, .6F);
    options.colors_precomp = Tensor::from_vector(
        std::vector<float>{1,0,0, 0,0,1}, {2,3}, device);
    const float a = .6F * std::exp(-.5F * 15.F*15.F / 625.F);
    auto radial = rasterizer.forward(model, camera, options).color.to_vector();
    close(radial[center], .4F*a, "Radial sorting did not place the blue Gaussian first");
    close(radial[2*32*32+center], .6F, "Radial sort compositing differs from oracle");
    options.rasterization.sort_by_camera_z = true;
    auto z_sorted = rasterizer.forward(model, camera, options).color.to_vector();
    close(z_sorted[center], a, "Camera-Z sorting did not place the red Gaussian first");
    close(z_sorted[2*32*32+center], (1-a)*.6F, "Camera-Z compositing differs from oracle");
    Camera second = camera;
    camera.fx = camera.fy = 100.F;
    second.fx = second.fy = 1000.F;
    second.world_to_camera[14] = 9.F;
    const auto point = Tensor::from_vector(std::vector<float>{0,0,1}, {1,3}, device);
    const auto floor = detail::compute_3d_filter(point, {camera,second}, .1F, true, true).to_vector()[0];
    close(floor, std::sqrt(.1F)*.01F,
          "3D filter combined a distance with a different camera's focal length");
    camera.fx = camera.fy = 20.F;
    model = make({0,0,2}, .1F, .9F);
    options = {};
    options.rasterization.pixel_center_offset = .5F;
    options.point_depth_bracket = .4F;
    rendered = rasterizer.forward(model, camera, options);
    const float depth = rendered.median_depth.to_vector()[center];
    require(depth > 0.F, "Half-pixel geometry render has no median surface");
    const auto query = Tensor::from_vector(std::vector<float>{.025F*depth,.025F*depth,depth}, {1,3}, device);
    const auto sampled = rasterizer.sample_depth(model,query,camera,options).camera_points.to_vector();
    require(std::abs(sampled[2]-depth) < .003F &&
            std::abs(sampled[0]-.025F*sampled[2]) < 1e-5F &&
            std::abs(sampled[1]-.025F*sampled[2]) < 1e-5F,
            "Half-pixel median render and geometric point queries use different rays");
}

inline void test_filter_mode_resolution() {
    using namespace photara::splat;
    for (auto mode : {SplatFilterMode::classic, SplatFilterMode::mip_2d,
                      SplatFilterMode::mip_3d, SplatFilterMode::mip}) {
        TrainingOptions training;
        training.filter_mode = mode;
        RasterizeOptions raster;
        configure_rasterization(raster, training);
        const bool mip2d = mode == SplatFilterMode::mip_2d || mode == SplatFilterMode::mip;
        require(raster.kernel_size == (mip2d ? .1F : .3F) &&
                raster.rasterization.compensate_2d == mip2d,
                "Explicit filter mode has inconsistent 2D settings");
    }
}
