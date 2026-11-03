// End-to-end finite differences exercise the cap through projection, parameter
// activation and the actual compositor, including native fisheye duals.
inline void test_footprint_gradients(tinytensor::Device device) {
    using namespace photara::splat;
    using tinytensor::Tensor;
    Rasterizer rasterizer;
    for (const auto mode : {photara::CameraModel::pinhole,
                           photara::CameraModel::opencv_fisheye,
                           photara::CameraModel::equirectangular}) {
        Camera camera;
        camera.world_to_camera = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        camera.width = mode == photara::CameraModel::equirectangular ? 64 : 32;
        camera.height = 32; camera.model = mode;
        camera.fx = camera.fy = mode == photara::CameraModel::equirectangular
            ? 32.F/3.141592653589793F : 20.F;
        camera.cx = camera.width*.5F; camera.cy = 16.F;
        const std::size_t area = camera.width*camera.height;
        for (bool capped : {false,true}) {
            const float size = capped ? (mode == photara::CameraModel::equirectangular ? 8.F : 2.F) : .06F;
            std::vector<float> p{.23F,.17F,2.2F,std::log(size),std::log(size*.6F),std::log(size*.3F),.9F,.1F,.2F,-.1F,.4F};
            const auto make = [&](const std::vector<float>& values) {
                GaussianModel m;
                m.means=Tensor::from_vector(std::vector<float>(values.begin(),values.begin()+3),{1,3},device);
                m.log_scales=Tensor::from_vector(std::vector<float>(values.begin()+3,values.begin()+6),{1,3},device);
                m.quaternions=Tensor::from_vector(std::vector<float>(values.begin()+6,values.begin()+10),{1,4},device);
                m.opacity_logits=Tensor::full({1,1},values[10],device);
                m.sh=Tensor::zeros({1,1,3},device);
                return m;
            };
            const unsigned px = mode == photara::CameraModel::equirectangular ? 33 : 18;
            const unsigned py = mode == photara::CameraModel::equirectangular ? 17 : 18;
            const std::size_t pixel = py*camera.width+px;
            std::vector<float> loss(3*area,0.F);loss[pixel]=1.F;
            RasterizeOptions options; options.require_depth=false;
            options.kernel_size=.3F;options.rasterization.compensate_2d=false;
            auto m=make(p);
            auto frame=rasterizer.forward(m,camera,options);
            const int radius=frame.radii.to_vector()[0];
            require(capped ? radius == 34 : radius < 34,
                    "Footprint finite-difference fixture did not exercise the requested cap branch");
            require(frame.color.to_vector()[pixel] > .005F,
                    "Footprint finite-difference fixture has no useful image gradient");
            const auto g=rasterizer.backward(m,frame,Tensor::from_vector(loss,{3,camera.height,camera.width},device),
                Tensor::zeros({camera.height,camera.width},device),{},{});
            std::vector<float> analytic;
            for (const auto* tensor : {&g.means,&g.log_scales,&g.quaternions,&g.opacity_logits}) {
                const auto values=tensor->to_vector();analytic.insert(analytic.end(),values.begin(),values.end());
            }
            for (std::size_t k=0;k<p.size();++k) {
                constexpr float step=.002F;
                auto plus=p,minus=p;plus[k]+=step;minus[k]-=step;
                const auto sample=[&](const std::vector<float>& values) {
                    return rasterizer.forward(make(values),camera,options).color.to_vector()[pixel];
                };
                const float numerical=(sample(plus)-sample(minus))/(2*step);
                if (!std::isfinite(analytic[k]) || std::abs(analytic[k]-numerical)>.001F+.02F*std::abs(numerical)) {
                    std::cerr << "Footprint gradient mode=" << int(mode) << " capped=" << capped
                              << " parameter=" << k << " analytic=" << analytic[k] << " numerical=" << numerical << '\n';
                    throw std::runtime_error("Footprint gradient differs from finite differences");
                }
            }
        }
    }
}

// Independent single-pixel projection/compositing oracles for raster policy.
inline void test_raster_policy(tinytensor::Device device) {
    test_footprint_gradients(device);
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
          "Standard alpha ceiling changed");
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
          "Standard near plane failed to cull a near Gaussian");
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
    // min(width,height)^2/8 = 128 caps both isotropic footprints.
    const float a = .6F * std::exp(-.5F * 15.F*15.F / 128.F);
    // Red has smaller camera Z but greater radial distance (2.5 vs 2.2).
    // Blue must occlude red under the shared training/preview sort policy.
    rendered = rasterizer.forward(model, camera, options);
    const auto radial_sorted = rendered.color.to_vector();
    close(radial_sorted[center], (1-.6F)*a, "Radial sorting did not place the blue Gaussian first");
    close(radial_sorted[2*32*32+center], .6F, "Radial compositing differs from oracle");
    std::vector<float> red_loss(3*32*32, 0.F);
    red_loss[center] = 1.F;
    const auto sorted_grad = rasterizer.backward(model, rendered,
        Tensor::from_vector(red_loss, {3,32,32}, device),
        Tensor::zeros({32,32}, device), {}, {}).colors_precomp.to_vector();
    close(sorted_grad[0], (1-.6F)*a, "Backward did not retain radial-order red visibility");
    close(sorted_grad[3], .6F, "Backward did not retain radial-order blue visibility");

    camera.model = photara::CameraModel::opencv_fisheye;
    const float theta = std::atan2(1.5F, 2.F);
    // The fisheye transverse eigenvalue is larger than the horizontal one.
    const float fish_vertical_variance = std::pow(40.F*theta/1.5F,2.F);
    const float fish_horizontal_variance = 256.F*(128.F/fish_vertical_variance);
    const float fisheye_alpha = .6F * std::exp(-.5F * 20.F*20.F * theta*theta / fish_horizontal_variance);
    const auto fish_sorted = rasterizer.forward(model, camera, options).color.to_vector();
    close(fish_sorted[center], (1-.6F)*fisheye_alpha, "Fisheye radial order differs from oracle");
    close(fish_sorted[2*32*32+center], .6F, "Fisheye did not place the radially nearer Gaussian first");
    camera.model = photara::CameraModel::pinhole;
    camera.orthographic = true;
    const float ortho_alpha = .6F * std::exp(-.5F * 30.F*30.F / 128.F);
    const auto ortho_sorted = rasterizer.forward(model, camera, options).color.to_vector();
    close(ortho_sorted[center], ortho_alpha, "Orthographic rays must retain camera-Z order");
    close(ortho_sorted[2*32*32+center], (1-ortho_alpha)*.6F, "Orthographic compositing differs from oracle");
    camera.orthographic = false;
    Camera second = camera;
    camera.fx = camera.fy = 100.F;
    second.fx = second.fy = 1000.F;
    second.world_to_camera[14] = 9.F;
    const auto point = Tensor::from_vector(std::vector<float>{0,0,1}, {1,3}, device);
    const auto floor = detail::compute_3d_filter(point, {camera,second}, .1F).to_vector()[0];
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
    if (!(std::abs(sampled[2]-depth) < .003F &&
          std::abs(sampled[0]-.025F*sampled[2]) < 1e-5F &&
          std::abs(sampled[1]-.025F*sampled[2]) < 1e-5F))
        std::cerr << "Point-query device=" << int(device) << " rendered=" << depth
                  << " sampled=" << sampled[0] << ',' << sampled[1] << ',' << sampled[2] << '\n';
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
        require(raster.rasterization.alpha_clip == .99F &&
                raster.rasterization.near_plane == .2F &&
                raster.rasterization.pixel_center_offset == 0.F,
                "Filter mode unexpectedly changed the standard raster policy");
    }
}
