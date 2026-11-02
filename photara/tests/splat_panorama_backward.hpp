#pragma once

// Native RGB uses spherical derivatives even when camera-space Z is zero.
// The unused perspective geometry chain must not inject 0 * infinity.
void test_panorama_side_ray_backward() {
    using namespace photara::splat;
    using tinytensor::Tensor;
    const auto device = tinytensor::Device::CUDA;
    Camera camera;
    camera.model = photara::CameraModel::equirectangular;
    camera.width = 64; camera.height = 32;
    camera.fx = 64.F / (2.F * std::numbers::pi_v<float>);
    camera.fy = 32.F / std::numbers::pi_v<float>;
    camera.cx = 32.F; camera.cy = 16.F;
    camera.world_to_camera = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    RasterizeOptions options;
    options.require_depth = false;
    options.kernel_size = .3F;
    options.rasterization.compensate_2d = false;
    options.background = {.11F,.07F,.17F};
    options.colors_precomp = Tensor::from_vector(
        std::vector<float>{.8F,.2F,.1F,.1F,.4F,.7F},{2,3},device);
    Rasterizer rasterizer;
    for (float z : {0.F, 1e-5F, -1e-5F}) {
        std::vector<float> means{1.2F,.07F,z,2.16F,.126F,z*1.8F};
        auto make_model = [&](const std::vector<float>& positions) {
            GaussianModel model;
            model.means = Tensor::from_vector(positions,{2,3},device);
            model.log_scales = Tensor::from_vector(std::vector<float>{
                std::log(.25F),std::log(.14F),std::log(.19F),
                std::log(.4F),std::log(.3F),std::log(.35F)},{2,3},device);
            model.quaternions = Tensor::from_vector(
                std::vector<float>{.94F,.13F,-.18F,.21F,1,0,0,0},{2,4},device);
            model.opacity_logits = Tensor::from_vector(std::vector<float>{.7F,.3F},{2,1},device);
            model.sh = Tensor::zeros({2,1,3},device); model.sh_degree = 0;
            return model;
        };
        auto model = make_model(means);
        auto frame = rasterizer.forward(model,camera,options);
        const auto alpha = frame.alpha.to_vector();
        const auto pixel = std::max_element(alpha.begin(),alpha.end()) - alpha.begin();
        require(alpha[pixel] > .1F,"Side-ray regression needs visible splats");
        constexpr std::size_t pixels = 64 * 32;
        for (bool active : {false,true}) {
            std::vector<float> upstream(3*pixels,0.F);
            if (active) { upstream[pixel]=.7F; upstream[pixels+pixel]=-.3F; upstream[2*pixels+pixel]=.2F; }
            auto gradients = rasterizer.backward(model,frame,
                Tensor::from_vector(upstream,{3,32,64},device),Tensor::zeros({32,64},device),{},{});
            for (const auto* tensor : {&gradients.means,&gradients.log_scales,&gradients.quaternions,&gradients.opacity_logits})
                for (float value : tensor->to_vector())
                    require(std::isfinite(value) && (active || value==0.F),
                        "Panorama side-ray RGB backward must be finite; zero upstream must remain zero");
            if (!active) continue;
            const auto analytic = gradients.means.to_vector();
            auto objective = [&](const std::vector<float>& positions) {
                const auto rgb = rasterizer.forward(make_model(positions),camera,options).color.to_vector();
                return .7F*rgb[pixel]-.3F*rgb[pixels+pixel]+.2F*rgb[2*pixels+pixel];
            };
            for (int axis=0;axis<3;++axis) {
                auto plus=means,minus=means;
                constexpr float step=2e-4F;
                plus[axis]+=step; minus[axis]-=step;
                const float numerical=(objective(plus)-objective(minus))/(2*step);
                require(std::abs(analytic[axis]-numerical)<.003F+.025F*std::abs(numerical),
                    "Panorama side-ray mean gradient must match RGB finite differences");
            }
        }
    }
    std::cout << "Panorama side-ray backward tests passed\n";
}
