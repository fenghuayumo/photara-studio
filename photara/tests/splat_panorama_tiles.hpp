#pragma once

// Regression for duplicate seam-wrapped instances. One Gaussian cannot have
// alpha above its activated opacity; its opacity-logit VJP is alpha*(1-o).
inline void panorama_single_contributor_regression() {
    using tinytensor::Device;
    using tinytensor::Tensor;
    for (auto device : {Device::CUDA, Device::Vulkan})
    for (auto method : {RenderMethod::gaussian_3dgs, RenderMethod::gaussian_3dgut})
    for (const unsigned width : {7U, 35U, 96U})
    for (const bool compact : {false, true})
    for (const bool pole : {false, true}) {
        Camera camera;
        camera.width=width;camera.height=64;camera.model=CameraModel::equirectangular;
        camera.fx=width/(2.F*3.14159265359F);camera.fy=64.F/3.14159265359F;
        camera.cx=width*.5F;camera.cy=32;
        camera.world_to_camera={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
        auto seed=gut_validation::model(device,false,1);
        seed.render_method=method;
        seed.means=Tensor::from_vector(pole ? std::vector<float>{0,2,.02F}
                                            : std::vector<float>{.01F,.1F,-2},{1U,3U},device);
        const float scale=pole ? .2F : .7F;
        seed.log_scales=Tensor::from_vector(std::vector<float>(3,std::log(scale)),{1U,3U},device);
        seed.quaternions=Tensor::from_vector(std::vector<float>{1,0,0,0},{1U,4U},device);
        seed.opacity_logits=Tensor::from_vector(std::vector<float>{std::log(9.F)},{1U,1U},device);
        RasterizeOptions options;options.require_depth=false;
        options.compact_instance_storage=compact;
        options.rasterization.compensate_2d=false;
        Rasterizer raster;
        auto frame=raster.forward(seed,camera,options);
        const auto alpha=frame.alpha.to_vector();
        const auto peak=std::max_element(alpha.begin(),alpha.end());
        std::cout<<"Panorama single contributor backend="<<int(device)<<" method="<<int(method)<<" width="<<width
                 <<" compact="<<compact<<" pole="<<pole<<" peak="<<*peak<<std::endl;
        // A narrow projected pole ellipse can have no pixel near its center.
        // Only require a visible contribution, not a resolution-dependent peak.
        require(*peak>=1.F/255.F && *peak<=.90001F,"Panorama repeats a single contributor's opacity");
        require(frame.rendered_instances<=((width+15)/16)*4U,"Panorama emits a Gaussian/tile pair more than once");
        const auto pixel=std::size_t(peak-alpha.begin());
        std::vector<float> da(width*64,0.F);da[pixel]=1.F;
        const auto gradients=raster.backward(seed,frame,Tensor::zeros({3U,64U,width},device),
            Tensor::from_vector(da,{64U,width},device),{},{});
        const float actual=gradients.opacity_logits.to_vector()[0];
        require(std::abs(actual-*peak*.1F)<2e-5F,"Panorama single-contributor opacity gradient was duplicated");
        // Independent parameter perturbation verifies the accepted instance
        // count and backward replay agree, not just the alpha ceiling.
        constexpr float epsilon=.002F;
        auto objective=[&](float logit) {
            auto perturbed=seed;
            perturbed.opacity_logits=Tensor::from_vector(std::vector<float>{logit},{1U,1U},device);
            options.record_backward_state=false;
            return raster.forward(perturbed,camera,options).alpha.to_vector()[pixel];
        };
        const float numeric=(objective(std::log(9.F)+epsilon)-objective(std::log(9.F)-epsilon))/(2*epsilon);
        require(std::abs(actual-numeric)<2e-4F,"Panorama opacity finite difference failed");
    }
    std::cout<<"Panorama unique instances, single alpha and opacity backward passed (CUDA/Vulkan, packed/compact, partial tiles)\n";
}
