// Included in the manual parity tool's anonymous namespace.
// Uses an immutable trained model; never grows, updates, or saves parameters.
int run_geometry_benchmark(int argc, char** argv) {
    using namespace photara::splat;
    using tinytensor::Device;
    using tinytensor::Tensor;
    require(argc>=6,"Usage: --bench-geometry MODEL IMAGES OUTPUT SFM [ITERS] [VIEW] [NF] [SAVE_GRADIENTS]");
    const std::filesystem::path output(argv[4]);
    std::filesystem::create_directories(output);
    const auto views=load_views(argv[3],argv[5]);
    const int iterations=argc>6?std::max(1,std::stoi(argv[6])):5;
    const std::size_t index=argc>7?std::stoul(argv[7]):0;
    const bool field=argc>8?std::stoi(argv[8])!=0:true;
    const bool save_gradients=argc>9?std::stoi(argv[9])!=0:false;
    require(index<views.size(),"Benchmark view outside dataset");
    std::ofstream csv(output/("geometry_view_"+std::to_string(index)+(field?"_nf.csv":".csv")));
    csv<<"backend,iteration,phase,ms\n";
    const std::vector<std::string> phases={"forward","photo_loss","DN_loss","unproject",
        "sample_forward","MV_loss","sample_backward","point_gradient","primary_backward",
        "NF_forward","NF_loss","NF_backward","merge","total"};
    for (const auto device : {Device::CUDA,Device::Vulkan}) {
        const bool vk=device==Device::Vulkan;
        const char* name=vk?"Vulkan":"CUDA";
        auto model=load_gaussians(argv[2],device);
        TrainingOptions options;
        options.backend=vk?TrainingBackend::vulkan:TrainingBackend::cuda;
        options.use_mask=false;
        options.use_source_resolution=true;
        options.max_image_dimension=1920;
        options.ignore_undistortion_border=true;
        options.progressive_resolution=false;
        options.use_depth_normal_loss=true;
        options.depth_normal_weight=.05F;
        options.multi_view_geo_weight=.02F;
        options.multi_view_ncc_weight=.6F;
        const auto reference=make_training_view(views[index],options);
        std::vector<Camera> cameras;
        std::vector<std::size_t> indices;
        for (std::size_t i=0;i<views.size();++i) {
            cameras.push_back(training_data::training_camera(views[i],options));
            indices.push_back(i);
        }
        const auto neighbours=training_data::compute_multi_view_neighbours(cameras,indices,options);
        require(!neighbours[index].empty(),"Benchmark has no neighbour");
        const auto neighbour=make_training_view(views[neighbours[index].front()],options);
        if (field) require(model.normal_features.is_valid(),"NF benchmark requires saved normal features");
        Rasterizer rasterizer;
        RasterizeOptions ro;
        ro.active_sh_degree=model.sh_degree;
        ro.require_depth=true;
        ro.copy_attachments=!vk;
        ro.defer_visibility=vk;
        const auto sync=[&] {
            if(vk) (void)Tensor::zeros({1},Device::Vulkan).to_vector();
            else require(cudaDeviceSynchronize()==cudaSuccess,"CUDA benchmark synchronization failed");
        };
        sync();
        std::vector<std::vector<double>> samples(phases.size());
        std::cout<<name<<" geometry benchmark gaussians="<<model.size()<<" view="<<index
                 <<" neighbour="<<neighbours[index].front()<<" size="<<reference.camera.width
                 <<'x'<<reference.camera.height<<" NF="<<field<<std::endl;
        for (int iteration=-2;iteration<iterations;++iteration) {
            std::vector<double> times(phases.size(),0.);
            std::size_t stage=0;
            const auto total_start=std::chrono::steady_clock::now();
            auto start=total_start;
            const auto mark=[&] {
                sync();
                const auto now=std::chrono::steady_clock::now();
                times[stage++]=std::chrono::duration<double,std::milli>(now-start).count();
                start=now;
            };
            auto rendered=rasterizer.forward(model,reference.camera,ro);mark();
            detail::LossGradients loss;
            if(vk) {
                loss.total=rasterizer.photometric_loss(rendered,reference.rgb,reference.mask,false,
                    options.ssim_weight,options.photometric_weight,false);
                loss.alpha=rendered.alpha.is_valid()?Tensor::zeros_like(rendered.alpha):Tensor{};
            } else loss=detail::compute_training_loss(rendered,reference,options,false,true,true);
            mark();
            if(vk) detail::add_depth_normal_loss_vulkan(rendered,reference.camera,.05F,loss,false);
            // CUDA computes DN in its combined training loss above.
            mark();
            const auto points=detail::unproject_depth_to_world(rendered.median_depth,reference.camera);mark();
            const auto sampled=rasterizer.sample_depth(model,points,neighbour.camera,ro);mark();
            Tensor point_gradient;
            detail::add_multi_view_loss(sampled.camera_points,sampled.inside,rendered,reference,
                neighbour,options,loss,point_gradient,false);mark();
            const auto sample_gradients=rasterizer.sample_depth_backward(model,sampled,point_gradient);mark();
            detail::add_sample_depth_point_gradients(reference.camera,sample_gradients.points,loss);mark();
            auto gradients=rasterizer.backward(model,rendered,loss.color,loss.alpha,loss.depth,loss.normal);mark();
            rendered.context.impl.reset();
            ModelGradients field_gradients;
            if(field) {
                auto normal_options=ro;
                normal_options.copy_attachments=true;
                normal_options.copy_color_only=false;
                normal_options.colors_precomp=detail::normal_features_to_normals(model.normal_features);
                const auto normal_render=rasterizer.forward(model,reference.camera,normal_options);mark();
                const auto normal_loss=detail::compute_normal_field_loss(normal_render,reference.camera,.03F,false);mark();
                field_gradients=rasterizer.backward(model,normal_render,normal_loss.color,
                    normal_loss.alpha,normal_loss.depth,normal_loss.normal);
                field_gradients.normal_features=detail::normal_features_backward(model.normal_features,
                    field_gradients.colors_precomp);mark();
            } else stage+=3;
            detail::add_sample_depth_model_gradients(sample_gradients,gradients);
            if(field) detail::add_model_gradients(field_gradients,gradients,false);
            mark();
            times.back()=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-total_start).count();
            if(iteration>=0) for(std::size_t i=0;i<times.size();++i) {
                csv<<name<<','<<iteration<<','<<phases[i]<<','<<times[i]<<'\n';
                samples[i].push_back(times[i]);
            }
            // Optional correctness capture stays outside all timing windows.
            // Compare this Vulkan capture with DISABLE_SUBGROUP_BACKWARD=1
            // on the same immutable model to isolate the optimized kernel.
            if(save_gradients && iteration==iterations-1) {
                const auto query_inside=sampled.inside.to(tinytensor::DataType::Float32);
                const std::pair<const char*,const Tensor*> groups[]={
                    {"means",&gradients.means},{"scales",&gradients.log_scales},
                    {"rotations",&gradients.quaternions},{"opacity",&gradients.opacity_logits},
                    {"sh",&gradients.sh},{"normal_features",&field_gradients.normal_features},
                    {"refine_weight",&gradients.refine_weight},
                    {"query_points",&sampled.camera_points},{"query_inside",&query_inside}};
                for(const auto& [label,tensor]:groups) {
                    if(!tensor->is_valid()) continue;
                    const auto values=tensor->to_vector();
                    require(std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);}),
                        std::string("Non-finite geometry gradient: ")+label);
                    std::ofstream file(output/(std::string(name)+"_"+label+".f32"),std::ios::binary);
                    file.write(reinterpret_cast<const char*>(values.data()),
                        static_cast<std::streamsize>(values.size()*sizeof(float)));
                    require(file.good(),"Failed to save geometry gradient capture");
                }
            }
        }
        for(std::size_t i=0;i<phases.size();++i)
            std::cout<<name<<" phase="<<phases[i]<<" median_ms="<<median_ms(samples[i])<<std::endl;
    }
    return 0;
}
