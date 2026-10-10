// Included by splat_test.cpp, using its require() helper and test headers.
void test_vulkan_rings_preview() {
    using namespace photara::splat;
    using tinytensor::Device;
    const auto tensor = [](std::vector<float> data, tinytensor::TensorShape shape) {
        return tinytensor::Tensor::from_vector(data, shape, Device::Vulkan);
    };
    GaussianModel model;
    model.means = tensor({0,0,3}, {1,3});
    model.log_scales = tensor({std::log(.4F),std::log(.4F),std::log(.04F)}, {1,3});
    model.quaternions = tensor({1,0,0,0}, {1,4});
    model.opacity_logits = tensor({10}, {1,1});
    model.sh = tensor({1,1,1}, {1,1,3}); model.sh_degree = 0;
    Camera camera;
    camera.world_to_camera = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    camera.width = camera.height = 96;
    camera.fx = camera.fy = 64; camera.cx = camera.cy = 48;
    const auto means_before = model.means.to_vector();
    const auto rgb = visualize(model, camera).to_vector();
    VisualizeOptions diagnostic;
    diagnostic.channel=VisualizationChannel::depth;
    diagnostic.automatic_depth=false; diagnostic.grayscale_depth=true;
    diagnostic.depth_near=0; diagnostic.depth_far=6;
    const auto depth_image=visualize(model,camera,diagnostic).to_vector();
    const auto center=48*96+48;
    require(std::abs(depth_image[center]-.5F)<.01F && depth_image[0]==0,
            "Live depth must map actual median depth and keep empty pixels black");
    diagnostic.automatic_depth=true; diagnostic.grayscale_depth=false;
    const auto automatic_depth=visualize(model,camera,diagnostic).to_vector();
    require(automatic_depth[center]>.85F && automatic_depth[center+2*96*96]>.5F,
            "Automatic depth must handle a constant-depth surface");
    diagnostic.channel=VisualizationChannel::normal;
    const auto normals=visualize(model,camera,diagnostic).to_vector();
    require(std::abs(normals[center]-.5F)<.01F && std::abs(normals[center+96*96]-.5F)<.01F &&
            (normals[center+2*96*96]<.01F || normals[center+2*96*96]>.99F),
            "Live normal must encode the Gaussian footprint direction");
    for(const auto& frame : {depth_image,automatic_depth,normals})
        for(float value : frame) require(std::isfinite(value)&&value>=0&&value<=1,"Diagnostic range");
    Camera tilted=camera;
    const float c=std::sqrt(.5F),s=c;
    tilted.world_to_camera={c,0,-s,0, 0,1,0,0, s,0,c,0, -3*s,0,3-3*c,1};
    tilted.position={3*s,0,3-3*c};
    diagnostic.world_normals=false;
    const auto camera_normal=visualize(model,tilted,diagnostic).to_vector();
    diagnostic.world_normals=true;
    const auto world_normal=visualize(model,tilted,diagnostic).to_vector();
    require(std::abs(world_normal[center]-.5F)<.02F &&
            std::abs(camera_normal[center]-world_normal[center])>.2F,
            "Live world normals must apply the inverse camera rotation");
    int cuda_devices=0;
    if(cudaGetDeviceCount(&cuda_devices)==cudaSuccess && cuda_devices>0) {
        GaussianModel cuda_model=model;
        cuda_model.means=model.means.to(Device::CUDA);
        cuda_model.log_scales=model.log_scales.to(Device::CUDA);
        cuda_model.quaternions=model.quaternions.to(Device::CUDA);
        cuda_model.opacity_logits=model.opacity_logits.to(Device::CUDA);
        cuda_model.sh=model.sh.to(Device::CUDA);
        const auto cuda_normal=visualize(cuda_model,tilted,diagnostic).to_vector();
        for(unsigned channel=0;channel<3;++channel)
            require(std::abs(cuda_normal[center+channel*96*96]-world_normal[center+channel*96*96])<.02F,
                    "CUDA/Vulkan normal previews must agree");
        diagnostic.channel=VisualizationChannel::depth;
        diagnostic.automatic_depth=false;diagnostic.grayscale_depth=true;
        const auto cuda_depth=visualize(cuda_model,camera,diagnostic).to_vector();
        require(std::abs(cuda_depth[center]-.5F)<.01F,"CUDA diagnostic depth conversion");
        diagnostic.automatic_depth=true;
        const auto cuda_auto=visualize(cuda_model,camera,diagnostic).to_vector();
        require(cuda_auto[center]>.99F && cuda_auto[0]==0,"CUDA automatic depth conversion");
    }
    VisualizeOptions vis; vis.mode = VisualizationMode::rings;
    vis.ring_scale = 1;
    const auto small = visualize(model, camera, vis).to_vector();
    vis.ring_scale = 4;
    const auto large = visualize(model, camera, vis).to_vector();
    require(rgb.size() == 3*96*96 && small.size() == rgb.size(), "Vulkan Rings image shape");
    std::size_t small_support = 0, large_support = 0;
    for (std::size_t i=0; i<96*96; ++i) {
        require(std::isfinite(large[i]) && large[i] >= 0 && large[i] <= 1, "Rings pixel range");
        small_support += small[i] > .001F;
        large_support += large[i] > .001F;
    }
    require(small != rgb && large_support > small_support*4, "Ring scale must grow the projected ellipse");
    // The rim is flat alpha=0.6; at sigma=4, a pixel near 34px lies on it.
    require(large[48*96+82] > large[48*96+80], "Vulkan Rings must draw a distinct outer rim");
    require(visualize(model, camera).to_vector() == rgb && model.means.to_vector() == means_before,
            "Preview must not mutate the model or the ordinary RGB render");

    // Exercise the actual trainer callback and visualization sidecar switches.
    const auto root = std::filesystem::temp_directory_path()/"photara_vulkan_rings_preview_test";
    std::filesystem::create_directories(root);
    photara::mvs::MvsScene scene;
    photara::io::RgbImage image{32,32,std::vector<std::uint8_t>(3*32*32,128)};
    photara::mvs::MvsView view;
    view.id=0; view.path=root/"input.png";
    photara::io::save_rgb_png(image,view.path);
    view.width=view.src_width=view.height=view.src_height=32;
    view.fx=view.fy=view.src_fx=view.src_fy=40;
    view.cx=view.cy=view.src_cx=view.src_cy=15.5F;
    scene.views.push_back(view);
    for(float x : {-.2F,.2F}) for(float y : {-.2F,.2F}) {
        photara::mvs::DensePoint p;
        p.position={x,y,2.F}; p.normal=photara::mvs::Vec3f::UnitZ();
        p.color=photara::mvs::Vec3f::Constant(.5F); p.views={0};
        scene.dense_cloud.points.push_back(p);
    }
    TrainingOptions options;
    options.backend=TrainingBackend::vulkan; options.iterations=5; options.sh_degree=0;
    options.initial_opacity=.8F; options.initialize_scale_from_knn=true;
    options.initial_scale=.3F; options.max_image_dimension=32;
    options.constrain_scale_range=false;
    options.mean_noise_weight=0; options.refine_start_iter=100;
    options.log_interval=1; options.preview_interval=1;
    options.preview_vis_file=root/"preview.vis";
    vis.mode=VisualizationMode::rings; vis.ring_scale=2.828427F;
    require(write_visualization_sidecar(options.preview_vis_file,vis,1), "Write Rings sidecar");
    unsigned callbacks=0;
    std::vector<std::vector<float>> frames;
    const auto trained = Trainer(options).train(scene,[&](const TrainingProgress& p) {
        require(std::isfinite(p.loss), "Training with Rings preview produced invalid loss");
        vis.mode=p.iteration==4 ? VisualizationMode::rings : VisualizationMode::splat;
        vis.channel=p.iteration==2 ? VisualizationChannel::depth
            : p.iteration==3 ? VisualizationChannel::normal : VisualizationChannel::color;
        require(write_visualization_sidecar(options.preview_vis_file,vis,p.iteration+1), "Switch live visualization");
        return true;
    }, {}, {}, [&](unsigned, std::size_t, const Camera& c, const tinytensor::Tensor& color) {
        require(color.device()==Device::Vulkan && color.numel()==3*c.width*c.height,
                "Live Rings frame must stay on the Vulkan device");
        ++callbacks; frames.push_back(color.to_vector());
        std::cout << "preview callback=" << callbacks << " size=" << color.numel()
                  << " max=" << *std::max_element(frames.back().begin(),frames.back().end()) << '\n';
    });
    require(trained.size()==4 && callbacks==5, "Vulkan training must deliver every preview frame");
    for(std::size_t i=1;i<frames.size();++i)
        require(frames[i]!=frames[i-1], "Live Rings/Color/Depth/Normal switching must update frames");
    VisualizeOptions loaded; std::uint64_t revision{};
    diagnostic.depth_near=.2F;diagnostic.depth_far=9;diagnostic.world_normals=false;
    require(write_visualization_sidecar(options.preview_vis_file,diagnostic,23)&&
            load_visualization_sidecar(options.preview_vis_file,loaded,revision)&&revision==23&&
            loaded.channel==diagnostic.channel&&loaded.depth_near==.2F&&loaded.depth_far==9&&
            !loaded.world_normals,"Diagnostic sidecar must preserve channel and display settings");
    std::ofstream(options.preview_vis_file)<<"24\nsplat\n2.5\n2.828427\n1 1";
    require(!load_visualization_sidecar(options.preview_vis_file,loaded,revision)&&revision==23&&
            loaded.channel==diagnostic.channel,"Partial sidecar must preserve the last request");
    std::ofstream(options.preview_vis_file)<<"24\nsplat\n2.5\n2.828427\n";
    require(load_visualization_sidecar(options.preview_vis_file,loaded,revision)&&
            loaded.channel==VisualizationChannel::color,"Legacy sidecar must restore color");
    std::cout << "Vulkan Rings, Depth, Normal and live training channel switches passed\n";
}
