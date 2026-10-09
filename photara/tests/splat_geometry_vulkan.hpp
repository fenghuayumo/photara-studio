// Included by splat_test.cpp so the geometry fixtures can also exercise its
// existing depth-normal finite differences and planar multi-view oracle.
void test_vulkan_geometry_chain() {
    using namespace photara::splat;
    using tinytensor::Tensor;
    using tinytensor::Device;
    constexpr std::size_t side = 32, pixels = side * side;
    Camera camera;
    camera.world_to_camera = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    camera.width = camera.height = side;
    camera.fx = camera.fy = 40.F; camera.cx = camera.cy = 15.5F;
    auto neighbour_camera = camera;
    neighbour_camera.position[0] = .1F;
    neighbour_camera.world_to_camera[12] = -.1F;
    const auto make_model = [](Device device) {
        GaussianModel m;
        m.means = Tensor::from_vector(std::vector<float>{-.08F,0.F,2.F, .10F,.02F,2.05F}, {2,3}, device);
        m.log_scales = Tensor::from_vector(std::vector<float>{
            std::log(.4F),std::log(.35F),std::log(.035F),
            std::log(.38F),std::log(.32F),std::log(.03F)}, {2,3}, device);
        m.quaternions = Tensor::from_vector(std::vector<float>{.985F,0.F,.174F,0.F, .98F,-.08F,-.17F,.015F}, {2,4}, device);
        m.opacity_logits = Tensor::from_vector(std::vector<float>{2.F,1.5F}, {2,1}, device);
        m.sh = Tensor::full({2,1,3}, .3F, device);
        m.sh_degree = 0;
        return m;
    };
    const auto make_view = [&](Device device, const Camera& c, float shift) {
        TrainingView v;
        v.camera = c;
        v.rgb = Tensor::zeros({3,side,side}, device);
        v.mask = Tensor::full({side,side}, 1.F, device);
        v.depth = Tensor::zeros({side,side}, device);
        v.normal = Tensor::zeros({3,side,side}, device);
        std::vector<float> gray(pixels);
        for (std::size_t y = 0; y < side; ++y)
            for (std::size_t x = 0; x < side; ++x)
                gray[y*side+x] = .5F+.22F*std::sin(.37F*(x+shift))+.18F*std::cos(.29F*y);
        v.gray = Tensor::from_vector(gray, {side,side}, device);
        return v;
    };
    const auto compare = [](const Tensor& a, const Tensor& b, float tolerance, const char* message) {
        const auto x = a.to_vector(), y = b.to_vector();
        require(x.size() == y.size(), message);
        double error = 0., norm = 0.;
        for (std::size_t i = 0; i < x.size(); ++i) {
            require(std::isfinite(x[i]) && std::isfinite(y[i]), message);
            error += (x[i]-y[i])*(x[i]-y[i]); norm += x[i]*x[i];
        }
        const double relative = std::sqrt(error/std::max(norm,1e-16));
        std::cout << message << " relative_l2=" << relative << '\n';
        require(relative < tolerance || std::sqrt(error) < 2e-6, message);
    };
    // Visible, behind-camera, off-screen, and all-invisible fallback radii.
    auto means = Tensor::from_vector(std::vector<float>{0,0,2, .1F,.2F,3, 100,0,2, 0,0,-2}, {4,3}, Device::CUDA);
    for (bool euclidean : {false,true}) {
        const auto a = detail::compute_3d_filter(means,{camera,neighbour_camera},.2F,euclidean);
        const auto b = detail::compute_3d_filter(means.to(Device::Vulkan),{camera,neighbour_camera},.2F,euclidean);
        compare(a,b,2e-5F,"Vulkan 3D filter parity");
    }
    auto invisible = Tensor::from_vector(std::vector<float>{0,0,-2,100,0,2}, {2,3}, Device::CUDA);
    compare(detail::compute_3d_filter(invisible,{camera}),
        detail::compute_3d_filter(invisible.to(Device::Vulkan),{camera}),2e-5F,"Vulkan invisible filter fallback");
    auto fisheye = camera;
    fisheye.model = photara::CameraModel::opencv_fisheye;
    fisheye.k1 = .08F; fisheye.k2 = -.025F;
    fisheye.k3 = .003F; fisheye.k4 = -.0002F;
    compare(detail::compute_3d_filter(means,{fisheye}),
        detail::compute_3d_filter(means.to(Device::Vulkan),{fisheye}),2e-5F,"Vulkan fisheye filter parity");
    auto panorama = camera;
    panorama.model = photara::CameraModel::equirectangular;
    compare(detail::compute_3d_filter(means,{panorama,camera}),
        detail::compute_3d_filter(means.to(Device::Vulkan),{panorama,camera}),2e-5F,"Vulkan panorama filter parity");

    struct Chain { ModelGradients total, sample; Tensor points; detail::MultiViewLoss loss; Tensor stability; };
    const auto run = [&](Device device, bool merge_sample, float photometric_weight = 0.F, bool collect_terms = true) {
        std::cout << "MV chain backend=" << (device == Device::CUDA ? "CUDA" : "Vulkan") << std::endl;
        auto model = make_model(device);
        model.filter_3d = detail::compute_3d_filter(model.means,{camera,neighbour_camera});
        auto ref = make_view(device,camera,0.F), near = make_view(device,neighbour_camera,2.F);
        RasterizeOptions ro; ro.require_depth = true;
        ro.point_depth_bracket = .2F; ro.point_depth_tolerance = 0.F;
        Rasterizer rasterizer;
        auto rendered = rasterizer.forward(model,camera,ro);
        TrainingOptions options;
        options.photometric_weight = photometric_weight; options.use_depth_normal_loss = true;
        options.depth_normal_weight = .05F;
        options.multi_view_geo_weight = .02F; options.multi_view_ncc_weight = .6F;
        auto image_gradient = detail::compute_training_loss(rendered,ref,options,collect_terms,true);
        auto world = detail::unproject_depth_to_world(rendered.median_depth,camera);
        auto sampled = rasterizer.sample_depth(model,world,neighbour_camera,ro);
        Tensor upstream;
        Chain result;
        result.stability = Tensor::zeros({3},device);
        result.loss = detail::add_multi_view_loss(sampled.camera_points,sampled.inside,rendered,
            ref,near,options,image_gradient,upstream,collect_terms,&result.stability);
        require(!collect_terms || (result.loss.geometry_pixels > 0 && result.loss.ncc_pixels > 0),
            "Vulkan geometry chain must have actual MV/NCC support");
        auto sample_grad = rasterizer.sample_depth_backward(model,sampled,upstream);
        result.sample = sample_grad.model; result.points = sample_grad.points;
        detail::add_sample_depth_point_gradients(camera,sample_grad.points,image_gradient);
        result.total = rasterizer.backward(model,rendered,image_gradient.color,image_gradient.alpha,
            image_gradient.depth,image_gradient.normal);
        if (merge_sample) detail::add_sample_depth_model_gradients(sample_grad,result.total);
        return result;
    };
    const auto cuda = run(Device::CUDA,true), vk = run(Device::Vulkan,true);
    require(cuda.loss.geometry_pixels == vk.loss.geometry_pixels && cuda.loss.ncc_pixels == vk.loss.ncc_pixels,
        "Vulkan full geometry chain changed accepted-pixel counts");
    compare(cuda.stability,vk.stability,1e-6F,"Vulkan stability accumulator");
    compare(cuda.points,vk.points,.003F,"Vulkan query point gradient");
    compare(cuda.sample.means,vk.sample.means,.001F,"Vulkan sample mean gradient");
    compare(cuda.sample.log_scales,vk.sample.log_scales,.001F,"Vulkan sample scale gradient");
    compare(cuda.sample.quaternions,vk.sample.quaternions,.001F,"Vulkan sample rotation gradient");
    compare(cuda.sample.opacity_logits,vk.sample.opacity_logits,.001F,"Vulkan sample opacity gradient");
    compare(cuda.total.means,vk.total.means,.001F,"Vulkan full mean gradient");
    compare(cuda.total.log_scales,vk.total.log_scales,.001F,"Vulkan full scale gradient");
    compare(cuda.total.quaternions,vk.total.quaternions,.001F,"Vulkan full rotation gradient");
    compare(cuda.total.opacity_logits,vk.total.opacity_logits,.001F,"Vulkan full opacity gradient");
    const auto unmerged = run(Device::Vulkan,false);
    auto difference = (vk.total.means-unmerged.total.means).to_vector();
    require(std::any_of(difference.begin(),difference.end(),[](float g){return std::abs(g)>1e-7F;}),
        "Neighbour Gaussian gradients were lost during model gradient merge");
    // Fused RGB loss and the neighbour query must preserve each other's state.
    const auto cuda_rgb = run(Device::CUDA,true,.7F), vk_rgb = run(Device::Vulkan,true,.7F);
    compare(cuda_rgb.total.means,vk_rgb.total.means,.001F,"Vulkan RGB plus MV mean gradient");
    compare(cuda_rgb.total.log_scales,vk_rgb.total.log_scales,.001F,"Vulkan RGB plus MV scale gradient");
    compare(cuda_rgb.total.quaternions,vk_rgb.total.quaternions,.001F,"Vulkan RGB plus MV rotation gradient");
    compare(cuda_rgb.total.opacity_logits,vk_rgb.total.opacity_logits,.001F,"Vulkan RGB plus MV opacity gradient");
    const auto deferred = run(Device::Vulkan,true,.7F,false);
    compare(vk_rgb.total.means,deferred.total.means,1e-5F,"Vulkan non-reporting MV gradient");
    compare(vk_rgb.stability,deferred.stability,1e-6F,"Vulkan non-reporting MV statistics");
    const auto cm = make_model(Device::CUDA), vm = make_model(Device::Vulkan);
    const auto cs = detail::summarize_geometry_distribution(cm), vs = detail::summarize_geometry_distribution(vm);
    require(std::abs(cs.log_scale_mean-vs.log_scale_mean)<1e-5F &&
            std::abs(cs.opacity_mean-vs.opacity_mean)<1e-5F &&
            std::abs(cs.log_anisotropy_stddev-vs.log_anisotropy_stddev)<1e-4F,
        "Vulkan adaptive MV geometry summary changed");
}

void test_vulkan_geometry_trainer() {
    using namespace photara::splat;
    const auto root = std::filesystem::temp_directory_path()/"photara_vulkan_geometry_smoke";
    std::filesystem::create_directories(root);
    photara::mvs::MvsScene scene;
    constexpr unsigned side = 64;
    for (unsigned i = 0; i < 3; ++i) {
        photara::io::RgbImage image{side,side,std::vector<std::uint8_t>(3*side*side)};
        for (unsigned y = 0; y < side; ++y)
            for (unsigned x = 0; x < side; ++x)
                for (unsigned c = 0; c < 3; ++c)
                    image.pixels[3*(y*side+x)+c] = static_cast<std::uint8_t>(
                        128+45*std::sin(.18F*(x+4*i))+35*std::cos(.15F*y));
        photara::mvs::MvsView view;
        view.id = i; view.path = root/("frame"+std::to_string(i)+".png");
        photara::io::save_rgb_png(image,view.path);
        view.width = view.src_width = view.height = view.src_height = side;
        view.fx = view.fy = view.src_fx = view.src_fy = 80.F;
        view.cx = view.cy = view.src_cx = view.src_cy = 31.5F;
        view.pose.C.x() = .1*i;
        scene.views.push_back(view);
    }
    for (float y : {-.3F,.3F}) for (float x : {-.3F,.3F}) {
        photara::mvs::DensePoint p;
        p.position = {x,y,2.F}; p.normal = photara::mvs::Vec3f::UnitZ();
        p.color = photara::mvs::Vec3f::Constant(.5F); p.views = {0,1,2};
        scene.dense_cloud.points.push_back(p);
    }
    TrainingOptions o;
    o.backend = TrainingBackend::vulkan; o.iterations = 10;
    o.sh_degree = 0; o.input_is_dense = false;
    o.initial_opacity = .9F;
    o.initialize_scale_from_knn = true; o.initial_scale = .3F;
    o.use_depth_normal_loss = true; o.depth_normal_from_iter = 1;
    o.multi_view_geo_weight = .02F; o.multi_view_ncc_weight = .6F;
    o.multi_view_adaptive_frequency = true; o.multi_view_adaptive_stable_refinements = 1;
    o.filter_3d_update_interval = 1;
    o.constrain_scale_range = false; o.mean_noise_weight = 0.F;
    o.densification_cap = 16; o.refine_start_iter = 1;
    o.refine_stop_iter = o.grow_stop_iter = 8; o.refine_every = 2;
    o.densify_gradient_threshold = -1.F; o.densify_select_fraction = 1.F;
    o.progressive_resolution = true; o.progressive_initial_scale = .5F;
    o.progressive_resolution_interval = 3; o.max_image_dimension = side;
    o.log_interval = 1;
    unsigned completed = 0; std::size_t mv_pixels = 0, ncc_pixels = 0;
    const auto model = Trainer(o).train(scene,[&](const TrainingProgress& p) {
        ++completed; mv_pixels += p.multi_view_geometry_pixels; ncc_pixels += p.multi_view_ncc_pixels;
        std::cout << "geometry trainer iter=" << p.iteration << " gaussians=" << p.gaussian_count
                  << " mv=" << p.multi_view_geometry_pixels << " ncc=" << p.multi_view_ncc_pixels << '\n';
        require(std::isfinite(p.loss),"Vulkan geometry trainer produced non-finite loss");
        return true;
    });
    require(completed == o.iterations && mv_pixels > 0 && ncc_pixels > 0,
        "Vulkan trainer did not complete actual MV geometry/NCC supervision");
    require(model.size() > 4 && model.filter_3d.numel() == model.size() &&
        model.filter_3d.device() == tinytensor::Device::Vulkan,
        "Vulkan filter was not recomputed after densification");
    require_finite(model.means,"Non-finite Vulkan trained means");
    require_finite(model.filter_3d,"Non-finite Vulkan final 3D filter");
    SplatMeshOptions mesh_options;
    mesh_options.max_depth = 3.F;
    mesh_options.fusion.mesh_method = photara::mvs::MeshMethod::tsdf;
    mesh_options.fusion.mesh_tsdf_voxel_size = .04F;
    const auto mesh = extract_splat_mesh(model,scene,o,mesh_options);
    require(mesh.valid_depth_pixels > 0 && !mesh.mesh.faces.empty(),
        "Vulkan geometry training did not produce a TSDF surface");
    for (const auto& view : scene.views) std::filesystem::remove(view.path);
    std::filesystem::remove(root);
    std::cout << "Vulkan geometry trainer completed with " << model.size() << " Gaussians, "
              << mesh.valid_depth_pixels << " depth pixels, " << mesh.mesh.faces.size() << " mesh faces\n";
}
