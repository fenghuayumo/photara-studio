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
    // Tiled query scatter must preserve original indices, empty tiles and
    // per-point tolerance decisions, including non-multiple group tails.
    {
        struct QueryEnvironment {
            bool existed;
            std::string previous;
            QueryEnvironment() : existed(std::getenv("SPLAT_DRENDER_DISABLE_TILED_DEPTH_QUERY") != nullptr),
                previous(existed ? std::getenv("SPLAT_DRENDER_DISABLE_TILED_DEPTH_QUERY") : "") {}
            static void set(const char* value) {
#ifdef _WIN32
                _putenv_s("SPLAT_DRENDER_DISABLE_TILED_DEPTH_QUERY",value);
#else
                if (*value) setenv("SPLAT_DRENDER_DISABLE_TILED_DEPTH_QUERY",value,1);
                else unsetenv("SPLAT_DRENDER_DISABLE_TILED_DEPTH_QUERY");
#endif
            }
            ~QueryEnvironment() { set(existed ? previous.c_str() : ""); }
        } environment;
        auto m=make_model(Device::Vulkan);
        auto c=camera;
        c.width=37; c.height=35; c.cx=18.F; c.cy=17.F;
        for (auto mode : {photara::CameraModel::pinhole,photara::CameraModel::opencv_fisheye,
                          photara::CameraModel::equirectangular}) {
            c.model=mode;
            c.k1=mode==photara::CameraModel::opencv_fisheye ? -.08F : 0.F;
            c.k2=mode==photara::CameraModel::opencv_fisheye ? .01F : 0.F;
            for (std::size_t n : {1u,127u,128u,129u,255u,256u,257u,513u,4099u}) {
                std::vector<float> world(3*n), dl(3*n);
                for (std::size_t i=0;i<n;++i) {
                    world[3*i]=.04F*float(int((i*13)%43)-21);
                    world[3*i+1]=.04F*float(int((i*17)%39)-19);
                    world[3*i+2]=i%11==0 ? -2.F : 2.F;
                    dl[3*i+2]=.2F/float(n);
                    if (i%19==0) world[3*i]=100.F;
                }
                const auto points=Tensor::from_vector(world,{n,3},Device::Vulkan);
                const auto upstream=Tensor::from_vector(dl,{n,3},Device::Vulkan);
                for (float tolerance : {0.F,.02F}) {
                    RasterizeOptions options;
                    options.point_depth_tolerance=tolerance;
                    Rasterizer rasterizer;
                    QueryEnvironment::set("1");
                    const auto reference=rasterizer.sample_depth(m,points,c,options);
                    const auto reference_points=reference.camera_points.to_vector();
                    const auto reference_inside=reference.inside.to(tinytensor::DataType::Float32).to_vector();
                    require(reference_inside.size()==n,"Query inside mask was not materialized");
                    if (n==4099) {
                        require(std::find(reference_inside.begin(),reference_inside.end(),1.F)!=reference_inside.end(),
                            "Tiled query fixture must contain valid depths");
                        require(std::find(reference_inside.begin(),reference_inside.end(),0.F)!=reference_inside.end(),
                            "Tiled query fixture must contain invalid depths");
                    }
                    const auto reference_grad=rasterizer.sample_depth_backward(m,reference,upstream);
                    QueryEnvironment::set("");
                    const auto tiled=rasterizer.sample_depth(m,points,c,options);
                    require(tiled.inside.to(tinytensor::DataType::Float32).to_vector()==reference_inside,
                        "Tiled query changed inside mask or original query indices");
                    const auto actual=tiled.camera_points.to_vector();
                    require(actual.size()==reference_points.size(),"Tiled query output shape changed");
                    for (std::size_t j=0;j<actual.size();++j)
                        require(std::isfinite(actual[j]) && std::abs(actual[j]-reference_points[j])<2e-5F,
                            "Tiled query changed a depth result or original query index");
                    const auto tiled_grad=rasterizer.sample_depth_backward(m,tiled,upstream);
                    compare(reference_grad.points,tiled_grad.points,.001F,"Tiled query point gradients");
                    compare(reference_grad.model.means,tiled_grad.model.means,.001F,"Tiled query model gradients");
                }
            }
        }
        std::cout<<"Tiled depth query tails, invalid points, scatter and tolerance passed\n";
    }
    // Exercise chunk boundaries and heavy overlap, with the CUDA query oracle.
    {
        constexpr std::size_t queries=270017;
        std::vector<float> world(3*queries), upstream(3*queries);
        for (std::size_t i=0;i<queries;++i) {
            world[3*i]=.01F*float(int(i%23)-11);
            world[3*i+1]=.01F*float(int(i%19)-9);
            world[3*i+2]=2.F;
            upstream[3*i]=.1F/queries;
            upstream[3*i+1]=-.2F/queries;
            upstream[3*i+2]=.3F/queries;
        }
        const auto run_query = [&](Device device) {
            auto m=make_model(device);
            Rasterizer rasterizer;
            RasterizeOptions options;
            options.require_depth=true;
            options.point_depth_tolerance=0.F;
            const auto sampled=rasterizer.sample_depth(m,
                Tensor::from_vector(world,{queries,3},device),camera,options);
            return rasterizer.sample_depth_backward(m,sampled,
                Tensor::from_vector(upstream,{queries,3},device));
        };
        const auto cuda=run_query(Device::CUDA), vk=run_query(Device::Vulkan);
        compare(cuda.points,vk.points,.003F,"Vulkan chunked query point gradients");
        compare(cuda.model.means,vk.model.means,.003F,"Vulkan chunked query mean gradients");
        compare(cuda.model.log_scales,vk.model.log_scales,.003F,"Vulkan chunked query scale gradients");
        compare(cuda.model.quaternions,vk.model.quaternions,.003F,"Vulkan chunked query rotation gradients");
        compare(cuda.model.opacity_logits,vk.model.opacity_logits,.003F,"Vulkan chunked query opacity gradients");
    }
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
    auto margin_means = Tensor::from_vector(
        std::vector<float>{1.F,0.F,2.F, 0.F,0.F,4.F}, {2,3}, Device::CUDA);
    compare(detail::compute_3d_filter(margin_means,{camera}),
        detail::compute_3d_filter(margin_means.to(Device::Vulkan),{camera}),
        2e-5F,"Vulkan 15-percent filter margin parity");
    auto offset_camera = camera;
    offset_camera.cx = 6.F;
    auto offset_means = Tensor::from_vector(
        std::vector<float>{-.6F,0.F,2.F, 0.F,0.F,4.F}, {2,3}, Device::CUDA);
    compare(detail::compute_3d_filter(offset_means,{offset_camera}),
        detail::compute_3d_filter(offset_means.to(Device::Vulkan),{offset_camera}),
        2e-5F,"Vulkan off-center filter parity");
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

void test_vulkan_normal_field() {
#ifdef TINYTENSOR_HAS_VULKAN
    using namespace photara::splat;
    using tinytensor::Tensor;
    using tinytensor::Device;
    const auto compare = [](const Tensor& a, const Tensor& b, double tolerance, const char* label) {
        const auto x = a.to_vector(), y = b.to_vector();
        require(x.size() == y.size(), label);
        double error = 0., norm = 0.;
        for (std::size_t i = 0; i < x.size(); ++i) {
            require(std::isfinite(x[i]) && std::isfinite(y[i]), label);
            error += (x[i]-y[i])*(x[i]-y[i]); norm += x[i]*x[i];
        }
        const double relative = std::sqrt(error/std::max(norm,1e-16));
        std::cout << label << " relative_l2=" << relative << '\n';
        require(relative < tolerance || std::sqrt(error) < 2e-6, label);
    };
    auto features = Tensor::from_vector(std::vector<float>{
        .2F,-.4F,1.3F,.7F, -.5F,.8F,.3F,-1.2F, 0,0,0,.2F, 1,0,0,0}, {4,4}, Device::CUDA);
    auto g = Tensor::from_vector(std::vector<float>{.1F,-.8F,.2F, -.3F,.6F,.9F, 1,2,3, .5F,0,0}, {4,3}, Device::CUDA);
    compare(detail::normal_features_to_normals(features),
        detail::normal_features_to_normals(features.to(Device::Vulkan)), 2e-5, "Vulkan normal feature forward");
    const auto analytic = detail::normal_features_backward(features.to(Device::Vulkan), g.to(Device::Vulkan));
    compare(detail::normal_features_backward(features,g), analytic, 2e-5, "Vulkan normal feature backward");
    auto fv = features.to_vector(); const auto gv = g.to_vector(), av = analytic.to_vector();
    const auto objective = [&](const std::vector<float>& values) {
        const auto n = detail::normal_features_to_normals(Tensor::from_vector(values,{4,4},Device::Vulkan)).to_vector();
        double sum = 0; for (std::size_t j=0;j<n.size();++j) sum += n[j]*gv[j]; return sum;
    };
    for (std::size_t j=0;j<8;++j) {
        auto plus=fv, minus=fv; plus[j]+=1e-3F; minus[j]-=1e-3F;
        const double fd=(objective(plus)-objective(minus))/.002;
        require(std::abs(fd-av[j])<2e-4, "Vulkan normal feature finite difference mismatch");
    }
    constexpr std::size_t side=16, pixels=side*side;
    Camera camera; camera.width=camera.height=side;
    camera.fx=camera.fy=24; camera.cx=camera.cy=7.5F;
    const float c=std::cos(.37F), s=std::sin(.37F);
    camera.world_to_camera={c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1};
    std::vector<float> depth(pixels), field(3*pixels);
    for (std::size_t y=0;y<side;++y) for (std::size_t x=0;x<side;++x) {
        const auto i=y*side+x; depth[i]=2.F+.02F*x+.01F*y+.003F*std::sin(float(x*y));
        field[i]=.15F+.01F*x; field[pixels+i]=-.1F+.02F*y; field[2*pixels+i]=-.7F;
    }
    depth[3*side+4]=0; // Invalid depth must mask the surrounding stencil.
    const auto render = [&](Device device) {
        RenderResult r;
        r.color=Tensor::from_vector(field,{3,side,side},device);
        r.alpha=Tensor::full({side,side},1.F,device);
        r.median_depth=Tensor::from_vector(depth,{side,side},device);
        r.normal=Tensor::zeros({3,side,side},device); return r;
    };
    const auto cuda=detail::compute_normal_field_loss(render(Device::CUDA),camera,.03F,true);
    const auto vk=detail::compute_normal_field_loss(render(Device::Vulkan),camera,.03F,true);
    require(std::abs(cuda.total-vk.total)<1e-6, "Vulkan normal field loss mismatch");
    compare(cuda.color,vk.color,2e-5,"Vulkan world normal field gradient");
    compare(cuda.depth,vk.depth,2e-4,"Vulkan normal field depth gradient");
    const auto quiet=detail::compute_normal_field_loss(render(Device::Vulkan),camera,.03F,false);
    compare(vk.color,quiet.color,2e-5,"Vulkan non-reporting normal field gradient");
    compare(vk.depth,quiet.depth,2e-5,"Vulkan non-reporting normal field depth gradient");
    // Full-resolution scalar accumulation used to serialize millions of CAS
    // retries on one address and reset Windows display GPUs.
    {
        constexpr std::size_t width=1920, height=1276, count=width*height;
        Camera large; large.width=width; large.height=height;
        large.fx=large.fy=1000; large.cx=width*.5F; large.cy=height*.5F;
        large.world_to_camera={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        RenderResult r;
        r.color=Tensor::zeros({3,height,width},Device::Vulkan);
        r.alpha=Tensor::full({height,width},1.F,Device::Vulkan);
        r.median_depth=Tensor::full({height,width},2.F,Device::Vulkan);
        r.normal=Tensor::zeros({3,height,width},Device::Vulkan);
        const float expected=.05F*float((width-2)*(height-2))/float(count);
        detail::LossGradients dn;
        detail::add_depth_normal_loss_vulkan(r,large,.05F,dn,true);
        require(std::abs(dn.total-expected)<2e-6F,"Full-resolution Vulkan DN reduction mismatch");
        const auto nf=detail::compute_normal_field_loss(r,large,.05F,true);
        require(std::abs(nf.total-expected)<2e-6F,"Full-resolution Vulkan normal field reduction mismatch");
        detail::LossGradients dn_quiet;
        detail::add_depth_normal_loss_vulkan(r,large,.05F,dn_quiet,false);
        compare(dn.normal,dn_quiet.normal,1e-6,"Full-resolution quiet DN gradients");
        const auto nf_quiet=detail::compute_normal_field_loss(r,large,.05F,false);
        compare(nf.color,nf_quiet.color,1e-6,"Full-resolution quiet normal field gradients");
        std::cout << "Full-resolution DN/normal field reduction passed\n";
        // More than 256*256 pixels exercises every level of MV reduction.
        large.fx=large.fy=1024.F;
        TrainingView reference, neighbour;
        reference.camera=neighbour.camera=large;
        const auto points=detail::unproject_depth_to_world(r.median_depth,large);
        const auto inside=Tensor::from_vector(std::vector<int>(count,1),{height,width},Device::Vulkan);
        TrainingOptions options;
        options.multi_view_geo_weight=.02F;
        options.multi_view_ncc_weight=0.F;
        detail::LossGradients mv_gradients;
        Tensor point_gradients;
        const auto mv=detail::add_multi_view_loss(points,inside,r,reference,neighbour,
            options,mv_gradients,point_gradients,true);
        require(mv.geometry_candidates==count && mv.geometry_pixels==count,
            "Full-resolution Vulkan MV count reduction mismatch");
        require(std::abs(mv.geometry-1e-6F)<1e-9F,"Full-resolution Vulkan MV loss reduction mismatch");
        std::cout << "Full-resolution MV reduction passed\n";
    }
    // Independent finite differences through a rotated camera and depth stencil.
    for (bool color : {true,false}) {
        const std::size_t index=8*side+9;
        const auto& gradient=color?vk.color:vk.depth;
        const auto grad=gradient.to_vector();
        auto values=color?field:depth;
        const auto loss = [&](const std::vector<float>& v) {
            auto r=render(Device::Vulkan);
            if(color) r.color=Tensor::from_vector(v,{3,side,side},Device::Vulkan);
            else r.median_depth=Tensor::from_vector(v,{side,side},Device::Vulkan);
            return detail::compute_normal_field_loss(r,camera,.03F,true).total;
        };
        auto plus=values, minus=values; plus[index]+=.001F; minus[index]-=.001F;
        const float fd=(loss(plus)-loss(minus))/.002F;
        require(std::abs(fd-grad[index])<1e-5F,"Vulkan normal field loss finite difference mismatch");
    }
    const auto chain = [&](Device device, bool reuse = false) {
        GaussianModel m;
        m.means=Tensor::from_vector(std::vector<float>{-.8F,0,2.F, -.6F,.05F,2.05F},{2,3},device);
        m.log_scales=Tensor::from_vector(std::vector<float>{std::log(.4F),std::log(.35F),std::log(.035F),
            std::log(.38F),std::log(.32F),std::log(.03F)},{2,3},device);
        m.quaternions=Tensor::from_vector(std::vector<float>{.985F,0,.174F,0, .98F,-.08F,-.17F,.015F},{2,4},device);
        m.opacity_logits=Tensor::from_vector(std::vector<float>{2,1.5F},{2,1},device);
        m.sh=Tensor::full({2,16,3},.3F,device); m.sh_degree=3;
        m.normal_features=Tensor::from_vector(std::vector<float>{.2F,-.4F,1.3F,.7F, -.5F,.8F,.3F,-1.2F},{2,4},device);
        m.filter_3d=Tensor::full({2,1},.005F,device);
        Rasterizer rasterizer; RasterizeOptions o; o.require_depth=true;
        if (reuse) {
            o.active_sh_degree = 3;
            o.copy_attachments = false;
            const auto primary = rasterizer.forward(m,camera,o);
            (void)rasterizer.backward(m,primary,
                Tensor::zeros({3,side,side},device),Tensor{},Tensor{},Tensor{});
            o.copy_attachments = true;
            o.reuse_projection = true;
        }
        o.colors_precomp=detail::normal_features_to_normals(m.normal_features);
        const auto r=rasterizer.forward(m,camera,o);
        const auto signed_colors=r.color.to_vector();
        require(std::any_of(signed_colors.begin(),signed_colors.end(),[](float value){return value < -.01F;}),
            "Normal field render must preserve signed colors");
        const auto l=detail::compute_normal_field_loss(r,camera,.03F,true);
        require(l.total>0 && l.total<.1F,"Normal field render did not contribute loss");
        auto gradients=rasterizer.backward(m,r,l.color,l.alpha,l.depth,l.normal);
        if (reuse) {
            auto changed_camera = camera;
            changed_camera.fx += 1.F;
            bool rejected = false;
            try { (void)rasterizer.forward(m,changed_camera,o); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected,"Vulkan projection reuse accepted a changed camera");
        }
        require(!gradients.sh.is_valid(),"Normal field must not differentiate SH");
        gradients.normal_features=detail::normal_features_backward(m.normal_features,gradients.colors_precomp);
        require_finite(gradients.normal_features,"Non-finite normal field feature gradient");
        const auto feature_gradient=gradients.normal_features.to_vector();
        require(std::any_of(feature_gradient.begin(),feature_gradient.end(),[](float value){return std::abs(value)>1e-6F;}),
            "Normal field chain must produce nonzero feature gradients");
        return std::make_pair(r.color,gradients);
    };
    const auto a=chain(Device::CUDA), b=chain(Device::Vulkan);
    compare(a.first,b.first,2e-3,"Vulkan signed precomputed render");
    compare(a.second.normal_features,b.second.normal_features,3e-3,"Vulkan complete normal feature gradient");
    compare(a.second.means,b.second.means,3e-3,"Vulkan normal field mean gradient");
    compare(a.second.log_scales,b.second.log_scales,3e-3,"Vulkan normal field scale gradient");
    compare(a.second.quaternions,b.second.quaternions,3e-3,"Vulkan normal field rotation gradient");
    compare(a.second.opacity_logits,b.second.opacity_logits,3e-3,"Vulkan normal field opacity gradient");
    const auto reused = chain(Device::Vulkan,true);
    compare(b.first,reused.first,2e-5,"Vulkan reused normal field render");
    compare(b.second.normal_features,reused.second.normal_features,2e-5,"Vulkan reused normal feature gradient");
    compare(b.second.means,reused.second.means,2e-5,"Vulkan reused normal field mean gradient");
    compare(b.second.log_scales,reused.second.log_scales,2e-5,"Vulkan reused normal field scale gradient");
    compare(b.second.quaternions,reused.second.quaternions,2e-5,"Vulkan reused normal field rotation gradient");
    compare(b.second.opacity_logits,reused.second.opacity_logits,2e-5,"Vulkan reused normal field opacity gradient");
    GaussianModel init;
    init.means=Tensor::zeros({3,3},Device::Vulkan);
    init.log_scales=Tensor::from_vector(std::vector<float>{-3,-1,-2, -1,-3,-2, -1,-2,-3},{3,3},Device::Vulkan);
    const std::vector<float> quaternions{2,.3F,-.4F,.1F, 1,-.2F,.3F,.4F, .7F,.1F,-.2F,.3F};
    init.quaternions=Tensor::from_vector(quaternions,{3,4},Device::Vulkan);
    const auto initialized=detail::normal_features_from_smallest_axis_vulkan(init).to_vector();
    for(std::size_t i=0;i<3;++i) {
        Eigen::Quaternionf q(quaternions[4*i],quaternions[4*i+1],quaternions[4*i+2],quaternions[4*i+3]);
        q.normalize(); const auto direction=q.toRotationMatrix().col(i).eval();
        for(std::size_t j=0;j<3;++j)
            require(std::abs(initialized[4*i+j]-direction(j))<2e-6F,"Vulkan smallest-axis initialization mismatch");
        require(initialized[4*i+3]==0,"Vulkan initial normal field sign must be zero");
    }
    init.log_scales=Tensor::from_vector(std::vector<float>{-30,-31,-32, -30,-31,-32, -30,-31,-32},{3,3},Device::Vulkan);
    init.filter_3d=Tensor::full({3,1},.01F,Device::Vulkan);
    const auto tied=detail::normal_features_from_smallest_axis_vulkan(init).to_vector();
    for(std::size_t i=0;i<3;++i) {
        Eigen::Quaternionf q(quaternions[4*i],quaternions[4*i+1],quaternions[4*i+2],quaternions[4*i+3]);
        q.normalize(); const auto direction=q.toRotationMatrix().col(0).eval();
        for(std::size_t j=0;j<3;++j)
            require(std::abs(tied[4*i+j]-direction(j))<2e-6F,"Vulkan filtered-scale tie initialization mismatch");
    }
#else
    std::cout << "SKIP Vulkan normal field: backend not built\n";
#endif
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
    o.use_normal_field = true; o.normal_field_weight = .05F; o.normal_field_from_iter = 4;
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
    require_finite(model.normal_features,"Non-finite Vulkan trained normal field");
    const auto feature_values=model.normal_features.to_vector();
    float orientation=0;
    for(std::size_t i=3;i<feature_values.size();i+=4) orientation+=std::abs(feature_values[i]);
    require(orientation>1e-6F,"Vulkan normal field Adam did not update orientation");
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
