// CPU double-precision oracle and finite differences, independent of GPU ray inversion.
void test_native_geometry(tinytensor::Device device) {
    using namespace photara::splat;
    using tinytensor::Tensor;
    constexpr unsigned w = 16, h = 10, n = w * h;
    for (auto model :
         {photara::CameraModel::pinhole, photara::CameraModel::opencv_fisheye,
          photara::CameraModel::equirectangular}) {
        Camera c;
        c.model = model;
        c.width = w;
        c.height = h;
        c.fx = c.fy = 4.F;
        c.cx = 8.F;
        c.cy = 5.F;
        c.k1 = .025F;
        c.k2 = -.001F;
        c.k3 = .0001F;
        c.k4 = -.000001F;
        c.world_to_camera = {0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1};
        if (model == photara::CameraModel::equirectangular) {
            c.fx = w / (2.F * std::numbers::pi_v<float>);
            c.fy = h / std::numbers::pi_v<float>;
        }
        std::vector<Eigen::Vector3d> rays(n, Eigen::Vector3d::Zero());
        for (unsigned y = 0; y < h; ++y)
            for (unsigned x = 0; x < w; ++x) {
                auto& r = rays[y * w + x];
                if (model == photara::CameraModel::pinhole)
                    r = {(x - c.cx) / c.fx, (y - c.cy) / c.fy, 1};
                else if (model == photara::CameraModel::equirectangular) {
                    auto q = photara::unproject_equirectangular_camera(x, y, w, h);
                    r = {q.x, q.y, q.z};
                } else {
                    auto q = photara::unproject_fisheye_camera(
                        x, y, c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.k3, c.k4);
                    if (q.valid && q.z > 1e-6)
                        r = {q.x / q.z, q.y / q.z, 1};
                }
            }
        std::vector<float> depths(n), normals(3 * n);
        for (unsigned i = 0; i < n; ++i) {
            depths[i] = 2.F + .1F * std::sin(float(i) * .4F);
            normals[i] = .3F;
            normals[n + i] = -.2F;
            normals[2 * n + i] = -.8F;
        }
        // A missing depth must invalidate every stencil that touches it.
        depths[4 * w + 6] = 0;
        for (bool field : {false, true}) {
            const auto cpu = [&](const std::vector<float>& d, const std::vector<float>& ns) {
                double loss = 0;
                for (unsigned y = 1; y + 1 < h; ++y)
                    for (unsigned x = 0; x < w; ++x) {
                        bool wrap = model == photara::CameraModel::equirectangular;
                        if (!wrap && (x == 0 || x + 1 == w))
                            continue;
                        unsigned i = y * w + x, l = y * w + (x + w - 1) % w,
                                 r = y * w + (x + 1) % w, t = i - w, b = i + w;
                        if (!(d[i] > 0 && d[l] > 0 && d[r] > 0 && d[t] > 0 && d[b] > 0))
                            continue;
                        if (rays[i].isZero() || rays[l].isZero() || rays[r].isZero() ||
                            rays[t].isZero() || rays[b].isZero())
                            continue;
                        Eigen::Vector3d dy = rays[b] * d[b] - rays[t] * d[t],
                                        dx = rays[r] * d[r] - rays[l] * d[l];
                        Eigen::Vector3d cross = dy.cross(dx);
                        if (cross.squaredNorm() <= 1e-20)
                            continue;
                        Eigen::Vector3d v(ns[i], ns[n + i], ns[2 * n + i]);
                        if (field)
                            v = Eigen::Vector3d(v.z(), v.y(), -v.x()).eval();
                        loss += .05 / n * (1 - v.dot(cross.normalized()));
                    }
                return loss;
            };
            RenderResult rendered;
            rendered.color = Tensor::zeros({3, h, w}, device);
            rendered.alpha = Tensor::zeros({h, w}, device);
            rendered.normal = Tensor::from_vector(normals, {3, h, w}, device);
            rendered.median_depth = Tensor::from_vector(depths, {h, w}, device);
            if (field)
                rendered.color = Tensor::from_vector(normals, {3, h, w}, device);
            TrainingView target;
            target.camera = c;
            target.rgb = Tensor::zeros({3, h, w}, device);
            target.depth = Tensor::zeros({h, w}, device);
            target.normal = Tensor::zeros({3, h, w}, device);
            target.mask = Tensor::full({h, w}, 1.F, device);
            TrainingOptions options;
            options.photometric_weight = 0;
            options.ssim_weight = 0;
            options.depth_normal_weight = .05F;
            auto loss = field
                            ? detail::compute_normal_field_loss(rendered, c, .05F, true)
                            : detail::compute_training_loss(rendered, target, options, true, true);
            require(
                std::abs(loss.normal_value - cpu(depths, normals)) < 2e-6,
                "Native geometry CPU forward oracle mismatch");
            auto gd = loss.depth.to_vector(), gn = (field ? loss.color : loss.normal).to_vector();
            constexpr float eps = .001F;
            for (unsigned i = 0; i < n; ++i) {
                if (depths[i] > 0) {
                    auto d = depths;
                    d[i] += eps;
                    double plus = cpu(d, normals);
                    d[i] -= 2 * eps;
                    double numeric = (plus - cpu(d, normals)) / (2 * eps);
                    require(
                        std::isfinite(gd[i]) &&
                            std::abs(gd[i] - numeric) < 2e-5 + std::abs(numeric) * .01,
                        "Native geometry depth finite difference mismatch (including panorama "
                        "seam)");
                }
                for (unsigned ch = 0; ch < 3; ++ch) {
                    auto ns = normals;
                    unsigned j = ch * n + i;
                    ns[j] += eps;
                    double plus = cpu(depths, ns);
                    ns[j] -= 2 * eps;
                    double numeric = (plus - cpu(depths, ns)) / (2 * eps);
                    require(
                        std::isfinite(gn[j]) && std::abs(gn[j] - numeric) < 2e-6,
                        "Native geometry normal finite difference mismatch");
                }
            }
        }
    }
    std::cout << "Native geometry CPU oracles and depth/normal gradients passed\n";
}

void test_native_geometry_trainer(photara::splat::TrainingBackend backend) {
    using namespace photara;
    using namespace photara::splat;
    auto path =
        std::filesystem::temp_directory_path() /
        ("photara_native_geometry_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".png");
    io::save_rgb_png(io::RgbImage{128, 64, std::vector<std::uint8_t>(128 * 64 * 3, 128)}, path);
    for (auto model : {CameraModel::opencv_fisheye, CameraModel::equirectangular}) {
        mvs::MvsScene scene;
        mvs::MvsView view;
        view.path = path;
        view.width = view.src_width = 128;
        view.height = view.src_height = 64;
        view.fx = view.fy = view.src_fx = view.src_fy = 40.F;
        view.cx = view.src_cx = 64.F;
        view.cy = view.src_cy = 32.F;
        view.source_model = model;
        scene.views = {view};
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
                scene.dense_cloud.points.push_back(
                    {mvs::Vec3f(x * .35F, y * .35F, 2.F), mvs::Vec3f(0, 0, -1),
                     mvs::Vec3f::Constant(.5F)});
        TrainingOptions o;
        o.backend = backend;
        o.iterations = 3;
        o.enable_densification = false;
        o.input_is_dense = true;
        o.constrain_scale_range = false;
        o.multi_view_depth_bracket = -1.F;
        o.initial_opacity = .8F;
        o.initial_scale = .4F;
        o.progressive_resolution = false;
        o.use_depth_normal_loss = true;
        o.depth_normal_from_iter = 1;
        o.use_normal_field = true;
        o.normal_field_from_iter = 1;
        o.photometric_weight = 0;
        o.ssim_weight = 0;
        bool saw_geometry = false;
        const auto trained = Trainer(o).train(scene, [&](const TrainingProgress& p) {
            require(std::isfinite(p.loss), "Native geometry training diverged");
            saw_geometry |= p.normal_loss > 0;
            return true;
        });
        require(
            saw_geometry,
            "Native geometry trainer still disables depth-normal/normal-field losses");
        require_finite(
            trained.normal_features, "Native geometry trainer produced invalid field features");
    }
    std::filesystem::remove(path);
    std::cout << "Native geometry trainer smoke tests passed\n";
}
