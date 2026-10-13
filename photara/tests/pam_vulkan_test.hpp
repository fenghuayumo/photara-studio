#pragma once
#ifdef TINYTENSOR_HAS_VULKAN
#include "../src/splat/rasterizer_vulkan.hpp"
void test_pam_vulkan() {
    using namespace photara::splat;
    using tinytensor::Device;
    using tinytensor::Tensor;
    if (!tinytensor::vulkan::available())
        throw std::runtime_error("PAM Vulkan test requires a Vulkan device");
    const std::vector<photara::mvs::Vec3f> shell{{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                                 {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    int cuda_count = 0;
    const bool has_cuda = cudaGetDeviceCount(&cuda_count) == cudaSuccess && cuda_count > 0;
    (void)cudaGetLastError();
    const auto reference_device = has_cuda ? Device::CUDA : Device::Vulkan;
    GaussianModel cpu;
    std::vector<float> means, features, rotations(24, 0.F);
    for (std::size_t i = 0; i < shell.size(); ++i) {
        means.insert(means.end(), shell[i].data(), shell[i].data() + 3);
        features.insert(features.end(), shell[i].data(), shell[i].data() + 3);
        features.push_back(3.F);
        rotations[4 * i] = 1.F;
    }
    cpu.means = Tensor::from_vector(means, {shell.size(), std::size_t{3}}, reference_device);
    cpu.log_scales = Tensor::from_vector(std::vector<float>(18, std::log(.65F)),
                                         {shell.size(), std::size_t{3}}, reference_device);
    cpu.quaternions =
        Tensor::from_vector(rotations, {shell.size(), std::size_t{4}}, reference_device);
    cpu.opacity_logits = Tensor::from_vector(std::vector<float>(6, 8.F),
                                             {shell.size(), std::size_t{1}}, reference_device);
    cpu.normal_features =
        Tensor::from_vector(features, {shell.size(), std::size_t{4}}, reference_device);
    cpu.sh = Tensor::zeros({shell.size(), std::size_t{1}, std::size_t{3}}, reference_device);
    cpu.filter_3d = Tensor::from_vector(std::vector<float>(6, .01F), {shell.size(), std::size_t{1}},
                                        reference_device);
    auto copy_to_vulkan = [](const GaussianModel& m) {
        GaussianModel v;
        v.means = m.means.to(Device::Vulkan);
        v.log_scales = m.log_scales.to(Device::Vulkan);
        v.quaternions = m.quaternions.to(Device::Vulkan);
        v.opacity_logits = m.opacity_logits.to(Device::Vulkan);
        v.normal_features = m.normal_features.to(Device::Vulkan);
        v.sh = m.sh.to(Device::Vulkan);
        v.filter_3d = m.filter_3d.to(Device::Vulkan);
        return v;
    };
    auto vk = copy_to_vulkan(cpu);
    // GW stores inverse_sigmoid(0/1) as infinities. Preserve those endpoints
    // when loading on Vulkan, but keep rejecting NaN opacity parameters.
    const auto ply =
        std::filesystem::temp_directory_path() /
        ("photara_pam_opacity_" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ply");
    save_gaussians_ply(cpu, ply);
    std::streamoff data_offset{}, opacity_offset{};
    std::size_t properties = 0;
    {
        std::fstream file(ply, std::ios::in | std::ios::out | std::ios::binary);
        std::string line;
        while (std::getline(file, line)) {
            if (line == "end_header") {
                data_offset = file.tellg();
                break;
            }
            if (line.rfind("property float ", 0) == 0) {
                if (line == "property float opacity")
                    opacity_offset = properties * sizeof(float);
                ++properties;
            }
        }
        const float positive = std::numeric_limits<float>::infinity(), negative = -positive;
        file.seekp(data_offset + opacity_offset);
        file.write(reinterpret_cast<const char*>(&positive), sizeof(float));
        file.seekp(data_offset + properties * sizeof(float) + opacity_offset);
        file.write(reinterpret_cast<const char*>(&negative), sizeof(float));
    }
    const auto loaded = load_gaussians_ply(ply, Device::Vulkan).opacity_logits.to_vector();
    require(loaded[0] == 104.F && loaded[1] == -104.F, "GW saturated opacity PLY compatibility");
    {
        std::fstream file(ply, std::ios::in | std::ios::out | std::ios::binary);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        file.seekp(data_offset + opacity_offset);
        file.write(reinterpret_cast<const char*>(&nan), sizeof(float));
    }
    bool rejected = false;
    try {
        (void)load_gaussians_ply(ply, Device::Vulkan);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    std::filesystem::remove(ply);
    require(rejected, "NaN opacity must remain invalid");
    std::vector<float> queries;
    for (unsigned i = 0; i < 1031; ++i)
        queries.insert(queries.end(),
                       {.11F * float(int(i % 23) - 11), .09F * float(int(i % 19) - 9),
                        .13F * float(int(i % 17) - 8)});
    // Empty rays, out-of-frustum points and points behind the camera.
    queries.insert(queries.end(), {100, 0, 0, 0, 0, -5, 0, 0, 0, 0, 0, 1, 0, 0, -1});
    auto points =
        Tensor::from_vector(queries, {queries.size() / 3, std::size_t{3}}, Device::Vulkan);
    const auto field = detail::make_pam_gpu_field(vk);
    for (unsigned k : {1U, 4U, 6U, 32U, 64U}) {
        const auto expected = detail::pam_field_gradients_reference(cpu, queries, k);
        const auto actual = detail::pam_field_gradients(field, points, k).to_vector();
        for (std::size_t i = 0; i < expected.size(); ++i)
            require(std::abs(expected[i] - actual[i]) < 3e-4F * (1.F + std::abs(expected[i])),
                    "PAM Vulkan field parity");
        auto refined = points.clone();
        auto occupancy = Tensor::from_vector(std::vector<float>(queries.size() / 3, .7F),
                                             {queries.size() / 3}, Device::Vulkan);
        detail::pam_refine_points(field, refined, occupancy, k, .5F, .5F, .5F);
        const auto result = refined.to_vector();
        for (std::size_t i = 0; i < queries.size() / 3; ++i) {
            Eigen::Vector3f g(expected[3 * i], expected[3 * i + 1], expected[3 * i + 2]);
            Eigen::Vector3f q(queries[3 * i], queries[3 * i + 1], queries[3 * i + 2]);
            if (g.squaredNorm() > .5F)
                q += .5F * std::clamp(.2F / g.squaredNorm(), -1.F, 1.F) * g;
            for (unsigned j = 0; j < 3; ++j)
                require(std::abs(q[j] - result[3 * i + j]) < 5e-5F, "PAM Vulkan projection parity");
        }
    }
    // More Gaussians than the largest query heap: exercise exact tree pruning,
    // anisotropic rotated covariance, signed normals and filtered opacity.
    {
        constexpr std::size_t n = 129;
        GaussianModel large;
        std::vector<float> m, s, r, f, o;
        for (std::size_t i = 0; i < n; ++i) {
            m.insert(m.end(), {.047F * float(int(i % 11) - 5), .053F * float(int(i % 7) - 3),
                               .061F * float(int(i % 5) - 2)});
            s.insert(s.end(), {std::log(.11F), std::log(.27F), std::log(.43F)});
            const float angle = .02F * float(i);
            r.insert(r.end(), {std::cos(angle), 0.F, 0.F, std::sin(angle)});
            f.insert(f.end(), {.3F, -.2F, .8F, i % 2 ? -2.F : 2.F});
            o.push_back(1.1F);
        }
        large.means = Tensor::from_vector(m, {n, std::size_t{3}}, reference_device);
        large.log_scales = Tensor::from_vector(s, {n, std::size_t{3}}, reference_device);
        large.quaternions = Tensor::from_vector(r, {n, std::size_t{4}}, reference_device);
        large.normal_features = Tensor::from_vector(f, {n, std::size_t{4}}, reference_device);
        large.opacity_logits = Tensor::from_vector(o, {n, std::size_t{1}}, reference_device);
        large.sh = Tensor::zeros({n, std::size_t{1}, std::size_t{3}}, reference_device);
        large.filter_3d =
            Tensor::from_vector(std::vector<float>(n, .02F), {n, std::size_t{1}}, reference_device);
        auto vk_large = copy_to_vulkan(large);
        const auto field_large = detail::make_pam_gpu_field(vk_large);
        for (unsigned k : {1U, 4U, 32U, 64U}) {
            const auto expected = detail::pam_field_gradients_reference(large, queries, k);
            const auto actual = detail::pam_field_gradients(field_large, points, k).to_vector();
            for (std::size_t i = 0; i < actual.size(); ++i) {
                if (std::abs(actual[i] - expected[i]) >= 4e-4F * (1.F + std::abs(expected[i])))
                    std::cerr << "PAM anisotropic k=" << k << " query=" << i / 3
                              << " component=" << i % 3 << " expected=" << expected[i]
                              << " actual=" << actual[i] << '\n';
                require(std::abs(actual[i] - expected[i]) < 4e-4F * (1.F + std::abs(expected[i])),
                        "PAM Vulkan anisotropic exact KD-tree parity");
            }
        }
    }
    Camera camera;
    camera.width = camera.height = 64;
    camera.fx = camera.fy = 50.F;
    camera.cx = camera.cy = 31.5F;
    camera.world_to_camera = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 4, 1};
    camera.position = {0, 0, -4};
    std::shared_ptr<void> backend;
    const auto actual = detail::vulkan_occupancy(backend, vk, points, camera, {});
    // A prepared view must serve nonzero-offset chunks and a short tail
    // identically to a complete query. Switching views starts a fresh frame.
    for (unsigned view_index = 0; view_index < 2; ++view_index) {
        auto test_camera = camera;
        test_camera.world_to_camera[12] = .25F * float(view_index);
        const auto reference = detail::vulkan_occupancy(
            backend, vk, points, test_camera, {});
        const auto expected_occ = reference.occupancy.to_vector();
        const auto expected_inside = reference.inside.to_vector();
        for (std::size_t begin = 0; begin < points.shape()[0]; begin += 257) {
            const auto end = std::min(begin + 257, points.shape()[0]);
            const auto chunk = detail::vulkan_occupancy(
                backend, vk, points.slice(0, begin, end), test_camera, {}, begin != 0);
            const auto occ = chunk.occupancy.to_vector();
            const auto inside = chunk.inside.to_vector();
            for (std::size_t i = 0; i < occ.size(); ++i) {
                require(occ[i] == expected_occ[begin + i], "PAM prepared occupancy chunk parity");
                require(inside[i] == expected_inside[begin + i], "PAM prepared frustum chunk parity");
            }
        }
    }
    if (has_cuda) {
        const auto reference = Rasterizer().evaluate_occupancy(
            cpu, Tensor::from_vector(queries, {queries.size() / 3, std::size_t{3}}, Device::CUDA),
            camera);
        const auto expected_occ = reference.occupancy.to_vector(),
                   actual_occ = actual.occupancy.to_vector();
        const auto expected_inside = reference.inside.to_vector(),
                   actual_inside = actual.inside.to_vector();
        for (std::size_t i = 0; i < actual_occ.size(); ++i) {
            require(expected_inside[i] == actual_inside[i], "PAM Vulkan frustum parity");
            if (std::abs(expected_occ[i] - actual_occ[i]) > 3e-5F)
                std::cerr << "occupancy mismatch " << i << " cuda=" << expected_occ[i]
                          << " vulkan=" << actual_occ[i] << '\n';
            require(std::abs(expected_occ[i] - actual_occ[i]) < 3e-5F,
                    "PAM Vulkan occupancy parity");
        }
    } else {
        const auto inside = actual.inside.to_vector();
        const auto occ = actual.occupancy.to_vector();
        const auto n = queries.size() / 3;
        require(!inside[n - 5] && !inside[n - 4] && occ[n - 5] == 0.F && occ[n - 4] == 0.F,
                "Vulkan-only occupancy invalid-frustum handling");
    }
    {
        Camera empty_camera = camera;
        empty_camera.world_to_camera[14] = -4.F;
        empty_camera.position = {0, 0, 4};
        std::shared_ptr<void> empty_backend;
        auto visible = Tensor::from_vector(std::vector<float>{0, 0, 5},
                                           {std::size_t{1}, std::size_t{3}}, Device::Vulkan);
        const auto empty = detail::vulkan_occupancy(empty_backend, vk, visible, empty_camera, {});
        require(empty.inside.to_vector()[0] && empty.occupancy.to_vector()[0] == 0.F,
                "Visible empty ray must vote zero occupancy");
    }
#ifdef PHOTARA_HAS_CGAL
    photara::mvs::MvsScene scene;
    photara::mvs::MvsView view;
    view.pose.C = {0, 0, -4};
    view.fx = view.fy = 50.F;
    view.cx = view.cy = 31.5F;
    view.width = view.height = 64;
    scene.views.push_back(view);
    photara::mvs::Mesh seed;
    seed.vertices = shell;
    seed.faces = {{0, 2, 4}, {2, 1, 4}, {1, 3, 4}, {3, 0, 4},
                  {2, 0, 5}, {1, 2, 5}, {3, 1, 5}, {0, 3, 5}};
    PamMeshOptions options;
    options.max_points = 64;
    options.scene_crop = false;
    options.oversampling_factor = 8;
    options.occupancy_chunk_size = 257;
    options.refinement_steps = 2;
    options.vector_field_neighbors = 6;
    options.points_per_tetrahedron = 1;
    options.occupancy_iso_value = .05F;
    options.vacancy_threshold = 1.F;
    for (bool masked : {false, true}) {
        if (masked) {
            scene.views[0].foreground_mask.assign(4096, 255);
            for (unsigned y = 0; y < 64; ++y)
                for (unsigned x = 0; x < 24; ++x)
                    scene.views[0].foreground_mask[64 * y + x] = 0;
        }
        auto chunk_scene = scene;
        auto shifted_view = scene.views[0];
        shifted_view.pose.C.x() += .25;
        chunk_scene.views.push_back(shifted_view);
        auto a = extract_pam_mesh(cpu, chunk_scene, seed, {}, options);
        auto b = extract_pam_mesh(vk, chunk_scene, seed, {}, options);
        require(!b.mesh.faces.empty() &&
                    a.candidate_cloud.points.size() == b.candidate_cloud.points.size(),
                "PAM Vulkan extraction failed");
        for (std::size_t i = 0; i < a.candidate_cloud.points.size(); ++i)
            require((a.candidate_cloud.points[i].position - b.candidate_cloud.points[i].position)
                            .norm() < 1e-4F,
                    "PAM Vulkan candidate parity");
        require(a.occupied_tetrahedron_count == b.occupied_tetrahedron_count,
                "PAM Vulkan tetrahedron classification parity");
        auto single_chunk_options = options;
        single_chunk_options.occupancy_chunk_size = 4096;
        const auto single_chunk = extract_pam_mesh(vk, chunk_scene, seed, {}, single_chunk_options);
        require(single_chunk.occupied_tetrahedron_count == b.occupied_tetrahedron_count &&
                    single_chunk.mesh.faces == b.mesh.faces &&
                    single_chunk.candidate_cloud.points.size() == b.candidate_cloud.points.size(),
                "PAM prepared view changed extraction topology");
        for (std::size_t i = 0; i < b.candidate_cloud.points.size(); ++i)
            require((single_chunk.candidate_cloud.points[i].position -
                     b.candidate_cloud.points[i].position).norm() < 1e-6F,
                    "PAM prepared view changed candidate positions");
    }
    // Also exercise the native empty-seed pivot path.
    scene.views[0].foreground_mask.clear();
    options.refinement_steps = 0;
    auto generated = extract_pam_mesh(vk, scene, {}, {}, options);
    require(!generated.seed_mesh.faces.empty() && !generated.mesh.faces.empty(),
            "PAM Vulkan generated pivot mesh failed");
    auto reference_seed = extract_pam_mesh(cpu, scene, {}, {}, options);
    require(reference_seed.seed_mesh.vertices.size() == generated.seed_mesh.vertices.size(),
            "PAM binary-search pivot count parity");
    for (std::size_t i = 0; i < generated.seed_mesh.vertices.size(); ++i)
        require((reference_seed.seed_mesh.vertices[i] - generated.seed_mesh.vertices[i]).norm() < 1e-4F,
                "PAM binary-search pivot position parity");
    // Held-out views must abstain from both occupancy and mesh sampling.
    // This deliberately incompatible camera would remove the training surface.
    auto held_out = view;
    held_out.foreground_mask.assign(4096, 0);
    auto split_scene = scene;
    split_scene.views.insert(split_scene.views.begin(), held_out);
    TrainingOptions split_options;
    split_options.evaluation_split_every = 2;
    auto split_result = extract_pam_mesh(vk, split_scene, {}, split_options, options);
    require(split_result.candidate_cloud.points.size() == generated.candidate_cloud.points.size(),
            "PAM training-view split candidate count");
    for (std::size_t i = 0; i < generated.candidate_cloud.points.size(); ++i)
        require((split_result.candidate_cloud.points[i].position -
                 generated.candidate_cloud.points[i].position).norm() < 1e-6F,
                "PAM held-out view affected extraction");
    auto refill_options = options;
    refill_options.oversampling_factor = 1;
    refill_options.max_resample_rounds = 1;
    auto probe = extract_pam_mesh(vk, scene, seed, {}, refill_options);
    std::vector<float> probe_points;
    for (const auto& point : probe.candidate_cloud.points)
        probe_points.insert(probe_points.end(), point.position.data(), point.position.data() + 3);
    auto probe_tensor = Tensor::from_vector(probe_points,
        {probe.candidate_cloud.points.size(), std::size_t{3}}, Device::Vulkan);
    std::shared_ptr<void> probe_backend;
    auto probe_values = detail::vulkan_occupancy(probe_backend, vk, probe_tensor,
        camera_from_mvs_view(view), {}).occupancy.to_vector();
    std::sort(probe_values.begin(), probe_values.end());
    const float low = probe_values[probe_values.size() / 4];
    const float high = probe_values[3 * probe_values.size() / 4];
    require(high > low, "PAM refill fixture has no occupancy variation");
    refill_options.occupancy_iso_value = .5F * (low + high);
    refill_options.vacancy_threshold = .5F * (high - low);
    auto partial = extract_pam_mesh(vk, scene, seed, {}, refill_options);
    require(partial.candidate_cloud.points.size() < refill_options.max_points,
            "PAM refill fixture must reject some initial samples");
    refill_options.max_resample_rounds = 0;
    auto filled = extract_pam_mesh(vk, scene, seed, {}, refill_options);
    require(filled.candidate_cloud.points.size() == refill_options.max_points,
            "PAM unlimited resampling did not fill the candidate budget");
    refill_options.occupancy_iso_value = .999F;
    refill_options.vacancy_threshold = 0.F;
    bool stalled = false;
    try {
        (void)extract_pam_mesh(vk, scene, seed, {}, refill_options);
    } catch (const std::runtime_error& error) {
        stalled = std::string(error.what()).find("three sampling rounds") != std::string::npos;
    }
    require(stalled, "PAM stalled sampling did not report an error");
    // Match Open3D/GW scene cropping: radius = 1.1*max distance from camera
    // mean, but the cube is centered at zero; crossing faces are discarded.
    auto crop_scene = scene;
    auto second_view = view;
    second_view.pose.C = {0, 0, 4};
    crop_scene.views.push_back(second_view);
    auto crop_options = options;
    crop_options.scene_crop = true;
    auto crossing_seed = seed;
    crossing_seed.vertices.emplace_back(10.F, 0.F, 0.F);
    crossing_seed.faces.emplace_back(0, 2, static_cast<int>(crossing_seed.vertices.size() - 1));
    auto cropped = extract_pam_mesh(vk, crop_scene, crossing_seed, {}, crop_options);
    auto crop_reference = extract_pam_mesh(vk, crop_scene, seed, {}, crop_options);
    require(cropped.candidate_cloud.points.size() == crop_reference.candidate_cloud.points.size(),
            "PAM scene crop candidate count");
    for (std::size_t i = 0; i < cropped.candidate_cloud.points.size(); ++i)
        require((cropped.candidate_cloud.points[i].position -
                 crop_reference.candidate_cloud.points[i].position).norm() < 1e-6F,
                "PAM scene crop retained a crossing face");
    // Translating camera centers must not translate GW's origin-centered box.
    for (auto& crop_view : crop_scene.views) crop_view.pose.C.x() += 20;
    bool empty_crop = false;
    try {
        auto far_seed = seed;
        for (auto& vertex : far_seed.vertices) vertex.x() += 20;
        (void)extract_pam_mesh(vk, crop_scene, far_seed, {}, crop_options);
    } catch (const std::runtime_error& error) {
        empty_crop = std::string(error.what()).find("empty seed mesh") != std::string::npos;
    }
    require(empty_crop, "PAM scene crop incorrectly followed the camera mean");
#endif
    std::cout
        << "PAM Vulkan field, projection, occupancy and extraction tests passed; CUDA available="
        << has_cuda << "\n";
}
#endif
