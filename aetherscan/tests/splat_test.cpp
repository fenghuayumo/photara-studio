#include "splat/trainer.hpp"
#include "splat/colmap.hpp"
#include "../src/splat/cuda_ops.hpp"
#include "io/image.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void write_binary(std::ostream& stream, const T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void require_finite(
    const tinytensor::Tensor& tensor, const char* message) {
    const std::vector<float> values = tensor.to_vector();
    require(!values.empty(), message);
    require(
        std::all_of(values.begin(), values.end(), [](const float value) {
            return std::isfinite(value);
        }),
        message);
}

void test_mvs_camera_conversion() {
    aetherscan::mvs::MvsView view;
    view.pose.R = Eigen::AngleAxisd(
        0.73, Eigen::Vector3d(0.2, 1.0, -0.1).normalized()).toRotationMatrix();
    view.pose.C = Eigen::Vector3d(-0.3, 0.25, 1.2);
    const aetherscan::splat::Camera camera =
        aetherscan::splat::camera_from_mvs_view(view);
    const Eigen::Vector3d world(0.4, -0.2, 2.1);
    const Eigen::Vector3d expected =
        view.pose.transform_world_to_camera(world);
    const auto& matrix = camera.world_to_camera;
    const Eigen::Vector3d actual(
        matrix[0] * world.x() + matrix[4] * world.y() +
            matrix[8] * world.z() + matrix[12],
        matrix[1] * world.x() + matrix[5] * world.y() +
            matrix[9] * world.z() + matrix[13],
        matrix[2] * world.x() + matrix[6] * world.y() +
            matrix[10] * world.z() + matrix[14]);
    require(
        (actual - expected).norm() < 1e-5,
        "MVS to GGGS camera conversion changed projection");
}

void test_forward_backward() {
    using namespace aetherscan::splat;
    GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(
        std::vector<float>{0.F, 0.F, 2.F}, {1, 3},
        tinytensor::Device::CUDA);
    model.log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>{std::log(0.15F), std::log(0.15F),
                           std::log(0.15F)},
        {1, 3}, tinytensor::Device::CUDA);
    model.quaternions = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, 0.F, 0.F, 0.F}, {1, 4},
        tinytensor::Device::CUDA);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        std::vector<float>{2.F}, {1, 1}, tinytensor::Device::CUDA);
    model.sh = tinytensor::Tensor::from_vector(
        std::vector<float>{0.5F, 0.25F, 0.1F}, {1, 1, 3},
        tinytensor::Device::CUDA);
    model.sh_degree = 0;

    Camera camera;
    camera.world_to_camera[0] = 1.F;
    camera.world_to_camera[5] = 1.F;
    camera.world_to_camera[10] = 1.F;
    camera.world_to_camera[15] = 1.F;
    camera.fx = 40.F;
    camera.fy = 40.F;
    camera.cx = 15.5F;
    camera.cy = 15.5F;
    camera.width = 32;
    camera.height = 32;

    Rasterizer rasterizer;
    RenderResult rendered = rasterizer.forward(model, camera);
    require(rendered.rendered_instances > 0, "Gaussian was not rasterized");
    const std::vector<float> alpha = rendered.alpha.to_vector();
    require(
        *std::max_element(alpha.begin(), alpha.end()) > 0.F,
        "Rasterized alpha is empty");

    const std::size_t pixels =
        static_cast<std::size_t>(camera.width) * camera.height;
    const auto grad_color = tinytensor::Tensor::from_vector(
        std::vector<float>(3 * pixels, 1.F / static_cast<float>(pixels)),
        {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    const auto grad_alpha = tinytensor::Tensor::zeros(
        {camera.height, camera.width}, tinytensor::Device::CUDA);
    const auto grad_depth = tinytensor::Tensor::zeros(
        {camera.height, camera.width}, tinytensor::Device::CUDA);
    const auto grad_normal = tinytensor::Tensor::zeros(
        {3, camera.height, camera.width}, tinytensor::Device::CUDA);
    const ModelGradients gradients = rasterizer.backward(
        model, rendered, grad_color, grad_alpha, grad_depth, grad_normal);
    require_finite(gradients.means, "Non-finite mean gradient");
    require_finite(gradients.log_scales, "Non-finite scale gradient");
    require_finite(gradients.quaternions, "Non-finite quaternion gradient");
    require_finite(gradients.opacity_logits, "Non-finite opacity gradient");
    require_finite(gradients.sh, "Non-finite SH gradient");
}

void test_adam_rejects_non_finite_gradients() {
    using namespace aetherscan::splat;
    auto parameter = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, 2.F, 3.F}, {3}, tinytensor::Device::CUDA);
    const auto gradient = tinytensor::Tensor::from_vector(
        std::vector<float>{
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
            -std::numeric_limits<float>::infinity()},
        {3}, tinytensor::Device::CUDA);
    auto state = detail::make_adam_state(parameter);
    detail::adam_step(
        parameter, gradient, state, 1e-3F, 1, TrainingOptions{});
    require_finite(parameter, "Adam accepted a non-finite gradient");
    require_finite(state.first, "Adam first moment became non-finite");
    require_finite(state.second, "Adam second moment became non-finite");
    const auto values = parameter.to_vector();
    require(
        values == std::vector<float>({1.F, 2.F, 3.F}),
        "Adam changed parameters for rejected gradients");
}

void test_mask_loading() {
    using namespace aetherscan;
    const auto root = std::filesystem::temp_directory_path() /
                      "aetherscan_splat_mask_test";
    const auto masks = root / "masks";
    std::filesystem::create_directories(masks);
    io::RgbImage source{2, 2, std::vector<std::uint8_t>(12, 128)};
    io::RgbImage mask{2, 2, {
        255, 255, 255, 0, 0, 0,
        0, 0, 0, 255, 255, 255}};
    const auto image_path = root / "frame.png";
    io::save_rgb_png(source, image_path);
    io::save_rgb_png(mask, masks / "frame.png");
    mvs::MvsView view;
    view.path = image_path;
    view.width = view.src_width = 2;
    view.height = view.src_height = 2;
    view.fx = view.fy = view.src_fx = view.src_fy = 1.F;
    view.cx = view.cy = view.src_cx = view.src_cy = 0.5F;
    splat::TrainingOptions options;
    options.use_mask = true;
    options.mask_dir = masks;
    const splat::TrainingView training =
        splat::make_training_view(view, options);
    require(training.has_mask, "GGGS did not load the matching mask file");
    require(
        training.mask.to_vector() == std::vector<float>({1.F, 0.F, 0.F, 1.F}),
        "GGGS mask threshold or pixel mapping differs from pygsplat");
    std::filesystem::remove_all(root);
}

void test_colmap_text_loading() {
    using namespace aetherscan;
    const auto root = std::filesystem::temp_directory_path() /
                      "aetherscan_colmap_splat_test";
    const auto model = root / "sparse" / "0";
    const auto images = root / "images";
    std::filesystem::create_directories(model);
    std::filesystem::create_directories(images);
    io::save_rgb_png(
        io::RgbImage{4, 3, std::vector<std::uint8_t>(36, 127)},
        images / "frame.png");
    {
        std::ofstream stream(model / "cameras.txt");
        stream << "1 OPENCV 4 3 100 101 2 1.5 0.01 -0.02 0.001 -0.002\n";
    }
    {
        std::ofstream stream(model / "images.txt");
        stream << "7 1 0 0 0 1 2 3 1 frame.png\n";
        stream << "2 1 42\n";
    }
    {
        std::ofstream stream(model / "points3D.txt");
        stream << "42 0.1 0.2 4 10 20 30 0.5 7 0\n";
    }
    const auto loaded = splat::load_colmap_scene(root, images);
    require(!loaded.binary, "COLMAP text model was reported as binary");
    require(loaded.scene.views.size() == 1, "COLMAP camera pose was not loaded");
    require(
        loaded.scene.dense_cloud.points.size() == 1 &&
            loaded.scene.sparse_points.size() == 1,
        "COLMAP sparse point was not loaded");
    const auto& view = loaded.scene.views.front();
    require(
        std::abs(view.fx - 100.F) < 1e-6F &&
            std::abs(view.fy - 101.F) < 1e-6F &&
            std::abs(view.k1 - 0.01F) < 1e-6F,
        "COLMAP intrinsics were decoded incorrectly");
    require(
        (view.pose.C - Eigen::Vector3d(-1, -2, -3)).norm() < 1e-9,
        "COLMAP world-to-camera translation was decoded incorrectly");
    const auto& point = loaded.scene.dense_cloud.points.front();
    require(
        point.views == std::vector<mvs::Index>{0} &&
            (point.position - mvs::Vec3f(0.1F, 0.2F, 4.F)).norm() < 1e-6F,
        "COLMAP track mapping or sparse position is wrong");

    {
        std::ofstream stream(model / "cameras.bin", std::ios::binary);
        write_binary<std::uint64_t>(stream, 1);
        write_binary<std::uint32_t>(stream, 1);
        write_binary<std::int32_t>(stream, 1);  // PINHOLE
        write_binary<std::uint64_t>(stream, 4);
        write_binary<std::uint64_t>(stream, 3);
        for (const double value : {100.0, 101.0, 2.0, 1.5})
            write_binary(stream, value);
    }
    {
        std::ofstream stream(model / "images.bin", std::ios::binary);
        write_binary<std::uint64_t>(stream, 1);
        write_binary<std::uint32_t>(stream, 7);
        for (const double value : {1.0, 0.0, 0.0, 0.0})
            write_binary(stream, value);
        for (const double value : {1.0, 2.0, 3.0})
            write_binary(stream, value);
        write_binary<std::uint32_t>(stream, 1);
        stream.write("frame.png", 10);
        write_binary<std::uint64_t>(stream, 0);
    }
    {
        std::ofstream stream(model / "points3D.bin", std::ios::binary);
        write_binary<std::uint64_t>(stream, 1);
        write_binary<std::uint64_t>(stream, 42);
        for (const double value : {0.1, 0.2, 4.0})
            write_binary(stream, value);
        for (const std::uint8_t value : {10, 20, 30})
            write_binary(stream, value);
        write_binary(stream, 0.5);
        write_binary<std::uint64_t>(stream, 1);
        write_binary<std::uint32_t>(stream, 7);
        write_binary<std::uint32_t>(stream, 0);
    }
    const auto binary_loaded = splat::load_colmap_scene(root, images);
    require(
        binary_loaded.binary && binary_loaded.scene.views.size() == 1 &&
            binary_loaded.scene.dense_cloud.points.size() == 1,
        "COLMAP binary model was not loaded");
    std::filesystem::remove_all(root);
}

void test_mask_loss_modes() {
    using namespace aetherscan::splat;
    RenderResult rendered;
    rendered.color = tinytensor::Tensor::from_vector(
        std::vector<float>(6, 0.8F), {3, 1, 2}, tinytensor::Device::CUDA);
    rendered.alpha = tinytensor::Tensor::from_vector(
        std::vector<float>{0.8F, 0.8F}, {1, 2}, tinytensor::Device::CUDA);
    rendered.median_depth = tinytensor::Tensor::zeros(
        {1, 2}, tinytensor::Device::CUDA);
    rendered.normal = tinytensor::Tensor::zeros(
        {3, 1, 2}, tinytensor::Device::CUDA);
    TrainingView target;
    target.camera.width = 2;
    target.camera.height = 1;
    target.rgb = tinytensor::Tensor::from_vector(
        std::vector<float>(6, 0.2F), {3, 1, 2}, tinytensor::Device::CUDA);
    target.depth = tinytensor::Tensor::zeros({1, 2}, tinytensor::Device::CUDA);
    target.normal = tinytensor::Tensor::zeros(
        {3, 1, 2}, tinytensor::Device::CUDA);
    target.mask = tinytensor::Tensor::from_vector(
        std::vector<float>{1.F, 0.F}, {1, 2}, tinytensor::Device::CUDA);
    target.has_mask = true;
    TrainingOptions options;
    options.use_mask = true;
    options.use_mvs_depth = false;
    options.use_mvs_normals = false;

    options.alpha_mode = AlphaMode::masked;
    auto loss = detail::compute_training_loss(rendered, target, options, true);
    auto rgb_gradient = loss.color.to_vector();
    auto alpha_gradient = loss.alpha.to_vector();
    require(rgb_gradient[0] != 0.F && rgb_gradient[1] == 0.F,
            "masked mode did not restrict RGB supervision to foreground");
    require(std::abs(alpha_gradient[0]) < 1e-6F &&
                std::abs(alpha_gradient[1] - 0.5F) < 1e-6F,
            "masked mode background-alpha penalty differs from pygsplat");
    require(std::abs(loss.alpha_value - 0.4F) < 1e-6F,
            "masked mode reported the wrong alpha loss");

    options.alpha_mode = AlphaMode::transparent;
    options.match_alpha_weight = 0.25F;
    loss = detail::compute_training_loss(rendered, target, options, true);
    alpha_gradient = loss.alpha.to_vector();
    require(std::abs(alpha_gradient[0] + 0.15625F) < 1e-5F &&
                std::abs(alpha_gradient[1] - 0.625F) < 1e-5F,
            "transparent mode BCE gradient differs from pygsplat");

    target.has_mask = false;
    loss = detail::compute_training_loss(rendered, target, options, false);
    rgb_gradient = loss.color.to_vector();
    alpha_gradient = loss.alpha.to_vector();
    require(rgb_gradient[0] != 0.F && rgb_gradient[1] != 0.F,
            "missing per-view mask incorrectly suppressed RGB training");
    require(std::all_of(alpha_gradient.begin(), alpha_gradient.end(),
                        [](float value) { return value == 0.F; }),
            "missing per-view mask incorrectly enabled alpha supervision");
}

void test_ssim_loss_and_scale_constraint() {
    using namespace aetherscan::splat;
    constexpr std::size_t side = 13;
    constexpr std::size_t pixels = side * side;
    std::vector<float> prediction(3 * pixels, 0.4F);
    std::vector<float> reference(3 * pixels, 0.4F);
    prediction[6 * side + 6] = 0.8F;
    RenderResult rendered;
    rendered.color = tinytensor::Tensor::from_vector(
        prediction, {3, side, side}, tinytensor::Device::CUDA);
    rendered.alpha = tinytensor::Tensor::zeros({side, side}, tinytensor::Device::CUDA);
    rendered.median_depth = tinytensor::Tensor::zeros(
        {side, side}, tinytensor::Device::CUDA);
    rendered.normal = tinytensor::Tensor::zeros(
        {3, side, side}, tinytensor::Device::CUDA);
    TrainingView target;
    target.camera.width = target.camera.height = side;
    target.rgb = tinytensor::Tensor::from_vector(
        reference, {3, side, side}, tinytensor::Device::CUDA);
    target.depth = tinytensor::Tensor::zeros({side, side}, tinytensor::Device::CUDA);
    target.normal = tinytensor::Tensor::zeros(
        {3, side, side}, tinytensor::Device::CUDA);
    target.mask = tinytensor::Tensor::from_vector(
        std::vector<float>(pixels, 1.F), {side, side}, tinytensor::Device::CUDA);
    TrainingOptions options;
    options.ssim_weight = 1.F;
    options.use_mvs_depth = false;
    options.use_mvs_normals = false;
    const auto loss = detail::compute_training_loss(
        rendered, target, options, true);
    require(
        std::abs(loss.rgb - 0.299324721F) < 2e-5F,
        "fused SSIM forward does not match the Python CUDA reference");
    require_finite(loss.color, "SSIM produced a non-finite gradient");
    const auto gradients = loss.color.to_vector();
    require(
        std::any_of(gradients.begin(), gradients.end(),
                    [](float value) { return std::abs(value) > 1e-6F; }),
        "SSIM did not backpropagate into rendered color");
    const std::size_t center = 6 * side + 6;
    require(
        std::abs(gradients[center] - 0.152631640F) < 2e-5F,
        "fused SSIM backward does not match the Python CUDA reference");
    const float gradient_sum = std::accumulate(
        gradients.begin(), gradients.end(), 0.F);
    const float gradient_abs_sum = std::accumulate(
        gradients.begin(), gradients.end(), 0.F,
        [](float total, float value) { return total + std::abs(value); });
    require(
        std::abs(gradient_sum - 0.004020121F) < 2e-5F &&
            std::abs(gradient_abs_sum - 0.301243156F) < 2e-5F,
        "fused SSIM gradient field differs from the Python CUDA reference");

    auto log_scales = tinytensor::Tensor::from_vector(
        std::vector<float>{std::log(0.01F), std::log(1.F), std::log(0.001F)},
        {1, 3}, tinytensor::Device::CUDA);
    detail::constrain_scale_ratio(log_scales, 10.F);
    const auto constrained = log_scales.to_vector();
    const auto [minimum, maximum] = std::minmax_element(
        constrained.begin(), constrained.end());
    require(
        std::exp(*maximum - *minimum) <= 10.0001F,
        "Gaussian scale-ratio constraint was not applied");
}

void test_densification_strategies_and_dense_bypass() {
    using namespace aetherscan;
    const auto root = std::filesystem::temp_directory_path() /
                      "aetherscan_densification_test";
    std::filesystem::create_directories(root);
    const auto image_path = root / "frame.png";
    io::save_rgb_png(
        io::RgbImage{32, 32, std::vector<std::uint8_t>(32 * 32 * 3, 128)},
        image_path);
    mvs::MvsScene scene;
    mvs::MvsView view;
    view.id = 0;
    view.path = image_path;
    view.width = view.src_width = 32;
    view.height = view.src_height = 32;
    view.fx = view.fy = view.src_fx = view.src_fy = 20.F;
    view.cx = view.cy = view.src_cx = view.src_cy = 15.5F;
    scene.views.push_back(view);
    for (const float x : {-1.F, 1.F}) {
        mvs::DensePoint point;
        point.position = mvs::Vec3f(x, 0.F, 2.F);
        point.normal = mvs::Vec3f::UnitZ();
        point.color = mvs::Vec3f::Constant(0.5F);
        point.views = {0};
        scene.dense_cloud.points.push_back(point);
    }
    splat::TrainingOptions options;
    options.iterations = 3;
    options.sh_degree = 0;
    options.use_mvs_depth = false;
    options.use_mvs_normals = false;
    options.input_is_dense = false;
    options.densification_cap = 8;
    options.refine_start_iter = 1;
    options.refine_stop_iter = 3;
    options.refine_every = 2;
    options.densify_gradient_threshold = -1.F;
    options.densify_select_fraction = 1.F;
    options.log_interval = 1;
    for (const auto strategy : {
             splat::DensificationStrategy::default_strategy,
             splat::DensificationStrategy::adc_plus,
             splat::DensificationStrategy::adc_igs}) {
        options.densification_strategy = strategy;
        const auto model = splat::Trainer(options).train(scene);
        require(
            model.size() > scene.dense_cloud.points.size(),
            "sparse-input densification strategy did not grow Gaussians");
        require(model.size() <= options.densification_cap,
                "densification exceeded its hard Gaussian cap");
    }
    options.input_is_dense = true;
    options.densification_strategy = splat::DensificationStrategy::adc_igs;
    const auto dense_model = splat::Trainer(options).train(scene);
    require(
        dense_model.size() == scene.dense_cloud.points.size(),
        "dense point-cloud initialization incorrectly enabled densification");
    std::filesystem::remove_all(root);
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            std::cout << "SKIP: no CUDA device\n";
            return 0;
        }
        test_mvs_camera_conversion();
        test_forward_backward();
        test_adam_rejects_non_finite_gradients();
        test_mask_loading();
        test_colmap_text_loading();
        test_mask_loss_modes();
        test_ssim_loss_and_scale_constraint();
        test_densification_strategies_and_dense_bypass();
        std::cout << "splat tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "splat test failed: " << error.what() << '\n';
        return 1;
    }
}
