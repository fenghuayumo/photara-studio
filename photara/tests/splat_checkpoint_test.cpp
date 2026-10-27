#include "splat/checkpoint.hpp"
#include "splat/trainer.hpp"
#include "io/image.hpp"
#include "mvs/export.hpp"
#include "project/archive.hpp"
#include "project/document.hpp"
#include "../src/splat/checkpoint.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

using namespace photara;
static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static void compare(const splat::GaussianModel& a, const splat::GaussianModel& b) {
    require(a.size() == b.size(), "Resumed topology differs from uninterrupted training");
    for (const auto& pair : {std::pair{&a.means, &b.means}, {&a.log_scales, &b.log_scales},
        {&a.quaternions, &b.quaternions}, {&a.opacity_logits, &b.opacity_logits},
        {&a.sh, &b.sh}, {&a.normal_features, &b.normal_features}, {&a.filter_3d, &b.filter_3d}}) {
        require(pair.first->is_valid() == pair.second->is_valid(), "Resumed optional model state differs");
        if (!pair.first->is_valid()) continue;
        const auto x = pair.first->to_vector(); const auto y = pair.second->to_vector();
        require(x.size() == y.size(), "Resumed tensor size differs");
        for (std::size_t i = 0; i < x.size(); ++i)
            if (!std::isfinite(y[i]) || std::abs(x[i] - y[i]) >= 2e-5F) {
                std::cerr << "Tensor mismatch at " << i << " baseline=" << x[i] << " resumed=" << y[i] << "\n";
                throw std::runtime_error("Resumed model differs from uninterrupted training");
            }
    }
}
int main(int argc, char** argv) {
    const bool vulkan = argc > 1 && std::string(argv[1]) == "--vulkan";
    const auto root = std::filesystem::temp_directory_path() /
        ("photara-resume-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directories(root);
        mvs::MvsScene scene;
        const auto image = root / "frame.png";
        io::save_rgb_png(io::RgbImage{32, 32, std::vector<std::uint8_t>(32 * 32 * 3, 128)}, image);
        for (unsigned i = 0; i < 3; ++i) {
            mvs::MvsView view;
            view.id = i; view.path = image;
            view.width = view.height = view.src_width = view.src_height = 32;
            view.fx = view.fy = view.src_fx = view.src_fy = 20.F;
            view.cx = view.cy = view.src_cx = view.src_cy = 15.5F;
            view.pose.C.x() = 0.1 * i;
            scene.views.push_back(view);
        }
        for (float x : {-0.2F, 0.2F}) {
            mvs::DensePoint point;
            point.position = {x, 0.F, 2.F}; point.normal = mvs::Vec3f::UnitZ();
            point.color = mvs::Vec3f::Constant(0.5F);
            scene.dense_cloud.points.push_back(point);
        }
        splat::TrainingOptions options;
        options.backend = vulkan ? splat::TrainingBackend::vulkan : splat::TrainingBackend::cuda;
        options.iterations = 10; options.sh_degree = 1; options.sh_degree_interval = 3;
        options.input_is_dense = false; options.enable_densification = true;
        options.refine_start_iter = 1; options.refine_stop_iter = 9; options.grow_stop_iter = 9;
        options.refine_every = 3; options.refine_stop_num_iter = 0;
        options.densification_cap = 16; options.densify_gradient_threshold = -1.F;
        options.densify_select_fraction = 1.F; options.maximum_scale_fraction = 1.F;
        options.prune_opacity = 0.F; options.opacity_reset_every = 0;
        options.log_interval = 1; options.use_ppisp = true; options.use_bilateral_grid = true;
        options.background_noise_strength = 0.1F;
        // Keep this tiny scene away from crop/reseed discontinuities so GPU
        // reduction roundoff cannot change its topology between identical runs.
        options.initialize_scale_from_knn = true; options.initial_scale = 0.1F;
        options.mean_noise_weight = 0.F;
        options.progressive_resolution = true; options.progressive_resolution_interval = 3;
        options.progressive_initial_scale = 0.5F;
        options.filter_mode = splat::SplatFilterMode::mip_3d; options.filter_3d_update_interval = 3;
        options.training_async_upload = false;
        scene.subject_bounds.valid = true;
        scene.subject_bounds.center = {0.F, 0.F, 2.F};
        scene.subject_bounds.half_extent = {1.F, 1.F, 1.F};
        options.live_subject_bounds_file = root / "bounds.txt";
        mvs::save_subject_bounds(scene.subject_bounds, options.live_subject_bounds_file);
        const auto uninterrupted = splat::Trainer(options).train(scene);
        require(uninterrupted.filter_mode == splat::SplatFilterMode::mip_3d,
            "Densification lost the 3D filter mode");
        compare(uninterrupted, splat::Trainer(options).train(scene));
        auto light = options;
        const auto light_cache = root / "light_cache";
        light.checkpoint_path = light_cache / "training.checkpoint";
        light.checkpoint_interval_seconds = 0;
        light.checkpoint_interval_iterations = 0;
        splat::Trainer(light).train(scene, [](const splat::TrainingProgress& p) { return p.iteration < 3; });
        const auto light_info = splat::inspect_training_checkpoint(light.checkpoint_path);
        require(light_info && light_info->iteration == 3, "Model checkpoint metadata missing");
        require(std::filesystem::file_size(light.checkpoint_path) < 4096, "Model manifest contains tensor payloads");
        require(light_info->model_path.extension() == ".ply" && std::filesystem::exists(light_info->model_path),
            "Model checkpoint does not reference an inspectable PLY");
        const auto standalone = root / "standalone.ascan";
        auto snapshot = project::Archive::from_bytes(splat::training_checkpoint_bytes(light.checkpoint_path));
        require(snapshot.has(project::ChunkType::gaussians) && snapshot.has(project::ChunkType::training_model_info) &&
            !snapshot.has(project::ChunkType::training_checkpoint), "Project snapshot duplicates optimizer/model tensors");
        snapshot.save(standalone);
        auto blocked_manifest = light.checkpoint_path; blocked_manifest += ".tmp";
        std::filesystem::create_directory(blocked_manifest);
        bool light_save_failed = false;
        try { splat::detail::save_model_checkpoint(light.checkpoint_path, *light_info, splat::checkpoint_model(light.checkpoint_path)); }
        catch (...) { light_save_failed = true; }
        require(light_save_failed && splat::inspect_training_checkpoint(light.checkpoint_path)->model_path == light_info->model_path,
            "Failed manifest publication replaced the last valid model reference");
        require(std::filesystem::exists(light_info->model_path), "Failed publication deleted the last valid PLY");
        std::filesystem::remove_all(light_cache);
        require(splat::inspect_training_checkpoint(standalone).has_value(), "Portable project lost model progress");
        light.checkpoint_path = root / "restored_cache" / "training.checkpoint";
        splat::restore_training_checkpoint(splat::training_checkpoint_bytes(standalone), light.checkpoint_path);
        light.resume_checkpoint = true;
        unsigned first_model_iteration = 0;
        const auto warm_model = splat::Trainer(light).train(scene, [&](const splat::TrainingProgress& p) {
            if (!first_model_iteration) first_model_iteration = p.iteration;
            return true;
        });
        require(first_model_iteration == 4 && warm_model.size() > 0, "Model restart did not continue from saved progress");
        for (float value : warm_model.means.to_vector()) require(std::isfinite(value), "Model restart produced invalid means");
        std::filesystem::remove(standalone);
        options.checkpoint_path = light.checkpoint_path;
        options.resume_checkpoint = true;
        {
            splat::detail::TrainingCheckpointLease lease(options.checkpoint_path);
            require(splat::training_checkpoint_in_use(options.checkpoint_path), "Active checkpoint lease was not detected");
            bool locked = false;
            try { splat::Trainer(options).train(scene); } catch (...) { locked = true; }
            require(locked, "Concurrent checkpoint writer was accepted");
        }
        require(!splat::training_checkpoint_in_use(options.checkpoint_path), "Checkpoint lease was not released");
        auto changed = options; changed.means_lr *= 2;
        bool rejected = false;
        try { splat::Trainer(changed).train(scene); } catch (...) { rejected = true; }
        require(rejected, "Changed training settings accepted a stale checkpoint");
        // Retired tensor checkpoints are not inspected, restored or migrated.
        const auto retired = root / "retired.checkpoint";
        auto old = project::Archive::create();
        old.set_chunk(project::ChunkType::training_checkpoint_info, splat::training_checkpoint_info_bytes(*light_info));
        old.set_chunk(project::ChunkType::training_checkpoint, {1, 2, 3});
        old.save(retired);
        require(!splat::inspect_training_checkpoint(retired), "Retired tensor checkpoint was accepted");
        bool restore_rejected = false;
        try { splat::restore_training_checkpoint(old.encode(), root / "unsupported.checkpoint"); }
        catch (...) { restore_rejected = true; }
        require(restore_rejected && !std::filesystem::exists(root / "unsupported.checkpoint"),
            "Retired checkpoint was restored");
        auto unsupported = options; unsupported.checkpoint_path = retired;
        rejected = false;
        try { splat::Trainer(unsupported).train(scene); } catch (...) { rejected = true; }
        require(rejected, "Trainer accepted retired tensor state");
        // Clearing an embedded progress record must preserve project content.
        auto native_project = project::Archive::from_bytes(splat::training_checkpoint_bytes(options.checkpoint_path));
        const auto marker_type = static_cast<project::ChunkType>(100);
        native_project.set_chunk(marker_type, {9, 8, 7});
        const auto native_path = root / "embedded.ascan"; native_project.save(native_path);
        splat::remove_training_checkpoint(native_path);
        auto cleared = project::Archive::open(native_path);
        require(!splat::inspect_training_checkpoint(native_path) && cleared.has(project::ChunkType::gaussians) &&
            cleared.chunk(marker_type) == std::vector<std::uint8_t>{9, 8, 7},
            "Clearing progress discarded saved project content");
        // Cooperative save/hold acknowledgement: cancel releases the trainer.
        options.resume_checkpoint = false;
        options.checkpoint_control_file = root / "save_control";
        auto reply = options.checkpoint_control_file; reply += ".reply";
        std::jthread controller([&] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (std::chrono::steady_clock::now() < deadline) {
                std::ifstream input(reply); std::string id, response;
                std::getline(input, id); std::getline(input, response);
                if (id == "test-request" && response.starts_with("saved ")) {
                    std::error_code error;
                    std::filesystem::remove(options.checkpoint_control_file, error);
                    if (!error) return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            std::error_code ignored; std::filesystem::remove(options.checkpoint_control_file, ignored);
        });
        { std::ofstream request(options.checkpoint_control_file); request << "test-request\n"; }
        splat::Trainer(options).train(scene);
        controller.join();
        require(std::filesystem::exists(reply), "Save request was not acknowledged");
        { std::ifstream input(reply); std::string id, response; std::getline(input, id); std::getline(input, response);
          require(id == "test-request" && response.starts_with("saved "), "Save request failed"); }
        // Automatic and explicit save failures must leave training alive and
        // preserve the last complete generation. Blocking the staged write
        // exercises the actual trainer error path without exhausting disk space.
        const auto before_failure = splat::training_checkpoint_bytes(options.checkpoint_path);
        auto blocked_stage = options.checkpoint_path; blocked_stage += ".tmp";
        std::filesystem::create_directory(blocked_stage);
        std::filesystem::remove(reply);
        { std::ofstream request(options.checkpoint_control_file); request << "failed-request\n"; }
        unsigned failure_progress = 0;
        splat::Trainer(options).train(scene, [&](const splat::TrainingProgress& p) {
            failure_progress = p.iteration; return true;
        });
        require(failure_progress == options.iterations, "Failed checkpoint stopped automatic training");
        require(before_failure == splat::training_checkpoint_bytes(options.checkpoint_path),
            "Failed checkpoint overwrote the last good generation");
        { std::ifstream input(reply); std::string id, response; std::getline(input, id); std::getline(input, response);
          require(id == "failed-request" && response.starts_with("error "), "Save failure was not acknowledged"); }
        std::filesystem::remove(options.checkpoint_control_file);
        std::filesystem::remove(blocked_stage);
        // Image content changes must invalidate the checkpoint even at the same path.
        auto modified = io::RgbImage{32, 32, std::vector<std::uint8_t>(32 * 32 * 3, 129)};
        io::save_rgb_png(modified, image); options.resume_checkpoint = true;
        rejected = false;
        try { splat::Trainer(options).train(scene); } catch (...) { rejected = true; }
        require(rejected, "Modified input image accepted a stale checkpoint");
        std::filesystem::remove_all(root);
        std::cout << "Training checkpoint resume tests passed (" << (vulkan ? "Vulkan" : "CUDA") << ")\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\nArtifacts: " << root << '\n'; return 1;
    }
}
