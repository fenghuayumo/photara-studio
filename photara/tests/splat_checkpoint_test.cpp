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
    if (argc == 3 && std::string(argv[1]) == "--migrate-project") {
        try {
            const auto path = std::filesystem::u8path(argv[2]);
            splat::migrate_training_checkpoints(path);
            const auto info = splat::inspect_training_checkpoint(path);
            require(info.has_value(), "Migrated project has no readable training state");
            std::cout << "Project migration verified: " << info->iteration << '/' << info->total_iterations << '\n';
            return 0;
        } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
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
        options.use_3d_filter = true; options.filter_3d_update_interval = 3;
        options.training_async_upload = false;
        scene.subject_bounds.valid = true;
        scene.subject_bounds.center = {0.F, 0.F, 2.F};
        scene.subject_bounds.half_extent = {1.F, 1.F, 1.F};
        options.live_subject_bounds_file = root / "bounds.txt";
        mvs::save_subject_bounds(scene.subject_bounds, options.live_subject_bounds_file);
        const auto uninterrupted = splat::Trainer(options).train(scene);
        compare(uninterrupted, splat::Trainer(options).train(scene));
        auto task_options = options;
        task_options.checkpoint_path = root / "training.checkpoint";
        task_options.checkpoint_interval_iterations = 0;
        task_options.checkpoint_interval_seconds = 0;
        splat::Trainer(task_options).train(scene,
            [](const splat::TrainingProgress& p) { return p.iteration < 3; });
        require(splat::inspect_training_checkpoint(task_options.checkpoint_path)->iteration == 3,
            "Task checkpoint did not retain interrupted progress");
        task_options.resume_checkpoint = true;
        compare(uninterrupted, splat::Trainer(task_options).train(scene));
        for (const auto& entry : std::filesystem::directory_iterator(root))
            require(entry.path().extension() != ".ascan", "Task training unexpectedly created a project file");
        options.checkpoint_path = splat::training_checkpoint_path(root / "project.ascan");
        options.checkpoint_interval_iterations = 2;
        const auto marker_type = static_cast<project::ChunkType>(100);
        auto native_project = project::Archive::create();
        project::Settings project_settings;
        project_settings.image_directory = root; project_settings.iterations = 10;
        project::write_settings(native_project, project_settings, options.checkpoint_path);
        native_project.set_chunk(marker_type, {9, 8, 7});
        native_project.save(options.checkpoint_path);
        // Cover both sides of a progressive-resolution / SH / refine boundary.
        for (unsigned stop_iteration : {3U, 4U}) {
            splat::remove_training_checkpoint(options.checkpoint_path);
            options.resume_checkpoint = false;
            splat::Trainer(options).train(scene, [=](const splat::TrainingProgress& p) {
                if (p.iteration == 2)
                    require(!splat::inspect_training_checkpoint(options.checkpoint_path),
                        "First iteration unexpectedly forced a checkpoint save");
                return p.iteration < stop_iteration;
            });
            const auto saved = splat::inspect_training_checkpoint(options.checkpoint_path);
            require(saved && saved->iteration == stop_iteration && saved->focus_bounds.valid,
                "Interrupted iteration or focus bounds were not saved");
            options.resume_checkpoint = true;
            unsigned first_iteration = 0;
            const auto resumed = splat::Trainer(options).train(scene, [&](const splat::TrainingProgress& p) {
                if (!first_iteration) first_iteration = p.iteration; return true;
            });
            require(first_iteration == stop_iteration + 1, "Resume restarted instead of continuing at the next iteration");
            compare(uninterrupted, resumed);
        }
        {
            splat::detail::TrainingCheckpointLease lease(options.checkpoint_path);
            require(splat::training_checkpoint_in_use(options.checkpoint_path), "Active checkpoint lease was not detected");
            bool locked = false;
            try { splat::Trainer(options).train(scene); } catch (...) { locked = true; }
            require(locked, "Concurrent checkpoint writer was accepted");
        }
        require(!splat::training_checkpoint_in_use(options.checkpoint_path), "Checkpoint lease was not released");
        // Corrupt the current training payload, not the in-project backup.
        {
            std::fstream corrupt(options.checkpoint_path, std::ios::binary | std::ios::in | std::ios::out);
            std::uint32_t count; corrupt.seekg(16); corrupt.read(reinterpret_cast<char*>(&count), 4);
            corrupt.seekg(24);
            struct Entry { std::uint32_t type, flags; std::uint64_t offset, size, checksum; };
            std::uint64_t position = 0;
            for (unsigned i = 0; i < count; ++i) {
                Entry entry; corrupt.read(reinterpret_cast<char*>(&entry), sizeof(entry));
                if (entry.type == std::uint32_t(project::ChunkType::training_checkpoint)) position = entry.offset + entry.size - 1;
            }
            require(position != 0, "Project did not contain the training-state chunk");
            char last; corrupt.seekg(position); corrupt.read(&last, 1);
            last ^= 0x55; corrupt.seekp(position); corrupt.write(&last, 1);
        }
        const auto recovered = splat::Trainer(options).train(scene);
        compare(uninterrupted, recovered);
        auto changed = options; changed.means_lr *= 2;
        bool rejected = false;
        try { splat::Trainer(changed).train(scene); } catch (const std::exception&) { rejected = true; }
        require(rejected, "Changed training settings accepted an old checkpoint");
        // A copied native project is sufficient for recovery; no sidecar files.
        const auto bytes = splat::training_checkpoint_bytes(options.checkpoint_path);
        auto archive = project::Archive::open(options.checkpoint_path);
        require(archive.chunk(marker_type) == std::vector<std::uint8_t>{9, 8, 7}, "Automatic save discarded existing project data");
        require(project::read_settings(archive).iterations == 10, "Automatic save discarded project settings");
        const auto copied_project = root / "copied.ascan";
        archive.save(copied_project);
        const auto original_project = options.checkpoint_path;
        options.checkpoint_path = copied_project;
        compare(uninterrupted, splat::Trainer(options).train(scene));
        options.checkpoint_path = original_project;
        splat::remove_training_checkpoint(options.checkpoint_path);
        require(std::filesystem::exists(options.checkpoint_path) && !splat::inspect_training_checkpoint(options.checkpoint_path),
            "Clearing a checkpoint removed the project itself");
        require(project::Archive::open(options.checkpoint_path).chunk(marker_type) == std::vector<std::uint8_t>{9, 8, 7},
            "Clearing training state discarded other project chunks");
        splat::restore_training_checkpoint(bytes, options.checkpoint_path);
        compare(uninterrupted, splat::Trainer(options).train(scene));
        {
            const auto legacy_project = root / "legacy-project.ascan";
            auto base = project::Archive::create();
            project::write_settings(base, project_settings, legacy_project);
            base.set_chunk(marker_type, {9, 8, 7}); base.save(legacy_project);
            auto legacy = legacy_project; legacy += ".training.ascan";
            auto old = project::Archive::from_bytes(bytes); old.save(legacy);
            const auto hidden = root / ".photara-recovery" / legacy_project.filename() / "training.ascan";
            old.save(hidden);
            {
                splat::detail::TrainingCheckpointLease active_legacy(legacy);
                bool blocked = false;
                try { splat::migrate_training_checkpoints(legacy_project); } catch (...) { blocked = true; }
                require(blocked && std::filesystem::exists(legacy), "Migration moved an active older trainer's checkpoint");
            }
            splat::migrate_training_checkpoints(legacy_project);
            require(!std::filesystem::exists(legacy) && !std::filesystem::exists(hidden),
                "Migration left external checkpoint files");
            require(splat::training_checkpoint_bytes(legacy_project) == bytes, "Migration changed the saved training state");
            require(project::Archive::open(legacy_project).chunk(marker_type) == std::vector<std::uint8_t>{9, 8, 7},
                "Migration discarded existing project chunks");
            options.checkpoint_path = legacy_project;
            compare(uninterrupted, splat::Trainer(options).train(scene));
            options.checkpoint_path = original_project;
            splat::migrate_training_checkpoints(legacy_project); // Idempotent reopen.
            // Different old runs are preserved as project history, never separate files.
            tinytensor::Tensor tensor = tinytensor::Tensor::ones({2}, tinytensor::Device::CPU);
            splat::detail::save_training_checkpoint(legacy, {1, 2, 2, 42}, {}, {&tensor});
            splat::migrate_training_checkpoints(legacy_project);
            require(!std::filesystem::exists(legacy) &&
                project::Archive::open(legacy_project).has(project::ChunkType::training_checkpoint_history),
                "Conflicting legacy state was not retained inside the project");
        }
        // Saving to an unwritable path preserves the previous valid file.
        {
            const auto blocked = root / "blocked";
            std::ofstream output(blocked); output << "not a directory"; output.close();
            bool failed = false;
            tinytensor::Tensor tensor = tinytensor::Tensor::ones({2}, tinytensor::Device::CPU);
            try { splat::detail::save_training_checkpoint(blocked / "checkpoint",
                {1, 2, 2, 42}, {}, {&tensor}); } catch (...) { failed = true; }
            require(failed, "Checkpoint write failure was silently accepted");
        }
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
