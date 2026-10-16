#include "pipeline.hpp"
#include "sparse_view.hpp"
#include "sfm/asfm.hpp"
#include "project/document.hpp"
#include "splat/checkpoint.hpp"
#include "splat/trainer.hpp"
#include "../src/splat/checkpoint.hpp"
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}
int main() {
    try {
        editor::RunMonitor monitor;
        monitor.begin(editor::JobKind::train);
        require(monitor.training_elapsed_seconds() < 0, "initialization does not start the training clock");
        require(monitor.headline().find("Initializing") != std::string::npos, "training initialization has an explicit status");
        monitor.consume("splat iteration=1/10 loss=0.1");
        require(monitor.training_elapsed_seconds() >= 0, "first iteration starts the training clock");
        monitor.pause_clock();
        const auto paused_elapsed = monitor.training_elapsed_seconds();
        require(monitor.training_elapsed_seconds() == paused_elapsed, "paused training clock is frozen");
        monitor.resume_clock();
        monitor.consume("splat_checkpoint_saving iteration=5");
        require(monitor.saving_checkpoint(), "checkpoint save has an explicit status");
        monitor.consume("splat_checkpoint_saved iteration=5");
        require(!monitor.saving_checkpoint(), "checkpoint completion restores the training status");
        monitor.consume("splat_training_completed iterations=10");
        const auto completed_elapsed = monitor.training_elapsed_seconds();
        require(monitor.training_elapsed_seconds() == completed_elapsed, "export does not extend training time");
        monitor.mark_finished(0);
        monitor.begin(editor::JobKind::train);
        require(monitor.training_elapsed_seconds() < 0, "new training resets its clock");
        editor::ProjectSettings settings;
        editor::ProjectLayout layout;
        layout.cache = std::filesystem::temp_directory_path() / "photara alignment test cache";
        layout.sparse_ply = layout.cache / "sparse.ply";
        layout.model_output = layout.cache / "splat.ply";
        settings.alignment_backend = 1;
        require(!settings.reuse_cache, "new Studio projects recompute alignment by default");
        auto command = editor::build_align_command("photara.exe", settings, layout);
        require(command.find("--ba-backend cpu") != std::string::npos &&
                command.find("--positioning-backend cpu") != std::string::npos,
                "Studio CPU selection controls both positioning and BA");
        require(command.find("--cache-dir -") != std::string::npos,
                "default alignment explicitly disables stage caching");
        require(command.find("--camera-warm-start") == std::string::npos,
                "Studio leaves warm start policy internal instead of exposing tuning");
        settings.reuse_cache = true;
        command = editor::build_align_command("photara.exe", settings, layout);
        require(command.find("--cache-dir \"") != std::string::npos,
                "explicit cache reuse receives a cache path even before it exists");
        settings.reuse_cache = false;
        command = editor::build_align_command("photara.exe", settings, layout);
        require(command.find("--cache-dir -") != std::string::npos,
                "fresh alignment explicitly disables stage caching");
        command = editor::build_train_command("photara.exe", settings, layout, {});
        require(command.find("--cache-dir -") != std::string::npos &&
                command.find("--positioning-backend cpu") != std::string::npos,
                "training-triggered alignment uses the same cache/backend policy");
        settings.speedy_pruning = true;
        command = editor::build_train_command("photara.exe", settings, layout, {});
        require(command.find("--splat-speedy-pruning=true") != std::string::npos &&
                command.find("--splat-speedy-start") == std::string::npos &&
                command.find("--splat-speedy-every") == std::string::npos,
                "Studio uses the automatic conservative pruning schedule");

        photara::sfm::Scene source;
        photara::sfm::PinholeCamera camera;
        camera.id=0;camera.width=640;camera.height=480;
        camera.fx=camera.fy=500;camera.cx=320;camera.cy=240;
        source.cameras.push_back(camera);
        source.images.resize(3);
        for (photara::sfm::Index i=0;i<3;++i) {
            auto& image=source.images[i];image.id=i;image.camera_id=0;
            image.path=std::to_string(i)+".jpg";image.registered=i<2;
            image.position_audit=i==0?photara::sfm::PositionAudit::graph_supported:
                                     photara::sfm::PositionAudit::unverified;
        }
        source.images[0].path = std::filesystem::path(u8"标准数据集-办公室") / u8"图像01.jpg";
        const auto decoded=photara::sfm::decode_asfm(photara::sfm::encode_asfm(source));
        const auto task_root = std::filesystem::temp_directory_path() /
            ("photara-transient-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        editor::ProjectSettings transient;
        const auto put_path = [](auto& field, const std::filesystem::path& path) {
            const auto text = editor::path_to_utf8(path);
            std::snprintf(field.data(), field.size(), "%s", text.c_str());
        };
        std::filesystem::create_directories(task_root / "aligned priors");
        auto prior_settings = settings;
        prior_settings.ai_priors = true; prior_settings.ai_prior_source = 1;
        prior_settings.geometry_training = false;
        prior_settings.ai_depth_weight = 0.F; prior_settings.ai_normal_weight = .02F;
        put_path(prior_settings.ai_prior_directory, task_root / "aligned priors");
        command = editor::build_train_command("photara.exe",prior_settings,layout,{});
        require(command.find("--splat-lambda-d 0") != std::string::npos &&
            command.find("--splat-lambda-n 0.02") != std::string::npos &&
            command.find("--splat-geometry-loss=false") != std::string::npos &&
            command.find("--splat-prior-model") == std::string::npos,
            "cached normal-only priors remain independent of DN/MV and inference");
        prior_settings.ai_depth_weight = prior_settings.ai_normal_weight = 0.F;
        prior_settings.ai_prior_directory.fill('\0');
        command = editor::build_train_command("photara.exe",prior_settings,layout,{});
        require(command.find("--splat-prior-") == std::string::npos,
            "zero weights skip prior paths and generation");
        prior_settings.ai_prior_source = 0; prior_settings.ai_depth_weight = .01F;
        const auto model_file = task_root / "moge normal.gguf";
        std::ofstream(model_file).put('x'); put_path(prior_settings.ai_prior_model,model_file);
        command = editor::build_train_command("photara.exe",prior_settings,layout,{});
        require(command.find("--splat-prior-model \"") != std::string::npos &&
            command.find("--splat-prior-tokens 1200") != std::string::npos &&
            command.find("--splat-undistort=true") != std::string::npos &&
            command.find("ai_priors") != std::string::npos,
            "automatic preparation receives the model, token budget and task cache");
        prior_settings.training_mode = 0; prior_settings.ai_prior_model.fill('\0');
        command = editor::build_train_command("photara.exe",prior_settings,layout,{});
        require(command.find("--splat-prior-") == std::string::npos,
            "fast mode skips AI priors without requiring model files");
        monitor.begin(editor::JobKind::train);
        monitor.consume("stage started: splat.prepare_ai_priors");
        monitor.consume("progress: Prepare AI priors completed=2/4 percent=50.0 items/s=1 elapsed_s=2 eta_s=2");
        require(monitor.headline().find("Prepare AI priors") != std::string::npos &&
            monitor.training_elapsed_seconds() < 0,
            "prior generation has progress before the training timer starts");
        put_path(transient.images_dir, task_root / "dataset_a" / "images");
        put_path(transient.cache_dir, task_root / "cache");
        const auto first_task = editor::resolve_layout(transient);
        require(first_task.project_file.empty(), "dataset selection does not assign a user project file");
        require(first_task.training_checkpoint.extension() == ".checkpoint" &&
                first_task.model_output.extension() == ".ply", "training outputs are task cache artifacts");
        auto train_command = editor::build_train_command("photara.exe", transient, first_task, {});
        require(train_command.find(".ascan") == std::string::npos,
                "unsaved training command does not create an ascan project");
        editor::prepare_training_inputs(first_task, task_root / "dataset_a" / "images", false);
        require(!std::filesystem::exists(first_task.training_checkpoint), "fresh preparation does not create a project or checkpoint");
        put_path(transient.images_dir, task_root / "dataset_b" / "images");
        const auto second_task = editor::resolve_layout(transient);
        require(second_task.working_sfm.parent_path() != first_task.working_sfm.parent_path(),
                "different datasets have independent runtime caches");
        editor::prepare_training_inputs(second_task, {}, false, [] { return true; });
        require(!std::filesystem::exists(second_task.working_sfm.parent_path()), "cancelled preparation creates no artifacts");

        const auto project = task_root / "saved.ascan";
        auto archive = photara::project::Archive::create();
        photara::project::write_sfm(archive, source, project);
        photara::splat::GaussianModel saved_model;
        saved_model.sh_degree = 0;
        const auto cpu = tinytensor::Device::CPU;
        saved_model.means = tinytensor::Tensor::zeros({2, 3}, cpu);
        saved_model.log_scales = tinytensor::Tensor::zeros({2, 3}, cpu);
        saved_model.quaternions = tinytensor::Tensor::from_vector({1.F, 0.F, 0.F, 0.F, 1.F, 0.F, 0.F, 0.F}, {2, 4}, cpu);
        saved_model.opacity_logits = tinytensor::Tensor::zeros({2, 1}, cpu);
        saved_model.sh = tinytensor::Tensor::zeros({2, 1, 3}, cpu);
        archive.set_chunk(photara::project::ChunkType::gaussians, photara::splat::encode_gaussians(saved_model));
        archive.set_chunk(photara::project::ChunkType::training_model_info,
            photara::splat::training_checkpoint_info_bytes({3, 10, 2, 42}));
        archive.save(project);
        const auto bytes_before = photara::project::Archive::open(project).encode();
        const auto stamp_before = std::filesystem::last_write_time(project);
        put_path(transient.project_dir, project);
        const auto saved_task = editor::resolve_layout(transient);
        require(saved_task.training_checkpoint != project && saved_task.model_output != project,
                "opened projects use independent task output and checkpoints");
        require(editor::build_train_command("photara.exe", transient, saved_task, {}).find(".ascan") == std::string::npos,
                "saved project is never passed as the training output");
        editor::prepare_training_inputs(saved_task, task_root / "dataset_b" / "images", true);
        require(std::filesystem::exists(saved_task.working_sfm), "saved camera data is restored into the task cache");
        require(photara::splat::inspect_training_checkpoint(saved_task.training_checkpoint)->iteration == 3,
                "saved progress can resume from an independent task checkpoint");
        editor::prepare_training_inputs(saved_task, task_root / "dataset_b" / "images", false);
        require(!std::filesystem::exists(saved_task.training_checkpoint), "new training clears only the task checkpoint");
        require(photara::project::Archive::open(project).encode() == bytes_before &&
                std::filesystem::last_write_time(project) == stamp_before,
                "preparation and resume leave the saved project byte-for-byte unchanged");
        for (const auto& entry : std::filesystem::recursive_directory_iterator(task_root / "cache"))
            require(entry.path().extension() != ".ascan", "task cache contains no ascan files");
        std::filesystem::remove_all(task_root);
        const auto imported=editor::sparse_scene_from_sfm(decoded,false);
        require(imported.scene.views[0].name == editor::path_to_utf8(source.images[0].path.filename()),
                "Chinese camera image names reach the UI as UTF-8");
        const auto counts=imported.scene.position_audit_counts();
        require(counts.audited==2 && counts.unverified==1 && counts.unaudited==0,
                "native project loading preserves risk and excludes unregistered cameras");
        require(imported.scene.views[1].position_audit==photara::sfm::PositionAudit::unverified,
                "per-camera position risk reaches Studio");
        auto cleared=imported.scene;cleared.clear();
        require(cleared.position_audit_counts().unverified==0,"clearing a scene clears risk status");

        const auto csv=std::filesystem::temp_directory_path()/"photara-studio-position-audit-test.csv";
        const auto write_csv=[&](bool audit) {
            std::ofstream out(csv);
            for(int i=0;i<28;++i)out<<(i?",":"")<<"column"<<i;
            if(audit)out<<",position_audit";
            out<<'\n';
            for(int i=0;i<28;++i)out<<(i?",":"")<<(i==1?"camera.jpg":i==2?"1":"0");
            if(audit)out<<",unverified";
            out<<'\n';
        };
        write_csv(true);
        editor::SparseScene csv_scene;
        require(editor::load_view_poses(csv,csv_scene),"risk CSV loads");
        require(csv_scene.position_audit_counts().unverified==1,"CSV risk reaches Studio");
        write_csv(false);
        editor::SparseScene legacy;
        require(editor::load_view_poses(csv,legacy),"legacy CSV loads");
        require(legacy.position_audit_counts().unaudited==1 &&
                legacy.position_audit_counts().unverified==0,
                "legacy cameras remain unaudited instead of implicitly supported");
        std::filesystem::remove(csv);
        std::cout<<"Studio alignment policy and risk tests passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
