#include "pipeline.hpp"
#include "sparse_view.hpp"
#include "sfm/asfm.hpp"
#include <filesystem>
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
        require(settings.reuse_cache, "new Studio projects reuse compatible stages");
        auto command = editor::build_align_command("photara.exe", settings, layout);
        require(command.find("--ba-backend cpu") != std::string::npos &&
                command.find("--positioning-backend cpu") != std::string::npos,
                "Studio CPU selection controls both positioning and BA");
        require(command.find("--cache-dir \"") != std::string::npos,
                "first alignment receives a cache path even before it exists");
        settings.reuse_cache = false;
        command = editor::build_align_command("photara.exe", settings, layout);
        require(command.find("--cache-dir -") != std::string::npos,
                "fresh alignment explicitly disables stage caching");
        command = editor::build_train_command("photara.exe", settings, layout, {});
        require(command.find("--cache-dir -") != std::string::npos &&
                command.find("--positioning-backend cpu") != std::string::npos,
                "training-triggered alignment uses the same cache/backend policy");

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
        const auto decoded=photara::sfm::decode_asfm(photara::sfm::encode_asfm(source));
        const auto imported=editor::sparse_scene_from_sfm(decoded,false);
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
