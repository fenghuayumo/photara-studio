#include "pipeline.hpp"
#include "sparse_view.hpp"
#include "sfm/asfm.hpp"
#include "project/document.hpp"
#include "splat/checkpoint.hpp"
#include "splat/trainer.hpp"
#include "studio_image_qa_test.hpp"
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
void test_camera_navigation() {
    auto* context = ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.DisplaySize = {800, 600}; io.DeltaTime = 1.F / 60.F;
    unsigned char* pixels; int w, h;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &w, &h);
    ImGui::NewFrame();
    const auto eye = [](const editor::OrbitCamera& c) { return editor::make_preview_camera(c, 640, 480).position; };
    for (const auto mode : {editor::CameraNavigation::turntable, editor::CameraNavigation::trackball,
                           editor::CameraNavigation::first_person, editor::CameraNavigation::fly}) {
        editor::OrbitCamera c; c.navigation = mode; c.interacting = true;
        const auto before = eye(c);
        io.MouseDown[1] = true; io.MouseDelta = {14.F, 8.F};
        editor::update_orbit_camera(c, true, 1.F, true);
        const auto after = eye(c);
        const float distance = std::abs(after[0]-before[0]) + std::abs(after[1]-before[1]) + std::abs(after[2]-before[2]);
        require(std::isfinite(distance), "camera control remains finite");
        if (mode == editor::CameraNavigation::first_person || mode == editor::CameraNavigation::fly)
            require(distance < 1e-4F, "look modes keep eye fixed");
        else require(distance > .01F, "orbit modes move around target");
    }
    ImGui::EndFrame();
    io.AddKeyEvent(ImGuiKey_W, true);
    ImGui::NewFrame();
    io.MouseDown[1] = true; io.MouseDelta = {0.F, 0.F};
    for (const auto mode : {editor::CameraNavigation::first_person, editor::CameraNavigation::fly}) {
        editor::OrbitCamera moving; moving.navigation = mode; moving.pitch = .8F;
        const auto before = eye(moving);
        editor::update_orbit_camera(moving, true, 1.F, true);
        const auto after = eye(moving);
        require(std::abs(after[0]-before[0]) + std::abs(after[2]-before[2]) > .001F, "W moves the camera");
        if (mode == editor::CameraNavigation::first_person)
            require(std::abs(after[1]-before[1]) < 1e-5F, "first person movement stays horizontal");
        else require(std::abs(after[1]-before[1]) > .001F, "flight follows pitched view");
    }
    editor::OrbitCamera trackball; trackball.navigation = editor::CameraNavigation::trackball; trackball.interacting = true;
    io.MouseDelta = {7.F, 25.F};
    for (int i = 0; i < 100; ++i) editor::update_orbit_camera(trackball, true, 1.F, true);
    require(std::isfinite(trackball.yaw) && std::isfinite(trackball.roll), "trackball remains stable through poles");
    editor::OrbitCamera c; c.roll = .7F;
    const auto preview = editor::make_preview_camera(c, 640, 480);
    std::array<float, 16> matrix;
    editor::camera_view_matrix(c, {0,0}, {640,480}, matrix);
    require(std::abs(matrix[0] - preview.world_to_camera[0]) < 1e-5F,
        "rolled viewport and render camera share their basis");
    ImGui::EndFrame(); ImGui::DestroyContext(context);
}
void test_camera_markers() {
    editor::ViewPose pose;
    pose.width=640; pose.height=480; pose.fx=pose.fy=300; pose.cx=320; pose.cy=240;
    const auto pinhole=editor::make_camera_marker_geometry(pose);
    require(pinhole.vertices.size()==4 && pinhole.triangles.size()==2, "pinhole marker retains its plane");
    require(std::abs(editor::camera_marker_hit_distance(pose,1,{0,0,-3},{0,0,1})-4)<1e-5F,
        "pinhole marker picking hits its image plane");
    pose.model=photara::CameraModel::opencv_fisheye;
    const auto fish=editor::make_camera_marker_geometry(pose,true);
    require(fish.vertices.size()>100 && !fish.triangles.empty() && !fish.rays.empty(), "fisheye marker has a curved sensor surface");
    for(const auto& v:fish.vertices) {
        const auto& p=v.position;
        require(std::abs(p.x*p.x+p.y*p.y+p.z*p.z-1)<1e-5F && p.z>0, "fisheye vertices stay on the calibrated forward unit sphere");
        const auto uv=photara::project_fisheye_camera(p.x,p.y,p.z,300,300,320,240,0,0,0,0);
        require(uv.valid && std::abs(uv.u-v.uv.x*640)<.01 && std::abs(uv.v-v.uv.y*480)<.01,
            "fisheye texture coordinates match calibration");
    }
    require(std::abs(editor::camera_marker_hit_distance(pose,1,{0,0,3},{0,0,-1})-2)<1e-5F,
        "fisheye picking hits the front spherical cap");
    require(!std::isfinite(editor::camera_marker_hit_distance(pose,1,{3,0,0},{-1,0,0})),
        "fisheye picking rejects rays outside the captured sensor");
    pose.k1=.1F;
    const auto distorted=editor::make_camera_marker_geometry(pose,true);
    require(std::abs(distorted.vertices.back().position.z-fish.vertices.back().position.z)>1e-4F,
        "fisheye distortion changes marker coverage");
    pose.fx=pose.fy=100; pose.k1=0;
    const auto circular=editor::make_camera_marker_geometry(pose,true);
    require(!circular.triangles.empty(), "invalid fisheye corners retain the visible circular sensor coverage");
    pose.model=photara::CameraModel::equirectangular;
    const auto pano=editor::make_camera_marker_geometry(pose,true);
    bool back=false,top=false,bottom=false;
    for(const auto& v:pano.vertices) {
        const auto p=v.position;
        require(std::abs(p.x*p.x+p.y*p.y+p.z*p.z-1)<1e-5F,"panorama marker is a complete unit sphere");
        back|=p.z<-.99F; top|=p.y<-.99F; bottom|=p.y>.99F;
    }
    require(back&&top&&bottom&&pano.rays.empty(),"panorama covers rear and poles without a fake forward frustum");
    for(const auto& face:pano.triangles) {
        float lo=1,hi=0;
        for(auto id:face) {lo=std::min(lo,pano.vertices[id].uv.x);hi=std::max(hi,pano.vertices[id].uv.x);}
        require(hi-lo<.1F,"panorama texture triangles do not cross the UV seam");
    }
    require(std::abs(editor::camera_marker_hit_distance(pose,1,{0,0,0},{0,0,-1})-1)<1e-5F,
        "panorama can be picked from inside its shell");
    // Exercise the actual ImGui draw path, including textured triangle indices,
    // without a native window or GPU context.
    ImGui::CreateContext();
    auto& io=ImGui::GetIO();
    io.IniFilename=nullptr; io.DisplaySize={640,480}; io.DeltaTime=1.F/60;
    unsigned char* pixels; int width,height;
    io.Fonts->GetTexDataAsRGBA32(&pixels,&width,&height);
    editor::SceneRenderer renderer;
    editor::SparseScene scene;
    pose.registered=true; scene.views={pose}; scene.radius=1;
    editor::OrbitCamera orbit; orbit.distance=3;
    editor::ViewOptions options;
    options.show_cloud=options.show_grid=options.show_axes=options.show_region=false;
    options.view_scale=.5F;
    const ImTextureID photo=static_cast<ImTextureID>(123);
    for(auto model:{photara::CameraModel::pinhole,photara::CameraModel::opencv_fisheye,photara::CameraModel::equirectangular}) {
        scene.views[0].model=model;
        for(auto projection:{editor::EditorProjection::perspective,editor::EditorProjection::panorama}) {
            orbit.projection=projection;
            ImGui::NewFrame();
            auto* draw=ImGui::GetBackgroundDrawList();
            const auto stats=renderer.draw(draw,{0,0},{640,480},scene,orbit,options,false,&photo,1,nullptr,nullptr,0);
            require(stats.drawn_views==1,"focused camera marker is rendered");
            bool textured=false;
            for(const auto& command:draw->CmdBuffer) textured|=command.TextureId==photo&&command.ElemCount>0;
            require(textured,"focused camera photo is rendered on its projection surface");
            for(const auto& vertex:draw->VtxBuffer)
                require(std::isfinite(vertex.pos.x)&&std::isfinite(vertex.pos.y)&&std::isfinite(vertex.uv.x)&&std::isfinite(vertex.uv.y),
                    "camera surface drawing emits finite positions and UVs");
            ImGui::Render();
        }
    }
    ImGui::DestroyContext();
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 3) {
            test_studio_image_qa_dataset(argv[1], argv[2]);
            return 0;
        }
        if (argc != 1) throw std::runtime_error("Usage: studio_alignment_test [native-sfm.bin image-directory]");
        test_studio_image_qa_projection();
        test_camera_markers();
        test_camera_navigation();
        editor::RunMonitor monitor;
        monitor.begin(editor::JobKind::align, editor::Stage::preparing);
        monitor.consume(R"(capture_frame kind=video ready=1 completed=20 total=20 image="D:\\capture frames\\00019.jpg")");
        require(monitor.video_frame().completed == 20 && monitor.video_frame().ready &&
                monitor.video_frame().image == R"(D:\capture frames\00019.jpg)",
                "video progress decodes quoted Windows paths");
        monitor.consume(R"(capture_frame kind=mask ready=1 completed=1 total=20 image="D:\\capture frames\\00000.jpg")");
        require(monitor.mask_frame().completed == 1 && monitor.mask_frame().total == 20 &&
                monitor.video_frame().completed == 20 && monitor.video_frame().total == 20,
                "mask progress preserves completed video progress");
        monitor.begin(editor::JobKind::align);
        require(monitor.video_frame().revision == 0 && monitor.mask_frame().revision == 0,
                "a new alignment clears progress from both previous stages");
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
        {
            auto backend_settings = settings;
            // Legacy projects may have saved a particular hardware variant.
            backend_settings.feature_extractor = "vulkan_sift";
            backend_settings.feature_matcher = "vulkan_mutual_ratio";
            const auto check_backend = [&](int backend, const char* extractor, const char* matcher) {
                backend_settings.alignment_backend = backend;
                for (const auto& cmd : {
                        editor::build_align_command("photara.exe", backend_settings, layout),
                        editor::build_train_command("photara.exe", backend_settings, layout, {})}) {
                    require(cmd.find(std::string(" --extractor ") + extractor + " ") != std::string::npos &&
                            cmd.find(std::string(" --matcher ") + matcher + " ") != std::string::npos,
                            "SIFT and brute force implementations follow the alignment backend");
                }
            };
            check_backend(1, "sift", "mutual_ratio");
            check_backend(0, "siftgpu", "gpu_mutual_ratio");
            if (editor::studio_ba_cuda()) check_backend(2, "siftgpu", "gpu_mutual_ratio");
            if (editor::studio_ba_vulkan()) check_backend(3, "vulkan_sift", "vulkan_mutual_ratio");
        }
        {
            auto matching_settings = settings;
            const auto check_pairing = [&](int strategy, bool retrieval,
                                            int window, bool expected_retrieval) {
                matching_settings.matching_strategy = strategy;
                matching_settings.retrieval_augmentation = retrieval;
                const auto align = editor::build_align_command(
                    "photara.exe", matching_settings, layout);
                const auto train = editor::build_train_command(
                    "photara.exe", matching_settings, layout, {});
                const auto window_flag = " --window " + std::to_string(window) + " ";
                const auto retrieval_flag = std::string(" --retrieval-augmentation=") +
                    (expected_retrieval ? "true" : "false");
                require(align.find(window_flag) != std::string::npos &&
                        align.find(retrieval_flag) != std::string::npos &&
                        train.find(window_flag) != std::string::npos &&
                        train.find(retrieval_flag) != std::string::npos,
                        "Studio align and train preserve the selected pairing policy");
            };
            matching_settings.sequential_overlap = 10;
            check_pairing(0, true, 0, true); // Unordered photos: automatic retrieval.
            check_pairing(0, false, 0, false); // Explicitly disabled: all pairs.
            check_pairing(1, false, 10, false);
            check_pairing(1, true, 10, true);
            check_pairing(2, true, 0, false); // Exhaustive overrides the checkbox.
            std::snprintf(matching_settings.images_dir.data(),
                          matching_settings.images_dir.size(), "capture.mp4");
            matching_settings.alignment_capture_type = 1;
            check_pairing(0, true, 10, true); // Video: temporal pairs plus retrieval.
            check_pairing(0, false, 10, false);
            check_pairing(2, true, 0, false);
            std::snprintf(matching_settings.images_dir.data(),
                          matching_settings.images_dir.size(), "extracted_frames");
            matching_settings.alignment_capture_type = 1;
            check_pairing(0, true, 10, true); // A frame folder can retain sequence ordering.
            matching_settings.alignment_capture_type = 0;
            check_pairing(0, true, 0, true); // Same folder can be treated as unordered photos.
            editor::apply_alignment_quality_preset(matching_settings, 0);
            require(matching_settings.retrieval_augmentation,
                    "Fast quality preserves automatic retrieval for unordered photos");
        }
        require(!settings.refine_weak_cameras && command.find("--refine-weak-cameras") == std::string::npos,
                "weak-camera refinement stays opt-in for new Studio projects");
        settings.refine_weak_cameras = true;
        command = editor::build_align_command("photara.exe", settings, layout);
        require(command.find("--refine-weak-cameras") != std::string::npos,
                "Studio forwards the optional camera refinement to the product CLI");
        command = editor::build_train_command("photara.exe", settings, layout, {});
        require(command.find("--refine-weak-cameras") != std::string::npos,
                "training-triggered alignment uses the same refinement policy");
        for (int mode = 0; mode != 3; ++mode) {
            settings.filter_mode = mode;
            const char* flags[] = {"classic", "mip2d", "mip3d"};
            const auto filtered = editor::build_train_command("photara.exe", settings, layout, {});
            require(filtered.find(std::string("--splat-filter-mode ") + flags[mode]) != std::string::npos,
                    "Studio did not forward the selected filter mode");
        }
        settings.filter_mode = 0;
        settings.refine_weak_cameras = false;
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
