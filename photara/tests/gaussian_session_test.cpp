#include "app.hpp"
#include "ui.hpp"
#include "imgui_internal.h"
#include "splat/trainer.hpp"
#include "splat/formats.hpp"
#include "splat/model_handoff.hpp"
#include "io/video_frames.hpp"
#include "project/document.hpp"
#include "studio_lasso_test.hpp"
#include "studio_view_shortcuts_test.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <fstream>
#include <thread>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace editor {
bool save_project_to_path(App&, const std::filesystem::path&);
void apply_opened_project(App&);
void request_gaussian_scene_load(App&);
}

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void wait_load(editor::App& app) {
    editor::process_gaussian_composition_refresh(app);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (app.loading_scene) {
        editor::poll_scene_load(app);
        require(std::chrono::steady_clock::now() < deadline, "composition load timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
void wait_gpu_load(editor::App& app) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (!editor::ensure_splat_renderer(app)) {
        editor::poll_splat_model_load(app);
        require(std::chrono::steady_clock::now() < deadline, "GPU model preparation timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(app.splat_renderer->set_model_transforms(app.gaussian_placements), "GPU placement table rejected");
}
photara::splat::GaussianModel model(float x) {
    using tinytensor::Tensor;
    constexpr auto cpu = tinytensor::Device::CPU;
    photara::splat::GaussianModel result;
    result.sh_degree = 0;
    result.means = Tensor::from_vector({x, 0.F, 0.F}, {1U, 3U}, cpu);
    result.log_scales = Tensor::from_vector({-1.F, -1.F, -1.F}, {1U, 3U}, cpu);
    result.quaternions = Tensor::from_vector({1.F, 0.F, 0.F, 0.F}, {1U, 4U}, cpu);
    result.opacity_logits = Tensor::from_vector({1.F}, {1U, 1U}, cpu);
    result.sh = Tensor::from_vector({0.F, 0.F, 0.F}, {1U, 1U, 3U}, cpu);
    return result;
}
void test_delete_open_row(editor::App& app) {
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = {500, 500}; io.DeltaTime = 1.F / 60.F;
    io.Fonts->AddFontDefault(); io.Fonts->Build();
    float x = 0, top = 0;
    const auto frame = [&] {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos({0, 0}); ImGui::SetNextWindowSize({400, 400});
        ImGui::Begin("Gaussian list test");
        ImGui::PushID(0);
        const std::string label = app.gaussian_models[0].name + "###gaussian_model";
        ImGui::GetStateStorage()->SetInt(ImGui::GetID(label.c_str()), 1);
        ImGui::PopID();
        x = ImGui::GetCurrentWindow()->WorkRect.Max.x - 10.F;
        top = ImGui::GetCursorScreenPos().y;
        const int depth = ImGui::GetCurrentWindow()->DC.TreeDepth;
        editor::draw_gaussian_models_panel(app);
        require(ImGui::GetCurrentWindow()->DC.TreeDepth == depth, "model row leaked tree stack");
        ImGui::End(); ImGui::Render();
    };
    frame();
    const auto original = app.gaussian_models.size();
    // Click the right-hand remove column through the first row. Actual ImGui
    // input drives the production panel, with the model tree explicitly open.
    for (float offset = 15; offset < 90 && app.gaussian_models.size() == original; offset += 3) {
        io.AddMousePosEvent(x, top + offset);
        io.AddMouseButtonEvent(0, true); frame();
        io.AddMouseButtonEvent(0, false); frame();
    }
    require(app.gaussian_models.size() == original - 1, "open model row was not removed");
    ImGui::DestroyContext();
    wait_load(app);
}
int test_gpu_session(const std::filesystem::path& folder,
                     const std::vector<editor::GaussianModelEntry>& entries) {
    if (!glfwInit()) return 77;
    if (!glfwVulkanSupported()) { glfwTerminate(); return 77; }
    std::uint32_t count = 0;
    const auto required = glfwGetRequiredInstanceExtensions(&count);
    ImVector<const char*> extensions;
    for (std::uint32_t i = 0; i < count; ++i) extensions.push_back(required[i]);
    editor::gpu::create_context(extensions);
    {
        editor::App gpu;
        gpu.gaussian_models = entries;
        for (auto& entry : gpu.gaussian_models) entry.visible = true;
        editor::store_path_field(gpu.settings.project_dir, folder / "models.ascan");
        editor::refresh_artifacts(gpu);
        editor::request_gaussian_scene_load(gpu); wait_load(gpu);
        wait_gpu_load(gpu);
        const auto epoch = gpu.splat_renderer->frames_epoch();
        const auto key = gpu.splat_renderer->source_key();
        const auto source = gpu.gaussian_models[0].cached_model;
        const auto upload = gpu.splat_renderer->model_upload_revision();
        for (int i = 0; i < 20; ++i) {
            gpu.gaussian_models[0].offset[0] += 0.05F;
            editor::refresh_gaussian_transform(gpu);
            require(gpu.splat_renderer->set_model_transforms(gpu.gaussian_placements), "GPU drag placement");
            require(editor::ensure_splat_renderer(gpu), "live drag GPU preview");
            require(gpu.splat_renderer->frames_epoch() == epoch && gpu.splat_renderer->source_key() == key &&
                    gpu.splat_renderer->model_upload_revision() == upload,
                    "live drag re-uploaded the composition");
        }
        require(gpu.gaussian_models[0].cached_model == source && !gpu.loading_scene,
                "GPU placement invalidated source cache or rebuilt scene");
        editor::prepare_gaussian_editing(gpu);
        photara::splat::GaussianModel editable;
        require(gpu.splat_edit.copy_model(editable), "world-space editing model missing");
        const auto expected = editor::load_export_model(gpu, tinytensor::Device::CPU);
        require(editable.means.to_vector() == expected.means.to_vector(), "picking/editing placement differs from export");
        require(gpu.splat_renderer->model_upload_revision() == upload, "opening editing reuploaded the GPU cloud");
        const auto ffmpeg = photara::io::locate_ffmpeg();
        if (!ffmpeg.empty()) {
            photara::project::CameraKeyframe first{}, second{};
            first.time = 0.F; first.distance = 4.F; first.fov_degrees = 50.F;
            first.target[0] = 10.F;
            second = first; second.time = .2F; second.target[0] = 10.8F;
            gpu.camera_keyframes = {first, second};
            gpu.video_render.width = 128; gpu.video_render.height = 96; gpu.video_render.fps = 10;
            const auto video = folder / "camera path test.mp4";
            require(editor::start_video_render(gpu, video, ffmpeg), "video export did not start");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(25);
            while (gpu.video_render.job.wait_for(std::chrono::milliseconds(20)) != std::future_status::ready)
                require(std::chrono::steady_clock::now() < deadline, "video export timed out");
            const auto failure = gpu.video_render.job.get();
            if (!failure.empty()) throw std::runtime_error("video export: " + failure);
            require(gpu.video_render.progress->completed == 3, "video did not render all sampled frames");
            photara::io::VideoProbe probe;
            require(photara::io::probe_video(ffmpeg, video, probe) && probe.width == 128 && probe.height == 96 &&
                        std::abs(probe.fps - 10.) < .01, "encoded video dimensions or frame rate mismatch");
            photara::io::VideoPreviewOptions preview;
            preview.video = video; preview.ffmpeg = ffmpeg; preview.output_dir = folder / "decoded";
            preview.fps = 10.F; preview.max_width = 128; preview.max_height = 96;
            require(photara::io::generate_video_preview(preview) >= 3, "cannot decode rendered video");
            const auto first_image = photara::io::load_rgb(preview.output_dir / "p_000001.jpg");
            const auto last_image = photara::io::load_rgb(preview.output_dir / "p_000003.jpg");
            require(!first_image.pixels.empty() && first_image.pixels != last_image.pixels,
                    "rendered video did not follow the camera path");
            require(*std::max_element(first_image.pixels.begin(), first_image.pixels.end()) > 30,
                    "rendered video frame is empty");
            gpu.video_render.progress.reset();
            std::cout << "Video render, decode, camera movement and cancellation passed\n";
            gpu.video_render.codec = 1;
            const auto hevc_video = folder / "camera path hevc.mp4";
            require(editor::start_video_render(gpu, hevc_video, ffmpeg), "H.265 export did not start");
            const auto hevc_error = gpu.video_render.job.get();
            if (hevc_error.find("Unknown encoder") == std::string::npos &&
                hevc_error.find("not found") == std::string::npos) {
                if (!hevc_error.empty()) throw std::runtime_error("H.265 video export: " + hevc_error);
                std::ifstream hevc_stream(hevc_video, std::ios::binary);
                const std::string hevc_bytes((std::istreambuf_iterator<char>(hevc_stream)), {});
                require(hevc_bytes.find("hvc1") != std::string::npos, "H.265 selection did not produce HEVC video");
            }
            gpu.video_render.progress.reset();
            gpu.video_render.codec = 0;
            const auto retained = folder / "existing.mp4";
            { std::ofstream stream(retained, std::ios::binary); stream << "existing content"; }
            require(editor::start_video_render(gpu, retained, folder / "missing ffmpeg.exe"), "encoder failure test did not start");
            require(!gpu.video_render.job.get().empty(), "missing encoder incorrectly succeeded");
            std::ifstream existing(retained, std::ios::binary);
            const std::string retained_data((std::istreambuf_iterator<char>(existing)), {});
            require(retained_data == "existing content", "failed encoder overwrote the previous video");
            gpu.video_render.progress.reset();
            const auto cancelled = folder / "cancelled.mp4";
            require(editor::start_video_render(gpu, cancelled, ffmpeg), "cancelled video export did not start");
            gpu.video_render.progress->cancel = true;
            require(gpu.video_render.job.get() == "Cancelled" && !std::filesystem::exists(cancelled),
                    "cancelled export created an output");
            gpu.video_render.progress.reset();
        }
        gpu.splat_edit.replace_from_model(gpu, model(17.F), false);
        require(gpu.splat_edit.dirty(), "edit fixture did not bind to composed cloud");
        const auto project = folder / "edited.ascan";
        require(editor::save_project_to_path(gpu, project), "edited composition save");
        editor::App restored;
        editor::store_path_field(restored.settings.project_dir, project);
        editor::apply_opened_project(restored); wait_load(restored);
        require(restored.gaussian_models.size() == 1, "saved edit model list did not flatten");
        require(bool(restored.gaussian_models[0].embedded), "saved edits missing embedded model");
        require(bool(restored.composed_model), restored.message.c_str());
        require(restored.composed_model->means.to_vector()[0] == 17.F, "saved edited positions changed");
        restored.layout.working_splat = folder / "stale_working.ply";
        photara::splat::save_gaussians(model(99.F), restored.layout.working_splat);
        const auto exported = folder / "edited_export.ply";
        editor::store_path_field(restored.splat_export.path, exported);
        editor::export_trained_model(restored);
        require(photara::splat::load_gaussians(exported, tinytensor::Device::CPU).means.to_vector()[0] == 17.F,
                "embedded model export copied a stale working file");
        wait_gpu_load(restored);
        restored.splat_edit.replace_from_model(restored, model(23.F), false);
        require(restored.splat_edit.dirty(), "close fixture has no edits");
        editor::export_trained_model(restored);
        require(restored.splat_edit.dirty(), "export should preserve the editing session");
        test_delete_open_row(restored);
        require(restored.gaussian_models.empty() && !restored.composed_model &&
                !restored.final_splat_ready && !restored.splat_edit.dirty() &&
                restored.settings.splat_model_source[0] == '\0',
                "closing edited model left preview or edit state behind");
        require(std::filesystem::exists(exported), "closing model deleted exported file");
    }
    editor::gpu::destroy_context();
    glfwTerminate();
    return 0;
}

// Drive the production Inspector with ImGui input, without starting jobs or a
// graphics device. In particular, an existing mesh must not redirect the Train
// step's primary action to texture baking, and scrolling must not hide actions.
void test_inspector_workflow() {
    using namespace editor;
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = {1000.F, 800.F};
    io.DeltaTime = 1.F / 60.F;
    io.Fonts->AddFontDefault(); io.Fonts->Build();
    theme::apply_style();
    App app;
    app.model_workspace = 0;
    store_path_field(app.settings.images_dir, "images");
    app.has_sparse = true;
    app.has_mesh = true;
    app.settings.build_mesh = true;
    Action action = Action::none;
    ImGuiWindow* footer = nullptr;
    const auto frame = [&] {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos({10.F, 10.F});
        ImGui::SetNextWindowSize({340.F, 660.F});
        action = draw_inspector(app);
        footer = nullptr;
        for (auto* window : ImGui::GetCurrentContext()->Windows)
            if (window->Active && std::strstr(window->Name, "##inspector_actions"))
                footer = window;
        ImGui::Render();
        require(footer != nullptr, "Inspector footer missing");
    };
    const auto click_primary = [&] {
        frame(); frame();
        const float x = footer->WorkRect.Min.x + 40.F;
        const float y = footer->DC.CursorPosPrevLine.y + 20.F;
        require(y < footer->InnerClipRect.Max.y, "primary action is clipped");
        io.AddMousePosEvent(x, y); frame();
        io.AddMouseButtonEvent(0, true); frame();
        io.AddMouseButtonEvent(0, false); frame();
        return action;
    };
    require(current_inspector_section(app) == InspectorSection::texture,
        "automatic step should follow available artifacts");
    select_inspector_section(app, InspectorSection::training);
    require(click_primary() == Action::train, "Train step dispatched another stage");
    // Force a long parameter page to its end; the separate footer stays put.
    for (auto* window : ImGui::GetCurrentContext()->Windows)
        if (std::strstr(window->Name, "##inspector_parameters_"))
            window->Scroll.y = window->ScrollMax.y;
    require(click_primary() == Action::train, "scrolling hid the Train action");
    select_inspector_section(app, InspectorSection::mesh);
    require(click_primary() == Action::dense, "Mesh step dispatched another stage");
    app.has_sparse = false;
    require(click_primary() == Action::none, "Mesh action accepted missing alignment");
    select_inspector_section(app, InspectorSection::alignment);
    require(click_primary() == Action::align, "Alignment step dispatched another stage");
    select_inspector_section(app, InspectorSection::texture);
#if defined(PHOTARA_HAS_TEXTURE)
    require(click_primary() == Action::texture, "Texture step dispatched another stage");
#endif
    app.has_mesh = false;
    require(click_primary() == Action::none, "Texture action accepted a missing mesh");
    // Every view-settings branch must render with balanced ImGui stacks.
    for (const auto workspace : {ViewportWorkspace::scene_3d, ViewportWorkspace::image_2d}) {
        app.workspace = workspace;
        for (const auto mode : {VisualizationMode::points, VisualizationMode::mesh, VisualizationMode::splat}) {
            app.view_mode = mode;
            ImGui::NewFrame();
            ImGui::SetNextWindowSize({340.F, 660.F});
            ImGui::Begin("View settings test");
            draw_view_settings(app);
            ImGui::End(); ImGui::Render();
        }
    }
    ImGui::DestroyContext();
}

void test_video_progress_layout() {
    ImGui::CreateContext();
    auto& io = ImGui::GetIO();
    io.IniFilename = nullptr; io.DisplaySize = {1000.F, 800.F}; io.DeltaTime = 1.F / 60.F;
    io.Fonts->AddFontDefault(); io.Fonts->Build();
    editor::App app;
    std::promise<std::string> result;
    app.video_render.job = result.get_future();
    app.video_render.progress = std::make_shared<editor::App::VideoRenderProgress>();
    app.video_render.progress->total = 181;
    const auto frame = [&] {
        ImGui::NewFrame();
        editor::draw_video_render_settings(app);
        ImGui::Render();
        auto* popup = ImGui::FindWindowByName("##render_video_progress");
        require(popup && popup->Active, "video progress popup missing");
        require(std::abs(popup->Size.x - 420.F) < .1F && std::abs(popup->Size.y - 210.F) < .1F,
                "video progress popup changed size between stages");
    };
    frame();
    app.video_render.progress->completed = 7; frame();
    app.video_render.progress->completed = 181; frame();
    app.video_render.progress->encoding = true; frame();
    app.video_render.progress->cancel = true; frame();
    result.set_value({}); app.video_render.job.get();
    ImGui::DestroyContext();
}

int main(int argc, char** argv) {
    {
        editor::App viewer;
        require(editor::model_viewer_active(viewer), "empty launch should open model viewer");
        viewer.settings.images_dir[0] = 'x';
        require(!editor::model_viewer_active(viewer), "capture should open reconstruction workspace");
        viewer.model_workspace = 1;
        require(editor::model_viewer_active(viewer), "explicit viewer selection ignored");
        viewer.model_workspace = 0;
        require(!editor::model_viewer_active(viewer), "explicit reconstruction selection ignored");
    }
    if (argc > 1 && std::string(argv[1]) == "--view-shortcuts-only") {
        try {
            test_studio_view_shortcuts();
            std::cout << "View shortcut checks passed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    if (argc > 1 && std::string(argv[1]) == "--lasso-only") {
        try {
            test_studio_lasso();
            std::cout << "Lasso gesture checks passed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    if (argc > 1 && std::string(argv[1]) == "--inspector-only") {
        try {
            test_inspector_workflow();
            test_video_progress_layout();
            std::cout << "Inspector workflow checks passed\n";
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    const auto folder = std::filesystem::temp_directory_path() / ("photara_session_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(folder);
    try {
        {
            auto tagged = model(1.F);
            tagged.filter_mode = photara::splat::SplatFilterMode::mip_2d;
            const auto ready = folder / "filter_handoff.ready";
            photara::splat::ModelHandoff publisher(tagged, ready);
            const auto received = photara::splat::receive_model_handoff(ready);
            require(received.filter_mode == tagged.filter_mode, "model handoff lost filter mode");
        }
        const auto first = folder / "first.ply", second = folder / "second.ply";
        photara::splat::save_gaussians(model(1.F), first);
        photara::splat::save_gaussians(model(3.F), second);
        editor::App app;
        editor::import_gaussian_models(app, {first, second}, false); wait_load(app);
        require(app.composed_model && app.composed_model->size() == 2, "initial composition");
        const auto source = app.gaussian_models[0].cached_model;
        const auto preview = app.composed_model;
        app.gaussian_models[0].offset[0] = 7.F; app.gaussian_models[0].scale = 2.F;
        require(app.composed_model == preview, "live drag changed committed composition");
        editor::refresh_gaussian_transform(app);
        require(app.gaussian_models[0].cached_model == source, "placement reloaded source data");
        require(app.composed_model == preview && app.composed_model->means.to_vector()[0] == 1.F,
                "GPU placement modified uploaded source attributes");
        require(editor::load_export_model(app, tinytensor::Device::CPU).means.to_vector()[0] == 9.F &&
                    app.scene.world_point(0).x == 9.F, "preview/export placement not applied");
        app.gaussian_models[1].visible = false;
        editor::refresh_gaussian_composition(app); wait_load(app);
        const auto project = folder / "models.ascan";
        require(editor::save_project_to_path(app, project), "project save failed");
        std::filesystem::remove(first); std::filesystem::remove(second);
        editor::App restored;
        editor::store_path_field(restored.settings.project_dir, project);
        editor::apply_opened_project(restored); wait_load(restored);
        require(restored.gaussian_models.size() == 2 && !restored.gaussian_models[1].visible,
                "model list/visibility not restored");
        require(restored.gaussian_models[0].scale == 2.F && restored.gaussian_models[0].offset[0] == 7.F,
                "placement not restored");
        require(restored.composed_model && editor::load_export_model(restored, tinytensor::Device::CPU).means.to_vector()[0] == 9.F,
                "embedded source restore depended on deleted files");
        restored.gaussian_models[1].visible = true;
        editor::refresh_gaussian_composition(restored); wait_load(restored);
        require(restored.composed_model->size() == 2, "hidden embedded source could not be restored");
        if (argc > 1 && std::string(argv[1]) == "--vulkan") {
            const int result = test_gpu_session(folder, restored.gaussian_models);
            if (result != 0) { std::filesystem::remove_all(folder); return result; }
        }
        // Exercise the same edited multi-source state that used to disable X.
        const float means[]{0.F, 0.F, 0.F};
        const float scales[]{-1.F, -1.F, -1.F};
        const float quats[]{1.F, 0.F, 0.F, 0.F};
        const float opacity[]{1.F};
        const float sh[]{0.F, 0.F, 0.F};
        editor::SplatEdit::Host host;
        host.means = means; host.log_scales = scales;
        host.quaternions = quats; host.opacity_logits = opacity;
        host.sh = sh; host.count = 1;
        host.filter_mode = static_cast<int>(photara::splat::SplatFilterMode::mip_2d);
        restored.splat_edit.sync("edited-close-test", host);
        photara::splat::GaussianModel edited_model;
        require(restored.splat_edit.copy_model(edited_model) &&
                    edited_model.filter_mode == photara::splat::SplatFilterMode::mip_2d,
                "Gaussian editor lost filter mode");
        restored.splat_edit.replace_from_model(restored, model(23.F), false);
        require(restored.splat_edit.dirty(), "multi-source close fixture has no edits");
        const auto extra = folder / "extra.ply";
        photara::splat::save_gaussians(model(31.F), extra);
        editor::import_gaussian_models(restored, {extra}, false); wait_load(restored);
        require(restored.gaussian_models.size() == 2 && restored.composed_model &&
                restored.composed_model->size() == 2,
                "edited composition did not accept another model");
        require(restored.composed_model->means.to_vector()[0] == 23.F &&
                restored.composed_model->means.to_vector()[3] == 31.F,
                "import discarded edited positions or incoming model");
        require(bool(restored.gaussian_models[0].embedded), "edits were not preserved as a source");
        test_delete_open_row(restored);
        require(!restored.splat_edit.dirty(), "removal left stale composition edits");
        test_delete_open_row(restored);
        require(restored.gaussian_models.empty() && !restored.composed_model &&
                restored.view_mode == editor::VisualizationMode::points,
                "last model removal did not clear the viewport");
        editor::App hidden;
        editor::store_path_field(hidden.settings.project_dir, project);
        editor::apply_opened_project(hidden); wait_load(hidden);
        for (auto& entry : hidden.gaussian_models) entry.visible = false;
        editor::refresh_gaussian_composition(hidden); wait_load(hidden);
        require(!hidden.has_model && !hidden.composed_model, "all-hidden composition remained visible");
        const auto hidden_project = folder / "hidden.ascan";
        require(editor::save_project_to_path(hidden, hidden_project), "all-hidden save failed");
        editor::App reopened;
        editor::store_path_field(reopened.settings.project_dir, hidden_project);
        editor::apply_opened_project(reopened); wait_load(reopened);
        require(reopened.gaussian_models.size() == 2 && !reopened.has_model, "all-hidden restore");
        for (auto& entry : reopened.gaussian_models) entry.visible = true;
        editor::refresh_gaussian_composition(reopened); wait_load(reopened);
        require(reopened.has_model && reopened.composed_model->size() == 2, "show restored hidden models");
        reopened.gaussian_models[1].scale = .75F;
        reopened.gaussian_models[1].rotation_degrees[2] = 90.F;
        reopened.gaussian_models[1].offset[1] = -1.F;
        reopened.gaussian_models[1].offset[2] = 2.F;
        editor::refresh_gaussian_transform(reopened);
        const auto before_merge = editor::load_export_model(reopened, tinytensor::Device::CPU).means.to_vector();
        const auto merged = folder / "merged.ply";
        editor::start_gaussian_merge(reopened, merged);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (reopened.gaussian_merge.running || reopened.gaussian_merge.apply_pending) {
            editor::poll_gaussian_merge(reopened);
            require(std::chrono::steady_clock::now() < deadline, "embedded merge timed out");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        wait_load(reopened);
        require(reopened.gaussian_models.size() == 1 && reopened.composed_model &&
                reopened.composed_model->size() == 2 && std::filesystem::exists(merged),
                "merge did not use embedded sources");
        const auto after_merge = reopened.composed_model->means.to_vector();
        require(after_merge.size() == before_merge.size(), "merge changed Gaussian count");
        for (std::size_t i = 0; i < before_merge.size(); ++i)
            require(std::abs(after_merge[i] - before_merge[i]) < 1e-5F, "merge lost or reapplied model placement");
        require(reopened.gaussian_models[0].scale == 1.F &&
                    reopened.gaussian_models[0].offset[0] == 0.F &&
                    reopened.gaussian_models[0].rotation_degrees[2] == 0.F,
                "merged model retained baked placement parameters");
        std::filesystem::remove_all(folder);
        std::cout << "Gaussian session checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(folder);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
