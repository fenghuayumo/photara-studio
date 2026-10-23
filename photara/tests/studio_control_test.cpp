#include "app.hpp"
#include "control/control_bus.hpp"
#include "control/control_state.hpp"
#include "splat/formats.hpp"
#include "project/document.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace editor::control;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

photara::splat::GaussianModel fixture() {
    using tinytensor::Tensor;
    constexpr auto cpu = tinytensor::Device::CPU;
    photara::splat::GaussianModel result;
    result.sh_degree = 0;
    result.means = Tensor::from_vector({1.F, 0.F, 0.F}, {1U, 3U}, cpu);
    result.log_scales = Tensor::from_vector({-1.F, -1.F, -1.F}, {1U, 3U}, cpu);
    result.quaternions = Tensor::from_vector({1.F, 0.F, 0.F, 0.F}, {1U, 4U}, cpu);
    result.opacity_logits = Tensor::from_vector({1.F}, {1U, 1U}, cpu);
    result.sh = Tensor::from_vector({0.F, 0.F, 0.F}, {1U, 1U, 3U}, cpu);
    return result;
}
}

int main() {
    const auto folder = std::filesystem::temp_directory_path() /
        ("photara_control_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(folder);
    try {
        editor::App app;
        ControlBus bus;
        bus.set_session_id("contract-test");
        const auto call = [&](const std::string& method, const json& params = json::object(),
                              const std::string& request = "") {
            auto future = bus.submit(method, params, "test", request);
            bus.pump_main_thread(app);
            bus.pump_post_present(app);
            return future.get();
        };
        const int iterations = app.settings.iterations;
        require(!ok(call("studio.params.patch", {{"patch", {{"iterations", 17}, {"strategy", "invalid"}}}})), "invalid patch accepted");
        require(app.settings.iterations == iterations, "failed patch partially modified settings");
        require(ok(call("studio.params.patch", {{"patch", {{"iterations", 17}}}}, "patch-1")), "valid patch failed");
        require(ok(call("studio.params.patch", {{"patch", {{"iterations", 18}}}})), "second patch failed");
        require(ok(call("studio.params.patch", {{"patch", {{"iterations", 17}}}}, "patch-1")), "retry failed");
        require(app.settings.iterations == 18, "retry executed twice");
        require(error_code(call("studio.params.patch", {{"patch", {{"iterations", 19}}}}, "patch-1")) == "CONFLICT", "request ID conflict not detected");

        // Duplicate submissions before execution must share the same outcome.
        auto a = bus.submit("studio.params.patch", {{"patch", {{"iterations", 20}}}}, "test", "pending");
        auto b = bus.submit("studio.params.patch", {{"patch", {{"iterations", 20}}}}, "test", "pending");
        bus.pump_main_thread(app);
        require(a.get() == b.get(), "in-flight retry differs");
        require(error_code(call("studio.job.control", {{"action", 7}})) == "INVALID_REQUEST", "malformed deferred command escaped validation");

        auto expired = bus.submit("studio.params.patch", {{"patch", {{"iterations", 21}}}}, "test", "expired");
        std::this_thread::sleep_for(std::chrono::milliseconds(5100));
        bus.pump_main_thread(app);
        require(error_code(expired.get()) == "EXPIRED" && app.settings.iterations == 20, "expired command changed settings");

        const auto id = bus.begin_operation("train", "test");
        bus.mark_operation_cancelled();
        require(call("studio.job.status", {{"op_id", id}})["job"]["status"] == "cancelled", "cancelled result lost");
        const auto newer = bus.begin_operation("align", "test");
        require(call("studio.job.status", {{"op_id", id}})["job"]["op_id"] == id, "status followed a newer operation");
        require(error_code(call("studio.job.status", {{"op_id", "missing"}})) == "NOT_FOUND", "unknown operation accepted");
        require(call("studio.job.status", {{"op_id", newer}})["job"]["status"] == "failed", "failed operation reported success");

        require(error_code(call("studio.job.start", {{"stage", "export_sfm"}, {"dry_run", true}})) == "PRECONDITION_FAILED", "SfM dry-run action missing");
        const float yaw = app.camera.yaw;
        require(!ok(call("studio.view.patch", {{"patch", {{"yaw", 1.0}, {"distance", -1}}}})), "invalid camera patch accepted");
        require(app.camera.yaw == yaw, "failed camera patch partially applied");
        const auto view = call("studio.view.patch", {{"patch", {{"target", {1, 2, 3}}, {"yaw", 1.0}, {"projection", "orthographic"}, {"ortho_height", 7}}}});
        require(ok(view) && app.camera.target.z == 3 && app.camera.ortho_height == 7, "camera patch failed");

        // Run save/export against a small in-memory cloud without a GPU or model download.
        app.composed_model = std::make_shared<photara::splat::GaussianModel>(fixture());
        editor::GaussianModelEntry entry;
        entry.name = "fixture";
        entry.cached_model = app.composed_model;
        app.gaussian_models.push_back(std::move(entry));
        app.has_model = true;
        const auto exported = folder / "model.ply";
        require(ok(call("studio.export.model", {{"path", editor::path_to_utf8(exported)}, {"sh_degree", 0}})), "model export failed");
        require(photara::splat::load_gaussians(exported, tinytensor::Device::CPU).size() == 1, "exported model invalid");
        require(error_code(call("studio.export.model", {{"path", editor::path_to_utf8(exported)}})) == "ALREADY_EXISTS", "export overwrote without request");
        require(ok(call("studio.export.model", {{"path", editor::path_to_utf8(exported)}, {"overwrite", true}})), "explicit overwrite failed");
        const auto project = folder / "session.ascan";
        require(ok(call("studio.project.save", {{"path", editor::path_to_utf8(project)}})), "project save failed");
        const auto archive = photara::project::Archive::open(project);
        require(!archive.chunk(photara::project::ChunkType::gaussians).empty(), "saved project lacks gaussians");
        require(ok(call("studio.project.save")), "save to current project failed");
        require(!ok(call("studio.project.save", {{"path", "relative.ascan"}})), "relative output accepted");
        auto pending = bus.submit("studio.ping", json::object(), "test");
        bus.stop();
        require(error_code(pending.get()) == "SHUTDOWN", "shutdown left an unresolved command");
        std::filesystem::remove_all(folder);
        std::cout << "Studio control contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(folder);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
