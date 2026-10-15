#include "splat/model_handoff.hpp"
#include "splat/formats.hpp"
#include "splat/trainer.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
void require(bool ok, const char* text) {
    if (!ok) throw std::runtime_error(text);
}

photara::splat::GaussianModel make_model(unsigned flags) {
    using tinytensor::Tensor;
    using tinytensor::Device;
    constexpr std::size_t n = 1024;
    photara::splat::GaussianModel model;
    std::vector<float> xyz(n * 3), scales(n * 3, -2.F), quat(n * 4, 0.F),
        opacity(n, 0.25F), sh(n * 48), normals(n * 4, .5F), filter(n, .01F);
    for (std::size_t i = 0; i < n; ++i) {
        xyz[3 * i] = static_cast<float>(i) * .001F;
        xyz[3 * i + 2] = 2.F;
        quat[4 * i] = 1.F;
    }
    for (std::size_t i = 0; i < sh.size(); ++i)
        sh[i] = static_cast<float>(i % 31) * .001F;
    model.means = Tensor::from_vector(xyz, {n, 3U}, Device::CPU);
    model.log_scales = Tensor::from_vector(scales, {n, 3U}, Device::CPU);
    model.quaternions = Tensor::from_vector(quat, {n, 4U}, Device::CPU);
    model.opacity_logits = Tensor::from_vector(opacity, {n, 1U}, Device::CPU);
    model.sh = Tensor::from_vector(sh, {n, 16U, 3U}, Device::CPU);
    if (flags & 2U) model.normal_features = Tensor::from_vector(normals, {n, 4U}, Device::CPU);
    if (flags & 1U) model.filter_3d = Tensor::from_vector(filter, {n, 1U}, Device::CPU);
    model.sh_degree = 3;
    return model;
}

void compare(const photara::splat::GaussianModel& a,
             const photara::splat::GaussianModel& b, unsigned flags) {
    require(a.size() == b.size() && a.sh_degree == b.sh_degree, "Model dimensions changed");
    const auto equal = [](const tinytensor::Tensor& x, const tinytensor::Tensor& y) {
        return x.to_vector() == y.to_vector();
    };
    require(equal(a.means, b.means) && equal(a.log_scales, b.log_scales) &&
            equal(a.quaternions, b.quaternions) && equal(a.opacity_logits, b.opacity_logits) &&
            equal(a.sh, b.sh), "Shared memory changed Gaussian parameters");
    if (flags & 1U) require(equal(a.filter_3d, b.filter_3d), "3D filter changed");
    if (flags & 2U) require(equal(a.normal_features, b.normal_features), "Normal features changed");
}
}

int main(int argc, char** argv) {
    try {
        using namespace photara::splat;
        if (argc == 4 && std::string(argv[1]) == "--publish") {
            const std::filesystem::path ready(argv[2]);
            const unsigned flags = std::stoul(argv[3]);
            ModelHandoff publisher(make_model(flags), ready);
            save_gaussians(publisher.model(), ready.parent_path() / "saved.ply");
            require(publisher.wait_for_reader(std::chrono::seconds(10)), "Reader did not attach");
            return 0;
        }
        const auto root = std::filesystem::temp_directory_path() /
            ("photara_handoff_test_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const auto exe = std::filesystem::absolute(argv[0]);
        for (unsigned flags = 0; flags < 4; ++flags) {
            const auto ready = root / ("model_" + std::to_string(flags) + ".ready");
#ifdef _WIN32
            std::wstring command = L"\"" + exe.wstring() + L"\" --publish \"" +
                ready.wstring() + L"\" " + std::to_wstring(flags);
            STARTUPINFOW startup{}; startup.cb = sizeof(startup);
            PROCESS_INFORMATION process{};
            require(CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE,
                                  CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process),
                    "Cannot start publisher process");
            CloseHandle(process.hThread);
#else
            const auto child = fork();
            require(child >= 0, "Cannot fork publisher process");
            if (child == 0) {
                const auto flag = std::to_string(flags);
                execl(exe.c_str(), exe.c_str(), "--publish", ready.c_str(), flag.c_str(), nullptr);
                _exit(3);
            }
#endif
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!std::filesystem::exists(ready) && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            auto received = receive_model_handoff(ready);
#ifdef _WIN32
            require(WaitForSingleObject(process.hProcess, 15000) == WAIT_OBJECT_0,
                    "Publisher failed to exit after reader acknowledgement");
            DWORD code{};
            GetExitCodeProcess(process.hProcess, &code);
            CloseHandle(process.hProcess);
            require(code == 0, "Publisher failed");
#else
            int status{};
            require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                    "Publisher failed");
#endif
            // The publisher and its mapping are gone. The received model owns
            // its data; no CUDA device/context or PLY parser was needed to receive it.
            compare(make_model(flags), received, flags);
            bool expired = false;
            try { receive_model_handoff(ready); } catch (const std::exception&) { expired = true; }
            require(expired, "Publisher mapping was leaked");
            const auto saved = load_gaussians(root / "saved.ply", tinytensor::Device::CPU);
            compare(received, saved, flags);
        }
        const auto invalid = root / "invalid.ready";
        { std::ofstream out(invalid); out << "PHOTARA_MODEL_HANDOFF 1\nmissing\n8\n"; }
        bool rejected = false;
        try { receive_model_handoff(invalid); } catch (const std::exception&) { rejected = true; }
        require(rejected, "Truncated manifest was accepted");
        require(std::filesystem::weakly_canonical(root).parent_path() ==
                    std::filesystem::weakly_canonical(std::filesystem::temp_directory_path()),
                "Test cleanup escaped its temporary directory");
        std::filesystem::remove_all(root);
        std::cout << "Cross-process handoff, all optional attributes, publisher exit, PLY fallback and invalid manifests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
